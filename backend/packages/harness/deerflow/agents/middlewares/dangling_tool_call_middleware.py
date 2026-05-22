"""中间件：修复消息历史中的悬挂工具调用（dangling tool calls）。

在LangChain/LangGraph的消息结构里，如果AIMessage发起了工具调用，理论上后面应该有对应的ToolMessage。
但真实运行中可能发生中断、异常、用户暂停等情况，导致历史消息里出现悬挂的工具调用。这会导致LLM因消息格式不完整而报错。

本中间件拦截模型调用，检测并修补这些缺口：在每个发起工具调用的 AIMessage 之后立即插入合成的 ToolMessage（带有错误指示），确保消息顺序正确。

注意：使用wrap_model_call而非before_model，是为了确保补丁插入到正确的位置（紧接在每个悬挂的AIMessage之后），而非追加到消息列表末尾
（before_model + add_messages归约器会追加到末尾）。
"""

import json
import logging
from collections.abc import Awaitable, Callable
from typing import override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse
from langchain_core.messages import ToolMessage

logger = logging.getLogger(__name__)


class DanglingToolCallMiddleware(AgentMiddleware[AgentState]):
    """在模型调用之前，为悬挂的工具调用插入占位 ToolMessage。

    扫描消息历史，找到那些 tool_calls 缺少对应 ToolMessage 的 AIMessage，
    并在这些 AIMessage 之后立即注入合成的错误响应，使 LLM 收到格式完整的对话。
    """

    @staticmethod
    def _message_tool_calls(msg) -> list[dict]:
        """从结构化字段或原始提供商载荷中提取并归一化工具调用。

        LangChain 会将格式不正确的提供商函数调用存储在 ``invalid_tool_calls`` 中。
        这些调用不会实际执行，但提供商适配器可能仍会将其中的 call id/name
        序列化到下一次请求中，而严格的 OpenAI 兼容校验器要求有匹配的 ToolMessage。
        因此将它们也视为悬挂调用，使下一次模型请求保持格式正确，
        模型也能看到一个可恢复的工具错误，而非再次收到提供商的 400 错误。

        Args:
            msg: 消息对象，通常是 AIMessage 实例

        Returns:
            归一化后的工具调用字典列表，每个字典包含 id、name、args 等字段
        """
        # 归一化结果列表
        normalized: list[dict] = []

        # 1. 首先尝试从标准 tool_calls 字段获取
        tool_calls = getattr(msg, "tool_calls", None) or []
        normalized.extend(list(tool_calls))

        # 2. 如果标准字段为空，尝试从 additional_kwargs 中的原始提供商载荷提取
        #    某些提供商（如 OpenAI）在特定情况下只将工具调用信息放在 additional_kwargs 中
        raw_tool_calls = (getattr(msg, "additional_kwargs", None) or {}).get("tool_calls") or []
        if not tool_calls:
            for raw_tc in raw_tool_calls:
                if not isinstance(raw_tc, dict):
                    continue

                # 尝试从不同结构中提取函数名
                function = raw_tc.get("function")
                name = raw_tc.get("name")
                if not name and isinstance(function, dict):
                    name = function.get("name")

                # 尝试从不同结构中提取参数
                args = raw_tc.get("args", {})
                if not args and isinstance(function, dict):
                    raw_args = function.get("arguments")
                    if isinstance(raw_args, str):
                        # arguments 字段通常是 JSON 字符串，需要解析
                        try:
                            parsed_args = json.loads(raw_args)
                        except (TypeError, ValueError, json.JSONDecodeError):
                            parsed_args = {}
                        args = parsed_args if isinstance(parsed_args, dict) else {}

                # 将原始工具调用归一化为统一格式
                normalized.append(
                    {
                        "id": raw_tc.get("id"),
                        "name": name or "unknown",
                        "args": args if isinstance(args, dict) else {},
                    }
                )

        # 3. 处理无效工具调用（invalid_tool_calls）
        #    这些是格式不正确、无法执行的工具调用，仍需要对应的 ToolMessage
        for invalid_tc in getattr(msg, "invalid_tool_calls", None) or []:
            if not isinstance(invalid_tc, dict):
                continue
            normalized.append(
                {
                    "id": invalid_tc.get("id"),
                    "name": invalid_tc.get("name") or "unknown",
                    "args": {},  # 无效调用没有可用参数
                    "invalid": True,  # 标记为无效调用
                    "error": invalid_tc.get("error"),  # 保留原始错误信息
                }
            )

        return normalized

    @staticmethod
    def _synthetic_tool_message_content(tool_call: dict) -> str:
        """为悬挂的工具调用生成合成的 ToolMessage 内容。

        根据工具调用的类型（无效调用 vs 被中断的调用）生成不同的错误提示文本。

        Args:
            tool_call: 工具调用字典，可能包含 invalid 和 error 字段

        Returns:
            合成的错误提示文本字符串
        """
        # 如果是无效工具调用（参数格式不正确）
        if tool_call.get("invalid"):
            error = tool_call.get("error")
            if isinstance(error, str) and error:
                return f"[Tool call could not be executed because its arguments were invalid: {error}]"
            return "[Tool call could not be executed because its arguments were invalid.]"
        # 如果是被中断的正常工具调用（用户取消等）
        return "[Tool call was interrupted and did not return a result.]"

    def _build_patched_messages(self, messages: list) -> list | None:
        """构建修补后的消息列表，将工具结果紧跟在其对应的工具调用 AIMessage 之后。

        在提供商序列化之前，将消息归一化为因果顺序，同时保持已有效的对话记录不变。
        如果没有需要修补的悬挂调用，返回 None 表示无需修改。

        Args:
            messages: 原始消息列表

        Returns:
            修补后的消息列表，或 None（如果无需修改）
        """
        # ===== 第一步：构建 tool_call_id → ToolMessage 的映射表 =====
        # 用于快速查找某个工具调用是否已有对应的 ToolMessage
        tool_messages_by_id: dict[str, ToolMessage] = {}
        for msg in messages:
            if isinstance(msg, ToolMessage):
                # 使用 setdefault 确保同一个 tool_call_id 只记录第一个 ToolMessage
                tool_messages_by_id.setdefault(msg.tool_call_id, msg)

        # ===== 第二步：收集所有 AIMessage 中发起的工具调用 ID =====
        tool_call_ids: set[str] = set()
        for msg in messages:
            if getattr(msg, "type", None) != "ai":
                continue
            for tc in self._message_tool_calls(msg):
                tc_id = tc.get("id")
                if tc_id:
                    tool_call_ids.add(tc_id)

        # ===== 第三步：重建消息列表，将 ToolMessage 紧跟其 AIMessage =====
        patched: list = []
        # 记录已处理的 ToolMessage ID，避免重复插入
        consumed_tool_msg_ids: set[str] = set()
        # 统计插入的合成 ToolMessage 数量
        patch_count = 0
        for msg in messages:
            # 跳过原位置的工具消息（它们将在 AIMessage 之后重新插入）
            if isinstance(msg, ToolMessage) and msg.tool_call_id in tool_call_ids:
                continue

            # 将当前消息加入修补列表
            patched.append(msg)

            # 只处理 AI 消息，检查其工具调用是否有对应的 ToolMessage
            if getattr(msg, "type", None) != "ai":
                continue

            # 遍历该 AIMessage 的所有工具调用
            for tc in self._message_tool_calls(msg):
                tc_id = tc.get("id")
                # 跳过无 ID 或已处理过的工具调用
                if not tc_id or tc_id in consumed_tool_msg_ids:
                    continue

                # 检查是否已有真实的 ToolMessage
                existing_tool_msg = tool_messages_by_id.get(tc_id)
                if existing_tool_msg is not None:
                    # 已有 ToolMessage，将其插入到 AIMessage 之后
                    patched.append(existing_tool_msg)
                    consumed_tool_msg_ids.add(tc_id)
                else:
                    # 没有对应的 ToolMessage（悬挂调用），插入合成的错误 ToolMessage
                    patched.append(
                        ToolMessage(
                            content=self._synthetic_tool_message_content(tc),
                            tool_call_id=tc_id,
                            name=tc.get("name", "unknown"),
                            status="error",  # 标记为错误状态
                        )
                    )
                    consumed_tool_msg_ids.add(tc_id)
                    patch_count += 1

        # 如果修补后的列表与原始列表相同，说明没有悬挂调用，无需修改
        if patched == messages:
            return None

        # 记录日志，提示修补了几个悬挂调用
        if patch_count:
            logger.warning(f"Injecting {patch_count} placeholder ToolMessage(s) for dangling tool calls")
        return patched

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """同步包装模型调用，在调用前修补悬挂的工具调用。

        拦截模型请求，对消息列表进行悬挂调用检测和修补，然后将修补后的请求传递给实际的模型处理函数。

        Args:
            request: 模型请求，包含消息列表等信息
            handler: 实际的模型调用处理函数

        Returns:
            模型调用结果
        """
        patched = self._build_patched_messages(request.messages)
        if patched is not None:
            request = request.override(messages=patched)
        return handler(request)

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """异步包装模型调用，在调用前修补悬挂的工具调用。

        异步版本的 wrap_model_call，逻辑完全相同，
        适用于异步模型调用场景。

        Args:
            request: 模型请求，包含消息列表等信息
            handler: 实际的异步模型调用处理函数

        Returns:
            模型调用结果
        """
        patched = self._build_patched_messages(request.messages)
        if patched is not None:
            request = request.override(messages=patched)
        return await handler(request)
