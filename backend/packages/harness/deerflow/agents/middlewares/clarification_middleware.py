"""澄清中间件：拦截 ask_clarification 工具调用（clarification_tool.py）并中断执行，向用户呈现问题。

当模型调用 ask_clarification 工具时，该中间件会：
1. 在工具执行之前拦截该调用
2. 提取澄清问题及相关元数据
3. 将信息格式化为用户友好的消息
4. 返回 Command 指令中断执行并向用户呈现问题
5. 等待用户回复后再继续执行

此机制替代了之前基于工具的方案——在旧方案中，澄清请求会继续在对话流中传递，
而新方案通过中断执行确保用户必须先回应问题，Agent 才会继续后续操作。
"""

import json
import logging
from collections.abc import Callable
from hashlib import sha256
from typing import override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.graph import END
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

logger = logging.getLogger(__name__)


class ClarificationMiddlewareState(AgentState):
    """与 ThreadState 模式兼容的状态类。"""

    pass


class ClarificationMiddleware(AgentMiddleware[ClarificationMiddlewareState]):
    """拦截 ask_clarification 工具调用，中断执行并向用户呈现澄清问题。

    当模型调用 ask_clarification 工具时，此中间件：
    1. 在工具执行之前拦截该工具调用
    2. 提取澄清问题及相关元数据（问题内容、类型、上下文、选项）
    3. 将信息格式化为用户友好的消息（含类型图标和选项编号）
    4. 返回 Command 指令中断执行（goto=END）并向用户呈现格式化后的问题
    5. 等待用户回复后再继续执行

    此机制替代了之前基于工具的方案——在旧方案中，澄清请求会继续在对话流中传递，
    而新方案通过中断执行确保用户必须先回应问题，Agent 才会继续后续操作。

    注意：该中间件必须位于中间件链的最后位置，以确保所有前置中间件
    （如工具错误处理、循环检测等）已处理完毕后才执行澄清拦截。
    """

    state_schema = ClarificationMiddlewareState

    def _stable_message_id(self, tool_call_id: str, formatted_message: str) -> str:
        """构建确定性的消息 ID，使重试的澄清调用替换而非追加历史消息。

        当 LangGraph 重试同一工具调用时，如果每次都生成不同的消息 ID，
        会导致消息历史中出现重复的澄清消息。通过基于 tool_call_id 或
        消息内容生成确定性 ID，确保重试的调用会覆盖之前的消息，
        而非在对话历史中追加新的消息。

        Args:
            tool_call_id: 工具调用的唯一标识符，由 LangGraph 分配
            formatted_message: 格式化后的澄清消息文本，用作无 ID 时的回退哈希源

        Returns:
            以 "clarification:" 为前缀的确定性消息 ID 字符串
        """
        if tool_call_id:
            # 有 tool_call_id 时直接使用，确保同一工具调用的重试使用相同的消息 ID
            return f"clarification:{tool_call_id}"
        # 无 tool_call_id 时，对格式化消息内容进行 SHA-256 哈希取前 16 位
        # 确保相同内容产生相同的 ID，不同内容产生不同的 ID
        digest = sha256(formatted_message.encode("utf-8")).hexdigest()[:16]
        return f"clarification:{digest}"

    def _is_chinese(self, text: str) -> bool:
        """检查文本是否包含中文字符。

        通过检测 Unicode 范围 一-鿿（CJK 统一表意文字基本区）
        来判断文本中是否包含中文字符。此方法可用于后续的消息格式化逻辑中，
        根据语言类型选择不同的排版风格。

        Args:
            text: 待检查的文本字符串

        Returns:
            如果文本中包含至少一个中文字符则返回 True，否则返回 False
        """
        return any("一" <= char <= "鿿" for char in text)

    def _format_clarification_message(self, args: dict) -> str:
        """将澄清工具调用的参数格式化为用户友好的消息。

        从工具调用参数中提取问题、类型、上下文和选项，然后组装成
        带有类型图标和编号选项的易读消息。不同类型的澄清请求会显示
        对应的图标（如缺失信息用问号、模糊需求用思考图标等），
        选项以编号列表的形式呈现，方便用户快速选择。

        Args:
            args: 工具调用参数字典，可能包含以下字段：
                - question: 澄清问题文本（必需）
                - clarification_type: 澄清类型，默认为 "missing_info"
                - context: 问题背景上下文（可选）
                - options: 供用户选择的选项列表（可选）

        Returns:
            格式化后的消息字符串，包含图标、问题、上下文和编号选项
        """
        # 提取各参数字段，缺失时使用合理默认值
        question = args.get("question", "")
        clarification_type = args.get("clarification_type", "missing_info")
        context = args.get("context")
        options = args.get("options", [])

        # 某些模型（如 Qwen3-Max）会将数组参数序列化为 JSON 字符串而非原生数组。
        # 这里进行反序列化和归一化，确保 options 始终为列表类型，
        # 以便后续的渲染逻辑统一处理。
        if isinstance(options, str):
            try:
                options = json.loads(options)
            except (json.JSONDecodeError, TypeError):
                # JSON 解析失败时，将原始字符串作为单元素列表
                options = [options]

        if options is None:
            # options 为 None 时初始化为空列表
            options = []
        elif not isinstance(options, list):
            # options 为非列表类型（如单个字符串、数字等），包装为单元素列表
            options = [options]

        # 不同澄清类型对应的图标映射
        type_icons = {
            "missing_info": "❓",  # 缺失信息
            "ambiguous_requirement": "🤔",  # 需求模糊
            "approach_choice": "🔀",  # 方案选择
            "risk_confirmation": "⚠️",  # 风险确认
            "suggestion": "💡",  # 建议
        }

        # 获取当前类型对应的图标，未识别的类型默认使用问号图标
        icon = type_icons.get(clarification_type, "❓")

        # 逐段组装消息内容
        message_parts = []

        # 根据是否存在上下文信息，采用不同的排版方式
        if context:
            # 有上下文时，先展示背景信息，再展示问题
            message_parts.append(f"{icon} {context}")
            message_parts.append(f"\n{question}")
        else:
            # 无上下文时，直接展示带图标的问题
            message_parts.append(f"{icon} {question}")

        # 添加编号选项列表
        if options and len(options) > 0:
            # 在选项前添加空行，使排版更清晰
            message_parts.append("")
            for i, option in enumerate(options, 1):
                message_parts.append(f"  {i}. {option}")

        return "\n".join(message_parts)

    def _handle_clarification(self, request: ToolCallRequest) -> Command:
        """处理澄清请求，返回中断执行的 Command 指令。

        当检测到 ask_clarification 工具调用时，此方法提取参数、
        格式化消息，并构建一个 Command 指令，该指令会：
        1. 将格式化后的 ToolMessage 添加到消息历史中
        2. 通过 goto=END 中断当前执行流程

        前端会检测 ask_clarification 类型的 ToolMessage 并直接展示
        给用户，无需额外添加 AIMessage。

        Args:
            request: 工具调用请求，包含 tool_call 字典（name、id、args 等字段）

        Returns:
            Command 指令，包含更新消息和跳转到 END 的控制流
        """
        # 从工具调用请求中提取参数
        args = request.tool_call.get("args", {})
        question = args.get("question", "")

        # 记录拦截日志，便于调试和追踪
        logger.info("Intercepted clarification request")
        logger.debug("Clarification question: %s", question)

        # 将参数格式化为用户友好的消息文本
        formatted_message = self._format_clarification_message(args)

        # 获取工具调用的唯一标识符
        tool_call_id = request.tool_call.get("id", "")

        # 创建 ToolMessage 来承载格式化后的问题
        # 该消息会被添加到消息历史中，前端据此展示澄清问题
        tool_message = ToolMessage(
            id=self._stable_message_id(tool_call_id, formatted_message),
            content=formatted_message,
            tool_call_id=tool_call_id,
            name="ask_clarification",
        )

        # 返回 Command 指令：
        # 1. update: 将格式化后的 ToolMessage 添加到消息历史
        # 2. goto=END: 中断当前执行流程，等待用户回复
        # 注意：这里不需要额外添加 AIMessage，前端会检测并直接展示
        # ask_clarification 类型的 ToolMessage
        return Command(
            update={"messages": [tool_message]},
            goto=END,
        )

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """拦截 ask_clarification 工具调用并中断执行（同步版本）。

        检查当前工具调用是否为 ask_clarification：
        - 如果是，则调用 _handle_clarification 进行拦截处理，
          返回中断执行的 Command 指令
        - 如果不是，则正常执行原始工具处理逻辑

        Args:
            request: 工具调用请求，包含工具名称、调用 ID 和参数等信息
            handler: 原始的工具执行处理函数

        Returns:
            如果是 ask_clarification 调用，返回中断执行的 Command；
            否则返回原始工具处理函数的执行结果
        """
        # 检查是否为 ask_clarification 工具调用
        if request.tool_call.get("name") != "ask_clarification":
            # 非澄清调用，正常执行原始工具逻辑
            return handler(request)

        # 拦截澄清调用，返回中断执行的 Command
        return self._handle_clarification(request)

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """拦截 ask_clarification 工具调用并中断执行（异步版本）。

        逻辑与同步版本 wrap_tool_call 完全一致，区别仅在于
        使用 await 调用原始的异步工具处理函数。

        Args:
            request: 工具调用请求，包含工具名称、调用 ID 和参数等信息
            handler: 原始的异步工具执行处理函数

        Returns:
            如果是 ask_clarification 调用，返回中断执行的 Command；
            否则返回原始工具处理函数的执行结果
        """
        # 检查是否为 ask_clarification 工具调用
        if request.tool_call.get("name") != "ask_clarification":
            # 非澄清调用，正常执行原始异步工具逻辑
            return await handler(request)

        # 拦截澄清调用，返回中断执行的 Command
        return self._handle_clarification(request)
