"""AIMessage 工具调用元数据一致性辅助模块。

本模块负责在中间件修改 AIMessage 的结构化 tool_calls 时，
同步维护原始提供商（raw provider）工具调用元数据的一致性。

LangChain 的 AIMessage 同时携带两套工具调用信息：
1. 结构化的 tool_calls 列表 —— 标准化的、可供下游直接使用的工具调用表示
2. additional_kwargs["tool_calls"] —— 原始提供商返回的未处理工具调用载荷
   （某些中间件，如 DanglingToolCallMiddleware，依赖此原始数据）

当中间件对结构化 tool_calls 进行增删改时，如果不同步清理原始提供商元数据，
就会导致两套数据不一致，从而引发下游处理错误。

此外，本模块还负责：
- 当所有 tool_calls 被移除时，清除已废弃的 function_call 字段
- 当没有工具调用时，将 finish_reason 从 "tool_calls" 更正为 "stop"
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage


def _raw_tool_call_id(raw_tool_call: Any) -> str | None:
    """从原始提供商工具调用条目中安全提取 id 字段。

    原始提供商返回的 tool_calls 条目格式不固定，本函数对其做防御性处理：
    只接受字典类型，且仅当 id 为非空字符串时才返回，否则返回 None。

    Args:
        raw_tool_call: 原始提供商工具调用条目，预期为 dict，但可能为其他类型。

    Returns:
        工具调用的 id 字符串，如果不存在或无效则返回 None。
    """
    if not isinstance(raw_tool_call, dict):
        return None

    raw_id = raw_tool_call.get("id")
    return raw_id if isinstance(raw_id, str) and raw_id else None


def clone_ai_message_with_tool_calls(
    message: AIMessage,
    tool_calls: list[dict[str, Any]],
    *,
    content: Any | None = None,
) -> AIMessage:
    """克隆 AIMessage 并保持原始提供商工具调用元数据与结构化 tool_calls 同步。

    当中间件需要修改 AIMessage 的 tool_calls 列表时（例如过滤、截断或替换），
    应使用本函数而非直接修改，以确保 AIMessage 内部的三处关联数据保持一致：
      - message.tool_calls（结构化工具调用列表）
      - message.additional_kwargs["tool_calls"]（原始提供商工具调用载荷）
      - message.response_metadata["finish_reason"]（完成原因标记）

    同步规则如下：
      1. 从新的 tool_calls 中收集所有保留的 id 集合（kept_ids）
      2. 过滤 additional_kwargs["tool_calls"]，仅保留 id 在 kept_ids 中的条目；
         若过滤后为空，则移除整个 "tool_calls" 键
      3. 如果新的 tool_calls 为空列表，则清除 additional_kwargs 中遗留的
         "function_call" 键（该字段是旧版 LangChain 的遗留字段，与工具调用相关）
      4. 如果新的 tool_calls 为空且 finish_reason 为 "tool_calls"，则将其
         更正为 "stop"（因为没有工具调用意味着模型实际上已正常结束）

    Args:
        message: 需要克隆的原始 AIMessage。
        tool_calls: 新的结构化工具调用列表，将替换原始消息中的 tool_calls。
        content: 可选的新内容，如果提供则覆盖原始消息的 content。

    Returns:
        克隆后的 AIMessage，其中 tool_calls、additional_kwargs 和 response_metadata
        已按照上述规则同步更新。
    """
    # 收集新 tool_calls 中所有有效的 id，用于后续过滤原始提供商工具调用
    kept_ids = {tc["id"] for tc in tool_calls if isinstance(tc.get("id"), str) and tc["id"]}

    # 构建更新字典：始终设置 tool_calls，可选覆盖 content
    update: dict[str, Any] = {"tool_calls": tool_calls}
    if content is not None:
        update["content"] = content

    # --- 同步 additional_kwargs 中的原始提供商工具调用元数据 ---
    additional_kwargs = dict(getattr(message, "additional_kwargs", {}) or {})
    raw_tool_calls = additional_kwargs.get("tool_calls")

    if isinstance(raw_tool_calls, list):
        # 仅保留 id 仍在 kept_ids 中的原始工具调用条目，确保与结构化 tool_calls 一致
        synced_raw_tool_calls = [raw_tc for raw_tc in raw_tool_calls if _raw_tool_call_id(raw_tc) in kept_ids]
        if synced_raw_tool_calls:
            # 过滤后仍有残留的原始工具调用，更新 additional_kwargs
            additional_kwargs["tool_calls"] = synced_raw_tool_calls
        else:
            # 过滤后为空，移除 "tool_calls" 键，避免留下空列表造成歧义
            additional_kwargs.pop("tool_calls", None)

    # 当没有工具调用时，清除遗留的 function_call 字段
    # function_call 是旧版 LangChain/OpenAI 单工具调用格式，当所有 tool_calls 被移除后
    # 它也应该被清理，否则下游可能会误认为仍有工具调用
    if not tool_calls:
        additional_kwargs.pop("function_call", None)

    update["additional_kwargs"] = additional_kwargs

    # --- 同步 response_metadata 中的 finish_reason ---
    # 当所有工具调用被移除后，原来的 "tool_calls" 完成原因已不适用，
    # 需要更正为 "stop" 以表示模型已正常结束生成
    response_metadata = dict(getattr(message, "response_metadata", {}) or {})
    if not tool_calls and response_metadata.get("finish_reason") == "tool_calls":
        response_metadata["finish_reason"] = "stop"
    update["response_metadata"] = response_metadata

    # 使用 model_copy 创建浅拷贝，仅更新指定的字段
    return message.model_copy(update=update)
