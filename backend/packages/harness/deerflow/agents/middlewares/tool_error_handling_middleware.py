"""
工具错误处理中间件及共享运行时中间件构建器。

本模块包含两部分核心功能：

1. ToolErrorHandlingMiddleware —— 工具错误处理中间件
   拦截工具调用过程中抛出的异常，将其转换为带有错误状态的 ToolMessage，
   使得 Agent 运行流程不会因工具异常而中断，而是继续使用已有的上下文信息
   或选择替代工具进行后续处理。同时保留 LangGraph 的 GraphBubbleUp 控制流
   信号（中断/暂停/恢复），确保框架内部的状态管理机制不受影响。

2. 运行时中间件构建函数
   - _build_runtime_middlewares: 构建所有 Agent 类型共享的基础中间件链
   - build_lead_runtime_middlewares: 为主 Agent 构建运行时中间件（含上传、悬空工具调用修复）
   - build_subagent_runtime_middlewares: 为子 Agent 构建运行时中间件（含视觉支持判断）

中间件的组装顺序至关重要，具体顺序如下：
   ThreadDataMiddleware -> UploadsMiddleware(可选) -> SandboxMiddleware ->
   DanglingToolCallMiddleware(可选) -> LLMErrorHandlingMiddleware ->
   GuardrailMiddleware(可选) -> SandboxAuditMiddleware -> ToolErrorHandlingMiddleware
"""

import logging
from collections.abc import Awaitable, Callable
from typing import override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphBubbleUp
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from deerflow.config.app_config import AppConfig

logger = logging.getLogger(__name__)

# 当工具调用的 id 缺失时使用的默认标识符
_MISSING_TOOL_CALL_ID = "missing_tool_call_id"


class ToolErrorHandlingMiddleware(AgentMiddleware[AgentState]):
    """
    工具错误处理中间件。

    将工具执行过程中抛出的异常转换为带有错误状态的 ToolMessage，
    使 Agent 运行可以继续，而非因未捕获的异常而终止。

    该中间件同时处理同步（wrap_tool_call）和异步（awrap_tool_call）
    两种工具调用方式，逻辑完全一致。对于 LangGraph 内部的
    GraphBubbleUp 控制流信号（用于中断/暂停/恢复），该中间件会
    直接重新抛出，不将其视为错误，以保证 LangGraph 的状态管理
    机制正常运作。
    """

    def _build_error_message(self, request: ToolCallRequest, exc: Exception) -> ToolMessage:
        """
        根据工具调用请求和捕获的异常构建错误 ToolMessage。

        构建流程：
        1. 从 tool_call 字典中提取工具名称和调用 id；
           若字段缺失则分别回退到 "unknown_tool" 和 _MISSING_TOOL_CALL_ID。
        2. 将异常信息转为字符串并去除首尾空白；若为空则使用异常类名。
        3. 对过长的错误详情进行截断（上限 500 字符），尾部附加 "..."。
        4. 组装包含工具名、异常类名和错误详情的内容字符串，
           并提示 Agent 继续使用已有上下文或选择替代工具。
        5. 返回 status="error" 的 ToolMessage，LangGraph 据此识别该消息为错误结果。

        Args:
            request: 工具调用请求，包含 tool_call 字典（name、id、args 等字段）。
            exc: 工具执行过程中抛出的异常。

        Returns:
            带有 status="error" 的 ToolMessage，告知 LLM 该工具调用失败。
        """
        # 提取工具名称，缺失时回退到 "unknown_tool"
        tool_name = str(request.tool_call.get("name") or "unknown_tool")
        # 提取工具调用 id，缺失时回退到默认标识符；此 id 必须与对应的 AIMessage tool_call id 匹配
        tool_call_id = str(request.tool_call.get("id") or _MISSING_TOOL_CALL_ID)
        # 将异常转为字符串并去除首尾空白；若结果为空则使用异常类名作为详情
        detail = str(exc).strip() or exc.__class__.__name__
        # 截断过长的错误详情，防止超长消息影响上下文窗口
        if len(detail) > 500:
            detail = detail[:497] + "..."

        # 组装错误消息内容，包含工具名、异常类型、详情以及给 LLM 的后续操作提示
        content = f"Error: Tool '{tool_name}' failed with {exc.__class__.__name__}: {detail}. Continue with available context, or choose an alternative tool."
        return ToolMessage(
            content=content,
            tool_call_id=tool_call_id,
            name=tool_name,
            status="error",  # 标记为错误状态，LangGraph 和下游中间件据此识别工具调用失败
        )

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """
        同步工具调用的错误拦截。

        执行流程：
        1. 调用 handler 执行实际的工具逻辑。
        2. 若 handler 抛出 GraphBubbleUp，直接重新抛出——这是 LangGraph 的控制流信号
           （中断/暂停/恢复），不能被吞掉或转换为错误消息。
        3. 若 handler 抛出其他异常，记录日志并返回错误 ToolMessage，使运行继续。

        Args:
            request: 工具调用请求。
            handler: 实际执行工具调用的处理函数。

        Returns:
            工具调用的正常结果，或在异常情况下的错误 ToolMessage。
        """
        try:
            return handler(request)
        except GraphBubbleUp:
            # 保留 LangGraph 控制流信号（中断/暂停/恢复），不将其视为工具错误
            raise
        except Exception as exc:
            # 记录工具执行失败的异常日志，包含工具名称和调用 id 以便排查
            logger.exception("Tool execution failed (sync): name=%s id=%s", request.tool_call.get("name"), request.tool_call.get("id"))
            return self._build_error_message(request, exc)

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """
        异步工具调用的错误拦截。

        逻辑与同步版本 wrap_tool_call 完全一致，区别仅在于使用 await 调用 handler。
        同样需要保留 GraphBubbleUp 信号，并将其他异常转换为错误 ToolMessage。

        Args:
            request: 工具调用请求。
            handler: 实际执行工具调用的异步处理函数。

        Returns:
            工具调用的正常结果，或在异常情况下的错误 ToolMessage。
        """
        try:
            return await handler(request)
        except GraphBubbleUp:
            # 保留 LangGraph 控制流信号（中断/暂停/恢复），不将其视为工具错误
            raise
        except Exception as exc:
            # 记录工具执行失败的异常日志，包含工具名称和调用 id 以便排查
            logger.exception("Tool execution failed (async): name=%s id=%s", request.tool_call.get("name"), request.tool_call.get("id"))
            return self._build_error_message(request, exc)


def _build_runtime_middlewares(
    *,
    app_config: AppConfig,
    include_uploads: bool,
    include_dangling_tool_call_patch: bool,
    lazy_init: bool = True,
) -> list[AgentMiddleware]:
    """
    构建所有 Agent 类型（主 Agent 和子 Agent）共享的基础中间件链。

    中间件的组装顺序非常重要，因为各中间件之间存在依赖关系：
    - ThreadDataMiddleware 必须最先执行，为后续中间件创建线程级目录和用户数据路径
    - UploadsMiddleware 需要在线程数据就绪后注入上传文件信息（插入到位置 1，即 SandboxMiddleware 之前）
    - SandboxMiddleware 依赖线程数据来获取沙箱路径映射
    - DanglingToolCallMiddleware 在沙箱就绪后修复悬空工具调用
    - LLMErrorHandlingMiddleware 在工具执行之前规范化 LLM 调用错误
    - GuardrailMiddleware 在 LLM 错误处理后进行工具调用前的授权检查
    - SandboxAuditMiddleware 在工具执行前审计沙箱操作
    - ToolErrorHandlingMiddleware 必须在最后，确保所有前置中间件处理完毕后才捕获工具执行异常并转换为错误 ToolMessage

    Args:
        app_config: 应用配置对象，提供模型、防护栏等配置信息。
        include_uploads: 是否包含上传文件中间件（主 Agent 需要，子 Agent 不需要）。
        include_dangling_tool_call_patch: 是否包含悬空工具调用修复中间件。
        lazy_init: 是否延迟初始化中间件（默认 True），适用于运行时按需加载的场景。

    Returns:
        按正确顺序排列的中间件列表。
    """
    from deerflow.agents.middlewares.llm_error_handling_middleware import LLMErrorHandlingMiddleware
    from deerflow.agents.middlewares.thread_data_middleware import ThreadDataMiddleware
    from deerflow.sandbox.middleware import SandboxMiddleware

    # 基础中间件链：ThreadDataMiddleware 最先，SandboxMiddleware 其次
    middlewares: list[AgentMiddleware] = [
        ThreadDataMiddleware(lazy_init=lazy_init),
        SandboxMiddleware(lazy_init=lazy_init),
    ]

    # 如果需要上传支持（主 Agent 场景），将 UploadsMiddleware 插入到位置 1
    # 即 ThreadDataMiddleware 之后、SandboxMiddleware 之前
    # 这样上传文件信息在沙箱初始化之前就已注入到对话状态中
    if include_uploads:
        from deerflow.agents.middlewares.uploads_middleware import UploadsMiddleware

        middlewares.insert(1, UploadsMiddleware())

    # 如果需要悬空工具调用修复（主 Agent 和子 Agent 均需要），
    # 将 DanglingToolCallMiddleware 追加到基础链末尾
    # 它负责为缺少响应的 AIMessage tool_calls 注入占位 ToolMessage
    if include_dangling_tool_call_patch:
        from deerflow.agents.middlewares.dangling_tool_call_middleware import DanglingToolCallMiddleware

        middlewares.append(DanglingToolCallMiddleware())

    # LLMErrorHandlingMiddleware 必须在工具相关中间件之前，
    # 确保模型调用失败时先规范化为可恢复的错误，再交给后续中间件处理
    middlewares.append(LLMErrorHandlingMiddleware(app_config=app_config))

    # 防护栏中间件（如果配置中启用）：在工具调用前进行授权检查
    guardrails_config = app_config.guardrails
    if guardrails_config.enabled and guardrails_config.provider:
        import inspect

        from deerflow.guardrails.middleware import GuardrailMiddleware
        from deerflow.reflection import resolve_variable

        # 通过反射机制动态加载防护栏提供者类
        provider_cls = resolve_variable(guardrails_config.provider.use)
        # 提取提供者的配置参数，若无配置则使用空字典
        provider_kwargs = dict(guardrails_config.provider.config) if guardrails_config.provider.config else {}
        # 如果提供者构造函数接受 'framework' 参数或 **kwargs，
        # 则自动注入 framework="deerflow"，用于配置发现等用途。
        # 内置提供者（如 AllowlistProvider）不需要此参数，因此仅在构造函数
        # 声明了 'framework' 参数或接受 **kwargs 时才注入。
        if "framework" not in provider_kwargs:
            try:
                sig = inspect.signature(provider_cls.__init__)
                if "framework" in sig.parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                    provider_kwargs["framework"] = "deerflow"
            except (ValueError, TypeError):
                pass
        # 实例化防护栏提供者，并创建 GuardrailMiddleware
        # fail_closed 控制当防护栏检查失败时是否完全阻止（闭锁模式）
        # passport 用于在工具调用间传递防护栏的检查结果
        provider = provider_cls(**provider_kwargs)
        middlewares.append(GuardrailMiddleware(provider, fail_closed=guardrails_config.fail_closed, passport=guardrails_config.passport))

    # SandboxAuditMiddleware 在工具执行前审计沙箱内的 shell/文件操作，用于安全日志记录
    from deerflow.agents.middlewares.sandbox_audit_middleware import SandboxAuditMiddleware

    middlewares.append(SandboxAuditMiddleware())
    # ToolErrorHandlingMiddleware 必须在中间件链的最后位置，
    # 确保所有前置中间件处理完毕后才捕获工具执行异常，
    # 避免将前置中间件的正常逻辑误判为工具错误
    middlewares.append(ToolErrorHandlingMiddleware())
    return middlewares


def build_lead_runtime_middlewares(*, app_config: AppConfig, lazy_init: bool = True) -> list[AgentMiddleware]:
    """
    为主 Agent（Lead Agent）构建运行时中间件链。

    与子 Agent 的区别：
    - 启用上传支持（include_uploads=True）：主 Agent 需要处理用户上传的文件，
      将文件信息注入对话状态，而子 Agent 不直接接收上传文件。
    - 启用悬空工具调用修复（include_dangling_tool_call_patch=True）：
      主 Agent 可能因用户中断等原因产生无响应的 tool_calls，需要补全。

    此函数构建的中间件链会在 agent.py 的 _build_middlewares 中被调用，
    后续还会追加主 Agent 专属的中间件（如 SummarizationMiddleware、
    TodoListMiddleware、TokenUsageMiddleware、TitleMiddleware、
    MemoryMiddleware、LoopDetectionMiddleware、ClarificationMiddleware 等）。

    Args:
        app_config: 应用配置对象。
        lazy_init: 是否延迟初始化中间件（默认 True）。

    Returns:
        主 Agent 的运行时中间件列表。
    """
    return _build_runtime_middlewares(
        app_config=app_config,
        include_uploads=True,  # 主 Agent 需要处理上传文件
        include_dangling_tool_call_patch=True,  # 主 Agent 需要修复悬空工具调用
        lazy_init=lazy_init,
    )


def build_subagent_runtime_middlewares(
    *,
    app_config: AppConfig | None = None,
    model_name: str | None = None,
    lazy_init: bool = True,
) -> list[AgentMiddleware]:
    """
    为子 Agent（Subagent）构建运行时中间件链。

    与主 Agent 的区别：
    - 不启用上传支持（include_uploads=False）：子 Agent 不直接处理用户上传的文件。
    - 启用悬空工具调用修复（include_dangling_tool_call_patch=True）：
      子 Agent 同样可能因超时等原因产生悬空的 tool_calls。
    - 额外的视觉支持判断：如果子 Agent 使用的模型支持视觉功能（supports_vision），
      则追加 ViewImageMiddleware，使子 Agent 能够处理图像内容。

    此函数在 subagents/executor.py 中被调用，用于为每个子 Agent 实例构建中间件链。

    Args:
        app_config: 应用配置对象；若为 None 则自动通过 get_app_config() 获取。
        model_name: 模型名称；若为 None 则使用配置中的第一个模型。
            用于判断是否需要追加 ViewImageMiddleware。
        lazy_init: 是否延迟初始化中间件（默认 True）。

    Returns:
        子 Agent 的运行时中间件列表，可能包含 ViewImageMiddleware。
    """
    # 若未传入 app_config，则从全局配置中获取
    if app_config is None:
        from deerflow.config import get_app_config

        app_config = get_app_config()

    # 构建子 Agent 的基础中间件链（不含上传支持）
    middlewares = _build_runtime_middlewares(
        app_config=app_config,
        include_uploads=False,  # 子 Agent 不直接处理上传文件
        include_dangling_tool_call_patch=True,  # 子 Agent 需要修复悬空工具调用
        lazy_init=lazy_init,
    )

    # 若未指定模型名称，则回退到配置中的第一个模型
    if model_name is None and app_config.models:
        model_name = app_config.models[0].name

    # 查询模型配置，判断是否支持视觉功能
    model_config = app_config.get_model_config(model_name) if model_name else None
    # 如果模型支持视觉（supports_vision），追加 ViewImageMiddleware
    # 使子 Agent 能够将图像以 base64 编码注入到对话状态中
    if model_config is not None and model_config.supports_vision:
        from deerflow.agents.middlewares.view_image_middleware import ViewImageMiddleware

        middlewares.append(ViewImageMiddleware())

    return middlewares
