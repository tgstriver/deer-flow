"""子代理任务工具，用于将工作委托给专门的子代理。

本模块提供 task_tool，允许主代理将复杂任务委托给专门的子代理执行。
子代理在独立的上下文中运行，有助于：
- 保持上下文分离（探索和实现分开）
- 处理复杂的多步骤任务
- 在隔离环境中执行命令
- 实现并行研究或探索任务

主要特性：
- 支持多种子代理类型（通用、bash 等）
- 后台异步执行
- 实时状态监控和进度报告
- 资源使用统计（Token）
- 取消和超时处理
- 分布式追踪（trace_id）
"""

import asyncio
import logging
import uuid
from dataclasses import replace
from typing import TYPE_CHECKING, Annotated, Any, cast

from langchain.tools import InjectedToolCallId, tool
from langgraph.config import get_stream_writer

from deerflow.config import get_app_config
from deerflow.sandbox.security import LOCAL_BASH_SUBAGENT_DISABLED_MESSAGE, is_host_bash_allowed
from deerflow.subagents import SubagentExecutor, get_available_subagent_names, get_subagent_config
from deerflow.subagents.config import resolve_subagent_model_name
from deerflow.subagents.executor import (
    SubagentStatus,
    cleanup_background_task,
    get_background_task_result,
    request_cancel_background_task,
)
from deerflow.tools.types import Runtime

if TYPE_CHECKING:
    from deerflow.config.app_config import AppConfig

logger = logging.getLogger(__name__)

# 缓存子代理的 Token 使用情况，按 tool_call_id 分组，
# 以便 TokenUsageMiddleware 可以将其写回到触发的 AIMessage 的 usage_metadata 中
_subagent_usage_cache: dict[str, dict[str, int]] = {}


def _token_usage_cache_enabled(app_config: "AppConfig | None") -> bool:
    """检查是否启用了 Token 使用缓存。

    通过检查应用配置中的 token_usage.enabled 设置来决定是否启用缓存。

    Args:
        app_config: 应用配置对象，可能为 None

    Returns:
        True 如果启用了 Token 使用缓存，否则 False

    Note:
        - 如果 app_config 为 None，尝试获取全局配置
        - 如果配置文件不存在，返回 False
    """
    if app_config is None:
        try:
            app_config = get_app_config()
        except FileNotFoundError:
            return False
    return bool(getattr(getattr(app_config, "token_usage", None), "enabled", False))


def _cache_subagent_usage(tool_call_id: str, usage: dict | None, *, enabled: bool = True) -> None:
    """缓存子代理的 Token 使用情况。

    将子代理的使用情况记录到缓存中，供后续处理使用。

    Args:
        tool_call_id: 工具调用 ID，用作缓存键
        usage: Token 使用情况字典，可能为 None
        enabled: 是否启用缓存（默认 True）

    Note:
        - 只有在启用且 usage 不为 None 时才进行缓存
        - 缓存按 tool_call_id 分组
    """
    if enabled and usage:
        _subagent_usage_cache[tool_call_id] = usage


def pop_cached_subagent_usage(tool_call_id: str) -> dict | None:
    """从缓存中弹出（移除并返回）子代理的 Token 使用情况。

    用于获取并清除指定工具调用的使用情况记录。

    Args:
        tool_call_id: 工具调用 ID

    Returns:
        Token 使用情况字典，如果不存在则返回 None
    """
    return _subagent_usage_cache.pop(tool_call_id, None)


def _is_subagent_terminal(result: Any) -> bool:
    """检查后台子代理结果是否处于终端状态（可安全清理）。

    终端状态包括：已完成、失败、已取消、已超时或已设置完成时间。

    Args:
        result: 子代理执行结果

    Returns:
        True 如果结果处于终端状态，否则 False

    Note:
        - 终端状态：COMPLETED、FAILED、CANCELLED、TIMED_OUT
        - 如果设置了 completed_at 时间戳，也认为是终端状态
    """
    return result.status in {SubagentStatus.COMPLETED, SubagentStatus.FAILED, SubagentStatus.CANCELLED, SubagentStatus.TIMED_OUT} or getattr(result, "completed_at", None) is not None


async def _await_subagent_terminal(task_id: str, max_polls: int) -> Any | None:
    """轮询直到后台子代理达到终端状态或轮询次数用完。

    用于等待子代理完成，最多轮询指定次数。

    Args:
        task_id: 任务 ID
        max_polls: 最大轮询次数

    Returns:
        子代理结果，如果任务不存在或超时则返回 None

    Note:
        - 每次轮询间隔 5 秒
        - 如果结果达到终端状态，立即返回
        - 如果轮询次数用完仍未到达终端状态，返回 None
    """
    for _ in range(max_polls):
        result = get_background_task_result(task_id)
        if result is None:
            return None
        if _is_subagent_terminal(result):
            return result
        await asyncio.sleep(5)
    return None


async def _deferred_cleanup_subagent_task(task_id: str, trace_id: str, max_polls: int) -> None:
    """延迟清理已取消的子代理任务，直到它可以安全移除。

    当子代理被取消但仍需要等待其达到终端状态时使用此函数。

    Args:
        task_id: 任务 ID
        trace_id: 追踪 ID，用于日志记录
        max_polls: 最大轮询次数

    Note:
        - 持续轮询直到任务达到终端状态
        - 达到终端状态后执行清理
        - 如果轮询超时，记录警告日志
        - 每次轮询间隔 5 秒
    """
    cleanup_poll_count = 0
    while True:
        result = get_background_task_result(task_id)
        if result is None:
            return
        if _is_subagent_terminal(result):
            cleanup_background_task(task_id)
            return
        if cleanup_poll_count >= max_polls:
            logger.warning(f"[trace={trace_id}] Deferred cleanup for task {task_id} timed out after {cleanup_poll_count} polls")
            return
        await asyncio.sleep(5)
        cleanup_poll_count += 1


def _log_cleanup_failure(cleanup_task: asyncio.Task[None], *, trace_id: str, task_id: str) -> None:
    """记录清理任务失败的日志。

    当延迟清理任务失败时记录错误信息。

    Args:
        cleanup_task: 清理任务
        trace_id: 追踪 ID，用于日志记录
        task_id: 任务 ID

    Note:
        - 如果任务被取消，不记录错误
        - 如果任务异常，记录错误日志
    """
    if cleanup_task.cancelled():
        return

    exc = cleanup_task.exception()
    if exc is not None:
        logger.error(f"[trace={trace_id}] Deferred cleanup failed for task {task_id}: {exc}")


def _schedule_deferred_subagent_cleanup(task_id: str, trace_id: str, max_polls: int) -> None:
    """调度延迟子代理清理任务。

    当子代理被取消但未能立即清理时，安排一个延迟清理任务。

    Args:
        task_id: 任务 ID
        trace_id: 追踪 ID，用于日志记录
        max_polls: 最大轮询次数

    Note:
        - 创建异步任务执行延迟清理
        - 添加完成回调记录失败情况
    """
    logger.debug(f"[trace={trace_id}] Scheduling deferred cleanup for cancelled task {task_id}")
    cleanup_task = asyncio.create_task(_deferred_cleanup_subagent_task(task_id, trace_id, max_polls))
    cleanup_task.add_done_callback(lambda task: _log_cleanup_failure(task, trace_id=trace_id, task_id=task_id))


def _find_usage_recorder(runtime: Any) -> Any | None:
    """在运行时配置中查找具有 ``record_external_llm_usage_records`` 方法的回调处理器。

    用于找到可以记录外部 LLM 使用情况的回调处理器。

    Args:
        runtime: 运行时对象

    Returns:
        找到的回调处理器，如果未找到则返回 None

    Note:
        - 检查运行时配置中的回调列表
        - 查找具有 record_external_llm_usage_records 方法的对象
    """
    if runtime is None:
        return None
    config = getattr(runtime, "config", None)
    if not isinstance(config, dict):
        return None
    callbacks = config.get("callbacks", [])
    if not callbacks:
        return None
    for cb in callbacks:
        if hasattr(cb, "record_external_llm_usage_records"):
            return cb
    return None


def _summarize_usage(records: list[dict] | None) -> dict | None:
    """将 Token 使用记录汇总为紧凑字典，用于 SSE 事件。

    将多个使用记录合并为一个摘要字典，包含输入、输出和总 Token 数。

    Args:
        records: Token 使用记录列表，可能为 None

    Returns:
        汇总的使用情况字典，如果没有记录则返回 None

    Note:
        - 计算输入 Token 总数
        - 计算输出 Token 总数
        - 计算总 Token 数
        - 处理可能为 None 的记录值
    """
    if not records:
        return None
    return {
        "input_tokens": sum(r.get("input_tokens", 0) or 0 for r in records),
        "output_tokens": sum(r.get("output_tokens", 0) or 0 for r in records),
        "total_tokens": sum(r.get("total_tokens", 0) or 0 for r in records),
    }


def _report_subagent_usage(runtime: Any, result: Any) -> None:
    """向父级 RunJournal 报告子代理的 Token 使用情况（如果可用）。

    每个子代理任务只能报告一次使用情况（通过 usage_reported 保护）。

    Args:
        runtime: 运行时对象
        result: 子代理执行结果

    Note:
        - 检查是否已经报告过使用情况
        - 查找使用记录器
        - 记录外部 LLM 使用情况
        - 标记已报告状态
        - 处理记录过程中的异常
    """
    if getattr(result, "usage_reported", True):
        return
    records = getattr(result, "token_usage_records", None) or []
    if not records:
        return
    journal = _find_usage_recorder(runtime)
    if journal is None:
        logger.debug("No usage recorder found in runtime callbacks — subagent token usage not recorded")
        return
    try:
        journal.record_external_llm_usage_records(records)
        result.usage_reported = True
    except Exception:
        logger.warning("Failed to report subagent token usage", exc_info=True)


def _get_runtime_app_config(runtime: Any) -> "AppConfig | None":
    """从运行时对象中获取应用配置。

    尝试从运行时上下文中的 app_config 获取配置。

    Args:
        runtime: 运行时对象

    Returns:
        应用配置对象，如果未找到则返回 None
    """
    context = getattr(runtime, "context", None)
    if isinstance(context, dict):
        app_config = context.get("app_config")
        if app_config is not None:
            return cast("AppConfig", app_config)
    return None


def _merge_skill_allowlists(parent: list[str] | None, child: list[str] | None) -> list[str] | None:
    """返回在父级策略下的有效子代理技能允许列表。

    合并父级和子级的技能允许列表，确保子级只能使用父级允许的技能。

    Args:
        parent: 父级技能允许列表，可能为 None
        child: 子级技能允许列表，可能为 None

    Returns:
        合并后的技能允许列表，如果两个都为 None 则返回 None

    Note:
        - 如果父级为 None，返回子级
        - 如果子级为 None，返回父级副本
        - 如果两者都不为 None，返回子级中在父级中存在的技能
    """
    if parent is None:
        return child
    if child is None:
        return list(parent)

    parent_set = set(parent)
    return [skill for skill in child if skill in parent_set]


@tool("task", parse_docstring=True)
async def task_tool(
    runtime: Runtime,
    description: str,
    prompt: str,
    subagent_type: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> str:
    """Delegate a task to a specialized subagent that runs in its own context.

    Subagents help you:
    - Preserve context by keeping exploration and implementation separate
    - Handle complex multi-step tasks autonomously
    - Execute commands or operations in isolated contexts

    Built-in subagent types:
    - **general-purpose**: A capable agent for complex, multi-step tasks that require
      both exploration and action. Use when the task requires complex reasoning,
      multiple dependent steps, or would benefit from isolated context.
    - **bash**: Command execution specialist for running bash commands. This is only
      available when host bash is explicitly allowed or when using an isolated shell
      sandbox such as `AioSandboxProvider`.

    Additional custom subagent types may be defined in config.yaml under
    `subagents.custom_agents`. Each custom type can have its own system prompt,
    tools, skills, model, and timeout configuration. If an unknown subagent_type
    is provided, the error message will list all available types.

    When to use this tool:
    - Complex tasks requiring multiple steps or tools
    - Tasks that produce verbose output
    - When you want to isolate context from the main conversation
    - Parallel research or exploration tasks

    When NOT to use this tool:
    - Simple, single-step operations (use tools directly)
    - Tasks requiring user interaction or clarification

    Args:
        description: A short (3-5 word) description of the task for logging/display. ALWAYS PROVIDE THIS PARAMETER FIRST.
        prompt: The task description for the subagent. Be specific and clear about what needs to be done. ALWAYS PROVIDE THIS PARAMETER SECOND.
        subagent_type: The type of subagent to use. ALWAYS PROVIDE THIS PARAMETER THIRD.
    """
    # 从运行时获取应用配置
    runtime_app_config = _get_runtime_app_config(runtime)
    # 检查是否启用 Token 使用缓存
    cache_token_usage = _token_usage_cache_enabled(runtime_app_config)
    # 获取可用的子代理名称列表
    available_subagent_names = get_available_subagent_names(app_config=runtime_app_config) if runtime_app_config is not None else get_available_subagent_names()

    # 获取子代理配置
    config = get_subagent_config(subagent_type, app_config=runtime_app_config) if runtime_app_config is not None else get_subagent_config(subagent_type)
    if config is None:
        available = ", ".join(available_subagent_names)
        return f"Error: Unknown subagent type '{subagent_type}'. Available: {available}"
    if subagent_type == "bash":
        # 检查是否允许主机 bash
        host_bash_allowed = is_host_bash_allowed(runtime_app_config) if runtime_app_config is not None else is_host_bash_allowed()
        if not host_bash_allowed:
            return f"Error: {LOCAL_BASH_SUBAGENT_DISABLED_MESSAGE}"

    # 构建配置覆盖
    overrides: dict = {}

    # 技能由 SubagentExecutor 按会话加载（与 Codex 模式对齐：
    # 每个子代理根据配置加载自己的技能，作为对话项目注入）。
    # 不再追加到 system_prompt 中。

    # 从运行时提取父级上下文
    sandbox_state = None
    thread_data = None
    thread_id = None
    parent_model = None
    trace_id = None
    metadata: dict = {}

    if runtime is not None:
        sandbox_state = runtime.state.get("sandbox")
        thread_data = runtime.state.get("thread_data")
        thread_id = runtime.context.get("thread_id") if runtime.context else None
        if thread_id is None:
            thread_id = runtime.config.get("configurable", {}).get("thread_id")

        # 尝试从可配置项获取父级模型
        metadata = runtime.config.get("metadata", {})
        parent_model = metadata.get("model_name")

        # 获取或生成 trace_id 用于分布式追踪
        trace_id = metadata.get("trace_id") or str(uuid.uuid4())[:8]

    parent_available_skills = metadata.get("available_skills")
    if parent_available_skills is not None:
        overrides["skills"] = _merge_skill_allowlists(list(parent_available_skills), config.skills)

    if overrides:
        config = replace(config, **overrides)

    # 获取可用工具（排除 task 工具以防止嵌套）
    # 延迟导入以避免循环依赖
    from deerflow.tools import get_available_tools

    # 继承父代理的 tool_groups，以便子代理遵守相同的限制
    parent_tool_groups = metadata.get("tool_groups")
    resolved_app_config = runtime_app_config
    if config.model == "inherit" and parent_model is None and resolved_app_config is None:
        resolved_app_config = get_app_config()
    effective_model = resolve_subagent_model_name(config, parent_model, app_config=resolved_app_config)

    # 子代理不应启用子代理工具（防止递归嵌套）
    available_tools_kwargs = {
        "model_name": effective_model,
        "groups": parent_tool_groups,
        "subagent_enabled": False,
    }
    if resolved_app_config is not None:
        available_tools_kwargs["app_config"] = resolved_app_config
    tools = get_available_tools(**available_tools_kwargs)

    # 创建执行器
    executor_kwargs = {
        "config": config,
        "tools": tools,
        "parent_model": parent_model,
        "sandbox_state": sandbox_state,
        "thread_data": thread_data,
        "thread_id": thread_id,
        "trace_id": trace_id,
    }
    if resolved_app_config is not None:
        executor_kwargs["app_config"] = resolved_app_config
    executor = SubagentExecutor(**executor_kwargs)

    # 开始后台执行（始终异步以防止阻塞）
    # 使用 tool_call_id 作为 task_id 以获得更好的可追溯性
    task_id = executor.execute_async(prompt, task_id=tool_call_id)

    # 在后台轮询任务完成（消除 LLM 轮询的需要）
    poll_count = 0
    last_status = None
    last_message_count = 0  # 跟踪我们已经发送了多少 AI 消息
    # 轮询超时：执行超时 + 60s 缓冲，每 5s 检查一次
    max_poll_count = (config.timeout_seconds + 60) // 5

    logger.info(f"[trace={trace_id}] Started background task {task_id} (subagent={subagent_type}, timeout={config.timeout_seconds}s, polling_limit={max_poll_count} polls)")

    writer = get_stream_writer()
    # 发送任务开始消息
    writer({"type": "task_started", "task_id": task_id, "description": description})

    try:
        while True:
            result = get_background_task_result(task_id)

            if result is None:
                logger.error(f"[trace={trace_id}] Task {task_id} not found in background tasks")
                writer({"type": "task_failed", "task_id": task_id, "error": "Task disappeared from background tasks"})
                cleanup_background_task(task_id)
                return f"Error: Task {task_id} disappeared from background tasks"

            # 为调试记录状态变化
            if result.status != last_status:
                logger.info(f"[trace={trace_id}] Task {task_id} status: {result.status.value}")
                last_status = result.status

            # 检查新的 AI 消息并发送 task_running 事件
            ai_messages = result.ai_messages or []
            current_message_count = len(ai_messages)
            if current_message_count > last_message_count:
                # 为每个新消息发送 task_running 事件
                for i in range(last_message_count, current_message_count):
                    message = ai_messages[i]
                    writer(
                        {
                            "type": "task_running",
                            "task_id": task_id,
                            "message": message,
                            "message_index": i + 1,  # 用于显示的 1 基索引
                            "total_messages": current_message_count,
                        }
                    )
                    logger.info(f"[trace={trace_id}] Task {task_id} sent message #{i + 1}/{current_message_count}")
                last_message_count = current_message_count

            # 检查任务是否完成、失败或超时
            usage = _summarize_usage(getattr(result, "token_usage_records", None))
            if result.status == SubagentStatus.COMPLETED:
                _cache_subagent_usage(tool_call_id, usage, enabled=cache_token_usage)
                _report_subagent_usage(runtime, result)
                writer({"type": "task_completed", "task_id": task_id, "result": result.result, "usage": usage})
                logger.info(f"[trace={trace_id}] Task {task_id} completed after {poll_count} polls")
                cleanup_background_task(task_id)
                return f"Task Succeeded. Result: {result.result}"
            elif result.status == SubagentStatus.FAILED:
                _cache_subagent_usage(tool_call_id, usage, enabled=cache_token_usage)
                _report_subagent_usage(runtime, result)
                writer({"type": "task_failed", "task_id": task_id, "error": result.error, "usage": usage})
                logger.error(f"[trace={trace_id}] Task {task_id} failed: {result.error}")
                cleanup_background_task(task_id)
                return f"Task failed. Error: {result.error}"
            elif result.status == SubagentStatus.CANCELLED:
                _cache_subagent_usage(tool_call_id, usage, enabled=cache_token_usage)
                _report_subagent_usage(runtime, result)
                writer({"type": "task_cancelled", "task_id": task_id, "error": result.error, "usage": usage})
                logger.info(f"[trace={trace_id}] Task {task_id} cancelled: {result.error}")
                cleanup_background_task(task_id)
                return "Task cancelled by user."
            elif result.status == SubagentStatus.TIMED_OUT:
                _cache_subagent_usage(tool_call_id, usage, enabled=cache_token_usage)
                _report_subagent_usage(runtime, result)
                writer({"type": "task_timed_out", "task_id": task_id, "error": result.error, "usage": usage})
                logger.warning(f"[trace={trace_id}] Task {task_id} timed out: {result.error}")
                cleanup_background_task(task_id)
                return f"Task timed out. Error: {result.error}"

            # 仍在运行，等待下次轮询
            await asyncio.sleep(5)
            poll_count += 1

            # 轮询超时作为安全网（以防线程池超时不工作）
            # 设置为执行超时 + 60s 缓冲，在 5s 轮询间隔中
            # 这捕获后台任务卡住的边缘情况
            # 注意：我们不在此处调用 cleanup_background_task，因为任务可能
            # 仍在后台运行。当执行器完成并设置终端状态时，清理将在那时发生。
            if poll_count > max_poll_count:
                timeout_minutes = config.timeout_seconds // 60
                logger.error(f"[trace={trace_id}] Task {task_id} polling timed out after {poll_count} polls (should have been caught by thread pool timeout)")
                _report_subagent_usage(runtime, result)
                usage = _summarize_usage(getattr(result, "token_usage_records", None))
                _cache_subagent_usage(tool_call_id, usage, enabled=cache_token_usage)
                writer({"type": "task_timed_out", "task_id": task_id, "usage": usage})
                return f"Task polling timed out after {timeout_minutes} minutes. This may indicate the background task is stuck. Status: {result.status.value}"
    except asyncio.CancelledError:
        # 信号后台子代理线程合作性停止。
        request_cancel_background_task(task_id)

        # 等待（受保护）子代理达到终端状态，以便在父级 RunJournal
        # 报告最终 Token 使用快照之前，父级工作者完成 get_completion_data() 的持久化。
        terminal_result = None
        try:
            terminal_result = await asyncio.shield(_await_subagent_terminal(task_id, max_poll_count))
        except asyncio.CancelledError:
            pass

        # 报告子代理收集的任何内容（即使我们超时了）。
        final_result = terminal_result or get_background_task_result(task_id)
        if final_result is not None:
            _report_subagent_usage(runtime, final_result)
        if final_result is not None and _is_subagent_terminal(final_result):
            cleanup_background_task(task_id)
        else:
            _schedule_deferred_subagent_cleanup(task_id, trace_id, max_poll_count)
        _subagent_usage_cache.pop(tool_call_id, None)
        raise
    except Exception:
        _subagent_usage_cache.pop(tool_call_id, None)
        raise
