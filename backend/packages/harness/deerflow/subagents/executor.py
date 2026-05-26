"""子代理执行引擎。

本模块提供子代理的完整执行能力，包括：
- 同步和异步执行模式
- 隔离事件循环管理（防止与父代理冲突）
- 后台任务调度和轮询
- 实时状态更新和进度报告
- Token 使用统计收集
- 协作式取消和超时处理
- 技能加载和工具过滤
- 分布式追踪（trace_id）

架构设计：
- 使用持久化事件循环避免每次执行创建新循环
- 线程池调度器管理后台任务
- ContextVar 复制保持上下文隔离
- 线程安全的结果存储和状态管理
"""

import asyncio
import atexit
import logging
import threading
import uuid
from collections.abc import Callable, Coroutine
from concurrent.futures import Future, ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from contextvars import Context, copy_context
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from langchain.agents import create_agent
from langchain.tools import BaseTool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from deerflow.agents.thread_state import SandboxState, ThreadDataState, ThreadState
from deerflow.config import get_app_config
from deerflow.config.app_config import AppConfig
from deerflow.models import create_chat_model
from deerflow.skills.tool_policy import filter_tools_by_skill_allowed_tools
from deerflow.skills.types import Skill
from deerflow.subagents.config import SubagentConfig, resolve_subagent_model_name
from deerflow.subagents.token_collector import SubagentTokenCollector

logger = logging.getLogger(__name__)


# 程序启动时清理可能存在的旧事件循环实例
_previous_shutdown_isolated_subagent_loop = globals().get("_shutdown_isolated_subagent_loop")
if callable(_previous_shutdown_isolated_subagent_loop):
    atexit.unregister(_previous_shutdown_isolated_subagent_loop)
    _previous_shutdown_isolated_subagent_loop()


class SubagentStatus(Enum):
    """子代理执行状态枚举。

    Attributes:
        PENDING: 任务已创建，等待执行
        RUNNING: 任务正在执行
        COMPLETED: 任务成功完成
        FAILED: 任务执行失败
        CANCELLED: 任务被用户取消
        TIMED_OUT: 任务执行超时
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"

    @property
    def is_terminal(self) -> bool:
        """检查是否为终端状态（不可再转换的状态）。

        Returns:
            True 如果是终端状态，否则 False
        """
        return self in {
            type(self).COMPLETED,
            type(self).FAILED,
            type(self).CANCELLED,
            type(self).TIMED_OUT,
        }


@dataclass
class SubagentResult:
    """子代理执行结果。

    该数据类保存子代理执行的完整状态和结果信息。

    Attributes:
        task_id: 唯一标识符，用于追踪执行
        trace_id: 分布式追踪 ID（连接父代理和子代理日志）
        status: 当前执行状态
        result: 最终结果消息（如果完成）
        error: 错误消息（如果失败）
        started_at: 执行开始时间
        completed_at: 执行完成时间
        ai_messages: 执行期间生成的 AI 消息列表（字典格式）
        token_usage_records: Token 使用记录列表
        usage_reported: 是否已报告使用情况
        cancel_event: 用于协作取消的线程事件
        _state_lock: 用于保护状态转换的线程锁
    """

    task_id: str
    trace_id: str
    status: SubagentStatus
    result: str | None = None
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    ai_messages: list[dict[str, Any]] | None = None
    token_usage_records: list[dict[str, int | str]] = field(default_factory=list)
    usage_reported: bool = False
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    _state_lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self):
        """初始化可变默认值。"""
        if self.ai_messages is None:
            self.ai_messages = []

    def try_set_terminal(
        self,
        status: SubagentStatus,
        *,
        result: str | None = None,
        error: str | None = None,
        completed_at: datetime | None = None,
        ai_messages: list[dict[str, Any]] | None = None,
        token_usage_records: list[dict[str, int | str]] | None = None,
    ) -> bool:
        """原子性地设置终端状态（仅一次）。

        后台超时/取消和执行工作者可能在同一结果对象上竞争。
        第一个终端转换获胜；延迟的终端写入不能更改状态或有效载荷字段。

        Args:
            status: 要设置的终端状态
            result: 成功结果消息
            error: 错误消息
            completed_at: 完成时间（默认当前时间）
            ai_messages: AI 消息列表
            token_usage_records: Token 使用记录

        Returns:
            True 如果成功设置，False 如果已经是终端状态

        Raises:
            ValueError: 如果 status 不是终端状态
        """
        if not status.is_terminal:
            raise ValueError(f"Status {status} is not terminal")

        with self._state_lock:
            if self.status.is_terminal:
                return False

            if result is not None:
                self.result = result
            if error is not None:
                self.error = error
            if ai_messages is not None:
                self.ai_messages = ai_messages
            if token_usage_records is not None:
                self.token_usage_records = token_usage_records
            self.completed_at = completed_at or datetime.now()
            self.status = status
            return True


# 后台任务结果的全局存储
_background_tasks: dict[str, SubagentResult] = {}
_background_tasks_lock = threading.Lock()

# 后台任务调度和编排的线程池
_scheduler_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="subagent-scheduler-")

# 持久化事件循环，用于从已运行的父循环触发的隔离子代理执行。
# 重用这个长期存在的循环避免每次执行创建新循环并关闭绑定到它的异步资源。
_isolated_subagent_loop: asyncio.AbstractEventLoop | None = None
_isolated_subagent_loop_thread: threading.Thread | None = None
_isolated_subagent_loop_started: threading.Event | None = None
_isolated_subagent_loop_lock = threading.Lock()


def _run_isolated_subagent_loop(
    loop: asyncio.AbstractEventLoop,
    started_event: threading.Event,
) -> None:
    """在专用守护线程中运行持久化隔离子代理循环。

    Args:
        loop: 事件循环实例
        started_event: 启动事件，用于信号循环已准备就绪
    """
    asyncio.set_event_loop(loop)
    loop.call_soon(started_event.set)
    try:
        loop.run_forever()
    finally:
        started_event.clear()


def _shutdown_isolated_subagent_loop() -> None:
    """停止并关闭持久化隔离子代理循环。

    程序退出时调用，确保事件循环正确清理。
    """
    global _isolated_subagent_loop, _isolated_subagent_loop_thread, _isolated_subagent_loop_started

    with _isolated_subagent_loop_lock:
        loop = _isolated_subagent_loop
        thread = _isolated_subagent_loop_thread
        _isolated_subagent_loop = None
        _isolated_subagent_loop_thread = None
        _isolated_subagent_loop_started = None

    if loop is None:
        return

    if loop.is_running():
        loop.call_soon_threadsafe(loop.stop)

    if thread is not None and thread.is_alive() and thread is not threading.current_thread():
        thread.join(timeout=1)

    thread_stopped = thread is None or not thread.is_alive()
    loop_stopped = not loop.is_running()

    if not loop.is_closed():
        if thread_stopped and loop_stopped:
            loop.close()
        else:
            logger.warning(
                "Skipping close of isolated subagent loop because shutdown did not complete within timeout (thread_alive=%s, loop_running=%s)",
                thread is not None and thread.is_alive(),
                loop.is_running(),
            )


atexit.register(_shutdown_isolated_subagent_loop)


def _get_isolated_subagent_loop() -> asyncio.AbstractEventLoop:
    """返回隔离子代理执行使用的持久化事件循环。

    如果循环不存在或不再可用，则创建新循环。

    Returns:
        事件循环实例

    Raises:
        RuntimeError: 如果启动循环超时
    """
    global _isolated_subagent_loop, _isolated_subagent_loop_thread, _isolated_subagent_loop_started
    with _isolated_subagent_loop_lock:
        thread_is_alive = _isolated_subagent_loop_thread is not None and _isolated_subagent_loop_thread.is_alive()
        loop_is_usable = _isolated_subagent_loop is not None and not _isolated_subagent_loop.is_closed() and _isolated_subagent_loop.is_running() and thread_is_alive

        if not loop_is_usable:
            loop = asyncio.new_event_loop()
            started_event = threading.Event()
            thread = threading.Thread(
                target=_run_isolated_subagent_loop,
                args=(loop, started_event),
                name="subagent-persistent-loop",
                daemon=True,
            )
            thread.start()
            if not started_event.wait(timeout=5):
                loop.call_soon_threadsafe(loop.stop)
                thread.join(timeout=1)
                loop.close()
                raise RuntimeError("Timed out starting isolated subagent event loop")
            _isolated_subagent_loop = loop
            _isolated_subagent_loop_thread = thread
            _isolated_subagent_loop_started = started_event

        if _isolated_subagent_loop is None:
            raise RuntimeError("Isolated subagent event loop is not initialized")
        return _isolated_subagent_loop


def _submit_to_isolated_loop_in_context(
    context: Context,
    coro_factory: Callable[[], Coroutine[Any, Any, SubagentResult]],
) -> Future[SubagentResult]:
    """在保留 ContextVar 状态的情况下向隔离循环提交协程。

    Args:
        context: 父级上下文，包含所有 ContextVar 的值
        coro_factory: 创建协程的工厂函数

    Returns:
        Future 对象，用于获取执行结果
    """
    return context.run(
        lambda: asyncio.run_coroutine_threadsafe(
            coro_factory(),
            _get_isolated_subagent_loop(),
        )
    )


def _filter_tools(
    all_tools: list[BaseTool],
    allowed: list[str] | None,
    disallowed: list[str] | None,
) -> list[BaseTool]:
    """根据子代理配置过滤工具。

    Args:
        all_tools: 所有可用工具的列表
        allowed: 可选的工具名称允许列表。如果提供，只包含这些工具。
        disallowed: 可选的工具名称禁止列表。这些工具始终被排除。

    Returns:
        过滤后的工具列表
    """
    filtered = all_tools

    # 如果指定了允许列表，应用过滤
    if allowed is not None:
        allowed_set = set(allowed)
        filtered = [t for t in filtered if t.name in allowed_set]

    # 应用禁止列表
    if disallowed is not None:
        disallowed_set = set(disallowed)
        filtered = [t for t in filtered if t.name not in disallowed_set]

    return filtered


class SubagentExecutor:
    """子代理执行器。

    负责初始化和执行子代理任务，包括：
    - 模型和工具配置
    - 技能加载和工具过滤
    - 同步和异步执行模式
    - 后台任务调度
    - 分布式追踪
    """

    def __init__(
        self,
        config: SubagentConfig,
        tools: list[BaseTool],
        app_config: AppConfig | None = None,
        parent_model: str | None = None,
        sandbox_state: SandboxState | None = None,
        thread_data: ThreadDataState | None = None,
        thread_id: str | None = None,
        trace_id: str | None = None,
    ):
        """初始化执行器。

        Args:
            config: 子代理配置
            tools: 所有可用工具的列表（将被过滤）
            app_config: 已解析的 AppConfig。当为 None 时，``_create_agent`` 将
                回退到 ``get_app_config()``（与主代理工厂模式匹配）。
            parent_model: 父代理的模型名称，用于继承
            sandbox_state: 父代理的沙箱状态
            thread_data: 父代理的线程数据
            thread_id: 线程 ID，用于沙箱操作
            trace_id: 父代理的追踪 ID，用于分布式追踪
        """
        self.config = config
        self.app_config = app_config
        self.parent_model = parent_model
        # 仅在不需要加载 config.yaml 时预先解析；否则延迟到 _create_agent
        # （已加载 app_config），以便单元测试可以在没有配置文件的情况下构造执行器
        if config.model != "inherit" or parent_model is not None or app_config is not None:
            self.model_name: str | None = resolve_subagent_model_name(config, parent_model, app_config=app_config)
        else:
            self.model_name = None
        self.sandbox_state = sandbox_state
        self.thread_data = thread_data
        self.thread_id = thread_id
        # 如果未提供则生成 trace_id（用于顶层调用）
        self.trace_id = trace_id or str(uuid.uuid4())[:8]

        self._base_tools = _filter_tools(
            tools,
            config.tools,
            config.disallowed_tools,
        )
        self.tools = self._base_tools

        logger.info(f"[trace={self.trace_id}] SubagentExecutor initialized: {config.name} with {len(self.tools)} tools")

    def _create_agent(self, tools: list[BaseTool] | None = None):
        """创建代理实例。

        Args:
            tools: 可选的工具列表，如果不提供则使用 self.tools

        Returns:
            创建好的代理实例

        Note:
            - system_prompt 包含在初始状态消息中（见 _build_initial_state）
            - 避免创建多个 SystemMessage，某些 LLM API 不支持
        """
        app_config = self.app_config or get_app_config()
        if self.model_name is None:
            self.model_name = resolve_subagent_model_name(self.config, self.parent_model, app_config=app_config)
        model = create_chat_model(name=self.model_name, thinking_enabled=False, app_config=app_config)

        from deerflow.agents.middlewares.tool_error_handling_middleware import build_subagent_runtime_middlewares

        # 与主代理重用共享中间件组合
        middlewares = build_subagent_runtime_middlewares(app_config=app_config, model_name=self.model_name, lazy_init=True)

        # system_prompt 包含在初始状态消息中（见 _build_initial_state）
        # 避免多个 SystemMessage，某些 LLM API 不支持
        return create_agent(
            model=model,
            tools=tools if tools is not None else self.tools,
            middleware=middlewares,
            system_prompt=None,
            state_schema=ThreadState,
        )

    async def _load_skills(self) -> list[Skill]:
        """根据 config.skills 加载启用的技能元数据。

        Returns:
            加载的技能列表

        Raises:
            Exception: 如果技能加载失败

        Note:
            - 如果 config.skills 为空列表，跳过加载
            - 使用 asyncio.to_thread 避免阻塞事件循环（LangGraph ASGI 要求）
            - 如果 config.skills 为 None，加载所有启用的技能
            - 如果 config.skills 有值，只加载白名单中的技能
        """
        if self.config.skills is not None and len(self.config.skills) == 0:
            logger.info(f"[trace={self.trace_id}] Subagent {self.config.name} skills=[] — skipping skill loading")
            return []

        try:
            from deerflow.skills.storage import get_or_new_skill_storage

            storage_kwargs = {"app_config": self.app_config} if self.app_config is not None else {}
            storage = await asyncio.to_thread(get_or_new_skill_storage, **storage_kwargs)
            # 使用 asyncio.to_thread 避免阻塞事件循环（LangGraph ASGI 要求）
            all_skills = await asyncio.to_thread(storage.load_skills, enabled_only=True)
            logger.info(f"[trace={self.trace_id}] Subagent {self.config.name} loaded {len(all_skills)} enabled skills from disk")
        except Exception:
            logger.exception(f"[trace={self.trace_id}] Failed to load skills for subagent {self.config.name}")
            raise

        if not all_skills:
            logger.info(f"[trace={self.trace_id}] Subagent {self.config.name} no enabled skills found")
            return []

        # 按 config.skills 白名单过滤
        if self.config.skills is not None:
            allowed = set(self.config.skills)
            return [s for s in all_skills if s.name in allowed]
        return all_skills

    def _apply_skill_allowed_tools(self, skills: list[Skill]) -> list[BaseTool]:
        """根据技能允许的工具列表过滤工具。

        Args:
            skills: 技能列表

        Returns:
            过滤后的工具列表
        """
        return filter_tools_by_skill_allowed_tools(self._base_tools, skills)

    async def _load_skill_messages(self, skills: list[Skill]) -> list[SystemMessage]:
        """根据 config.skills 加载技能内容作为对话项目。

        与 Codex 模式对齐：每个子代理按会话加载自己的技能，
        并将它们作为对话项目（开发者消息）注入，而不是作为系统提示文本。
        config.skills 白名单控制加载哪些技能：
        - None: 加载所有启用的技能
        - []: 不加载技能
        - ["skill-a", "skill-b"]: 只加载这些技能

        Args:
            skills: 技能列表

        Returns:
            包含技能内容的 SystemMessage 列表
        """
        if not skills:
            return []

        # 读取每个技能的 SKILL.md 内容并创建对话项目
        messages = []
        for skill in skills:
            try:
                content = await asyncio.to_thread(skill.skill_file.read_text, encoding="utf-8")
                content = content.strip()
                if content:
                    messages.append(SystemMessage(content=f'<skill name="{skill.name}">\n{content}\n</skill>'))
                    logger.info(f"[trace={self.trace_id}] Subagent {self.config.name} loaded skill: {skill.name}")
            except Exception:
                logger.debug(f"[trace={self.trace_id}] Failed to read skill {skill.name}", exc_info=True)

        return messages

    async def _build_initial_state(self, task: str) -> tuple[dict[str, Any], list[BaseTool]]:
        """构建代理执行的初始状态。

        Args:
            task: 任务描述

        Returns:
            二元组：（初始状态字典，按加载的技能元数据过滤的工具列表）

        Note:
            - 将 system_prompt 和技能合并为单个 SystemMessage
            - 某些 LLM API 拒绝多个 SystemMessage
            - 传递父代理的沙箱和线程数据
        """

        # 加载技能作为对话项目（Codex 模式）
        skills = await self._load_skills()
        filtered_tools = self._apply_skill_allowed_tools(skills)
        skill_messages = await self._load_skill_messages(skills)

        # 将 system_prompt 和技能合并为单个 SystemMessage。
        # 某些 LLM API 拒绝多个 SystemMessage，报错 "System message must be at the beginning."
        system_parts: list[str] = []
        if self.config.system_prompt:
            system_parts.append(self.config.system_prompt)
        for skill_msg in skill_messages:
            system_parts.append(skill_msg.content)

        messages: list[Any] = []
        if system_parts:
            messages.append(SystemMessage(content="\n\n".join(system_parts)))

        # 然后是实际任务
        messages.append(HumanMessage(content=task))

        state: dict[str, Any] = {
            "messages": messages,
        }

        # 从父代理传递沙箱和线程数据
        if self.sandbox_state is not None:
            state["sandbox"] = self.sandbox_state
        if self.thread_data is not None:
            state["thread_data"] = self.thread_data

        return state, filtered_tools

    async def _aexecute(self, task: str, result_holder: SubagentResult | None = None) -> SubagentResult:
        """异步执行任务。

        Args:
            task: 子代理的任务描述
            result_holder: 可选的预创建结果对象，在执行期间更新

        Returns:
            包含执行结果的 SubagentResult

        Note:
            - 使用流式执行以获取实时更新
            - 收集 AI 消息用于进度报告
            - 支持协作式取消
            - 收集 Token 使用统计
        """
        if result_holder is not None:
            # 使用提供的结果持有者（用于带实时更新的异步执行）
            result = result_holder
        else:
            # 为同步执行创建新结果
            task_id = str(uuid.uuid4())[:8]
            result = SubagentResult(
                task_id=task_id,
                trace_id=self.trace_id,
                status=SubagentStatus.RUNNING,
                started_at=datetime.now(),
            )
        ai_messages = result.ai_messages
        if ai_messages is None:
            ai_messages = []
            result.ai_messages = ai_messages

        collector: SubagentTokenCollector | None = None
        try:
            state, filtered_tools = await self._build_initial_state(task)
            agent = self._create_agent(filtered_tools)

            # 子代理 LLM 调用的 Token 收集器
            collector_caller = f"subagent:{self.config.name}"
            collector = SubagentTokenCollector(caller=collector_caller)

            # 构建包含 thread_id 的配置，用于沙箱访问和递归限制
            run_config: RunnableConfig = {
                "recursion_limit": self.config.max_turns,
                "callbacks": [collector],
                "tags": [collector_caller],
            }
            context: dict[str, Any] = {}
            if self.thread_id:
                run_config["configurable"] = {"thread_id": self.thread_id}
                context["thread_id"] = self.thread_id
            if self.app_config is not None:
                context["app_config"] = self.app_config

            logger.info(f"[trace={self.trace_id}] Subagent {self.config.name} starting async execution with max_turns={self.config.max_turns}")

            # 使用 stream 而不是 invoke 以获取实时更新
            # 这允许我们在 AI 消息生成时收集它们
            final_state = None

            # 预检查：如果已经在流式传输开始前被取消，立即退出
            if result.cancel_event.is_set():
                logger.info(f"[trace={self.trace_id}] Subagent {self.config.name} cancelled before streaming")
                result.try_set_terminal(
                    SubagentStatus.CANCELLED,
                    error="Cancelled by user",
                    token_usage_records=collector.snapshot_records(),
                )
                return result

            async for chunk in agent.astream(state, config=run_config, context=context, stream_mode="values"):  # type: ignore[arg-type]
                # 协作式取消：检查父代理是否请求停止。
                # 注意：取消只在 astream 迭代边界检测，
                # 所以单次迭代中的长时间运行的工具调用不会被中断，直到下一个 chunk 产生。
                if result.cancel_event.is_set():
                    logger.info(f"[trace={self.trace_id}] Subagent {self.config.name} cancelled by parent")
                    result.try_set_terminal(
                        SubagentStatus.CANCELLED,
                        error="Cancelled by user",
                        token_usage_records=collector.snapshot_records(),
                    )
                    return result

                final_state = chunk

                # 从当前状态提取 AI 消息
                messages = chunk.get("messages", [])
                if messages:
                    last_message = messages[-1]
                    # 检查这是否是一个新的 AI 消息
                    if isinstance(last_message, AIMessage):
                        # 将消息转换为字典以进行序列化
                        message_dict = last_message.model_dump()
                        # 只有在列表中不存在时才添加（避免重复）
                        # 如果有 message ID 则通过 ID 比较，否则比较完整字典
                        message_id = message_dict.get("id")
                        is_duplicate = False
                        if message_id:
                            is_duplicate = any(msg.get("id") == message_id for msg in ai_messages)
                        else:
                            is_duplicate = message_dict in ai_messages

                        if not is_duplicate:
                            ai_messages.append(message_dict)
                            logger.info(f"[trace={self.trace_id}] Subagent {self.config.name} captured AI message #{len(ai_messages)}")

            logger.info(f"[trace={self.trace_id}] Subagent {self.config.name} completed async execution")
            token_usage_records = collector.snapshot_records()
            final_result: str | None = None

            if final_state is None:
                logger.warning(f"[trace={self.trace_id}] Subagent {self.config.name} no final state")
                final_result = "No response generated"
            else:
                # 提取最终消息 - 查找最后一个 AIMessage
                messages = final_state.get("messages", [])
                logger.info(f"[trace={self.trace_id}] Subagent {self.config.name} final messages count: {len(messages)}")

                # 在对话中查找最后一个 AIMessage
                last_ai_message = None
                for msg in reversed(messages):
                    if isinstance(msg, AIMessage):
                        last_ai_message = msg
                        break

                if last_ai_message is not None:
                    content = last_ai_message.content
                    # 为最终结果处理 str 和 list 两种内容类型
                    if isinstance(content, str):
                        final_result = content
                    elif isinstance(content, list):
                        # 从内容块列表中提取文本，仅用于最终结果。
                        # 直接连接原始字符串块，但保留完整文本块之间的分隔以提高可读性。
                        text_parts = []
                        pending_str_parts = []
                        for block in content:
                            if isinstance(block, str):
                                pending_str_parts.append(block)
                            elif isinstance(block, dict):
                                if pending_str_parts:
                                    text_parts.append("".join(pending_str_parts))
                                    pending_str_parts.clear()
                                text_val = block.get("text")
                                if isinstance(text_val, str):
                                    text_parts.append(text_val)
                        if pending_str_parts:
                            text_parts.append("".join(pending_str_parts))
                        final_result = "\n".join(text_parts) if text_parts else "No text content in response"
                    else:
                        final_result = str(content)
                elif messages:
                    # 回退：如果没找到 AIMessage，使用最后一条消息
                    last_message = messages[-1]
                    logger.warning(f"[trace={self.trace_id}] Subagent {self.config.name} no AIMessage found, using last message: {type(last_message)}")
                    raw_content = last_message.content if hasattr(last_message, "content") else str(last_message)
                    if isinstance(raw_content, str):
                        final_result = raw_content
                    elif isinstance(raw_content, list):
                        parts = []
                        pending_str_parts = []
                        for block in raw_content:
                            if isinstance(block, str):
                                pending_str_parts.append(block)
                            elif isinstance(block, dict):
                                if pending_str_parts:
                                    parts.append("".join(pending_str_parts))
                                    pending_str_parts.clear()
                                text_val = block.get("text")
                                if isinstance(text_val, str):
                                    parts.append(text_val)
                        if pending_str_parts:
                            parts.append("".join(pending_str_parts))
                        final_result = "\n".join(parts) if parts else "No text content in response"
                    else:
                        final_result = str(raw_content)
                else:
                    logger.warning(f"[trace={self.trace_id}] Subagent {self.config.name} no messages in final state")
                    final_result = "No response generated"

            if final_result is None:
                final_result = "No response generated"

            result.try_set_terminal(
                SubagentStatus.COMPLETED,
                result=final_result,
                token_usage_records=token_usage_records,
            )

        except Exception as e:
            logger.exception(f"[trace={self.trace_id}] Subagent {self.config.name} async execution failed")
            result.try_set_terminal(
                SubagentStatus.FAILED,
                error=str(e),
                token_usage_records=collector.snapshot_records() if collector is not None else None,
            )

        return result

    def _execute_in_isolated_loop(self, task: str, result_holder: SubagentResult | None = None) -> SubagentResult:
        """在持久化隔离子代理循环上执行任务。

        此方法被同步 ``execute()`` 路径使用，当调用者已经在事件循环中运行时。
        因为 ``execute()`` 是同步 API，此路径在实际协程在长期存在的隔离循环上运行时
        阻塞调用者。重用该循环避免共享异步客户端（如 httpx 客户端）
        被绑定到每次执行关闭的短期循环。

        Args:
            task: 任务描述
            result_holder: 可选的结果持有者

        Returns:
            执行结果

        Raises:
            FuturesTimeoutError: 如果执行超时
        """
        future: Future[SubagentResult] | None = None
        parent_context = copy_context()
        try:
            future = _submit_to_isolated_loop_in_context(
                parent_context,
                lambda: self._aexecute(task, result_holder),
            )
            return future.result(timeout=self.config.timeout_seconds)
        except FuturesTimeoutError:
            if result_holder is not None:
                result_holder.cancel_event.set()
            if future is not None:
                future.cancel()
            raise
        except Exception:
            if future is None:
                logger.debug(
                    f"[trace={self.trace_id}] Failed to submit subagent {self.config.name} to the isolated event loop",
                    exc_info=True,
                )
            else:
                logger.debug(
                    f"[trace={self.trace_id}] Subagent {self.config.name} failed while executing on the isolated event loop",
                    exc_info=True,
                )
            raise

    def execute(self, task: str, result_holder: SubagentResult | None = None) -> SubagentResult:
        """同步执行任务（异步执行的包装器）。

        此方法在新的事件循环中运行异步执行，允许在
        线程池中使用异步工具（如 MCP 工具）。

        当从已在运行的事件循环内调用时（例如，当父代理是异步时），
        此方法同步等待持久化隔离循环，以避免与共享
        异步原语（如 httpx 客户端）的事件循环冲突。

        Args:
            task: 子代理的任务描述
            result_holder: 可选的预创建结果对象，在执行期间更新

        Returns:
            包含执行结果的 SubagentResult
        """
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop is not None and loop.is_running():
                logger.debug(f"[trace={self.trace_id}] Subagent {self.config.name} detected running event loop, using isolated loop")
                return self._execute_in_isolated_loop(task, result_holder)

            # 标准路径：没有运行的事件循环，使用 asyncio.run
            return asyncio.run(self._aexecute(task, result_holder))
        except Exception as e:
            logger.exception(f"[trace={self.trace_id}] Subagent {self.config.name} execution failed")
            # 如果没有结果对象，创建带错误的新结果
            if result_holder is not None:
                result = result_holder
            else:
                result = SubagentResult(
                    task_id=str(uuid.uuid4())[:8],
                    trace_id=self.trace_id,
                    status=SubagentStatus.RUNNING,
                )
            result.try_set_terminal(SubagentStatus.FAILED, error=str(e))
            return result

    def execute_async(self, task: str, task_id: str | None = None) -> str:
        """在后台启动任务执行。

        Args:
            task: 子代理的任务描述
            task_id: 可选的任务 ID。如果未提供，将生成随机 UUID。

        Returns:
            任务 ID，可用于稍后检查状态

        Note:
            - 创建初始待处理结果
            - 提交到调度器线程池
            - 直接提交到持久化隔离循环，避免 execute() 创建临时循环
        """
        # 使用提供的 task_id 或生成新的
        if task_id is None:
            task_id = str(uuid.uuid4())[:8]

        # 创建初始待处理结果
        result = SubagentResult(
            task_id=task_id,
            trace_id=self.trace_id,
            status=SubagentStatus.PENDING,
        )

        logger.info(f"[trace={self.trace_id}] Subagent {self.config.name} starting async execution, task_id={task_id}, timeout={self.config.timeout_seconds}s")

        with _background_tasks_lock:
            _background_tasks[task_id] = result

        parent_context = copy_context()

        # 提交到调度器线程池
        def run_task():
            """在后台线程中运行任务。

            Note:
                - 更新任务状态为 RUNNING
                - 直接提交到持久化隔离循环
                - 处理超时和异常
                - 使用协作式取消机制
            """
            with _background_tasks_lock:
                _background_tasks[task_id].status = SubagentStatus.RUNNING
                _background_tasks[task_id].started_at = datetime.now()
                result_holder = _background_tasks[task_id]

            try:
                # 直接提交执行到持久化隔离循环，避免后台路径通过 execute() 创建临时循环。
                execution_future = _submit_to_isolated_loop_in_context(
                    parent_context,
                    lambda: self._aexecute(task, result_holder),
                )
                try:
                    # 等待执行带超时
                    execution_future.result(timeout=self.config.timeout_seconds)
                except FuturesTimeoutError:
                    logger.error(f"[trace={self.trace_id}] Subagent {self.config.name} execution timed out after {self.config.timeout_seconds}s")
                    # 信号协作式取消并取消 future
                    result_holder.cancel_event.set()
                    result_holder.try_set_terminal(
                        SubagentStatus.TIMED_OUT,
                        error=f"Execution timed out after {self.config.timeout_seconds} seconds",
                    )
                    execution_future.cancel()
            except Exception as e:
                logger.exception(f"[trace={self.trace_id}] Subagent {self.config.name} async execution failed")
                with _background_tasks_lock:
                    task_result = _background_tasks[task_id]
                task_result.try_set_terminal(SubagentStatus.FAILED, error=str(e))

        _scheduler_pool.submit(run_task)
        return task_id


# 最大并发子代理数
# 用于限制同时执行的子代理数量，避免资源耗尽
MAX_CONCURRENT_SUBAGENTS = 3


def request_cancel_background_task(task_id: str) -> None:
    """请求取消运行中的后台任务。

    设置任务的 cancel_event，由 ``_aexecute`` 在 ``agent.astream()``
    迭代期间协作检查。这允许子代理线程（不能通过 ``Future.cancel()``
    强制终止）在下一次迭代边界停止。

    Args:
        task_id: 要取消的任务 ID

    Note:
        - 使用协作式取消机制，非强制终止
        - 取消请求是异步的，任务可能不会立即停止
        - 线程安全，使用锁保护
    """
    with _background_tasks_lock:
        result = _background_tasks.get(task_id)
        if result is not None:
            result.cancel_event.set()
            logger.info("Requested cancellation for background task %s", task_id)


def get_background_task_result(task_id: str) -> SubagentResult | None:
    """获取后台任务的结果。

    用于查询后台任务的执行状态和结果。
    调用者可以轮询此方法直到任务进入终端状态。

    Args:
        task_id: execute_async 返回的任务 ID

    Returns:
        如果找到则返回 SubagentResult，否则返回 None

    Note:
        - 线程安全，使用锁保护
        - 返回结果对象的引用，调用者应只读访问
    """
    with _background_tasks_lock:
        return _background_tasks.get(task_id)


def list_background_tasks() -> list[SubagentResult]:
    """列出所有后台任务。

    返回所有后台任务的快照，包括待处理、运行中和已完成的任务。
    可用于监控和调试目的。

    Returns:
        所有 SubagentResult 实例的列表（副本）

    Note:
        - 返回列表的副本，不会影响内部状态
        - 线程安全，使用锁保护
    """
    with _background_tasks_lock:
        return list(_background_tasks.values())


def cleanup_background_task(task_id: str) -> None:
    """清理已完成的任务，释放内存。

    应该在 task_tool 完成轮询并返回结果后调用。
    这防止已完成任务累积导致内存泄漏。

    只移除处于终端状态（COMPLETED/FAILED/TIMED_OUT）的任务，
    避免与后台执行器仍在更新任务条目产生竞争条件。

    Args:
        task_id: 要移除的任务 ID

    Note:
        - 只清理终端状态的任务
        - 如果任务不存在，仅记录 debug 日志
        - 线程安全，使用锁保护
        - 防止内存泄漏的重要机制
    """
    with _background_tasks_lock:
        result = _background_tasks.get(task_id)
        if result is None:
            # 没有需要清理的；可能已被移除
            logger.debug("Requested cleanup for unknown background task %s", task_id)
            return

        # 只清理处于终端状态的任务，避免与后台执行器仍在更新任务条目产生竞争
        if result.status.is_terminal or result.completed_at is not None:
            del _background_tasks[task_id]
            logger.debug("Cleaned up background task: %s", task_id)
        else:
            logger.debug(
                "Skipping cleanup for non-terminal background task %s (status=%s)",
                task_id,
                result.status.value if hasattr(result.status, "value") else result.status,
            )
