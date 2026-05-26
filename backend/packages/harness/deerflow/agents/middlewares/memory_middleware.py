"""记忆中间件模块。

本模块实现了Agent执行完成后的异步记忆更新机制（重写了after_agent函数）。
核心思路是：在每次Agent对话结束后（即模型已经给用户输出了最终答案），将经过过滤的对话消息入队，
由后台的防抖队列（debounce queue）批量处理，通过LLM总结提取用户偏好、事实和上下文信息，并原子性地写入持久化存储。

关键设计要点：
- 仅保留用户输入和最终助手回复，过滤掉工具调用等中间过程消息，以避免噪声信息干扰记忆提取质量。
- 在入队时捕获user_id，而非在后台Timer线程中读取ContextVar，因为Python的threading.Timer在新线程上执行，
ContextVar的值不会自动传播到新线程上下文中。
- 支持按Agent维度隔离记忆（通过agent_name参数），也可使用全局记忆。
"""

import logging
from typing import TYPE_CHECKING, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langgraph.config import get_config
from langgraph.runtime import Runtime

from deerflow.agents.memory.message_processing import detect_correction, detect_reinforcement, filter_messages_for_memory
from deerflow.agents.memory.queue import get_memory_queue
from deerflow.config.memory_config import get_memory_config
from deerflow.runtime.user_context import get_effective_user_id

if TYPE_CHECKING:
    from deerflow.config.memory_config import MemoryConfig

logger = logging.getLogger(__name__)


class MemoryMiddlewareState(AgentState):
    """记忆中间件的状态模式，与 ThreadState 的 schema 保持兼容。

    继承自 AgentState，当前不需要额外字段，
    仅用于为 MemoryMiddleware 提供类型安全的状态类型标注。
    """

    pass


class MemoryMiddleware(AgentMiddleware[MemoryMiddlewareState]):
    """记忆中间件：在 Agent 执行完成后将对话入队，以触发异步记忆更新。

    工作流程：
    1. Agent 执行完成后，在 after_agent 钩子中将当前对话入队
    2. 仅保留用户输入和最终助手回复（过滤掉工具调用等中间步骤消息）
    3. 队列使用防抖机制（debounce），将短时间内同一线程的多次更新合并为一次
    4. 后台线程通过 LLM 总结提取记忆更新，并原子性地写入存储文件
    """

    state_schema = MemoryMiddlewareState

    def __init__(self, agent_name: str | None = None, *, memory_config: "MemoryConfig | None" = None):
        """初始化记忆中间件。

        Args:
            agent_name: Agent 名称。如果提供，则按 Agent 维度隔离记忆存储；
                如果为 None，则使用全局共享记忆。
            memory_config: 显式的记忆配置对象。当省略时，
                回退到全局配置（通过 get_memory_config() 获取）。
        """
        super().__init__()
        self._agent_name = agent_name
        self._memory_config = memory_config

    @override
    def after_agent(self, state: MemoryMiddlewareState, runtime: Runtime) -> dict | None:
        """在 Agent 执行完成后，将对话入队以触发异步记忆更新。

        该方法在中间件链的 after_agent 阶段被调用，负责：
        1. 检查记忆功能是否启用
        2. 从运行时上下文中提取 thread_id
        3. 过滤消息，仅保留用户输入和最终助手回复
        4. 检测用户是否在纠正或强化之前的信息
        5. 在入队时捕获当前请求上下文中的 user_id
        6. 将过滤后的对话提交到记忆更新队列

        Args:
            state: 当前 Agent 的状态，包含 messages 等字段。
            runtime: LangGraph 运行时上下文，用于获取 thread_id 等元信息。

        Returns:
            始终返回 None，本中间件不修改 Agent 状态。
        """
        # 获取记忆配置：优先使用初始化时传入的配置，否则从全局配置读取
        config = self._memory_config or get_memory_config()
        if not config.enabled:
            return None

        # 优先从 runtime.context 中获取 thread_id，若无则回退到 LangGraph 的 configurable 元数据
        # runtime.context 是 DeerFlow 运行时注入的上下文字典，通常包含 thread_id
        thread_id = runtime.context.get("thread_id") if runtime.context else None
        if thread_id is None:
            # 回退路径：从 LangGraph 的 RunnableConfig 中读取 configurable.thread_id
            # 这种情况发生在 runtime.context 未被正确注入时
            config_data = get_config()
            thread_id = config_data.get("configurable", {}).get("thread_id")
        if not thread_id:
            logger.debug("No thread_id in context, skipping memory update")
            return None

        # 从状态中提取消息列表
        messages = state.get("messages", [])
        if not messages:
            logger.debug("No messages in state, skipping memory update")
            return None

        # 过滤消息：仅保留用户输入和最终助手回复，移除工具调用等中间过程消息
        # 这样做是因为工具调用的细节对记忆提取没有意义，反而会引入噪声
        filtered_messages = filter_messages_for_memory(messages)

        # 只有当存在至少一条用户消息和一条助手回复时，才有入队的意义
        # 这是最低限度的有效对话条件，避免将不完整的对话提交给记忆更新
        user_messages = [m for m in filtered_messages if getattr(m, "type", None) == "human"]
        assistant_messages = [m for m in filtered_messages if getattr(m, "type", None) == "ai"]

        if not user_messages or not assistant_messages:
            return None

        # 检测用户意图：是否在纠正之前的信息（例如"不对，应该是..."）
        # 纠正检测优先级高于强化检测，因为纠正通常包含更重要的记忆更新信号
        correction_detected = detect_correction(filtered_messages)
        # 仅在未检测到纠正时才检测强化（例如"对，就是这样"），
        # 避免同时标记为纠正和强化，两者语义互斥
        reinforcement_detected = not correction_detected and detect_reinforcement(filtered_messages)

        # 【关键】在入队时捕获 user_id，而非在后台 Timer 线程中读取。
        # Python 的 threading.Timer 在新线程上执行回调，而 ContextVar 的值
        # 不会自动传播到新线程的上下文中（与 asyncio 的任务传播机制不同）。
        # 因此必须在当前请求上下文仍存活时显式读取并存储 user_id，
        # 否则后台线程中 get_effective_user_id() 会返回默认值而非真实用户。
        user_id = get_effective_user_id()

        # 将过滤后的对话提交到记忆更新队列
        # 队列内部使用防抖机制（debounce），短时间内的多次入队会合并为一次更新
        queue = get_memory_queue()
        queue.add(
            thread_id=thread_id,
            messages=filtered_messages,
            agent_name=self._agent_name,
            user_id=user_id,
            correction_detected=correction_detected,
            reinforcement_detected=reinforcement_detected,
        )

        return None
