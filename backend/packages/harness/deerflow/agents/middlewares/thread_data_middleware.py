"""
线程数据目录中间件

本模块实现了 ThreadDataMiddleware，负责为每次线程执行创建对应的数据目录结构。
它是中间件链中的第一个中间件，在 Agent 执行之前运行，确保后续中间件和工具
可以访问到正确的线程级别工作空间路径。

目录结构:
    {base_dir}/users/{user_id}/threads/{thread_id}/user-data/workspace   — 工作空间目录
    {base_dir}/users/{user_id}/threads/{thread_id}/user-data/uploads     — 上传文件目录
    {base_dir}/users/{user_id}/threads/{thread_id}/user-data/outputs     — 输出文件目录

初始化策略:
    - lazy_init=True（默认）: 仅计算路径，不立即创建目录，待实际使用时按需创建
    - lazy_init=False: 在 before_agent() 阶段立即创建所有目录

该中间件还会为最后一条 HumanMessage 附加 run_id 和时间戳元数据，
以便后续中间件（如 MemoryMiddleware）追踪消息来源和时间。
"""

import logging
from datetime import UTC, datetime
from typing import NotRequired, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage
from langgraph.config import get_config
from langgraph.runtime import Runtime

from deerflow.agents.thread_state import ThreadDataState
from deerflow.config.paths import Paths, get_paths
from deerflow.runtime.user_context import get_effective_user_id

logger = logging.getLogger(__name__)


class ThreadDataMiddlewareState(AgentState):
    """线程数据中间件状态，与 ThreadState 模式兼容。

    在 Agent 状态中扩展了 thread_data 字段，
    用于存储当前线程的工作空间、上传和输出目录路径。
    该字段为 NotRequired，允许在中间件未执行时不存在。
    """

    thread_data: NotRequired[ThreadDataState | None]


class ThreadDataMiddleware(AgentMiddleware[ThreadDataMiddlewareState]):
    """线程数据目录中间件，为每次线程执行创建对应的数据目录。

    在 before_agent 阶段执行以下操作：
        1. 从运行时上下文或 LangGraph 配置中解析 thread_id
        2. 通过 get_effective_user_id() 解析当前用户 ID，实现用户级路径隔离
        3. 根据 lazy_init 配置，计算路径或立即创建目录
        4. 为最后一条 HumanMessage 附加 run_id 和时间戳元数据

    目录结构:
        - {base_dir}/users/{user_id}/threads/{thread_id}/user-data/workspace
        - {base_dir}/users/{user_id}/threads/{thread_id}/user-data/uploads
        - {base_dir}/users/{user_id}/threads/{thread_id}/user-data/outputs

    生命周期管理:
        - lazy_init=True（默认）: 仅计算路径，目录按需创建
        - lazy_init=False: 在 before_agent() 中立即创建目录
    """

    state_schema = ThreadDataMiddlewareState

    def __init__(self, base_dir: str | None = None, lazy_init: bool = True):
        """初始化线程数据中间件。

        Args:
            base_dir: 线程数据的基础目录。若为 None，则通过 get_paths() 自动解析
                      （默认解析为 backend/.deer-flow/）。若提供显式路径，则使用该路径
                      构造 Paths 实例，适用于自定义存储位置的场景。
            lazy_init: 是否采用延迟初始化策略。
                       True（默认）: 仅计算路径字符串，不实际创建目录。
                           目录将在 Sandbox 或工具首次访问时按需创建，
                           适用于大多数场景，避免不必要的磁盘 I/O。
                       False: 在 before_agent() 阶段立即创建所有目录。
                           适用于需要确保目录在 Agent 执行前就存在的场景，
                           例如某些外部进程需要提前写入目录的情况。
        """
        super().__init__()
        # 根据 base_dir 参数决定使用显式路径还是自动解析的默认路径
        self._paths = Paths(base_dir) if base_dir else get_paths()
        # 保存初始化策略，影响 before_agent 中的目录创建行为
        self._lazy_init = lazy_init

    def _get_thread_paths(self, thread_id: str, user_id: str | None = None) -> dict[str, str]:
        """获取线程数据目录的路径映射（不创建目录）。

        根据 thread_id 和可选的 user_id 计算工作空间、上传和输出目录的物理路径。
        当提供 user_id 时，路径包含用户级隔离前缀
        （如 {base_dir}/users/{user_id}/threads/...），
        实现多用户环境下的数据隔离。

        Args:
            thread_id: 线程 ID，对应一次完整的对话会话。
            user_id: 可选的用户 ID，用于按用户隔离路径。
                     在无认证模式下默认为 "default"。

        Returns:
            包含三个路径的字典:
                - workspace_path: 工作空间目录的物理路径
                - uploads_path: 上传文件目录的物理路径
                - outputs_path: 输出文件目录的物理路径
        """
        return {
            "workspace_path": str(self._paths.sandbox_work_dir(thread_id, user_id=user_id)),
            "uploads_path": str(self._paths.sandbox_uploads_dir(thread_id, user_id=user_id)),
            "outputs_path": str(self._paths.sandbox_outputs_dir(thread_id, user_id=user_id)),
        }

    def _create_thread_directories(self, thread_id: str, user_id: str | None = None) -> dict[str, str]:
        """创建线程数据目录并返回路径映射。

        调用 Paths.ensure_thread_dirs() 确保目录存在（若已存在则为空操作），
        然后返回与 _get_thread_dirs() 相同的路径字典。
        此方法仅在 lazy_init=False 时被调用。

        Args:
            thread_id: 线程 ID，对应一次完整的对话会话。
            user_id: 可选的用户 ID，用于按用户隔离路径。

        Returns:
            包含已创建目录路径的字典，结构与 _get_thread_dirs() 相同。
        """
        # 确保目录存在（内部使用 os.makedirs(exist_ok=True) 实现）
        self._paths.ensure_thread_dirs(thread_id, user_id=user_id)
        # 返回路径映射
        return self._get_thread_paths(thread_id, user_id=user_id)

    @override
    def before_agent(self, state: ThreadDataMiddlewareState, runtime: Runtime) -> dict | None:
        """Agent 执行前的钩子方法，在线程上下文中初始化数据目录。

        执行流程：
            1. 解析 thread_id —— 优先从 runtime.context 获取，
               若不存在则回退到 LangGraph 的 config.configurable.thread_id
            2. 解析 user_id —— 通过 get_effective_user_id() 获取当前有效用户 ID，
               在无认证模式下回退为 "default"
            3. 根据 lazy_init 配置，仅计算路径或立即创建目录
            4. 为最后一条 HumanMessage 附加 run_id 和时间戳元数据

        Args:
            state: 当前 Agent 状态，包含 messages 等字段。
            runtime: LangGraph 运行时实例，提供 context（含 thread_id、run_id 等）。

        Returns:
            更新后的状态字典，包含:
                - thread_data: 线程数据路径映射（workspace_path, uploads_path, outputs_path）
                - messages: 可能被修改过的消息列表（最后一条 HumanMessage 附带了元数据）

        Raises:
            ValueError: 当 thread_id 在 runtime.context 和 config.configurable 中均不存在时抛出。
        """
        # 从运行时上下文中获取 thread_id（由 Gateway 或客户端传入）
        context = runtime.context or {}
        thread_id = context.get("thread_id")

        # 若运行时上下文中没有 thread_id，则回退到 LangGraph 配置的 configurable 字段
        # 这兼容了通过 LangGraph Studio 或直接 API 调用启动的场景
        if thread_id is None:
            config = get_config()
            thread_id = config.get("configurable", {}).get("thread_id")

        # thread_id 是必须的，无法在缺少的情况下继续执行
        if thread_id is None:
            raise ValueError("Thread ID is required in runtime context or config.configurable")

        # 解析当前有效用户 ID，实现用户级路径隔离
        # 在无认证模式下，get_effective_user_id() 返回 "default"
        user_id = get_effective_user_id()

        if self._lazy_init:
            # 延迟初始化：仅计算路径字符串，不实际创建目录
            # 目录将在 Sandbox 或工具首次访问时按需创建
            paths = self._get_thread_paths(thread_id, user_id=user_id)
        else:
            # 立即初始化：在 Agent 执行前创建所有必要的目录
            paths = self._create_thread_directories(thread_id, user_id=user_id)
            logger.debug("Created thread data directories for thread %s", thread_id)

        # --- 为最后一条 HumanMessage 附加元数据 ---
        # 获取当前消息列表的副本，避免修改原始状态
        messages = list(state.get("messages", []))
        last_message = messages[-1] if messages else None

        if last_message and isinstance(last_message, HumanMessage):
            # 重新构造 HumanMessage，在 additional_kwargs 中追加 run_id 和 timestamp
            # - run_id: 标识当前运行，用于关联消息与特定的执行实例
            # - timestamp: 记录消息被中间件处理的时间（UTC），用于时间线追踪
            # 保留原始 content、id、name 等字段，仅在 additional_kwargs 中追加新字段
            messages[-1] = HumanMessage(
                content=last_message.content,
                id=last_message.id,
                # 若原始消息没有 name，则使用默认值 "user-input"
                name=last_message.name or "user-input",
                # 合并原有的 additional_kwargs，并追加 run_id 和 timestamp
                additional_kwargs={
                    **last_message.additional_kwargs,
                    "run_id": runtime.context.get("run_id"),
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )

        # 返回更新后的状态：thread_data 包含路径映射，messages 包含可能修改过的消息列表
        return {
            "thread_data": {
                **paths,
            },
            "messages": messages,
        }
