"""带防抖机制的记忆更新队列。

本模块实现记忆更新的异步处理:
- 收集对话上下文并批量处理
- 可配置的防抖期，避免频繁更新
- 线程安全的队列操作
- 支持立即处理和强制刷新

工作流程:
1. 对话结束时将上下文添加到队列
2. 重置防抖计时器
3. 防抖期过后批量处理所有待处理更新
4. 使用 MemoryUpdater 调用 LLM 更新记忆
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from deerflow.config.memory_config import get_memory_config

logger = logging.getLogger(__name__)


@dataclass
class ConversationContext:
    """待处理用于记忆更新的对话上下文。
    
    封装单次对话的完整信息，包括消息内容、时间戳和检测到的信号。
    
    Attributes:
        thread_id: 线程 ID，用于标识对话所属的会话
        messages: 对话消息列表，包含用户和助手的交互
        timestamp: 上下文创建时间(UTC)，用于追踪和排序
        agent_name: 代理名称，如果提供则存储到该代理的独立记忆中；None 表示全局记忆
        user_id: 用户 ID，在入队时捕获，确保跨线程边界传递(ContextVar 不会传播到原始线程)
        correction_detected: 是否检测到明确的纠正信号(如"不对"、"你理解错了")
        reinforcement_detected: 是否检测到正面强化信号(如"完全正确"、"正是我想要的")
        
    Note:
        - correction_detected 和 reinforcement_detected 用于提高事实置信度
        - user_id 必须显式存储，因为 threading.Timer 不继承 ContextVar
    """

    thread_id: str
    messages: list[Any]
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    agent_name: str | None = None
    user_id: str | None = None
    correction_detected: bool = False
    reinforcement_detected: bool = False


class MemoryUpdateQueue:
    """带防抖机制的记忆更新队列。
    
    该队列收集对话上下文并在可配置的防抖期后处理它们。
    在防抖窗口内接收的多个对话会被批量处理。
    
    核心特性:
    - 防抖机制：避免频繁调用 LLM，节省成本和减少延迟
    - 批量处理：同一目标的多次对话合并处理
    - 线程安全：使用锁保护所有队列操作
    - 灵活调度：支持延迟处理和立即处理两种模式
    
    工作流程:
    1. add/add_nowait 将对话添加到队列
    2. _enqueue_locked 合并同一目标的上下文
    3. _reset_timer/_schedule_timer 设置防抖计时器
    4. 防抖期过后调用 _process_queue 批量处理
    5. MemoryUpdater 调用 LLM 更新记忆并持久化
    
    """

    def __init__(self):
        """初始化记忆更新队列。
        
        创建空的队列数据结构、线程锁和计时器引用。
        
        Attributes:
            _queue: 待处理的对话上下文列表
            _lock: 保护队列操作的线程锁
            _timer: 防抖计时器，到期后触发批量处理
            _processing: 标记是否正在处理队列，防止并发处理
        """
        self._queue: list[ConversationContext] = []
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._processing = False

    @staticmethod
    def _queue_key(
        thread_id: str,
        user_id: str | None,
        agent_name: str | None,
    ) -> tuple[str, str | None, str | None]:
        """返回记忆更新目标的防抖身份标识。
        
        基于三元组 (thread_id, user_id, agent_name) 生成唯一键，
        用于识别和合并同一目标的多次对话更新。
        
        Args:
            thread_id: 线程 ID
            user_id: 用户 ID(可为 None，表示全局用户)
            agent_name: 代理名称(可为 None，表示全局代理)
            
        Returns:
            三元组键，用于队列去重和合并
            
        Note:
            - 相同键的对话会被合并，只保留最新的上下文
            - 这是防抖机制的核心，避免对同一目标重复更新
        """
        return (thread_id, user_id, agent_name)

    def add(
        self,
        thread_id: str,
        messages: list[Any],
        agent_name: str | None = None,
        user_id: str | None = None,
        correction_detected: bool = False,
        reinforcement_detected: bool = False,
    ) -> None:
        """将对话添加到更新队列(带防抖延迟)。
        
        此方法将对话上下文加入队列，并重置防抖计时器。
        如果在防抖期内再次调用，会合并同一目标的上下文并重新计时。
        
        Args:
            thread_id: 线程 ID，标识对话所属的会话
            messages: 对话消息列表，包含用户和助手的交互
            agent_name: 如果提供，记忆存储到该代理的独立记忆中；None 表示全局记忆
            user_id: 入队时捕获的用户 ID，存储在 ConversationContext 中以便跨线程边界传递
                   (ContextVar 不会传播到原始线程，所以需要显式存储)
            correction_detected: 最近的对话轮次是否包含明确的纠正信号
            reinforcement_detected: 最近的对话轮次是否包含正面强化信号
            
        Note:
            - 如果记忆功能未启用，直接返回不执行任何操作
            - 使用线程锁保证队列操作的原子性
            - 同一目标的多次调用会合并，只保留最新的上下文
            - 防抖期由配置中的 debounce_seconds 决定
        """
        config = get_memory_config()
        if not config.enabled:
            return

        with self._lock:
            self._enqueue_locked(
                thread_id=thread_id,
                messages=messages,
                agent_name=agent_name,
                user_id=user_id,
                correction_detected=correction_detected,
                reinforcement_detected=reinforcement_detected,
            )
            self._reset_timer()

        logger.info("Memory update queued for thread %s, queue size: %d", thread_id, len(self._queue))

    def add_nowait(
        self,
        thread_id: str,
        messages: list[Any],
        agent_name: str | None = None,
        user_id: str | None = None,
        correction_detected: bool = False,
        reinforcement_detected: bool = False,
    ) -> None:
        """将对话添加到队列并立即开始后台处理。
        
        与 add() 不同，此方法不等待防抖期，而是立即调度处理。
        适用于摘要钩子等需要立即保存记忆的场景。
        
        Args:
            thread_id: 线程 ID，标识对话所属的会话
            messages: 对话消息列表
            agent_name: 代理名称(可选)
            user_id: 用户 ID(可选)
            correction_detected: 是否检测到纠正信号
            reinforcement_detected: 是否检测到正面强化信号
            
        Note:
            - 使用 delay=0 调度计时器，立即触发 _process_queue
            - 仍在后台线程中执行，不会阻塞调用者
            - 适合需要立即持久化的关键对话
        """
        config = get_memory_config()
        if not config.enabled:
            return

        with self._lock:
            self._enqueue_locked(
                thread_id=thread_id,
                messages=messages,
                agent_name=agent_name,
                user_id=user_id,
                correction_detected=correction_detected,
                reinforcement_detected=reinforcement_detected,
            )
            self._schedule_timer(0)

        logger.info("Memory update queued for immediate processing on thread %s, queue size: %d", thread_id, len(self._queue))

    def _enqueue_locked(
        self,
        *,
        thread_id: str,
        messages: list[Any],
        agent_name: str | None,
        user_id: str | None,
        correction_detected: bool,
        reinforcement_detected: bool,
    ) -> None:
        """在锁保护下将对话上下文加入队列(内部方法)。
        
        此方法实现防抖合并逻辑：如果队列中已存在相同目标的上下文，
        则移除旧上下文并添加新的，同时合并纠正/强化信号。
        
        Args:
            thread_id: 线程 ID
            messages: 对话消息列表
            agent_name: 代理名称
            user_id: 用户 ID
            correction_detected: 是否检测到纠正信号
            reinforcement_detected: 是否检测到强化信号
            
        Note:
            - 必须在持有 _lock 的情况下调用
            - 使用 OR 逻辑合并信号：任一对话检测到信号即保留
            - 新上下文替换旧上下文，但保留检测到的信号
        """
        # 生成队列键用于去重
        queue_key = self._queue_key(thread_id, user_id, agent_name)
        # 查找是否已存在相同目标的上下文
        existing_context = next(
            (context for context in self._queue if self._queue_key(context.thread_id, context.user_id, context.agent_name) == queue_key),
            None,
        )
        # 合并纠正和强化信号(OR 逻辑)
        merged_correction_detected = correction_detected or (existing_context.correction_detected if existing_context is not None else False)
        merged_reinforcement_detected = reinforcement_detected or (existing_context.reinforcement_detected if existing_context is not None else False)
        # 创建新的上下文对象
        context = ConversationContext(
            thread_id=thread_id,
            messages=messages,
            agent_name=agent_name,
            user_id=user_id,
            correction_detected=merged_correction_detected,
            reinforcement_detected=merged_reinforcement_detected,
        )

        # 移除旧的相同目标上下文，添加新的
        self._queue = [context for context in self._queue if self._queue_key(context.thread_id, context.user_id, context.agent_name) != queue_key]
        self._queue.append(context)

    def _reset_timer(self) -> None:
        """重置防抖计时器。
        
        取消现有计时器并设置新的计时器，延迟时间由配置决定。
        每次调用 add() 时都会重置计时器，实现防抖效果。
        
        Note:
            - 从配置中读取 debounce_seconds
            - 如果频繁调用 add()，处理会一直延迟直到停止调用
        """
        config = get_memory_config()
        self._schedule_timer(config.debounce_seconds)

        logger.debug("Memory update timer set for %ss", config.debounce_seconds)

    def _schedule_timer(self, delay_seconds: float) -> None:
        """在指定延迟后调度队列处理。
        
        创建一个新的 threading.Timer，到期后调用 _process_queue。
        如果已有计时器，先取消它再创建新的。
        
        Args:
            delay_seconds: 延迟秒数，0 表示立即处理
            
        Note:
            - Timer 设置为 daemon 线程，进程退出时自动终止
            - delay=0 用于立即处理(add_nowait 和 flush_nowait)
            - 每次调用都会取消旧计时器，确保只有一个待处理的计时器
        """
        # 取消现有计时器(如果有)
        if self._timer is not None:
            self._timer.cancel()

        self._timer = threading.Timer(
            delay_seconds,
            self._process_queue,
        )
        self._timer.daemon = True
        self._timer.start()

    def _process_queue(self) -> None:
        """处理所有队列中的对话上下文。
        
        此方法在后台线程中执行，批量处理所有待处理的记忆更新。
        对每个上下文调用 MemoryUpdater 进行 LLM 分析和记忆更新。
        
        工作流程:
        1. 检查是否正在处理，如果是则重新调度(保留立即刷新语义)
        2. 复制当前队列并清空，释放锁
        3. 创建 MemoryUpdater 实例
        4. 遍历所有上下文，逐个更新记忆
        5. 在多个更新之间添加小延迟避免速率限制
        6. 最终将 _processing 标记设为 False
        
        Note:
            - 在循环内导入 MemoryUpdater 避免循环依赖
            - 使用深拷贝的队列副本，处理期间新添加的项不会被处理
            - 单个更新失败不影响其他更新
            - 多上下文处理时，每个更新间有 0.5s 延迟
        """
        # 在此处导入以避免循环依赖
        from deerflow.agents.memory.updater import MemoryUpdater

        with self._lock:
            if self._processing:
                # 即使有其他工作线程活动，也保留立即刷新语义
                self._schedule_timer(0)
                return

            if not self._queue:
                return

            self._processing = True
            contexts_to_process = self._queue.copy()
            self._queue.clear()
            self._timer = None

        logger.info("Processing %d queued memory updates", len(contexts_to_process))

        try:
            updater = MemoryUpdater()

            for context in contexts_to_process:
                try:
                    logger.info("Updating memory for thread %s", context.thread_id)
                    success = updater.update_memory(
                        messages=context.messages,
                        thread_id=context.thread_id,
                        agent_name=context.agent_name,
                        correction_detected=context.correction_detected,
                        reinforcement_detected=context.reinforcement_detected,
                        user_id=context.user_id,
                    )
                    if success:
                        logger.info("Memory updated successfully for thread %s", context.thread_id)
                    else:
                        logger.warning("Memory update skipped/failed for thread %s", context.thread_id)
                except Exception as e:
                    logger.error("Error updating memory for thread %s: %s", context.thread_id, e)

                # 在更新之间添加小延迟以避免速率限制
                if len(contexts_to_process) > 1:
                    time.sleep(0.5)

        finally:
            with self._lock:
                self._processing = False

    def flush(self) -> None:
        """强制立即处理队列。
        
        取消待处理的计时器并同步执行 _process_queue。
        这会阻塞直到所有队列项处理完成。
        
        Use Cases:
            - 测试：确保所有记忆更新已完成
            - 优雅关闭：在进程退出前处理所有待处理项
            - 调试：立即看到记忆更新结果
            
        Note:
            - 此方法是同步的，会阻塞调用者
            - 如果队列很大，可能需要较长时间
            - 适合在测试或关闭流程中使用
        """
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None

        self._process_queue()

    def flush_nowait(self) -> None:
        """在后台线程中立即开始队列处理。
        
        与 flush() 不同，此方法不阻塞调用者，而是立即调度后台处理。
        适用于需要快速返回但希望尽快处理记忆的场景。
        
        Note:
            - 使用 daemon 线程：如果进程在 _process_queue 完成前退出，
              队列中的消息可能会丢失。对于尽力而为的记忆更新来说这是可接受的。
            - 不会阻塞调用者，适合在请求处理流程中使用
            - 比 flush() 更适合生产环境
        """
        with self._lock:
            # Daemon 线程：如果进程在 _process_queue 完成前退出，
            # 队列中的消息可能会丢失。对于尽力而为的记忆更新来说这是可接受的。
            self._schedule_timer(0)

    def clear(self) -> None:
        """清空队列而不处理。
        
        丢弃所有待处理的记忆更新，重置队列状态。
        主要用于测试和调试场景。
        
        Note:
            - 不会触发任何记忆更新
            - 取消待处理的计时器
            - 重置 _processing 标志
            - 适合在测试之间清理状态
        """
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            self._queue.clear()
            self._processing = False

    @property
    def pending_count(self) -> int:
        """获取待处理更新的数量。
        
        Returns:
            队列中待处理的对话上下文数量
            
        Note:
            - 线程安全，使用锁保护
            - 反映当前时刻的队列状态
        """
        with self._lock:
            return len(self._queue)

    @property
    def is_processing(self) -> bool:
        """检查队列是否正在被处理。
        
        Returns:
            如果 _process_queue 正在执行则返回 True
            
        Note:
            - 线程安全，使用锁保护
            - 可用于监控和调试
            - 处理期间为 True，处理完成后为 False
        """
        with self._lock:
            return self._processing


# 全局单例实例
_memory_queue: MemoryUpdateQueue | None = None
_queue_lock = threading.Lock()


def get_memory_queue() -> MemoryUpdateQueue:
    """获取全局记忆更新队列单例。
    
    使用线程安全的双重检查锁定模式创建和返回单例队列实例。
    
    Returns:
        记忆更新队列实例
        
    Note:
        - 线程安全，使用 _queue_lock 保护
        - 整个应用生命周期内只有一个队列实例
        - 所有模块都应该通过此函数获取队列，而不是直接创建
        
    Example:
        >>> queue = get_memory_queue()
        >>> queue.add(thread_id="t1", messages=[...])
    """
    global _memory_queue
    with _queue_lock:
        if _memory_queue is None:
            _memory_queue = MemoryUpdateQueue()
        return _memory_queue


def reset_memory_queue() -> None:
    """重置全局记忆队列。
    
    清空并销毁当前队列实例，下次调用 get_memory_queue() 时会创建新实例。
    主要用于测试场景，确保测试之间的状态隔离。
    
    Warning:
        - 仅用于测试！生产环境中不应调用此函数
        - 会丢失所有待处理的记忆更新
        - 可能导致正在进行的更新失败
        
    Use Cases:
        - 单元测试：在每个测试前重置队列状态
        - 集成测试：确保测试之间没有状态污染
    """
    global _memory_queue
    with _queue_lock:
        if _memory_queue is not None:
            _memory_queue.clear()
        _memory_queue = None
