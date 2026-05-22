"""扩展 TodoListMiddleware 的中间件，增加了上下文丢失检测与过早退出防护机制。

上下文丢失检测（Context-Loss Detection）:
    当消息历史被截断时（例如被 SummarizationMiddleware 摘要化后），
    原始的 `write_todos` 工具调用及其对应的 ToolMessage 可能会滚出当前活跃的上下文窗口。
    这会导致模型"忘记"自己曾经创建过待办事项列表，从而无法继续追踪进度。
    本中间件在 `before_model` / `abefore_model` 中检测这一情况，
    并注入一条提醒消息，使模型仍然知晓尚未完成的待办列表。

过早退出防护（Premature-Exit Prevention）:
    当模型产出了最终响应（不含工具调用）但待办事项尚未全部完成时，
    本中间件会阻止代理退出循环。具体做法是：为下一次模型请求排队一条提醒消息，
    并跳转回模型节点（jump_to="model"），强制代理继续处理待办列表。
    完成提醒通过 ``wrap_model_call`` 注入到请求中，而不是作为普通用户可见消息
    持久化到图状态中，从而避免将控制指令泄露到用户界面或保存的对话记录中。
"""

from __future__ import annotations

import threading
from collections.abc import Awaitable, Callable
from typing import Any, override

from langchain.agents.middleware import TodoListMiddleware
from langchain.agents.middleware.todo import PlanningState, Todo
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse, hook_config
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime


def _todos_in_messages(messages: list[Any]) -> bool:
    """检查消息列表中是否存在包含 write_todos 工具调用的 AIMessage。

    遍历所有消息，如果发现任何 AIMessage 携带名为 "write_todos" 的工具调用，
    则说明 write_todos 的调用记录仍在当前上下文窗口中，模型能够感知到待办列表。

    Args:
        messages: 消息列表，通常来自 PlanningState 中的 "messages" 字段。

    Returns:
        如果存在包含 write_todos 工具调用的 AIMessage 则返回 True，否则返回 False。
    """
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.get("name") == "write_todos":
                    return True
    return False


def _reminder_in_messages(messages: list[Any]) -> bool:
    """检查消息列表中是否已存在 todo_reminder 类型的 HumanMessage。

    用于防止重复注入上下文丢失提醒。如果之前的提醒尚未被截断出上下文窗口，
    则无需再次注入。

    Args:
        messages: 消息列表。

    Returns:
        如果已存在 name 为 "todo_reminder" 的 HumanMessage 则返回 True。
    """
    for msg in messages:
        if isinstance(msg, HumanMessage) and getattr(msg, "name", None) == "todo_reminder":
            return True
    return False


def _completion_reminder_count(messages: list[Any]) -> int:
    """统计消息列表中 todo_completion_reminder 类型 HumanMessage 的数量。

    用于在 after_model 中判断是否已经达到完成提醒的上限次数。

    Args:
        messages: 消息列表。

    Returns:
        name 为 "todo_completion_reminder" 的 HumanMessage 数量。
    """
    return sum(1 for msg in messages if isinstance(msg, HumanMessage) and getattr(msg, "name", None) == "todo_completion_reminder")


def _format_todos(todos: list[Todo]) -> str:
    """将待办事项列表格式化为人类可读的字符串。

    每个待办项格式为 "- [状态] 内容"，例如 "- [pending] 实现用户登录功能"。

    Args:
        todos: 待办事项列表，每项包含 "status" 和 "content" 字段。

    Returns:
        格式化后的多行字符串。
    """
    lines: list[str] = []
    for todo in todos:
        status = todo.get("status", "pending")
        content = todo.get("content", "")
        lines.append(f"- [{status}] {content}")
    return "\n".join(lines)


def _format_completion_reminder(todos: list[Todo]) -> str:
    """为未完成的待办事项生成完成提醒的格式化文本。

    仅筛选 status 不为 "completed" 的待办项，生成系统提醒消息，
    要求模型继续处理这些未完成的任务。

    Args:
        todos: 待办事项列表。

    Returns:
        包裹在 <system_reminder> 标签中的提醒文本。
    """
    # 筛选出所有未完成的待办项
    incomplete = [t for t in todos if t.get("status") != "completed"]
    # 格式化未完成待办项为多行文本
    incomplete_text = "\n".join(f"- [{t.get('status', 'pending')}] {t.get('content', '')}" for t in incomplete)
    return (
        "<system_reminder>\n"
        "You have incomplete todo items that must be finished before giving your final response:\n\n"
        f"{incomplete_text}\n\n"
        "Please continue working on these tasks. Call `write_todos` to mark items as completed "
        "as you finish them, and only respond when all items are done.\n"
        "</system_reminder>"
    )


# 工具调用相关的结束原因集合，用于判断模型是否意图发起工具调用
_TOOL_CALL_FINISH_REASONS = {"tool_calls", "function_call"}


def _has_tool_call_intent_or_error(message: AIMessage) -> bool:
    """判断 AIMessage 是否包含工具调用意图或工具调用解析错误。

    完成提醒只在模型产出了干净的最终回答（无工具调用意图）时才触发。
    如果模型仍有工具调用意图或存在工具解析错误，应该让工具路径来处理，
    而不是用待办提醒去遮蔽这些信号。

    该辅助函数将所有工具调用意图/错误的检测逻辑集中在一处，
    避免在调用点分散检查不同字段。由于 LangChain 不同版本和集成提供商
    在工具调用信息的存放位置上存在差异，此函数覆盖了所有可能的字段：

    1. message.tool_calls —— 结构化的工具调用列表
    2. message.invalid_tool_calls —— 解析失败的工具调用
    3. message.additional_kwargs["tool_calls"] / ["function_call"] ——
       部分集成在结构化字段为空时仍保留原始/遗留的工具调用意图
    4. message.response_metadata["finish_reason"] ——
       某些提供商通过结束原因标识工具调用意图

    如果修改此函数，需同步更新测试
    `TestToolCallIntentOrError.test_langchain_ai_message_tool_fields_are_explicitly_handled`；
    LangChain 升级后若该测试失败，应审查此函数以确保新的工具调用/错误字段
    不会被静默当作干净的最终回答处理。

    Args:
        message: 待检查的 AIMessage。

    Returns:
        如果消息包含工具调用意图或错误则返回 True，否则返回 False。
    """
    # 检查结构化的工具调用
    if message.tool_calls:
        return True

    # 检查无效的工具调用（解析失败）
    if getattr(message, "invalid_tool_calls", None):
        return True

    # 向后/提供商兼容性：某些集成在结构化 tool_calls 为空时，
    # 仍会在 additional_kwargs 中保留原始或遗留的工具调用意图。
    additional_kwargs = getattr(message, "additional_kwargs", {}) or {}
    if additional_kwargs.get("tool_calls") or additional_kwargs.get("function_call"):
        return True

    # 检查响应元数据中的结束原因是否为工具调用
    response_metadata = getattr(message, "response_metadata", {}) or {}
    return response_metadata.get("finish_reason") in _TOOL_CALL_FINISH_REASONS


class TodoMiddleware(TodoListMiddleware):
    """扩展 TodoListMiddleware，增加 write_todos 上下文丢失检测与过早退出防护。

    上下文丢失检测：
        当原始的 write_todos 工具调用已从消息历史中被截断（例如经过摘要化后），
        模型会丢失对当前待办列表的感知。本中间件在 before_model / abefore_model
        中检测这一间隙，并注入提醒消息使模型能够继续追踪进度。

    过早退出防护：
        当模型产出不含工具调用的最终响应但待办事项尚未全部完成时，
        本中间件在 after_model 中拦截该响应，排队完成提醒并跳转回模型节点，
        强制代理继续工作，直到所有待办项完成或达到提醒次数上限。
    """

    @override
    def before_model(
        self,
        state: PlanningState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """在模型调用前检测上下文丢失并注入待办列表提醒。

        检测逻辑：
        1. 如果状态中没有待办事项，无需处理。
        2. 如果消息历史中仍存在 write_todos 工具调用记录，说明模型能感知待办列表，无需提醒。
        3. 如果已经注入过提醒且该提醒仍在上下文中，无需重复注入。
        4. 否则，说明待办列表存在于状态中但原始调用记录已丢失，需要注入提醒。

        Args:
            state: 当前规划状态，包含 "todos" 和 "messages"。
            runtime: LangGraph 运行时。

        Returns:
            包含提醒消息的字典 {"messages": [reminder]}，或 None（无需提醒时）。
        """
        todos: list[Todo] = state.get("todos") or []  # type: ignore[assignment]
        if not todos:
            # 没有待办事项，无需任何处理
            return None

        messages = state.get("messages") or []
        if _todos_in_messages(messages):
            # write_todos 调用记录仍在上下文窗口中 —— 模型能感知到待办列表，无需操作
            return None

        if _reminder_in_messages(messages):
            # 之前注入的提醒尚未被截断，无需重复注入
            return None

        # 待办列表存在于状态中，但原始的 write_todos 调用记录已从上下文中丢失。
        # 注入一条 HumanMessage 提醒，让模型保持对待办列表的感知。
        formatted = _format_todos(todos)
        reminder = HumanMessage(
            name="todo_reminder",
            additional_kwargs={"hide_from_ui": True},  # 对用户界面隐藏此提醒
            content=(
                "<system_reminder>\n"
                "Your todo list from earlier is no longer visible in the current context window, "
                "but it is still active. Here is the current state:\n\n"
                f"{formatted}\n\n"
                "Continue tracking and updating this todo list as you work. "
                "Call `write_todos` whenever the status of any item changes.\n"
                "</system_reminder>"
            ),
        )
        return {"messages": [reminder]}

    @override
    async def abefore_model(
        self,
        state: PlanningState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """before_model 的异步版本，逻辑完全一致。"""
        return self.before_model(state, runtime)

    # 完成提醒的最大次数上限。
    # 达到此上限后允许代理退出，防止代理无法继续推进时陷入无限循环。
    _MAX_COMPLETION_REMINDERS = 2
    # 长生命周期中间件实例中，每个运行（run）的提醒账本键数量硬上限。
    # 防止在极长时间运行的中间件实例中累积过多已完成的运行记录导致内存泄漏。
    _MAX_COMPLETION_REMINDER_KEYS = 4096

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """初始化中间件实例。

        创建线程安全的内部数据结构来管理完成提醒的排队和计数：

        - _pending_completion_reminders: 排队等待注入的完成提醒文本，
          键为 (thread_id, run_id) 元组，值为提醒文本列表。
        - _completion_reminder_counts: 每个运行已触发的完成提醒次数，
          用于判断是否达到上限。
        - _completion_reminder_touch_order: 每个键的最后访问顺序号，
          用于 LRU 淘汰策略。
        - _completion_reminder_next_order: 全局递增的访问顺序计数器。
        """
        super().__init__(*args, **kwargs)
        self._lock = threading.Lock()
        self._pending_completion_reminders: dict[tuple[str, str], list[str]] = {}
        self._completion_reminder_counts: dict[tuple[str, str], int] = {}
        self._completion_reminder_touch_order: dict[tuple[str, str], int] = {}
        self._completion_reminder_next_order = 0

    @staticmethod
    def _get_thread_id(runtime: Runtime) -> str:
        """从运行时上下文中提取线程 ID。

        Args:
            runtime: LangGraph 运行时，其 context 属性包含线程信息。

        Returns:
            线程 ID 字符串，无法获取时返回 "default"。
        """
        context = getattr(runtime, "context", None)
        thread_id = context.get("thread_id") if context else None
        return str(thread_id) if thread_id else "default"

    @staticmethod
    def _get_run_id(runtime: Runtime) -> str:
        """从运行时上下文中提取运行 ID。

        Args:
            runtime: LangGraph 运行时，其 context 属性包含运行信息。

        Returns:
            运行 ID 字符串，无法获取时返回 "default"。
        """
        context = getattr(runtime, "context", None)
        run_id = context.get("run_id") if context else None
        return str(run_id) if run_id else "default"

    def _pending_key(self, runtime: Runtime) -> tuple[str, str]:
        """生成用于标识当前线程-运行组合的字典键。

        返回 (thread_id, run_id) 元组，确保不同线程和不同运行之间的
        提醒状态互不干扰。

        Args:
            runtime: LangGraph 运行时。

        Returns:
            (thread_id, run_id) 元组。
        """
        return self._get_thread_id(runtime), self._get_run_id(runtime)

    def _touch_completion_reminder_key_locked(self, key: tuple[str, str]) -> None:
        """更新指定键的访问顺序号（必须在持有锁的情况下调用）。

        用于 LRU 淘汰策略，每次访问键时递增全局计数器并记录该键的顺序号。

        Args:
            key: (thread_id, run_id) 元组。
        """
        self._completion_reminder_next_order += 1
        self._completion_reminder_touch_order[key] = self._completion_reminder_next_order

    def _completion_reminder_keys_locked(self) -> set[tuple[str, str]]:
        """收集所有提醒状态字典中存在的键集合（必须在持有锁的情况下调用）。

        合并三个内部字典的键：排队提醒、提醒计数、访问顺序，
        返回它们的并集。

        Returns:
            所有可能存在的键的集合。
        """
        keys = set(self._pending_completion_reminders)
        keys.update(self._completion_reminder_counts)
        keys.update(self._completion_reminder_touch_order)
        return keys

    def _drop_completion_reminder_key_locked(self, key: tuple[str, str]) -> None:
        """从所有提醒状态字典中移除指定键（必须在持有锁的情况下调用）。

        用于清理已完成或过期的运行记录，释放内存。

        Args:
            key: 要移除的 (thread_id, run_id) 元组。
        """
        self._pending_completion_reminders.pop(key, None)
        self._completion_reminder_counts.pop(key, None)
        self._completion_reminder_touch_order.pop(key, None)

    def _prune_completion_reminder_state_locked(self, protected_key: tuple[str, str]) -> None:
        """当键数量超过上限时，按 LRU 策略淘汰最久未访问的键（必须在持有锁的情况下调用）。

        protected_key 参数指定的键不会被淘汰，确保当前活跃的运行记录不会被意外清理。

        Args:
            protected_key: 受保护的键，通常是当前正在处理的运行。
        """
        keys = self._completion_reminder_keys_locked()
        overflow = len(keys) - self._MAX_COMPLETION_REMINDER_KEYS
        if overflow <= 0:
            # 未超出上限，无需淘汰
            return

        # 排除受保护的键，按访问顺序升序排列（最旧的在前）
        candidates = [key for key in keys if key != protected_key]
        candidates.sort(key=lambda key: self._completion_reminder_touch_order.get(key, 0))
        # 淘汰最旧的 overflow 个键
        for key in candidates[:overflow]:
            self._drop_completion_reminder_key_locked(key)

    def _queue_completion_reminder(self, runtime: Runtime, reminder: str) -> None:
        """将完成提醒文本排队等待下一次模型调用时注入。

        在 after_model 中检测到模型试图过早退出时调用此方法，
        将提醒文本追加到当前运行的排队列表中，并递增该运行的提醒计数。
        同时更新访问顺序并在键数量超限时执行淘汰。

        Args:
            runtime: LangGraph 运行时，用于确定当前线程和运行 ID。
            reminder: 格式化后的提醒文本。
        """
        key = self._pending_key(runtime)
        with self._lock:
            self._pending_completion_reminders.setdefault(key, []).append(reminder)
            self._completion_reminder_counts[key] = self._completion_reminder_counts.get(key, 0) + 1
            self._touch_completion_reminder_key_locked(key)
            self._prune_completion_reminder_state_locked(protected_key=key)

    def _completion_reminder_count_for_runtime(self, runtime: Runtime) -> int:
        """获取当前运行已触发的完成提醒次数。

        用于在 after_model 中判断是否已达到 _MAX_COMPLETION_REMINDERS 上限。

        Args:
            runtime: LangGraph 运行时。

        Returns:
            当前运行已触发的完成提醒次数。
        """
        key = self._pending_key(runtime)
        with self._lock:
            return self._completion_reminder_counts.get(key, 0)

    def _drain_completion_reminders(self, runtime: Runtime) -> list[str]:
        """排空当前运行的所有排队提醒，返回提醒文本列表。

        在 wrap_model_call 的 _augment_request 中调用，将排队的提醒
        从内部存储中取出并注入到模型请求中。取出的提醒会从排队列表中移除，
        但计数器不会被重置（用于持续跟踪已发送的总提醒次数）。

        Args:
            runtime: LangGraph 运行时。

        Returns:
            排队的提醒文本列表，可能为空。
        """
        key = self._pending_key(runtime)
        with self._lock:
            reminders = self._pending_completion_reminders.pop(key, [])
            if reminders or key in self._completion_reminder_counts:
                # 更新访问顺序，保持该键不被 LRU 淘汰
                self._touch_completion_reminder_key_locked(key)
            return reminders

    def _clear_other_run_completion_reminders(self, runtime: Runtime) -> None:
        """清除同一线程下其他运行的完成提醒状态。

        在 before_agent 中调用。当新运行开始时，清理同一线程中
        之前运行遗留的提醒状态，避免不同运行之间的状态干扰。

        Args:
            runtime: LangGraph 运行时。
        """
        thread_id, current_run_id = self._pending_key(runtime)
        with self._lock:
            for key in self._completion_reminder_keys_locked():
                # 仅清除同一线程下不同运行 ID 的记录
                if key[0] == thread_id and key[1] != current_run_id:
                    self._drop_completion_reminder_key_locked(key)

    def _clear_current_run_completion_reminders(self, runtime: Runtime) -> None:
        """清除当前运行的完成提醒状态。

        在 after_agent 中调用。当代理运行结束时，清理当前运行的
        所有提醒状态，释放内存。

        Args:
            runtime: LangGraph 运行时。
        """
        key = self._pending_key(runtime)
        with self._lock:
            self._drop_completion_reminder_key_locked(key)

    @override
    def before_agent(self, state: PlanningState, runtime: Runtime) -> dict[str, Any] | None:
        """代理开始前，清除同线程其他运行的提醒状态。

        确保新运行不会继承之前运行遗留的排队提醒或计数器。
        """
        self._clear_other_run_completion_reminders(runtime)
        return None

    @override
    async def abefore_agent(self, state: PlanningState, runtime: Runtime) -> dict[str, Any] | None:
        """before_agent 的异步版本。"""
        self._clear_other_run_completion_reminders(runtime)
        return None

    @hook_config(can_jump_to=["model"])
    @override
    def after_model(
        self,
        state: PlanningState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """在模型产出响应后检测是否需要阻止过早退出。

        核心逻辑流程：
        1. 保留基类逻辑（检测并行的 write_todos 调用）。
        2. 仅在模型试图干净退出（无工具调用意图或错误）时介入。
           如果模型仍有工具调用意图，应走工具处理路径，而非用提醒遮蔽。
        3. 如果所有待办项已完成或没有待办项，允许退出。
        4. 如果已完成提醒次数已达上限（_MAX_COMPLETION_REMINDERS），允许退出，
           防止无限循环。
        5. 否则，排队一条完成提醒并跳转回模型节点，强制代理继续工作。

        注意：完成提醒不能作为普通 HumanMessage 持久化到图状态中，
        否则会泄露到用户可见的消息流和保存的对话记录中。
        因此通过 wrap_model_call 在请求层面注入。

        Args:
            state: 当前规划状态。
            runtime: LangGraph 运行时。

        Returns:
            {"jump_to": "model"} 表示跳转回模型节点，或 None 表示允许继续。
        """
        # 步骤 1：保留基类逻辑（并行 write_todos 检测）
        base_result = super().after_model(state, runtime)
        if base_result is not None:
            return base_result

        # 步骤 2：仅在代理试图干净退出时介入。
        # 工具调用意图或工具解析错误应交由工具路径处理，不应被待办提醒遮蔽。
        messages = state.get("messages") or []
        last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        if not last_ai or _has_tool_call_intent_or_error(last_ai):
            return None

        # 步骤 3：所有待办项已完成或没有待办项时允许退出
        todos: list[Todo] = state.get("todos") or []  # type: ignore[assignment]
        if not todos or all(t.get("status") == "completed" for t in todos):
            return None

        # 步骤 4：执行提醒次数上限，防止无限重新介入循环
        if self._completion_reminder_count_for_runtime(runtime) >= self._MAX_COMPLETION_REMINDERS:
            return None

        # 步骤 5：为下一次模型请求排队提醒并跳转回模型节点。
        # 不能将此控制指令持久化为普通 HumanMessage，否则会泄露到
        # 用户可见的消息流和保存的对话记录中。
        self._queue_completion_reminder(runtime, _format_completion_reminder(todos))
        return {"jump_to": "model"}

    @override
    @hook_config(can_jump_to=["model"])
    async def aafter_model(
        self,
        state: PlanningState,
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """after_model 的异步版本。"""
        return self.after_model(state, runtime)

    @staticmethod
    def _format_pending_completion_reminders(reminders: list[str]) -> str:
        """将排队的完成提醒列表格式化为单一文本。

        使用 dict.fromkeys 去重并保持顺序，避免重复的提醒内容
        在同一次模型请求中被注入多次。

        Args:
            reminders: 排队的提醒文本列表。

        Returns:
            用双换行符连接的去重提醒文本。
        """
        return "\n\n".join(dict.fromkeys(reminders))

    def _augment_request(self, request: ModelRequest) -> ModelRequest:
        """将排队的完成提醒注入到模型请求的消息列表中。

        排空（drain）当前运行的所有排队提醒，将其格式化为
        HumanMessage 追加到请求消息末尾。此消息标记为
        name="todo_completion_reminder" 并设置 hide_from_ui=True，
        确保不暴露给用户。

        这是 "drain + augment" 模式的核心：在 after_model 中排队提醒，
        在 wrap_model_call 中排空并注入，避免了将控制消息持久化到图状态。

        Args:
            request: 原始模型请求。

        Returns:
            追加了完成提醒消息的新模型请求，或原始请求（无排队提醒时）。
        """
        # 排空当前运行的所有排队提醒
        reminders = self._drain_completion_reminders(request.runtime)
        if not reminders:
            return request
        # 将提醒文本格式化并追加为 HumanMessage
        new_messages = [
            *request.messages,
            HumanMessage(
                content=self._format_pending_completion_reminders(reminders),
                name="todo_completion_reminder",
                additional_kwargs={"hide_from_ui": True},  # 对用户界面隐藏
            ),
        ]
        return request.override(messages=new_messages)

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """包装同步模型调用，在调用前注入排队的完成提醒。

        wrap_model_call 是提醒注入的执行点。after_model 仅负责排队提醒
        和跳转回模型节点，而实际的提醒注入在此处完成。这确保了提醒消息
        仅存在于模型请求层面，不会被持久化到图状态。

        Args:
            request: 原始模型请求。
            handler: 实际执行模型调用的函数。

        Returns:
            模型调用结果。
        """
        return handler(self._augment_request(request))

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """wrap_model_call 的异步版本，在调用前注入排队的完成提醒。"""
        return await handler(self._augment_request(request))

    @override
    def after_agent(self, state: PlanningState, runtime: Runtime) -> dict[str, Any] | None:
        """代理运行结束后，清除当前运行的完成提醒状态。

        确保已结束运行的提醒计数和排队记录不会残留，
        避免对后续运行产生干扰或造成内存泄漏。
        """
        self._clear_current_run_completion_reminders(runtime)
        return None

    @override
    async def aafter_agent(self, state: PlanningState, runtime: Runtime) -> dict[str, Any] | None:
        """after_agent 的异步版本。"""
        self._clear_current_run_completion_reminders(runtime)
        return None
