"""DeerFlow 摘要化中间件扩展模块。

本模块在 LangGraph 原生 SummarizationMiddleware 基础上，增加了以下核心能力：

1. **技能包救援 (Skill Rescue)**：当对话历史因 token 超限被摘要压缩时，
   最近加载的技能文件内容（位于 /mnt/skills 下的 read_file 调用及其 ToolMessage 响应）
   会被识别并保留，避免摘要化后丢失关键技能上下文，从而防止 Agent 重复加载相同的技能文件。

2. **动态上下文提醒保留 (Dynamic Context Reminder Preservation)**：
   DynamicContextMiddleware 注入的隐藏提醒消息（携带当前日期和可选记忆信息）
   必须被排除在摘要压缩之外。如果这些提醒被摘要化，
   DynamicContextMiddleware 会误判摘要 HumanMessage 为首条用户消息，
   导致提醒被注入到错误的位置。

3. **摘要前钩子派发 (Before-Summarization Hook Dispatch)**：
   在消息被实际摘要化移除之前，触发已注册的 BeforeSummarizationHook 回调，
   允许外部系统（如审计日志、监控等）在消息丢失前捕获完整的对话快照。

4. **自定义摘要消息构建**：覆写基类的 _build_new_messages 方法，
   为摘要消息设置 name="summary"，使前端可以识别并隐藏该消息，
   同时仍作为上下文供模型使用。
"""

from __future__ import annotations

import logging
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any, Protocol, override, runtime_checkable

from langchain.agents import AgentState
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.config import get_config
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

from deerflow.agents.middlewares.dynamic_context_middleware import is_dynamic_context_reminder
from deerflow.agents.middlewares.tool_call_metadata import clone_ai_message_with_tool_calls

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SummarizationEvent:
    """摘要化事件上下文，在对话历史被摘要化移除之前触发并传递给钩子。

    该数据类封装了摘要化操作的完整上下文信息，包括即将被摘要化的消息、
    被保留的消息、线程 ID、Agent 名称以及运行时实例。

    Attributes:
        messages_to_summarize: 即将被摘要压缩的消息元组（这些消息的内容将被摘要替换）
        preserved_messages: 将被保留（不参与摘要）的消息元组
        thread_id: 当前线程 ID，可能为 None（无法解析时）
        agent_name: 当前 Agent 名称，可能为 None（无法解析时）
        runtime: LangGraph 运行时实例，用于访问上下文和配置
    """

    messages_to_summarize: tuple[AnyMessage, ...]
    preserved_messages: tuple[AnyMessage, ...]
    thread_id: str | None
    agent_name: str | None
    runtime: Runtime


@runtime_checkable
class BeforeSummarizationHook(Protocol):
    """摘要前钩子协议，在摘要化移除消息之前被调用。

    实现此协议的钩子会在对话历史被摘要压缩之前收到 SummarizationEvent，
    可用于审计日志、指标采集、消息备份等场景。
    钩子异常不会中断摘要化流程，但会被记录到日志中。
    """

    def __call__(self, event: SummarizationEvent) -> None: ...


def _resolve_thread_id(runtime: Runtime) -> str | None:
    """从运行时上下文或 LangGraph 配置中解析当前线程 ID。

    解析优先级：
    1. runtime.context 中的 "thread_id" 字段
    2. LangGraph get_config() 返回的 configurable.thread_id

    如果两种方式都无法获取，返回 None。

    Args:
        runtime: LangGraph 运行时实例

    Returns:
        线程 ID 字符串，或 None（无法解析时）
    """
    # 首先尝试从运行时上下文获取 thread_id
    thread_id = runtime.context.get("thread_id") if runtime.context else None
    if thread_id is None:
        # 运行时上下文中没有，尝试从 LangGraph 配置中获取
        try:
            config_data = get_config()
        except RuntimeError:
            # get_config() 在非 LangGraph 执行上下文中会抛出 RuntimeError
            return None
        thread_id = config_data.get("configurable", {}).get("thread_id")
    return thread_id


def _resolve_agent_name(runtime: Runtime) -> str | None:
    """从运行时上下文或 LangGraph 配置中解析当前 Agent 名称。

    解析优先级：
    1. runtime.context 中的 "agent_name" 字段
    2. LangGraph get_config() 返回的 configurable.agent_name

    如果两种方式都无法获取，返回 None。

    Args:
        runtime: LangGraph 运行时实例

    Returns:
        Agent 名称字符串，或 None（无法解析时）
    """
    # 首先尝试从运行时上下文获取 agent_name
    agent_name = runtime.context.get("agent_name") if runtime.context else None
    if agent_name is None:
        # 运行时上下文中没有，尝试从 LangGraph 配置中获取
        try:
            config_data = get_config()
        except RuntimeError:
            # get_config() 在非 LangGraph 执行上下文中会抛出 RuntimeError
            return None
        agent_name = config_data.get("configurable", {}).get("agent_name")
    return agent_name


def _tool_call_path(tool_call: dict[str, Any]) -> str | None:
    """尽力从类 read_file 工具调用中提取文件路径参数。

    遍历常见的路径参数键名（path, file_path, filepath），
    返回第一个非空字符串值。如果参数不是字典类型或找不到有效路径，返回 None。

    Args:
        tool_call: 工具调用字典，包含 "args" 键

    Returns:
        文件路径字符串，或 None（无法提取时）
    """
    args = tool_call.get("args") or {}
    if not isinstance(args, dict):
        return None
    # 按优先级遍历常见的路径参数键名
    for key in ("path", "file_path", "filepath"):
        value = args.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _clone_ai_message(
    message: AIMessage,
    tool_calls: list[dict[str, Any]],
    *,
    content: Any | None = None,
) -> AIMessage:
    """克隆 AIMessage，同时替换其 tool_calls 列表和可选的 content。

    在技能包救援过程中，需要将一个 AIMessage 的 tool_calls 拆分为
    "被救援的"和"留在摘要中的"两部分，分别生成两个新的 AIMessage。
    本函数用于创建这些克隆消息。

    Args:
        message: 原始 AIMessage
        tool_calls: 替换后的 tool_calls 列表
        content: 可选的替换内容，默认为 None 表示保留原始内容

    Returns:
        克隆后的新 AIMessage 实例
    """
    return clone_ai_message_with_tool_calls(message, tool_calls, content=content)


@dataclass
class _SkillBundle:
    """与单条 AIMessage 关联的技能相关工具调用和工具结果集合。

    当 Agent 加载技能文件时，会产生如下消息序列：
    - AIMessage（包含 read_file 工具调用，路径指向 /mnt/skills/...）
    - ToolMessage（包含技能文件内容）

    _SkillBundle 将这些消息聚合为一个逻辑单元，用于在摘要化时
    识别和救援最近加载的技能内容。

    Attributes:
        ai_index: 所属 AIMessage 在消息列表中的索引位置
        skill_tool_indices: 关联的 ToolMessage 在消息列表中的索引元组
        skill_tool_call_ids: 关联的工具调用 ID 集合（用于匹配 AIMessage.tool_calls 和 ToolMessage）
        skill_tool_tokens: 关联的 ToolMessage 的总 token 数
        skill_key: 去重用的技能键，由排序后的技能路径用 "|" 连接而成，
                   用于识别同一技能的重复加载
    """

    ai_index: int
    skill_tool_indices: tuple[int, ...]
    skill_tool_call_ids: frozenset[str]
    skill_tool_tokens: int
    skill_key: str


class DeerFlowSummarizationMiddleware(SummarizationMiddleware):
    """DeerFlow 摘要化中间件，在基类基础上增加了预压缩钩子派发和技能包救援能力。

    继承自 LangGraph 的 SummarizationMiddleware，扩展了以下功能：

    - **技能包救援**：识别并保留最近加载的技能文件内容，避免摘要化后丢失关键上下文
    - **动态上下文提醒保留**：确保 DynamicContextMiddleware 注入的隐藏提醒消息不被摘要化
    - **摘要前钩子派发**：在消息被摘要化移除之前触发外部钩子回调
    - **自定义摘要消息**：为摘要消息设置 name="summary" 以便前端识别和隐藏

    配置参数通过 __init__ 传入，控制技能救援的行为：
    - skills_container_path: 技能文件在容器中的根路径
    - skill_file_read_tool_names: 被视为读取技能文件的工具名称集合
    - before_summarization: 摘要前钩子列表
    - preserve_recent_skill_count: 最多保留的近期技能包数量
    - preserve_recent_skill_tokens: 近期技能包的总 token 预算
    - preserve_recent_skill_tokens_per_skill: 单个技能包的 token 上限
    """

    def __init__(
        self,
        *args,
        skills_container_path: str | None = None,
        skill_file_read_tool_names: Collection[str] | None = None,
        before_summarization: list[BeforeSummarizationHook] | None = None,
        preserve_recent_skill_count: int = 5,
        preserve_recent_skill_tokens: int = 25_000,
        preserve_recent_skill_tokens_per_skill: int = 5_000,
        **kwargs,
    ) -> None:
        """初始化 DeerFlow 摘要化中间件。

        Args:
            *args: 传递给基类 SummarizationMiddleware 的位置参数
            skills_container_path: 技能文件在容器中的根路径，默认为 "/mnt/skills"
            skill_file_read_tool_names: 被视为读取技能文件的工具名称集合，
                                        默认为 {"read_file", "read", "view", "cat"}
            before_summarization: 摘要前钩子列表，默认为空列表
            preserve_recent_skill_count: 最多保留的近期技能包数量，默认为 5
            preserve_recent_skill_tokens: 近期技能包的总 token 预算，默认为 25000
            preserve_recent_skill_tokens_per_skill: 单个技能包的 token 上限，默认为 5000
            **kwargs: 传递给基类 SummarizationMiddleware 的关键字参数
        """
        super().__init__(*args, **kwargs)
        self._skills_container_path = skills_container_path or "/mnt/skills"
        self._skill_file_read_tool_names = frozenset(skill_file_read_tool_names or {"read_file", "read", "view", "cat"})
        self._before_summarization_hooks = before_summarization or []
        # 确保 count 和 tokens 参数非负
        self._preserve_recent_skill_count = max(0, preserve_recent_skill_count)
        self._preserve_recent_skill_tokens = max(0, preserve_recent_skill_tokens)
        self._preserve_recent_skill_tokens_per_skill = max(0, preserve_recent_skill_tokens_per_skill)

    def before_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        """同步版本的模型前拦截入口，由中间件链调用。

        在模型调用之前检查是否需要摘要化，如果需要则执行摘要化流程。

        Args:
            state: 当前 Agent 状态，包含 messages 等字段
            runtime: LangGraph 运行时实例

        Returns:
            状态更新字典（包含新的 messages 列表），或 None（无需摘要化时）
        """
        return self._maybe_summarize(state, runtime)

    async def abefore_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        """异步版本的模型前拦截入口，由中间件链调用。

        在模型调用之前检查是否需要摘要化，如果需要则异步执行摘要化流程。

        Args:
            state: 当前 Agent 状态，包含 messages 等字段
            runtime: LangGraph 运行时实例

        Returns:
            状态更新字典（包含新的 messages 列表），或 None（无需摘要化时）
        """
        return await self._amaybe_summarize(state, runtime)

    def _maybe_summarize(self, state: AgentState, runtime: Runtime) -> dict | None:
        """执行同步摘要化的核心逻辑。

        摘要化流程按以下步骤执行：
        1. 确保所有消息都有 ID（用于 RemoveMessage 定位）
        2. 计算 token 总数，判断是否需要摘要化
        3. 确定截断索引（cutoff_index），将消息分为"待摘要"和"保留"两部分
        4. 执行技能包救援，将近期技能内容从"待摘要"部分移至"保留"部分
        5. 保留动态上下文提醒消息，确保不被摘要化
        6. 触发摘要前钩子
        7. 生成摘要文本
        8. 构建新的消息列表：摘要消息 + 保留的消息

        最终返回的消息列表以 RemoveMessage(id=REMOVE_ALL_MESSAGES) 开头，
        表示先移除所有现有消息，然后添加摘要和保留的消息。

        Args:
            state: 当前 Agent 状态
            runtime: LangGraph 运行时实例

        Returns:
            状态更新字典，或 None（无需摘要化时）
        """
        messages = state["messages"]
        # 步骤 1：确保所有消息都有唯一 ID，RemoveMessage 需要通过 ID 定位消息
        self._ensure_message_ids(messages)

        # 步骤 2：计算总 token 数并判断是否需要摘要化
        total_tokens = self.token_counter(messages)
        if not self._should_summarize(messages, total_tokens):
            return None

        # 步骤 3：确定截断索引，划分"待摘要"和"保留"的分界点
        cutoff_index = self._determine_cutoff_index(messages)
        if cutoff_index <= 0:
            return None

        # 步骤 4：执行技能包救援，将近期技能内容从待摘要区移至保留区
        messages_to_summarize, preserved_messages = self._partition_with_skill_rescue(messages, cutoff_index)
        # 步骤 5：保留动态上下文提醒消息，确保不被摘要化
        messages_to_summarize, preserved_messages = self._preserve_dynamic_context_reminders(messages_to_summarize, preserved_messages)
        # 步骤 6：触发摘要前钩子
        self._fire_hooks(messages_to_summarize, preserved_messages, runtime)
        # 步骤 7：生成摘要文本
        summary = self._create_summary(messages_to_summarize)
        # 步骤 8：构建新的消息列表
        new_messages = self._build_new_messages(summary)

        # 返回状态更新：先移除所有消息，再添加摘要和保留的消息
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *new_messages,
                *preserved_messages,
            ]
        }

    async def _amaybe_summarize(self, state: AgentState, runtime: Runtime) -> dict | None:
        """执行异步摘要化的核心逻辑。

        与 _maybe_summarize 流程完全一致，唯一区别是摘要生成步骤使用异步方法
        （await self._acreate_summary），适用于异步 LLM 调用场景。

        摘要化流程按以下步骤执行：
        1. 确保所有消息都有 ID（用于 RemoveMessage 定位）
        2. 计算 token 总数，判断是否需要摘要化
        3. 确定截断索引（cutoff_index），将消息分为"待摘要"和"保留"两部分
        4. 执行技能包救援，将近期技能内容从"待摘要"部分移至"保留"部分
        5. 保留动态上下文提醒消息，确保不被摘要化
        6. 触发摘要前钩子
        7. 异步生成摘要文本
        8. 构建新的消息列表：摘要消息 + 保留的消息

        Args:
            state: 当前 Agent 状态
            runtime: LangGraph 运行时实例

        Returns:
            状态更新字典，或 None（无需摘要化时）
        """
        messages = state["messages"]
        # 步骤 1：确保所有消息都有唯一 ID
        self._ensure_message_ids(messages)

        # 步骤 2：计算总 token 数并判断是否需要摘要化
        total_tokens = self.token_counter(messages)
        if not self._should_summarize(messages, total_tokens):
            return None

        # 步骤 3：确定截断索引
        cutoff_index = self._determine_cutoff_index(messages)
        if cutoff_index <= 0:
            return None

        # 步骤 4：执行技能包救援
        messages_to_summarize, preserved_messages = self._partition_with_skill_rescue(messages, cutoff_index)
        # 步骤 5：保留动态上下文提醒消息
        messages_to_summarize, preserved_messages = self._preserve_dynamic_context_reminders(messages_to_summarize, preserved_messages)
        # 步骤 6：触发摘要前钩子
        self._fire_hooks(messages_to_summarize, preserved_messages, runtime)
        # 步骤 7：异步生成摘要文本
        summary = await self._acreate_summary(messages_to_summarize)
        # 步骤 8：构建新的消息列表
        new_messages = self._build_new_messages(summary)

        # 返回状态更新：先移除所有消息，再添加摘要和保留的消息
        return {
            "messages": [
                RemoveMessage(id=REMOVE_ALL_MESSAGES),
                *new_messages,
                *preserved_messages,
            ]
        }

    @override
    def _build_new_messages(self, summary: str) -> list[HumanMessage]:
        """覆写基类实现，为摘要消息设置特殊名称 'summary'。

        生成的 HumanMessage 带有 name="summary" 属性，前端据此识别该消息为摘要，
        在界面中隐藏不显示，但仍然作为上下文供 LLM 模型使用。
        这样用户不会看到历史对话被压缩成一条摘要消息，但模型仍然可以
        利用摘要内容理解之前的对话上下文。

        Args:
            summary: 摘要文本内容

        Returns:
            包含单条带 name="summary" 的 HumanMessage 列表
        """
        return [HumanMessage(content=f"Here is a summary of the conversation to date:\n\n{summary}", name="summary")]

    def _preserve_dynamic_context_reminders(
        self,
        messages_to_summarize: list[AnyMessage],
        preserved_messages: list[AnyMessage],
    ) -> tuple[list[AnyMessage], list[AnyMessage]]:
        """将隐藏的动态上下文提醒消息从待摘要区移至保留区。

        DynamicContextMiddleware 会在对话中注入隐藏的提醒消息（携带当前日期和可选记忆信息）。
        这些提醒消息必须被排除在摘要压缩之外，原因如下：

        - 如果提醒消息被摘要化，DynamicContextMiddleware 在下一轮对话中
          会误判摘要 HumanMessage（name="summary"）为第一条用户消息，
          从而在错误的位置注入新的提醒消息。

        - 保留这些提醒消息可以确保 DynamicContextMiddleware 能正确识别
          已有提醒的位置，避免重复注入。

        具体操作：从 messages_to_summarize 中筛选出所有动态上下文提醒消息，
        将它们添加到 preserved_messages 的开头（保持时间顺序）。

        Args:
            messages_to_summarize: 待摘要化的消息列表
            preserved_messages: 已保留的消息列表

        Returns:
            元组 (remaining, reminders + preserved)：
            - remaining: 从待摘要列表中移除提醒消息后的剩余消息
            - reminders + preserved: 提醒消息 + 原保留消息
        """
        # 识别待摘要消息中的所有动态上下文提醒
        reminders = [msg for msg in messages_to_summarize if is_dynamic_context_reminder(msg)]
        if not reminders:
            # 没有提醒消息需要保留，直接返回原列表
            return messages_to_summarize, preserved_messages

        # 从待摘要消息中移除提醒消息，保留其余消息
        remaining = [msg for msg in messages_to_summarize if not is_dynamic_context_reminder(msg)]
        # 提醒消息添加到保留列表开头，保持时间顺序
        return remaining, reminders + preserved_messages

    def _partition_with_skill_rescue(
        self,
        messages: list[AnyMessage],
        cutoff_index: int,
    ) -> tuple[list[AnyMessage], list[AnyMessage]]:
        """在基类分区基础上，救援最近加载的技能包内容。

        该方法执行以下步骤：
        1. 调用基类的 _partition_messages 按 cutoff_index 分区
        2. 在待摘要区中查找所有技能包（_SkillBundle）
        3. 按预算选择需要救援的技能包
        4. 将被救援的技能相关消息从待摘要区移至保留区

        技能包救援的核心逻辑：
        - 当一个 AIMessage 的 tool_calls 中部分属于技能加载、部分不属于时，
          需要将该 AIMessage 拆分为两个克隆消息：
          - 一个只包含被救援的技能 tool_calls（content 设为空字符串）
          - 一个只包含剩余的 tool_calls 和原始 content
        - 对应的 ToolMessage 如果属于被救援的 tool_call_id，则整体移至保留区

        这种拆分确保了：
        - 被救援的技能内容完整保留（AIMessage + ToolMessage 配对）
        - 非技能内容仍然被摘要化
        - AIMessage 的 tool_calls 与 ToolMessage 的 tool_call_id 对应关系不被破坏

        如果技能包查找过程中出现异常，安全回退到基类的默认分区结果。

        Args:
            messages: 完整的消息列表
            cutoff_index: 截断索引，此索引之前的消息为待摘要区

        Returns:
            元组 (remaining, rescued + preserved)：
            - remaining: 待摘要区中未被救援的消息
            - rescued + preserved: 被救援的技能消息 + 原保留区的消息
        """
        # 步骤 1：调用基类分区方法
        to_summarize, preserved = self._partition_messages(messages, cutoff_index)

        # 如果技能救援功能被禁用（count 或 tokens 为 0）或待摘要区为空，直接返回
        if self._preserve_recent_skill_count == 0 or self._preserve_recent_skill_tokens == 0 or not to_summarize:
            return to_summarize, preserved

        # 步骤 2：在待摘要区中查找技能包
        try:
            bundles = self._find_skill_bundles(to_summarize, self._skills_container_path)
        except Exception:
            # 查找过程出错时安全回退，记录异常但不中断摘要化流程
            logger.exception("Skill-preserving summarization rescue failed; falling back to default partition")
            return to_summarize, preserved

        if not bundles:
            # 没有找到任何技能包，直接返回
            return to_summarize, preserved

        # 步骤 3：按预算选择需要救援的技能包
        rescue_bundles = self._select_bundles_to_rescue(bundles)
        if not rescue_bundles:
            # 没有符合预算的技能包需要救援
            return to_summarize, preserved

        # 步骤 4：执行救援，将技能相关消息从待摘要区移至保留区
        bundles_by_ai_index = {bundle.ai_index: bundle for bundle in rescue_bundles}
        rescue_tool_indices = {idx for bundle in rescue_bundles for idx in bundle.skill_tool_indices}
        rescued: list[AnyMessage] = []  # 被救援的消息
        remaining: list[AnyMessage] = []  # 留在待摘要区的消息

        for i, msg in enumerate(to_summarize):
            bundle = bundles_by_ai_index.get(i)
            # 处理包含技能 tool_calls 的 AIMessage：需要拆分
            if bundle is not None and isinstance(msg, AIMessage):
                # 将 tool_calls 分为"被救援的"和"留在摘要中的"两组
                rescued_tool_calls = [tc for tc in msg.tool_calls if tc.get("id") in bundle.skill_tool_call_ids]
                remaining_tool_calls = [tc for tc in msg.tool_calls if tc.get("id") not in bundle.skill_tool_call_ids]

                # 被救援的 tool_calls 生成一个克隆消息（content 设为空，因为内容在 ToolMessage 中）
                if rescued_tool_calls:
                    rescued.append(_clone_ai_message(msg, rescued_tool_calls, content=""))
                # 剩余的 tool_calls 或非空 content 生成另一个克隆消息
                if remaining_tool_calls or msg.content:
                    remaining.append(_clone_ai_message(msg, remaining_tool_calls))
                continue

            # 处理被救援的 ToolMessage：整体移至保留区
            if i in rescue_tool_indices:
                rescued.append(msg)
                continue

            # 其他消息留在待摘要区
            remaining.append(msg)

        # 被救援的消息添加到保留列表开头，保持时间顺序
        return remaining, rescued + preserved

    def _find_skill_bundles(
        self,
        messages: list[AnyMessage],
        skills_root: str,
    ) -> list[_SkillBundle]:
        """在消息列表中定位所有技能加载相关的 AIMessage + ToolMessage 组合。

        扫描消息列表，识别以下模式：
        - AIMessage 包含指向 skills_root 下的 read_file 工具调用
        - 紧随其后的 ToolMessage 的 tool_call_id 与上述工具调用匹配

        将每组匹配的消息聚合为一个 _SkillBundle，包含：
        - AIMessage 的索引位置
        - 关联的 ToolMessage 索引位置
        - 工具调用 ID 集合（用于后续拆分 AIMessage）
        - ToolMessage 的总 token 数（用于预算控制）
        - 技能键（用于去重，由路径排序拼接而成）

        扫描策略：顺序遍历消息列表，遇到带 tool_calls 的 AIMessage 时，
        检查其中是否有技能相关的工具调用；如果有，继续扫描后续连续的
        ToolMessage 以匹配对应的响应。

        Args:
            messages: 消息列表（通常是待摘要区的消息）
            skills_root: 技能文件根路径（如 "/mnt/skills"）

        Returns:
            找到的 _SkillBundle 列表，按消息出现顺序排列
        """
        bundles: list[_SkillBundle] = []
        n = len(messages)
        i = 0
        while i < n:
            msg = messages[i]
            # 跳过没有 tool_calls 的非 AIMessage
            if not (isinstance(msg, AIMessage) and msg.tool_calls):
                i += 1
                continue

            # 检查当前 AIMessage 的 tool_calls 中是否有技能相关的调用
            tool_calls = list(msg.tool_calls)
            skill_paths_by_id: dict[str, str] = {}  # tool_call_id -> 技能文件路径
            for tc in tool_calls:
                if self._is_skill_tool_call(tc, skills_root):
                    tc_id = tc.get("id")
                    path = _tool_call_path(tc)
                    if tc_id and path:
                        skill_paths_by_id[tc_id] = path

            if not skill_paths_by_id:
                # 当前 AIMessage 没有技能相关的工具调用，跳过
                i += 1
                continue

            # 扫描 AIMessage 之后的连续 ToolMessage，收集匹配的技能工具响应
            skill_tool_tokens = 0  # 技能 ToolMessage 的总 token 数
            skill_key_parts: list[str] = []  # 技能路径列表，用于生成去重键
            skill_tool_indices: list[int] = []  # 技能 ToolMessage 的索引列表
            matched_skill_call_ids: set[str] = set()  # 匹配到的 tool_call_id 集合

            # 找到连续 ToolMessage 消息块的结束位置
            j = i + 1
            while j < n and isinstance(messages[j], ToolMessage):
                j += 1

            # 在 ToolMessage 块中匹配技能相关的响应
            for k in range(i + 1, j):
                tool_msg = messages[k]
                if isinstance(tool_msg, ToolMessage) and tool_msg.tool_call_id in skill_paths_by_id:
                    skill_tool_tokens += self.token_counter([tool_msg])
                    skill_key_parts.append(skill_paths_by_id[tool_msg.tool_call_id])
                    skill_tool_indices.append(k)
                    matched_skill_call_ids.add(tool_msg.tool_call_id)

            if not skill_tool_indices:
                # 没有找到匹配的 ToolMessage，跳过到 ToolMessage 块之后
                i = j
                continue

            # 构建技能包，skill_key 用排序后的路径拼接以支持去重
            bundles.append(
                _SkillBundle(
                    ai_index=i,
                    skill_tool_indices=tuple(skill_tool_indices),
                    skill_tool_call_ids=frozenset(matched_skill_call_ids),
                    skill_tool_tokens=skill_tool_tokens,
                    skill_key="|".join(sorted(skill_key_parts)),
                )
            )
            # 继续从 ToolMessage 块之后扫描
            i = j

        return bundles

    def _select_bundles_to_rescue(self, bundles: list[_SkillBundle]) -> list[_SkillBundle]:
        """按照预算约束选择需要救援的技能包，优先保留最新的。

        选择策略：
        1. 从最新的技能包开始向前遍历（reversed）
        2. 同一技能键（skill_key）只保留一次，避免重复加载同一技能
        3. 单个技能包的 token 数不得超过 preserve_recent_skill_tokens_per_skill
        4. 已选技能包的总 token 数不得超过 preserve_recent_skill_tokens
        5. 已选技能包的数量不得超过 preserve_recent_skill_count

        这种"最新优先"策略确保 Agent 始终保留最近使用的技能上下文，
        因为最近的技能更可能与当前任务相关。

        遍历完成后将结果反转回时间顺序（最早在前），保证消息顺序正确。

        Args:
            bundles: 按时间顺序排列的技能包列表

        Returns:
            按时间顺序排列的待救援技能包列表
        """
        selected: list[_SkillBundle] = []
        if not bundles:
            return selected

        seen_skill_keys: set[str] = set()  # 已选技能的去重键集合
        total_tokens = 0  # 已选技能包的累计 token 数
        kept = 0  # 已选技能包的数量

        # 从最新的技能包开始向前遍历
        for bundle in reversed(bundles):
            # 达到数量上限，停止选择
            if kept >= self._preserve_recent_skill_count:
                break
            # 同一技能已经选择过，跳过（避免重复保留）
            if bundle.skill_key in seen_skill_keys:
                continue
            # 单个技能包超过 token 上限，跳过
            if bundle.skill_tool_tokens > self._preserve_recent_skill_tokens_per_skill:
                continue
            # 累计 token 超过总预算，跳过
            if total_tokens + bundle.skill_tool_tokens > self._preserve_recent_skill_tokens:
                continue

            # 选中此技能包
            selected.append(bundle)
            total_tokens += bundle.skill_tool_tokens
            kept += 1
            seen_skill_keys.add(bundle.skill_key)

        # 反转回时间顺序（最早在前），保证消息顺序正确
        selected.reverse()
        return selected

    def _is_skill_tool_call(self, tool_call: dict[str, Any], skills_root: str) -> bool:
        """判断给定的工具调用是否为读取技能文件的操作。

        判断条件：
        1. 工具名称在 skill_file_read_tool_names 集合中
        2. 工具调用参数中包含有效的文件路径
        3. 文件路径等于技能根路径或位于技能根路径之下

        Args:
            tool_call: 工具调用字典，包含 "name" 和 "args" 键
            skills_root: 技能文件根路径

        Returns:
            True 如果该工具调用是读取技能文件的操作，否则 False
        """
        name = tool_call.get("name") or ""
        # 检查工具名称是否为文件读取类工具
        if name not in self._skill_file_read_tool_names:
            return False
        # 提取文件路径参数
        path = _tool_call_path(tool_call)
        if not path:
            return False
        # 规范化根路径（移除末尾斜杠），判断路径是否在技能目录下
        normalized_root = skills_root.rstrip("/")
        return path == normalized_root or path.startswith(normalized_root + "/")

    def _fire_hooks(
        self,
        messages_to_summarize: list[AnyMessage],
        preserved_messages: list[AnyMessage],
        runtime: Runtime,
    ) -> None:
        """触发所有已注册的摘要前钩子。

        在消息被实际摘要化移除之前，构造 SummarizationEvent 并依次调用
        所有 before_summarization 钩子。每个钩子的异常会被捕获并记录到日志，
        不会中断其他钩子的执行或摘要化流程。

        这种容错设计确保了：
        - 单个钩子失败不影响其他钩子的执行
        - 钩子异常不会阻止摘要化操作（摘要化是必要的，否则会超出 token 限制）

        Args:
            messages_to_summarize: 即将被摘要化的消息列表
            preserved_messages: 将被保留的消息列表
            runtime: LangGraph 运行时实例
        """
        if not self._before_summarization_hooks:
            # 没有注册任何钩子，直接返回
            return

        # 构造摘要化事件，将列表转为元组以防止外部钩子修改
        event = SummarizationEvent(
            messages_to_summarize=tuple(messages_to_summarize),
            preserved_messages=tuple(preserved_messages),
            thread_id=_resolve_thread_id(runtime),
            agent_name=_resolve_agent_name(runtime),
            runtime=runtime,
        )

        # 依次调用所有钩子，每个钩子的异常独立捕获
        for hook in self._before_summarization_hooks:
            try:
                hook(event)
            except Exception:
                hook_name = getattr(hook, "__name__", None) or type(hook).__name__
                logger.exception("before_summarization hook %s failed", hook_name)
