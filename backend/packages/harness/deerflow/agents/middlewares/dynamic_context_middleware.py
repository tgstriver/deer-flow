"""动态上下文中间件 —— 将记忆和当前日期以 <system-reminder> 的形式注入对话。

设计思路
========

**冻结快照模式（frozen-snapshot pattern）**

系统提示词（system prompt）在所有用户和会话之间保持完全静态，以最大化前缀缓存（prefix-cache）
的命中率。动态内容（当前日期、用户记忆）不混入系统提示词，而是作为一条独立的
<system-reminder> HumanMessage 插入到第一条用户消息之前。该消息一旦注入即被"冻结"——
后续轮次不会再修改其内容，从而保证前缀缓存在整个会话期间持续有效。

**午夜跨越检测（midnight crossing detection）**

当对话持续到次日时，中间件会检测到当前日期与之前注入的日期不一致，随后在当前轮次之前
插入一条轻量的日期更新提醒。该修正会被持久化到消息历史中，因此新日期的后续轮次能看到
一致的日期记录，不会重复注入。

注入格式
--------

首次完整提醒：

    <system-reminder>
    <memory>...</memory>

    <current_date>2026-05-08, Friday</current_date>
    </system-reminder>

午夜跨越后仅更新日期：

    <system-reminder>
    <current_date>2026-05-09, Saturday</current_date>
    </system-reminder>
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, override

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

if TYPE_CHECKING:
    from deerflow.config.app_config import AppConfig

logger = logging.getLogger(__name__)

# ── 正则与常量 ──────────────────────────────────────────────────────────
# 用于从消息内容中提取 <current_date> 标签值的正则表达式
_DATE_RE = re.compile(r"<current_date>([^<]+)</current_date>")
# HumanMessage.additional_kwargs 中的标记键，用于标识本中间件注入的提醒消息
_DYNAMIC_CONTEXT_REMINDER_KEY = "dynamic_context_reminder"
# 摘要消息的 name 标识，这类消息不应被视为动态上下文的注入目标
_SUMMARY_MESSAGE_NAME = "summary"


def _extract_date(content: str) -> str | None:
    """从文本内容中提取第一个 <current_date> 标签的值，未找到则返回 None。"""
    m = _DATE_RE.search(content)
    return m.group(1) if m else None


def is_dynamic_context_reminder(message: object) -> bool:
    """判断给定消息是否为本中间件注入的隐藏动态上下文提醒。

    通过检查 additional_kwargs 中的 ``dynamic_context_reminder`` 标记来判断，
    而非依赖内容子串匹配，以避免将用户消息中恰好包含 ``<system-reminder>`` 的
    情况误判为注入提醒。该函数是公共 API，被 summarization_middleware、
    title_middleware、prompt 等模块引用。
    """
    return isinstance(message, HumanMessage) and bool(message.additional_kwargs.get(_DYNAMIC_CONTEXT_REMINDER_KEY))


def _last_injected_date(messages: list) -> str | None:
    """从消息列表中逆序扫描，返回最近一次注入的日期字符串。

    使用 ``dynamic_context_reminder`` additional_kwargs 标记进行检测，
    而非内容子串匹配，因此用户消息中包含 ``<system-reminder>`` 时
    不会被误识别为注入提醒。返回 None 表示尚未注入过任何动态上下文提醒。
    """
    # 逆序遍历，找到最近一条动态上下文提醒消息
    for msg in reversed(messages):
        if is_dynamic_context_reminder(msg):
            content_str = msg.content if isinstance(msg.content, str) else str(msg.content)
            return _extract_date(content_str)
    return None


def _is_user_injection_target(message: object) -> bool:
    """判断给定消息是否可以作为动态上下文提醒的注入目标。

    符合条件的消息必须同时满足：
    1. 是 HumanMessage 实例
    2. 不是本中间件注入的隐藏提醒消息（避免对已注入消息重复注入）
    3. 不是摘要消息（summary 消息由 SummarizationMiddleware 生成，不应插入提醒）
    """
    return isinstance(message, HumanMessage) and not is_dynamic_context_reminder(message) and message.name != _SUMMARY_MESSAGE_NAME


class DynamicContextMiddleware(AgentMiddleware):
    """动态上下文中间件 —— 将记忆和当前日期以 <system-reminder> 注入 HumanMessage。

    首轮注入
    --------
    在第一条 HumanMessage 前插入完整的 system-reminder（记忆 + 日期），并通过
    ID 交换技术（ID-swap technique）使其与原消息一同持久化。注入后该消息内容
    在整个会话期间保持冻结（frozen），从而保证前缀缓存持续命中。

    午夜跨越
    --------
    如果对话跨越午夜，当前日期与之前注入的日期不同，则在当前（最后一条）
    HumanMessage 前插入一条轻量的日期更新提醒并持久化。新日期的后续轮次
    能在历史记录中看到修正后的日期，不会重复注入。
    """

    def __init__(self, agent_name: str | None = None, *, app_config: AppConfig | None = None):
        super().__init__()
        self._agent_name = agent_name  # 智能体名称，用于获取对应智能体的记忆上下文
        self._app_config = app_config  # 应用配置，控制记忆注入开关等行为

    def _build_full_reminder(self) -> str:
        """构建完整的动态上下文提醒内容，包含记忆和当前日期。

        记忆注入受 ``memory.injection_enabled`` 配置项控制；当前日期始终包含。
        返回格式为包含 <system-reminder> 标签的字符串。
        """
        from deerflow.agents.lead_agent.prompt import _get_memory_context

        # 记忆注入开关：配置存在且 injection_enabled 为 True 时才注入记忆
        injection_enabled = self._app_config.memory.injection_enabled if self._app_config else True
        # 获取该智能体的记忆上下文（用户偏好、事实等）
        memory_context = _get_memory_context(self._agent_name, app_config=self._app_config) if injection_enabled else ""
        # 格式化当前日期，如 "2026-05-08, Friday"
        current_date = datetime.now().strftime("%Y-%m-%d, %A")

        lines: list[str] = ["<system-reminder>"]
        if memory_context:
            lines.append(memory_context.strip())
            lines.append("")  # 记忆与日期之间的空行分隔
        lines.append(f"<current_date>{current_date}</current_date>")
        lines.append("</system-reminder>")

        return "\n".join(lines)

    def _build_date_update_reminder(self) -> str:
        """构建仅包含当前日期的轻量更新提醒，用于午夜跨越场景。

        与完整提醒不同，此提醒不包含记忆内容，仅更新日期信息。
        """
        current_date = datetime.now().strftime("%Y-%m-%d, %A")
        return "\n".join(
            [
                "<system-reminder>",
                f"<current_date>{current_date}</current_date>",
                "</system-reminder>",
            ]
        )

    @staticmethod
    def _make_reminder_and_user_messages(original: HumanMessage, reminder_content: str) -> tuple[HumanMessage, HumanMessage]:
        """使用 ID 交换技术（ID-swap technique）生成提醒消息和用户消息。

        核心思路：让提醒消息"继承"原始消息的 ID，从而利用 LangGraph 的
        add_messages 机制实现原地替换（保留原始消息的位置）。原始用户内容则
        携带派生 ID ``{id}__user``，由 add_messages 紧跟在提醒消息之后追加。

        这样做的好处是：
        - 提醒消息出现在用户消息之前（符合 LLM 的阅读顺序）
        - 原始消息的 ID 被保留在提醒消息上，确保 add_messages 正确替换
        - 提醒消息被标记为 hide_from_ui 和 dynamic_context_reminder，前端可隐藏

        当原始消息没有 ID 时，会生成一个稳定的 UUID，以避免派生 ID
        退化为歧义的 ``None__user`` 字符串。

        参数:
            original: 原始的 HumanMessage，需要在其前插入提醒
            reminder_content: 提醒消息的文本内容

        返回:
            (reminder_msg, user_msg) 元组，提醒消息在前，用户消息在后
        """
        # 确保有一个稳定的 ID：优先使用原始消息的 ID，否则生成新的 UUID
        stable_id = original.id or str(uuid.uuid4())

        # 提醒消息：继承原始 ID（实现原地替换），标记为隐藏且为动态上下文提醒
        reminder_msg = HumanMessage(
            content=reminder_content,
            id=stable_id,
            additional_kwargs={"hide_from_ui": True, _DYNAMIC_CONTEXT_REMINDER_KEY: True},
        )

        # 用户消息：保留原始内容，使用派生 ID（{stable_id}__user），
        # add_messages 会将其追加到同名 ID 消息之后
        user_msg = HumanMessage(
            content=original.content,
            id=f"{stable_id}__user",
            name=original.name,
            additional_kwargs=original.additional_kwargs,
        )
        return reminder_msg, user_msg

    def _inject(self, state) -> dict | None:
        """核心注入逻辑：根据对话状态决定注入完整提醒、日期更新提醒或不操作。

        三种情况：
        1. last_date 为 None → 首轮，注入完整提醒（记忆 + 日期）
        2. last_date == current_date → 同日，无需操作
        3. last_date != current_date → 跨越午夜，注入日期更新提醒

        参数:
            state: LangGraph 状态字典，包含 "messages" 键

        返回:
            包含新消息列表的字典，或 None（无需注入时）
        """
        messages = list(state.get("messages", []))
        if not messages:
            return None

        # 获取当前日期和上次注入的日期，用于午夜跨越检测
        current_date = datetime.now().strftime("%Y-%m-%d, %A")
        last_date = _last_injected_date(messages)
        logger.debug(
            "DynamicContextMiddleware._inject: msg_count=%d last_date=%r current_date=%r",
            len(messages),
            last_date,
            current_date,
        )

        if last_date is None:
            # ── 首轮：注入完整提醒（记忆 + 日期）作为独立 HumanMessage ────
            # 找到第一条可作为注入目标的用户消息
            first_idx = next((i for i, m in enumerate(messages) if _is_user_injection_target(m)), None)
            if first_idx is None:
                return None
            full_reminder = self._build_full_reminder()
            logger.info(
                "DynamicContextMiddleware: injecting full reminder (len=%d, has_memory=%s) into first HumanMessage id=%r",
                len(full_reminder),
                "<memory>" in full_reminder,
                messages[first_idx].id,
            )
            # 使用 ID 交换技术：提醒消息替换原消息位置，用户消息紧随其后
            reminder_msg, user_msg = self._make_reminder_and_user_messages(messages[first_idx], full_reminder)
            return {"messages": [reminder_msg, user_msg]}

        if last_date == current_date:
            # ── 同日：无需操作 ──────────────────────────────────────────────
            return None

        # ── 午夜跨越：注入日期更新提醒作为独立 HumanMessage ───────────────
        # 找到最后一条可作为注入目标的用户消息（即当前轮次）
        last_human_idx = next((i for i in reversed(range(len(messages))) if _is_user_injection_target(messages[i])), None)
        if last_human_idx is None:
            return None

        # 使用 ID 交换技术在当前用户消息前插入日期更新
        reminder_msg, user_msg = self._make_reminder_and_user_messages(messages[last_human_idx], self._build_date_update_reminder())
        logger.info("DynamicContextMiddleware: midnight crossing detected — injected date update before current turn")
        return {"messages": [reminder_msg, user_msg]}

    @override
    def before_agent(self, state, runtime: Runtime) -> dict | None:
        """同步版本的 Agent 前置钩子，在智能体处理消息前注入动态上下文。"""
        return self._inject(state)

    @override
    async def abefore_agent(self, state, runtime: Runtime) -> dict | None:
        """异步版本的 Agent 前置钩子，逻辑与同步版本完全一致。"""
        return self._inject(state)
