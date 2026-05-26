"""纯参数代理工厂模块，用于创建 DeerFlow 代理。

``create_deerflow_agent`` 接受纯 Python 参数 —— 无需 YAML 文件，无需
全局单例。它是位于原始 ``langchain.agents.create_agent`` 原语与
配置驱动的 ``make_lead_agent`` 应用工厂之间的 SDK 级入口。

注意：工厂组装本身是无配置的，但部分注入的运行时组件（例如用于子代理的
``task_tool``）在调用时仍可能读取全局配置。完全无配置的运行时是第二阶段
（Phase 2）的目标。
"""
# Pure-argument factory for DeerFlow agents.
#
# ``create_deerflow_agent`` accepts plain Python arguments — no YAML files, no
# global singletons.  It is the SDK-level entry point sitting between the raw
# ``langchain.agents.create_agent`` primitive and the config-driven
# ``make_lead_agent`` application factory.
#
# Note: the factory assembly itself is config-free, but some injected runtime
# components (e.g. ``task_tool`` for subagent) may still read global config at
# invocation time.  Full config-free runtime is a Phase 2 goal.

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware

from deerflow.agents.features import RuntimeFeatures
from deerflow.agents.middlewares.clarification_middleware import ClarificationMiddleware
from deerflow.agents.middlewares.dangling_tool_call_middleware import DanglingToolCallMiddleware
from deerflow.agents.middlewares.tool_error_handling_middleware import ToolErrorHandlingMiddleware
from deerflow.agents.thread_state import ThreadState
from deerflow.tools.builtins import ask_clarification_tool

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langchain_core.tools import BaseTool
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TodoMiddleware prompts (minimal SDK version)
# TodoMiddleware 提示词（最小化 SDK 版本）
# ---------------------------------------------------------------------------

_TODO_SYSTEM_PROMPT = """
<todo_list_system>
You have access to the `write_todos` tool to help you manage and track complex multi-step objectives.

**CRITICAL RULES:**
- Mark todos as completed IMMEDIATELY after finishing each step - do NOT batch completions
- Keep EXACTLY ONE task as `in_progress` at any time (unless tasks can run in parallel)
- Update the todo list in REAL-TIME as you work - this gives users visibility into your progress
- DO NOT use this tool for simple tasks (< 3 steps) - just complete them directly
</todo_list_system>
"""

_TODO_TOOL_DESCRIPTION = "Use this tool to create and manage a structured task list for complex work sessions.  Only use for complex tasks (3+ steps)."


# ---------------------------------------------------------------------------
# Public API / 公共 API
# ---------------------------------------------------------------------------


def create_deerflow_agent(
    model: BaseChatModel,
    tools: list[BaseTool] | None = None,
    *,
    system_prompt: str | None = None,
    middleware: list[AgentMiddleware] | None = None,
    features: RuntimeFeatures | None = None,
    extra_middleware: list[AgentMiddleware] | None = None,
    plan_mode: bool = False,
    state_schema: type | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    name: str = "default",
) -> CompiledStateGraph:
    """从纯 Python 参数创建 DeerFlow 代理。

    工厂组装本身不读取配置文件。部分注入的运行时组件（例如 ``task_tool``）
    在调用时可能仍依赖全局配置 —— 参见 Phase 2 路线图以实现完全无配置运行时。

    Parameters / 参数
    ----------
    model:
        Chat model instance. / 聊天模型实例。
    tools:
        User-provided tools.  Feature-injected tools are appended automatically.
        / 用户提供的工具。特性注入的工具会自动追加。
    system_prompt:
        System message.  ``None`` uses a minimal default.
        / 系统消息。``None`` 使用最小默认值。
    middleware:
        **Full takeover** — if provided, this exact list is used.
        Cannot be combined with *features* or *extra_middleware*.
        / **完全接管** —— 如果提供，则使用此精确列表。
        不能与 *features* 或 *extra_middleware* 组合使用。
    features:
        Declarative feature flags.  Cannot be combined with *middleware*.
        / 声明式特性标志。不能与 *middleware* 组合使用。
    extra_middleware:
        Additional middlewares inserted into the auto-assembled chain via
        ``@Next``/``@Prev`` positioning.  Cannot be used with *middleware*.
        / 通过 ``@Next``/``@Prev`` 定位插入到自动组装链中的额外中间件。
        不能与 *middleware* 组合使用。
    plan_mode:
        Enable TodoMiddleware for task tracking.
        / 启用 TodoMiddleware 进行任务跟踪。
    state_schema:
        LangGraph state type.  Defaults to ``ThreadState``.
        / LangGraph 状态类型。默认为 ``ThreadState``。
    checkpointer:
        Optional persistence backend. / 可选的持久化后端。
    name:
        Agent name (passed to middleware that cares, e.g. ``MemoryMiddleware``).
        / 代理名称（传递给需要的中间件，例如 ``MemoryMiddleware``）。

    Raises / 异常
    ------
    ValueError
        If both *middleware* and *features*/*extra_middleware* are provided.
        / 如果同时提供了 *middleware* 和 *features*/*extra_middleware*。
    """
    # 参数互斥校验：middleware（完全接管）与 features/extra_middleware 不可共存
    if middleware is not None and features is not None:
        raise ValueError("Cannot specify both 'middleware' and 'features'.  Use one or the other.")
    if middleware is not None and extra_middleware:
        raise ValueError("Cannot use 'extra_middleware' with 'middleware' (full takeover).")
    if extra_middleware:
        for mw in extra_middleware:
            if not isinstance(mw, AgentMiddleware):
                raise TypeError(f"extra_middleware items must be AgentMiddleware instances, got {type(mw).__name__}")

    effective_tools: list[BaseTool] = list(tools or [])  # 有效工具列表
    effective_state = state_schema or ThreadState  # 有效状态类型，默认 ThreadState

    if middleware is not None:
        # 完全接管模式：直接使用用户提供的中间件列表
        effective_middleware = list(middleware)
    else:
        # 特性驱动模式：从 RuntimeFeatures 自动组装中间件链
        feat = features or RuntimeFeatures()
        effective_middleware, extra_tools = _assemble_from_features(
            feat,
            name=name,
            plan_mode=plan_mode,
            extra_middleware=extra_middleware or [],
        )
        # Deduplicate by tool name — user-provided tools take priority.
        # 按工具名去重 —— 用户提供的工具优先级更高。
        existing_names = {t.name for t in effective_tools}
        for t in extra_tools:
            if t.name not in existing_names:
                effective_tools.append(t)
                existing_names.add(t.name)

    return create_agent(
        model=model,
        tools=effective_tools or None,
        middleware=effective_middleware,
        system_prompt=system_prompt,
        state_schema=effective_state,
        checkpointer=checkpointer,
        name=name,
    )


# ---------------------------------------------------------------------------
# Internal: feature-driven middleware assembly
# 内部：特性驱动的中间件组装
# ---------------------------------------------------------------------------


def _assemble_from_features(
    feat: RuntimeFeatures,
    *,
    name: str = "default",
    plan_mode: bool = False,
    extra_middleware: list[AgentMiddleware] | None = None,
) -> tuple[list[AgentMiddleware], list[BaseTool]]:
    """根据 *feat* 构建有序的中间件链 + 额外工具。

    Middleware order matches ``make_lead_agent`` (14 middlewares):
    中间件顺序与 ``make_lead_agent`` 一致（14 个中间件）：

      0-2. Sandbox infrastructure (ThreadData → Uploads → Sandbox)
            沙箱基础设施（ThreadData → Uploads → Sandbox）
      3.   DanglingToolCallMiddleware (always) / 悬挂工具调用中间件（始终启用）
      4.   GuardrailMiddleware (guardrail feature) / 护栏中间件（guardrail 特性）
      5.   ToolErrorHandlingMiddleware (always) / 工具错误处理中间件（始终启用）
      6.   SummarizationMiddleware (summarization feature) / 摘要中间件（summarization 特性）
      7.   TodoMiddleware (plan_mode parameter) / 待办事项中间件（plan_mode 参数）
      8.   TitleMiddleware (auto_title feature) / 标题中间件（auto_title 特性）
      9.   MemoryMiddleware (memory feature) / 记忆中间件（memory 特性）
      10.  ViewImageMiddleware (vision feature) / 图像查看中间件（vision 特性）
      11.  SubagentLimitMiddleware (subagent feature) / 子代理限制中间件（subagent 特性）
      12.  LoopDetectionMiddleware (loop_detection feature) / 循环检测中间件（loop_detection 特性）
      13.  ClarificationMiddleware (always last) / 澄清中间件（始终在最后）

    Two-phase ordering / 两阶段排序：
      1. Built-in chain — fixed sequential append. / 内置链 — 固定顺序追加。
      2. Extra middleware — inserted via @Next/@Prev. / 额外中间件 — 通过 @Next/@Prev 插入。

    Each feature value is handled as / 每个特性值的处理规则：
      - ``False``: skip / 跳过
      - ``True``: create the built-in default middleware (not available for
        ``summarization`` and ``guardrail`` — these require a custom instance)
        / 创建内置默认中间件（``summarization`` 和 ``guardrail`` 不支持此选项
          —— 它们需要自定义实例）
      - ``AgentMiddleware`` instance: use directly (custom replacement)
        / 直接使用（自定义替换）
    """
    chain: list[AgentMiddleware] = []  # 中间件链
    extra_tools: list[BaseTool] = []  # 特性注入的额外工具

    # --- [0-2] Sandbox infrastructure / 沙箱基础设施 ---
    if feat.sandbox is not False:
        if isinstance(feat.sandbox, AgentMiddleware):
            # 用户提供了自定义沙箱中间件
            chain.append(feat.sandbox)
        else:
            # 使用内置默认沙箱中间件组：ThreadData → Uploads → Sandbox
            from deerflow.agents.middlewares.thread_data_middleware import ThreadDataMiddleware
            from deerflow.agents.middlewares.uploads_middleware import UploadsMiddleware
            from deerflow.sandbox.middleware import SandboxMiddleware

            chain.append(ThreadDataMiddleware(lazy_init=True))
            chain.append(UploadsMiddleware())
            chain.append(SandboxMiddleware(lazy_init=True))

    # --- [3] DanglingToolCall (always) / 悬挂工具调用中间件（始终启用） ---
    chain.append(DanglingToolCallMiddleware())

    # --- [4] Guardrail / 护栏中间件 ---
    if feat.guardrail is not False:
        if isinstance(feat.guardrail, AgentMiddleware):
            chain.append(feat.guardrail)
        else:
            raise ValueError("guardrail=True requires a custom AgentMiddleware instance (no built-in GuardrailMiddleware yet)")

    # --- [5] ToolErrorHandling (always) / 工具错误处理中间件（始终启用） ---
    chain.append(ToolErrorHandlingMiddleware())

    # --- [6] Summarization / 摘要中间件 ---
    if feat.summarization is not False:
        if isinstance(feat.summarization, AgentMiddleware):
            chain.append(feat.summarization)
        else:
            raise ValueError("summarization=True requires a custom AgentMiddleware instance (SummarizationMiddleware needs a model argument)")

    # --- [7] TodoMiddleware (plan_mode) / 待办事项中间件（计划模式） ---
    if plan_mode:
        from deerflow.agents.middlewares.todo_middleware import TodoMiddleware

        chain.append(TodoMiddleware(system_prompt=_TODO_SYSTEM_PROMPT, tool_description=_TODO_TOOL_DESCRIPTION))

    # --- [8] Auto Title / 自动标题中间件 ---
    if feat.auto_title is not False:
        if isinstance(feat.auto_title, AgentMiddleware):
            chain.append(feat.auto_title)
        else:
            from deerflow.agents.middlewares.title_middleware import TitleMiddleware

            chain.append(TitleMiddleware())

    # --- [9] Memory / 记忆中间件 ---
    if feat.memory is not False:
        if isinstance(feat.memory, AgentMiddleware):
            chain.append(feat.memory)
        else:
            from deerflow.agents.middlewares.memory_middleware import MemoryMiddleware

            chain.append(MemoryMiddleware(agent_name=name))

    # --- [10] Vision / 视觉中间件 ---
    if feat.vision is not False:
        if isinstance(feat.vision, AgentMiddleware):
            chain.append(feat.vision)
        else:
            from deerflow.agents.middlewares.view_image_middleware import ViewImageMiddleware

            chain.append(ViewImageMiddleware())

        # 视觉特性启用时，如果沙箱也启用，则追加 view_image 工具
        if feat.sandbox is not False:
            from deerflow.tools.builtins import view_image_tool

            extra_tools.append(view_image_tool)

    # --- [11] Subagent / 子代理中间件 ---
    if feat.subagent is not False:
        if isinstance(feat.subagent, AgentMiddleware):
            chain.append(feat.subagent)
        else:
            from deerflow.agents.middlewares.subagent_limit_middleware import SubagentLimitMiddleware

            chain.append(SubagentLimitMiddleware())
        # 子代理特性启用时，始终追加 task 工具
        from deerflow.tools.builtins import task_tool

        extra_tools.append(task_tool)

    # --- [12] LoopDetection / 循环检测中间件 ---
    if feat.loop_detection is not False:
        if isinstance(feat.loop_detection, AgentMiddleware):
            chain.append(feat.loop_detection)
        else:
            from deerflow.agents.middlewares.loop_detection_middleware import LoopDetectionMiddleware
            from deerflow.config.loop_detection_config import LoopDetectionConfig

            chain.append(LoopDetectionMiddleware.from_config(LoopDetectionConfig()))

    # --- [13] Clarification (always last among built-ins) / 澄清中间件（内置链中始终最后） ---
    chain.append(ClarificationMiddleware())
    extra_tools.append(ask_clarification_tool)  # 澄清工具始终追加

    # --- Insert extra_middleware via @Next/@Prev ---
    # --- 通过 @Next/@Prev 插入额外中间件 ---
    if extra_middleware:
        _insert_extra(chain, extra_middleware)
        # Invariant: ClarificationMiddleware must always be last.
        # @Next(ClarificationMiddleware) could push it off the tail.
        # 不变式：ClarificationMiddleware 必须始终在最后。
        # @Next(ClarificationMiddleware) 可能将其推离尾部。
        clar_idx = next(i for i, m in enumerate(chain) if isinstance(m, ClarificationMiddleware))
        if clar_idx != len(chain) - 1:
            # 将 ClarificationMiddleware 移回末尾
            chain.append(chain.pop(clar_idx))

    return chain, extra_tools


# ---------------------------------------------------------------------------
# Internal: extra middleware insertion with @Next/@Prev
# 内部：使用 @Next/@Prev 插入额外中间件
# ---------------------------------------------------------------------------


def _insert_extra(chain: list[AgentMiddleware], extras: list[AgentMiddleware]) -> None:
    """使用 ``@Next``/``@Prev`` 锚点将额外中间件插入 *chain* 中。

    Algorithm / 算法：
      1. Validate: no middleware has both @Next and @Prev.
         校验：没有中间件同时拥有 @Next 和 @Prev。
      2. Conflict detection: two extras targeting same anchor (same or opposite direction) → error.
         冲突检测：两个额外中间件指向同一锚点（相同或相反方向）→ 报错。
      3. Insert unanchored extras before ClarificationMiddleware.
         将无锚点的额外中间件插入到 ClarificationMiddleware 之前。
      4. Insert anchored extras iteratively (supports cross-external anchoring).
         迭代插入有锚点的额外中间件（支持跨外部锚点）。
      5. If an anchor cannot be resolved after all rounds → error.
         如果所有轮次后仍无法解析锚点 → 报错。
    """
    next_targets: dict[type, type] = {}  # @Next 锚点目标映射：锚点类型 → 中间件类型
    prev_targets: dict[type, type] = {}  # @Prev 锚点目标映射：锚点类型 → 中间件类型

    anchored: list[tuple[AgentMiddleware, str, type]] = []  # 有锚点的额外中间件：(中间件实例, 方向, 锚点类型)
    unanchored: list[AgentMiddleware] = []  # 无锚点的额外中间件

    for mw in extras:
        next_anchor = getattr(type(mw), "_next_anchor", None)  # 读取 @Next 装饰器设置的锚点
        prev_anchor = getattr(type(mw), "_prev_anchor", None)  # 读取 @Prev 装饰器设置的锚点

        if next_anchor and prev_anchor:
            raise ValueError(f"{type(mw).__name__} cannot have both @Next and @Prev")

        if next_anchor:
            # 冲突检测：同一锚点不能被两个 @Next 指向
            if next_anchor in next_targets:
                raise ValueError(f"Conflict: {type(mw).__name__} and {next_targets[next_anchor].__name__} both @Next({next_anchor.__name__})")
            # 冲突检测：同一锚点不能同时被 @Next 和 @Prev 指向
            if next_anchor in prev_targets:
                raise ValueError(f"Conflict: {type(mw).__name__} @Next({next_anchor.__name__}) and {prev_targets[next_anchor].__name__} @Prev({next_anchor.__name__}) — use cross-anchoring between extras instead")
            next_targets[next_anchor] = type(mw)
            anchored.append((mw, "next", next_anchor))
        elif prev_anchor:
            # 冲突检测：同一锚点不能被两个 @Prev 指向
            if prev_anchor in prev_targets:
                raise ValueError(f"Conflict: {type(mw).__name__} and {prev_targets[prev_anchor].__name__} both @Prev({prev_anchor.__name__})")
            # 冲突检测：同一锚点不能同时被 @Prev 和 @Next 指向
            if prev_anchor in next_targets:
                raise ValueError(f"Conflict: {type(mw).__name__} @Prev({prev_anchor.__name__}) and {next_targets[prev_anchor].__name__} @Next({prev_anchor.__name__}) — use cross-anchoring between extras instead")
            prev_targets[prev_anchor] = type(mw)
            anchored.append((mw, "prev", prev_anchor))
        else:
            # 无锚点的中间件
            unanchored.append(mw)

    # Unanchored → before ClarificationMiddleware
    # 无锚点的额外中间件插入到 ClarificationMiddleware 之前
    clarification_idx = next(i for i, m in enumerate(chain) if isinstance(m, ClarificationMiddleware))
    for mw in unanchored:
        chain.insert(clarification_idx, mw)
        clarification_idx += 1

    # Anchored → iterative insertion (supports external-to-external anchoring)
    # 有锚点的中间件 → 迭代插入（支持额外中间件之间的交叉锚定）
    pending = list(anchored)  # 待插入的有锚点中间件
    max_rounds = len(pending) + 1  # 最大迭代轮数
    for _ in range(max_rounds):
        if not pending:
            break
        remaining = []
        for mw, direction, anchor in pending:
            # 在链中查找锚点中间件的位置
            idx = next(
                (i for i, m in enumerate(chain) if isinstance(m, anchor)),
                None,
            )
            if idx is None:
                # 锚点未找到，可能是跨额外中间件锚定，留到下一轮处理
                remaining.append((mw, direction, anchor))
                continue
            if direction == "next":
                # @Next：插入到锚点之后
                chain.insert(idx + 1, mw)
            else:
                # @Prev：插入到锚点之前
                chain.insert(idx, mw)
        if len(remaining) == len(pending):
            # 没有进展，说明存在无法解析的锚点
            names = [type(m).__name__ for m, _, _ in remaining]
            anchor_types = {a for _, _, a in remaining}
            remaining_types = {type(m) for m, _, _ in remaining}
            circular = anchor_types & remaining_types  # 检测循环依赖
            if circular:
                raise ValueError(f"Circular dependency among extra middlewares: {', '.join(t.__name__ for t in circular)}")
            raise ValueError(f"Cannot resolve positions for {', '.join(names)} — anchors {', '.join(a.__name__ for _, _, a in remaining)} not found in chain")
        pending = remaining
