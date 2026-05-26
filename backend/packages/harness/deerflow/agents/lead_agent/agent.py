"""Lead Agent 工厂模块 — DeerFlow 的主入口代理。

本模块负责创建和配置 DeerFlow 的主代理（Lead Agent），包括：
- 从运行时配置中解析模型、中间件、工具等参数
- 按严格顺序组装中间件链（18 个中间件组件）
- 根据配置动态启用/禁用可选功能（计划模式、子代理、摘要、循环检测等）
- 通过 LangGraph 的 create_agent 创建可运行的代理图

LangGraph 通过 langgraph.json 中注册的 make_lead_agent 入口调用本模块，
签名必须保持与 LangGraph Server 兼容。
"""

import logging

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware
from langchain_core.runnables import RunnableConfig

from deerflow.agents.lead_agent.prompt import apply_prompt_template
from deerflow.agents.memory.summarization_hook import memory_flush_hook
from deerflow.agents.middlewares.clarification_middleware import ClarificationMiddleware
from deerflow.agents.middlewares.loop_detection_middleware import LoopDetectionMiddleware
from deerflow.agents.middlewares.memory_middleware import MemoryMiddleware
from deerflow.agents.middlewares.subagent_limit_middleware import SubagentLimitMiddleware
from deerflow.agents.middlewares.summarization_middleware import BeforeSummarizationHook, DeerFlowSummarizationMiddleware
from deerflow.agents.middlewares.title_middleware import TitleMiddleware
from deerflow.agents.middlewares.todo_middleware import TodoMiddleware
from deerflow.agents.middlewares.token_usage_middleware import TokenUsageMiddleware
from deerflow.agents.middlewares.tool_error_handling_middleware import build_lead_runtime_middlewares
from deerflow.agents.middlewares.view_image_middleware import ViewImageMiddleware
from deerflow.agents.thread_state import ThreadState
from deerflow.config.agents_config import load_agent_config, validate_agent_name
from deerflow.config.app_config import AppConfig, get_app_config
from deerflow.models import create_chat_model
from deerflow.skills.tool_policy import filter_tools_by_skill_allowed_tools
from deerflow.skills.types import Skill

logger = logging.getLogger(__name__)


def _get_runtime_config(config: RunnableConfig) -> dict:
    """合并旧的 configurable 选项与 LangGraph 运行时上下文。

    LangGraph 的运行时配置分布在两个位置：
    - config["configurable"]: 传统的可配置参数（如 thread_id、model_name 等）
    - config["context"]: LangGraph 运行时注入的上下文信息

    本函数将两者合并为一个字典，context 中的值优先级更高（覆盖 configurable 中的同名键）。

    Args:
        config: LangGraph 的 RunnableConfig 对象

    Returns:
        合并后的配置字典
    """
    cfg = dict(config.get("configurable", {}) or {})
    context = config.get("context", {}) or {}
    if isinstance(context, dict):
        cfg.update(context)
    return cfg


def _resolve_model_name(requested_model_name: str | None = None, *, app_config: AppConfig | None = None) -> str:
    """安全地解析运行时模型名称，如果请求的模型无效则回退到默认模型。

    解析优先级：
    1. 请求的模型名称 — 如果在配置中存在则使用
    2. 配置中的第一个模型 — 作为默认回退
    3. 抛出异常 — 如果没有任何模型配置

    Args:
        requested_model_name: 运行时请求的模型名称，可能为 None
        app_config: 应用配置，为 None 时自动获取

    Returns:
        解析后的有效模型名称

    Raises:
        ValueError: 没有配置任何模型时抛出
    """
    app_config = app_config or get_app_config()
    default_model_name = app_config.models[0].name if app_config.models else None
    if default_model_name is None:
        raise ValueError("No chat models are configured. Please configure at least one model in config.yaml.")

    # 请求的模型在配置中存在，直接使用
    if requested_model_name and app_config.get_model_config(requested_model_name):
        return requested_model_name

    # 请求了模型但配置中找不到，回退到默认模型并记录警告
    if requested_model_name and requested_model_name != default_model_name:
        logger.warning(f"Model '{requested_model_name}' not found in config; fallback to default model '{default_model_name}'.")
    return default_model_name


def _create_summarization_middleware(*, app_config: AppConfig | None = None) -> DeerFlowSummarizationMiddleware | None:
    """根据应用配置创建并配置摘要中间件。

    当配置中启用摘要功能时，创建 DeerFlowSummarizationMiddleware 实例，
    配置触发条件、保留策略、摘要模型、技能救援参数等。

    如果启用了记忆功能，还会注册 memory_flush_hook 作为摘要前钩子，
    确保在摘要删除消息之前将对话内容刷新到记忆存储中。

    Args:
        app_config: 应用配置，为 None 时自动获取

    Returns:
        配置好的 DeerFlowSummarizationMiddleware 实例，未启用时返回 None
    """
    resolved_app_config = app_config or get_app_config()
    config = resolved_app_config.summarization

    if not config.enabled:
        return None

    # 准备触发参数：将 Pydantic 触发条件转换为元组格式
    trigger = None
    if config.trigger is not None:
        if isinstance(config.trigger, list):
            trigger = [t.to_tuple() for t in config.trigger]
        else:
            trigger = config.trigger.to_tuple()

    # 准备保留策略参数
    keep = config.keep.to_tuple()

    # 准备摘要模型。
    # 绑定 "middleware:summarize" 标签，使 RunJournal 将这些 LLM 调用
    # 归类为中间件而非 lead_agent（SummarizationMiddleware 是 LangChain 内置的，
    # 因此我们在创建模型时打标签）。
    if config.model_name:
        model = create_chat_model(name=config.model_name, thinking_enabled=False, app_config=resolved_app_config)
    else:
        model = create_chat_model(thinking_enabled=False, app_config=resolved_app_config)
    model = model.with_config(tags=["middleware:summarize"])

    # 组装中间件参数
    kwargs = {
        "model": model,
        "trigger": trigger,
        "keep": keep,
    }

    if config.trim_tokens_to_summarize is not None:
        kwargs["trim_tokens_to_summarize"] = config.trim_tokens_to_summarize

    if config.summary_prompt is not None:
        kwargs["summary_prompt"] = config.summary_prompt

    # 注册摘要前钩子：记忆刷新钩子确保在消息被摘要删除前持久化到记忆存储
    hooks: list[BeforeSummarizationHook] = []
    if resolved_app_config.memory.enabled:
        hooks.append(memory_flush_hook)

    # 技能容器路径：用于识别加载技能文件的工具调用，在摘要时优先保留
    skills_container_path = resolved_app_config.skills.container_path or "/mnt/skills"

    return DeerFlowSummarizationMiddleware(
        **kwargs,
        skills_container_path=skills_container_path,
        skill_file_read_tool_names=config.skill_file_read_tool_names,
        before_summarization=hooks,
        preserve_recent_skill_count=config.preserve_recent_skill_count,
        preserve_recent_skill_tokens=config.preserve_recent_skill_tokens,
        preserve_recent_skill_tokens_per_skill=config.preserve_recent_skill_tokens_per_skill,
    )


def _create_todo_list_middleware(is_plan_mode: bool) -> TodoMiddleware | None:
    """创建并配置待办事项中间件。

    仅在计划模式（plan_mode）启用时创建 TodoMiddleware 实例。
    自定义了系统提示和工具描述，匹配 DeerFlow 的风格。

    Args:
        is_plan_mode: 是否启用计划模式

    Returns:
        TodoMiddleware 实例（计划模式启用时），否则返回 None
    """
    if not is_plan_mode:
        return None

    # 自定义系统提示，匹配 DeerFlow 的风格
    system_prompt = """
<todo_list_system>
You have access to the `write_todos` tool to help you manage and track complex multi-step objectives.

**CRITICAL RULES:**
- Mark todos as completed IMMEDIATELY after finishing each step - do NOT batch completions
- Keep EXACTLY ONE task as `in_progress` at any time (unless tasks can run in parallel)
- Update the todo list in REAL-TIME as you work - this gives users visibility into your progress
- DO NOT use this tool for simple tasks (< 3 steps) - just complete them directly

**When to Use:**
This tool is designed for complex objectives that require systematic tracking:
- Complex multi-step tasks requiring 3+ distinct steps
- Non-trivial tasks needing careful planning and execution
- User explicitly requests a todo list
- User provides multiple tasks (numbered or comma-separated list)
- The plan may need revisions based on intermediate results

**When NOT to Use:**
- Single, straightforward tasks
- Trivial tasks (< 3 steps)
- Purely conversational or informational requests
- Simple tool calls where the approach is obvious

**Best Practices:**
- Break down complex tasks into smaller, actionable steps
- Use clear, descriptive task names
- Remove tasks that become irrelevant
- Add new tasks discovered during implementation
- Don't be afraid to revise the todo list as you learn more

**Task Management:**
Writing todos takes time and tokens - use it when helpful for managing complex problems, not for simple requests.
</todo_list_system>
"""

    tool_description = """Use this tool to create and manage a structured task list for complex work sessions.

**IMPORTANT: Only use this tool for complex tasks (3+ steps). For simple requests, just do the work directly.**

## When to Use

Use this tool in these scenarios:
1. **Complex multi-step tasks**: When a task requires 3 or more distinct steps or actions
2. **Non-trivial tasks**: Tasks requiring careful planning or multiple operations
3. **User explicitly requests todo list**: When the user directly asks you to track tasks
4. **Multiple tasks**: When users provide a list of things to be done
5. **Dynamic planning**: When the plan may need updates based on intermediate results

## When NOT to Use

Skip this tool when:
1. The task is straightforward and takes less than 3 steps
2. The task is trivial and tracking provides no benefit
3. The task is purely conversational or informational
4. It's clear what needs to be done and you can just do it

## How to Use

1. **Starting a task**: Mark it as `in_progress` BEFORE beginning work
2. **Completing a task**: Mark it as `completed` IMMEDIATELY after finishing
3. **Updating the list**: Add new tasks, remove irrelevant ones, or update descriptions as needed
4. **Multiple updates**: You can make several updates at once (e.g., complete one task and start the next)

## Task States

- `pending`: Task not yet started
- `in_progress`: Currently working on (can have multiple if tasks run in parallel)
- `completed`: Task finished successfully

## Task Completion Requirements

**CRITICAL: Only mark a task as completed when you have FULLY accomplished it.**

Never mark a task as completed if:
- There are unresolved issues or errors
- Work is partial or incomplete
- You encountered blockers preventing completion
- You couldn't find necessary resources or dependencies
- Quality standards haven't been met

If blocked, keep the task as `in_progress` and create a new task describing what needs to be resolved.

## Best Practices

- Create specific, actionable items
- Break complex tasks into smaller, manageable steps
- Use clear, descriptive task names
- Update task status in real-time as you work
- Mark tasks complete IMMEDIATELY after finishing (don't batch completions)
- Remove tasks that are no longer relevant
- **IMPORTANT**: When you write the todo list, mark your first task(s) as `in_progress` immediately
- **IMPORTANT**: Unless all tasks are completed, always have at least one task `in_progress` to show progress

Being proactive with task management demonstrates thoroughness and ensures all requirements are completed successfully.

**Remember**: If you only need a few tool calls to complete a task and it's clear what to do, it's better to just do the task directly and NOT use this tool at all.
"""

    return TodoMiddleware(system_prompt=system_prompt, tool_description=tool_description)


def _build_middlewares(
    config: RunnableConfig,
    model_name: str | None,
    agent_name: str | None = None,
    custom_middlewares: list[AgentMiddleware] | None = None,
    *,
    app_config: AppConfig | None = None,
):
    """根据运行时配置构建中间件链。

    中间件的顺序至关重要，每个中间件在特定位置有其原因：

    顺序依赖关系：
    - ThreadDataMiddleware 必须在 SandboxMiddleware 之前，确保 thread_id 可用
    - UploadsMiddleware 应在 ThreadDataMiddleware 之后，以访问 thread_id
    - DanglingToolCallMiddleware 在模型看到历史消息前修补缺失的 ToolMessage
    - SummarizationMiddleware 应较早执行，在其他处理之前减少上下文
    - TodoListMiddleware 应在 ClarificationMiddleware 之前，允许管理待办事项
    - TitleMiddleware 在首次对话后生成标题
    - MemoryMiddleware 在 TitleMiddleware 之后排队记忆更新
    - ViewImageMiddleware 应在 ClarificationMiddleware 之前注入图像详情
    - ClarificationMiddleware 必须在最后，以便在模型调用后拦截澄清请求

    完整中间件链（18 个组件）：
    1.  ThreadDataMiddleware        — 创建线程目录、解析路径
    2.  UploadsMiddleware           — 注入上传文件信息
    3.  SandboxMiddleware           — 获取沙箱、存储 sandbox_id
    4.  DanglingToolCallMiddleware  — 修补悬挂的工具调用
    5.  LLMErrorHandlingMiddleware  — LLM 错误重试与熔断
    6.  GuardrailMiddleware (可选)  — 工具调用授权拦截
    7.  SandboxAuditMiddleware      — 沙箱命令安全审计
    8.  ToolErrorHandlingMiddleware — 工具异常转错误 ToolMessage
    9.  DynamicContextMiddleware    — 注入记忆和当前日期
    10. SummarizationMiddleware (可选) — 上下文摘要压缩
    11. TodoListMiddleware (可选)   — 待办事项跟踪
    12. TokenUsageMiddleware (可选) — Token 用量记录
    13. TitleMiddleware             — 自动生成标题
    14. MemoryMiddleware            — 异步记忆更新排队
    15. ViewImageMiddleware (可选)  — 图像详情注入
    16. DeferredToolFilterMiddleware (可选) — 延迟工具过滤
    17. SubagentLimitMiddleware (可选) — 子代理并发限制
    18. LoopDetectionMiddleware (可选) — 循环检测
    19. CustomMiddlewares (可选)    — 自定义中间件
    20. ClarificationMiddleware     — 澄清请求拦截（必须最后）

    Args:
        config: 运行时配置，包含 is_plan_mode 等可配置选项
        model_name: 解析后的模型名称，用于判断是否支持视觉等
        agent_name: 代理名称，如果提供则使用按代理隔离的记忆存储
        custom_middlewares: 可选的自定义中间件列表，插入到 ClarificationMiddleware 之前
        app_config: 应用配置

    Returns:
        中间件实例列表
    """
    resolved_app_config = app_config or get_app_config()

    # 构建基础运行时中间件（前 8 个：ThreadData → Uploads → Sandbox →
    # DanglingToolCall → LLMError → Guardrail → SandboxAudit → ToolError）
    middlewares = build_lead_runtime_middlewares(app_config=resolved_app_config, lazy_init=True)

    # 9. 注入当前日期（以及可选的记忆）作为 <system-reminder>
    #    注入到第一个 HumanMessage 中，保持系统提示完全静态以最大化前缀缓存复用
    from deerflow.agents.middlewares.dynamic_context_middleware import DynamicContextMiddleware

    middlewares.append(DynamicContextMiddleware(agent_name=agent_name, app_config=resolved_app_config))

    # 10. 添加摘要中间件（如果启用）
    summarization_middleware = _create_summarization_middleware(app_config=resolved_app_config)
    if summarization_middleware is not None:
        middlewares.append(summarization_middleware)

    # 11. 添加待办事项中间件（计划模式启用时）
    cfg = _get_runtime_config(config)
    is_plan_mode = cfg.get("is_plan_mode", False)
    todo_list_middleware = _create_todo_list_middleware(is_plan_mode)
    if todo_list_middleware is not None:
        middlewares.append(todo_list_middleware)

    # 12. 添加 Token 用量中间件（token_usage 跟踪启用时）
    if resolved_app_config.token_usage.enabled:
        middlewares.append(TokenUsageMiddleware())

    # 13. 添加标题中间件
    middlewares.append(TitleMiddleware(app_config=resolved_app_config))

    # 14. 添加记忆中间件（在标题中间件之后，确保标题生成完成后再排队记忆更新）
    middlewares.append(MemoryMiddleware(agent_name=agent_name, memory_config=resolved_app_config.memory))

    # 15. 添加图像查看中间件（仅当当前模型支持视觉时）
    #    使用 make_lead_agent 中解析的运行时 model_name，避免使用过时的配置值
    model_config = resolved_app_config.get_model_config(model_name) if model_name else None
    if model_config is not None and model_config.supports_vision:
        middlewares.append(ViewImageMiddleware())

    # 16. 添加延迟工具过滤中间件，在模型绑定时隐藏延迟工具的 schema
    if resolved_app_config.tool_search.enabled:
        from deerflow.agents.middlewares.deferred_tool_filter_middleware import DeferredToolFilterMiddleware

        middlewares.append(DeferredToolFilterMiddleware())

    # 17. 添加子代理限制中间件，截断超出的并行 task 工具调用
    subagent_enabled = cfg.get("subagent_enabled", False)
    if subagent_enabled:
        max_concurrent_subagents = cfg.get("max_concurrent_subagents", 3)
        middlewares.append(SubagentLimitMiddleware(max_concurrent=max_concurrent_subagents))

    # 18. 循环检测中间件 — 检测并打断重复的工具调用循环
    loop_detection_config = resolved_app_config.loop_detection
    if loop_detection_config.enabled:
        middlewares.append(LoopDetectionMiddleware.from_config(loop_detection_config))

    # 19. 注入自定义中间件（在 ClarificationMiddleware 之前）
    if custom_middlewares:
        middlewares.extend(custom_middlewares)

    # 20. ClarificationMiddleware 必须始终在最后
    middlewares.append(ClarificationMiddleware())
    return middlewares


def _available_skill_names(agent_config, is_bootstrap: bool) -> set[str] | None:
    """获取当前代理可用的技能名称集合。

    Args:
        agent_config: 代理配置对象，可能包含 skills 列表
        is_bootstrap: 是否为引导模式

    Returns:
        可用技能名称集合，None 表示不限制（所有技能可用）
    """
    if is_bootstrap:
        # 引导模式只允许 bootstrap 技能
        return {"bootstrap"}
    if agent_config and agent_config.skills is not None:
        # 代理配置中指定了技能列表，只使用这些技能
        return set(agent_config.skills)
    # 未指定技能列表，不限制
    return None


def _load_enabled_skills_for_tool_policy(available_skills: set[str] | None, *, app_config: AppConfig) -> list[Skill]:
    """加载已启用的技能列表，用于工具策略过滤。

    根据可用技能名称集合过滤全局已启用技能，只保留名称匹配的技能。
    如果 available_skills 为 None，则返回所有已启用技能（不限制）。

    Args:
        available_skills: 允许的技能名称集合，None 表示不限制
        app_config: 应用配置

    Returns:
        过滤后的技能列表
    """
    try:
        from deerflow.agents.lead_agent.prompt import get_enabled_skills_for_config

        skills = get_enabled_skills_for_config(app_config)
    except Exception:
        logger.exception("Failed to load skills for allowed-tools policy")
        raise

    if available_skills is None:
        # 不限制，返回所有已启用技能
        return skills
    # 只保留名称在 available_skills 中的技能
    return [skill for skill in skills if skill.name in available_skills]


def make_lead_agent(config: RunnableConfig):
    """LangGraph 图工厂入口；保持签名与 LangGraph Server 兼容。

    此函数是 langgraph.json 中注册的入口点，LangGraph Server 通过它
    创建代理图实例。从运行时配置中提取 app_config，然后委托给
    _make_lead_agent 执行实际的代理创建逻辑。

    Args:
        config: LangGraph 的 RunnableConfig，包含 configurable 和 context

    Returns:
        LangGraph 代理图实例
    """
    runtime_config = _get_runtime_config(config)
    runtime_app_config = runtime_config.get("app_config")
    return _make_lead_agent(config, app_config=runtime_app_config or get_app_config())


def _make_lead_agent(config: RunnableConfig, *, app_config: AppConfig):
    """实际的 Lead Agent 创建逻辑。

    执行流程：
    1. 解析运行时配置参数（模型、思考模式、计划模式、子代理等）
    2. 加载代理配置（如果是自定义代理）
    3. 解析最终模型名称（请求 → 代理配置 → 全局默认，带回退）
    4. 注入 LangSmith 追踪元数据
    5. 加载工具并根据技能策略过滤
    6. 构建中间件链
    7. 应用系统提示模板
    8. 创建代理图

    Args:
        config: LangGraph 的 RunnableConfig
        app_config: 应用配置

    Returns:
        LangGraph 代理图实例

    Raises:
        ValueError: 无法解析任何可用模型时抛出
    """
    # 延迟导入以避免循环依赖
    from deerflow.tools import get_available_tools
    from deerflow.tools.builtins import setup_agent, update_agent

    cfg = _get_runtime_config(config)
    resolved_app_config = app_config

    # ===== 第一步：解析运行时配置参数 =====
    thinking_enabled = cfg.get("thinking_enabled", True)  # 是否启用扩展思考
    reasoning_effort = cfg.get("reasoning_effort", None)  # 推理努力程度
    requested_model_name: str | None = cfg.get("model_name") or cfg.get("model")  # 请求的模型名称
    is_plan_mode = cfg.get("is_plan_mode", False)  # 是否启用计划模式
    subagent_enabled = cfg.get("subagent_enabled", False)  # 是否启用子代理
    max_concurrent_subagents = cfg.get("max_concurrent_subagents", 3)  # 最大并发子代理数
    is_bootstrap = cfg.get("is_bootstrap", False)  # 是否为引导模式
    agent_name = validate_agent_name(cfg.get("agent_name"))  # 代理名称（经过验证）

    # ===== 第二步：加载代理配置 =====
    agent_config = load_agent_config(agent_name) if not is_bootstrap else None
    available_skills = _available_skill_names(agent_config, is_bootstrap)
    # 自定义代理的模型配置（如有），None 则使用 _resolve_model_name 选择的默认模型
    agent_model_name = agent_config.model if agent_config and agent_config.model else None

    # ===== 第三步：最终模型名称解析 =====
    # 优先级：请求参数 → 代理配置 → 全局默认，对未知名称提供回退
    model_name = _resolve_model_name(requested_model_name or agent_model_name, app_config=resolved_app_config)

    model_config = resolved_app_config.get_model_config(model_name)

    if model_config is None:
        raise ValueError("No chat model could be resolved. Please configure at least one model in config.yaml or provide a valid 'model_name'/'model' in the request.")

    # 如果启用了思考模式但模型不支持，回退到非思考模式
    if thinking_enabled and not model_config.supports_thinking:
        logger.warning(f"Thinking mode is enabled but model '{model_name}' does not support it; fallback to non-thinking mode.")
        thinking_enabled = False

    logger.info(
        "Create Agent(%s) -> thinking_enabled: %s, reasoning_effort: %s, model_name: %s, is_plan_mode: %s, subagent_enabled: %s, max_concurrent_subagents: %s",
        agent_name or "default",
        thinking_enabled,
        reasoning_effort,
        model_name,
        is_plan_mode,
        subagent_enabled,
        max_concurrent_subagents,
    )

    # ===== 第四步：注入 LangSmith 追踪元数据 =====
    if "metadata" not in config:
        config["metadata"] = {}

    config["metadata"].update(
        {
            "agent_name": agent_name or "default",
            "model_name": model_name or "default",
            "thinking_enabled": thinking_enabled,
            "reasoning_effort": reasoning_effort,
            "is_plan_mode": is_plan_mode,
            "subagent_enabled": subagent_enabled,
            "tool_groups": agent_config.tool_groups if agent_config else None,
            "available_skills": sorted(available_skills) if available_skills is not None else None,
        }
    )

    # ===== 第五步：加载技能并过滤工具 =====
    skills_for_tool_policy = _load_enabled_skills_for_tool_policy(available_skills, app_config=resolved_app_config)

    # ===== 第六步：创建代理 =====
    if is_bootstrap:
        # 引导模式：使用精简提示的特殊代理，用于初始自定义代理创建流程
        # 额外提供 setup_agent 工具，允许用户创建新的自定义代理
        tools = get_available_tools(model_name=model_name, subagent_enabled=subagent_enabled, app_config=resolved_app_config) + [setup_agent]
        return create_agent(
            model=create_chat_model(name=model_name, thinking_enabled=thinking_enabled, app_config=resolved_app_config),
            tools=filter_tools_by_skill_allowed_tools(tools, skills_for_tool_policy),
            middleware=_build_middlewares(config, model_name=model_name, app_config=resolved_app_config),
            system_prompt=apply_prompt_template(
                subagent_enabled=subagent_enabled,
                max_concurrent_subagents=max_concurrent_subagents,
                available_skills=set(["bootstrap"]),
                app_config=resolved_app_config,
            ),
            state_schema=ThreadState,
        )

    # 自定义代理可以通过 update_agent 工具更新自身的 SOUL.md / config.yaml
    # 默认代理（无 agent_name）不会看到此工具
    extra_tools = [update_agent] if agent_name else []

    # 标准代理创建流程
    tools = get_available_tools(model_name=model_name, groups=agent_config.tool_groups if agent_config else None, subagent_enabled=subagent_enabled, app_config=resolved_app_config)

    return create_agent(
        model=create_chat_model(name=model_name, thinking_enabled=thinking_enabled, reasoning_effort=reasoning_effort, app_config=resolved_app_config),
        tools=filter_tools_by_skill_allowed_tools(tools + extra_tools, skills_for_tool_policy),
        middleware=_build_middlewares(config, model_name=model_name, agent_name=agent_name, app_config=resolved_app_config),
        system_prompt=apply_prompt_template(
            subagent_enabled=subagent_enabled,
            max_concurrent_subagents=max_concurrent_subagents,
            agent_name=agent_name,
            available_skills=set(agent_config.skills) if agent_config and agent_config.skills is not None else None,
            app_config=resolved_app_config,
        ),
        state_schema=ThreadState,
    )
