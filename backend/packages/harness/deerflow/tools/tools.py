"""DeerFlow 工具管理系统。

本模块负责加载、配置和管理 DeerFlow 中可用的所有工具，包括：
- 内置工具（如文件展示、澄清提问等）
- 子代理工具（任务管理）
- MCP（Model Context Protocol）工具
- ACP（Agent Communication Protocol）工具
- 自定义配置工具

工具去重和名称冲突处理遵循 issue #1803 的修复方案。
"""

import logging

from langchain.tools import BaseTool

from deerflow.config import get_app_config
from deerflow.config.app_config import AppConfig
from deerflow.reflection import resolve_variable
from deerflow.sandbox.security import is_host_bash_allowed
from deerflow.tools.builtins import ask_clarification_tool, present_file_tool, task_tool, view_image_tool
from deerflow.tools.builtins.tool_search import get_deferred_registry
from deerflow.tools.sync import make_sync_tool_wrapper

logger = logging.getLogger(__name__)

# 内置工具列表：始终可用的基础工具
# - present_file_tool: 用于向用户展示文件内容
# - ask_clarification_tool: 用于在信息不足时向用户提问澄清
BUILTIN_TOOLS = [
    present_file_tool,
    ask_clarification_tool,
]

# 子代理工具列表：仅在启用子代理功能时添加
# - task_tool: 用于创建和管理子任务
# 注意：task_status_tool 不再暴露给 LLM，后端内部处理轮询逻辑
SUBAGENT_TOOLS = [
    task_tool,
    # task_status_tool is no longer exposed to LLM (backend handles polling internally)
]


def _is_host_bash_tool(tool: object) -> bool:
    """判断工具是否为主机 bash 执行工具。

    通过检查工具的 group 或 use 属性来识别 bash 工具。

    Args:
        tool: 要检查的工具对象

    Returns:
        True 如果该工具是主机 bash 执行工具，否则 False
    """
    group = getattr(tool, "group", None)
    use = getattr(tool, "use", None)
    if group == "bash":
        return True
    if use == "deerflow.sandbox.tools:bash_tool":
        return True
    return False


def _ensure_sync_invocable_tool(tool: BaseTool) -> BaseTool:
    """为纯异步工具添加同步调用包装器，以支持同步代理调用者。

    某些工具只实现了异步方法（coroutine），但同步代码路径需要调用它们。
    此函数检测这种情况并为工具附加一个同步包装器。

    Args:
        tool: 需要确保可同步调用的工具对象

    Returns:
        已添加同步包装器的工具对象（如果原本就支持同步调用则原样返回）
    """
    if getattr(tool, "func", None) is None and getattr(tool, "coroutine", None) is not None:
        tool.func = make_sync_tool_wrapper(tool.coroutine, tool.name)
    return tool


def get_available_tools(
    groups: list[str] | None = None,
    include_mcp: bool = True,
    model_name: str | None = None,
    subagent_enabled: bool = False,
    *,
    app_config: AppConfig | None = None,
) -> list[BaseTool]:
    """获取所有可用的工具列表。

    本函数根据配置加载并组装完整的工具集，包括配置工具、内置工具、
    MCP 工具和 ACP 工具。同时处理工具去重、名称验证和安全过滤。

    注意：MCP 工具应在应用启动时使用 `initialize_mcp_tools()` 进行初始化，
    本函数从缓存中读取已初始化的 MCP 工具。

    Args:
        groups: 可选的工具组过滤器，仅返回指定组的工具（如 ["search", "code"]）
        include_mcp: 是否包含 MCP 服务器提供的工具（默认：True）
        model_name: 可选的模型名称，用于确定是否包含视觉相关工具
        subagent_enabled: 是否启用子代理工具（task 等）
        app_config: 可选的应用配置对象，未提供时使用全局配置

    Returns:
        可用工具列表（已去重，按优先级排序）

    Raises:
        ImportError: 当 MCP 模块不可用时发出警告但不中断执行
        Exception: MCP 工具加载失败时记录错误但继续执行
    """
    # 获取应用配置（使用传入的配置或全局配置）
    config = app_config or get_app_config()
    # 根据工具组过滤器筛选配置中的工具
    tool_configs = [tool for tool in config.tools if groups is None or tool.group in groups]

    # 安全检查：当 LocalSandboxProvider 激活时，默认不暴露主机 bash 工具
    # 这是为了防止 AI 代理直接在主机上执行命令，提高安全性
    if not is_host_bash_allowed(config):
        tool_configs = [tool for tool in tool_configs if not _is_host_bash_tool(tool)]

    # 解析并加载所有配置的工具实例
    # resolve_variable 通过反射机制从配置字符串（如 "module.path:ClassName"）实例化工具对象
    loaded_tools_raw = [(cfg, resolve_variable(cfg.use, BaseTool)) for cfg in tool_configs]

    # 验证工具名称一致性：检查配置中的 name 字段与工具对象的 .name 属性是否匹配
    # 名称不一致是导致 issue #1803 的根本原因：LLM 接收到的工具schema名称与运行时
    # 路由器识别的名称不同，导致 "not a valid tool" 错误
    for cfg, loaded in loaded_tools_raw:
        if cfg.name != loaded.name:
            logger.warning(
                "Tool name mismatch: config name %r does not match tool .name %r (use: %s). The tool's own .name will be used for binding.",
                cfg.name,
                loaded.name,
                cfg.use,
            )

    # 为所有加载的工具确保同步调用能力（必要时添加包装器）
    loaded_tools = [_ensure_sync_invocable_tool(t) for _, t in loaded_tools_raw]

    # 根据配置条件性添加工具
    builtin_tools = BUILTIN_TOOLS.copy()
    # 检查是否启用了技能演进功能
    skill_evolution_config = getattr(config, "skill_evolution", None)
    if getattr(skill_evolution_config, "enabled", False):
        # 启用技能管理工具，允许动态添加/删除技能
        from deerflow.tools.skill_manage_tool import skill_manage_tool

        builtin_tools.append(skill_manage_tool)

    # 仅在运行时参数启用子代理时添加子代理工具
    if subagent_enabled:
        builtin_tools.extend(SUBAGENT_TOOLS)
        logger.info("Including subagent tools (task)")

    # 如果未指定模型名称，使用配置中的第一个模型作为默认值
    if model_name is None and config.models:
        model_name = config.models[0].name

    # 仅在模型支持视觉功能时添加 view_image_tool
    # 这可以避免不支持视觉的模型接收到无法处理的工具
    model_config = config.get_model_config(model_name) if model_name else None
    if model_config is not None and model_config.supports_vision:
        builtin_tools.append(view_image_tool)
        logger.info(f"Including view_image_tool for model '{model_name}' (supports_vision=True)")

    # 获取缓存的 MCP 工具（如果启用）
    # 重要：使用 ExtensionsConfig.from_file() 而非 config.extensions，
    # 以确保始终从磁盘读取最新配置。这是因为 Gateway API（运行在独立进程中）
    # 对配置的修改需要立即反映在 MCP 工具加载中。
    mcp_tools = []
    if include_mcp:
        try:
            from deerflow.config.extensions_config import ExtensionsConfig
            from deerflow.mcp.cache import get_cached_mcp_tools

            # 从配置文件加载扩展配置
            extensions_config = ExtensionsConfig.from_file()
            if extensions_config.get_enabled_mcp_servers():
                # 从缓存中获取已初始化的 MCP 工具
                mcp_tools = get_cached_mcp_tools()
                if mcp_tools:
                    logger.info(f"Using {len(mcp_tools)} cached MCP tool(s)")

                    # 当启用工具搜索功能时，将 MCP 工具注册到延迟注册表中，并添加 tool_search 到内置工具列表
                    if config.tool_search.enabled:
                        from deerflow.tools.builtins.tool_search import DeferredToolRegistry, set_deferred_registry
                        from deerflow.tools.builtins.tool_search import tool_search as tool_search_tool

                        # 检查当前异步上下文中是否已存在注册表
                        # get_available_tools 会在生成子代理时被重新调用（task_tool 调用它来构建子代理的工具集），
                        # 之前我们无条件地重建注册表——这会清除父代理的 tool_search 提升记录。
                        # DeferredToolFilterMiddleware 随后会将这些工具重新隐藏，导致代理能看到工具名但无法调用
                        # （issue #2884）。contextvars 已经提供了我们需要的生命周期语义：新的请求/图运行
                        # 在新的 asyncio 任务中启动，ContextVar 处于默认值 None，因此重用仅在同一运行内的
                        # 重入调用时触发。
                        #
                        # 故意不与当前 mcp_tools 快照进行协调。MCP 缓存仅在 extensions_config.json 的
                        # mtime 变化时刷新，这实际上发生在图运行之间，而不是运行内部。即使发生了刷新，
                        # 已构建的主代理的 ToolNode 仍然持有之前的工具集（LangGraph 在图构建时绑定工具），
                        # 所以全新的 MCP 工具无论如何也无法被调用。DeferredToolRegistry 不保留之前提升的
                        # 工具名称（promote() 会完全删除条目），所以将注册表与新的 mcp_tools 列表重新同步
                        # 会将那些提升误分类为新工具并重新注册为延迟——这正是此修复要防止的错误。
                        existing_registry = get_deferred_registry()
                        if existing_registry is None:
                            # 创建新注册表并注册所有 MCP 工具
                            registry = DeferredToolRegistry()
                            for t in mcp_tools:
                                registry.register(t)
                            set_deferred_registry(registry)
                            logger.info(f"Tool search active: {len(mcp_tools)} tools deferred")
                        else:
                            # 复用现有注册表，保持已提升的工具状态
                            mcp_tool_names = {t.name for t in mcp_tools}
                            still_deferred = len(existing_registry)
                            promoted_count = max(0, len(mcp_tool_names) - still_deferred)
                            logger.info(f"Tool search active (preserved promotions): {still_deferred} tools deferred, {promoted_count} already promoted")
                        builtin_tools.append(tool_search_tool)
        except ImportError:
            # MCP 适配器包未安装时发出警告，但不中断执行
            logger.warning("MCP module not available. Install 'langchain-mcp-adapters' package to enable MCP tools.")
        except Exception as e:
            # 其他异常（如配置错误、网络问题等）记录错误但继续执行
            logger.error(f"Failed to get cached MCP tools: {e}")

    # 如果配置了任何 ACP 代理，添加 invoke_acp_agent 工具
    # ACP（Agent Communication Protocol）用于代理间通信
    acp_tools: list[BaseTool] = []
    try:
        from deerflow.tools.builtins.invoke_acp_agent_tool import build_invoke_acp_agent_tool

        # 获取 ACP 代理配置（优先使用传入的配置，否则从全局配置读取）
        if app_config is None:
            from deerflow.config.acp_config import get_acp_agents

            acp_agents = get_acp_agents()
        else:
            acp_agents = getattr(config, "acp_agents", {}) or {}
        if acp_agents:
            # 构建 ACP 代理调用工具，支持调用多个配置的代理
            acp_tools.append(build_invoke_acp_agent_tool(acp_agents))
            logger.info(f"Including invoke_acp_agent tool ({len(acp_agents)} agent(s): {list(acp_agents.keys())})")
    except Exception as e:
        # ACP 工具加载失败时发出警告，但不影响其他工具的正常使用
        logger.warning(f"Failed to load ACP tool: {e}")

    # 记录工具加载统计信息
    logger.info(f"Total tools loaded: {len(loaded_tools)}, built-in tools: {len(builtin_tools)}, MCP tools: {len(mcp_tools)}, ACP tools: {len(acp_tools)}")

    # 按工具名称去重 —— 配置加载的工具优先级最高，其次是内置工具、MCP 工具和 ACP 工具
    # 重复的工具名称会导致 LLM 收到模糊或拼接的函数 schema（issue #1803）
    # 去重策略：保留首次出现的工具，跳过后续同名工具
    all_tools = loaded_tools + builtin_tools + mcp_tools + acp_tools
    seen_names: set[str] = set()
    unique_tools: list[BaseTool] = []
    for t in all_tools:
        if t.name not in seen_names:
            unique_tools.append(t)
            seen_names.add(t.name)
        else:
            # 发现重复工具名称时发出警告，提示检查配置和 MCP 服务器注册
            logger.warning(
                "Duplicate tool name %r detected and skipped — check your config.yaml and MCP server registrations (issue #1803).",
                t.name,
            )
    return unique_tools
