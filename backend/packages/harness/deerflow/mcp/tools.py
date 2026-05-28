"""使用 langchain-mcp-adapters 加载 MCP 工具。

本模块负责从所有启用的 MCP 服务器加载工具，核心功能包括：
- 动态加载 langchain-mcp-adapters 库
- 从配置文件读取 MCP 服务器配置
- 自动注入 OAuth 认证头（初始连接和工具调用）
- 支持自定义工具拦截器扩展
- 将异步工具包装为同步工具以兼容 DeerFlow 客户端

工作流程：
    1. 检查 langchain-mcp-adapters 是否已安装
    2. 从磁盘读取最新的扩展配置（确保跨进程配置更新可见）
    3. 构建服务器配置字典
    4. 为需要 OAuth 的服务器注入初始认证头
    5. 构建工具拦截器链（OAuth + 自定义拦截器）
    6. 创建 MultiServerMCPClient 并加载所有工具
    7. 将异步工具包装为同步工具
    8. 返回工具列表

架构说明：
    - ExtensionsConfig.from_file() 每次都从磁盘读取最新配置
    - 这与 get_extensions_config() 不同，后者使用缓存单例
    - 这种设计确保 Gateway API（独立进程）的配置变更能立即生效
"""

import logging

from langchain_core.tools import BaseTool

from deerflow.config.extensions_config import ExtensionsConfig
from deerflow.mcp.client import build_servers_config
from deerflow.mcp.oauth import build_oauth_tool_interceptor, get_initial_oauth_headers
from deerflow.reflection import resolve_variable
from deerflow.tools.sync import make_sync_tool_wrapper

logger = logging.getLogger(__name__)


async def get_mcp_tools() -> list[BaseTool]:
    """从所有启用的 MCP 服务器获取工具。
    
    这是加载 MCP 工具的主要入口函数，负责：
    - 检查依赖库是否安装
    - 读取最新的配置文件
    - 初始化 OAuth 认证
    - 加载自定义拦截器
    - 创建多服务器客户端并获取工具
    - 将异步工具转换为同步工具
    
    Returns:
        来自所有已启用 MCP 服务器的 LangChain 工具列表
        如果发生错误或没有配置服务器，返回空列表
        
    Note:
        - 此函数是异步的，需要在事件循环中调用
        - 每次调用都会从磁盘重新读取配置（不使用缓存）
        - 工具名称会自动添加服务器前缀（tool_name_prefix=True）
        - 异步工具会被自动包装为同步工具以兼容 DeerFlow
        
    Example:
        >>> # 在异步上下文中调用
        >>> tools = await get_mcp_tools()
        >>> print(f"Loaded {len(tools)} tools")
        >>> for tool in tools:
        ...     print(f"  - {tool.name}: {tool.description}")
    """
    try:
        # 动态导入 langchain-mcp-adapters
        from langchain_mcp_adapters.client import MultiServerMCPClient
    except ImportError:
        logger.warning("langchain-mcp-adapters not installed. Install it to enable MCP tools: pip install langchain-mcp-adapters")
        return []

    # 注意：我们使用 ExtensionsConfig.from_file() 而不是 get_extensions_config()
    # 以便始终从磁盘读取最新配置。这确保了通过 Gateway API（运行在独立进程中）
    # 所做的更改在初始化 MCP 工具时能够立即反映出来。
    extensions_config = ExtensionsConfig.from_file()
    servers_config = build_servers_config(extensions_config)

    if not servers_config:
        logger.info("No enabled MCP servers configured")
        return []

    try:
        # 创建多服务器 MCP 客户端
        logger.info(f"Initializing MCP client with {len(servers_config)} server(s)")

        # 为服务器连接注入初始 OAuth 头（用于工具发现/会话初始化）
        initial_oauth_headers = await get_initial_oauth_headers(extensions_config)
        for server_name, auth_header in initial_oauth_headers.items():
            if server_name not in servers_config:
                continue
            # 只为 SSE 和 HTTP 传输类型注入认证头
            if servers_config[server_name].get("transport") in ("sse", "http"):
                existing_headers = dict(servers_config[server_name].get("headers", {}))
                existing_headers["Authorization"] = auth_header
                servers_config[server_name]["headers"] = existing_headers

        # 构建工具拦截器链
        tool_interceptors = []
        
        # 1. 添加 OAuth 拦截器（用于工具调用时的认证）
        oauth_interceptor = build_oauth_tool_interceptor(extensions_config)
        if oauth_interceptor is not None:
            tool_interceptors.append(oauth_interceptor)

        # 2. 加载在 extensions_config.json 中声明的自定义拦截器
        # 格式："mcpInterceptors": ["pkg.module:builder_func", ...]
        raw_interceptor_paths = (extensions_config.model_extra or {}).get("mcpInterceptors")
        if isinstance(raw_interceptor_paths, str):
            # 如果配置的是单个字符串，转换为列表
            raw_interceptor_paths = [raw_interceptor_paths]
        elif not isinstance(raw_interceptor_paths, list):
            # 如果不是列表且不是 None，发出警告
            if raw_interceptor_paths is not None:
                logger.warning(f"mcpInterceptors must be a list of strings, got {type(raw_interceptor_paths).__name__}; skipping")
            raw_interceptor_paths = []
        
        # 遍历并加载每个自定义拦截器
        for interceptor_path in raw_interceptor_paths:
            try:
                # 使用 resolve_variable 动态导入构建器函数
                builder = resolve_variable(interceptor_path)
                # 调用构建器函数创建拦截器实例
                interceptor = builder()
                if callable(interceptor):
                    # 如果返回的是可调用对象，添加到拦截器链
                    tool_interceptors.append(interceptor)
                    logger.info(f"Loaded MCP interceptor: {interceptor_path}")
                elif interceptor is not None:
                    # 如果返回的不是可调用对象也不是 None，发出警告
                    logger.warning(f"Builder {interceptor_path} returned non-callable {type(interceptor).__name__}; skipping")
            except Exception as e:
                # 如果加载失败，记录警告但继续处理其他拦截器
                logger.warning(f"Failed to load MCP interceptor {interceptor_path}: {e}", exc_info=True)

        # 创建多服务器 MCP 客户端
        # tool_name_prefix=True 表示工具名称会自动添加服务器名前缀
        client = MultiServerMCPClient(servers_config, tool_interceptors=tool_interceptors, tool_name_prefix=True)

        # 从所有服务器获取工具
        tools = await client.get_tools()
        logger.info(f"Successfully loaded {len(tools)} tool(s) from MCP servers")

        # 修补工具以支持同步调用，因为 DeerFlow 客户端以同步方式流式传输
        # 检查工具是否有 coroutine 但没有 func，如果是则创建同步包装器
        for tool in tools:
            if getattr(tool, "func", None) is None and getattr(tool, "coroutine", None) is not None:
                # 使用 make_sync_tool_wrapper 将异步协程包装为同步函数
                tool.func = make_sync_tool_wrapper(tool.coroutine, tool.name)

        return tools

    except Exception as e:
        logger.error(f"Failed to load MCP tools: {e}", exc_info=True)
        return []
