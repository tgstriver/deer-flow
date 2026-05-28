"""MCP 工具缓存，避免重复加载。

本模块提供 MCP 工具的缓存机制，核心功能包括：
- 懒加载初始化：首次使用时自动加载工具
- 基于配置文件修改时间的缓存失效：配置文件变更时自动重新加载
- 线程安全的事件循环兼容处理：支持 FastAPI 和 LangGraph Studio 等不同上下文
- 异步锁防止并发初始化竞争

设计目标：
    - 避免每次请求都重新加载 MCP 工具，提高性能
    - 检测配置文件变化，确保配置更新后能及时生效
    - 在不同事件循环环境下都能正常工作（同步/异步、有/无运行中的循环）

使用场景：
    - FastAPI 网关进程：应用启动时调用 initialize_mcp_tools()
    - LangGraph Server 进程：通过 get_cached_mcp_tools() 懒加载
    - 跨进程配置更新：Gateway API 修改配置后，LangGraph Server 能检测到并重新加载
"""

import asyncio
import logging
import os

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

# 全局 MCP 工具缓存，None 表示尚未初始化
# 类型：list[BaseTool] | None
_mcp_tools_cache: list[BaseTool] | None = None

# 缓存是否已初始化的标志
# True 表示已完成初始化（即使加载失败也算初始化过）
_cache_initialized = False

# 初始化锁，防止并发初始化竞争
# 使用 asyncio.Lock 确保同一时间只有一个协程在执行初始化
_initialization_lock = asyncio.Lock()

# 配置文件的最后修改时间，用于缓存失效检测
# 记录缓存初始化时的配置文件 mtime，后续对比判断是否过期
_config_mtime: float | None = None


def _get_config_mtime() -> float | None:
    """获取扩展配置文件的最后修改时间。
    
    通过 ExtensionsConfig.resolve_config_path() 解析配置文件路径，
    然后使用 os.path.getmtime() 获取文件的修改时间戳。
    
    Returns:
        以浮点数返回修改时间（Unix 时间戳），若文件不存在则返回 None
        
    Note:
        - 修改时间用于检测配置文件是否被修改
        - 如果配置文件不存在，返回 None，表示无法进行失效检测
        - 此函数在每次检查缓存状态时都会被调用
        
    Example:
        >>> mtime = _get_config_mtime()
        >>> if mtime:
        ...     print(f"Config last modified at: {mtime}")
    """
    from deerflow.config.extensions_config import ExtensionsConfig

    config_path = ExtensionsConfig.resolve_config_path()
    if config_path and config_path.exists():
        return os.path.getmtime(config_path)
    return None


def _is_cache_stale() -> bool:
    """检查缓存是否因配置文件变更而过期。
    
    通过对比当前配置文件修改时间与缓存时记录的修改时间来判断。
    如果配置文件在缓存后被修改，则认为缓存过期，需要重新加载。
    
    Returns:
        若缓存应失效返回 True，否则返回 False
        
    Note:
        - 如果缓存尚未初始化（_cache_initialized=False），返回 False
        - 如果无法获取修改时间（之前或现在为 None），返回 False（保守策略）
        - 只有当 current_mtime > _config_mtime 时才认为过期
        - 这种设计避免了因文件系统问题导致的误判
        
    Example:
        >>> # 假设缓存时 mtime=1000.0
        >>> # 现在配置文件被修改，mtime=1005.0
        >>> if _is_cache_stale():
        ...     reset_mcp_tools_cache()  # 重置缓存
        ...     await initialize_mcp_tools()  # 重新初始化
    """
    global _config_mtime

    if not _cache_initialized:
        return False  # 尚未初始化，不算过期

    current_mtime = _get_config_mtime()

    # 若之前或现在无法获取修改时间，假定缓存未过期
    # 这是一种保守策略，避免因文件系统问题导致频繁重载
    if _config_mtime is None or current_mtime is None:
        return False

    # 若配置文件在缓存后已被修改，则缓存过期
    if current_mtime > _config_mtime:
        logger.info(f"MCP config file has been modified (mtime: {_config_mtime} -> {current_mtime}), cache is stale")
        return True

    return False


async def initialize_mcp_tools() -> list[BaseTool]:
    """初始化并缓存 MCP 工具。
    
    此方法应在应用启动时调用一次，从所有启用的 MCP 服务器加载工具并缓存。
    使用异步锁确保并发安全，防止多个协程同时初始化。
    
    工作流程：
        1. 获取初始化锁（防止并发）
        2. 检查是否已初始化，如果是则直接返回缓存
        3. 调用 get_mcp_tools() 从配置文件加载工具
        4. 缓存工具列表和配置文件修改时间
        5. 释放锁并返回工具列表
    
    Returns:
        来自所有已启用 MCP 服务器的 LangChain 工具列表
        
    Note:
        - 使用 _initialization_lock 确保线程安全
        - 记录配置文件的 mtime，用于后续的失效检测
        - 如果已经初始化过，直接返回缓存而不重新加载
        - 日志会记录加载的工具数量和配置文件修改时间
        
    Example:
        >>> # 在 FastAPI 启动钩子中调用
        >>> @app.on_event("startup")
        >>> async def startup():
        ...     tools = await initialize_mcp_tools()
        ...     logger.info(f"Loaded {len(tools)} MCP tools")
    """
    global _mcp_tools_cache, _cache_initialized, _config_mtime

    async with _initialization_lock:
        # 双重检查：如果已初始化，直接返回缓存
        if _cache_initialized:
            logger.info("MCP tools already initialized")
            return _mcp_tools_cache or []

        from deerflow.mcp.tools import get_mcp_tools

        logger.info("Initializing MCP tools...")
        # 从配置文件加载所有启用的 MCP 服务器的工具
        _mcp_tools_cache = await get_mcp_tools()
        _cache_initialized = True
        _config_mtime = _get_config_mtime()  # 记录配置文件修改时间
        logger.info(f"MCP tools initialized: {len(_mcp_tools_cache)} tool(s) loaded (config mtime: {_config_mtime})")

        return _mcp_tools_cache


def get_cached_mcp_tools() -> list[BaseTool]:
    """获取缓存的 MCP 工具，支持懒加载初始化。
    
    如果工具尚未初始化，会自动进行初始化。
    这确保了 MCP 工具在 FastAPI 和 LangGraph Studio 等不同上下文中都能正常工作。
    
    同时检查配置文件自上次初始化以来是否被修改，
    如果已修改则重新初始化。这确保了通过 Gateway API（运行在独立进程中）
    所做的更改能够反映在 LangGraph Server 中。
    
    工作流程：
        1. 检查缓存是否因配置文件变更而过期
        2. 如果过期，重置缓存
        3. 如果未初始化，执行懒加载初始化
        4. 根据事件循环状态选择合适的初始化方式
        5. 返回缓存的工具列表
    
    Returns:
        缓存的 MCP 工具列表，如果初始化失败则返回空列表
        
    Note:
        - 支持三种事件循环场景：
          a) 有运行中的事件循环（如 LangGraph Studio）：使用 ThreadPoolExecutor 在新线程中运行
          b) 有未运行的事件循环：使用 run_until_complete
          c) 没有事件循环：使用 asyncio.run 创建新循环
        - 异常处理：任何初始化失败都会记录日志并返回空列表
        - 跨进程配置同步：通过配置文件 mtime 检测实现
        
    Example:
        >>> # 在 LangGraph 节点中使用
        >>> def my_node(state):
        ...     tools = get_cached_mcp_tools()
        ...     # 使用工具进行处理
        ...     return {"result": process_with_tools(tools)}
    """
    global _cache_initialized

    # 检查缓存是否因配置文件变更而过期
    if _is_cache_stale():
        logger.info("MCP cache is stale, resetting for re-initialization...")
        reset_mcp_tools_cache()

    if not _cache_initialized:
        logger.info("MCP tools not initialized, performing lazy initialization...")
        try:
            # 尝试在当前事件循环中初始化
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 如果循环已经在运行（如在 LangGraph Studio 中），
                # 需要在线程中创建新循环来运行异步初始化
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    # 在新线程中运行 asyncio.run，避免与主循环冲突
                    future = executor.submit(asyncio.run, initialize_mcp_tools())
                    future.result()  # 等待初始化完成
            else:
                # 如果没有循环在运行，可以使用当前循环
                loop.run_until_complete(initialize_mcp_tools())
        except RuntimeError:
            # 没有事件循环存在，创建一个
            try:
                asyncio.run(initialize_mcp_tools())
            except Exception:
                logger.exception("Failed to lazy-initialize MCP tools")
                return []
        except Exception:
            logger.exception("Failed to lazy-initialize MCP tools")
            return []

    return _mcp_tools_cache or []


def reset_mcp_tools_cache() -> None:
    """重置 MCP 工具缓存。
    
    清除所有缓存状态，包括工具列表、初始化标志和配置文件修改时间。
    下次调用 get_cached_mcp_tools() 时会重新初始化。
    
    Note:
        - 将所有全局变量重置为初始状态
        - 主要用于测试场景，避免测试之间的状态污染
        - 也可用于强制重新加载 MCP 工具（如配置发生重大变更）
        - 不会主动关闭已创建的 MCP 客户端连接，依赖垃圾回收
        
    Example:
        >>> # 测试场景
        >>> reset_mcp_tools_cache()
        >>> tools = get_cached_mcp_tools()  # 会重新初始化
        >>> 
        >>> # 配置变更后强制重载
        >>> reload_extensions_config()
        >>> reset_mcp_tools_cache()
        >>> tools = get_cached_mcp_tools()  # 会使用新配置重新加载
    """
    global _mcp_tools_cache, _cache_initialized, _config_mtime
    _mcp_tools_cache = None
    _cache_initialized = False
    _config_mtime = None
    logger.info("MCP tools cache reset")
