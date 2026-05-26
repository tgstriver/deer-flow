"""Cache for MCP tools to avoid repeated loading.

MCP 工具缓存模块，用于避免重复加载 MCP 工具。
提供懒加载初始化、基于配置文件修改时间的缓存失效机制，以及线程安全的事件循环兼容处理。
"""

import asyncio
import logging
import os

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

# 全局 MCP 工具缓存，None 表示尚未初始化
_mcp_tools_cache: list[BaseTool] | None = None
# 缓存是否已初始化的标志
_cache_initialized = False
# 初始化锁，防止并发初始化竞争
_initialization_lock = asyncio.Lock()
# 配置文件的最后修改时间，用于缓存失效检测
_config_mtime: float | None = None  # Track config file modification time


def _get_config_mtime() -> float | None:
    """Get the modification time of the extensions config file.

    获取扩展配置文件的最后修改时间。

    Returns:
        The modification time as a float, or None if the file doesn't exist.
        以浮点数返回修改时间，若文件不存在则返回 None。
    """
    from deerflow.config.extensions_config import ExtensionsConfig

    config_path = ExtensionsConfig.resolve_config_path()
    if config_path and config_path.exists():
        return os.path.getmtime(config_path)
    return None


def _is_cache_stale() -> bool:
    """Check if the cache is stale due to config file changes.

    检查缓存是否因配置文件变更而过期。
    通过对比当前配置文件修改时间与缓存时记录的修改时间来判断。

    Returns:
        True if the cache should be invalidated, False otherwise.
        若缓存应失效返回 True，否则返回 False。
    """
    global _config_mtime

    if not _cache_initialized:
        return False  # Not initialized yet, not stale — 尚未初始化，不算过期

    current_mtime = _get_config_mtime()

    # If we couldn't get mtime before or now, assume not stale
    # 若之前或现在无法获取修改时间，假定缓存未过期
    if _config_mtime is None or current_mtime is None:
        return False

    # If the config file has been modified since we cached, it's stale
    # 若配置文件在缓存后已被修改，则缓存过期
    if current_mtime > _config_mtime:
        logger.info(f"MCP config file has been modified (mtime: {_config_mtime} -> {current_mtime}), cache is stale")
        return True

    return False


async def initialize_mcp_tools() -> list[BaseTool]:
    """Initialize and cache MCP tools.

    初始化并缓存 MCP 工具。
    此方法应在应用启动时调用一次。使用异步锁确保并发安全。

    This should be called once at application startup.

    Returns:
        List of LangChain tools from all enabled MCP servers.
        来自所有已启用 MCP 服务器的 LangChain 工具列表。
    """
    global _mcp_tools_cache, _cache_initialized, _config_mtime

    async with _initialization_lock:
        if _cache_initialized:
            logger.info("MCP tools already initialized")
            return _mcp_tools_cache or []

        from deerflow.mcp.tools import get_mcp_tools

        logger.info("Initializing MCP tools...")
        _mcp_tools_cache = await get_mcp_tools()
        _cache_initialized = True
        _config_mtime = _get_config_mtime()  # Record config file mtime
        logger.info(f"MCP tools initialized: {len(_mcp_tools_cache)} tool(s) loaded (config mtime: {_config_mtime})")

        return _mcp_tools_cache


def get_cached_mcp_tools() -> list[BaseTool]:
    """Get cached MCP tools with lazy initialization.

    If tools are not initialized, automatically initializes them.
    This ensures MCP tools work in both FastAPI and LangGraph Studio contexts.

    Also checks if the config file has been modified since last initialization,
    and re-initializes if needed. This ensures that changes made through the
    Gateway API (which runs in a separate process) are reflected in the
    LangGraph Server.

    Returns:
        List of cached MCP tools.
    """
    global _cache_initialized

    # Check if cache is stale due to config file changes
    if _is_cache_stale():
        logger.info("MCP cache is stale, resetting for re-initialization...")
        reset_mcp_tools_cache()

    if not _cache_initialized:
        logger.info("MCP tools not initialized, performing lazy initialization...")
        try:
            # Try to initialize in the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is already running (e.g., in LangGraph Studio),
                # we need to create a new loop in a thread
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, initialize_mcp_tools())
                    future.result()
            else:
                # If no loop is running, we can use the current loop
                loop.run_until_complete(initialize_mcp_tools())
        except RuntimeError:
            # No event loop exists, create one
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
    """Reset the MCP tools cache.

    This is useful for testing or when you want to reload MCP tools.
    """
    global _mcp_tools_cache, _cache_initialized, _config_mtime
    _mcp_tools_cache = None
    _cache_initialized = False
    _config_mtime = None
    logger.info("MCP tools cache reset")
