"""异步检查点工厂。

为长时间运行的异步服务器提供**异步上下文管理器**，确保正确的资源清理。

支持的后端：memory（内存）、sqlite、postgres。

用法示例（如 FastAPI lifespan）::

    from deerflow.runtime.checkpointer.async_provider import make_checkpointer

    async with make_checkpointer() as checkpointer:
        app.state.checkpointer = checkpointer  # InMemorySaver if not configured

同步用法请参考 :mod:`deerflow.runtime.checkpointer.provider`。

本模块提供：
- 异步上下文管理器，用于创建和管理检查点存储
- 支持多种后端（内存、SQLite、PostgreSQL）
- 自动资源清理和初始化
- 配置驱动的优先级选择机制
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator

from langgraph.types import Checkpointer

from deerflow.config.app_config import AppConfig, get_app_config
from deerflow.runtime.checkpointer.provider import (
    POSTGRES_CONN_REQUIRED,
    POSTGRES_INSTALL,
    SQLITE_INSTALL,
)
from deerflow.runtime.store._sqlite_utils import ensure_sqlite_parent_dir, resolve_sqlite_conn_str

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 异步工厂 / Async factory
# ---------------------------------------------------------------------------


@contextlib.asynccontextmanager
async def _async_checkpointer(config) -> AsyncIterator[Checkpointer]:
    """异步上下文管理器，构造并销毁检查点存储。
    
    根据配置类型返回对应的检查点存储实例，退出时自动清理资源。
    
    Args:
        config: 检查点配置对象，包含 type 和 connection_string 字段
        
    Yields:
        Checkpointer 实例（InMemorySaver、AsyncSqliteSaver 或 AsyncPostgresSaver）
        
    Raises:
        ImportError: 如果缺少必要的依赖包（sqlite 或 postgres）
        ValueError: 如果配置类型未知或 postgres 缺少连接字符串
        
    Note:
        - 支持三种后端：memory、sqlite、postgres
        - SQLite 模式会自动创建父目录
        - 所有异步 saver 都会调用 setup() 进行初始化
        - 使用 async with 确保资源正确清理
    """
    if config.type == "memory":
        # 内存模式：使用 InMemorySaver，进程内非持久化
        from langgraph.checkpoint.memory import InMemorySaver

        yield InMemorySaver()
        return

    if config.type == "sqlite":
        # SQLite模式：使用异步 SQLite 检查点存储
        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        except ImportError as exc:
            raise ImportError(SQLITE_INSTALL) from exc

        conn_str = resolve_sqlite_conn_str(config.connection_string or "store.db")
        # 在线程中确保 SQLite 父目录存在（避免阻塞事件循环）
        await asyncio.to_thread(ensure_sqlite_parent_dir, conn_str)
        async with AsyncSqliteSaver.from_conn_string(conn_str) as saver:
            await saver.setup()
            yield saver
        return

    if config.type == "postgres":
        # PostgreSQL模式：使用异步 Postgres 检查点存储
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        except ImportError as exc:
            raise ImportError(POSTGRES_INSTALL) from exc

        if not config.connection_string:
            raise ValueError(POSTGRES_CONN_REQUIRED)

        async with AsyncPostgresSaver.from_conn_string(config.connection_string) as saver:
            await saver.setup()
            yield saver
        return

    raise ValueError(f"Unknown checkpointer type: {config.type!r}")


# ---------------------------------------------------------------------------
# Public async context manager
# ---------------------------------------------------------------------------


@contextlib.asynccontextmanager
async def _async_checkpointer_from_database(db_config) -> AsyncIterator[Checkpointer]:
    """从统一的 DatabaseConfig 构造检查点的异步上下文管理器。
    
    根据数据库配置的后端类型返回对应的检查点存储实例。
    
    Args:
        db_config: 数据库配置对象，包含 backend、checkpointer_sqlite_path、postgres_url 字段
        
    Yields:
        Checkpointer 实例
        
    Raises:
        ImportError: 如果缺少必要的依赖包
        ValueError: 如果后端类型未知或 postgres 缺少连接 URL
        
    Note:
        - 与 _async_checkpointer 不同，此函数使用统一的 database 配置结构
        - SQLite 路径直接从配置读取，不经过 resolve_sqlite_conn_str
        - PostgreSQL 需要 postgres_url 字段
    """
    if db_config.backend == "memory":
        # 内存模式：使用 InMemorySaver，进程内非持久化
        from langgraph.checkpoint.memory import InMemorySaver

        yield InMemorySaver()
        return

    if db_config.backend == "sqlite":
        # SQLite模式：使用异步 SQLite 检查点存储
        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        except ImportError as exc:
            raise ImportError(SQLITE_INSTALL) from exc

        conn_str = db_config.checkpointer_sqlite_path
        ensure_sqlite_parent_dir(conn_str)
        async with AsyncSqliteSaver.from_conn_string(conn_str) as saver:
            await saver.setup()
            yield saver
        return

    if db_config.backend == "postgres":
        # PostgreSQL模式：使用异步 Postgres 检查点存储
        try:
            from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        except ImportError as exc:
            raise ImportError(POSTGRES_INSTALL) from exc

        if not db_config.postgres_url:
            raise ValueError("database.postgres_url is required for the postgres backend")

        async with AsyncPostgresSaver.from_conn_string(db_config.postgres_url) as saver:
            await saver.setup()
            yield saver
        return

    raise ValueError(f"Unknown database backend: {db_config.backend!r}")


@contextlib.asynccontextmanager
async def make_checkpointer(app_config: AppConfig | None = None) -> AsyncIterator[Checkpointer]:
    """异步上下文管理器，为调用者的生命周期生成检查点。
    
    资源在进入时打开，在退出时关闭 —— 无全局状态::

        async with make_checkpointer(app_config) as checkpointer:
            app.state.checkpointer = checkpointer

    当 *config.yaml* 中未配置检查点时，生成 ``InMemorySaver``。
    
    Args:
        app_config: 应用配置对象，如果为 None 则从全局获取
        
    Yields:
        Checkpointer 实例
        
    Priority（优先级从高到低）:
        1. 遗留的 ``checkpointer:`` 配置节（向后兼容）
        2. 统一的 ``database:`` 配置节
        3. 默认的 InMemorySaver
        
    Note:
        - 优先使用遗留的 checkpointer 配置以保持向后兼容
        - 如果没有配置或后端为 memory，则使用内存存储
        - 所有资源都在 async with 块结束时自动清理
        - 不会创建全局状态，每次调用都创建新的实例
    """

    if app_config is None:
        app_config = get_app_config()

    # 遗留配置：独立的 checkpointer 配置优先（向后兼容）
    if app_config.checkpointer is not None:
        async with _async_checkpointer(app_config.checkpointer) as saver:
            yield saver
            return

    # 统一的 database 配置
    db_config = getattr(app_config, "database", None)
    if db_config is not None and db_config.backend != "memory":
        async with _async_checkpointer_from_database(db_config) as saver:
            yield saver
            return

    # 默认：内存存储
    from langgraph.checkpoint.memory import InMemorySaver

    yield InMemorySaver()
