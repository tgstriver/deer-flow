"""异步检查点工厂。

Async checkpointer factory.

为长时间运行的异步服务器提供**异步上下文管理器**，确保正确的资源清理。

Provides an **async context manager** for long-running async servers that need
proper resource cleanup.

支持的后端：memory（内存）、sqlite、postgres。

Supported backends: memory, sqlite, postgres.

用法示例（如 FastAPI lifespan）::

    from deerflow.runtime.checkpointer.async_provider import make_checkpointer

    async with make_checkpointer() as checkpointer:
        app.state.checkpointer = checkpointer  # InMemorySaver if not configured

同步用法请参考 :mod:`deerflow.runtime.checkpointer.provider`。

For sync usage see :mod:`deerflow.runtime.checkpointer.provider`.
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

    Async context manager that constructs and tears down a checkpointer.

    根据配置类型返回对应的检查点存储实例，退出时自动清理资源。
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
    """Async context manager that constructs a checkpointer from unified DatabaseConfig."""
    if db_config.backend == "memory":
        from langgraph.checkpoint.memory import InMemorySaver

        yield InMemorySaver()
        return

    if db_config.backend == "sqlite":
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
    """Async context manager that yields a checkpointer for the caller's lifetime.
    Resources are opened on enter and closed on exit -- no global state::

        async with make_checkpointer(app_config) as checkpointer:
            app.state.checkpointer = checkpointer

    Yields an ``InMemorySaver`` when no checkpointer is configured in *config.yaml*.

    Priority:
    1. Legacy ``checkpointer:`` config section (backward compatible)
    2. Unified ``database:`` config section
    3. Default InMemorySaver
    """

    if app_config is None:
        app_config = get_app_config()

    # Legacy: standalone checkpointer config takes precedence
    if app_config.checkpointer is not None:
        async with _async_checkpointer(app_config.checkpointer) as saver:
            yield saver
            return

    # Unified database config
    db_config = getattr(app_config, "database", None)
    if db_config is not None and db_config.backend != "memory":
        async with _async_checkpointer_from_database(db_config) as saver:
            yield saver
            return

    # Default: in-memory
    from langgraph.checkpoint.memory import InMemorySaver

    yield InMemorySaver()
