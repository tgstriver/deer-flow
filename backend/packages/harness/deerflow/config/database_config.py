"""统一数据库后端配置模块。

同时控制 LangGraph 检查点和 DeerFlow 应用持久化层（运行记录、线程元数据、用户等）。
用户只需配置一个后端，系统自动处理物理隔离细节。

SQLite 模式：检查点和应用共享同一个 .db 文件（{sqlite_dir}/deerflow.db），
每个连接启用 WAL 日志模式。WAL 允许并发读取和单一写入而不阻塞，
使统一文件对两种工作负载安全可用。写入竞争者通过默认 5 秒 sqlite3
busy timeout 等待而非立即失败。

Postgres 模式：两者使用相同的数据库 URL，但维护独立连接池，生命周期不同。

Memory 模式：检查点使用 MemorySaver，应用使用内存存储，不初始化数据库。

敏感值（postgres_url）应在 config.yaml 中使用 $VAR 语法引用 .env 中的环境变量：

    database:
      backend: postgres
      postgres_url: $DATABASE_URL

$VAR 的解析由 AppConfig.resolve_env_variables() 在此配置实例化之前完成——
DatabaseConfig 本身不需要做任何环境变量处理。
"""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, Field


class DatabaseConfig(BaseModel):
    backend: Literal["memory", "sqlite", "postgres"] = Field(
        default="memory",
        description=("Storage backend for both checkpointer and application data. 'memory' for development (no persistence across restarts), 'sqlite' for single-node deployment, 'postgres' for production multi-node deployment."),
    )
    sqlite_dir: str = Field(
        default=".deer-flow/data",
        description=("Directory for the SQLite database file. Both checkpointer and application data share {sqlite_dir}/deerflow.db."),
    )
    postgres_url: str = Field(
        default="",
        description=(
            "PostgreSQL connection URL, shared by checkpointer and app. "
            "Use $DATABASE_URL in config.yaml to reference .env. "
            "Example: postgresql://user:pass@host:5432/deerflow "
            "(the +asyncpg driver suffix is added automatically where needed)."
        ),
    )
    echo_sql: bool = Field(
        default=False,
        description="Echo all SQL statements to log (debug only).",
    )
    pool_size: int = Field(
        default=5,
        description="Connection pool size for the app ORM engine (postgres only).",
    )

    # -- Derived helpers (not user-configured) --

    @property
    def _resolved_sqlite_dir(self) -> str:
        """Resolve sqlite_dir to an absolute path (relative to CWD)."""
        from pathlib import Path

        return str(Path(self.sqlite_dir).resolve())

    @property
    def sqlite_path(self) -> str:
        """Unified SQLite file path shared by checkpointer and app."""
        return os.path.join(self._resolved_sqlite_dir, "deerflow.db")

    # Backward-compatible aliases
    @property
    def checkpointer_sqlite_path(self) -> str:
        """SQLite file path for the LangGraph checkpointer (alias for sqlite_path)."""
        return self.sqlite_path

    @property
    def app_sqlite_path(self) -> str:
        """SQLite file path for application ORM data (alias for sqlite_path)."""
        return self.sqlite_path

    @property
    def app_sqlalchemy_url(self) -> str:
        """SQLAlchemy async URL for the application ORM engine."""
        if self.backend == "sqlite":
            return f"sqlite+aiosqlite:///{self.sqlite_path}"
        if self.backend == "postgres":
            url = self.postgres_url
            if url.startswith("postgresql://"):
                url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
            return url
        raise ValueError(f"No SQLAlchemy URL for backend={self.backend!r}")
