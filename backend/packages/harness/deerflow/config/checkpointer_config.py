"""检查点（Checkpointer）配置模块。

定义 LangGraph 状态持久化检查点的配置参数。检查点用于在对话过程中保存和恢复
Agent 的状态，确保中断或重启后能继续之前的执行流程。

支持三种后端类型：
- memory: 进程内存储，仅用于开发调试，重启后数据丢失
- sqlite: 本地文件持久化，适用于单节点部署
- postgres: PostgreSQL 数据库持久化，适用于生产环境多节点部署
"""

from typing import Literal

from pydantic import BaseModel, Field

# 检查点后端类型枚举，限定为三种可选值
CheckpointerType = Literal["memory", "sqlite", "postgres"]


class CheckpointerConfig(BaseModel):
    """LangGraph 状态持久化检查点配置类。

    定义检查点后端的类型和连接参数。不同后端类型有不同的持久化策略：
    - memory: 纯内存存储，进程退出即丢失，仅适合开发调试
    - sqlite: 将状态持久化到本地 SQLite 文件，适合单节点部署
    - postgres: 将状态持久化到 PostgreSQL 数据库，适合生产环境，
      安装时需使用 deerflow-harness[postgres] 依赖
    """

    type: CheckpointerType = Field(description="检查点后端类型。'memory' 仅进程内有效（重启后丢失）。'sqlite' 持久化到本地文件（需安装 langgraph-checkpoint-sqlite）。'postgres' 持久化到 PostgreSQL（安装时需 deerflow-harness[postgres]）。")
    connection_string: str | None = Field(
        default=None,
        description="连接字符串，用于 sqlite（文件路径）或 postgres（DSN）。"
        "sqlite 可省略，默认为 'store.db'。"
        "postgres 必须提供。"
        "sqlite 示例：'.deer-flow/checkpoints.db' 或 ':memory:'（内存模式）。"
        "postgres 示例：'postgresql://user:pass@localhost:5432/db'。",
    )


# 全局配置实例 — None 表示未配置检查点
_checkpointer_config: CheckpointerConfig | None = None


def get_checkpointer_config() -> CheckpointerConfig | None:
    """获取当前检查点配置实例。

    Returns:
        当前 CheckpointerConfig 实例，若未配置则返回 None
    """
    return _checkpointer_config


def set_checkpointer_config(config: CheckpointerConfig | None) -> None:
    """设置检查点配置实例。

    Args:
        config: 要设置的 CheckpointerConfig 实例，可传入 None 清除配置
    """
    global _checkpointer_config
    _checkpointer_config = config


def load_checkpointer_config_from_dict(config_dict: dict | None) -> None:
    """从字典加载检查点配置。

    通常在 AppConfig 加载过程中调用，将 config.yaml 中对应段落解析为配置实例。

    Args:
        config_dict: 包含检查点配置参数的字典，若为 None 则清除全局配置
    """
    global _checkpointer_config
    if config_dict is None:
        # 未提供配置字典时，清除全局配置
        _checkpointer_config = None
        return
    _checkpointer_config = CheckpointerConfig(**config_dict)
