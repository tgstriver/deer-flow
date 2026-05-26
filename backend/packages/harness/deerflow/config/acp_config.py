"""ACP（Agent Client Protocol，代理客户端协议）代理配置模块。

本模块负责加载和管理 ACP 兼容外部代理的配置信息。
ACP 代理通过子进程方式启动，由 DeerFlow 主代理在运行时调用。

配置来源：config.yaml 中的 acp_agents 字段。
"""

import logging
from collections.abc import Mapping

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ACPAgentConfig(BaseModel):
    """单个 ACP 兼容代理的配置模型（Configuration for a single ACP-compatible agent）。

    每个 ACP 代理作为独立子进程运行，DeerFlow 通过 ACP 协议与其通信。
    配置项包括启动命令、环境变量、模型提示以及权限自动审批等。
    """

    command: str = Field(description="Command to launch the ACP agent subprocess / 启动 ACP 代理子进程的命令")
    args: list[str] = Field(default_factory=list, description="Additional command arguments / 附加命令行参数列表")
    env: dict[str, str] = Field(
        default_factory=dict, description="Environment variables to inject into the agent subprocess. Values starting with $ are resolved from host environment variables. / 注入代理子进程的环境变量，以 $ 开头的值将从宿主环境变量中解析"
    )
    description: str = Field(description="Description of the agent's capabilities (shown in tool description) / 代理能力描述（显示在工具描述中）")
    model: str | None = Field(default=None, description="Model hint passed to the agent (optional) / 传递给代理的模型提示（可选）")
    auto_approve_permissions: bool = Field(
        default=False,
        description=(
            "When True, DeerFlow automatically approves all ACP permission requests from this agent "
            "(allow_once preferred over allow_always). When False (default), all permission requests "
            "are denied — the agent must be configured to operate without requesting permissions."
            " / 为 True 时，DeerFlow 自动批准该代理的所有 ACP 权限请求（优先 allow_once）；"
            "为 False（默认）时，所有权限请求均被拒绝——代理必须配置为无需请求权限即可运行。"
        ),
    )


# 全局 ACP 代理配置字典，键为代理名称，值为 ACPAgentConfig
_acp_agents: dict[str, ACPAgentConfig] = {}


def get_acp_agents() -> dict[str, ACPAgentConfig]:
    """获取当前已配置的 ACP 代理列表（Get the currently configured ACP agents）。

    Returns:
        Mapping of agent name -> ACPAgentConfig.  Empty dict if no ACP agents are configured.
        代理名称到 ACPAgentConfig 的映射字典，若未配置则返回空字典。
    """
    return _acp_agents


def load_acp_config_from_dict(config_dict: Mapping[str, Mapping[str, object]] | None) -> None:
    """从字典加载 ACP 代理配置（Load ACP agent configuration from a dictionary）。

    通常从 config.yaml 解析后调用此函数。会将全局 _acp_agents 替换为
    新解析的配置字典。

    Args:
        config_dict: Mapping of agent name -> config fields.
            代理名称到配置字段的映射，通常来自 config.yaml 的 acp_agents 节。
    """
    global _acp_agents
    if config_dict is None:
        config_dict = {}
    _acp_agents = {name: ACPAgentConfig(**cfg) for name, cfg in config_dict.items()}
    logger.info("ACP config loaded: %d agent(s): %s", len(_acp_agents), list(_acp_agents.keys()))
