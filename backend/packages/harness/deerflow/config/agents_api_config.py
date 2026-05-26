"""自定义代理管理 API 配置模块（Configuration for the custom agents management API）。

本模块负责管理自定义代理和用户配置文件的 HTTP API 开关。
当启用时，Gateway 暴露读写自定义代理 SOUL.md、config 和 USER.md 的路由；
当禁用时，所有相关路由均被拒绝。
"""

from pydantic import BaseModel, Field


class AgentsApiConfig(BaseModel):
    """自定义代理与用户配置文件管理 API 的配置模型（Configuration for custom-agent and user-profile management routes）。

    控制是否通过 HTTP 暴露自定义代理管理接口，包括 SOUL.md、配置文件
    和 USER.md 提示词管理路由。
    """

    enabled: bool = Field(
        default=False,
        description=(
            "Whether to expose the custom-agent management API over HTTP. When disabled, the gateway rejects read/write access to custom agent SOUL.md, config, and USER.md prompt-management routes."
            " / 是否通过 HTTP 暴露自定义代理管理 API。禁用时，Gateway 拒绝对自定义代理 SOUL.md、"
            "config 和 USER.md 提示词管理路由的读写访问。"
        ),
    )


# 全局单例配置实例
_agents_api_config: AgentsApiConfig = AgentsApiConfig()


def get_agents_api_config() -> AgentsApiConfig:
    """获取当前代理 API 配置（Get the current agents API configuration）。

    Returns:
        当前的 AgentsApiConfig 实例。
    """
    return _agents_api_config


def set_agents_api_config(config: AgentsApiConfig) -> None:
    """设置代理 API 配置（Set the agents API configuration）。

    Args:
        config: 要设置的 AgentsApiConfig 实例。
    """
    global _agents_api_config
    _agents_api_config = config


def load_agents_api_config_from_dict(config_dict: dict) -> None:
    """从字典加载代理 API 配置（Load agents API configuration from a dictionary）。

    Args:
        config_dict: 包含 AgentsApiConfig 字段的字典，通常来自 config.yaml 的 agents_api 节。
    """
    global _agents_api_config
    _agents_api_config = AgentsApiConfig(**config_dict)
