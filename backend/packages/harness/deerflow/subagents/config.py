"""子代理配置定义模块 —— 定义子代理的配置数据类及模型解析逻辑。

本模块包含：
- SubagentConfig: 子代理的核心配置数据类，涵盖名称、描述、系统提示词、
  工具白名单/黑名单、技能列表、模型、最大轮次和超时时间等
- resolve_subagent_model_name: 解析子代理实际使用的模型名称，
  支持继承（inherit）父代理模型、显式指定模型名、或从 AppConfig 获取默认模型

Subagent configuration definitions.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from deerflow.config.app_config import AppConfig


@dataclass
class SubagentConfig:
    """子代理配置数据类 —— 定义子代理的所有可配置属性。

    该数据类描述了子代理的完整行为规范，包括身份标识、行为引导、
    工具访问控制、技能加载策略和执行约束等。

    Configuration for a subagent.

    Attributes:
        name: 子代理的唯一标识符。/ Unique identifier for the subagent.
        description: 描述何时应将任务委派给此子代理。/ When Claude should delegate to this subagent.
        system_prompt: 指导子代理行为的系统提示词。/ The system prompt that guides the subagent's behavior.
        tools: 可选的工具名称白名单。为 None 时继承所有工具。/ Optional list of tool names to allow. If None, inherits all tools.
        disallowed_tools: 可选的工具名称黑名单。/ Optional list of tool names to deny.
        skills: 可选的技能名称列表。为 None 时继承所有已启用的技能，
                为空列表时不加载任何技能。/ Optional list of skill names to load. If None, inherits all enabled skills.
                If an empty list, no skills are loaded.
        model: 使用的模型名称。'inherit' 表示继承父代理的模型。/ Model to use - 'inherit' uses parent's model.
        max_turns: 子代理执行的最大轮次数。/ Maximum number of agent turns before stopping.
        timeout_seconds: 最大执行时间（秒），默认 900 秒（15 分钟）。/ Maximum execution time in seconds (default: 900 = 15 minutes).
    """

    name: str
    description: str
    system_prompt: str | None = None
    tools: list[str] | None = None
    # 默认禁止 task 工具，防止子代理嵌套委派 / Default disallow task tool to prevent subagent nesting
    disallowed_tools: list[str] | None = field(default_factory=lambda: ["task"])
    skills: list[str] | None = None
    model: str = "inherit"
    max_turns: int = 50
    timeout_seconds: int = 900


def _default_model_name(app_config: "AppConfig") -> str:
    """从 AppConfig 获取默认模型名称。

    当子代理配置为继承模型且无法从父代理获取模型名时，使用此函数
    从应用配置中获取第一个已配置的模型名称作为默认值。

    Args:
        app_config: 应用配置对象。/ Application configuration object.

    Returns:
        默认模型名称。/ Default model name.

    Raises:
        ValueError: 当没有配置任何模型时抛出。/ When no models are configured.
    """
    if not app_config.models:
        raise ValueError("No chat models are configured. Please configure at least one model in config.yaml.")
    return app_config.models[0].name


def resolve_subagent_model_name(config: SubagentConfig, parent_model: str | None, *, app_config: "AppConfig | None" = None) -> str:
    """解析子代理实际使用的模型名称。

    模型名称解析优先级：
    1. 如果配置中指定了非 'inherit' 的模型名，直接使用该名称
    2. 如果父代理模型名不为 None，使用父代理的模型
    3. 否则从 AppConfig 获取默认模型名称

    Resolve the effective model name a subagent should use.

    Args:
        config: 子代理配置对象。/ Subagent configuration object.
        parent_model: 父代理的模型名称，可为 None。/ Parent agent's model name, may be None.
        app_config: 可选的应用配置对象。为 None 时自动加载。/ Optional AppConfig. Auto-loaded when None.

    Returns:
        子代理应使用的模型名称。/ The effective model name a subagent should use.
    """
    # 优先级1：显式指定的模型名（非继承）/ Priority 1: explicitly specified model (not inherit)
    if config.model != "inherit":
        return config.model

    # 优先级2：继承父代理的模型 / Priority 2: inherit from parent agent
    if parent_model is not None:
        return parent_model

    # 优先级3：从 AppConfig 获取默认模型 / Priority 3: get default from AppConfig
    if app_config is None:
        from deerflow.config import get_app_config

        app_config = get_app_config()
    return _default_model_name(app_config)
