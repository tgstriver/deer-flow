"""自定义代理配置与加载器模块（Configuration and loaders for custom agents）。

自定义代理按用户隔离存储在 ``{base_dir}/users/{user_id}/agents/{name}/`` 目录下。
旧版共享布局 ``{base_dir}/agents/{name}/`` 仍然可读，以便在用户运行
``scripts/migrate_user_isolation.py`` 迁移脚本之前，早期安装的代理继续可用。
新的写入操作始终指向按用户隔离的布局。
"""

import logging
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from deerflow.config.paths import get_paths
from deerflow.runtime.user_context import get_effective_user_id

logger = logging.getLogger(__name__)

# SOUL.md 文件名常量，定义代理的个性、价值观和行为准则
SOUL_FILENAME = "SOUL.md"
# 代理名称合法模式：仅允许字母、数字和连字符
AGENT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9-]+$")


def validate_agent_name(name: str | None) -> str | None:
    """在将代理名称用于文件系统路径之前进行校验（Validate a custom agent name before using it in filesystem paths）。

    代理名称必须由字母、数字和连字符组成，以防止路径遍历等安全风险。

    Args:
        name: 待校验的代理名称，None 表示使用默认代理。

    Returns:
        校验通过后的名称，或 None（当输入为 None 时）。

    Raises:
        ValueError: 名称不合法（非字符串或包含非法字符）。
    """
    if name is None:
        return None
    if not isinstance(name, str):
        raise ValueError("Invalid agent name. Expected a string or None. / 无效的代理名称，期望字符串或 None。")
    if not AGENT_NAME_PATTERN.fullmatch(name):
        raise ValueError(f"Invalid agent name '{name}'. Must match pattern: {AGENT_NAME_PATTERN.pattern} / 无效的代理名称 '{name}'，必须匹配模式: {AGENT_NAME_PATTERN.pattern}")
    return name


class AgentConfig(BaseModel):
    """自定义代理的配置模型（Configuration for a custom agent）。

    定义代理的基本元数据和工具/技能绑定关系。
    配置文件为代理目录下的 config.yaml。
    """

    name: str  # 代理唯一名称
    description: str = ""  # 代理功能描述
    model: str | None = None  # 代理使用的模型名称，None 表示使用默认模型
    tool_groups: list[str] | None = None  # 代理可用的工具组列表，None 表示使用默认工具组
    # skills controls which skills are loaded into the agent's prompt:
    # - None (or omitted): load all enabled skills (default fallback behavior)
    # - [] (explicit empty list): disable all skills
    # - ["skill1", "skill2"]: load only the specified skills
    # skills 控制加载到代理提示词中的技能：
    # - None（或省略）：加载所有已启用的技能（默认回退行为）
    # - []（显式空列表）：禁用所有技能
    # - ["skill1", "skill2"]：仅加载指定技能
    skills: list[str] | None = None


def resolve_agent_dir(name: str, *, user_id: str | None = None) -> Path:
    """返回代理在磁盘上的目录，优先使用按用户隔离的布局（Return the on-disk directory for an agent, preferring the per-user layout）。

    解析顺序：
    1. ``{base_dir}/users/{user_id}/agents/{name}/``（按用户隔离，当前布局）。
    2. ``{base_dir}/agents/{name}/``（旧版共享布局——只读回退）。

    若两者均不存在，则返回按用户隔离的路径，以便调用者写入新布局。

    Args:
        name: 已校验的代理名称。
        user_id: 代理所有者。默认使用请求上下文中的有效用户
            （无认证模式下为 ``"default"``）。

    Returns:
        代理目录的 Path 对象。
    """
    paths = get_paths()
    effective_user = user_id or get_effective_user_id()
    user_path = paths.user_agent_dir(effective_user, name)
    if user_path.exists():
        return user_path

    legacy_path = paths.agent_dir(name)
    if legacy_path.exists():
        return legacy_path

    return user_path


def load_agent_config(name: str | None, *, user_id: str | None = None) -> AgentConfig | None:
    """从代理目录加载自定义或默认代理的配置（Load the custom or default agent's config from its directory）。

    优先从按用户隔离的布局读取；对于尚未迁移的安装，
    回退到旧版共享布局。

    Args:
        name: 代理名称。
        user_id: 代理所有者。默认使用当前请求上下文中的有效用户。

    Returns:
        AgentConfig 实例，或 ``None``（当 ``name`` 为 ``None`` 时）。

    Raises:
        FileNotFoundError: 代理目录或 config.yaml 不存在。
        ValueError: config.yaml 无法解析。
    """

    if name is None:
        return None

    name = validate_agent_name(name)
    agent_dir = resolve_agent_dir(name, user_id=user_id)
    config_file = agent_dir / "config.yaml"

    if not agent_dir.exists():
        raise FileNotFoundError(f"Agent directory not found: {agent_dir} / 未找到代理目录: {agent_dir}")

    if not config_file.exists():
        raise FileNotFoundError(f"Agent config not found: {config_file} / 未找到代理配置: {config_file}")

    try:
        with open(config_file, encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse agent config {config_file}: {e} / 解析代理配置失败 {config_file}: {e}") from e

    # Ensure name is set from directory name if not in file
    # 如果文件中没有 name 字段，则从目录名推断
    if "name" not in data:
        data["name"] = name

    # Strip unknown fields before passing to Pydantic (e.g. legacy prompt_file)
    # 传给 Pydantic 前剥离未知字段（如旧版 prompt_file），避免校验错误
    known_fields = set(AgentConfig.model_fields.keys())
    data = {k: v for k, v in data.items() if k in known_fields}

    return AgentConfig(**data)


def load_agent_soul(agent_name: str | None, *, user_id: str | None = None) -> str | None:
    """读取自定义代理的 SOUL.md 文件内容（Read the SOUL.md file for a custom agent, if it exists）。

    SOUL.md 定义代理的个性、价值观和行为准则。
    其内容会被注入到主代理的系统提示词中作为额外上下文。

    Args:
        agent_name: 代理名称，None 表示默认代理。
        user_id: 代理所有者。默认使用当前请求上下文中的有效用户。

    Returns:
        SOUL.md 的文本内容，若文件不存在则返回 None。
    """
    if agent_name:
        agent_dir = resolve_agent_dir(agent_name, user_id=user_id)
    else:
        agent_dir = get_paths().base_dir
    soul_path = agent_dir / SOUL_FILENAME
    if not soul_path.exists():
        return None
    content = soul_path.read_text(encoding="utf-8").strip()
    return content or None


def list_custom_agents(*, user_id: str | None = None) -> list[AgentConfig]:
    """扫描代理目录并返回所有有效的自定义代理（Scan the agents directory and return all valid custom agents）。

    返回按用户隔离布局和旧版共享布局中代理的并集，
    以便未迁移的安装仍能看到旧代理。按用户布局的条目会覆盖
    同名的旧版条目。

    Args:
        user_id: 要列出代理的用户。默认使用当前请求上下文中的有效用户。

    Returns:
        找到的每个有效代理目录对应的 AgentConfig 列表，按名称排序。
    """
    paths = get_paths()
    effective_user = user_id or get_effective_user_id()

    seen: set[str] = set()
    agents: list[AgentConfig] = []

    user_root = paths.user_agents_dir(effective_user)
    legacy_root = paths.agents_dir

    for root in (user_root, legacy_root):
        if not root.exists():
            continue
        for entry in sorted(root.iterdir()):
            if not entry.is_dir():
                continue
            if entry.name in seen:
                continue
            config_file = entry / "config.yaml"
            if not config_file.exists():
                logger.debug(f"Skipping {entry.name}: no config.yaml / 跳过 {entry.name}: 无 config.yaml")
                continue

            try:
                agent_cfg = load_agent_config(entry.name, user_id=effective_user)
                if agent_cfg is None:
                    continue
                agents.append(agent_cfg)
                seen.add(entry.name)
            except Exception as e:
                logger.warning(f"Skipping agent '{entry.name}': {e} / 跳过代理 '{entry.name}': {e}")

    agents.sort(key=lambda a: a.name)
    return agents
