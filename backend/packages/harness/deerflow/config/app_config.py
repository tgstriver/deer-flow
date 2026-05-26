"""DeerFlow 应用全局配置模块。

本模块是配置系统的核心，定义了 AppConfig 主配置类及其加载、缓存、
热重载机制。AppConfig 聚合了所有子配置（模型、工具、沙箱、内存、
代理等），并提供从 config.yaml 文件加载配置的完整流程。

配置优先级：
1. 显式 config_path 参数
2. DEER_FLOW_CONFIG_PATH 环境变量
3. 调用方项目根目录下的 config.yaml
4. 旧版 backend/ 根目录和仓库根目录的 config.yaml（兼容单体仓库）

配置值以 $ 开头的字段会被解析为环境变量引用（如 $OPENAI_API_KEY）。
"""

import logging
import os
from collections.abc import Mapping
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Self

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from deerflow.config.acp_config import ACPAgentConfig, load_acp_config_from_dict
from deerflow.config.agents_api_config import AgentsApiConfig, load_agents_api_config_from_dict
from deerflow.config.checkpointer_config import CheckpointerConfig, load_checkpointer_config_from_dict
from deerflow.config.database_config import DatabaseConfig
from deerflow.config.extensions_config import ExtensionsConfig
from deerflow.config.guardrails_config import GuardrailsConfig, load_guardrails_config_from_dict
from deerflow.config.loop_detection_config import LoopDetectionConfig
from deerflow.config.memory_config import MemoryConfig, load_memory_config_from_dict
from deerflow.config.model_config import ModelConfig
from deerflow.config.run_events_config import RunEventsConfig
from deerflow.config.runtime_paths import existing_project_file
from deerflow.config.sandbox_config import SandboxConfig
from deerflow.config.skill_evolution_config import SkillEvolutionConfig
from deerflow.config.skills_config import SkillsConfig
from deerflow.config.stream_bridge_config import StreamBridgeConfig, load_stream_bridge_config_from_dict
from deerflow.config.subagents_config import SubagentsAppConfig, load_subagents_config_from_dict
from deerflow.config.summarization_config import SummarizationConfig, load_summarization_config_from_dict
from deerflow.config.title_config import TitleConfig, load_title_config_from_dict
from deerflow.config.token_usage_config import TokenUsageConfig
from deerflow.config.tool_config import ToolConfig, ToolGroupConfig
from deerflow.config.tool_search_config import ToolSearchConfig, load_tool_search_config_from_dict

load_dotenv()

logger = logging.getLogger(__name__)


# config.yaml 中 database 节缺失时的默认值
CONFIG_FILE_DATABASE_DEFAULTS = {
    "backend": "sqlite",  # 默认后端类型：sqlite
    "sqlite_dir": ".deer-flow/data",  # SQLite 数据文件目录
}


class CircuitBreakerConfig(BaseModel):
    """LLM 熔断器配置（Configuration for the LLM Circuit Breaker）。

    当 LLM 调用连续失败达到阈值时触发熔断，阻止后续请求直达 LLM，
    等待恢复超时后重新尝试。
    """

    failure_threshold: int = Field(default=5, description="Number of consecutive failures before tripping the circuit / 触发熔断前的连续失败次数（默认 5）")
    recovery_timeout_sec: int = Field(default=60, description="Time in seconds before attempting to recover the circuit / 熔断后尝试恢复的等待时间（秒，默认 60）")


def _legacy_config_candidates() -> tuple[Path, ...]:
    """返回源码树中的 config.yaml 候选路径，用于单体仓库兼容（Return source-tree config.yaml locations for monorepo compatibility）。"""
    backend_dir = Path(__file__).resolve().parents[4]
    repo_root = backend_dir.parent
    return (backend_dir / "config.yaml", repo_root / "config.yaml")


def logging_level_from_config(name: str | None) -> int:
    """将 config.yaml 中的 log_level 字符串映射为 logging 模块的级别常量（Map ``config.yaml`` ``log_level`` string to a :mod:`logging` level constant）。

    Args:
        name: 日志级别名称（如 "debug"、"info"、"warning"、"error"），None 默认为 "info"。

    Returns:
        对应的 logging 级别常量（整数），无法识别时返回 logging.INFO。
    """
    mapping = logging.getLevelNamesMapping()
    return mapping.get((name or "info").strip().upper(), logging.INFO)


def apply_logging_level(name: str | None) -> None:
    """解析日志级别名称并应用到 ``deerflow``/``app`` 日志器层级（Resolve *name* to a logging level and apply it to the ``deerflow``/``app`` logger hierarchies）。

    仅修改 ``deerflow`` 和 ``app`` 日志器的级别，不影响第三方库
    （如 uvicorn、sqlalchemy）的日志输出。根处理器的级别只会降低（不会升高），
    以确保已配置日志器的消息可以传播通过，同时保留可能为第三方日志输出
    设置的限制性阈值。

    Args:
        name: 日志级别名称，如 "debug"、"info"、"warning"、"error"。
    """
    level = logging_level_from_config(name)
    for logger_name in ("deerflow", "app"):
        logging.getLogger(logger_name).setLevel(level)
    for handler in logging.root.handlers:
        if level < handler.level:
            handler.setLevel(level)


class AppConfig(BaseModel):
    """DeerFlow 应用主配置模型（Config for the DeerFlow application）。

    聚合了所有子系统的配置项，是整个配置体系的核心入口。
    从 config.yaml 加载并解析后，会同步更新各子模块的全局单例配置。
    """

    log_level: str = Field(default="info", description="Logging level for deerflow and app modules (debug/info/warning/error); third-party libraries are not affected / deerflow 和 app 模块的日志级别；不影响第三方库")
    token_usage: TokenUsageConfig = Field(default_factory=TokenUsageConfig, description="Token usage tracking configuration / Token 使用量追踪配置")
    models: list[ModelConfig] = Field(default_factory=list, description="Available models / 可用的模型列表")
    sandbox: SandboxConfig = Field(description="Sandbox configuration / 沙箱配置")
    tools: list[ToolConfig] = Field(default_factory=list, description="Available tools / 可用的工具列表")
    tool_groups: list[ToolGroupConfig] = Field(default_factory=list, description="Available tool groups / 可用的工具组列表")
    skills: SkillsConfig = Field(default_factory=SkillsConfig, description="Skills configuration / 技能系统配置")
    skill_evolution: SkillEvolutionConfig = Field(default_factory=SkillEvolutionConfig, description="Agent-managed skill evolution configuration / 代理管理的技能演化配置")
    extensions: ExtensionsConfig = Field(default_factory=ExtensionsConfig, description="Extensions configuration (MCP servers and skills state) / 扩展配置（MCP 服务器和技能状态）")
    tool_search: ToolSearchConfig = Field(default_factory=ToolSearchConfig, description="Tool search / deferred loading configuration / 工具搜索/延迟加载配置")
    title: TitleConfig = Field(default_factory=TitleConfig, description="Automatic title generation configuration / 自动标题生成配置")
    summarization: SummarizationConfig = Field(default_factory=SummarizationConfig, description="Conversation summarization configuration / 对话摘要配置")
    memory: MemoryConfig = Field(default_factory=MemoryConfig, description="Memory subsystem configuration / 记忆子系统配置")
    agents_api: AgentsApiConfig = Field(default_factory=AgentsApiConfig, description="Custom-agent management API configuration / 自定义代理管理 API 配置")
    acp_agents: dict[str, ACPAgentConfig] = Field(default_factory=dict, description="ACP-compatible agent configuration / ACP 兼容代理配置")
    subagents: SubagentsAppConfig = Field(default_factory=SubagentsAppConfig, description="Subagent runtime configuration / 子代理运行时配置")
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig, description="Guardrail middleware configuration / 护栏中间件配置")
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig, description="LLM circuit breaker configuration / LLM 熔断器配置")
    loop_detection: LoopDetectionConfig = Field(default_factory=LoopDetectionConfig, description="Loop detection middleware configuration / 循环检测中间件配置")
    model_config = ConfigDict(extra="allow")  # 允许额外字段，兼容配置文件中的自定义项
    database: DatabaseConfig = Field(default_factory=DatabaseConfig, description="Unified database backend configuration / 统一数据库后端配置")
    run_events: RunEventsConfig = Field(default_factory=RunEventsConfig, description="Run event storage configuration / 运行事件存储配置")
    checkpointer: CheckpointerConfig | None = Field(default=None, description="Checkpointer configuration / 检查点配置")
    stream_bridge: StreamBridgeConfig | None = Field(default=None, description="Stream bridge configuration / 流桥接配置")

    @classmethod
    def resolve_config_path(cls, config_path: str | None = None) -> Path:
        """解析配置文件路径（Resolve the config file path）。

        优先级：
        1. 若提供了 `config_path` 参数，直接使用。
        2. 若设置了 `DEER_FLOW_CONFIG_PATH` 环境变量，使用它。
        3. 否则，搜索调用方项目根目录。
        4. 最后，搜索旧版 backend/ 根目录和仓库根目录（兼容单体仓库）。

        Args:
            config_path: 可选的配置文件路径。

        Returns:
            解析后的配置文件路径。

        Raises:
            FileNotFoundError: 配置文件未找到。
        """
        if config_path:
            path = Path(config_path)
            if not Path.exists(path):
                raise FileNotFoundError(f"Config file specified by param `config_path` not found at {path}")
            return path
        elif os.getenv("DEER_FLOW_CONFIG_PATH"):
            path = Path(os.getenv("DEER_FLOW_CONFIG_PATH"))
            if not Path.exists(path):
                raise FileNotFoundError(f"Config file specified by environment variable `DEER_FLOW_CONFIG_PATH` not found at {path}")
            return path
        else:
            project_config = existing_project_file(("config.yaml",))
            if project_config is not None:
                return project_config

            for path in _legacy_config_candidates():
                if path.exists():
                    return path
            raise FileNotFoundError("`config.yaml` file not found in the project root or legacy backend/repository root locations")

    @classmethod
    def from_file(cls, config_path: str | None = None) -> Self:
        """Load config from YAML file.

        See `resolve_config_path` for more details.

        Args:
            config_path: Path to the config file.

        Returns:
            AppConfig: The loaded config.
        """
        resolved_path = cls.resolve_config_path(config_path)
        with open(resolved_path, encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

        # Check config version before processing
        cls._check_config_version(config_data, resolved_path)

        config_data = cls.resolve_env_variables(config_data)
        cls._apply_database_defaults(config_data)

        # Load circuit_breaker config if present
        if "circuit_breaker" in config_data:
            config_data["circuit_breaker"] = config_data["circuit_breaker"]

        # Load extensions config separately (it's in a different file)
        extensions_config = ExtensionsConfig.from_file()
        config_data["extensions"] = extensions_config.model_dump()

        result = cls.model_validate(config_data)
        acp_agents = cls._validate_acp_agents(config_data.get("acp_agents", {}))
        cls._apply_singleton_configs(result, acp_agents)
        return result

    @classmethod
    def _validate_acp_agents(
        cls,
        config_data: Mapping[str, Mapping[str, object]] | None,
    ) -> dict[str, ACPAgentConfig]:
        if config_data is None:
            config_data = {}
        return {name: ACPAgentConfig(**cfg) for name, cfg in config_data.items()}

    @classmethod
    def _apply_singleton_configs(cls, config: Self, acp_agents: dict[str, ACPAgentConfig]) -> None:
        from deerflow.config.checkpointer_config import get_checkpointer_config

        previous_checkpointer_config = get_checkpointer_config()

        load_title_config_from_dict(config.title.model_dump())
        load_summarization_config_from_dict(config.summarization.model_dump())
        load_memory_config_from_dict(config.memory.model_dump())
        load_agents_api_config_from_dict(config.agents_api.model_dump())
        load_subagents_config_from_dict(config.subagents.model_dump())
        load_tool_search_config_from_dict(config.tool_search.model_dump())
        load_guardrails_config_from_dict(config.guardrails.model_dump())
        load_checkpointer_config_from_dict(config.checkpointer.model_dump() if config.checkpointer is not None else None)
        load_stream_bridge_config_from_dict(config.stream_bridge.model_dump() if config.stream_bridge is not None else None)
        load_acp_config_from_dict({name: agent.model_dump() for name, agent in acp_agents.items()})

        if previous_checkpointer_config != config.checkpointer:
            # These runtime singletons derive their backend from checkpointer config.
            # Keep imports local to avoid cycles: both providers import get_app_config.
            from deerflow.runtime.checkpointer import reset_checkpointer
            from deerflow.runtime.store import reset_store

            reset_checkpointer()
            reset_store()

    @classmethod
    def _apply_database_defaults(cls, config_data: dict[str, Any]) -> None:
        """Apply config.yaml defaults for persistence when the section is absent."""
        database_config = config_data.get("database")
        if database_config is None:
            database_config = {}
            config_data["database"] = database_config
        if not isinstance(database_config, dict):
            return
        for key, value in CONFIG_FILE_DATABASE_DEFAULTS.items():
            database_config.setdefault(key, value)

    @classmethod
    def _check_config_version(cls, config_data: dict, config_path: Path) -> None:
        """Check if the user's config.yaml is outdated compared to config.example.yaml.

        Emits a warning if the user's config_version is lower than the example's.
        Missing config_version is treated as version 0 (pre-versioning).
        """
        try:
            user_version = int(config_data.get("config_version", 0))
        except (TypeError, ValueError):
            user_version = 0

        # Find config.example.yaml by searching config.yaml's directory and its parents
        example_path = None
        search_dir = config_path.parent
        for _ in range(5):  # search up to 5 levels
            candidate = search_dir / "config.example.yaml"
            if candidate.exists():
                example_path = candidate
                break
            parent = search_dir.parent
            if parent == search_dir:
                break
            search_dir = parent
        if example_path is None:
            return

        try:
            with open(example_path, encoding="utf-8") as f:
                example_data = yaml.safe_load(f)
            raw = example_data.get("config_version", 0) if example_data else 0
            try:
                example_version = int(raw)
            except (TypeError, ValueError):
                example_version = 0
        except Exception:
            return

        if user_version < example_version:
            logger.warning(
                "Your config.yaml (version %d) is outdated — the latest version is %d. Run `make config-upgrade` to merge new fields into your config.",
                user_version,
                example_version,
            )

    @classmethod
    def resolve_env_variables(cls, config: Any) -> Any:
        """Recursively resolve environment variables in the config.

        Environment variables are resolved using the `os.getenv` function. Example: $OPENAI_API_KEY

        Args:
            config: The config to resolve environment variables in.

        Returns:
            The config with environment variables resolved.
        """
        if isinstance(config, str):
            if config.startswith("$"):
                env_value = os.getenv(config[1:])
                if env_value is None:
                    raise ValueError(f"Environment variable {config[1:]} not found for config value {config}")
                return env_value
            return config
        elif isinstance(config, dict):
            return {k: cls.resolve_env_variables(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [cls.resolve_env_variables(item) for item in config]
        return config

    def get_model_config(self, name: str) -> ModelConfig | None:
        """Get the model config by name.

        Args:
            name: The name of the model to get the config for.

        Returns:
            The model config if found, otherwise None.
        """
        return next((model for model in self.models if model.name == name), None)

    def get_tool_config(self, name: str) -> ToolConfig | None:
        """Get the tool config by name.

        Args:
            name: The name of the tool to get the config for.

        Returns:
            The tool config if found, otherwise None.
        """
        return next((tool for tool in self.tools if tool.name == name), None)

    def get_tool_group_config(self, name: str) -> ToolGroupConfig | None:
        """Get the tool group config by name.

        Args:
            name: The name of the tool group to get the config for.

        Returns:
            The tool group config if found, otherwise None.
        """
        return next((group for group in self.tool_groups if group.name == name), None)


# Compatibility singleton layer for code paths that have not yet been
# migrated to explicit ``AppConfig`` threading. New composition roots should
# prefer constructing ``AppConfig`` once and passing it down directly.
_app_config: AppConfig | None = None
_app_config_path: Path | None = None
_app_config_mtime: float | None = None
_app_config_is_custom = False
_current_app_config: ContextVar[AppConfig | None] = ContextVar("deerflow_current_app_config", default=None)
_current_app_config_stack: ContextVar[tuple[AppConfig | None, ...]] = ContextVar("deerflow_current_app_config_stack", default=())


def _get_config_mtime(config_path: Path) -> float | None:
    """Get the modification time of a config file if it exists."""
    try:
        return config_path.stat().st_mtime
    except OSError:
        return None


def _load_and_cache_app_config(config_path: str | None = None) -> AppConfig:
    """Load config from disk and refresh cache metadata."""
    global _app_config, _app_config_path, _app_config_mtime, _app_config_is_custom

    resolved_path = AppConfig.resolve_config_path(config_path)
    _app_config = AppConfig.from_file(str(resolved_path))
    _app_config_path = resolved_path
    _app_config_mtime = _get_config_mtime(resolved_path)
    _app_config_is_custom = False
    return _app_config


def get_app_config() -> AppConfig:
    """Get the DeerFlow config instance.

    Returns a cached singleton instance and automatically reloads it when the
    underlying config file path or modification time changes. Use
    `reload_app_config()` to force a reload, or `reset_app_config()` to clear
    the cache.
    """
    global _app_config, _app_config_path, _app_config_mtime

    runtime_override = _current_app_config.get()
    if runtime_override is not None:
        return runtime_override

    if _app_config is not None and _app_config_is_custom:
        return _app_config

    resolved_path = AppConfig.resolve_config_path()
    current_mtime = _get_config_mtime(resolved_path)

    should_reload = _app_config is None or _app_config_path != resolved_path or _app_config_mtime != current_mtime
    if should_reload:
        if _app_config_path == resolved_path and _app_config_mtime is not None and current_mtime is not None and _app_config_mtime != current_mtime:
            logger.info(
                "Config file has been modified (mtime: %s -> %s), reloading AppConfig",
                _app_config_mtime,
                current_mtime,
            )
        _load_and_cache_app_config(str(resolved_path))
    return _app_config


def reload_app_config(config_path: str | None = None) -> AppConfig:
    """Reload the config from file and update the cached instance.

    This is useful when the config file has been modified and you want
    to pick up the changes without restarting the application.

    Args:
        config_path: Optional path to config file. If not provided,
                     uses the default resolution strategy.

    Returns:
        The newly loaded AppConfig instance.
    """
    return _load_and_cache_app_config(config_path)


def reset_app_config() -> None:
    """Reset the cached config instance.

    This clears the singleton cache, causing the next call to
    `get_app_config()` to reload from file. Useful for testing
    or when switching between different configurations.
    """
    global _app_config, _app_config_path, _app_config_mtime, _app_config_is_custom
    _app_config = None
    _app_config_path = None
    _app_config_mtime = None
    _app_config_is_custom = False


def set_app_config(config: AppConfig) -> None:
    """Set a custom config instance.

    This allows injecting a custom or mock config for testing purposes.

    Args:
        config: The AppConfig instance to use.
    """
    global _app_config, _app_config_path, _app_config_mtime, _app_config_is_custom
    _app_config = config
    _app_config_path = None
    _app_config_mtime = None
    _app_config_is_custom = True


def peek_current_app_config() -> AppConfig | None:
    """Return the runtime-scoped AppConfig override, if one is active."""
    return _current_app_config.get()


def push_current_app_config(config: AppConfig) -> None:
    """Push a runtime-scoped AppConfig override for the current execution context."""
    stack = _current_app_config_stack.get()
    _current_app_config_stack.set(stack + (_current_app_config.get(),))
    _current_app_config.set(config)


def pop_current_app_config() -> None:
    """Pop the latest runtime-scoped AppConfig override for the current execution context."""
    stack = _current_app_config_stack.get()
    if not stack:
        _current_app_config.set(None)
        return
    previous = stack[-1]
    _current_app_config_stack.set(stack[:-1])
    _current_app_config.set(previous)
