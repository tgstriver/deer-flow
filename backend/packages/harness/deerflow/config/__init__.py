"""DeerFlow 配置包（Configuration Package）。

本模块是 deerflow.config 包的入口，负责对外暴露配置相关的公共 API。
包括应用全局配置、扩展配置、循环检测、内存、技能、路径和追踪等
核心配置对象的获取函数和类型定义。

典型用法::

    from deerflow.config import get_app_config, get_paths

导入规范：
- 新代码应优先通过 ``AppConfig`` 实例直接传递配置，
  而非依赖此处的兼容性单例获取函数。
"""

from .app_config import get_app_config
from .extensions_config import ExtensionsConfig, get_extensions_config
from .loop_detection_config import LoopDetectionConfig
from .memory_config import MemoryConfig, get_memory_config
from .paths import Paths, get_paths
from .skill_evolution_config import SkillEvolutionConfig
from .skills_config import SkillsConfig
from .tracing_config import (
    get_enabled_tracing_providers,
    get_explicitly_enabled_tracing_providers,
    get_tracing_config,
    is_tracing_enabled,
    validate_enabled_tracing_providers,
)

__all__ = [
    "get_app_config",
    "SkillEvolutionConfig",
    "Paths",
    "get_paths",
    "SkillsConfig",
    "ExtensionsConfig",
    "get_extensions_config",
    "LoopDetectionConfig",
    "MemoryConfig",
    "get_memory_config",
    "get_tracing_config",
    "get_explicitly_enabled_tracing_providers",
    "get_enabled_tracing_providers",
    "is_tracing_enabled",
    "validate_enabled_tracing_providers",
]
