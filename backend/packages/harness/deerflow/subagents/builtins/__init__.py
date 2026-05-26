"""内置子代理配置模块 —— 注册系统预定义的子代理。

内置子代理包括：
- general-purpose: 通用多步骤任务代理，继承父代理的所有工具
- bash: 命令执行专家代理，仅允许沙箱相关工具

Built-in subagent configurations.
"""

from .bash_agent import BASH_AGENT_CONFIG
from .general_purpose import GENERAL_PURPOSE_CONFIG

__all__ = [
    "GENERAL_PURPOSE_CONFIG",
    "BASH_AGENT_CONFIG",
]

# 内置子代理注册表 / Registry of built-in subagents
BUILTIN_SUBAGENTS = {
    "general-purpose": GENERAL_PURPOSE_CONFIG,
    "bash": BASH_AGENT_CONFIG,
}
