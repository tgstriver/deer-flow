"""子代理（Subagent）模块 —— 提供子代理的配置、注册与执行能力。

子代理是 DeerFlow 系统中将复杂任务委派给专门代理的机制。
本模块对外暴露核心类和函数：
- SubagentConfig: 子代理配置定义
- SubagentExecutor: 子代理执行引擎
- SubagentResult: 子代理执行结果
- get_available_subagent_names: 获取当前运行时可用的子代理名称列表
- get_subagent_config: 按名称获取子代理配置（含 config.yaml 覆盖）
- list_subagents: 列出所有已注册的子代理配置
"""

from .config import SubagentConfig
from .executor import SubagentExecutor, SubagentResult
from .registry import get_available_subagent_names, get_subagent_config, list_subagents

__all__ = [
    "SubagentConfig",
    "SubagentExecutor",
    "SubagentResult",
    "get_available_subagent_names",
    "get_subagent_config",
    "list_subagents",
]
