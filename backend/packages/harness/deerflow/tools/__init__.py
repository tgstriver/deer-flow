"""DeerFlow 工具模块 (Tools Package)

本模块是 DeerFlow 工具系统的入口，负责导出核心工具获取函数和技能管理工具。
工具系统为 Agent 提供各种可调用的能力，包括内置工具、MCP 工具、社区工具等。
"""

from .tools import get_available_tools

__all__ = ["get_available_tools", "skill_manage_tool"]


def __getattr__(name: str):
    """模块级属性访问钩子，实现 skill_manage_tool 的延迟导入。

    当访问 skill_manage_tool 时才进行导入，避免不必要的模块加载开销。
    这是一种常见的延迟导入 (lazy import) 模式。

    Args:
        name: 被访问的属性名称。

    Returns:
        对应的模块属性值。

    Raises:
        AttributeError: 当请求的属性名不存在时抛出。
    """
    if name == "skill_manage_tool":
        from .skill_manage_tool import skill_manage_tool

        return skill_manage_tool
    raise AttributeError(name)
