"""DeerFlow Lead Agent 包。

本包实现了 DeerFlow 的主代理（Lead Agent），是系统的核心入口。
包含代理工厂函数、系统提示模板生成、技能缓存预热等组件。

模块概览：
- agent.py   — Lead Agent 工厂函数 make_lead_agent，LangGraph 图入口
- prompt.py  — 系统提示模板生成、技能缓存管理、子代理提示构建
"""

from .agent import make_lead_agent

__all__ = ["make_lead_agent"]
