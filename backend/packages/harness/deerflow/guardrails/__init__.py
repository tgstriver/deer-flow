"""工具调用前授权中间件。

本模块提供工具调用前的安全防护机制,包括:
- 可插拔的防护策略提供者(GuardrailProvider)
- 内置的白名单/黑名单提供者(AllowlistProvider)
- LangGraph 代理中间件集成(GuardrailMiddleware)
- 结构化的决策和原因对象

使用场景:
- 限制某些工具的使用权限
- 实现工具调用的审计和日志
- 集成外部策略引擎进行动态授权
"""

from deerflow.guardrails.builtin import AllowlistProvider
from deerflow.guardrails.middleware import GuardrailMiddleware
from deerflow.guardrails.provider import GuardrailDecision, GuardrailProvider, GuardrailReason, GuardrailRequest

__all__ = [
    "AllowlistProvider",  # 内置白名单/黑名单提供者
    "GuardrailDecision",  # 防护决策对象
    "GuardrailMiddleware",  # LangGraph 中间件
    "GuardrailProvider",  # 提供者协议接口
    "GuardrailReason",  # 决策原因对象
    "GuardrailRequest",  # 防护请求对象
]
