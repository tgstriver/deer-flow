"""DeerFlow 内置的防护策略提供者。

本模块提供开箱即用的防护策略实现:
- AllowlistProvider: 基于白名单/黑名单的简单工具访问控制
- 无外部依赖，轻量级实现
- 支持同步和异步评估
"""

from deerflow.guardrails.provider import GuardrailDecision, GuardrailReason, GuardrailRequest


class AllowlistProvider:
    """简单的白名单/黑名单提供者。
    
    通过配置允许或拒绝的工具列表来控制工具访问权限。
    评估逻辑:
    1. 如果设置了白名单，工具必须在白名单中
    2. 如果工具在黑名单中，直接拒绝
    3. 其他情况允许
    
    Attributes:
        name: 提供者名称，用于标识和日志
        _allowed: 允许的工具名称集合，None 表示不限制
        _denied: 拒绝的工具名称集合
    """

    name = "allowlist"

    def __init__(self, *, allowed_tools: list[str] | None = None, denied_tools: list[str] | None = None):
        """初始化白名单/黑名单提供者。
        
        Args:
            allowed_tools: 允许的工具名称列表，如果为 None 则不限制
            denied_tools: 拒绝的工具名称列表，默认为空集合
            
        Note:
            - 白名单和黑名单可以同时设置，白名单优先级更高
            - 工具名称必须完全匹配
        """
        self._allowed = set(allowed_tools) if allowed_tools else None
        self._denied = set(denied_tools) if denied_tools else set()

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        """评估工具调用是否应该被允许。
        
        Args:
            request: 包含工具名称和输入的防护请求
            
        Returns:
            防护决策对象，包含允许/拒绝决定和原因
            
        Note:
            评估顺序:
            1. 检查白名单：如果设置了白名单且工具不在其中，拒绝
            2. 检查黑名单：如果工具在黑名单中，拒绝
            3. 其他情况：允许
        """
        # 检查白名单
        if self._allowed is not None and request.tool_name not in self._allowed:
            return GuardrailDecision(
                allow=False,
                reasons=[GuardrailReason(
                    code="oap.tool_not_allowed",
                    message=f"tool '{request.tool_name}' not in allowlist"
                )]
            )
        # 检查黑名单
        if request.tool_name in self._denied:
            return GuardrailDecision(
                allow=False,
                reasons=[GuardrailReason(
                    code="oap.tool_not_allowed",
                    message=f"tool '{request.tool_name}' is denied"
                )]
            )
        # 允许
        return GuardrailDecision(allow=True, reasons=[GuardrailReason(code="oap.allowed")])

    async def aevaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        """异步评估工具调用。
        
        Args:
            request: 包含工具名称和输入的防护请求
            
        Returns:
            防护决策对象
            
        Note:
            由于 AllowlistProvider 无 I/O 操作，直接调用同步版本
        """
        return self.evaluate(request)
