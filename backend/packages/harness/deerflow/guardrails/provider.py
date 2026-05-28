"""GuardrailProvider 协议和数据结构，用于工具调用前授权。

本模块定义了工具调用防护的核心接口和数据模型:
- GuardrailRequest: 防护请求上下文
- GuardrailReason: 决策原因对象
- GuardrailDecision: 防护决策结果
- GuardrailProvider: 可插拔提供者的协议接口

设计理念:
- 使用 Protocol 而非基类，实现更灵活的接口契约
- 支持同步和异步评估
- 与 OAP (Open Authorization Policy) 标准对齐
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class GuardrailRequest:
    """传递给提供者的工具调用上下文。
    
    包含评估工具调用是否应该被允许所需的所有信息。
    
    Attributes:
        tool_name: 要调用的工具名称
        tool_input: 工具的输入参数
        agent_id: 代理标识符（可选，用于区分不同代理）
        thread_id: 线程标识符（可选，用于审计和追踪）
        is_subagent: 是否为子代理调用
        timestamp: 请求时间戳（ISO 格式）
    """

    tool_name: str
    tool_input: dict[str, Any]
    agent_id: str | None = None
    thread_id: str | None = None
    is_subagent: bool = False
    timestamp: str = ""


@dataclass
class GuardrailReason:
    """允许/拒绝决策的结构化原因（OAP 原因对象）。
    
    提供机器可读的原因代码和人类可读的描述信息。
    
    Attributes:
        code: 原因代码，用于程序化处理（如 "oap.tool_not_allowed"）
        message: 人类可读的详细描述（可选）
        
    Note:
        原因代码遵循命名空间约定，如 "oap.*" 表示系统原因
    """

    code: str
    message: str = ""


@dataclass
class GuardrailDecision:
    """提供者的允许/拒绝判决（与 OAP Decision 对象对齐）。
    
    包含防护策略的评估结果和相关元数据。
    
    Attributes:
        allow: 是否允许工具调用
        reasons: 决策原因列表，支持多个原因
        policy_id: 应用的策略标识符（可选）
        metadata: 附加元数据，用于审计和调试
        
    Note:
        - allow=True 时，reasons 通常包含允许的原因
        - allow=False 时，reasons 必须包含至少一个拒绝原因
        - metadata 可以包含策略版本、评估时间等信息
    """

    allow: bool
    reasons: list[GuardrailReason] = field(default_factory=list)
    policy_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class GuardrailProvider(Protocol):
    """可插拔工具调用授权的契约。
    
    定义防护策略提供者必须实现的接口。
    任何具有这些方法的类都可以作为提供者，无需继承基类。
    
    提供者通过类路径加载（使用 resolve_variable()），
    与 DeerFlow 加载模型、工具和沙箱的机制相同。
    
    Attributes:
        name: 提供者名称，用于标识和日志
        
    Methods:
        evaluate: 同步评估工具调用
        aevaluate: 异步评估工具调用
        
    Example:
        >>> class MyCustomProvider:
        ...     name = "my_custom"
        ...     
        ...     def evaluate(self, request):
        ...         # 自定义逻辑
        ...         return GuardrailDecision(allow=True)
        ...     
        ...     async def aevaluate(self, request):
        ...         return self.evaluate(request)
    """

    name: str

    def evaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        """评估工具调用是否应该继续。
        
        Args:
            request: 包含工具名称和输入的防护请求
            
        Returns:
            防护决策对象
        """
        ...

    async def aevaluate(self, request: GuardrailRequest) -> GuardrailDecision:
        """异步版本的评估方法。
        
        Args:
            request: 包含工具名称和输入的防护请求
            
        Returns:
            防护决策对象
            
        Note:
            如果提供者没有异步 I/O 操作，可以直接调用 evaluate()
        """
        ...
