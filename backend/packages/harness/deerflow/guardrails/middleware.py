"""GuardrailMiddleware - 在执行前根据 GuardrailProvider 评估工具调用。

本模块实现 LangGraph 代理中间件，在工具调用执行前进行安全防护评估:
- 集成 GuardrailProvider 进行工具调用授权
- 被拒绝的调用返回错误 ToolMessage，代理可以自适应调整
- 支持 fail-closed（失败时阻止）和 fail-open（失败时允许）模式
- 保留 LangGraph 控制流信号（中断/暂停/恢复）

工作流程:
1. 捕获工具调用请求
2. 构建 GuardrailRequest 对象
3. 调用提供者评估方法
4. 根据决策结果：允许执行或返回错误消息
5. 处理提供者异常（根据 fail_closed 配置）
"""

import logging
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.errors import GraphBubbleUp
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from deerflow.guardrails.provider import GuardrailDecision, GuardrailProvider, GuardrailReason, GuardrailRequest

logger = logging.getLogger(__name__)


class GuardrailMiddleware(AgentMiddleware[AgentState]):
    """在执行前根据 GuardrailProvider 评估工具调用。
    
    该中间件集成到 LangGraph 代理的工作流中，在每个工具调用
    执行前进行安全防护检查。
    
    被拒绝的调用返回错误 ToolMessage，代理可以据此调整策略。
    如果提供者抛出异常，行为取决于 fail_closed 配置:
      - True（默认）：阻止调用（fail-closed 模式）
      - False：允许通过并记录警告（fail-open 模式）
    
    Attributes:
        provider: 防护策略提供者实例
        fail_closed: 失败时是否阻止调用（默认 True）
        passport: 代理标识符，用于审计
    """

    def __init__(self, provider: GuardrailProvider, *, fail_closed: bool = True, passport: str | None = None):
        """初始化防护中间件。
        
        Args:
            provider: 防护策略提供者，负责评估工具调用
            fail_closed: 如果为 True，提供者错误时阻止调用；
                        如果为 False，提供者错误时允许调用
            passport: 代理标识符，可选，用于追踪和审计
            
        Note:
            - fail_closed=True 更安全，但可能影响可用性
            - fail_closed=False 更可用，但可能绕过安全防护
        """
        self.provider = provider
        self.fail_closed = fail_closed
        self.passport = passport

    def _build_request(self, request: ToolCallRequest) -> GuardrailRequest:
        """从工具调用请求构建防护请求。
        
        Args:
            request: LangGraph 的工具调用请求
            
        Returns:
            包含工具信息的 GuardrailRequest 对象
            
        Note:
            - 从 tool_call 字典中提取工具名称和参数
            - 使用 UTC 时间戳
            - agent_id 从 passport 配置中获取
        """
        return GuardrailRequest(
            tool_name=str(request.tool_call.get("name", "")),
            tool_input=request.tool_call.get("args", {}),
            agent_id=self.passport,
            timestamp=datetime.now(UTC).isoformat(),
        )

    def _build_denied_message(self, request: ToolCallRequest, decision: GuardrailDecision) -> ToolMessage:
        """构建被拒绝的工具调用错误消息。
        
        创建一个 ToolMessage 对象，告知代理工具调用被阻止，
        并提供拒绝原因，让代理可以选择替代方案。
        
        Args:
            request: 原始工具调用请求
            decision: 防护决策对象
            
        Returns:
            包含错误信息的 ToolMessage
            
        Note:
            - 消息包含工具名称、原因代码和详细描述
            - status 设置为 "error" 以标识失败
            - 建议代理选择替代方法
        """
        tool_name = str(request.tool_call.get("name", "unknown_tool"))
        tool_call_id = str(request.tool_call.get("id", "missing_id"))
        reason_text = decision.reasons[0].message if decision.reasons else "blocked by guardrail policy"
        reason_code = decision.reasons[0].code if decision.reasons else "oap.denied"
        return ToolMessage(
            content=f"Guardrail denied: tool '{tool_name}' was blocked ({reason_code}). Reason: {reason_text}. Choose an alternative approach.",
            tool_call_id=tool_call_id,
            name=tool_name,
            status="error",
        )

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """同步包装工具调用。
        
        在工具调用执行前进行防护评估。
        
        Args:
            request: 工具调用请求
            handler: 原始的工具调用处理器
            
        Returns:
            ToolMessage 或 Command 对象
            
        Note:
            - 如果决策允许，调用原始处理器
            - 如果决策拒绝，返回错误消息
            - GraphBubbleUp 异常会被重新抛出，保留 LangGraph 控制流
            - 其他异常根据 fail_closed 配置处理
        """
        gr = self._build_request(request)
        try:
            decision = self.provider.evaluate(gr)
        except GraphBubbleUp:
            # 保留 LangGraph 控制流信号（中断/暂停/恢复）
            raise
        except Exception:
            logger.exception("Guardrail provider error (sync)")
            if self.fail_closed:
                # 失败时阻止：返回拒绝决策
                decision = GuardrailDecision(
                    allow=False,
                    reasons=[GuardrailReason(
                        code="oap.evaluator_error",
                        message="guardrail provider error (fail-closed)"
                    )]
                )
            else:
                # 失败时允许：直接调用原始处理器
                return handler(request)
        if not decision.allow:
            logger.warning(
                "Guardrail denied: tool=%s policy=%s code=%s",
                gr.tool_name,
                decision.policy_id,
                decision.reasons[0].code if decision.reasons else "unknown"
            )
            return self._build_denied_message(request, decision)
        return handler(request)

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """异步包装工具调用。
        
        在工具调用执行前进行防护评估（异步版本）。
        
        Args:
            request: 工具调用请求
            handler: 原始的工具调用处理器（异步）
            
        Returns:
            ToolMessage 或 Command 对象
            
        Note:
            - 逻辑与同步版本相同
            - 使用提供者的 aevaluate 方法
            - 支持异步 I/O 操作（如远程策略引擎）
        """
        gr = self._build_request(request)
        try:
            decision = await self.provider.aevaluate(gr)
        except GraphBubbleUp:
            # 保留 LangGraph 控制流信号（中断/暂停/恢复）
            raise
        except Exception:
            logger.exception("Guardrail provider error (async)")
            if self.fail_closed:
                # 失败时阻止：返回拒绝决策
                decision = GuardrailDecision(
                    allow=False,
                    reasons=[GuardrailReason(
                        code="oap.evaluator_error",
                        message="guardrail provider error (fail-closed)"
                    )]
                )
            else:
                # 失败时允许：直接调用原始处理器
                return await handler(request)
        if not decision.allow:
            logger.warning(
                "Guardrail denied: tool=%s policy=%s code=%s",
                gr.tool_name,
                decision.policy_id,
                decision.reasons[0].code if decision.reasons else "unknown"
            )
            return self._build_denied_message(request, decision)
        return await handler(request)
