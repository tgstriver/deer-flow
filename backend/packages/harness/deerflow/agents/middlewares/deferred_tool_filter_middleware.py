"""延迟工具过滤中间件 —— 在模型绑定阶段过滤掉延迟工具的 schema。

背景：什么是「延迟工具」(Deferred Tool)?
  当 tool_search（工具搜索）功能启用时，MCP 工具不会直接暴露给 LLM，而是先注册到
  DeferredToolRegistry（延迟工具注册表）中。这些工具的 schema 不会被发送给模型的
  bind_tools 调用——这正是"延迟"的含义：在初始阶段节省上下文 token。

  LLM 在运行时通过 tool_search 工具按需搜索和发现延迟工具，被搜索到的工具才会被
  "提升"(promote)为可见工具，此后 LLM 才能调用它们。

为什么需要过滤?
  1. 节省上下文窗口：将大量 MCP 工具 schema 全部发送给 LLM 会消耗宝贵的 token 额度。
  2. 按需加载：LLM 只在需要时才搜索和发现相关工具，避免无关工具干扰推理。
  3. 执行路由保留：ToolNode 仍然持有所有工具（包括延迟工具）的执行路由能力，
     只是 LLM 看不到延迟工具的 schema 而已。

本中间件的拦截点:
  - wrap_model_call / awrap_model_call：在模型调用前，从 request.tools 中移除
    延迟工具的 schema，确保 model.bind_tools 只接收活跃工具。
  - wrap_tool_call / awrap_tool_call：在工具调用前，阻止对尚未提升的延迟工具的
    直接调用，返回错误提示信息，引导 LLM 先调用 tool_search 提升该工具。
"""

import logging
from collections.abc import Awaitable, Callable
from typing import override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

logger = logging.getLogger(__name__)


class DeferredToolFilterMiddleware(AgentMiddleware[AgentState]):
    """延迟工具过滤中间件：在模型绑定前移除延迟工具 schema，并拦截对未提升延迟工具的调用。

    工作机制:
      - ToolNode 仍然持有全部工具（包括延迟工具）的执行路由，确保运行时工具调用
        能被正确分发。
      - LLM 只能看到活跃工具的 schema —— 延迟工具通过 tool_search 在运行时
        按需发现和提升。
      - 如果 LLM 试图直接调用一个尚未提升的延迟工具，中间件会拦截该调用并返回
        错误信息，提示先通过 tool_search 提升该工具。
    """

    def _filter_tools(self, request: ModelRequest) -> ModelRequest:
        """从模型请求的工具列表中过滤掉延迟工具的 schema。

        步骤:
          1. 获取延迟工具注册表（DeferredToolRegistry）。
          2. 若注册表不存在或为空，说明没有延迟工具，直接返回原请求。
          3. 获取所有延迟工具的名称集合。
          4. 从 request.tools 中筛除名称在延迟集合中的工具，保留活跃工具。
          5. 若确实过滤掉了工具，记录调试日志。
          6. 返回更新了工具列表的新请求对象。

        Args:
            request: 原始的模型请求，包含待绑定的工具列表。

        Returns:
            过滤后的模型请求，仅包含活跃工具的 schema。
        """
        # 延迟导入，避免循环依赖：tool_search 模块本身可能依赖中间件链
        from deerflow.tools.builtins.tool_search import get_deferred_registry

        # 获取延迟工具注册表实例
        registry = get_deferred_registry()
        if not registry:
            # 注册表不存在，无需过滤
            return request

        # 获取所有延迟工具的名称集合
        deferred_names = registry.deferred_names

        # 过滤：只保留不在延迟名称集合中的工具（即活跃工具）
        # 使用 getattr 安全获取工具名称，防止工具对象没有 name 属性时出错
        active_tools = [t for t in request.tools if getattr(t, "name", None) not in deferred_names]

        # 如果过滤后工具数量减少，说明确实移除了延迟工具，记录调试信息
        if len(active_tools) < len(request.tools):
            logger.debug(f"Filtered {len(request.tools) - len(active_tools)} deferred tool schema(s) from model binding")

        # 返回新的请求对象，用过滤后的工具列表替换原始列表
        return request.override(tools=active_tools)

    def _blocked_tool_message(self, request: ToolCallRequest) -> ToolMessage | None:
        """检查工具调用请求是否针对尚未提升的延迟工具，若是则返回错误 ToolMessage。

        当 LLM 试图直接调用一个延迟工具时（例如通过猜测工具名称），此方法会
        拦截该调用并返回错误信息，引导 LLM 先使用 tool_search 发现和提升该工具。

        步骤:
          1. 获取延迟工具注册表，若不存在则无需拦截。
          2. 从 tool_call 字典中提取工具名称，若名称为空则放行。
          3. 检查该工具名称是否在延迟注册表中，若不在则放行。
          4. 构造错误 ToolMessage，提示需要先调用 tool_search 提升工具。

        Args:
            request: 工具调用请求，包含 tool_call 字典（含 name 和 id）。

        Returns:
            若该工具是尚未提升的延迟工具，返回错误 ToolMessage；
            否则返回 None，表示不需要拦截，交给后续处理。
        """
        # 延迟导入，避免循环依赖
        from deerflow.tools.builtins.tool_search import get_deferred_registry

        # 获取延迟工具注册表实例
        registry = get_deferred_registry()
        if not registry:
            # 注册表不存在，无需拦截
            return None

        # 从 tool_call 字典中安全提取工具名称
        tool_name = str(request.tool_call.get("name") or "")
        if not tool_name:
            # 工具名称为空，无法判断，放行
            return None

        # 检查该工具是否属于延迟工具
        if not registry.contains(tool_name):
            # 不是延迟工具，放行
            return None

        # 构造错误 ToolMessage：
        # - content: 错误提示，告知 LLM 该工具是延迟的、尚未提升，需要先调用 tool_search
        # - tool_call_id: 关联到原始工具调用的 ID，确保消息配对正确
        # - name: 被拦截的工具名称
        # - status: 标记为错误状态
        tool_call_id = str(request.tool_call.get("id") or "missing_tool_call_id")
        return ToolMessage(
            content=(f"Error: Tool '{tool_name}' is deferred and has not been promoted yet. Call tool_search first to expose and promote this tool's schema, then retry."),
            tool_call_id=tool_call_id,
            name=tool_name,
            status="error",
        )

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """同步拦截模型调用：在传递给下游 handler 前过滤掉延迟工具 schema。

        调用链: 本方法 -> _filter_tools 过滤 -> handler 执行实际模型调用。
        """
        return handler(self._filter_tools(request))

    @override
    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
    ) -> ToolMessage | Command:
        """同步拦截工具调用：若调用的是未提升的延迟工具，返回错误消息；否则放行。

        调用链: 本方法 -> 检查是否被拦截 -> 被拦截则返回错误 ToolMessage，
                否则交给 handler 执行实际工具调用。
        """
        blocked = self._blocked_tool_message(request)
        if blocked is not None:
            # 被拦截：返回错误提示，阻止对未提升延迟工具的调用
            return blocked
        # 未被拦截：放行，执行实际工具调用
        return handler(request)

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """异步拦截模型调用：在传递给下游 handler 前过滤掉延迟工具 schema。

        与 wrap_model_call 逻辑相同，仅异步版本。
        """
        return await handler(self._filter_tools(request))

    @override
    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command]],
    ) -> ToolMessage | Command:
        """异步拦截工具调用：若调用的是未提升的延迟工具，返回错误消息；否则放行。

        与 wrap_tool_call 逻辑相同，仅异步版本。
        """
        blocked = self._blocked_tool_message(request)
        if blocked is not None:
            # 被拦截：返回错误提示，阻止对未提升延迟工具的调用
            return blocked
        # 未被拦截：放行，执行实际工具调用
        return await handler(request)
