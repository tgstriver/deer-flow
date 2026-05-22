"""子代理并发限制中间件。

本模块实现了对单次模型响应中并发的子代理（subagent）工具调用数量的硬性限制。
当大语言模型在一次回复中生成超过最大并发数的 "task" 工具调用时，
该中间件会截断多余的调用，仅保留前 max_concurrent 个，丢弃其余部分。
这种方式比基于提示词（prompt）的限制更加可靠，因为 LLM 并不总是严格遵守
提示词中关于并发数量的约束。

该中间件在 Agent 中间件链中排在第 16 位（可选，仅当 subagent_enabled 时启用），
作用于模型响应之后（after_model / aafter_model）的阶段。
"""

import logging
from typing import override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langgraph.runtime import Runtime

from deerflow.agents.middlewares.tool_call_metadata import clone_ai_message_with_tool_calls
from deerflow.subagents.executor import MAX_CONCURRENT_SUBAGENTS

logger = logging.getLogger(__name__)

# 子代理并发数的有效范围：最小 2，最大 4
# 该范围经过权衡：太小会过度限制并行能力，太大会导致资源争用和超时风险增加
MIN_SUBAGENT_LIMIT = 2
MAX_SUBAGENT_LIMIT = 4


def _clamp_subagent_limit(value: int) -> int:
    """将子代理并发限制值钳制到有效范围 [2, 4] 内。

    如果传入的值小于 MIN_SUBAGENT_LIMIT（2），则返回 2；
    如果传入的值大于 MAX_SUBAGENT_LIMIT（4），则返回 4；
    否则返回原始值。

    Args:
        value: 期望的最大并发子代理数量。

    Returns:
        被钳制到 [2, 4] 范围内的整数值。
    """
    return max(MIN_SUBAGENT_LIMIT, min(MAX_SUBAGENT_LIMIT, value))


class SubagentLimitMiddleware(AgentMiddleware[AgentState]):
    """截断模型响应中多余的 "task" 工具调用，强制执行最大并发子代理数限制。

    当 LLM 在一次响应中生成超过 max_concurrent 个并行的 "task" 工具调用时，
    该中间件仅保留前 max_concurrent 个调用，丢弃超出的部分。
    这比依赖提示词来限制并发数量更加可靠，因为模型并不总是遵守提示词约束。

    工作原理：
        1. 在模型响应之后（after_model）检查最新消息中的工具调用列表
        2. 筛选出名称为 "task" 的工具调用，统计其数量
        3. 若 "task" 调用数量超过限制，则保留前 max_concurrent 个，截断剩余的
        4. 用截断后的工具调用列表替换原始 AIMessage

    Args:
        max_concurrent: 允许的最大并发子代理调用数量。
            默认值为 MAX_CONCURRENT_SUBAGENTS（3）。
            传入的值会被钳制到 [2, 4] 的有效范围内。
    """

    def __init__(self, max_concurrent: int = MAX_CONCURRENT_SUBAGENTS):
        super().__init__()
        # 对传入的并发限制值进行钳制，确保其落在 [2, 4] 的有效范围内
        self.max_concurrent = _clamp_subagent_limit(max_concurrent)

    def _truncate_task_calls(self, state: AgentState) -> dict | None:
        """检查并截断最新的 AIMessage 中多余的 "task" 工具调用。

        处理流程：
            1. 从状态中获取消息列表，若无消息则返回 None
            2. 检查最新消息是否为 AI 消息（type == "ai"），若不是则返回 None
            3. 检查该消息是否包含工具调用，若无则返回 None
            4. 统计名称为 "task" 的工具调用的索引位置
            5. 若 "task" 调用数量未超过限制，则无需截断，返回 None
            6. 若超过限制，构建需要丢弃的工具调用索引集合（超出限制的部分），
               过滤掉这些索引对应的工具调用，生成截断后的列表
            7. 使用 clone_ai_message_with_tool_calls 创建替换后的 AIMessage，
               返回包含更新消息的字典

        Args:
            state: 当前 Agent 状态，包含消息列表等信息。

        Returns:
            若需要截断，返回 {"messages": [updated_msg]} 用于状态更新；
            若无需截断，返回 None。
        """
        messages = state.get("messages", [])
        if not messages:
            return None

        # 取最新的消息
        last_msg = messages[-1]
        # 只处理 AI 消息（即模型生成的回复），其他类型（如 ToolMessage）跳过
        if getattr(last_msg, "type", None) != "ai":
            return None

        tool_calls = getattr(last_msg, "tool_calls", None)
        if not tool_calls:
            return None

        # 收集所有名称为 "task" 的工具调用的索引位置
        # "task" 是子代理委派的工具名称，每个 task 调用代表一个子代理任务
        task_indices = [i for i, tc in enumerate(tool_calls) if tc.get("name") == "task"]
        if len(task_indices) <= self.max_concurrent:
            # "task" 调用数量在允许范围内，无需截断
            return None

        # 构建需要丢弃的工具调用索引集合：
        # 保留前 max_concurrent 个 "task" 调用，丢弃索引在 task_indices[max_concurrent:] 中的调用
        # 例如：max_concurrent=3，task_indices=[0,2,4,5,7]，则丢弃索引 {5, 7} 对应的调用
        indices_to_drop = set(task_indices[self.max_concurrent :])
        # 从工具调用列表中过滤掉被丢弃的索引，保留其余调用（包括非 "task" 类型的调用）
        truncated_tool_calls = [tc for i, tc in enumerate(tool_calls) if i not in indices_to_drop]

        dropped_count = len(indices_to_drop)
        logger.warning(f"Truncated {dropped_count} excess task tool call(s) from model response (limit: {self.max_concurrent})")

        # 使用克隆函数创建新的 AIMessage，保留原始消息的 id 等元数据，
        # 但用截断后的工具调用列表替换原始的 tool_calls。
        # 相同 id 会触发 LangGraph 的消息替换机制，而非追加新消息。
        updated_msg = clone_ai_message_with_tool_calls(last_msg, truncated_tool_calls)
        return {"messages": [updated_msg]}

    @override
    def after_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        """同步版本的模型后置钩子，在模型生成响应后执行截断逻辑。

        Args:
            state: 当前 Agent 状态。
            runtime: LangGraph 运行时实例。

        Returns:
            截断结果字典或 None。
        """
        return self._truncate_task_calls(state)

    @override
    async def aafter_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        """异步版本的模型后置钩子，在模型生成响应后执行截断逻辑。

        由于截断逻辑本身是纯 CPU 操作（不涉及 I/O），同步和异步版本
        共享同一个 _truncate_task_calls 实现。

        Args:
            state: 当前 Agent 状态。
            runtime: LangGraph 运行时实例。

        Returns:
            截断结果字典或 None。
        """
        return self._truncate_task_calls(state)
