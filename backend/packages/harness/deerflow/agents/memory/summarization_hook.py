"""在摘要移除消息前触发的钩子。

本模块实现与摘要中间件的集成:
- memory_flush_hook: 在消息被摘要前将其刷新到记忆队列
- 检测对话中的纠正和强化信号
- 过滤消息仅保留相关内容用于记忆更新

工作流程:
1. 摘要中间件在移除旧消息前调用此钩子
2. 过滤消息(仅保留用户输入和最终 AI 响应)
3. 检测纠正/强化信号
4. 将过滤后的消息添加到记忆队列(立即处理)
"""

from __future__ import annotations

from deerflow.agents.memory.message_processing import detect_correction, detect_reinforcement, filter_messages_for_memory
from deerflow.agents.memory.queue import get_memory_queue
from deerflow.agents.middlewares.summarization_middleware import SummarizationEvent
from deerflow.config.memory_config import get_memory_config
from deerflow.runtime.user_context import resolve_runtime_user_id


def memory_flush_hook(event: SummarizationEvent) -> None:
    """将即将被摘要的消息刷新到记忆队列。
    
    此钩子在摘要中间件移除旧消息前调用，确保重要对话
    内容被保存到长期记忆中。
    
    Args:
        event: 摘要事件，包含待摘要的消息
        
    Note:
        - 仅当记忆功能启用且有 thread_id 时处理
        - 需要同时有用户和助手消息
        - 使用 add_nowait 立即开始处理
        - 纠正信号优先级高于强化信号
    """
    # 检查记忆功能是否启用
    if not get_memory_config().enabled or not event.thread_id:
        return

    # 过滤消息，仅保留用于记忆更新的内容
    filtered_messages = filter_messages_for_memory(list(event.messages_to_summarize))
    user_messages = [message for message in filtered_messages if getattr(message, "type", None) == "human"]
    assistant_messages = [message for message in filtered_messages if getattr(message, "type", None) == "ai"]
    # 需要同时有用户和助手消息
    if not user_messages or not assistant_messages:
        return

    # 检测纠正和强化信号(纠正优先)
    correction_detected = detect_correction(filtered_messages)
    reinforcement_detected = not correction_detected and detect_reinforcement(filtered_messages)
    
    # 解析用户 ID 并添加到队列
    user_id = resolve_runtime_user_id(event.runtime)
    queue = get_memory_queue()
    queue.add_nowait(
        thread_id=event.thread_id,
        messages=filtered_messages,
        agent_name=event.agent_name,
        user_id=user_id,
        correction_detected=correction_detected,
        reinforcement_detected=reinforcement_detected,
    )
