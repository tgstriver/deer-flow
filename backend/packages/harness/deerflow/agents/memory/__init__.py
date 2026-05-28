"""DeerFlow 的记忆模块。

本模块提供全局记忆机制,包括:
- 在 memory.json 中存储用户上下文和对话历史
- 使用 LLM 总结和提取对话中的事实
- 将相关记忆注入系统提示以提供个性化响应

核心组件:
- storage: 记忆存储提供者(支持文件和自定义存储)
- queue: 带防抖机制的记忆更新队列
- updater: 基于 LLM 的记忆更新器
- prompt: 提示词模板和格式化工具
- summarization_hook: 与摘要中间件集成的钩子
- message_processing: 消息处理和信号检测
"""

from deerflow.agents.memory.prompt import (
    FACT_EXTRACTION_PROMPT,
    MEMORY_UPDATE_PROMPT,
    format_conversation_for_update,
    format_memory_for_injection,
)
from deerflow.agents.memory.queue import (
    ConversationContext,
    MemoryUpdateQueue,
    get_memory_queue,
    reset_memory_queue,
)
from deerflow.agents.memory.storage import (
    FileMemoryStorage,
    MemoryStorage,
    get_memory_storage,
)
from deerflow.agents.memory.updater import (
    MemoryUpdater,
    clear_memory_data,
    delete_memory_fact,
    get_memory_data,
    reload_memory_data,
    update_memory_from_conversation,
)

__all__ = [
    # 提示词工具
    "MEMORY_UPDATE_PROMPT",  # 记忆更新提示词模板
    "FACT_EXTRACTION_PROMPT",  # 事实提取提示词模板
    "format_memory_for_injection",  # 格式化记忆用于注入
    "format_conversation_for_update",  # 格式化对话用于更新
    # 队列
    "ConversationContext",  # 对话上下文
    "MemoryUpdateQueue",  # 记忆更新队列
    "get_memory_queue",  # 获取全局队列实例
    "reset_memory_queue",  # 重置队列(测试用)
    # 存储
    "MemoryStorage",  # 存储抽象基类
    "FileMemoryStorage",  # 文件存储实现
    "get_memory_storage",  # 获取配置的存储实例
    # 更新器
    "MemoryUpdater",  # 记忆更新器
    "clear_memory_data",  # 清空记忆数据
    "delete_memory_fact",  # 删除记忆事实
    "get_memory_data",  # 获取记忆数据
    "reload_memory_data",  # 重新加载记忆数据
    "update_memory_from_conversation",  # 从对话更新记忆的便捷函数
]
