"""模型提供者模块（Models Package）

本模块是 DeerFlow 模型层的入口，负责提供统一的聊天模型创建接口。
通过工厂模式，根据配置文件动态实例化不同的大语言模型提供商，
包括 OpenAI、Claude、vLLM、DeepSeek、MiniMax、MindIE、Codex 等。

主要导出:
    create_chat_model: 根据名称和配置创建聊天模型实例的工厂函数
"""

from .factory import create_chat_model

__all__ = ["create_chat_model"]
