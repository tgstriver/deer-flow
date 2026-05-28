"""修补版 ChatDeepSeek，在多轮对话中保留 reasoning_content。

本模块提供了一个修补版的 ChatDeepSeek，用于正确处理消息回传给 API 时的 reasoning_content。
原始实现将 reasoning_content 存储在 additional_kwargs 中，但在后续 API 调用时没有将其包含进去，
这会导致在启用思考模式时，需要所有助手消息都包含 reasoning_content 的 API 出现错误。

核心问题：
    - DeepSeek API 在启用 thinking/reasoning 模式时，要求所有助手消息都必须包含 reasoning_content
    - LangChain 的原始 ChatDeepSeek 实现只接收 reasoning_content，但不会在后续请求中发送它
    - 这导致多轮对话时 API 返回错误

解决方案：
    - 重写 _get_request_payload 方法
    - 从 original_messages 中提取 reasoning_content
    - 注入到 payload 的对应助手消息中
"""

from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage
from langchain_deepseek import ChatDeepSeek


class PatchedChatDeepSeek(ChatDeepSeek):
    """支持正确保存 reasoning_content 的 ChatDeepSeek。
    
    当使用启用思考/推理功能的模型时，API 期望在多轮对话中的**所有**助手消息
    都包含 reasoning_content。这个修补版本确保来自 additional_kwargs 的
    reasoning_content 被包含在请求负载中。
    
    继承自：
        ChatDeepSeek: LangChain 的 DeepSeek 聊天模型实现
        
    主要改进：
        - 重写 _get_request_payload 方法
        - 自动提取并注入 reasoning_content
        - 支持两种匹配策略（位置匹配和计数匹配）
        
    使用场景：
        - 需要启用 thinking 模式的 DeepSeek 模型
        - 多轮对话场景
        - 需要完整保留推理过程的对话历史
        
    Example:
        >>> from deerflow.models.patched_deepseek import PatchedChatDeepSeek
        >>> model = PatchedChatDeepSeek(model="deepseek-chat", enable_thinking=True)
        >>> response = model.invoke([HumanMessage(content="你好")])
        >>> # reasoning_content 会被自动保存到 additional_kwargs
        >>> # 下一轮对话时会自动包含在请求中
    """

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """检查此类是否可序列化。
        
        Returns:
            True，表示此类支持 LangChain 序列化协议
            
        Note:
            - 这是 LangChain 的标准接口方法
            - 用于模型配置的持久化和传输
        """
        return True

    @property
    def lc_secrets(self) -> dict[str, str]:
        """定义需要保密的环境变量映射。
        
        Returns:
            字典，键为参数名，值为环境变量名
            
        Note:
            - api_key 和 openai_api_key 都映射到 DEEPSEEK_API_KEY
            - 这是为了兼容 OpenAI 风格的 API 密钥命名
            - LangChain 会自动从环境变量中读取这些值
        """
        return {"api_key": "DEEPSEEK_API_KEY", "openai_api_key": "DEEPSEEK_API_KEY"}

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        """获取请求负载，同时保留 reasoning_content。
        
        重写父类方法，将 additional_kwargs 中的 reasoning_content 注入到
        负载中的助手消息里。
        
        Args:
            input_: 语言模型输入，可以是单条消息、消息列表或字符串
            stop: 停止词序列列表，遇到这些词时停止生成
            **kwargs: 其他关键字参数，传递给父类方法
            
        Returns:
            完整的请求负载字典，包含处理后的消息列表
            
        Note:
            - 首先获取原始消息列表（转换前）
            - 调用父类方法获取基础负载
            - 通过两种策略匹配并注入 reasoning_content：
              1. 位置匹配：如果消息数量相同，按位置一一对应
              2. 计数匹配：如果数量不同，按助手消息的顺序匹配
            - 只处理 role='assistant' 且 additional_kwargs 中有 reasoning_content 的消息
            
        Raises:
            无显式异常抛出，但可能传播父类的异常
            
        Example:
            原始消息结构：
                AIMessage(
                    content="答案",
                    additional_kwargs={"reasoning_content": "推理过程"}
                )
            
            处理后负载：
                {
                    "messages": [
                        {"role": "assistant", "content": "答案", "reasoning_content": "推理过程"}
                    ]
                }
        """
        # 获取转换前的原始消息列表
        original_messages = self._convert_input(input_).to_messages()

        # 调用父类方法获取基础负载
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)

        # 匹配负载消息与原始消息以恢复 reasoning_content
        payload_messages = payload.get("messages", [])

        # 负载消息和原始消息应该保持相同的顺序
        # 遍历两者并按位置匹配
        if len(payload_messages) == len(original_messages):
            # 策略1：位置匹配（优先）
            for payload_msg, orig_msg in zip(payload_messages, original_messages):
                if payload_msg.get("role") == "assistant" and isinstance(orig_msg, AIMessage):
                    # 从原始消息的 additional_kwargs 中提取 reasoning_content
                    reasoning_content = orig_msg.additional_kwargs.get("reasoning_content")
                    if reasoning_content is not None:
                        # 注入到负载消息中
                        payload_msg["reasoning_content"] = reasoning_content
        else:
            # 策略2：计数匹配（降级方案）
            # 当消息数量不一致时（例如系统消息被过滤），按助手消息的顺序匹配
            ai_messages = [m for m in original_messages if isinstance(m, AIMessage)]
            assistant_payloads = [(i, m) for i, m in enumerate(payload_messages) if m.get("role") == "assistant"]

            for (idx, payload_msg), ai_msg in zip(assistant_payloads, ai_messages):
                reasoning_content = ai_msg.additional_kwargs.get("reasoning_content")
                if reasoning_content is not None:
                    payload_messages[idx]["reasoning_content"] = reasoning_content

        return payload
