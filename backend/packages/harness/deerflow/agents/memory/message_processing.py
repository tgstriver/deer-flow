"""共享辅助函数，将对话转换为记忆更新输入。

本模块提供消息处理工具:
- 从消息内容中提取纯文本
- 过滤消息，仅保留用户输入和最终助手响应
- 检测用户纠正信号(如"不对"、"你理解错了")
- 检测正面强化信号(如"完全正确"、"正是我想要的")
- 清理上传文件标记，避免将临时文件路径存储到长期记忆
"""

from __future__ import annotations

import re
from copy import copy
from typing import Any

# 匹配上传文件块的正则表达式
_UPLOAD_BLOCK_RE = re.compile(r"<uploaded_files>[\s\S]*?</uploaded_files>\n*", re.IGNORECASE)
# 用户纠正信号模式(中英文)
_CORRECTION_PATTERNS = (
    re.compile(r"\bthat(?:'s| is) (?:wrong|incorrect)\b", re.IGNORECASE),  # 英文: "that's wrong"
    re.compile(r"\byou misunderstood\b", re.IGNORECASE),  # 英文: "you misunderstood"
    re.compile(r"\btry again\b", re.IGNORECASE),  # 英文: "try again"
    re.compile(r"\bredo\b", re.IGNORECASE),  # 英文: "redo"
    re.compile(r"不对"),  # 中文
    re.compile(r"你理解错了"),  # 中文
    re.compile(r"你理解有误"),  # 中文
    re.compile(r"重试"),  # 中文
    re.compile(r"重新来"),  # 中文
    re.compile(r"换一种"),  # 中文
    re.compile(r"改用"),  # 中文
)
# 正面强化信号模式(中英文)
_REINFORCEMENT_PATTERNS = (
    re.compile(r"\byes[,.]?\s+(?:exactly|perfect|that(?:'s| is) (?:right|correct|it))\b", re.IGNORECASE),
    re.compile(r"\bperfect(?:[.!?]|$)", re.IGNORECASE),
    re.compile(r"\bexactly\s+(?:right|correct)\b", re.IGNORECASE),
    re.compile(r"\bthat(?:'s| is)\s+(?:exactly\s+)?(?:right|correct|what i (?:wanted|needed|meant))\b", re.IGNORECASE),
    re.compile(r"\bkeep\s+(?:doing\s+)?that\b", re.IGNORECASE),
    re.compile(r"\bjust\s+(?:like\s+)?(?:that|this)\b", re.IGNORECASE),
    re.compile(r"\bthis is (?:great|helpful)\b(?:[.!?]|$)", re.IGNORECASE),
    re.compile(r"\bthis is what i wanted\b(?:[.!?]|$)", re.IGNORECASE),
    re.compile(r"对[，,]?\s*就是这样(?:[。！？!?.]|$)"),  # 中文
    re.compile(r"完全正确(?:[。！？!?.]|$)"),  # 中文
    re.compile(r"(?:对[，,]?\s*)?就是这个意思(?:[。！？!?.]|$)"),  # 中文
    re.compile(r"正是我想要的(?:[。！？!?.]|$)"),  # 中文
    re.compile(r"继续保持(?:[。！？!?.]|$)"),  # 中文
)


def extract_message_text(message: Any) -> str:
    """从消息内容中提取纯文本，用于过滤和信号检测。
    
    处理不同格式的消息内容:
    - 字符串: 直接返回
    - 列表: 遍历内容块，提取文本部分(支持多模态消息)
    - 其他类型: 转换为字符串
    
    Args:
        message: 消息对象，可以是 LangChain 消息或字典
        
    Returns:
        提取的纯文本字符串
        
    Note:
        支持结构化内容块格式，如 {"type": "text", "text": "..."}
    """
    content = getattr(message, "content", "")
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict):
                text_val = part.get("text")
                if isinstance(text_val, str):
                    text_parts.append(text_val)
        return " ".join(text_parts)
    return str(content)


def filter_messages_for_memory(messages: list[Any]) -> list[Any]:
    """仅保留用户输入和最终助手响应，用于记忆更新。
    
    过滤策略:
    - 保留所有 human 类型消息(用户输入)
    - 仅保留没有工具调用的 ai 类型消息(最终响应)
    - 跳过仅包含上传文件标记的消息
    - 跳过工具调用后的中间 AI 响应
    
    Args:
        messages: 原始消息列表
        
    Returns:
        过滤后的消息列表，适合用于记忆更新
        
    Note:
        - 使用 skip_next_ai 标记跳过上传文件后的 AI 响应
        - 工具调用的中间响应不会存储到记忆
    """
    filtered = []
    skip_next_ai = False
    for msg in messages:
        msg_type = getattr(msg, "type", None)

        if msg_type == "human":
            content_str = extract_message_text(msg)
            if "<uploaded_files>" in content_str:
                # 移除上传文件标记
                stripped = _UPLOAD_BLOCK_RE.sub("", content_str).strip()
                if not stripped:
                    # 如果只有上传文件标记，跳过此消息和下一个 AI 响应
                    skip_next_ai = True
                    continue
                # 保留清理后的消息
                clean_msg = copy(msg)
                clean_msg.content = stripped
                filtered.append(clean_msg)
                skip_next_ai = False
            else:
                filtered.append(msg)
                skip_next_ai = False
        elif msg_type == "ai":
            # 仅保留没有工具调用的最终响应
            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                if skip_next_ai:
                    # 跳过上传文件后的 AI 响应
                    skip_next_ai = False
                    continue
                filtered.append(msg)

    return filtered


def detect_correction(messages: list[Any]) -> bool:
    """在最近对话中检测明确的用户纠正信号。
    
    检查最近 6 条消息中的用户输入，查找纠正模式:
    - 英文: "that's wrong", "you misunderstood", "try again"
    - 中文: "不对", "你理解错了", "重试"
    
    Args:
        messages: 对话消息列表
        
    Returns:
        如果检测到纠正信号则返回 True
        
    Note:
        纠正信号用于提高事实置信度(category="correction", confidence >= 0.95)
    """
    # 检查最近 6 条消息中的用户输入
    recent_user_msgs = [msg for msg in messages[-6:] if getattr(msg, "type", None) == "human"]

    for msg in recent_user_msgs:
        content = extract_message_text(msg).strip()
        if content and any(pattern.search(content) for pattern in _CORRECTION_PATTERNS):
            return True

    return False


def detect_reinforcement(messages: list[Any]) -> bool:
    """在最近对话中检测明确的正面强化信号。
    
    检查最近 6 条消息中的用户输入，查找正面强化模式:
    - 英文: "perfect", "exactly right", "this is great"
    - 中文: "完全正确", "正是我想要的", "继续保持"
    
    Args:
        messages: 对话消息列表
        
    Returns:
        如果检测到正面强化信号则返回 True
        
    Note:
        正面强化信号用于提高偏好/行为事实的置信度
        如果同时检测到纠正信号，则忽略强化信号
    """
    # 检查最近 6 条消息中的用户输入
    recent_user_msgs = [msg for msg in messages[-6:] if getattr(msg, "type", None) == "human"]

    for msg in recent_user_msgs:
        content = extract_message_text(msg).strip()
        if content and any(pattern.search(content) for pattern in _REINFORCEMENT_PATTERNS):
            return True

    return False
