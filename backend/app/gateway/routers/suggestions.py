"""后续问题建议生成 API。

本模块提供基于对话上下文自动生成后续问题的功能，帮助用户继续对话。
使用 LLM 分析最近的对话历史，生成相关的简短问题建议。

主要特性：
- 基于对话上下文智能生成问题
- 支持自定义生成数量（1-5个）
- 自动检测用户语言并保持一致
- 容错处理：解析失败时返回空列表
- 支持模型覆盖配置
"""

import json
import logging

from fastapi import APIRouter, Depends, Request
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from app.gateway.authz import require_permission
from app.gateway.deps import get_config
from deerflow.config.app_config import AppConfig
from deerflow.models import create_chat_model

logger = logging.getLogger(__name__)

# 创建路由器，所有端点以 /api 为前缀
router = APIRouter(prefix="/api", tags=["suggestions"])


class SuggestionMessage(BaseModel):
    """建议请求中的单条消息模型。

    Attributes:
        role: 消息角色，必须是 'user' 或 'assistant'
        content: 消息的纯文本内容
    """

    role: str = Field(..., description="Message role: user|assistant")
    content: str = Field(..., description="Message content as plain text")


class SuggestionsRequest(BaseModel):
    """生成建议的请求体模型。

    Attributes:
        messages: 最近的对话消息列表，用于提供上下文
        n: 要生成的建议数量（默认3，范围1-5）
        model_name: 可选的模型名称覆盖，不指定则使用默认模型
    """

    messages: list[SuggestionMessage] = Field(..., description="Recent conversation messages")
    n: int = Field(default=3, ge=1, le=5, description="Number of suggestions to generate")
    model_name: str | None = Field(default=None, description="Optional model override")


class SuggestionsResponse(BaseModel):
    """生成建议的响应模型。

    Attributes:
        suggestions: 建议的后续问题列表
    """

    suggestions: list[str] = Field(default_factory=list, description="Suggested follow-up questions")


def _strip_markdown_code_fence(text: str) -> str:
    """移除 Markdown 代码围栏标记（```）。

    LLM 有时会将 JSON 输出包裹在 Markdown 代码块中，此函数将其剥离。

    Args:
        text: 可能包含代码围栏的文本

    Returns:
        移除代码围栏后的文本
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) >= 3 and lines[0].startswith("```") and lines[-1].startswith("```"):
        return "\n".join(lines[1:-1]).strip()
    return stripped


def _parse_json_string_list(text: str) -> list[str] | None:
    """从文本中解析 JSON 字符串数组。

    尝试从 LLM 的响应中提取 JSON 数组，支持以下情况：
    - 纯 JSON 数组
    - 包裹在 Markdown 代码块中的 JSON
    - JSON 前后有其他文本（通过查找 [ 和 ] 定位）

    Args:
        text: 待解析的文本

    Returns:
        解析后的字符串列表，如果解析失败则返回 None

    Note:
        - 只保留非空字符串项
        - 忽略非字符串类型的数组元素
    """
    # 先移除可能的 Markdown 代码围栏
    candidate = _strip_markdown_code_fence(text)
    # 查找 JSON 数组的起始和结束位置
    start = candidate.find("[")
    end = candidate.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    # 提取数组部分
    candidate = candidate[start : end + 1]
    try:
        data = json.loads(candidate)
    except Exception:
        return None
    if not isinstance(data, list):
        return None
    # 过滤和清理结果
    out: list[str] = []
    for item in data:
        if not isinstance(item, str):
            continue
        s = item.strip()
        if not s:
            continue
        out.append(s)
    return out


def _extract_response_text(content: object) -> str:
    """从 LLM 响应内容中提取纯文本。

    处理不同格式的响应内容：
    - 字符串：直接返回
    - 列表：遍历内容块，提取文本类型的内容
    - None：返回空字符串
    - 其他类型：转换为字符串

    Args:
        content: LLM 响应的内容对象

    Returns:
        提取的纯文本字符串

    Note:
        支持 LangChain 的消息格式，包括结构化内容块（如 {type: "text", text: "..."}）
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in {"text", "output_text"}:
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts) if parts else ""
    if content is None:
        return ""
    return str(content)


def _format_conversation(messages: list[SuggestionMessage]) -> str:
    """将消息列表格式化为可读的对话文本。

    将结构化的消息对象转换为自然语言的对话格式，便于 LLM 理解上下文。

    Args:
        messages: 消息列表

    Returns:
        格式化后的对话文本，每行格式为 "Role: content"
    """
    parts: list[str] = []
    for m in messages:
        role = m.role.strip().lower()
        if role in ("user", "human"):
            parts.append(f"User: {m.content.strip()}")
        elif role in ("assistant", "ai"):
            parts.append(f"Assistant: {m.content.strip()}")
        else:
            parts.append(f"{m.role}: {m.content.strip()}")
    return "\n".join(parts).strip()


@router.post(
    "/threads/{thread_id}/suggestions",
    response_model=SuggestionsResponse,
    summary="生成后续问题建议",
    description="基于最近的对话上下文，生成用户可能询问的简短后续问题。",
)
@require_permission("threads", "read", owner_check=True)
async def generate_suggestions(
    thread_id: str,
    body: SuggestionsRequest,
    request: Request,
    config: AppConfig = Depends(get_config),
) -> SuggestionsResponse:
    """生成后续问题建议。

    基于提供的对话历史，使用 LLM 生成相关的后续问题，帮助用户继续对话。

    **工作流程：**
    1. 验证输入：检查是否有消息和有效的对话内容
    2. 格式化对话：将消息列表转换为自然语言格式
    3. 构建提示词：创建系统指令和用户内容
    4. 调用 LLM：使用配置的模型生成建议
    5. 解析响应：从 LLM 输出中提取 JSON 数组
    6. 清理结果：移除换行符、限制数量
    7. 错误处理：任何异常都返回空列表

    **LLM 提示词要求：**
    - 生成指定数量的简短问题
    - 问题必须与对话相关
    - 使用与用户相同的语言
    - 每个问题简洁（<= 20个英文单词或 <= 40个中文字符）
    - 不包含编号、Markdown 或额外文本
    - 输出必须是纯 JSON 字符串数组

    Args:
        thread_id: 线程 ID（用于权限验证）
        body: 建议请求体，包含消息列表和配置
        request: FastAPI 请求对象
        config: 应用配置（依赖注入）

    Returns:
        建议响应，包含生成的问题列表

    Raises:
        HTTPException: 如果权限验证失败（由装饰器处理）

    Permissions:
        需要 'threads.read' 权限，且必须是线程所有者

    Note:
        - 如果消息为空或解析失败，返回空列表而非错误
        - 禁用 thinking 模式以提高响应速度
        - 运行名称设置为 "suggest_agent" 便于追踪
    """
    # 如果没有消息，直接返回空列表
    if not body.messages:
        return SuggestionsResponse(suggestions=[])

    n = body.n
    # 格式化对话为自然语言
    conversation = _format_conversation(body.messages)
    if not conversation:
        return SuggestionsResponse(suggestions=[])

    # 构建系统指令，明确告知 LLM 的输出格式和要求
    system_instruction = (
        "You are generating follow-up questions to help the user continue the conversation.\n"
        f"Based on the conversation below, produce EXACTLY {n} short questions the user might ask next.\n"
        "Requirements:\n"
        "- Questions must be relevant to the preceding conversation.\n"
        "- Questions must be written in the same language as the user.\n"
        "- Keep each question concise (ideally <= 20 words / <= 40 Chinese characters).\n"
        "- Do NOT include numbering, markdown, or any extra text.\n"
        "- Output MUST be a JSON array of strings only.\n"
    )
    # 构建用户内容，包含对话上下文
    user_content = f"Conversation Context:\n{conversation}\n\nGenerate {n} follow-up questions"

    try:
        # 创建聊天模型（禁用 thinking 模式以提高速度）
        model = create_chat_model(name=body.model_name, thinking_enabled=False, app_config=config)
        # 调用 LLM 生成建议
        response = await model.ainvoke([SystemMessage(content=system_instruction), HumanMessage(content=user_content)], config={"run_name": "suggest_agent"})
        # 提取响应文本
        raw = _extract_response_text(response.content)
        # 解析 JSON 数组
        suggestions = _parse_json_string_list(raw) or []
        # 清理结果：移除换行符、过滤空字符串、限制数量
        cleaned = [s.replace("\n", " ").strip() for s in suggestions if s.strip()]
        cleaned = cleaned[:n]
        return SuggestionsResponse(suggestions=cleaned)
    except Exception as exc:
        # 记录错误但返回空列表，保证 API 的稳定性
        logger.exception("Failed to generate suggestions: thread_id=%s err=%s", thread_id, exc)
        return SuggestionsResponse(suggestions=[])
