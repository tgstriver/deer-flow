"""记忆更新器，用于读取、写入和更新记忆数据。

本模块提供基于 LLM 的记忆更新能力:
- MemoryUpdater: 核心更新器类，使用 LLM 分析对话并更新记忆
- 事实管理: 创建、更新、删除记忆事实
- 数据导入/导出: 支持记忆数据的导入和清空
- 同步/异步支持: 提供 sync 和 async 两种更新路径

架构设计:
- 使用线程池执行同步 LLM 调用，避免事件循环冲突
- 深拷贝保护缓存数据不被污染
- 上传事件清理: 移除关于文件上传的临时信息
- 事实去重: 基于内容规范化避免重复事实
- 事实限制: 按置信度排序并保留 top N

更新流程:
1. 加载当前记忆
2. 格式化对话为更新提示词
3. 调用 LLM 生成更新
4. 解析 JSON 响应
5. 应用更新(用户上下文、历史、事实)
6. 保存前再次清理上传文件相关噪音
7. 持久化到存储
"""

import asyncio
import atexit
import concurrent.futures
import copy
import json
import logging
import math
import re
import uuid
from typing import Any

from deerflow.agents.memory.prompt import (
    MEMORY_UPDATE_PROMPT,
    format_conversation_for_update,
)
from deerflow.agents.memory.storage import (
    create_empty_memory,
    get_memory_storage,
    utc_now_iso_z,
)
from deerflow.config.memory_config import get_memory_config
from deerflow.models import create_chat_model

logger = logging.getLogger(__name__)


# Thread pool for offloading sync memory updates when called from an async
# context.  Unlike the previous asyncio.run() approach, this runs *sync*
# model.invoke() calls — no event loop is created, so the langchain async
# httpx client pool (globally cached via @lru_cache) is never touched and
# cross-loop connection reuse is impossible.
_SYNC_MEMORY_UPDATER_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=4,
    thread_name_prefix="memory-updater-sync",
)
atexit.register(lambda: _SYNC_MEMORY_UPDATER_EXECUTOR.shutdown(wait=False))


def _create_empty_memory() -> dict[str, Any]:
    """Backward-compatible wrapper around the storage-layer empty-memory factory."""
    return create_empty_memory()


def _save_memory_to_file(memory_data: dict[str, Any], agent_name: str | None = None, *, user_id: str | None = None) -> bool:
    """Backward-compatible wrapper around the configured memory storage save path."""
    return get_memory_storage().save(memory_data, agent_name, user_id=user_id)


def get_memory_data(agent_name: str | None = None, *, user_id: str | None = None) -> dict[str, Any]:
    """Get the current memory data via storage provider."""
    return get_memory_storage().load(agent_name, user_id=user_id)


def reload_memory_data(agent_name: str | None = None, *, user_id: str | None = None) -> dict[str, Any]:
    """Reload memory data via storage provider."""
    return get_memory_storage().reload(agent_name, user_id=user_id)


def import_memory_data(memory_data: dict[str, Any], agent_name: str | None = None, *, user_id: str | None = None) -> dict[str, Any]:
    """通过存储提供者持久化导入的记忆数据。
    
    将完整的记忆负载保存到存储中，支持按代理和用户进行隔离。
    
    Args:
        memory_data: 要持久化的完整记忆负载
        agent_name: 如果提供，导入到特定代理的记忆中
        user_id: 如果提供，将记忆限定到特定用户
        
    Returns:
        存储标准化后的已保存记忆数据
        
    Raises:
        OSError: 如果持久化导入的记忆失败
        
    Note:
        - 会覆盖现有的记忆数据
        - 保存后会重新加载以确保一致性
    """
    storage = get_memory_storage()
    if not storage.save(memory_data, agent_name, user_id=user_id):
        raise OSError("Failed to save imported memory data")
    return storage.load(agent_name, user_id=user_id)


def clear_memory_data(agent_name: str | None = None, *, user_id: str | None = None) -> dict[str, Any]:
    """清除所有存储的记忆数据并持久化空结构。
    
    用空的记忆结构替换现有的记忆数据，实现记忆重置功能。
    
    Args:
        agent_name: 代理名称(可选)，如果提供则清除特定代理的记忆
        user_id: 用户ID(可选)，如果提供则清除特定用户的数据
        
    Returns:
        清空后的空记忆数据
        
    Raises:
        OSError: 如果保存清空的记忆数据失败
        
    Note:
        - 彻底清除记忆，不可逆操作
        - 保留记忆结构的基本框架
    """
    cleared_memory = create_empty_memory()
    if not _save_memory_to_file(cleared_memory, agent_name, user_id=user_id):
        raise OSError("Failed to save cleared memory data")
    return cleared_memory


def _validate_confidence(confidence: float) -> float:
    """验证持久化的事实置信度，确保存储的JSON保持标准兼容性。
    
    检查置信度值是否在有效范围内 [0, 1]，并验证其为有限数值。
    
    Args:
        confidence: 要验证的置信度值
        
    Returns:
        验证后的置信度值
        
    Raises:
        ValueError: 如果置信度值无效
        
    Note:
        - 置信度必须是有限数值
        - 置信度必须在 [0, 1] 范围内
    """
    if not math.isfinite(confidence) or confidence < 0 or confidence > 1:
        raise ValueError("confidence")
    return confidence


def create_memory_fact(
    content: str,
    category: str = "context",
    confidence: float = 0.5,
    agent_name: str | None = None,
    *,
    user_id: str | None = None,
) -> dict[str, Any]:
    """创建新事实并持久化更新的记忆数据。
    
    向记忆中添加新的事实条目，并将其持久化到存储中。
    
    Args:
        content: 事实内容
        category: 事实类别，默认为 "context"
        confidence: 事实置信度，默认为 0.5
        agent_name: 代理名称(可选)
        user_id: 用户ID(可选)
        
    Returns:
        添加事实后的更新记忆数据
        
    Raises:
        ValueError: 如果内容为空
        OSError: 如果保存记忆数据失败
        
    Note:
        - 自动生成唯一的事实ID
        - 记录创建时间和来源
        - 验证置信度的有效性
    """
    normalized_content = content.strip()
    if not normalized_content:
        raise ValueError("content")

    normalized_category = category.strip() or "context"
    validated_confidence = _validate_confidence(confidence)
    now = utc_now_iso_z()
    memory_data = get_memory_data(agent_name, user_id=user_id)
    updated_memory = dict(memory_data)
    facts = list(memory_data.get("facts", []))
    facts.append(
        {
            "id": f"fact_{uuid.uuid4().hex[:8]}",
            "content": normalized_content,
            "category": normalized_category,
            "confidence": validated_confidence,
            "createdAt": now,
            "source": "manual",
        }
    )
    updated_memory["facts"] = facts

    if not _save_memory_to_file(updated_memory, agent_name, user_id=user_id):
        raise OSError("Failed to save memory data after creating fact")

    return updated_memory


def delete_memory_fact(fact_id: str, agent_name: str | None = None, *, user_id: str | None = None) -> dict[str, Any]:
    """根据ID删除事实并持久化更新的记忆数据。
    
    从记忆中移除指定ID的事实，并将更改持久化到存储中。
    
    Args:
        fact_id: 要删除的事实ID
        agent_name: 代理名称(可选)
        user_id: 用户ID(可选)
        
    Returns:
        删除事实后的更新记忆数据
        
    Raises:
        KeyError: 如果找不到指定ID的事实
        OSError: 如果保存记忆数据失败
        
    Note:
        - 事实ID必须精确匹配
        - 删除操作不可逆
    """
    memory_data = get_memory_data(agent_name, user_id=user_id)
    facts = memory_data.get("facts", [])
    updated_facts = [fact for fact in facts if fact.get("id") != fact_id]
    if len(updated_facts) == len(facts):
        raise KeyError(fact_id)

    updated_memory = dict(memory_data)
    updated_memory["facts"] = updated_facts

    if not _save_memory_to_file(updated_memory, agent_name, user_id=user_id):
        raise OSError(f"Failed to save memory data after deleting fact '{fact_id}'")

    return updated_memory


def update_memory_fact(
    fact_id: str,
    content: str | None = None,
    category: str | None = None,
    confidence: float | None = None,
    agent_name: str | None = None,
    *,
    user_id: str | None = None,
) -> dict[str, Any]:
    """更新现有事实并持久化更新的记忆数据。
    
    修改记忆中指定ID事实的属性，并将更改持久化到存储中。
    
    Args:
        fact_id: 要更新的事实ID
        content: 新的事实内容(可选)
        category: 新的事实类别(可选)
        confidence: 新的事实置信度(可选)
        agent_name: 代理名称(可选)
        user_id: 用户ID(可选)
        
    Returns:
        更新事实后的记忆数据
        
    Raises:
        KeyError: 如果找不到指定ID的事实
        ValueError: 如果提供的内容为空
        OSError: 如果保存记忆数据失败
        
    Note:
        - 只更新提供的参数，其他属性保持不变
        - 至少提供一个要更新的属性
        - 验证新值的有效性
    """
    memory_data = get_memory_data(agent_name, user_id=user_id)
    updated_memory = dict(memory_data)
    updated_facts: list[dict[str, Any]] = []
    found = False

    for fact in memory_data.get("facts", []):
        if fact.get("id") == fact_id:
            found = True
            updated_fact = dict(fact)
            if content is not None:
                normalized_content = content.strip()
                if not normalized_content:
                    raise ValueError("content")
                updated_fact["content"] = normalized_content
            if category is not None:
                updated_fact["category"] = category.strip() or "context"
            if confidence is not None:
                updated_fact["confidence"] = _validate_confidence(confidence)
            updated_facts.append(updated_fact)
        else:
            updated_facts.append(fact)

    if not found:
        raise KeyError(fact_id)

    updated_memory["facts"] = updated_facts

    if not _save_memory_to_file(updated_memory, agent_name, user_id=user_id):
        raise OSError(f"Failed to save memory data after updating fact '{fact_id}'")

    return updated_memory


def _extract_text(content: Any) -> str:
    """Extract plain text from LLM response content (str or list of content blocks).

    Modern LLMs may return structured content as a list of blocks instead of a
    plain string, e.g. [{"type": "text", "text": "..."}]. Using str() on such
    content produces Python repr instead of the actual text, breaking JSON
    parsing downstream.

    String chunks are concatenated without separators to avoid corrupting
    chunked JSON/text payloads. Dict-based text blocks are treated as full text
    blocks and joined with newlines for readability.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces: list[str] = []
        pending_str_parts: list[str] = []

        def flush_pending_str_parts() -> None:
            if pending_str_parts:
                pieces.append("".join(pending_str_parts))
                pending_str_parts.clear()

        for block in content:
            if isinstance(block, str):
                pending_str_parts.append(block)
            elif isinstance(block, dict):
                flush_pending_str_parts()
                text_val = block.get("text")
                if isinstance(text_val, str):
                    pieces.append(text_val)

        flush_pending_str_parts()
        return "\n".join(pieces)
    return str(content)


# Matches sentences that describe a file-upload *event* rather than general
# file-related work.  Deliberately narrow to avoid removing legitimate facts
# such as "User works with CSV files" or "prefers PDF export".
_UPLOAD_SENTENCE_RE = re.compile(
    r"[^.!?]*\b(?:"
    r"upload(?:ed|ing)?(?:\s+\w+){0,3}\s+(?:file|files?|document|documents?|attachment|attachments?)"
    r"|file\s+upload"
    r"|/mnt/user-data/uploads/"
    r"|<uploaded_files>"
    r")[^.!?]*[.!?]?\s*",
    re.IGNORECASE,
)


def _strip_upload_mentions_from_memory(memory_data: dict[str, Any]) -> dict[str, Any]:
    """Remove sentences about file uploads from all memory summaries and facts.

    Uploaded files are session-scoped; persisting upload events in long-term
    memory causes the agent to search for non-existent files in future sessions.
    """
    # Scrub summaries in user/history sections
    for section in ("user", "history"):
        section_data = memory_data.get(section, {})
        for _key, val in section_data.items():
            if isinstance(val, dict) and "summary" in val:
                cleaned = _UPLOAD_SENTENCE_RE.sub("", val["summary"]).strip()
                cleaned = re.sub(r"  +", " ", cleaned)
                val["summary"] = cleaned

    # Also remove any facts that describe upload events
    facts = memory_data.get("facts", [])
    if facts:
        memory_data["facts"] = [f for f in facts if not _UPLOAD_SENTENCE_RE.search(f.get("content", ""))]

    return memory_data


def _fact_content_key(content: Any) -> str | None:
    if not isinstance(content, str):
        return None
    stripped = content.strip()
    if not stripped:
        return None
    return stripped.casefold()


class MemoryUpdater:
    """基于LLM和对话上下文更新记忆。
    
    核心功能类，使用LLM分析对话并智能更新记忆数据。
    
    主要特性:
    - 支持同步和异步更新
    - 自动检测纠正和强化信号
    - 智能事实提取和管理
    - 上传事件清理
    - 事实去重和限制
    
    Attributes:
        _model_name: 用于记忆更新的模型名称
    """

    def __init__(self, model_name: str | None = None):
        """初始化记忆更新器。
        
        Args:
            model_name: 可选的模型名称，如果为None则使用配置或默认值
            
        Note:
            - 模型会在每次更新时重新获取
            - thinking_enabled=False 以提高性能
        """
        self._model_name = model_name

    def _get_model(self):
        """获取用于记忆更新的模型。
        
        Returns:
            配置好的聊天模型实例
            
        Note:
            - 使用配置的模型名称
            - 禁用思考模式以提高响应速度
        """
        config = get_memory_config()
        model_name = self._model_name or config.model_name
        return create_chat_model(name=model_name, thinking_enabled=False)

    def _build_correction_hint(
        self,
        correction_detected: bool,
        reinforcement_detected: bool,
    ) -> str:
        """构建纠正和强化信号的提示词提示。
        
        根据检测到的信号类型生成相应的提示词增强内容。
        
        Args:
            correction_detected: 是否检测到纠正信号
            reinforcement_detected: 是否检测到强化信号
            
        Returns:
            用于增强提示词的字符串
            
        Note:
            - 纠正信号优先级更高
            - 强化信号仅在未检测到纠正信号时使用
        """
        correction_hint = ""
        if correction_detected:
            correction_hint = (
                "IMPORTANT: Explicit correction signals were detected in this conversation. "
                "Pay special attention to what the agent got wrong, what the user corrected, "
                "and record the correct approach as a fact with category "
                '"correction" and confidence >= 0.95 when appropriate.'
            )
        if reinforcement_detected:
            reinforcement_hint = (
                "IMPORTANT: Positive reinforcement signals were detected in this conversation. "
                "The user explicitly confirmed the agent's approach was correct or helpful. "
                "Record the confirmed approach, style, or preference as a fact with category "
                '"preference" or "behavior" and confidence >= 0.9 when appropriate.'
            )
            correction_hint = (correction_hint + "\n" + reinforcement_hint).strip() if correction_hint else reinforcement_hint

        return correction_hint

    def _prepare_update_prompt(
        self,
        messages: list[Any],
        agent_name: str | None,
        correction_detected: bool,
        reinforcement_detected: bool,
        user_id: str | None = None,
    ) -> tuple[dict[str, Any], str] | None:
        """加载记忆并为对话构建更新提示词。
        
        此方法整合当前记忆、对话历史和信号提示，准备LLM的输入。
        
        Args:
            messages: 对话消息列表
            agent_name: 代理名称
            correction_detected: 是否检测到纠正信号
            reinforcement_detected: 是否检测到强化信号
            user_id: 用户ID
            
        Returns:
            包含当前记忆和提示词的元组，如果不需要更新则返回None
            
        Note:
            - 检查记忆功能是否启用
            - 验证消息列表非空
            - 格式化对话为适合LLM处理的文本
        """
        config = get_memory_config()
        if not config.enabled or not messages:
            return None

        current_memory = get_memory_data(agent_name, user_id=user_id)
        conversation_text = format_conversation_for_update(messages)
        if not conversation_text.strip():
            return None

        correction_hint = self._build_correction_hint(
            correction_detected=correction_detected,
            reinforcement_detected=reinforcement_detected,
        )
        prompt = MEMORY_UPDATE_PROMPT.format(
            current_memory=json.dumps(current_memory, indent=2),
            conversation=conversation_text,
            correction_hint=correction_hint,
        )
        return current_memory, prompt

    def _finalize_update(
        self,
        current_memory: dict[str, Any],
        response_content: Any,
        thread_id: str | None,
        agent_name: str | None,
        user_id: str | None = None,
    ) -> bool:
        """解析模型响应，应用更新并持久化记忆。
        
        此方法负责解析LLM的JSON响应，应用更新到记忆数据，
        清理上传提及，并将结果持久化到存储。
        
        Args:
            current_memory: 当前记忆数据
            response_content: 模型响应内容
            thread_id: 线程ID
            agent_name: 代理名称
            user_id: 用户ID
            
        Returns:
            保存成功返回True，否则返回False
            
        Note:
            - 使用深拷贝防止缓存污染
            - 在保存前再次清理上传提及
            - 处理可能的JSON格式问题
        """
        response_text = _extract_text(response_content).strip()

        if response_text.startswith("```"):
            lines = response_text.split("\n")
            response_text = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        update_data = json.loads(response_text)
        # 深拷贝以防止后续保存失败时损坏仍缓存的原始对象引用
        updated_memory = self._apply_updates(copy.deepcopy(current_memory), update_data, thread_id)
        updated_memory = _strip_upload_mentions_from_memory(updated_memory)
        return get_memory_storage().save(updated_memory, agent_name, user_id=user_id)

    async def aupdate_memory(
        self,
        messages: list[Any],
        thread_id: str | None = None,
        agent_name: str | None = None,
        correction_detected: bool = False,
        reinforcement_detected: bool = False,
        user_id: str | None = None,
    ) -> bool:
        """异步更新记忆，委托给同步路径。
        
        使用 ``asyncio.to_thread`` 在工作线程中运行 *同步* ``model.invoke()`` 路径
        这样就不会创建第二个事件循环，langchain异步httpx客户端池
        (与主代理共享) 将永远不会被触及。这消除了跨循环连接复用错误
        如issue #2615所述。
        
        Args:
            messages: 对话消息列表
            thread_id: 线程ID
            agent_name: 代理名称
            correction_detected: 最近轮次是否包含明确的纠正信号
            reinforcement_detected: 最近轮次是否包含积极强化信号
            user_id: 用户ID
            
        Returns:
            更新成功返回True，否则返回False
            
        Note:
            - 通过线程池避免事件循环冲突
            - 保持与同步版本相同的逻辑
            - 适用于在异步上下文中调用
        """
        return await asyncio.to_thread(
            self._do_update_memory_sync,
            messages=messages,
            thread_id=thread_id,
            agent_name=agent_name,
            correction_detected=correction_detected,
            reinforcement_detected=reinforcement_detected,
            user_id=user_id,
        )

    def _do_update_memory_sync(
        self,
        messages: list[Any],
        thread_id: str | None = None,
        agent_name: str | None = None,
        correction_detected: bool = False,
        reinforcement_detected: bool = False,
        user_id: str | None = None,
    ) -> bool:
        """使用 ``model.invoke()`` 进行纯同步记忆更新。
        
        使用 *同步* LLM调用路径，因此不会创建事件循环。这保证了
        langchain提供者的全局缓存异步httpx ``AsyncClient`` / 连接池
        (与主代理共享的那个) 永远不会被触及——不可能出现跨循环连接复用。
        
        Args:
            messages: 对话消息列表
            thread_id: 线程ID
            agent_name: 代理名称
            correction_detected: 最近轮次是否包含明确的纠正信号
            reinforcement_detected: 最近轮次是否包含积极强化信号
            user_id: 用户ID
            
        Returns:
            更新成功返回True，否则返回False
            
        Note:
            - 仅使用同步LLM调用
            - 避免事件循环冲突
            - 处理JSON解析和通用异常
        """
        try:
            prepared = self._prepare_update_prompt(
                messages=messages,
                agent_name=agent_name,
                correction_detected=correction_detected,
                reinforcement_detected=reinforcement_detected,
                user_id=user_id,
            )
            if prepared is None:
                return False

            current_memory, prompt = prepared
            model = self._get_model()
            response = model.invoke(prompt, config={"run_name": "memory_agent"})
            return self._finalize_update(
                current_memory=current_memory,
                response_content=response.content,
                thread_id=thread_id,
                agent_name=agent_name,
                user_id=user_id,
            )
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse LLM response for memory update: %s", e)
            return False
        except Exception as e:
            logger.exception("Memory update failed: %s", e)
            return False

    def update_memory(
        self,
        messages: list[Any],
        thread_id: str | None = None,
        agent_name: str | None = None,
        correction_detected: bool = False,
        reinforcement_detected: bool = False,
        user_id: str | None = None,
    ) -> bool:
        """使用同步LLM路径同步更新记忆。
        
        使用 ``model.invoke()`` (同步HTTP) 在与主代理
        共享的异步 ``AsyncClient`` 完全不同的连接池上操作。
        这消除了跨循环连接复用错误，如issue #2615所述。
        
        当从运行中的事件循环内调用时(例如从LangGraph节点)，
        阻塞同步调用被卸载到线程池，这样调用者的循环就不会被阻塞。
        
        Args:
            messages: 对话消息列表
            thread_id: 可选的线程ID，用于跟踪源
            agent_name: 如果提供，更新特定代理的记忆；如果为None，更新全局记忆
            correction_detected: 最近轮次是否包含明确的纠正信号
            reinforcement_detected: 最近轮次是否包含积极强化信号
            user_id: 如果提供，将记忆限定到特定用户
            
        Returns:
            更新成功返回True，否则返回False
            
        Note:
            - 使用同步HTTP调用
            - 在事件循环中自动卸载到线程池
            - 避免跨循环连接复用
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            try:
                future = _SYNC_MEMORY_UPDATER_EXECUTOR.submit(
                    self._do_update_memory_sync,
                    messages=messages,
                    thread_id=thread_id,
                    agent_name=agent_name,
                    correction_detected=correction_detected,
                    reinforcement_detected=reinforcement_detected,
                    user_id=user_id,
                )
                return future.result()
            except Exception:
                logger.exception("Failed to offload memory update to executor")
                return False

        return self._do_update_memory_sync(
            messages=messages,
            thread_id=thread_id,
            agent_name=agent_name,
            correction_detected=correction_detected,
            reinforcement_detected=reinforcement_detected,
            user_id=user_id,
        )

    def _apply_updates(
        self,
        current_memory: dict[str, Any],
        update_data: dict[str, Any],
        thread_id: str | None = None,
    ) -> dict[str, Any]:
        """应用LLM生成的更新到记忆。
        
        将LLM返回的更新数据应用到当前记忆，包括用户上下文、历史和事实。
        
        Args:
            current_memory: 当前记忆数据
            update_data: 来自LLM的更新数据
            thread_id: 可选的线程ID
            
        Returns:
            更新后的记忆数据
            
        Note:
            - 更新用户上下文部分(workContext, personalContext, topOfMind)
            - 更新历史部分(recentMonths, earlierContext, longTermBackground)
            - 处理要移除的事实
            - 添加新事实并进行去重
            - 应用最大事实数量限制
        """
        config = get_memory_config()
        now = utc_now_iso_z()

        # 更新用户部分
        user_updates = update_data.get("user", {})
        for section in ["workContext", "personalContext", "topOfMind"]:
            section_data = user_updates.get(section, {})
            if section_data.get("shouldUpdate") and section_data.get("summary"):
                current_memory["user"][section] = {
                    "summary": section_data["summary"],
                    "updatedAt": now,
                }

        # 更新历史部分
        history_updates = update_data.get("history", {})
        for section in ["recentMonths", "earlierContext", "longTermBackground"]:
            section_data = history_updates.get(section, {})
            if section_data.get("shouldUpdate") and section_data.get("summary"):
                current_memory["history"][section] = {
                    "summary": section_data["summary"],
                    "updatedAt": now,
                }

        # 移除事实
        facts_to_remove = set(update_data.get("factsToRemove", []))
        if facts_to_remove:
            current_memory["facts"] = [f for f in current_memory.get("facts", []) if f.get("id") not in facts_to_remove]

        # 添加新事实
        existing_fact_keys = {fact_key for fact_key in (_fact_content_key(fact.get("content")) for fact in current_memory.get("facts", [])) if fact_key is not None}
        new_facts = update_data.get("newFacts", [])
        for fact in new_facts:
            confidence = fact.get("confidence", 0.5)
            if confidence >= config.fact_confidence_threshold:
                raw_content = fact.get("content", "")
                if not isinstance(raw_content, str):
                    continue
                normalized_content = raw_content.strip()
                fact_key = _fact_content_key(normalized_content)
                if fact_key is not None and fact_key in existing_fact_keys:
                    continue

                fact_entry = {
                    "id": f"fact_{uuid.uuid4().hex[:8]}",
                    "content": normalized_content,
                    "category": fact.get("category", "context"),
                    "confidence": confidence,
                    "createdAt": now,
                    "source": thread_id or "unknown",
                }
                source_error = fact.get("sourceError")
                if isinstance(source_error, str):
                    normalized_source_error = source_error.strip()
                    if normalized_source_error:
                        fact_entry["sourceError"] = normalized_source_error
                current_memory["facts"].append(fact_entry)
                if fact_key is not None:
                    existing_fact_keys.add(fact_key)

        # 强制执行最大事实限制
        if len(current_memory["facts"]) > config.max_facts:
            # 按置信度排序并保留最高的
            current_memory["facts"] = sorted(
                current_memory["facts"],
                key=lambda f: f.get("confidence", 0),
                reverse=True,
            )[: config.max_facts]

        return current_memory


def update_memory_from_conversation(
    messages: list[Any],
    thread_id: str | None = None,
    agent_name: str | None = None,
    correction_detected: bool = False,
    reinforcement_detected: bool = False,
    user_id: str | None = None,
) -> bool:
    """从对话更新记忆的便利函数。
    
    便捷函数，用于从对话消息列表更新记忆，封装了MemoryUpdater的使用。
    
    Args:
        messages: 对话消息列表
        thread_id: 可选的线程ID
        agent_name: 如果提供，更新特定代理的记忆；如果为None，更新全局记忆
        correction_detected: 最近轮次是否包含明确的纠正信号
        reinforcement_detected: 最近轮次是否包含积极强化信号
        user_id: 如果提供，将记忆限定到特定用户
        
    Returns:
        成功返回True，否则返回False
        
    Note:
        - 创建MemoryUpdater实例
        - 调用update_memory方法
        - 适用于简单的记忆更新场景
    """
    updater = MemoryUpdater()
    return updater.update_memory(messages, thread_id, agent_name, correction_detected, reinforcement_detected, user_id=user_id)
