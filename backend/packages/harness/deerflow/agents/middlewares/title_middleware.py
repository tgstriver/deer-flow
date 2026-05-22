"""线程标题自动生成中间件。

当用户与助手完成第一轮对话后，该中间件会自动为线程生成一个简明的标题。
标题生成有两种路径：
  - 异步路径（aafter_model）：调用 LLM 根据对话内容生成标题，失败时回退到本地截断。
  - 同步路径（after_model）：直接使用用户消息的截断文本作为标题，不调用 LLM。

标题生成的触发条件：
  1. 标题功能已启用（title_config.enabled 为 True）
  2. 线程当前没有标题（state 中 title 为空）
  3. 至少存在一条用户消息和一条助手回复（第一轮完整对话完成）
  4. 用户消息恰好只有一条（仅首轮触发，避免后续轮次覆盖标题）

该中间件还会对消息内容进行规范化处理：
  - 将列表/字典等结构化内容递归提取为纯文本
  - 剥离推理模型（如 DeepSeek-R1）输出的 <think...</think 标签
  - 对标题长度进行截断控制
"""

import logging
import re
from typing import TYPE_CHECKING, Any, NotRequired, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langgraph.config import get_config
from langgraph.runtime import Runtime

from deerflow.agents.middlewares.dynamic_context_middleware import is_dynamic_context_reminder
from deerflow.config.title_config import get_title_config
from deerflow.models import create_chat_model

if TYPE_CHECKING:
    from deerflow.config.app_config import AppConfig
    from deerflow.config.title_config import TitleConfig

logger = logging.getLogger(__name__)


class TitleMiddlewareState(AgentState):
    """标题中间件使用的状态模式，与 ThreadState 模式兼容。

    在 AgentState 基础上增加了可选的 title 字段，
    用于在线程状态中传递自动生成的标题。
    """

    title: NotRequired[str | None]


class TitleMiddleware(AgentMiddleware[TitleMiddlewareState]):
    """线程标题自动生成中间件。

    在首轮完整对话（用户消息 + 助手回复）完成后自动生成线程标题。
    支持异步 LLM 生成和同步本地回退两种模式。
    """

    state_schema = TitleMiddlewareState

    def __init__(self, *, app_config: "AppConfig | None" = None, title_config: "TitleConfig | None" = None):
        super().__init__()
        self._app_config = app_config
        self._title_config = title_config

    def _get_title_config(self):
        """获取标题配置，按优先级依次尝试：显式传入 > 应用配置中获取 > 全局默认配置。"""
        # 优先使用构造时显式传入的 title_config
        if self._title_config is not None:
            return self._title_config
        # 其次从 app_config 的 title 子配置中获取
        if self._app_config is not None:
            return self._app_config.title
        # 最后回退到全局默认配置
        return get_title_config()

    def _normalize_content(self, content: object) -> str:
        """将结构化消息内容递归规范化为纯文本字符串。

        LLM 的消息内容可能是字符串、列表或字典等多种格式：
        - 字符串：直接返回
        - 列表：递归处理每个元素，用换行符拼接非空结果
          （适用于多模态消息，如文本 + 图片的混合内容列表）
        - 字典：优先提取 "text" 字段，其次提取 "content" 字段并递归处理
          （适用于 OpenAI 风格的消息块，如 {"type": "text", "text": "..."}）

        对于无法识别的格式，返回空字符串。
        """
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            # 递归处理列表中的每个元素，过滤掉空结果后用换行符拼接
            parts = [self._normalize_content(item) for item in content]
            return "\n".join(part for part in parts if part)

        if isinstance(content, dict):
            # 优先提取 dict 中的 "text" 字段（常见于 OpenAI 消息块格式）
            text_value = content.get("text")
            if isinstance(text_value, str):
                return text_value

            # 其次提取 "content" 字段并递归处理（可能嵌套更深层的结构）
            nested_content = content.get("content")
            if nested_content is not None:
                return self._normalize_content(nested_content)

        # 无法识别的格式，返回空字符串
        return ""

    @staticmethod
    def _is_user_message_for_title(message: object) -> bool:
        """判断消息是否为用于标题生成的用户消息。

        条件：消息类型为 "human" 且不是动态上下文提醒。
        动态上下文提醒（如日期更新、记忆注入）是系统内部插入的 HumanMessage，
        不应被计入标题生成的用户消息，否则会导致误判首轮对话已完成。
        """
        return getattr(message, "type", None) == "human" and not is_dynamic_context_reminder(message)

    def _should_generate_title(self, state: TitleMiddlewareState) -> bool:
        """判断是否应该为当前线程生成标题。

        触发条件（全部满足）：
        1. 标题功能已启用（title_config.enabled 为 True）
        2. 线程尚未设置标题（state 中 title 为空）
        3. 至少存在 2 条消息（一轮对话的最小量）
        4. 恰好只有 1 条用户消息（排除动态上下文提醒后），且有至少 1 条助手回复
           — 即仅首轮完整对话后触发，避免后续轮次覆盖已有标题
        """
        config = self._get_title_config()
        if not config.enabled:
            return False

        # 线程已有标题则跳过，避免覆盖
        if state.get("title"):
            return False

        # 消息数量不足一轮对话（至少需要用户消息 + 助手回复各一条）
        messages = state.get("messages", [])
        if len(messages) < 2:
            return False

        # 分别统计用户消息和助手消息数量
        user_messages = [m for m in messages if self._is_user_message_for_title(m)]
        assistant_messages = [m for m in messages if m.type == "ai"]

        # 仅在首轮完整对话后生成：恰好 1 条用户消息 + 至少 1 条助手回复
        return len(user_messages) == 1 and len(assistant_messages) >= 1

    def _build_title_prompt(self, state: TitleMiddlewareState) -> tuple[str, str]:
        """构建标题生成的提示词，并返回可用于回退标题的用户消息原文。

        返回值：(prompt_string, user_msg)
        - prompt_string：格式化后的提示词，传入 LLM 生成标题
        - user_msg：用户消息的纯文本内容，在 LLM 调用失败时用作回退标题

        处理细节：
        - 用户消息和助手消息各截取前 500 个字符，避免提示词过长
        - 助手消息会先剥离 <think... 标签（推理模型的内部思考过程不参与标题生成）
        """
        config = self._get_title_config()
        messages = state.get("messages", [])

        # 提取第一条用户消息和第一条助手消息的内容
        user_msg_content = next((m.content for m in messages if self._is_user_message_for_title(m)), "")
        assistant_msg_content = next((m.content for m in messages if m.type == "ai"), "")

        # 规范化消息内容为纯文本
        user_msg = self._normalize_content(user_msg_content)
        # 助手消息额外剥离推理标签，因为思考过程不应影响标题生成
        assistant_msg = self._strip_think_tags(self._normalize_content(assistant_msg_content))

        # 使用配置的模板格式化提示词，截取前 500 字符以控制长度
        prompt = config.prompt_template.format(
            max_words=config.max_words,
            user_msg=user_msg[:500],
            assistant_msg=assistant_msg[:500],
        )
        return prompt, user_msg

    def _strip_think_tags(self, text: str) -> str:
        """剥离推理模型输出的 <think...</think 标签。

        推理模型（如 DeepSeek-R1、MiniMax 等）会在回复中包含 <think...</think
        标签，其中是模型的内部推理过程。这些内容对标题生成无意义，需要清除。

        使用正则表达式以非贪婪模式匹配（包括跨行内容），
        并忽略大小写以兼容不同模型的标签格式。
        """
        return re.sub(r"<think[\s\S]*?</think\s*>", "", text, flags=re.IGNORECASE).strip()

    def _parse_title(self, content: object) -> str:
        """将 LLM 输出规范化为干净的标题字符串。

        处理步骤：
        1. 递归规范化消息内容为纯文本
        2. 剥离推理标签（部分模型在生成标题时也可能输出思考过程）
        3. 去除首尾空白和引号（LLM 有时会在标题外加引号）
        4. 按配置的 max_chars 截断标题长度
        """
        config = self._get_title_config()
        title_content = self._normalize_content(content)
        title_content = self._strip_think_tags(title_content)
        # 去除首尾空白和可能存在的引号（LLM 有时输出 "标题" 或 '标题'）
        title = title_content.strip().strip('"').strip("'")
        # 超过最大字符数时截断
        return title[: config.max_chars] if len(title) > config.max_chars else title

    def _fallback_title(self, user_msg: str) -> str:
        """当 LLM 标题生成失败时，使用用户消息的截断文本作为回退标题。

        回退策略：
        - 取用户消息的前 50 个字符（不超过 max_chars 上限）
        - 超过长度限制时追加省略号 "..."
        - 用户消息为空时使用默认标题 "New Conversation"
        """
        config = self._get_title_config()
        fallback_chars = min(config.max_chars, 50)
        if len(user_msg) > fallback_chars:
            # 截断并追加省略号
            return user_msg[:fallback_chars].rstrip() + "..."
        # 用户消息为空则使用默认标题
        return user_msg if user_msg else "New Conversation"

    def _get_runnable_config(self) -> dict[str, Any]:
        """继承父级 RunnableConfig 并添加中间件标签。

        通过继承当前上下文的 RunnableConfig，确保标题生成的 LLM 调用
        能够获取到正确的配置（如 API 密钥等）。

        同时添加 run_name="title_agent" 和 tags=["middleware:title"]，
        使 RunJournal 能将此 LLM 调用识别为标题生成中间件发起的，
        而非主代理（lead_agent）的调用。
        """
        try:
            # 尝试获取当前 LangGraph 运行上下文的配置
            parent = get_config()
        except Exception:
            # 无运行上下文时使用空配置（例如在非 LangGraph 环境中调用）
            parent = {}
        config = {**parent}
        # 标记此 LLM 调用的来源，便于日志追踪和指标统计
        config["run_name"] = "title_agent"
        config["tags"] = [*(config.get("tags") or []), "middleware:title"]
        return config

    def _generate_title_result(self, state: TitleMiddlewareState) -> dict | None:
        """同步路径：生成本地回退标题，不调用 LLM。

        这是 after_model 的同步实现，直接使用用户消息的截断文本作为标题。
        不发起 LLM 调用，因此不会阻塞主流程，适用于：
        - 不需要高质量标题的场景
        - 异步 LLM 调用不可用时的降级方案

        返回值：
        - {"title": ...}：生成的标题
        - None：无需生成标题（不满足触发条件时）
        """
        if not self._should_generate_title(state):
            return None

        # 仅提取用户消息用于回退标题，不调用 LLM
        _, user_msg = self._build_title_prompt(state)
        return {"title": self._fallback_title(user_msg)}

    async def _agenerate_title_result(self, state: TitleMiddlewareState) -> dict | None:
        """异步路径：调用 LLM 生成标题，失败时回退到本地截断标题。

        这是 aafter_model 的异步实现，流程如下：
        1. 检查是否满足标题生成条件
        2. 构建标题提示词，准备用户消息原文作为回退
        3. 创建专用的标题生成模型（禁用思考模式，避免输出推理过程）
        4. 异步调用 LLM 生成标题
        5. 解析 LLM 输出为干净标题字符串
        6. 若 LLM 调用失败或输出为空，使用用户消息截断文本作为回退标题

        注意：标题生成模型强制设置 thinking_enabled=False，
        因为标题生成不需要推理过程，且思考标签会增加延迟和解析复杂度。

        返回值：
        - {"title": ...}：生成的标题（LLM 生成或回退）
        - None：无需生成标题（不满足触发条件时）
        """
        if not self._should_generate_title(state):
            return None

        config = self._get_title_config()
        prompt, user_msg = self._build_title_prompt(state)

        try:
            # 禁用思考模式：标题生成不需要推理过程，避免输出 <think 标签
            model_kwargs = {"thinking_enabled": False}
            if self._app_config is not None:
                model_kwargs["app_config"] = self._app_config
            # 根据配置决定是否指定模型名称
            if config.model_name:
                model = create_chat_model(name=config.model_name, **model_kwargs)
            else:
                model = create_chat_model(**model_kwargs)
            # 异步调用 LLM 生成标题
            response = await model.ainvoke(prompt, config=self._get_runnable_config())
            title = self._parse_title(response.content)
            if title:
                return {"title": title}
        except Exception:
            # LLM 调用失败时记录调试日志，使用回退标题
            logger.debug("Failed to generate async title; falling back to local title", exc_info=True)
        # LLM 调用失败或输出为空时，回退到本地截断标题
        return {"title": self._fallback_title(user_msg)}

    @override
    def after_model(self, state: TitleMiddlewareState, runtime: Runtime) -> dict | None:
        """同步模式下的后处理钩子：在模型调用后生成本地回退标题。

        该方法由中间件框架在模型响应后同步调用。
        返回的状态更新将合并到线程状态中。
        """
        return self._generate_title_result(state)

    @override
    async def aafter_model(self, state: TitleMiddlewareState, runtime: Runtime) -> dict | None:
        """异步模式下的后处理钩子：在模型调用后异步生成 LLM 标题。

        该方法由中间件框架在模型响应后异步调用。
        优先使用 LLM 生成高质量标题，失败时回退到本地截断标题。
        """
        return await self._agenerate_title_result(state)
