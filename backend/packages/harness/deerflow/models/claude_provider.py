"""自定义 Claude 模型提供者，支持 OAuth Bearer 认证、提示缓存和智能思考预算。

本模块提供了 ChatAnthropic 的扩展子类 ClaudeChatModel，在标准 API 密钥认证的基础上，
新增了 Claude Code OAuth 令牌认证、提示缓存（prompt caching）和自动思考预算分配等功能。

支持两种认证模式:
  1. 标准 API 密钥（x-api-key 请求头）—— 默认 ChatAnthropic 行为
  2. Claude Code OAuth 令牌（Authorization: Bearer 请求头）
     - 通过 sk-ant-oat 前缀自动检测
     - 需要 anthropic-beta: oauth-2025-04-20,claude-code-20250219
     - 所有 OAuth 请求需要在系统提示中包含计费头（billing header）

自动从以下来源加载凭证:
  - $ANTHROPIC_API_KEY 环境变量
  - $CLAUDE_CODE_OAUTH_TOKEN 或 $ANTHROPIC_AUTH_TOKEN
  - $CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR
  - $CLAUDE_CODE_CREDENTIALS_PATH
  - ~/.claude/.credentials.json
"""

import hashlib
import json
import logging
import os
import socket
import time
import uuid
from typing import Any

import anthropic
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage
from pydantic import PrivateAttr

logger = logging.getLogger(__name__)

MAX_RETRIES = 3  # 最大重试次数
THINKING_BUDGET_RATIO = 0.8  # 思考预算占 max_tokens 的比例（80%）

# 计费头（Billing header）：Anthropic API 对 OAuth 令牌访问的必需字段。
# 必须作为系统提示的第一个块。格式与 Claude Code CLI 一致。
# 如果硬编码版本过时，可通过 ANTHROPIC_BILLING_HEADER 环境变量覆盖。
_DEFAULT_BILLING_HEADER = "x-anthropic-billing-header: cc_version=2.1.85.351; cc_entrypoint=cli; cch=6c6d5;"
OAUTH_BILLING_HEADER = os.environ.get("ANTHROPIC_BILLING_HEADER", _DEFAULT_BILLING_HEADER)


class ClaudeChatModel(ChatAnthropic):
    """支持 OAuth Bearer 认证、提示缓存和智能思考的 ChatAnthropic 扩展类。

    本类继承自 ChatAnthropic，新增以下核心能力：
    - OAuth Bearer 认证：自动检测 Claude Code OAuth 令牌并切换为 Bearer 认证模式
    - 提示缓存（Prompt Caching）：自动在系统提示、近期消息和工具定义上添加缓存断点
    - 智能思考预算：自动将 max_tokens 的 80% 分配给思考预算
    - 重试机制：对速率限制和服务端错误自动进行指数退避重试

    配置示例:
        - name: claude-sonnet-4.6
          use: deerflow.models.claude_provider:ClaudeChatModel
          model: claude-sonnet-4-6
          max_tokens: 16384
          enable_prompt_caching: true
    """

    # 自定义字段
    enable_prompt_caching: bool = True  # 是否启用提示缓存（默认启用）
    prompt_cache_size: int = 3  # 近期参与缓存的消息数量
    auto_thinking_budget: bool = True  # 是否自动分配思考预算（默认启用）
    retry_max_attempts: int = MAX_RETRIES  # 最大重试次数
    _is_oauth: bool = PrivateAttr(default=False)  # 内部标记：是否为 OAuth 认证模式
    _oauth_access_token: str = PrivateAttr(default="")  # 内部存储：OAuth 访问令牌

    model_config = {"arbitrary_types_allowed": True}

    def _validate_retry_config(self) -> None:
        """验证重试配置的合法性。"""
        if self.retry_max_attempts < 1:
            raise ValueError("retry_max_attempts must be >= 1")

    def model_post_init(self, __context: Any) -> None:
        """模型初始化后处理：自动加载凭证并配置 OAuth 认证。

        执行流程：
        1. 验证重试配置
        2. 提取当前的 API 密钥
        3. 若无有效密钥，尝试从 Claude Code 凭证源加载
        4. 检测是否为 OAuth 令牌，若是则配置 Bearer 认证模式
        5. 调用父类初始化
        6. 若为 OAuth 模式，修补客户端以使用 Bearer 认证
        """
        from pydantic import SecretStr

        from deerflow.models.credential_loader import (
            OAUTH_ANTHROPIC_BETAS,
            is_oauth_token,
            load_claude_code_credential,
        )

        self._validate_retry_config()

        # 提取实际的密钥值（SecretStr.str() 返回 '**********'，无法获取真实值）
        current_key = ""
        if self.anthropic_api_key:
            if hasattr(self.anthropic_api_key, "get_secret_value"):
                current_key = self.anthropic_api_key.get_secret_value()
            else:
                current_key = str(self.anthropic_api_key)

        # 若无有效密钥，尝试从 Claude Code OAuth 显式交接源加载
        if not current_key or current_key in ("your-anthropic-api-key",):
            cred = load_claude_code_credential()
            if cred:
                current_key = cred.access_token
                logger.info(f"Using Claude Code CLI credential (source: {cred.source})")
            else:
                logger.warning("No Anthropic API key or explicit Claude Code OAuth credential found.")

        # 检测 OAuth 令牌并配置 Bearer 认证
        if is_oauth_token(current_key):
            self._is_oauth = True
            self._oauth_access_token = current_key
            # 将令牌临时设置为 api_key（稍后在客户端上会交换为 auth_token）
            self.anthropic_api_key = SecretStr(current_key)
            # 添加 OAuth 所需的 beta 请求头
            self.default_headers = {
                **(self.default_headers or {}),
                "anthropic-beta": OAUTH_ANTHROPIC_BETAS,
            }
            # OAuth 令牌最多支持 4 个 cache_control 块 —— 禁用提示缓存
            self.enable_prompt_caching = False
            logger.info("OAuth token detected — will use Authorization: Bearer header")
        else:
            if current_key:
                self.anthropic_api_key = SecretStr(current_key)

        # 确保 api_key 为 SecretStr 类型
        if isinstance(self.anthropic_api_key, str):
            self.anthropic_api_key = SecretStr(self.anthropic_api_key)

        super().model_post_init(__context)

        # OAuth Bearer 认证：在客户端创建后立即修补。
        # 必须在 super() 之后执行，因为客户端是延迟创建的。
        if self._is_oauth:
            self._patch_client_oauth(self._client)
            self._patch_client_oauth(self._async_client)

    def _patch_client_oauth(self, client: Any) -> None:
        """将 Anthropic SDK 客户端的 api_key 替换为 auth_token 以支持 OAuth Bearer 认证。

        OAuth 令牌需要通过 Authorization: Bearer 请求头发送，而非标准的 x-api-key。
        此方法将客户端的 api_key 置空，并将 OAuth 访问令牌设置到 auth_token 字段。

        Args:
            client: Anthropic SDK 客户端实例（同步或异步）
        """
        if hasattr(client, "api_key") and hasattr(client, "auth_token"):
            client.api_key = None
            client.auth_token = self._oauth_access_token

    def _get_request_payload(
        self,
        input_: Any,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        """重写请求载荷构建，注入提示缓存、思考预算和 OAuth 计费头。

        在父类生成的基础载荷上，依次应用：
        1. OAuth 计费头注入（仅 OAuth 模式）
        2. 提示缓存断点标记（启用时）
        3. 思考预算自动分配（启用时）

        Args:
            input_: 输入消息（LangChain 消息列表）
            stop: 停止词列表
            **kwargs: 其他传递给父类的参数

        Returns:
            dict: 注入额外配置后的 API 请求载荷
        """
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)

        if self._is_oauth:
            self._apply_oauth_billing(payload)

        if self.enable_prompt_caching:
            self._apply_prompt_caching(payload)

        if self.auto_thinking_budget:
            self._apply_thinking_budget(payload)

        return payload

    def _apply_oauth_billing(self, payload: dict) -> None:
        """注入 OAuth 请求所需的计费头块。

        计费块始终放置在 system 列表的最前面，并移除任何已有的计费块以避免
        重复或顺序错位。同时添加 metadata.user_id 字段，这是 OAuth 计费
        验证所必需的。

        Args:
            payload: API 请求载荷字典，将被就地修改
        """
        billing_block = {"type": "text", "text": OAUTH_BILLING_HEADER}

        system = payload.get("system")
        if isinstance(system, list):
            # 移除已有的计费块，然后在索引 0 处插入新的计费块
            filtered = [b for b in system if not (isinstance(b, dict) and OAUTH_BILLING_HEADER in b.get("text", ""))]
            payload["system"] = [billing_block] + filtered
        elif isinstance(system, str):
            if OAUTH_BILLING_HEADER in system:
                payload["system"] = [billing_block]
            else:
                payload["system"] = [billing_block, {"type": "text", "text": system}]
        else:
            payload["system"] = [billing_block]

        # 添加 metadata.user_id 字段，OAuth 计费验证所需
        if not isinstance(payload.get("metadata"), dict):
            payload["metadata"] = {}
        if "user_id" not in payload["metadata"]:
            # 从机器主机名生成稳定的 device_id
            hostname = socket.gethostname()
            device_id = hashlib.sha256(f"deerflow-{hostname}".encode()).hexdigest()
            session_id = str(uuid.uuid4())
            payload["metadata"]["user_id"] = json.dumps(
                {
                    "device_id": device_id,
                    "account_uuid": "deerflow",
                    "session_id": session_id,
                }
            )

    def _apply_prompt_caching(self, payload: dict) -> None:
        """对系统提示、近期消息和最后一个工具定义应用临时缓存标记。

        使用最多 MAX_CACHE_BREAKPOINTS（4）个断点——这是 Anthropic API 和 AWS Bedrock
        共同强制执行的硬性限制。断点放在*最后*的候选块上，因为靠后的断点覆盖更大的
        前缀，能产生更高的缓存命中率。

        系统提示预期为完全静态内容（不含用户记忆或当前日期）。
        动态上下文通过 DynamicContextMiddleware 在每轮对话中以 <system-reminder>
        标签注入到第一个 HumanMessage 中。

        Args:
            payload: API 请求载荷字典，将被就地修改
        """
        MAX_CACHE_BREAKPOINTS = 4  # 最大缓存断点数，API 硬性限制

        # 按文档顺序收集候选块：
        #   1. 系统提示文本块
        #   2. 最近 prompt_cache_size 条消息的内容块
        #   3. 最后一个工具定义
        candidates: list[dict] = []

        # 1. 系统块
        system = payload.get("system")
        if system and isinstance(system, list):
            for block in system:
                if isinstance(block, dict) and block.get("type") == "text":
                    candidates.append(block)
        elif system and isinstance(system, str):
            new_block: dict = {"type": "text", "text": system}
            payload["system"] = [new_block]
            candidates.append(new_block)

        # 2. 近期消息块
        messages = payload.get("messages", [])
        cache_start = max(0, len(messages) - self.prompt_cache_size)
        for i in range(cache_start, len(messages)):
            msg = messages[i]
            if not isinstance(msg, dict):
                continue
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        candidates.append(block)
            elif isinstance(content, str) and content:
                new_block = {"type": "text", "text": content}
                msg["content"] = [new_block]
                candidates.append(new_block)

        # 3. 最后一个工具定义
        tools = payload.get("tools", [])
        if tools and isinstance(tools[-1], dict):
            candidates.append(tools[-1])

        # 仅对最后 MAX_CACHE_BREAKPOINTS 个候选块应用 cache_control，
        # 以保持在 API 限制之内。
        for block in candidates[-MAX_CACHE_BREAKPOINTS:]:
            block["cache_control"] = {"type": "ephemeral"}

    def _apply_thinking_budget(self, payload: dict) -> None:
        """自动分配思考预算（max_tokens 的 80%）。

        仅在思考模式已启用且未手动指定预算时生效。

        Args:
            payload: API 请求载荷字典，将被就地修改
        """
        thinking = payload.get("thinking")
        if not thinking or not isinstance(thinking, dict):
            return
        if thinking.get("type") != "enabled":
            return
        if thinking.get("budget_tokens"):
            return

        max_tokens = payload.get("max_tokens", 8192)
        thinking["budget_tokens"] = int(max_tokens * THINKING_BUDGET_RATIO)

    @staticmethod
    def _strip_cache_control(payload: dict) -> None:
        """在 OAuth 请求发送到 Anthropic 之前移除 cache_control 标记。

        OAuth 令牌最多支持 4 个 cache_control 块，为安全起见在 OAuth 请求中
        完全移除缓存标记。

        Args:
            payload: API 请求载荷字典，将被就地修改
        """
        for section in ("system", "messages"):
            items = payload.get(section)
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                item.pop("cache_control", None)
                content = item.get("content")
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            block.pop("cache_control", None)

        tools = payload.get("tools")
        if isinstance(tools, list):
            for tool in tools:
                if isinstance(tool, dict):
                    tool.pop("cache_control", None)

    def _create(self, payload: dict) -> Any:
        """同步创建请求，OAuth 模式下移除缓存控制标记。"""
        if self._is_oauth:
            self._strip_cache_control(payload)
        return super()._create(payload)

    async def _acreate(self, payload: dict) -> Any:
        """异步创建请求，OAuth 模式下移除缓存控制标记。"""
        if self._is_oauth:
            self._strip_cache_control(payload)
        return await super()._acreate(payload)

    def _generate(self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any) -> Any:
        """同步生成响应，包含 OAuth 修补和重试逻辑。

        对速率限制（RateLimitError）和服务端错误（InternalServerError）进行
        指数退避重试，最多重试 retry_max_attempts 次。

        Args:
            messages: LangChain 消息列表
            stop: 停止词列表
            **kwargs: 其他传递给父类的参数

        Returns:
            父类生成的 ChatResult
        """
        if self._is_oauth:
            self._patch_client_oauth(self._client)

        last_error = None
        for attempt in range(1, self.retry_max_attempts + 1):
            try:
                return super()._generate(messages, stop=stop, **kwargs)
            except anthropic.RateLimitError as e:
                last_error = e
                if attempt >= self.retry_max_attempts:
                    raise
                wait_ms = self._calc_backoff_ms(attempt, e)
                logger.warning(f"Rate limited, retrying attempt {attempt}/{self.retry_max_attempts} after {wait_ms}ms")
                time.sleep(wait_ms / 1000)
            except anthropic.InternalServerError as e:
                last_error = e
                if attempt >= self.retry_max_attempts:
                    raise
                wait_ms = self._calc_backoff_ms(attempt, e)
                logger.warning(f"Server error, retrying attempt {attempt}/{self.retry_max_attempts} after {wait_ms}ms")
                time.sleep(wait_ms / 1000)
        raise last_error

    async def _agenerate(self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any) -> Any:
        """异步生成响应，包含 OAuth 修补和重试逻辑。

        异步版本的 _generate，同样对速率限制和服务端错误进行指数退避重试。

        Args:
            messages: LangChain 消息列表
            stop: 停止词列表
            **kwargs: 其他传递给父类的参数

        Returns:
            父类生成的 ChatResult
        """
        import asyncio

        if self._is_oauth:
            self._patch_client_oauth(self._async_client)

        last_error = None
        for attempt in range(1, self.retry_max_attempts + 1):
            try:
                return await super()._agenerate(messages, stop=stop, **kwargs)
            except anthropic.RateLimitError as e:
                last_error = e
                if attempt >= self.retry_max_attempts:
                    raise
                wait_ms = self._calc_backoff_ms(attempt, e)
                logger.warning(f"Rate limited, retrying attempt {attempt}/{self.retry_max_attempts} after {wait_ms}ms")
                await asyncio.sleep(wait_ms / 1000)
            except anthropic.InternalServerError as e:
                last_error = e
                if attempt >= self.retry_max_attempts:
                    raise
                wait_ms = self._calc_backoff_ms(attempt, e)
                logger.warning(f"Server error, retrying attempt {attempt}/{self.retry_max_attempts} after {wait_ms}ms")
                await asyncio.sleep(wait_ms / 1000)
        raise last_error

    @staticmethod
    def _calc_backoff_ms(attempt: int, error: Exception) -> int:
        """计算指数退避等待时间（毫秒），附加固定 20% 抖动。

        基础退避公式：2000ms * 2^(attempt-1)，加上 20% 的抖动。
        如果错误响应中包含 Retry-After 头，则优先使用该值。

        Args:
            attempt: 当前重试次数（从 1 开始）
            error: 触发重试的异常

        Returns:
            int: 退避等待时间（毫秒）
        """
        backoff_ms = 2000 * (1 << (attempt - 1))
        jitter_ms = int(backoff_ms * 0.2)
        total_ms = backoff_ms + jitter_ms

        # 优先使用服务端返回的 Retry-After 头
        if hasattr(error, "response") and error.response is not None:
            retry_after = error.response.headers.get("Retry-After")
            if retry_after:
                try:
                    total_ms = int(retry_after) * 1000
                except (ValueError, TypeError):
                    pass

        return total_ms
