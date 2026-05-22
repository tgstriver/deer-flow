"""LLM 错误处理中间件：提供重试/退避机制和用户友好的降级消息。

当 LLM 提供商出现瞬时故障（如服务繁忙、超时、5xx 错误）时，自动进行指数退避重试；
当遇到不可恢复的错误（如配额耗尽、认证失败）或重试次数耗尽时，返回一条用户友好的 AIMessage 而非让整个 Agent 运行崩溃。

同时内置了熔断器（Circuit Breaker）机制：当连续失败达到阈值时自动"熔断"，快速拒绝后续请求，避免向已经不可用的服务持续发送请求；
经过恢复超时时间后进入"半开"状态放行一个探测请求，探测成功则恢复，探测失败则继续熔断。
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import Awaitable, Callable
from email.utils import parsedate_to_datetime
from typing import Any, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import (
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)
from langchain_core.messages import AIMessage
from langgraph.errors import GraphBubbleUp

from deerflow.config.app_config import AppConfig

logger = logging.getLogger(__name__)

# 可重试的 HTTP 状态码集合
# 408: 请求超时；409: 冲突；425: 过早请求；429: 请求过多（限流）
# 500: 服务器内部错误；502: 网关错误；503: 服务不可用；504: 网关超时
_RETRIABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}

# 服务繁忙相关的错误关键词模式（中英文）
# 用于匹配 LLM 提供商返回的错误消息，判断是否为临时性的负载问题
_BUSY_PATTERNS = (
    "server busy",
    "temporarily unavailable",
    "try again later",
    "please retry",
    "please try again",
    "overloaded",
    "high demand",
    "rate limit",
    "负载较高",
    "服务繁忙",
    "稍后重试",
    "请稍后重试",
)

# 配额/计费相关的错误关键词模式（中英文）
# 匹配到这些模式说明是账户层面的问题，重试无法解决
_QUOTA_PATTERNS = (
    "insufficient_quota",
    "quota",
    "billing",
    "credit",
    "payment",
    "余额不足",
    "超出限额",
    "额度不足",
    "欠费",
)

# 认证/授权相关的错误关键词模式（中英文）
# 匹配到这些模式说明是凭据问题，需要用户手动修复配置
_AUTH_PATTERNS = (
    "authentication",
    "unauthorized",
    "invalid api key",
    "invalid_api_key",
    "permission",
    "forbidden",
    "access denied",
    "无权",
    "未授权",
)


class LLMErrorHandlingMiddleware(AgentMiddleware[AgentState]):
    """LLM 错误处理中间件：重试瞬时错误并返回优雅的助手消息。

    核心机制：
    1. 重试：对可恢复的瞬时错误（超时、服务繁忙、5xx 等）进行指数退避重试
    2. 熔断器：连续失败达到阈值后熔断，快速拒绝请求，保护系统
    3. 降级：重试耗尽或不可恢复错误时，返回用户友好的 AIMessage 而非抛出异常
    """

    # 最大重试次数（含首次调用，即最多尝试 3 次）
    retry_max_attempts: int = 3
    # 重试的基础延迟（毫秒），实际延迟按指数退避计算
    retry_base_delay_ms: int = 1000
    # 重试延迟的上限（毫秒），防止退避时间过长
    retry_cap_delay_ms: int = 8000

    def __init__(self, *, app_config: AppConfig, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # 从应用配置中读取熔断器参数
        # 连续失败多少次后触发熔断
        self.circuit_failure_threshold = app_config.circuit_breaker.failure_threshold
        # 熔断后等待多久进入半开状态（秒）
        self.circuit_recovery_timeout_sec = app_config.circuit_breaker.recovery_timeout_sec

        # ===== 熔断器状态变量（线程安全，通过 _circuit_lock 保护） =====
        self._circuit_lock = threading.Lock()
        # 当前连续失败计数
        self._circuit_failure_count = 0
        # 熔断到期的时间戳（time.time() 基准），0 表示未熔断
        self._circuit_open_until = 0.0
        # 熔断器状态："closed"（正常）、"open"（熔断）、"half_open"（半开/探测）
        self._circuit_state = "closed"
        # 半开状态下是否已有探测请求在飞行中，防止并发探测
        self._circuit_probe_in_flight = False

    def _check_circuit(self) -> bool:
        """检查熔断器状态，判断是否应该快速拒绝请求。

        熔断器三种状态的转换逻辑：
        - closed（关闭/正常）：所有请求正常放行，返回 False
        - open（打开/熔断）：快速拒绝所有请求，返回 True
          直到 recovery_timeout_sec 到期后转为 half_open
        - half_open（半开/探测）：放行一个探测请求，其余拒绝
          探测成功 → closed；探测失败 → 重新 open

        Returns:
            True 表示熔断器处于打开状态，应快速拒绝请求；
            False 表示可以放行请求
        """
        with self._circuit_lock:
            now = time.time()

            if self._circuit_state == "open":
                # 熔断状态：检查恢复超时是否已到期
                if now < self._circuit_open_until:
                    # 未到期，继续快速拒绝
                    return True
                # 已到期，转为半开状态，准备放行探测请求
                self._circuit_state = "half_open"
                self._circuit_probe_in_flight = False

            if self._circuit_state == "half_open":
                # 半开状态：只允许一个探测请求通过
                if self._circuit_probe_in_flight:
                    # 已有探测请求在飞行中，拒绝其他请求
                    return True
                # 标记探测请求已发出，放行本次请求
                self._circuit_probe_in_flight = True
                return False

            # closed 状态：正常放行
            return False

    def _record_success(self) -> None:
        """记录一次成功调用，重置熔断器状态。

        当探测请求或正常请求成功时调用，将熔断器恢复到 closed 状态，
        清零失败计数。无论是从 half_open 探测成功还是 closed 状态正常调用，
        都会执行完整的重置。
        """
        with self._circuit_lock:
            if self._circuit_state != "closed" or self._circuit_failure_count > 0:
                logger.info("Circuit breaker reset (Closed). LLM service recovered.")
            self._circuit_failure_count = 0
            self._circuit_open_until = 0.0
            self._circuit_state = "closed"
            self._circuit_probe_in_flight = False

    def _record_failure(self) -> None:
        """记录一次失败调用，更新熔断器状态。

        根据当前熔断器状态执行不同逻辑：
        - half_open：探测请求失败，立即重新熔断
        - closed：累加失败计数，达到阈值后触发熔断
        """
        with self._circuit_lock:
            if self._circuit_state == "half_open":
                # 半开状态下的探测请求失败，立即重新熔断
                self._circuit_open_until = time.time() + self.circuit_recovery_timeout_sec
                self._circuit_state = "open"
                self._circuit_probe_in_flight = False
                logger.error(
                    "Circuit breaker probe failed (Open). Will probe again after %ds.",
                    self.circuit_recovery_timeout_sec,
                )
                return

            # closed 状态：累加失败计数
            self._circuit_failure_count += 1
            if self._circuit_failure_count >= self.circuit_failure_threshold:
                # 连续失败达到阈值，触发熔断
                self._circuit_open_until = time.time() + self.circuit_recovery_timeout_sec
                if self._circuit_state != "open":
                    self._circuit_state = "open"
                    self._circuit_probe_in_flight = False
                    logger.error(
                        "Circuit breaker tripped (Open). Threshold reached (%d). Will probe after %ds.",
                        self.circuit_failure_threshold,
                        self.circuit_recovery_timeout_sec,
                    )

    def _classify_error(self, exc: BaseException) -> tuple[bool, str]:
        """对异常进行分类，判断是否可重试以及错误类型。

        分类优先级（从高到低）：
        1. 配额/计费错误 → 不可重试 (quota)
        2. 认证/授权错误 → 不可重试 (auth)
        3. 已知可重试异常类名 → 可重试 (transient)
        4. 可重试 HTTP 状态码 → 可重试 (transient)
        5. 服务繁忙关键词 → 可重试 (busy)
        6. 其他 → 不可重试 (generic)

        Args:
            exc: 捕获到的异常对象

        Returns:
            (是否可重试, 错误分类标签) 的元组
        """
        # 提取错误详情文本和状态码用于分类判断
        detail = _extract_error_detail(exc)
        lowered = detail.lower()
        error_code = _extract_error_code(exc)
        status_code = _extract_status_code(exc)

        # 配额/计费错误：不可重试，需要用户手动解决
        if _matches_any(lowered, _QUOTA_PATTERNS) or _matches_any(str(error_code).lower(), _QUOTA_PATTERNS):
            return False, "quota"

        # 认证/授权错误：不可重试，需要用户修复凭据
        if _matches_any(lowered, _AUTH_PATTERNS):
            return False, "auth"

        # 已知的可重试异常类名
        exc_name = exc.__class__.__name__
        if exc_name in {
            "APITimeoutError",  # API 调用超时
            "APIConnectionError",  # API 连接失败
            "InternalServerError",  # 服务器内部错误
            "ReadError",  # httpx.ReadError: 流式传输中连接断开
            "RemoteProtocolError",  # httpx: 服务器意外关闭连接
        }:
            return True, "transient"

        # 可重试的 HTTP 状态码
        if status_code in _RETRIABLE_STATUS_CODES:
            return True, "transient"

        # 服务繁忙关键词匹配
        if _matches_any(lowered, _BUSY_PATTERNS):
            return True, "busy"

        # 未匹配任何已知模式，视为通用不可重试错误
        return False, "generic"

    def _build_retry_delay_ms(self, attempt: int, exc: BaseException) -> int:
        """计算重试等待时间（毫秒）。

        优先使用服务端返回的 Retry-After 头部值；
        如果服务端未指定，则使用指数退避算法：
        delay = base_delay * 2^(attempt-1)，且不超过上限值。

        Args:
            attempt: 当前重试次数（从 1 开始）
            exc: 触发重试的异常，可能包含 Retry-After 信息

        Returns:
            等待时间（毫秒）
        """
        # 优先使用服务端建议的重试等待时间
        retry_after = _extract_retry_after_ms(exc)
        if retry_after is not None:
            return retry_after
        # 指数退避：第1次重试等待 base_delay，第2次等待 2*base_delay，以此类推
        backoff = self.retry_base_delay_ms * (2 ** max(0, attempt - 1))
        return min(backoff, self.retry_cap_delay_ms)

    def _build_retry_message(self, attempt: int, wait_ms: int, reason: str) -> str:
        """构建重试提示消息，用于流式事件通知前端。

        Args:
            attempt: 当前重试次数
            wait_ms: 等待时间（毫秒）
            reason: 错误分类标签

        Returns:
            重试提示消息文本
        """
        seconds = max(1, round(wait_ms / 1000))
        reason_text = "provider is busy" if reason == "busy" else "provider request failed temporarily"
        return f"LLM request retry {attempt}/{self.retry_max_attempts}: {reason_text}. Retrying in {seconds}s."

    def _build_circuit_breaker_message(self) -> str:
        """构建熔断器触发时的用户提示消息。

        Returns:
            熔断器降级提示消息文本
        """
        return "The configured LLM provider is currently unavailable due to continuous failures. Circuit breaker is engaged to protect the system. Please wait a moment before trying again."

    def _build_user_message(self, exc: BaseException, reason: str) -> str:
        """根据错误分类构建用户友好的降级消息。

        不同类型的错误返回不同的提示，帮助用户理解问题并采取相应措施。

        Args:
            exc: 原始异常对象
            reason: 错误分类标签（quota/auth/busy/transient/generic）

        Returns:
            用户友好的错误提示消息
        """
        detail = _extract_error_detail(exc)
        if reason == "quota":
            # 配额/计费问题：提示用户检查账户
            return "The configured LLM provider rejected the request because the account is out of quota, billing is unavailable, or usage is restricted. Please fix the provider account and try again."
        if reason == "auth":
            # 认证/授权问题：提示用户检查凭据
            return "The configured LLM provider rejected the request because authentication or access is invalid. Please check the provider credentials and try again."
        if reason in {"busy", "transient"}:
            # 瞬时错误但重试耗尽：提示用户稍后再试
            return "The configured LLM provider is temporarily unavailable after multiple retries. Please wait a moment and continue the conversation."
        # 未知错误：返回原始错误详情
        return f"LLM request failed: {detail}"

    def _emit_retry_event(self, attempt: int, wait_ms: int, reason: str) -> None:
        """通过 LangGraph 的流式写入器发送重试事件，通知前端当前正在重试。

        该事件会被前端捕获并显示给用户，让用户知道系统正在自动重试。

        Args:
            attempt: 当前重试次数
            wait_ms: 等待时间（毫秒）
            reason: 错误分类标签
        """
        try:
            from langgraph.config import get_stream_writer

            writer = get_stream_writer()
            writer(
                {
                    "type": "llm_retry",
                    "attempt": attempt,
                    "max_attempts": self.retry_max_attempts,
                    "wait_ms": wait_ms,
                    "reason": reason,
                    "message": self._build_retry_message(attempt, wait_ms, reason),
                }
            )
        except Exception:
            # 流式写入器不可用时静默忽略（如在非流式调用上下文中）
            logger.debug("Failed to emit llm_retry event", exc_info=True)

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """同步包装模型调用，实现重试和熔断逻辑。

        执行流程：
        1. 检查熔断器状态，如果熔断则快速返回降级消息
        2. 调用模型处理器
        3. 成功 → 记录成功，返回结果
        4. GraphBubbleUp → 透传 LangGraph 控制流信号（中断/暂停/恢复）
        5. 可重试错误 → 重试（带退避和事件通知）
        6. 不可重试或重试耗尽 → 记录失败，返回用户友好的降级 AIMessage

        Args:
            request: 模型请求对象
            handler: 实际的模型调用处理函数

        Returns:
            模型响应结果或降级 AIMessage
        """
        # 首先检查熔断器是否处于打开状态
        if self._check_circuit():
            return AIMessage(content=self._build_circuit_breaker_message())

        attempt = 1
        while True:
            try:
                response = handler(request)
                # 调用成功，重置熔断器状态
                self._record_success()
                return response
            except GraphBubbleUp:
                # 透传 LangGraph 的控制流信号（中断/暂停/恢复等）
                # 不应被错误处理逻辑拦截，同时半开状态下释放探测标记
                with self._circuit_lock:
                    if self._circuit_state == "half_open":
                        self._circuit_probe_in_flight = False
                raise
            except Exception as exc:
                # 对异常进行分类
                retriable, reason = self._classify_error(exc)
                if retriable and attempt < self.retry_max_attempts:
                    # 可重试且未达到最大重试次数
                    wait_ms = self._build_retry_delay_ms(attempt, exc)
                    logger.warning(
                        "Transient LLM error on attempt %d/%d; retrying in %dms: %s",
                        attempt,
                        self.retry_max_attempts,
                        wait_ms,
                        _extract_error_detail(exc),
                    )
                    # 通知前端正在重试
                    self._emit_retry_event(attempt, wait_ms, reason)
                    # 同步等待后重试
                    time.sleep(wait_ms / 1000)
                    attempt += 1
                    continue
                # 不可重试或重试次数耗尽
                logger.warning(
                    "LLM call failed after %d attempt(s): %s",
                    attempt,
                    _extract_error_detail(exc),
                    exc_info=exc,
                )
                # 可重试但次数耗尽，记录失败（影响熔断器计数）
                if retriable:
                    self._record_failure()
                # 返回降级的 AIMessage，不让异常传播导致 Agent 崩溃
                return AIMessage(content=self._build_user_message(exc, reason))

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """异步包装模型调用，实现重试和熔断逻辑。

        逻辑与同步版本 wrap_model_call 完全相同，
        仅将同步等待替换为异步等待（asyncio.sleep）。

        Args:
            request: 模型请求对象
            handler: 实际的异步模型调用处理函数

        Returns:
            模型响应结果或降级 AIMessage
        """
        # 检查熔断器状态
        if self._check_circuit():
            return AIMessage(content=self._build_circuit_breaker_message())

        attempt = 1
        while True:
            try:
                response = await handler(request)
                self._record_success()
                return response
            except GraphBubbleUp:
                # 透传 LangGraph 控制流信号
                with self._circuit_lock:
                    if self._circuit_state == "half_open":
                        self._circuit_probe_in_flight = False
                raise
            except Exception as exc:
                retriable, reason = self._classify_error(exc)
                if retriable and attempt < self.retry_max_attempts:
                    wait_ms = self._build_retry_delay_ms(attempt, exc)
                    logger.warning(
                        "Transient LLM error on attempt %d/%d; retrying in %dms: %s",
                        attempt,
                        self.retry_max_attempts,
                        wait_ms,
                        _extract_error_detail(exc),
                    )
                    self._emit_retry_event(attempt, wait_ms, reason)
                    # 异步等待后重试
                    await asyncio.sleep(wait_ms / 1000)
                    attempt += 1
                    continue
                logger.warning(
                    "LLM call failed after %d attempt(s): %s",
                    attempt,
                    _extract_error_detail(exc),
                    exc_info=exc,
                )
                if retriable:
                    self._record_failure()
                return AIMessage(content=self._build_user_message(exc, reason))


# ===== 以下为模块级辅助函数，用于从异常中提取各类信息 =====


def _matches_any(detail: str, patterns: tuple[str, ...]) -> bool:
    """检查字符串是否匹配任一关键词模式（子串匹配）。

    Args:
        detail: 待检查的字符串（通常已转小写）
        patterns: 关键词模式元组

    Returns:
        是否匹配到至少一个模式
    """
    return any(pattern in detail for pattern in patterns)


def _extract_error_code(exc: BaseException) -> Any:
    """从异常对象中提取错误码。

    尝试从多个位置查找错误码：
    1. 异常对象的 code/error_code 属性
    2. 异常对象 body.error 中的 code/type 字段（OpenAI 风格的错误结构）

    Args:
        exc: 异常对象

    Returns:
        错误码值，未找到则返回 None
    """
    # 直接属性查找
    for attr in ("code", "error_code"):
        value = getattr(exc, attr, None)
        if value not in (None, ""):
            return value

    # OpenAI 风格的嵌套错误结构：exc.body.error.code / exc.body.error.type
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            for key in ("code", "type"):
                value = error.get(key)
                if value not in (None, ""):
                    return value
    return None


def _extract_status_code(exc: BaseException) -> int | None:
    """从异常对象中提取 HTTP 状态码。

    尝试从多个位置查找状态码：
    1. 异常对象的 status_code/status 属性
    2. 异常对象 response 属性中的 status_code（httpx/requests 风格）

    Args:
        exc: 异常对象

    Returns:
        HTTP 状态码（整数），未找到则返回 None
    """
    # 直接属性查找
    for attr in ("status_code", "status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    # 从 response 对象中查找（httpx.HTTPStatusError / requests.HTTPError）
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None)
    return status if isinstance(status, int) else None


def _extract_retry_after_ms(exc: BaseException) -> int | None:
    """从异常对象的响应头中提取 Retry-After 值，转换为毫秒。

    支持两种 Retry-After 格式：
    1. 数值格式：直接表示秒数或毫秒数（如 "5" 或 "5000"）
    2. HTTP 日期格式：表示重试的目标时间（如 "Fri, 22 May 2026 12:00:00 GMT"）

    同时支持 OpenAI 风格的 retry-after-ms 毫秒级头部。

    Args:
        exc: 异常对象

    Returns:
        重试等待时间（毫秒），未找到或解析失败则返回 None
    """
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None

    # 按优先级查找 Retry-After 相关头部
    raw = None
    header_name = ""
    for key in ("retry-after-ms", "Retry-After-Ms", "retry-after", "Retry-After"):
        header_name = key
        if hasattr(headers, "get"):
            raw = headers.get(key)
        if raw:
            break
    if not raw:
        return None

    try:
        # 数值格式：如果是毫秒级头部则直接使用，否则乘以 1000 转换为毫秒
        multiplier = 1 if "ms" in header_name.lower() else 1000
        return max(0, int(float(raw) * multiplier))
    except (TypeError, ValueError):
        # 数值解析失败，尝试作为 HTTP 日期格式解析
        try:
            target = parsedate_to_datetime(str(raw))
            # 计算距离当前时间的毫秒差
            delta = target.timestamp() - time.time()
            return max(0, int(delta * 1000))
        except (TypeError, ValueError, OverflowError):
            return None


def _extract_error_detail(exc: BaseException) -> str:
    """从异常对象中提取可读的错误详情文本。

    按优先级尝试：
    1. str(exc) 的结果（非空时使用）
    2. exc.message 属性（非空时使用）
    3. 异常类名作为兜底

    Args:
        exc: 异常对象

    Returns:
        错误详情文本
    """
    detail = str(exc).strip()
    if detail:
        return detail
    message = getattr(exc, "message", None)
    if isinstance(message, str) and message.strip():
        return message.strip()
    # 以上都为空时，返回异常类名作为兜底
    return exc.__class__.__name__
