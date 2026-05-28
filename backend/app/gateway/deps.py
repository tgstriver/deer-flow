"""集中访问存储在 ``app.state`` 上的单例对象。

**获取器**（由路由器使用）：当必需的依赖项缺失时抛出 503 错误，
除了 ``get_store`` 返回 ``None``。

初始化直接在 ``app.py`` 中通过 :class:`AsyncExitStack` 处理。

本模块提供：
- LangGraph 运行时组件的初始化和清理
- FastAPI 依赖注入函数，用于在请求中访问共享资源
- 认证相关的辅助函数
- 线程安全的单例缓存机制
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from typing import TYPE_CHECKING, TypeVar, cast

from fastapi import FastAPI, HTTPException, Request
from langgraph.types import Checkpointer

from deerflow.config.app_config import AppConfig
from deerflow.persistence.feedback import FeedbackRepository
from deerflow.runtime import RunContext, RunManager, StreamBridge
from deerflow.runtime.events.store.base import RunEventStore
from deerflow.runtime.runs.store.base import RunStore

if TYPE_CHECKING:
    from app.gateway.auth.local_provider import LocalAuthProvider
    from app.gateway.auth.repositories.sqlite import SQLiteUserRepository
    from deerflow.persistence.thread_meta.base import ThreadMetaStore


T = TypeVar("T")


def get_config(request: Request) -> AppConfig:
    """返回存储在 ``app.state`` 上的应用级 ``AppConfig``。
    
    Args:
        request: FastAPI 请求对象
        
    Returns:
        应用配置对象
        
    Raises:
        HTTPException: 如果配置不可用（503 状态码）
        
    Note:
        - 配置在应用启动时加载并存储在 app.state.config
        - 所有需要配置的路由器都应该使用此函数
    """
    config = getattr(request.app.state, "config", None)
    if config is None:
        raise HTTPException(status_code=503, detail="Configuration not available")
    return config


@asynccontextmanager
async def langgraph_runtime(app: FastAPI) -> AsyncGenerator[None, None]:
    """引导和拆除所有 LangGraph 运行时单例。
    
    负责初始化和清理以下组件：
    - StreamBridge：流式传输桥接器
    - Checkpointer：检查点管理器
    - Store：全局存储
    - RunRepository / MemoryRunStore：运行存储
    - FeedbackRepository：反馈仓库
    - ThreadMetaStore：线程元数据存储
    - RunEventStore：运行事件存储
    - RunManager：运行管理器
    
    使用示例（在 ``app.py`` 中）::
    
        async with langgraph_runtime(app):
            yield
    
    Args:
        app: FastAPI 应用实例
        
    Yields:
        None（在初始化和清理之间）
        
    Raises:
        RuntimeError: 如果 app.state.config 未初始化
        
    Note:
        - 必须在配置加载后调用
        - 使用 AsyncExitStack 确保正确的资源清理顺序
        - 持久化引擎在 checkpointer 之前初始化（postgres 后端自动创建数据库）
        - 关闭时会调用 close_engine() 清理数据库连接
    """
    from deerflow.persistence.engine import close_engine, get_session_factory, init_engine_from_config
    from deerflow.runtime import make_store, make_stream_bridge
    from deerflow.runtime.checkpointer.async_provider import make_checkpointer
    from deerflow.runtime.events.store import make_run_event_store

    async with AsyncExitStack() as stack:
        config = getattr(app.state, "config", None)
        if config is None:
            raise RuntimeError("langgraph_runtime() requires app.state.config to be initialized")

        # 初始化流式传输桥接器
        app.state.stream_bridge = await stack.enter_async_context(make_stream_bridge(config))

        # 在 checkpointer 之前初始化持久化引擎，以便自动创建数据库逻辑先运行（postgres 后端）
        await init_engine_from_config(config.database)

        # 初始化检查点管理器和全局存储
        app.state.checkpointer = await stack.enter_async_context(make_checkpointer(config))
        app.state.store = await stack.enter_async_context(make_store(config))

        # 初始化仓库 —— 所有仓库共用一个 get_session_factory() 调用
        sf = get_session_factory()
        if sf is not None:
            from deerflow.persistence.feedback import FeedbackRepository
            from deerflow.persistence.run import RunRepository

            app.state.run_store = RunRepository(sf)
            app.state.feedback_repo = FeedbackRepository(sf)
        else:
            # 如果没有会话工厂，使用内存存储
            from deerflow.runtime.runs.store.memory import MemoryRunStore

            app.state.run_store = MemoryRunStore()
            app.state.feedback_repo = None

        # 初始化线程元数据存储
        from deerflow.persistence.thread_meta import make_thread_store

        app.state.thread_store = make_thread_store(sf, app.state.store)

        # 运行事件存储（有自己的工厂，支持配置驱动的后端选择）
        run_events_config = getattr(config, "run_events", None)
        app.state.run_event_store = make_run_event_store(run_events_config)

        # 带存储支持的 RunManager，用于持久化
        app.state.run_manager = RunManager(store=app.state.run_store)

        try:
            yield
        finally:
            # 关闭时清理数据库引擎
            await close_engine()


# ---------------------------------------------------------------------------
# 获取器 —— 由路由器在每个请求中调用
# ---------------------------------------------------------------------------


def _require(attr: str, label: str) -> Callable[[Request], T]:
    """创建一个 FastAPI 依赖项，返回 ``app.state.<attr>`` 或在缺失时抛出 503。
    
    Args:
        attr: app.state 上的属性名
        label: 用于错误消息的标签
        
    Returns:
        依赖函数，接受 Request 并返回指定类型的值
        
    Raises:
        HTTPException: 如果属性不存在（503 状态码）
        
    Note:
        - 使用闭包捕获 attr 和 label
        - 动态设置函数名以便调试
        - 泛型类型 T 允许返回任意类型
    """

    def dep(request: Request) -> T:
        val = getattr(request.app.state, attr, None)
        if val is None:
            raise HTTPException(status_code=503, detail=f"{label} not available")
        return cast(T, val)

    dep.__name__ = dep.__qualname__ = f"get_{attr}"
    return dep


# 预定义的依赖注入函数，用于访问各种运行时组件
get_stream_bridge: Callable[[Request], StreamBridge] = _require("stream_bridge", "Stream bridge")
get_run_manager: Callable[[Request], RunManager] = _require("run_manager", "Run manager")
get_checkpointer: Callable[[Request], Checkpointer] = _require("checkpointer", "Checkpointer")
get_run_event_store: Callable[[Request], RunEventStore] = _require("run_event_store", "Run event store")
get_feedback_repo: Callable[[Request], FeedbackRepository] = _require("feedback_repo", "Feedback")
get_run_store: Callable[[Request], RunStore] = _require("run_store", "Run store")


def get_store(request: Request):
    """返回全局存储（如果未配置则可能为 ``None``）。
    
    Args:
        request: FastAPI 请求对象
        
    Returns:
        全局存储实例或 None
        
    Note:
        - 与其他获取器不同，此函数在缺失时不抛出异常
        - 某些场景下存储是可选的
    """
    return getattr(request.app.state, "store", None)


def get_thread_store(request: Request) -> ThreadMetaStore:
    """返回线程元数据存储（SQL 或内存支持）。
    
    Args:
        request: FastAPI 请求对象
        
    Returns:
        线程元数据存储实例
        
    Raises:
        HTTPException: 如果线程存储不可用（503 状态码）
        
    Note:
        - 支持 SQL 后端和内存后端
        - 必须在 langgraph_runtime 之后可用
    """
    val = getattr(request.app.state, "thread_store", None)
    if val is None:
        raise HTTPException(status_code=503, detail="Thread metadata store not available")
    return val


def get_run_context(request: Request) -> RunContext:
    """从 ``app.state`` 单例构建 :class:`RunContext`。
    
    返回一个带有基础设施依赖的*基础*上下文。
    
    Args:
        request: FastAPI 请求对象
        
    Returns:
        运行上下文对象，包含所有必需的依赖项
        
    Note:
        - 组合了 checkpointer、store、event_store、thread_store 等组件
        - 用于传递给 LangGraph 代理执行
    """
    config = get_config(request)
    return RunContext(
        checkpointer=get_checkpointer(request),
        store=get_store(request),
        event_store=get_run_event_store(request),
        run_events_config=getattr(config, "run_events", None),
        thread_store=get_thread_store(request),
        app_config=config,
    )


# ---------------------------------------------------------------------------
# 认证辅助函数（由 authz.py 和认证中间件使用）
# ---------------------------------------------------------------------------

# 缓存的单例，避免每个请求重复实例化
_cached_local_provider: LocalAuthProvider | None = None
_cached_repo: SQLiteUserRepository | None = None


def get_local_provider() -> LocalAuthProvider:
    """获取或创建缓存的 LocalAuthProvider 单例。
    
    必须在 ``init_engine_from_config()`` 之后调用 —— 需要共享的会话工厂来构造用户仓库。
    
    Returns:
        本地认证提供者实例
        
    Raises:
        RuntimeError: 如果在初始化引擎之前调用
        
    Note:
        - 使用全局变量缓存，避免重复创建
        - 首次调用时初始化仓库和提供者
        - 后续调用直接返回缓存实例
        - 线程安全：Python GIL 保证简单赋值的原子性
    """
    global _cached_local_provider, _cached_repo
    if _cached_repo is None:
        from app.gateway.auth.repositories.sqlite import SQLiteUserRepository
        from deerflow.persistence.engine import get_session_factory

        sf = get_session_factory()
        if sf is None:
            raise RuntimeError("get_local_provider() called before init_engine_from_config(); cannot access users table")
        _cached_repo = SQLiteUserRepository(sf)
    if _cached_local_provider is None:
        from app.gateway.auth.local_provider import LocalAuthProvider

        _cached_local_provider = LocalAuthProvider(repository=_cached_repo)
    return _cached_local_provider


async def get_current_user_from_request(request: Request):
    """从请求 cookie 中获取当前认证用户。
    
    Args:
        request: FastAPI 请求对象
        
    Returns:
        用户对象
        
    Raises:
        HTTPException: 
            - 401 如果未认证（没有 access_token cookie）
            - 401 如果 token 无效或过期
            - 401 如果用户不存在
            - 401 如果 token 版本不匹配（密码已更改）
            
    Note:
        - 从 cookie 中读取 access_token
        - 解码并验证 token
        - 检查用户是否存在
        - 验证 token 版本是否与数据库中的 token_version 匹配
        - 版本不匹配表示密码已更改，token 已失效
    """
    from app.gateway.auth import decode_token
    from app.gateway.auth.errors import AuthErrorCode, AuthErrorResponse, TokenError, token_error_to_code

    access_token = request.cookies.get("access_token")
    if not access_token:
        raise HTTPException(
            status_code=401,
            detail=AuthErrorResponse(code=AuthErrorCode.NOT_AUTHENTICATED, message="Not authenticated").model_dump(),
        )

    payload = decode_token(access_token)
    if isinstance(payload, TokenError):
        raise HTTPException(
            status_code=401,
            detail=AuthErrorResponse(code=token_error_to_code(payload), message=f"Token error: {payload.value}").model_dump(),
        )

    provider = get_local_provider()
    user = await provider.get_user(payload.sub)
    if user is None:
        raise HTTPException(
            status_code=401,
            detail=AuthErrorResponse(code=AuthErrorCode.USER_NOT_FOUND, message="User not found").model_dump(),
        )

    # Token 版本不匹配 → 密码已更改，token 已过期
    if user.token_version != payload.ver:
        raise HTTPException(
            status_code=401,
            detail=AuthErrorResponse(code=AuthErrorCode.TOKEN_INVALID, message="Token revoked (password changed)").model_dump(),
        )

    return user


async def get_optional_user_from_request(request: Request):
    """从请求中获取可选的认证用户。
    
    Args:
        request: FastAPI 请求对象
        
    Returns:
        用户对象或 None（如果未认证）
        
    Note:
        - 内部调用 get_current_user_from_request
        - 捕获 HTTPException 并返回 None
        - 用于不需要强制认证的场景
    """
    try:
        return await get_current_user_from_request(request)
    except HTTPException:
        return None


async def get_current_user(request: Request) -> str | None:
    """从请求 cookie 中提取 user_id，如果未认证则返回 None。
    
    轻量级适配器，返回字符串 ID 供只需要标识的调用者使用（例如 ``feedback.py``）。
    需要完整用户对象的调用者应该使用 ``get_current_user_from_request`` 或 
    ``get_optional_user_from_request``。
    
    Args:
        request: FastAPI 请求对象
        
    Returns:
        用户 ID 字符串或 None
        
    Note:
        - 内部调用 get_optional_user_from_request
        - 只返回用户 ID，不返回完整用户对象
        - 适用于只需要用户标识的场景
    """
    user = await get_optional_user_from_request(request)
    return str(user.id) if user else None
