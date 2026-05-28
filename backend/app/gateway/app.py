"""DeerFlow API 网关主应用。

本模块是 DeerFlow 项目的 FastAPI 网关入口，负责：
- 配置和管理所有 API 路由
- 处理应用生命周期（启动/关闭）
- 设置中间件（认证、CSRF、CORS）
- 管理 LangGraph 运行时初始化
- 处理首次启动的管理员引导和孤儿线程迁移

架构说明:
    - LangGraph 兼容的请求通过 nginx 路由到此网关
    - 网关提供代理运行的运行时端点
    - 以及模型、MCP 配置、技能、工件等自定义端点
"""
import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.gateway.auth_middleware import AuthMiddleware
from app.gateway.config import get_gateway_config
from app.gateway.csrf_middleware import CSRFMiddleware, get_configured_cors_origins
from app.gateway.deps import langgraph_runtime
from app.gateway.routers import (
    agents,
    artifacts,
    assistants_compat,
    auth,
    channels,
    feedback,
    mcp,
    memory,
    models,
    runs,
    skills,
    suggestions,
    thread_runs,
    threads,
    uploads,
)
from deerflow.config import app_config as deerflow_app_config
from deerflow.config.app_config import apply_logging_level

AppConfig = deerflow_app_config.AppConfig
get_app_config = deerflow_app_config.get_app_config

# 默认日志配置；lifespan 会从 config.yaml 的 log_level 覆盖此设置。
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

# 每个 lifespan 关闭钩子允许运行的上限时间（秒）。
# 限制工作进程退出时间，防止 uvicorn 的重载监督器持续
# 向卡在等待关闭清理的工作进程发送信号。
_SHUTDOWN_HOOK_TIMEOUT_SECONDS = 5.0


async def _ensure_admin_user(app: FastAPI) -> None:
    """启动钩子：处理首次启动并迁移孤儿线程。
    
    管理员创建后的行为:
        从 LangGraph 存储中迁移孤儿线程（metadata.user_id 未设置）到管理员账户。
        这是"无认证 → 有认证"的升级路径：在没有身份验证的情况下运行 DeerFlow 
        的用户需要为现有的 LangGraph 线程数据分配所有者。
    
    首次启动（不存在管理员）:
        - 不会自动创建任何用户账户
        - 操作员必须访问 /setup 来创建第一个管理员
    
    后续启动（管理员已存在）:
        - 运行一次性的"无认证 → 有认证"孤儿线程迁移
        - 针对没有 user_id 的现有 LangGraph 线程元数据
    
    不需要 SQL 持久化迁移：四个 user_id 列（threads_meta、runs、
    run_events、feedback）仅随 auth 模块通过 create_all 一起出现，
    因此新创建的表永远不会包含 NULL 所有者行。
    
    Args:
        app: FastAPI 应用实例
        
    Note:
        - 必须在 langgraph_runtime 之后调用，以便 app.state.store 可用
        - 如果认证持久化未就绪，会跳过而非失败
    """
    from sqlalchemy import select

    from app.gateway.deps import get_local_provider
    from deerflow.persistence.engine import get_session_factory
    from deerflow.persistence.user.model import UserRow

    try:
        provider = get_local_provider()
    except RuntimeError:
        # 在某些测试/启动路径中，认证持久化可能尚未初始化。
        # 跳过管理员迁移工作而不是让网关启动失败。
        logger.warning("Auth persistence not ready; skipping admin bootstrap check")
        return

    sf = get_session_factory()
    if sf is None:
        return

    admin_count = await provider.count_admin_users()

    if admin_count == 0:
        logger.info("=" * 60)
        logger.info("  检测到首次启动 — 不存在管理员账户。")
        logger.info("  请访问 /setup 完成管理员账户创建。")
        logger.info("=" * 60)
        return

    # 管理员已存在 — 运行孤儿线程迁移以处理任何
    # 早于认证模块的 LangGraph 线程元数据。
    async with sf() as session:
        stmt = select(UserRow).where(UserRow.system_role == "admin").limit(1)
        row = (await session.execute(stmt)).scalar_one_or_none()

    if row is None:
        return  # 不应发生（上面 admin_count > 0），但为了安全起见。

    admin_id = str(row.id)

    # LangGraph 存储孤儿迁移 — 非致命错误。
    # 这涵盖了"无认证 → 有认证"的升级路径，适用于
    # 现有 LangGraph 线程元数据没有设置 user_id 的用户。
    store = getattr(app.state, "store", None)
    if store is not None:
        try:
            migrated = await _migrate_orphaned_threads(store, admin_id)
            if migrated:
                logger.info("Migrated %d orphan LangGraph thread(s) to admin", migrated)
        except Exception:
            logger.exception("LangGraph thread migration failed (non-fatal)")


async def _iter_store_items(store, namespace, *, page_size: int = 500):
    """LangGraph 存储命名空间的异步分页迭代器。
    
    使用游标风格的循环替换旧的硬编码 limit=1000 调用，
    以便超过一页的孤儿环境不会静默丢失数据。
    当页面为空或短页到达（表示最后一页）时终止。
    
    Args:
        store: LangGraph 存储实例
        namespace: 要迭代的命名空间元组，如 ("threads",)
        page_size: 每页项目数，默认 500
        
    Yields:
        存储中的项目对象
        
    Note:
        - 使用 offset-based 分页
        - 自动检测最后一页（返回的项目数 < page_size）
    """
    offset = 0
    while True:
        batch = await store.asearch(namespace, limit=page_size, offset=offset)
        if not batch:
            return
        for item in batch:
            yield item
        if len(batch) < page_size:
            return
        offset += page_size


async def _migrate_orphaned_threads(store, admin_user_id: str) -> int:
    """将没有 user_id 的 LangGraph 存储线程迁移到指定的管理员。
    
    使用游标分页，无论数量多少都能迁移所有孤儿线程。
    
    Args:
        store: LangGraph 存储实例
        admin_user_id: 管理员用户 ID
        
    Returns:
        迁移的行数
        
    Note:
        - 只迁移 metadata.user_id 为空或未设置的线程
        - 更新后直接保存到存储
    """
    migrated = 0
    async for item in _iter_store_items(store, ("threads",)):
        metadata = item.value.get("metadata", {})
        if not metadata.get("user_id"):
            metadata["user_id"] = admin_user_id
            item.value["metadata"] = metadata
            await store.aput(("threads",), item.key, item.value)
            migrated += 1
    return migrated


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """应用生命周期处理器。
    
    管理应用的启动和关闭流程：
    1. 加载配置并设置日志级别
    2. 初始化 LangGraph 运行时组件
    3. 检查管理员引导状态并迁移孤儿线程
    4. 启动 IM 渠道服务
    5. 关闭时停止渠道服务（带超时限制）
    
    Args:
        app: FastAPI 应用实例
        
    Yields:
        None（在启动和关闭之间）
        
    Raises:
        RuntimeError: 如果配置加载失败
        
    Note:
        - 必须在 langgraph_runtime 之后调用 _ensure_admin_user
        - 关闭钩子有超时限制，防止工作进程挂起
    """

    # 启动时加载配置并检查必要的环境变量
    try:
        app.state.config = get_app_config()
        apply_logging_level(app.state.config.log_level)
        logger.info("Configuration loaded successfully")
    except Exception as e:
        error_msg = f"Failed to load configuration during gateway startup: {e}"
        logger.exception(error_msg)
        raise RuntimeError(error_msg) from e

    config = get_gateway_config()
    logger.info(f"Starting API Gateway on {config.host}:{config.port}")

    # 初始化 LangGraph 运行时组件（StreamBridge、RunManager、checkpointer、store）
    async with langgraph_runtime(app):
        logger.info("LangGraph runtime initialised")

        # 在管理员存在后，检查管理员引导状态并迁移孤儿线程。
        # 必须在 langgraph_runtime 之后运行，以便 app.state.store 可用于线程迁移
        await _ensure_admin_user(app)

        # 如果配置了任何渠道，则启动 IM 渠道服务
        try:
            from app.channels.service import start_channel_service

            channel_service = await start_channel_service(app.state.config)
            logger.info("Channel service started: %s", channel_service.get_status())
        except Exception:
            logger.exception("No IM channels configured or channel service failed to start")

        yield

        # 关闭时停止渠道服务（带超时限制以防止工作进程挂起）
        try:
            from app.channels.service import stop_channel_service

            await asyncio.wait_for(
                stop_channel_service(),
                timeout=_SHUTDOWN_HOOK_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            logger.warning(
                "Channel service shutdown exceeded %.1fs; proceeding with worker exit.",
                _SHUTDOWN_HOOK_TIMEOUT_SECONDS,
            )
        except Exception:
            logger.exception("Failed to stop channel service")

    logger.info("Shutting down API Gateway")


def create_app() -> FastAPI:
    """创建并配置 FastAPI 应用。
    
    负责：
    - 创建 FastAPI 实例并设置元数据
    - 配置中间件（认证、CSRF、CORS）
    - 注册所有 API 路由
    - 添加健康检查端点
    
    Returns:
        配置好的 FastAPI 应用实例
        
    Note:
        - 文档 URL 根据配置启用或禁用
        - 中间件按特定顺序添加：Auth → CSRF → CORS
        - 所有路由器在模块级别导入，避免循环依赖
    """
    config = get_gateway_config()
    docs_url = "/docs" if config.enable_docs else None
    redoc_url = "/redoc" if config.enable_docs else None
    openapi_url = "/openapi.json" if config.enable_docs else None

    app = FastAPI(
        title="DeerFlow API Gateway",
        description="""
## DeerFlow API 网关

基于 LangGraph 的 AI 代理后端的 API 网关，具有沙盒执行能力。

### 功能特性

- **模型管理**：查询和检索可用的 AI 模型
- **MCP 配置**：管理模型上下文协议（MCP）服务器配置
- **记忆管理**：访问和管理全局记忆数据以实现个性化对话
- **技能管理**：查询和管理技能及其启用状态
- **工件**：访问线程工件和生成的文件
- **健康监控**：系统健康检查端点

### 架构

LangGraph 兼容的请求通过 nginx 路由到此网关。
此网关提供代理运行的运行时端点，以及模型、MCP 配置、技能和工件的自定义端点。
        """,
        version="0.1.0",
        lifespan=lifespan,
        docs_url=docs_url,
        redoc_url=redoc_url,
        openapi_url=openapi_url,
        openapi_tags=[
            {
                "name": "models",
                "description": "查询可用 AI 模型及其配置的操作",
            },
            {
                "name": "mcp",
                "description": "管理模型上下文协议（MCP）服务器配置",
            },
            {
                "name": "memory",
                "description": "访问和管理全局记忆数据以实现个性化对话",
            },
            {
                "name": "skills",
                "description": "管理技能及其配置",
            },
            {
                "name": "artifacts",
                "description": "访问和下载线程工件及生成的文件",
            },
            {
                "name": "uploads",
                "description": "上传和管理用户的线程文件",
            },
            {
                "name": "threads",
                "description": "管理 DeerFlow 线程本地文件系统数据",
            },
            {
                "name": "agents",
                "description": "创建和管理具有每代理配置和提示的自定义代理",
            },
            {
                "name": "suggestions",
                "description": "为对话生成后续问题建议",
            },
            {
                "name": "channels",
                "description": "管理 IM 渠道集成（飞书、Slack、Telegram）",
            },
            {
                "name": "assistants-compat",
                "description": "LangGraph Platform 兼容的助手 API（存根）",
            },
            {
                "name": "runs",
                "description": "LangGraph Platform 兼容的运行生命周期（创建、流式传输、取消）",
            },
            {
                "name": "health",
                "description": "健康检查和系统状态端点",
            },
        ],
    )

    # 认证：拒绝未认证的请求访问非公共路径（fail-closed 安全网）
    app.add_middleware(AuthMiddleware)

    # CSRF：双重提交 Cookie 模式用于状态变更请求
    app.add_middleware(CSRFMiddleware)

    # CORS：统一的 nginx 端点默认是同源的。分离源浏览器客户端必须通过此显式网关
    # 允许列表来选择，以便 CORS 和 CSRF 来源检查共享相同的真实来源。
    cors_origins = sorted(get_configured_cors_origins())
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # 注册路由
    # Models API 挂载在 /api/models
    app.include_router(models.router)

    # MCP API 挂载在 /api/mcp
    app.include_router(mcp.router)

    # Memory API 挂载在 /api/memory
    app.include_router(memory.router)

    # Skills API 挂载在 /api/skills
    app.include_router(skills.router)

    # Artifacts API 挂载在 /api/threads/{thread_id}/artifacts
    app.include_router(artifacts.router)

    # Uploads API 挂载在 /api/threads/{thread_id}/uploads
    app.include_router(uploads.router)

    # Thread cleanup API 挂载在 /api/threads/{thread_id}
    app.include_router(threads.router)

    # Agents API 挂载在 /api/agents
    app.include_router(agents.router)

    # Suggestions API 挂载在 /api/threads/{thread_id}/suggestions
    app.include_router(suggestions.router)

    # Channels API 挂载在 /api/channels
    app.include_router(channels.router)

    # Assistants compatibility API（LangGraph Platform 存根）
    app.include_router(assistants_compat.router)

    # Auth API 挂载在 /api/v1/auth
    app.include_router(auth.router)

    # Feedback API 挂载在 /api/threads/{thread_id}/runs/{run_id}/feedback
    app.include_router(feedback.router)

    # Thread Runs API（LangGraph Platform 兼容的运行生命周期）
    app.include_router(thread_runs.router)

    # Stateless Runs API（无需预先存在的线程即可流式传输/等待）
    app.include_router(runs.router)

    @app.get("/health", tags=["health"])
    async def health_check() -> dict[str, str]:
        """健康检查端点。
        
        Returns:
            服务健康状态信息
        """
        return {"status": "healthy", "service": "deer-flow-gateway"}

    return app

# 为 uvicorn 创建应用实例
app = create_app()
