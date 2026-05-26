"""线程运行（Runs）API 端点 — 创建、流式传输、等待、取消。

本模块基于 :class:`deerflow.agents.runs.RunManager` 和
:class:`deerflow.agents.stream_bridge.StreamBridge` 实现 LangGraph Platform
的 runs API。

SSE（Server-Sent Events）格式与 LangGraph Platform 协议对齐，使得
来自 ``@langchain/langgraph-sdk/react`` 的 ``useStream`` React hook
可以无需修改直接使用。

主要功能：
- 创建后台运行任务（立即返回）
- 通过 SSE 流式传输运行事件
- 阻塞等待运行完成并返回最终状态
- 列出和管理线程的运行历史
- 取消正在运行或待处理的运行任务
- 加入现有运行的 SSE 流
- 查询消息、事件和 Token 使用情况
"""

# 导入标准库和第三方依赖
from __future__ import annotations

import asyncio
import logging
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

# 导入网关内部模块
from app.gateway.authz import require_permission
from app.gateway.deps import get_checkpointer, get_current_user, get_feedback_repo, get_run_event_store, get_run_manager, get_run_store, get_stream_bridge
from app.gateway.services import sse_consumer, start_run
from deerflow.runtime import RunRecord, RunStatus, serialize_channel_values

logger = logging.getLogger(__name__)
# 创建路由器，所有端点都以 /api/threads 为前缀
router = APIRouter(prefix="/api/threads", tags=["runs"])


# ---------------------------------------------------------------------------
# 请求/响应数据模型
# ---------------------------------------------------------------------------


class RunCreateRequest(BaseModel):
    """创建运行的请求体模型。

    包含启动 LangGraph 运行所需的所有配置选项，支持高级功能如断点续传、
    多任务策略、延迟执行等。
    """

    assistant_id: str | None = Field(default=None, description="要使用的代理/助手 ID")
    input: dict[str, Any] | None = Field(default=None, description="图输入（例如 {messages: [...]}）")
    command: dict[str, Any] | None = Field(default=None, description="LangGraph Command 对象")
    metadata: dict[str, Any] | None = Field(default=None, description="运行元数据")
    config: dict[str, Any] | None = Field(default=None, description="RunnableConfig 覆盖配置")
    context: dict[str, Any] | None = Field(default=None, description="DeerFlow 上下文覆盖（model_name、thinking_enabled 等）")
    webhook: str | None = Field(default=None, description="完成回调 URL")
    checkpoint_id: str | None = Field(default=None, description="从指定检查点恢复")
    checkpoint: dict[str, Any] | None = Field(default=None, description="完整检查点对象")
    interrupt_before: list[str] | Literal["*"] | None = Field(default=None, description="在这些节点之前中断")
    interrupt_after: list[str] | Literal["*"] | None = Field(default=None, description="在这些节点之后中断")
    stream_mode: list[str] | str | None = Field(default=None, description="流式传输模式")
    stream_subgraphs: bool = Field(default=False, description="包含子图事件")
    stream_resumable: bool | None = Field(default=None, description="SSE 可恢复模式")
    on_disconnect: Literal["cancel", "continue"] = Field(default="cancel", description="SSE 断开时的行为")
    on_completion: Literal["delete", "keep"] = Field(default="keep", description="完成后是否删除临时线程")
    multitask_strategy: Literal["reject", "rollback", "interrupt", "enqueue"] = Field(default="reject", description="并发策略")
    after_seconds: float | None = Field(default=None, description="延迟执行秒数")
    if_not_exists: Literal["reject", "create"] = Field(default="create", description="线程创建策略")
    feedback_keys: list[str] | None = Field(default=None, description="LangSmith 反馈键")


class RunResponse(BaseModel):
    """运行记录的响应模型。

    表示一个运行的基本信息，用于列表和详情接口返回。
    """

    run_id: str
    thread_id: str
    assistant_id: str | None = None
    status: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    kwargs: dict[str, Any] = Field(default_factory=dict)
    multitask_strategy: str = "reject"
    created_at: str = ""
    updated_at: str = ""


class ThreadTokenUsageModelBreakdown(BaseModel):
    """按模型分类的 Token 使用统计。

    Attributes:
        tokens: 总 Token 数
        runs: 运行次数
    """

    tokens: int = 0
    runs: int = 0


class ThreadTokenUsageCallerBreakdown(BaseModel):
    """按调用者分类的 Token 使用统计。

    Attributes:
        lead_agent: 主代理消耗的 Token
        subagent: 子代理消耗的 Token
        middleware: 中间件消耗的 Token
    """

    lead_agent: int = 0
    subagent: int = 0
    middleware: int = 0


class ThreadTokenUsageResponse(BaseModel):
    """线程级别的 Token 使用统计响应。

    聚合整个线程中所有运行的 Token 消耗情况，按模型和调用者分类。
    """

    thread_id: str
    total_tokens: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_runs: int = 0
    by_model: dict[str, ThreadTokenUsageModelBreakdown] = Field(default_factory=dict)
    by_caller: ThreadTokenUsageCallerBreakdown = Field(default_factory=ThreadTokenUsageCallerBreakdown)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _cancel_conflict_detail(run_id: str, record: RunRecord) -> str:
    """生成取消冲突的详细错误消息。

    当尝试取消一个无法取消的运行时，根据运行状态提供具体的错误说明。

    Args:
        run_id: 运行 ID
        record: 运行记录对象

    Returns:
        描述冲突原因的字符串
    """
    if record.status in (RunStatus.pending, RunStatus.running):
        return f"Run {run_id} is not active on this worker and cannot be cancelled"
    return f"Run {run_id} is not cancellable (status: {record.status.value})"


def _record_to_response(record: RunRecord) -> RunResponse:
    """将内部运行记录转换为 API 响应模型。

    Args:
        record: 内部运行记录对象

    Returns:
        符合 API 规范的响应对象
    """
    return RunResponse(
        run_id=record.run_id,
        thread_id=record.thread_id,
        assistant_id=record.assistant_id,
        status=record.status.value,
        metadata=record.metadata,
        kwargs=record.kwargs,
        multitask_strategy=record.multitask_strategy,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


# ---------------------------------------------------------------------------
# API 端点
# ---------------------------------------------------------------------------


@router.post("/{thread_id}/runs", response_model=RunResponse)
@require_permission("runs", "create", owner_check=True, require_existing=True)
async def create_run(thread_id: str, body: RunCreateRequest, request: Request) -> RunResponse:
    """创建后台运行任务（立即返回）。

    启动一个新的运行任务并在后台执行，立即返回运行记录而不等待完成。
    适用于需要异步执行的场景。

    Args:
        thread_id: 线程 ID
        body: 运行创建请求体
        request: FastAPI 请求对象

    Returns:
        运行记录响应

    Permissions:
        需要 'runs.create' 权限，且必须是线程所有者，线程必须已存在
    """
    record = await start_run(body, thread_id, request)
    return _record_to_response(record)


@router.post("/{thread_id}/runs/stream")
@require_permission("runs", "create", owner_check=True, require_existing=True)
async def stream_run(thread_id: str, body: RunCreateRequest, request: Request) -> StreamingResponse:
    """创建运行并通过 SSE 流式传输事件。

    启动运行任务并建立 SSE 连接，实时推送运行过程中的事件（如消息、工具调用、状态更新等）。

    响应包含 ``Content-Location`` 头，指向运行的资源 URL，符合LangGraph Platform 协议。
    SDK 的 ``useStream`` React hook 使用此头来提取运行元数据。

    Args:
        thread_id: 线程 ID
        body: 运行创建请求体
        request: FastAPI 请求对象

    Returns:
        SSE 流式响应

    Headers:
        Content-Location: 运行的资源路径，供 SDK 提取 run_id
        Cache-Control: no-cache（禁用缓存）
        X-Accel-Buffering: no（禁用 nginx 缓冲）
    """
    bridge = get_stream_bridge(request)
    run_mgr = get_run_manager(request)
    record = await start_run(body, thread_id, request)

    return StreamingResponse(
        sse_consumer(bridge, record, request, run_mgr),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            # LangGraph Platform 在此头中包含运行元数据
            # SDK 使用贪婪正则表达式从此路径中提取 run_id，
            # 因此它必须指向标准的运行资源路径，不能有额外后缀
            "Content-Location": f"/api/threads/{thread_id}/runs/{record.run_id}",
        },
    )


@router.post("/{thread_id}/runs/wait", response_model=dict)
@require_permission("runs", "create", owner_check=True, require_existing=True)
async def wait_run(thread_id: str, body: RunCreateRequest, request: Request) -> dict:
    """创建运行并阻塞直到完成，返回最终状态。

    启动运行任务并同步等待其完成，然后从检查点获取最终的通道值
    （channel values），即图的最终状态。

    适用于需要立即获取结果的场景，但会占用连接直到运行完成。

    Args:
        thread_id: 线程 ID
        body: 运行创建请求体
        request: FastAPI 请求对象

    Returns:
        最终的通道值（序列化后的图状态），如果获取失败则返回状态和错误信息
    """
    record = await start_run(body, thread_id, request)

    # 等待运行任务完成（如果任务存在）
    if record.task is not None:
        try:
            await record.task
        except asyncio.CancelledError:
            pass

    # 从检查点获取最终状态
    checkpointer = get_checkpointer(request)
    config = {"configurable": {"thread_id": thread_id}}
    try:
        checkpoint_tuple = await checkpointer.aget_tuple(config)
        if checkpoint_tuple is not None:
            checkpoint = getattr(checkpoint_tuple, "checkpoint", {}) or {}
            channel_values = checkpoint.get("channel_values", {})
            return serialize_channel_values(channel_values)
    except Exception:
        logger.exception("Failed to fetch final state for run %s", record.run_id)

    return {"status": record.status.value, "error": record.error}


@router.get("/{thread_id}/runs", response_model=list[RunResponse])
@require_permission("runs", "read", owner_check=True)
async def list_runs(thread_id: str, request: Request) -> list[RunResponse]:
    """列出线程的所有运行记录。

    返回指定线程的历史运行记录列表，按时间排序。

    Args:
        thread_id: 线程 ID
        request: FastAPI 请求对象

    Returns:
        运行记录列表

    Permissions:
        需要 'runs.read' 权限，且必须是线程所有者
    """
    run_mgr = get_run_manager(request)
    user_id = await get_current_user(request)
    records = await run_mgr.list_by_thread(thread_id, user_id=user_id)
    return [_record_to_response(r) for r in records]


@router.get("/{thread_id}/runs/{run_id}", response_model=RunResponse)
@require_permission("runs", "read", owner_check=True)
async def get_run(thread_id: str, run_id: str, request: Request) -> RunResponse:
    """获取特定运行的详细信息。

    Args:
        thread_id: 线程 ID
        run_id: 运行 ID
        request: FastAPI 请求对象

    Returns:
        运行记录详情

    Raises:
        HTTPException: 404 如果运行不存在或不属于指定线程

    Permissions:
        需要 'runs.read' 权限，且必须是线程所有者
    """
    run_mgr = get_run_manager(request)
    user_id = await get_current_user(request)
    record = await run_mgr.get(run_id, user_id=user_id)
    if record is None or record.thread_id != thread_id:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return _record_to_response(record)


@router.post("/{thread_id}/runs/{run_id}/cancel")
@require_permission("runs", "cancel", owner_check=True, require_existing=True)
async def cancel_run(
    thread_id: str,
    run_id: str,
    request: Request,
    wait: bool = Query(default=False, description="取消后阻塞直到运行完全停止"),
    action: Literal["interrupt", "rollback"] = Query(default="interrupt", description="取消动作类型"),
) -> Response:
    """取消正在运行或待处理的运行任务。

    支持两种取消动作：
    - **interrupt**: 停止执行，保留当前检查点（可以从中恢复）
    - **rollback**: 停止执行，回滚到运行前的检查点状态

    支持两种响应模式：
    - **wait=false** (默认): 立即返回 202 Accepted，取消操作在后台进行
    - **wait=true**: 阻塞直到运行完全停止，返回 204 No Content

    Args:
        thread_id: 线程 ID
        run_id: 运行 ID
        request: FastAPI 请求对象
        wait: 是否等待取消完成
        action: 取消动作类型

    Returns:
        202 Accepted（立即返回）或 204 No Content（等待模式）

    Raises:
        HTTPException: 404 如果运行不存在，409 如果无法取消

    Permissions:
        需要 'runs.cancel' 权限，且必须是线程所有者，运行必须已存在
    """
    run_mgr = get_run_manager(request)
    record = await run_mgr.get(run_id)
    if record is None or record.thread_id != thread_id:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")

    # 执行取消操作
    cancelled = await run_mgr.cancel(run_id, action=action)
    if not cancelled:
        raise HTTPException(status_code=409, detail=_cancel_conflict_detail(run_id, record))

    # 如果需要等待，则阻塞直到任务完成
    if wait and record.task is not None:
        try:
            await record.task
        except asyncio.CancelledError:
            pass
        return Response(status_code=204)

    return Response(status_code=202)


@router.get("/{thread_id}/runs/{run_id}/join")
@require_permission("runs", "read", owner_check=True)
async def join_run(thread_id: str, run_id: str, request: Request) -> StreamingResponse:
    """加入现有运行的 SSE 流。

    允许客户端连接到已经在运行的任务的 SSE 流，实时接收后续事件。
    这对于多个客户端监控同一运行或重新连接断开的流很有用。

    Args:
        thread_id: 线程 ID
        run_id: 运行 ID
        request: FastAPI 请求对象

    Returns:
        SSE 流式响应

    Raises:
        HTTPException: 404 如果运行不存在，409 如果运行不在当前 worker 上活跃

    Permissions:
        需要 'runs.read' 权限，且必须是线程所有者
    """
    run_mgr = get_run_manager(request)
    record = await run_mgr.get(run_id)
    if record is None or record.thread_id != thread_id:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    if record.store_only:
        raise HTTPException(status_code=409, detail=f"Run {run_id} is not active on this worker and cannot be streamed")

    bridge = get_stream_bridge(request)
    return StreamingResponse(
        sse_consumer(bridge, record, request, run_mgr),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.api_route("/{thread_id}/runs/{run_id}/stream", methods=["GET", "POST"], response_model=None)
@require_permission("runs", "read", owner_check=True)
async def stream_existing_run(
    thread_id: str,
    run_id: str,
    request: Request,
    action: Literal["interrupt", "rollback"] | None = Query(default=None, description="取消动作类型"),
    wait: int = Query(default=0, description="阻塞直到取消完成 (1) 或立即返回 (0)"),
):
    """加入现有运行的 SSE 流（GET），或先取消再流式传输（POST）。

    这是一个多功能端点，支持两种操作模式：

    **GET 模式**: 直接加入现有运行的 SSE 流，类似于 /join 端点。

    **POST 模式**: LangGraph SDK 的 ``joinStream`` 和 ``useStream`` 停止按钮
    都使用 POST 到此端点。当包含 ``action=interrupt`` 或 ``action=rollback``
    参数时，先取消运行，然后流式传输任何剩余的缓冲事件，使客户端观察到
    干净的关闭过程。

    Args:
        thread_id: 线程 ID
        run_id: 运行 ID
        request: FastAPI 请求对象
        action: 取消动作类型（仅在 POST 时使用）
        wait: 是否等待取消完成（1=等待，0=立即返回）

    Returns:
        SSE 流式响应，或在等待模式下返回 204 No Content

    Raises:
        HTTPException: 404 如果运行不存在，409 如果无法流式传输

    Permissions:
        需要 'runs.read' 权限，且必须是线程所有者
    """
    run_mgr = get_run_manager(request)
    record = await run_mgr.get(run_id)
    if record is None or record.thread_id != thread_id:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    if record.store_only and action is None:
        raise HTTPException(status_code=409, detail=f"Run {run_id} is not active on this worker and cannot be streamed")

    # 如果请求了取消动作，则先取消运行（停止按钮/中断流程）
    if action is not None:
        cancelled = await run_mgr.cancel(run_id, action=action)
        if not cancelled:
            raise HTTPException(status_code=409, detail=_cancel_conflict_detail(run_id, record))
        if wait and record.task is not None:
            try:
                await record.task
            except (asyncio.CancelledError, Exception):
                pass
            return Response(status_code=204)

    # 建立 SSE 连接
    bridge = get_stream_bridge(request)
    return StreamingResponse(
        sse_consumer(bridge, record, request, run_mgr),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# 消息/事件/Token 使用端点
# ---------------------------------------------------------------------------


@router.get("/{thread_id}/messages")
@require_permission("runs", "read", owner_check=True)
async def list_thread_messages(
    thread_id: str,
    request: Request,
    limit: int = Query(default=50, le=200),
    before_seq: int | None = Query(default=None),
    after_seq: int | None = Query(default=None),
) -> list[dict]:
    """返回线程的可显示消息列表（跨所有运行），附带反馈信息。

    获取线程中所有运行的消息，按序列号排序，并为每个运行的最后一条
    AI 消息附加用户反馈（如果有）。

    Args:
        thread_id: 线程 ID
        request: FastAPI 请求对象
        limit: 最大返回消息数（默认 50，最大 200）
        before_seq: 只返回此序列号之前的消息（分页）
        after_seq: 只返回此序列号之后的消息（分页）

    Returns:
        消息列表，每条消息包含 feedback 字段（如果有）

    Permissions:
        需要 'runs.read' 权限，且必须是线程所有者
    """
    event_store = get_run_event_store(request)
    messages = await event_store.list_messages(thread_id, limit=limit, before_seq=before_seq, after_seq=after_seq)

    # 为每个运行的最后一条 AI 消息附加反馈
    feedback_repo = get_feedback_repo(request)
    user_id = await get_current_user(request)
    feedback_map = await feedback_repo.list_by_thread_grouped(thread_id, user_id=user_id)

    # 找到每个运行的最后一条 ai_message 的索引
    last_ai_per_run: dict[str, int] = {}  # run_id -> messages 列表中的索引
    for i, msg in enumerate(messages):
        if msg.get("event_type") == "ai_message":
            last_ai_per_run[msg["run_id"]] = i

    # 附加反馈字段
    last_ai_indices = set(last_ai_per_run.values())
    for i, msg in enumerate(messages):
        if i in last_ai_indices:
            run_id = msg["run_id"]
            fb = feedback_map.get(run_id)
            msg["feedback"] = (
                {
                    "feedback_id": fb["feedback_id"],
                    "rating": fb["rating"],
                    "comment": fb.get("comment"),
                }
                if fb
                else None
            )
        else:
            msg["feedback"] = None

    return messages


@router.get("/{thread_id}/runs/{run_id}/messages")
@require_permission("runs", "read", owner_check=True)
async def list_run_messages(
    thread_id: str,
    run_id: str,
    request: Request,
    limit: int = Query(default=50, le=200, ge=1),
    before_seq: int | None = Query(default=None),
    after_seq: int | None = Query(default=None),
) -> dict:
    """返回特定运行的分页消息列表。

    获取指定运行中的所有消息，支持分页查询。

    Args:
        thread_id: 线程 ID
        run_id: 运行 ID
        request: FastAPI 请求对象
        limit: 每页最大消息数（默认 50，范围 1-200）
        before_seq: 只返回此序列号之前的消息（分页）
        after_seq: 只返回此序列号之后的消息（分页）

    Returns:
        包含 data（消息列表）和 has_more（是否有更多消息）的字典

    Permissions:
        需要 'runs.read' 权限，且必须是线程所有者
    """
    event_store = get_run_event_store(request)
    rows = await event_store.list_messages_by_run(
        thread_id,
        run_id,
        limit=limit + 1,  # 多取一条以判断是否有更多
        before_seq=before_seq,
        after_seq=after_seq,
    )
    has_more = len(rows) > limit
    data = rows[:limit] if has_more else rows
    return {"data": data, "has_more": has_more}


@router.get("/{thread_id}/runs/{run_id}/events")
@require_permission("runs", "read", owner_check=True)
async def list_run_events(
    thread_id: str,
    run_id: str,
    request: Request,
    event_types: str | None = Query(default=None),
    limit: int = Query(default=500, le=2000),
) -> list[dict]:
    """返回运行的完整事件流（用于调试/审计）。

    获取运行过程中产生的所有事件，包括消息、工具调用、状态更新等。
    可用于调试、审计或回放运行过程。

    Args:
        thread_id: 线程 ID
        run_id: 运行 ID
        request: FastAPI 请求对象
        event_types: 可选的事件类型过滤器（逗号分隔，如 "ai_message,tool_call"）
        limit: 最大返回事件数（默认 500，最大 2000）

    Returns:
        事件列表

    Permissions:
        需要 'runs.read' 权限，且必须是线程所有者
    """
    event_store = get_run_event_store(request)
    types = event_types.split(",") if event_types else None
    return await event_store.list_events(thread_id, run_id, event_types=types, limit=limit)


@router.get("/{thread_id}/token-usage", response_model=ThreadTokenUsageResponse)
@require_permission("threads", "read", owner_check=True)
async def thread_token_usage(thread_id: str, request: Request) -> ThreadTokenUsageResponse:
    """线程级别的 Token 使用统计聚合。

    计算整个线程中所有运行的 Token 消耗情况，包括：
    - 总 Token 数、输入 Token 数、输出 Token 数
    - 按模型分类的统计
    - 按调用者分类的统计（主代理、子代理、中间件）

    Args:
        thread_id: 线程 ID
        request: FastAPI 请求对象

    Returns:
        Token 使用统计响应

    Permissions:
        需要 'threads.read' 权限，且必须是线程所有者
    """
    run_store = get_run_store(request)
    agg = await run_store.aggregate_tokens_by_thread(thread_id)
    return ThreadTokenUsageResponse(thread_id=thread_id, **agg)
