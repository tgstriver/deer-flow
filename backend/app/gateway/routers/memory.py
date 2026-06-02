"""记忆 API 路由器，用于检索和管理全局记忆数据。

本模块提供完整的记忆系统 RESTful API 接口，核心功能包括：
- 获取当前全局记忆数据（用户上下文、历史、事实）
- 重新加载记忆数据（从存储文件刷新缓存）
- 清空所有记忆数据
- 手动创建、删除、更新记忆事实
- 导出和导入记忆数据（JSON 格式）
- 获取记忆系统配置和状态

架构说明：
    - 使用 Pydantic 模型定义请求和响应结构
    - 委托给 deerflow.agents.memory.updater 模块处理业务逻辑
    - 支持多用户隔离（通过 get_effective_user_id()）
    - 统一的错误处理和 HTTP 状态码映射

API 端点列表：
    - GET /api/memory: 获取记忆数据
    - POST /api/memory/reload: 重新加载记忆
    - DELETE /api/memory: 清空所有记忆
    - POST /api/memory/facts: 创建记忆事实
    - DELETE /api/memory/facts/{fact_id}: 删除记忆事实
    - PATCH /api/memory/facts/{fact_id}: 部分更新记忆事实
    - GET /api/memory/export: 导出记忆数据
    - POST /api/memory/import: 导入记忆数据
    - GET /api/memory/config: 获取记忆配置
    - GET /api/memory/status: 获取记忆状态（配置 + 数据）

安全特性：
    - 所有操作都基于当前用户的 user_id（多租户隔离）
    - 输入验证（置信度范围 0-1，内容非空）
    - 统一的错误响应格式
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from deerflow.agents.memory.updater import (
    clear_memory_data,
    create_memory_fact,
    delete_memory_fact,
    get_memory_data,
    import_memory_data,
    reload_memory_data,
    update_memory_fact,
)
from deerflow.config.memory_config import get_memory_config
from deerflow.runtime.user_context import get_effective_user_id

# 创建 API 路由器，设置前缀和标签
router = APIRouter(prefix="/api", tags=["memory"])


class ContextSection(BaseModel):
    """上下文节段的 Pydantic 模型（用户和历史）。
    
    封装记忆中的一个上下文部分，包含摘要内容和更新时间戳。
    
    Attributes:
        summary: 摘要内容，描述该上下文的关键信息
        updatedAt: 最后更新时间戳（ISO 8601 格式）
    
    Note:
        - 默认值为空字符串，表示尚未填充内容
        - 用于 UserContext 和 HistoryContext 的各个子节段
    """

    summary: str = Field(default="", description="Summary content")
    updatedAt: str = Field(default="", description="Last update timestamp")


class UserContext(BaseModel):
    """用户上下文的 Pydantic 模型。
    
    封装与用户相关的三类上下文信息，用于个性化对话。
    
    Attributes:
        workContext: 工作上下文（项目、技术栈、工作流程等）
        personalContext: 个人上下文（偏好、习惯、风格等）
        topOfMind: 当前关注点（最近的任务、目标、思考等）
    
    Note:
        - 每个子节段都是 ContextSection 类型
        - 使用 default_factory 确保每次创建新实例时生成独立的对象
        - 这些信息由 LLM 自动从对话中提取和更新
    """

    workContext: ContextSection = Field(default_factory=ContextSection)
    personalContext: ContextSection = Field(default_factory=ContextSection)
    topOfMind: ContextSection = Field(default_factory=ContextSection)


class HistoryContext(BaseModel):
    """历史上下文的 Pydantic 模型。
    
    封装对话历史的三个时间维度，用于长期记忆管理。
    
    Attributes:
        recentMonths: 最近几个月的详细上下文（高频访问）
        earlierContext: 较早的上下文（中频访问）
        longTermBackground: 长期背景信息（低频访问，高度压缩）
    
    Note:
        - 每个子节段都是 ContextSection 类型
        - 使用时间分层策略优化 token 使用和检索效率
        - 这些信息通过记忆系统的 summarization 机制定期整理
    """

    recentMonths: ContextSection = Field(default_factory=ContextSection)
    earlierContext: ContextSection = Field(default_factory=ContextSection)
    longTermBackground: ContextSection = Field(default_factory=ContextSection)


class Fact(BaseModel):
    """记忆事实的 Pydantic 模型。
    
    封装从对话中提取的单个事实，是记忆系统的核心数据单元。
    
    Attributes:
        id: 事实的唯一标识符（如 fact_abc123）
        content: 事实的内容描述（如 "用户偏好 TypeScript 而非 JavaScript"）
        category: 事实的分类（如 preference, context, skill 等），默认为 "context"
        confidence: 置信度分数（0-1），表示系统对该事实准确性的确信程度
        createdAt: 创建时间戳（ISO 8601 格式）
        source: 来源线程 ID，记录该事实是从哪个对话中提取的
        sourceError: 可选的错误描述，记录之前 AI 犯的错误或错误方法（用于纠正信号）
    
    Note:
        - 置信度用于过滤低质量事实（低于阈值的事实不会被注入到提示词中）
        - sourceError 字段支持纠正信号检测，提高事实准确性
        - 事实可以手动创建或通过 LLM 自动提取
    """

    id: str = Field(..., description="Unique identifier for the fact")
    content: str = Field(..., description="Fact content")
    category: str = Field(default="context", description="Fact category")
    confidence: float = Field(default=0.5, description="Confidence score (0-1)")
    createdAt: str = Field(default="", description="Creation timestamp")
    source: str = Field(default="unknown", description="Source thread ID")
    sourceError: str | None = Field(default=None, description="Optional description of the prior mistake or wrong approach")


class MemoryResponse(BaseModel):
    """记忆数据的响应模型。
    
    封装完整的记忆数据结构，包括版本、时间戳、用户上下文、历史和事实列表。
    
    Attributes:
        version: 记忆 schema 版本号（如 "1.0"），用于兼容性管理
        lastUpdated: 最后更新时间戳（ISO 8601 格式）
        user: 用户上下文（workContext, personalContext, topOfMind）
        history: 历史上下文（recentMonths, earlierContext, longTermBackground）
        facts: 事实列表，按置信度和相关性排序
    
    Note:
        - 这是记忆系统的核心数据结构
        - 所有记忆相关 API 都返回此模型或其变体
        - 使用 response_model_exclude_none=True 排除空值字段
    """

    version: str = Field(default="1.0", description="Memory schema version")
    lastUpdated: str = Field(default="", description="Last update timestamp")
    user: UserContext = Field(default_factory=UserContext)
    history: HistoryContext = Field(default_factory=HistoryContext)
    facts: list[Fact] = Field(default_factory=list)


def _map_memory_fact_value_error(exc: ValueError) -> HTTPException:
    """将 updater 验证错误转换为稳定的 API 响应。
    
    统一处理记忆事实相关的 ValueError，映射到适当的 HTTP 状态码和错误消息。
    
    Args:
        exc: ValueError 异常对象
        
    Returns:
        HTTPException 对象，包含 400 状态码和人类可读的错误详情
        
    Note:
        - 如果异常参数是 "confidence"，返回置信度无效的错误消息
        - 否则返回内容不能为空的错误消息
        - 这种映射确保前端收到一致的错误格式
    """
    if exc.args and exc.args[0] == "confidence":
        detail = "Invalid confidence value; must be between 0 and 1."
    else:
        detail = "Memory fact content cannot be empty."
    return HTTPException(status_code=400, detail=detail)


class FactCreateRequest(BaseModel):
    """创建记忆事实的请求模型。
    
    封装手动创建单个记忆事实所需的参数。
    
    Attributes:
        content: 事实的内容描述（必填，至少 1 个字符）
        category: 事实的分类（可选，默认为 "context"）
        confidence: 置信度分数（可选，默认为 0.5，范围 0-1）
    
    Note:
        - 使用 min_length=1 确保内容不为空
        - 使用 ge=0.0, le=1.0 限制置信度范围
        - 这是 POST /api/memory/facts 端点的请求体
    """

    content: str = Field(..., min_length=1, description="Fact content")
    category: str = Field(default="context", description="Fact category")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence score (0-1)")


class FactPatchRequest(BaseModel):
    """PATCH 请求模型，保留省略字段的现有值。
    
    用于部分更新记忆事实，只更新提供的字段，其他字段保持不变。
    
    Attributes:
        content: 事实的内容描述（可选，如果提供则至少 1 个字符）
        category: 事实的分类（可选）
        confidence: 置信度分数（可选，如果提供则范围 0-1）
    
    Note:
        - 所有字段都是可选的（str | None）
        - 未提供的字段保持原值不变
        - 这是 PATCH /api/memory/facts/{fact_id} 端点的请求体
        - 与 FactCreateRequest 不同，这里允许部分更新
    """

    content: str | None = Field(default=None, min_length=1, description="Fact content")
    category: str | None = Field(default=None, description="Fact category")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0, description="Confidence score (0-1)")


class MemoryConfigResponse(BaseModel):
    """记忆配置的响应模型。
    
    封装记忆系统的配置参数，用于前端显示和管理。
    
    Attributes:
        enabled: 是否启用记忆系统
        storage_path: 记忆存储文件的路径（如 .deer-flow/memory.json）
        debounce_seconds: 记忆更新的防抖时间（秒），避免频繁更新
        max_facts: 最大存储的事实数量
        fact_confidence_threshold: 事实的最小置信度阈值（低于此值的事实不会被注入）
        injection_enabled: 是否启用记忆注入（将记忆插入到提示词中）
        max_injection_tokens: 记忆注入的最大 token 数
    
    Note:
        - 这是 GET /api/memory/config 端点的响应体
        - 所有字段都是必填的（没有默认值）
        - 配置由 deerflow.config.memory_config 模块管理
    """

    enabled: bool = Field(..., description="Whether memory is enabled")
    storage_path: str = Field(..., description="Path to memory storage file")
    debounce_seconds: int = Field(..., description="Debounce time for memory updates")
    max_facts: int = Field(..., description="Maximum number of facts to store")
    fact_confidence_threshold: float = Field(..., description="Minimum confidence threshold for facts")
    injection_enabled: bool = Field(..., description="Whether memory injection is enabled")
    max_injection_tokens: int = Field(..., description="Maximum tokens for memory injection")


class MemoryStatusResponse(BaseModel):
    """记忆状态的响应模型。
    
    封装记忆系统的完整状态，包括配置和当前数据。
    
    Attributes:
        config: 记忆系统配置（MemoryConfigResponse）
        data: 当前记忆数据（MemoryResponse）
    
    Note:
        - 这是 GET /api/memory/status 端点的响应体
        - 组合配置和数据，减少前端请求次数
        - 用于记忆管理界面的初始化加载
    """

    config: MemoryConfigResponse
    data: MemoryResponse


@router.get(
    "/memory",
    response_model=MemoryResponse,
    response_model_exclude_none=True,
    summary="获取记忆数据",
    description="检索当前的全局记忆数据，包括用户上下文、历史和事实。",
)
async def get_memory() -> MemoryResponse:
    """获取当前的全局记忆数据。
    
    从存储中加载并返回完整的记忆数据，包括用户上下文、历史上下文和所有事实。
    
    Returns:
        当前的记忆数据，包含用户上下文、历史和事实
        
    Raises:
        HTTPException: 500 如果加载记忆失败（由底层模块处理）
        
    Note:
        - 使用当前用户的 user_id 进行隔离（多租户支持）
        - 返回的数据结构与 MemoryResponse 模型一致
        - 这是最常用的记忆 API 端点
        
    Example Response:
        ```json
        {
            "version": "1.0",
            "lastUpdated": "2024-01-15T10:30:00Z",
            "user": {
                "workContext": {"summary": "Working on DeerFlow project", "updatedAt": "..."},
                "personalContext": {"summary": "Prefers concise responses", "updatedAt": "..."},
                "topOfMind": {"summary": "Building memory API", "updatedAt": "..."}
            },
            "history": {
                "recentMonths": {"summary": "Recent development activities", "updatedAt": "..."},
                "earlierContext": {"summary": "", "updatedAt": ""},
                "longTermBackground": {"summary": "", "updatedAt": ""}
            },
            "facts": [
                {
                    "id": "fact_abc123",
                    "content": "User prefers TypeScript over JavaScript",
                    "category": "preference",
                    "confidence": 0.9,
                    "createdAt": "2024-01-15T10:30:00Z",
                    "source": "thread_xyz"
                }
            ]
        }
        ```
    """
    memory_data = get_memory_data(user_id=get_effective_user_id())
    return MemoryResponse(**memory_data)


@router.post(
    "/memory/reload",
    response_model=MemoryResponse,
    response_model_exclude_none=True,
    summary="重新加载记忆数据",
    description="从存储文件重新加载记忆数据，刷新内存缓存。",
)
async def reload_memory() -> MemoryResponse:
    """从文件重新加载记忆数据。
    
    强制从存储文件重新加载记忆数据，用于外部修改文件后刷新缓存。
    
    Returns:
        重新加载后的记忆数据
        
    Note:
        - 绕过内存缓存，直接从磁盘读取
        - 适用于外部工具修改了记忆文件的场景
        - 使用当前用户的 user_id 进行隔离
    """
    memory_data = reload_memory_data(user_id=get_effective_user_id())
    return MemoryResponse(**memory_data)


@router.delete(
    "/memory",
    response_model=MemoryResponse,
    response_model_exclude_none=True,
    summary="清空所有记忆数据",
    description="删除所有保存的记忆数据，将记忆结构重置为空状态。",
)
async def clear_memory() -> MemoryResponse:
    """清空所有持久化的记忆数据。
    
    删除当前用户的所有记忆数据，包括用户上下文、历史和事实。
    
    Returns:
        清空后的记忆数据（所有字段为空或空列表）
        
    Raises:
        HTTPException: 500 如果清空记忆失败（如权限不足、磁盘错误等）
        
    Note:
        - 此操作不可逆，请谨慎使用
        - 会保留记忆文件，但将所有内容重置为空
        - 使用当前用户的 user_id 进行隔离
    """
    try:
        memory_data = clear_memory_data(user_id=get_effective_user_id())
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Failed to clear memory data.") from exc

    return MemoryResponse(**memory_data)


@router.post(
    "/memory/facts",
    response_model=MemoryResponse,
    response_model_exclude_none=True,
    summary="创建记忆事实",
    description="手动创建单个保存的记忆事实。",
)
async def create_memory_fact_endpoint(request: FactCreateRequest) -> MemoryResponse:
    """手动创建单个事实。
    
    允许用户或管理员手动添加新的记忆事实，而不是等待 LLM 自动提取。
    
    Args:
        request: 包含事实内容的请求对象（content, category, confidence）
        
    Returns:
        更新后的记忆数据（包含新创建的事实）
        
    Raises:
        HTTPException:
            - 400 如果内容无效或置信度超出范围
            - 500 如果创建失败（如磁盘错误、权限不足等）
            
    Note:
        - 使用当前用户的 user_id 进行隔离
        - 会自动生成唯一的 fact_id 和 createdAt 时间戳
        - 新事实的 source 字段设置为 "manual" 表示手动创建
    """
    try:
        memory_data = create_memory_fact(
            content=request.content,
            category=request.category,
            confidence=request.confidence,
            user_id=get_effective_user_id(),
        )
    except ValueError as exc:
        raise _map_memory_fact_value_error(exc) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Failed to create memory fact.") from exc

    return MemoryResponse(**memory_data)


@router.delete(
    "/memory/facts/{fact_id}",
    response_model=MemoryResponse,
    response_model_exclude_none=True,
    summary="删除记忆事实",
    description="根据事实 ID 删除单个保存的记忆事实。",
)
async def delete_memory_fact_endpoint(fact_id: str) -> MemoryResponse:
    """根据事实 ID 从记忆中删除单个事实。
    
    永久删除指定的记忆事实，无法恢复。
    
    Args:
        fact_id: 要删除的事实 ID（如 fact_abc123）
        
    Returns:
        更新后的记忆数据（不包含已删除的事实）
        
    Raises:
        HTTPException:
            - 404 如果找不到指定的事实 ID
            - 500 如果删除失败（如磁盘错误、权限不足等）
            
    Note:
        - 使用当前用户的 user_id 进行隔离
        - 删除操作不可逆，请谨慎使用
        - fact_id 必须精确匹配（区分大小写）
    """
    try:
        memory_data = delete_memory_fact(fact_id, user_id=get_effective_user_id())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Memory fact '{fact_id}' not found.") from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Failed to delete memory fact.") from exc

    return MemoryResponse(**memory_data)


@router.patch(
    "/memory/facts/{fact_id}",
    response_model=MemoryResponse,
    response_model_exclude_none=True,
    summary="部分更新记忆事实",
    description="根据事实 ID 部分更新单个保存的记忆事实，保留省略的字段。",
)
async def update_memory_fact_endpoint(fact_id: str, request: FactPatchRequest) -> MemoryResponse:
    """部分更新单个事实。
    
    允许更新事实的部分字段，未提供的字段保持原值不变。
    
    Args:
        fact_id: 要更新的事实 ID（如 fact_abc123）
        request: 包含更新字段的请求对象（content, category, confidence 都是可选的）
        
    Returns:
        更新后的记忆数据（包含修改后的事实）
        
    Raises:
        HTTPException:
            - 400 如果内容无效或置信度超出范围
            - 404 如果找不到指定的事实 ID
            - 500 如果更新失败（如磁盘错误、权限不足等）
            
    Note:
        - 使用当前用户的 user_id 进行隔离
        - 只更新提供的字段，其他字段保持不变
        - 这是 PATCH 方法，不是 PUT（PUT 会替换整个资源）
    """
    try:
        memory_data = update_memory_fact(
            fact_id=fact_id,
            content=request.content,
            category=request.category,
            confidence=request.confidence,
            user_id=get_effective_user_id(),
        )
    except ValueError as exc:
        raise _map_memory_fact_value_error(exc) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Memory fact '{fact_id}' not found.") from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Failed to update memory fact.") from exc

    return MemoryResponse(**memory_data)


@router.get(
    "/memory/export",
    response_model=MemoryResponse,
    response_model_exclude_none=True,
    summary="导出记忆数据",
    description="将当前的全局记忆数据导出为 JSON 格式，用于备份或传输。",
)
async def export_memory() -> MemoryResponse:
    """导出当前的记忆数据。
    
    返回完整的记忆数据，可用于备份、迁移或在不同系统之间传输。
    
    Returns:
        当前的记忆数据（与 GET /api/memory 相同）
        
    Note:
        - 使用当前用户的 user_id 进行隔离
        - 导出的数据可以直接用于 POST /api/memory/import 导入
        - 数据格式为标准的 JSON，便于人工阅读和编辑
    """
    memory_data = get_memory_data(user_id=get_effective_user_id())
    return MemoryResponse(**memory_data)


@router.post(
    "/memory/import",
    response_model=MemoryResponse,
    response_model_exclude_none=True,
    summary="导入记忆数据",
    description="从 JSON 负载导入并覆盖当前的全局记忆数据。",
)
async def import_memory(request: MemoryResponse) -> MemoryResponse:
    """导入并持久化记忆数据。
    
    从请求体中读取记忆数据并覆盖当前的所有记忆，用于恢复备份或迁移数据。
    
    Args:
        request: 包含完整记忆数据的请求对象（version, lastUpdated, user, history, facts）
        
    Returns:
        导入后的记忆数据（与请求体相同，但可能经过标准化处理）
        
    Raises:
        HTTPException: 500 如果导入失败（如磁盘错误、权限不足、数据格式无效等）
        
    Note:
        - 使用当前用户的 user_id 进行隔离
        - 会完全覆盖现有的记忆数据（不是合并）
        - 建议在导入前先调用 GET /api/memory/export 备份当前数据
        - 导入的数据必须符合 MemoryResponse 模型的格式要求
    """
    try:
        memory_data = import_memory_data(request.model_dump(), user_id=get_effective_user_id())
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Failed to import memory data.") from exc

    return MemoryResponse(**memory_data)


@router.get(
    "/memory/config",
    response_model=MemoryConfigResponse,
    summary="获取记忆配置",
    description="检索当前的记忆系统配置。",
)
async def get_memory_config_endpoint() -> MemoryConfigResponse:
    """获取记忆系统的配置。
    
    返回记忆系统的所有配置参数，用于前端显示和管理。
    
    Returns:
        当前的记忆配置设置
        
    Note:
        - 配置是全局的，不按用户隔离
        - 配置由 deerflow.config.memory_config 模块管理
        - 修改配置需要编辑配置文件或通过管理接口
        
    Example Response:
        ```json
        {
            "enabled": true,
            "storage_path": ".deer-flow/memory.json",
            "debounce_seconds": 30,
            "max_facts": 100,
            "fact_confidence_threshold": 0.7,
            "injection_enabled": true,
            "max_injection_tokens": 2000
        }
        ```
    """
    config = get_memory_config()
    return MemoryConfigResponse(
        enabled=config.enabled,
        storage_path=config.storage_path,
        debounce_seconds=config.debounce_seconds,
        max_facts=config.max_facts,
        fact_confidence_threshold=config.fact_confidence_threshold,
        injection_enabled=config.injection_enabled,
        max_injection_tokens=config.max_injection_tokens,
    )


@router.get(
    "/memory/status",
    response_model=MemoryStatusResponse,
    response_model_exclude_none=True,
    summary="获取记忆状态",
    description="在单个请求中同时获取记忆配置和当前数据。",
)
async def get_memory_status() -> MemoryStatusResponse:
    """获取记忆系统的状态，包括配置和数据。
    
    组合返回记忆系统的配置和当前数据，减少前端请求次数。
    
    Returns:
        组合的记忆配置和当前数据
        
    Note:
        - 这是最高效的获取完整记忆状态的端点
        - 适用于记忆管理界面的初始化加载
        - 使用当前用户的 user_id 获取数据部分
        - 配置部分是全局的，不按用户隔离
    """
    config = get_memory_config()
    memory_data = get_memory_data(user_id=get_effective_user_id())

    return MemoryStatusResponse(
        config=MemoryConfigResponse(
            enabled=config.enabled,
            storage_path=config.storage_path,
            debounce_seconds=config.debounce_seconds,
            max_facts=config.max_facts,
            fact_confidence_threshold=config.fact_confidence_threshold,
            injection_enabled=config.injection_enabled,
            max_injection_tokens=config.max_injection_tokens,
        ),
        data=MemoryResponse(**memory_data),
    )
