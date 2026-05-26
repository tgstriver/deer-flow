"""文件上传路由处理器。

本模块处理线程级别的文件上传、列表和删除操作，提供以下功能：
- 多文件上传（支持批量上传）
- 文件大小和数量限制
- 文件名安全验证和去重
- 自动文档转换（可选）
- 沙箱同步（非本地沙箱模式）
- 文件列表查询
- 安全文件删除

安全特性：
- 路径遍历攻击防护
- 符号链接检测
- 文件大小限制
- 权限控制（基于用户隔离）
"""

import logging
import os
import stat

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel, Field

from app.gateway.authz import require_permission
from app.gateway.deps import get_config
from deerflow.config.app_config import AppConfig
from deerflow.config.paths import get_paths
from deerflow.runtime.user_context import get_effective_user_id
from deerflow.sandbox.sandbox_provider import SandboxProvider, get_sandbox_provider
from deerflow.uploads.manager import (
    PathTraversalError,
    UnsafeUploadPathError,
    claim_unique_filename,
    delete_file_safe,
    enrich_file_listing,
    ensure_uploads_dir,
    get_uploads_dir,
    list_files_in_dir,
    normalize_filename,
    open_upload_file_no_symlink,
    upload_artifact_url,
    upload_virtual_path,
)
from deerflow.utils.file_conversion import CONVERTIBLE_EXTENSIONS, convert_file_to_markdown

logger = logging.getLogger(__name__)

# 创建路由器，所有端点以 /api/threads/{thread_id}/uploads 为前缀
router = APIRouter(prefix="/api/threads/{thread_id}/uploads", tags=["uploads"])

# 文件上传常量配置
UPLOAD_CHUNK_SIZE = 8192  # 每次读取的块大小（8KB）
DEFAULT_MAX_FILES = 10  # 默认最大文件数量
DEFAULT_MAX_FILE_SIZE = 50 * 1024 * 1024  # 默认单个文件最大大小（50MB）
DEFAULT_MAX_TOTAL_SIZE = 100 * 1024 * 1024  # 默认总上传大小限制（100MB）


class UploadResponse(BaseModel):
    """文件上传响应模型。

    Attributes:
        success: 是否全部成功（没有跳过任何文件）
        files: 成功上传的文件信息列表
        message: 响应消息
        skipped_files: 因安全问题被跳过的文件名列表
    """

    success: bool
    files: list[dict[str, str]]
    message: str
    skipped_files: list[str] = Field(default_factory=list)


class UploadLimits(BaseModel):
    """应用级上传限制配置，暴露给客户端使用。

    Attributes:
        max_files: 单次请求最大文件数量
        max_file_size: 单个文件最大字节数
        max_total_size: 单次请求总字节数限制
    """

    max_files: int
    max_file_size: int
    max_total_size: int


def _make_file_sandbox_writable(file_path: os.PathLike[str] | str) -> None:
    """确保上传文件在挂载到非本地沙箱时可写。

    在 AIO 沙箱模式下，网关先写入主机侧的权威文件，然后沙箱运行时
    可能会重写相同的挂载路径。授予全局可写权限可以防止网关用户和
    沙箱运行时用户之间的权限不匹配。

    Args:
        file_path: 文件路径

    Note:
        - 跳过符号链接文件的 chmod 操作
        - 添加用户、组和其他用户的写权限
        - 如果系统支持，使用 follow_symlinks=False 避免跟随符号链接
    """
    file_stat = os.lstat(file_path)
    if stat.S_ISLNK(file_stat.st_mode):
        logger.warning("Skipping sandbox chmod for symlinked upload path: %s", file_path)
        return

    writable_mode = stat.S_IMODE(file_stat.st_mode) | stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH
    chmod_kwargs = {"follow_symlinks": False} if os.chmod in os.supports_follow_symlinks else {}
    os.chmod(file_path, writable_mode, **chmod_kwargs)


def _uses_thread_data_mounts(sandbox_provider: SandboxProvider) -> bool:
    """检查沙箱提供者是否使用线程数据挂载。

    如果使用线程数据挂载，则不需要手动同步文件到沙箱。

    Args:
        sandbox_provider: 沙箱提供者实例

    Returns:
        True 如果使用线程数据挂载，否则 False
    """
    return bool(getattr(sandbox_provider, "uses_thread_data_mounts", False))


def _get_uploads_config_value(app_config: AppConfig, key: str, default: object) -> object:
    """从上传配置中读取值，支持字典和属性访问两种方式。

    兼容不同的配置格式：
    - 字典格式：uploads_cfg.get(key, default)
    - 对象格式：getattr(uploads_cfg, key, default)

    Args:
        app_config: 应用配置对象
        key: 配置键名
        default: 默认值

    Returns:
        配置值或默认值
    """
    uploads_cfg = getattr(app_config, "uploads", None)
    if isinstance(uploads_cfg, dict):
        return uploads_cfg.get(key, default)
    return getattr(uploads_cfg, key, default)


def _get_upload_limit(app_config: AppConfig, key: str, default: int, *, legacy_key: str | None = None) -> int:
    """获取上传限制配置值，包含向后兼容性和验证逻辑。

    尝试从配置中读取限制值，如果无效则回退到默认值。
    支持遗留配置键名以保证向后兼容。

    Args:
        app_config: 应用配置对象
        key: 主要配置键名
        default: 默认值
        legacy_key: 可选的遗留配置键名（用于向后兼容）

    Returns:
        有效的限制值（正整数），如果配置无效则返回默认值

    Note:
        - 如果值为 None，尝试使用遗留键名
        - 如果值 <= 0，视为无效并使用默认值
        - 任何异常都会记录警告并返回默认值
    """
    try:
        value = _get_uploads_config_value(app_config, key, None)
        if value is None and legacy_key is not None:
            value = _get_uploads_config_value(app_config, legacy_key, None)
        if value is None:
            value = default
        limit = int(value)
        if limit <= 0:
            raise ValueError
        return limit
    except Exception:
        logger.warning("Invalid uploads.%s value; falling back to %d", key, default)
        return default


def _get_upload_limits(app_config: AppConfig) -> UploadLimits:
    """获取完整的上传限制配置。

    从应用配置中读取所有上传限制，支持遗留键名以保证向后兼容。

    Args:
        app_config: 应用配置对象

    Returns:
        包含所有限制值的 UploadLimits 对象

    Note:
        - max_files 支持遗留键名 max_file_count
        - max_file_size 支持遗留键名 max_single_file_size
    """
    return UploadLimits(
        max_files=_get_upload_limit(app_config, "max_files", DEFAULT_MAX_FILES, legacy_key="max_file_count"),
        max_file_size=_get_upload_limit(app_config, "max_file_size", DEFAULT_MAX_FILE_SIZE, legacy_key="max_single_file_size"),
        max_total_size=_get_upload_limit(app_config, "max_total_size", DEFAULT_MAX_TOTAL_SIZE),
    )


def _cleanup_uploaded_paths(paths: list[os.PathLike[str] | str]) -> None:
    """清理已上传的文件路径（用于失败回滚）。

    当上传请求被拒绝或失败时，删除已成功写入的文件以避免磁盘泄漏。
    按相反顺序删除以确保一致性。

    Args:
        paths: 需要清理的文件路径列表

    Note:
        - 忽略 FileNotFoundError（文件可能已被删除）
        - 其他异常仅记录警告，不中断清理流程
    """
    for path in reversed(paths):
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
        except Exception:
            logger.warning("Failed to clean up upload path after rejected request: %s", path, exc_info=True)


async def _write_upload_file_with_limits(
    file: UploadFile,
    *,
    uploads_dir: os.PathLike[str] | str,
    display_filename: str,
    max_single_file_size: int,
    max_total_size: int,
    total_size: int,
) -> tuple[os.PathLike[str] | str, int, int]:
    """带限制检查的文件写入函数。

    流式读取上传文件并写入磁盘，同时检查单个文件大小和总大小限制。
    如果超过限制，会抛出 HTTPException 并清理已写入的部分文件。

    Args:
        file: FastAPI 上传文件对象
        uploads_dir: 上传目录路径
        display_filename: 显示用的文件名（已 sanitized）
        max_single_file_size: 单个文件最大字节数
        max_total_size: 本次请求总大小最大字节数
        total_size: 当前累计的总字节数

    Returns:
        三元组：(文件路径, 文件大小, 更新后的总大小)

    Raises:
        HTTPException: 413 如果超过文件大小限制

    Note:
        - 使用分块读取（UPLOAD_CHUNK_SIZE）以支持大文件
        - 异常时会关闭文件句柄并删除部分写入的文件
        - 成功后关闭文件句柄但不删除文件
    """
    file_size = 0
    # 打开文件进行写入（防止符号链接攻击）
    file_path, fh = open_upload_file_no_symlink(uploads_dir, display_filename)
    try:
        # 分块读取并写入文件
        while chunk := await file.read(UPLOAD_CHUNK_SIZE):
            file_size += len(chunk)
            total_size += len(chunk)
            # 检查单个文件大小限制
            if file_size > max_single_file_size:
                raise HTTPException(status_code=413, detail=f"File too large: {display_filename}")
            # 检查总大小限制
            if total_size > max_total_size:
                raise HTTPException(status_code=413, detail="Total upload size too large")
            fh.write(chunk)
    except Exception:
        # 发生异常时关闭文件句柄并删除部分写入的文件
        fh.close()
        try:
            os.unlink(file_path)
        except FileNotFoundError:
            pass
        raise
    else:
        # 成功完成后关闭文件句柄
        fh.close()
    return file_path, file_size, total_size


def _auto_convert_documents_enabled(app_config: AppConfig) -> bool:
    """检查是否启用了自动主机端文档转换功能。

    安全默认值为禁用，除非操作员在 config.yaml 中通过
    uploads.auto_convert_documents 显式启用。

    Args:
        app_config: 应用配置对象

    Returns:
        True 如果启用了自动转换，否则 False

    Note:
        - 支持字符串格式的布尔值（"1", "true", "yes", "on"）
        - 任何异常都返回 False（安全默认值）
    """
    try:
        raw = _get_uploads_config_value(app_config, "auto_convert_documents", False)
        if isinstance(raw, str):
            return raw.strip().lower() in {"1", "true", "yes", "on"}
        return bool(raw)
    except Exception:
        return False


@router.post("", response_model=UploadResponse)
@require_permission("threads", "write", owner_check=True, require_existing=False)
async def upload_files(
    thread_id: str,
    request: Request,
    files: list[UploadFile] = File(...),
    config: AppConfig = Depends(get_config),
) -> UploadResponse:
    """上传多个文件到线程的上传目录。

    支持批量文件上传，包含完整的安全检查和限制验证。

    **工作流程：**
    1. 验证输入：检查文件数量和大小限制
    2. 准备目录：确保上传目录存在
    3. 初始化沙箱：如果需要，获取沙箱实例
    4. 逐个处理文件：
       - 验证文件名安全性
       - 检查文件名重复
       - 流式写入文件（带大小限制）
       - 可选：自动转换为 Markdown
       - 同步到沙箱（非挂载模式）
    5. 返回结果：包含成功文件和跳过文件的信息

    **安全特性：**
    - 文件名规范化（移除危险字符）
    - 文件名去重（防止覆盖）
    - 路径遍历防护
    - 符号链接检测
    - 文件大小限制
    - 总大小限制
    - 自动回滚（失败时清理已上传文件）

    Args:
        thread_id: 线程 ID
        request: FastAPI 请求对象
        files: 上传文件列表
        config: 应用配置（依赖注入）

    Returns:
        上传响应，包含成功文件列表和跳过文件列表

    Raises:
        HTTPException:
            - 400: 没有提供文件或线程 ID 无效
            - 413: 文件数量过多、单个文件过大或总大小超限
            - 500: 沙箱获取失败或其他内部错误

    Permissions:
        需要 'threads.write' 权限，且必须是线程所有者

    Note:
        - 同一请求中的重复文件名会自动重命名（添加序号）
        - 如果启用了 auto_convert_documents，支持的文档格式会自动转换为 Markdown
        - 在非线程数据挂载模式下，文件会同步到沙箱
    """
    # 验证是否有文件上传
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    # 获取上传限制配置
    limits = _get_upload_limits(config)
    # 检查文件数量限制
    if len(files) > limits.max_files:
        raise HTTPException(status_code=413, detail=f"Too many files: maximum is {limits.max_files}")

    # 确保上传目录存在
    try:
        uploads_dir = ensure_uploads_dir(thread_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 获取沙箱上传目录路径
    sandbox_uploads = get_paths().sandbox_uploads_dir(thread_id, user_id=get_effective_user_id())

    # 初始化跟踪变量
    uploaded_files = []  # 成功上传的文件信息
    written_paths = []  # 已写入的文件路径（用于回滚）
    sandbox_sync_targets = []  # 需要同步到沙箱的文件
    skipped_files = []  # 因安全问题跳过的文件
    total_size = 0  # 累计总大小
    # 跟踪当前请求中的文件名，防止重复表单字段相互覆盖
    # 已有上传保留历史覆盖行为（用于单个替换上传）
    seen_filenames: set[str] = set()

    # 获取沙箱提供者并检查是否需要同步
    sandbox_provider = get_sandbox_provider()
    sync_to_sandbox = not _uses_thread_data_mounts(sandbox_provider)
    sandbox = None
    if sync_to_sandbox:
        # 获取沙箱实例
        sandbox_id = sandbox_provider.acquire(thread_id)
        sandbox = sandbox_provider.get(sandbox_id)
        if sandbox is None:
            raise HTTPException(status_code=500, detail="Failed to acquire sandbox")

    # 检查是否启用自动文档转换
    auto_convert_documents = _auto_convert_documents_enabled(config)

    # 逐个处理上传的文件
    for file in files:
        if not file.filename:
            continue

        try:
            # 规范化文件名（移除危险字符）
            original_filename = normalize_filename(file.filename)
            # 确保文件名唯一（如果重复则添加序号）
            safe_filename = claim_unique_filename(original_filename, seen_filenames)
        except ValueError:
            logger.warning(f"Skipping file with unsafe filename: {file.filename!r}")
            continue

        try:
            # 写入文件（带大小限制检查）
            file_path, file_size, total_size = await _write_upload_file_with_limits(
                file,
                uploads_dir=uploads_dir,
                display_filename=safe_filename,
                max_single_file_size=limits.max_file_size,
                max_total_size=limits.max_total_size,
                total_size=total_size,
            )
            written_paths.append(file_path)

            # 构建虚拟路径和文件信息
            virtual_path = upload_virtual_path(safe_filename)

            if sync_to_sandbox:
                sandbox_sync_targets.append((file_path, virtual_path))

            file_info = {
                "filename": safe_filename,
                "size": str(file_size),
                "path": str(sandbox_uploads / safe_filename),
                "virtual_path": virtual_path,
                "artifact_url": upload_artifact_url(thread_id, safe_filename),
            }
            # 如果文件名被修改，记录原始文件名
            if safe_filename != original_filename:
                file_info["original_filename"] = original_filename

            logger.info(f"Saved file: {safe_filename} ({file_size} bytes) to {file_info['path']}")

            # 如果启用了自动转换且文件格式支持，转换为 Markdown
            file_ext = file_path.suffix.lower()
            if auto_convert_documents and file_ext in CONVERTIBLE_EXTENSIONS:
                md_path = await convert_file_to_markdown(file_path)
                if md_path:
                    written_paths.append(md_path)
                    md_virtual_path = upload_virtual_path(md_path.name)

                    if sync_to_sandbox:
                        sandbox_sync_targets.append((md_path, md_virtual_path))

                    file_info["markdown_file"] = md_path.name
                    file_info["markdown_path"] = str(sandbox_uploads / md_path.name)
                    file_info["markdown_virtual_path"] = md_virtual_path
                    file_info["markdown_artifact_url"] = upload_artifact_url(thread_id, md_path.name)

            uploaded_files.append(file_info)

        except HTTPException as e:
            # HTTP 异常：清理已上传的文件并重新抛出
            _cleanup_uploaded_paths(written_paths)
            raise e
        except UnsafeUploadPathError as e:
            # 不安全路径：跳过该文件但继续处理其他文件
            logger.warning("Skipping upload with unsafe destination %s: %s", file.filename, e)
            skipped_files.append(safe_filename)
            continue
        except Exception as e:
            # 其他异常：清理并返回 500 错误
            logger.error(f"Failed to upload {file.filename}: {e}")
            _cleanup_uploaded_paths(written_paths)
            raise HTTPException(status_code=500, detail=f"Failed to upload {file.filename}: {str(e)}")

    # 如果需要同步到沙箱，设置文件权限并更新沙箱
    if sync_to_sandbox:
        for file_path, virtual_path in sandbox_sync_targets:
            _make_file_sandbox_writable(file_path)
            sandbox.update_file(virtual_path, file_path.read_bytes())

    # 构建响应消息
    message = f"Successfully uploaded {len(uploaded_files)} file(s)"
    if skipped_files:
        message += f"; skipped {len(skipped_files)} unsafe file(s)"

    return UploadResponse(
        success=not skipped_files,
        files=uploaded_files,
        message=message,
        skipped_files=skipped_files,
    )


@router.get("/limits", response_model=UploadLimits)
@require_permission("threads", "read", owner_check=True)
async def get_upload_limits(
    thread_id: str,
    request: Request,
    config: AppConfig = Depends(get_config),
) -> UploadLimits:
    """获取当前线程的上传限制配置。

    返回应用配置的上传限制，客户端可以使用这些信息来验证上传前的文件大小。

    Args:
        thread_id: 线程 ID（用于权限验证）
        request: FastAPI 请求对象
        config: 应用配置（依赖注入）

    Returns:
        上传限制配置对象

    Permissions:
        需要 'threads.read' 权限，且必须是线程所有者
    """
    return _get_upload_limits(config)


@router.get("/list", response_model=dict)
@require_permission("threads", "read", owner_check=True)
async def list_uploaded_files(thread_id: str, request: Request) -> dict:
    """列出线程上传目录中的所有文件。

    获取指定线程的所有上传文件信息，包括文件名、大小、路径等。

    Args:
        thread_id: 线程 ID
        request: FastAPI 请求对象

    Returns:
        包含文件列表的字典，每个文件包含：
        - filename: 文件名
        - size: 文件大小
        - path: 沙箱相对路径
        - virtual_path: 虚拟路径
        - artifact_url: 资源 URL

    Raises:
        HTTPException: 400 如果线程 ID 无效

    Permissions:
        需要 'threads.read' 权限，且必须是线程所有者

    Note:
        - 网关额外包含沙箱相对路径（sandbox-relative path）
        - 文件列表经过 enrich_file_listing 增强，添加了虚拟路径和 URL
    """
    try:
        uploads_dir = get_uploads_dir(thread_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    result = list_files_in_dir(uploads_dir)
    enrich_file_listing(result, thread_id)

    # 网关额外包含沙箱相对路径
    sandbox_uploads = get_paths().sandbox_uploads_dir(thread_id, user_id=get_effective_user_id())
    for f in result["files"]:
        f["path"] = str(sandbox_uploads / f["filename"])

    return result


@router.delete("/{filename}")
@require_permission("threads", "delete", owner_check=True, require_existing=True)
async def delete_uploaded_file(thread_id: str, filename: str, request: Request) -> dict:
    """从线程上传目录中删除文件。

    安全地删除指定文件，包括其关联的 Markdown 转换文件（如果存在）。

    Args:
        thread_id: 线程 ID
        filename: 要删除的文件名
        request: FastAPI 请求对象

    Returns:
        删除结果字典

    Raises:
        HTTPException:
            - 400: 线程 ID 无效或路径遍历攻击
            - 404: 文件不存在
            - 500: 删除失败

    Permissions:
        需要 'threads.delete' 权限，且必须是线程所有者，文件必须已存在

    Note:
        - 使用 delete_file_safe 确保路径安全（防止路径遍历）
        - 如果文件有对应的 Markdown 转换文件，会一并删除
        - CONVERTIBLE_EXTENSIONS 定义了哪些扩展名的文件可能有 Markdown 版本
    """
    try:
        uploads_dir = get_uploads_dir(thread_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        return delete_file_safe(uploads_dir, filename, convertible_extensions=CONVERTIBLE_EXTENSIONS)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    except PathTraversalError:
        raise HTTPException(status_code=400, detail="Invalid path")
    except Exception as e:
        logger.error(f"Failed to delete {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete {filename}: {str(e)}")
