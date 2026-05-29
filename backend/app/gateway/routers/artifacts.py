"""Artifact 文件访问路由器。

本模块提供 AI 代理生成的 artifact 文件的访问接口，核心功能包括：
- 支持文本和二进制文件的在线查看
- 自动检测文件类型并返回适当的 Content-Type
- 对活跃 Web 内容（HTML/XHTML/SVG）强制下载以防止脚本执行
- 支持从 .skill ZIP 归档中提取文件
- 路径遍历保护和安全验证
- 可选的强制下载模式

安全特性：
- 使用 resolve_thread_virtual_path 防止路径遍历攻击
- 活跃 Web 内容始终作为附件下载，避免 XSS 风险
- .skill 归档文件大小限制（16MB）
- RFC 5987 编码的 Content-Disposition 头支持 Unicode 文件名
"""
import logging
import mimetypes
import zipfile
from pathlib import Path
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, PlainTextResponse, Response

from app.gateway.authz import require_permission
from app.gateway.path_utils import resolve_thread_virtual_path

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["artifacts"])

# 活跃内容 MIME 类型集合 —— 这些类型可能包含可执行脚本，必须作为附件下载
# 以防止在应用源中执行脚本导致 XSS 攻击
ACTIVE_CONTENT_MIME_TYPES = {
    "text/html",              # HTML 文档
    "application/xhtml+xml",  # XHTML 文档
    "image/svg+xml",          # SVG 图像（可能包含 JavaScript）
}

# .skill 归档成员的最大未压缩大小（16 MB）
# 超过此大小的文件不允许预览，防止内存耗尽攻击
MAX_SKILL_ARCHIVE_MEMBER_BYTES = 16 * 1024 * 1024

# .skill 归档读取的块大小（64 KB）
# 用于流式读取大文件，避免一次性加载到内存
_SKILL_ARCHIVE_READ_CHUNK_SIZE = 64 * 1024


def _build_content_disposition(disposition_type: str, filename: str) -> str:
    """构建 RFC 5987 编码的 Content-Disposition 头值。
    
    使用 UTF-8 编码和百分号编码来支持 Unicode 文件名。
    
    Args:
        disposition_type: 处置类型，'attachment'（下载）或 'inline'（内联显示）
        filename: 文件名，可能包含 Unicode 字符
        
    Returns:
        RFC 5987 编码的 Content-Disposition 头值
        
    Note:
        - 格式：{type}; filename*=UTF-8''{encoded_filename}
        - 使用 urllib.parse.quote 进行百分号编码
        - safe='' 表示所有特殊字符都会被编码
        - 符合 RFC 5987 标准，广泛被现代浏览器支持
        
    Example:
        >>> _build_content_disposition("attachment", "报告.pdf")
        "attachment; filename*=UTF-8''%E6%8A%A5%E5%91%8A.pdf"
    """
    return f"{disposition_type}; filename*=UTF-8''{quote(filename)}"


def _build_attachment_headers(filename: str, extra_headers: dict[str, str] | None = None) -> dict[str, str]:
    """构建附件下载的响应头。
    
    创建 Content-Disposition 头以触发文件下载，并可选择性地添加额外头。
    
    Args:
        filename: 下载时使用的文件名
        extra_headers: 额外的响应头字典（如 Cache-Control）
        
    Returns:
        包含 Content-Disposition 和可选额外头的字典
        
    Note:
        - 始终设置 Content-Disposition 为 attachment 模式
        - 如果提供 extra_headers，会合并到返回的字典中
        - 用于强制浏览器下载文件而不是内联显示
    """
    headers = {"Content-Disposition": _build_content_disposition("attachment", filename)}
    if extra_headers:
        headers.update(extra_headers)
    return headers


def is_text_file_by_content(path: Path, sample_size: int = 8192) -> bool:
    """通过检查内容中的空字节来判断文件是否为文本文件。
    
    读取文件的前 N 个字节，检查是否包含空字节（\x00）。
    文本文件不应包含空字节，而二进制文件通常包含。
    
    Args:
        path: 要检查的文件路径
        sample_size: 采样大小（字节），默认 8192
        
    Returns:
        True 如果文件看起来是文本文件，False 如果是二进制文件
        
    Note:
        - 仅检查前 sample_size 字节，不是整个文件
        - 空字节（\x00）是二进制文件的强指标
        - 如果读取失败（权限不足、文件不存在等），返回 False
        - 这是一种启发式方法，不是 100% 准确
        
    Example:
        - 纯文本文件：不包含 \x00 → 返回 True
        - PDF/图片/可执行文件：包含 \x00 → 返回 False
    """
    try:
        with open(path, "rb") as f:
            chunk = f.read(sample_size)
            # 文本文件不应包含空字节
            return b"\x00" not in chunk
    except Exception:
        return False


def _read_skill_archive_member(zip_ref: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    """读取 .skill 归档成员，同时强制执行未压缩大小上限。
    
    从 ZIP 归档中安全地读取单个文件，防止内存耗尽攻击。
    
    Args:
        zip_ref: 已打开的 ZipFile 对象
        info: ZipInfo 对象，包含文件的元数据
        
    Returns:
        文件内容的字节数据
        
    Raises:
        HTTPException: 如果文件超过 MAX_SKILL_ARCHIVE_MEMBER_BYTES（413 状态码）
        
    Note:
        - 首先检查 info.file_size（未压缩大小）是否超过限制
        - 使用流式读取（块大小 _SKILL_ARCHIVE_READ_CHUNK_SIZE）
        - 在读取过程中持续检查累计大小，防止压缩率极高的文件绕过检查
        - 如果超过限制，立即停止读取并抛出异常
        - 这种双重检查确保即使恶意构造的 ZIP 文件也无法消耗过多内存
    """
    if info.file_size > MAX_SKILL_ARCHIVE_MEMBER_BYTES:
        raise HTTPException(status_code=413, detail="Skill archive member is too large to preview")

    chunks: list[bytes] = []
    total_read = 0
    with zip_ref.open(info, "r") as src:
        while chunk := src.read(_SKILL_ARCHIVE_READ_CHUNK_SIZE):
            total_read += len(chunk)
            if total_read > MAX_SKILL_ARCHIVE_MEMBER_BYTES:
                raise HTTPException(status_code=413, detail="Skill archive member is too large to preview")
            chunks.append(chunk)
    return b"".join(chunks)


def _extract_file_from_skill_archive(zip_path: Path, internal_path: str) -> bytes | None:
    """从 .skill ZIP 归档中提取文件。
    
    支持从技能包归档中提取指定文件，用于预览和访问。
    
    Args:
        zip_path: .skill 文件的路径（ZIP 归档）
        internal_path: 归档内文件的路径（如 "SKILL.md"）
        
    Returns:
        文件内容的字节数据，如果找不到则返回 None
        
    Note:
        - 首先验证文件是否是有效的 ZIP 归档
        - 构建文件名到 ZipInfo 的映射以提高查找效率
        - 尝试直接路径匹配（精确匹配）
        - 如果直接匹配失败，尝试带顶层目录前缀的匹配（如 "skill-name/SKILL.md"）
        - 捕获 BadZipFile 和 KeyError 异常，返回 None 而不是抛出
        - 通过 _read_skill_archive_member 读取时会自动检查大小限制
        
    Example:
        - 直接匹配：internal_path="SKILL.md" → 查找 "SKILL.md"
        - 带前缀匹配：internal_path="SKILL.md" → 查找 "my-skill/SKILL.md"
    """
    if not zipfile.is_zipfile(zip_path):
        return None

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # 列出归档中的所有文件，构建名称到信息的映射
            infos_by_name = {info.filename: info for info in zip_ref.infolist()}

            # 首先尝试直接路径匹配
            if internal_path in infos_by_name:
                return _read_skill_archive_member(zip_ref, infos_by_name[internal_path])

            # 尝试带任何顶层目录前缀的匹配（如 "skill-name/SKILL.md"）
            for name, info in infos_by_name.items():
                if name.endswith("/" + internal_path) or name == internal_path:
                    return _read_skill_archive_member(zip_ref, info)

            # 未找到
            return None
    except (zipfile.BadZipFile, KeyError):
        return None


@router.get(
    "/threads/{thread_id}/artifacts/{path:path}",
    summary="获取 Artifact 文件",
    description="检索 AI 代理生成的 artifact 文件。文本和二进制文件可以内联查看，而活跃的 Web 内容始终作为下载提供。",
)
@require_permission("threads", "read", owner_check=True)
async def get_artifact(thread_id: str, path: str, request: Request, download: bool = False) -> Response:
    """根据路径获取 artifact 文件。
    
    此端点自动检测文件类型并返回适当的内容类型。
    使用 `download` 查询参数强制非活跃内容的文件下载。
    
    Args:
        thread_id: 线程 ID
        path: 带有虚拟前缀的 artifact 路径（如 mnt/user-data/outputs/file.txt）
        request: FastAPI 请求对象（自动注入）
        download: 如果为 True，强制附件下载；默认为 False
        
    Returns:
        带有适当内容类型的 FileResponse：
        - 活跃内容（HTML/XHTML/SVG）：作为下载附件提供
        - 文本文件：带有正确 MIME 类型的纯文本
        - 二进制文件：内联显示并提供下载选项
        
    Raises:
        HTTPException:
            - 400 如果路径无效或不是文件
            - 403 如果访问被拒绝（检测到路径遍历）
            - 404 如果文件未找到
            - 413 如果 .skill 归档成员过大
        
    Query Parameters:
        download (bool): 如果为 true，强制附件下载那些通常以内联或纯文本返回的文件类型。
                        活跃的 HTML/XHTML/SVG 内容无论此标志如何都会下载。
        
    Note:
        - 使用 @require_permission 装饰器进行权限验证（owner_check=True 仅允许所有者访问）
        - 支持从 .skill ZIP 归档中提取文件（路径包含 ".skill/" 时）
        - 对活跃 Web 内容强制下载以防止 XSS 攻击
        - 使用 resolve_thread_virtual_path 防止路径遍历攻击
        - .skill 归档文件有 5 分钟缓存（Cache-Control: private, max-age=300）
        
    Example:
        - 内联获取文本文件：`/api/threads/abc123/artifacts/mnt/user-data/outputs/notes.txt`
        - 下载文件：`/api/threads/abc123/artifacts/mnt/user-data/outputs/data.csv?download=true`
        - 活跃的 Web 内容（.html、.xhtml、.svg）始终下载
        - 从 .skill 归档提取：`/api/threads/abc123/artifacts/mnt/skills/my-skill.skill/SKILL.md`
    """
    # 检查这是对 .skill 归档内文件的请求（如 xxx.skill/SKILL.md）
    if ".skill/" in path:
        # 在 ".skill/" 处分割路径以获取 ZIP 文件路径和内部路径
        skill_marker = ".skill/"
        marker_pos = path.find(skill_marker)
        skill_file_path = path[: marker_pos + len(".skill")]  # 例如："mnt/user-data/outputs/my-skill.skill"
        internal_path = path[marker_pos + len(skill_marker) :]  # 例如："SKILL.md"

        actual_skill_path = resolve_thread_virtual_path(thread_id, skill_file_path)

        if not actual_skill_path.exists():
            raise HTTPException(status_code=404, detail=f"Skill file not found: {skill_file_path}")

        if not actual_skill_path.is_file():
            raise HTTPException(status_code=400, detail=f"Path is not a file: {skill_file_path}")

        # 从 .skill 归档中提取文件
        content = _extract_file_from_skill_archive(actual_skill_path, internal_path)
        if content is None:
            raise HTTPException(status_code=404, detail=f"File '{internal_path}' not found in skill archive")

        # 根据内部文件确定 MIME 类型
        mime_type, _ = mimetypes.guess_type(internal_path)
        # 添加缓存头以避免重复的 ZIP 提取（缓存 5 分钟）
        cache_headers = {"Cache-Control": "private, max-age=300"}
        download_name = Path(internal_path).name or actual_skill_path.stem
        if download or mime_type in ACTIVE_CONTENT_MIME_TYPES:
            return Response(content=content, media_type=mime_type or "application/octet-stream", headers=_build_attachment_headers(download_name, cache_headers))

        if mime_type and mime_type.startswith("text/"):
            return PlainTextResponse(content=content.decode("utf-8"), media_type=mime_type, headers=cache_headers)

        # 对于看起来像文本的未知类型，默认为纯文本
        try:
            return PlainTextResponse(content=content.decode("utf-8"), media_type="text/plain", headers=cache_headers)
        except UnicodeDecodeError:
            return Response(content=content, media_type=mime_type or "application/octet-stream", headers=cache_headers)

    # 普通文件处理（非 .skill 归档）
    actual_path = resolve_thread_virtual_path(thread_id, path)

    logger.info(f"Resolving artifact path: thread_id={thread_id}, requested_path={path}, actual_path={actual_path}")

    if not actual_path.exists():
        raise HTTPException(status_code=404, detail=f"Artifact not found: {path}")

    if not actual_path.is_file():
        raise HTTPException(status_code=400, detail=f"Path is not a file: {path}")

    mime_type, _ = mimetypes.guess_type(actual_path)

    if download:
        # 用户请求强制下载，使用附件模式
        return FileResponse(path=actual_path, filename=actual_path.name, media_type=mime_type, headers=_build_attachment_headers(actual_path.name))

    # 始终强制下载活跃内容类型，以防止用户在打开生成的 artifacts 时在应用源中执行脚本
    if mime_type in ACTIVE_CONTENT_MIME_TYPES:
        return FileResponse(path=actual_path, filename=actual_path.name, media_type=mime_type, headers=_build_attachment_headers(actual_path.name))

    if mime_type and mime_type.startswith("text/"):
        # 已知文本类型，使用 PlainTextResponse
        return PlainTextResponse(content=actual_path.read_text(encoding="utf-8"), media_type=mime_type)

    if is_text_file_by_content(actual_path):
        # 通过内容检测为文本文件，使用 PlainTextResponse
        return PlainTextResponse(content=actual_path.read_text(encoding="utf-8"), media_type=mime_type)

    # 二进制文件，使用内联显示并提供下载选项
    return Response(content=actual_path.read_bytes(), media_type=mime_type, headers={"Content-Disposition": _build_content_disposition("inline", actual_path.name)})
