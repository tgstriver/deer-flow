"""共享的上传管理逻辑。

纯业务逻辑模块 —— 不依赖 FastAPI/HTTP。
Gateway 和 Client 都委托给这些函数处理上传操作。

核心功能：
- 线程安全的上传目录管理
- 文件名安全验证和规范化
- 防路径遍历攻击（Path Traversal）
- 防符号链接攻击（Symlink Attack）
- 文件列表和删除操作
- 虚拟路径和 artifact URL 生成

安全特性：
- 严格的 thread_id 验证（仅允许字母数字、连字符、下划线、点号）
- 文件名清理（去除路径组件，拒绝遍历模式）
- POSIX O_NOFOLLOW 防止符号链接跟随
- Windows 双重 lstat 检查减少 TOCTOU 窗口
- 路径遍历验证确保文件在允许的目录内
"""

import errno
import os
import re
import stat
from pathlib import Path
from urllib.parse import quote

from deerflow.config.paths import VIRTUAL_PATH_PREFIX, get_paths
from deerflow.runtime.user_context import get_effective_user_id


class PathTraversalError(ValueError):
    """Raised when a path escapes its allowed base directory."""


class UnsafeUploadPathError(ValueError):
    """Raised when an upload destination is not a safe regular file path."""


# thread_id 必须仅包含字母数字、连字符、下划线或点号。
# 这是文件系统路径的安全要求，防止特殊字符导致的路径注入。
_SAFE_THREAD_ID = re.compile(r"^[a-zA-Z0-9._-]+$")


def validate_thread_id(thread_id: str) -> None:
    """拒绝包含不安全字符的 thread_id。
    
    验证 thread_id 是否符合文件系统路径安全要求。
    
    Args:
        thread_id: 要验证的线程 ID
        
    Raises:
        ValueError: 如果 thread_id 为空或包含不安全字符
        
    Note:
        - 仅允许字母数字、连字符(-)、下划线(_)、点号(.)
        - 空字符串也会被拒绝
        - 此验证在所有上传操作之前执行
    """
    if not thread_id or not _SAFE_THREAD_ID.match(thread_id):
        raise ValueError(f"Invalid thread_id: {thread_id!r}")


def get_uploads_dir(thread_id: str) -> Path:
    """返回线程的上传目录路径（无副作用）。
    
    计算并返回指定线程的上传目录路径，但不创建目录。
    
    Args:
        thread_id: 线程 ID
        
    Returns:
        上传目录的 Path 对象
        
    Raises:
        ValueError: 如果 thread_id 无效
        
    Note:
        - 此函数不会创建目录，仅计算路径
        - 使用 get_effective_user_id() 获取当前用户 ID
        - 路径格式：sandbox_uploads_dir/{thread_id}/{user_id}
    """
    validate_thread_id(thread_id)
    return get_paths().sandbox_uploads_dir(thread_id, user_id=get_effective_user_id())


def ensure_uploads_dir(thread_id: str) -> Path:
    """返回线程的上传目录，如需要则创建它。
    
    确保上传目录存在，如果不存在则递归创建。
    
    Args:
        thread_id: 线程 ID
        
    Returns:
        上传目录的 Path 对象（已确保存在）
        
    Raises:
        ValueError: 如果 thread_id 无效
        OSError: 如果无法创建目录（权限不足等）
        
    Note:
        - 使用 parents=True 递归创建所有父目录
        - 使用 exist_ok=True 避免目录已存在时抛出异常
    """
    base = get_uploads_dir(thread_id)
    base.mkdir(parents=True, exist_ok=True)
    return base


def normalize_filename(filename: str) -> str:
    """通过提取 basename 来清理文件名。
    
    去除任何目录组件并拒绝路径遍历模式，确保文件名安全。
    
    Args:
        filename: 来自用户输入的原始文件名（可能包含路径组件）
        
    Returns:
        安全的文件名（仅 basename）
        
    Raises:
        ValueError: 如果文件名为空或解析为遍历模式
        
    Note:
        - 使用 Path.name 提取 basename，自动去除路径分隔符
        - 拒绝 '.' 和 '..' 等特殊名称
        - 拒绝包含反斜杠的文件名（Windows 风格路径）
        - 限制 UTF-8 编码后的长度不超过 255 字节
        - 此函数是防止路径遍历攻击的第一道防线
    """
    if not filename:
        raise ValueError("Filename is empty")
    safe = Path(filename).name
    if not safe or safe in {".", ".."}:
        raise ValueError(f"Filename is unsafe: {filename!r}")
    # 拒绝反斜杠 —— 在 Linux 上 Path.name 会将其保留为字面字符，
    # 但它们表示应该被清理或拒绝的 Windows 风格路径。
    if "\\" in safe:
        raise ValueError(f"Filename contains backslash: {filename!r}")
    if len(safe.encode("utf-8")) > 255:
        raise ValueError(f"Filename too long: {len(safe)} chars")
    return safe


def claim_unique_filename(name: str, seen: set[str]) -> str:
    """通过在碰撞时附加 ``_N`` 后缀来生成唯一文件名。
    
    自动将返回的名称添加到 *seen* 集合中，调用者无需手动添加。
    
    Args:
        name: 候选文件名
        seen: 已声明的文件名集合（会被原地修改）
        
    Returns:
        不在 *seen* 中的文件名（已添加到 *seen*）
        
    Note:
        - 如果名称未冲突，直接返回原名称
        - 如果冲突，在 stem 和 suffix 之间插入 _1, _2, ... 直到找到唯一名称
        - 例如：file.txt → file_1.txt → file_2.txt ...
        - 此函数会修改 seen 集合，属于有副作用的操作
    """
    if name not in seen:
        seen.add(name)
        return name
    stem, suffix = Path(name).stem, Path(name).suffix
    counter = 1
    candidate = f"{stem}_{counter}{suffix}"
    while candidate in seen:
        counter += 1
        candidate = f"{stem}_{counter}{suffix}"
    seen.add(candidate)
    return candidate


def validate_path_traversal(path: Path, base: Path) -> None:
    """验证 *path* 是否在 *base* 目录内。
    
    通过解析符号链接并检查相对路径来防止路径遍历攻击。
    
    Args:
        path: 要验证的路径
        base: 基准目录（允许的路径根）
        
    Raises:
        PathTraversalError: 如果检测到路径遍历
        
    Note:
        - 使用 resolve() 解析所有符号链接和 '..' 组件
        - 使用 relative_to() 检查 path 是否是 base 的子路径
        - 这是防止访问 uploads 目录外文件的关键安全检查
        - 必须在所有文件操作之前调用
    """
    try:
        path.resolve().relative_to(base.resolve())
    except ValueError:
        raise PathTraversalError("Path traversal detected") from None


def open_upload_file_no_symlink(base_dir: Path, filename: str) -> tuple[Path, object]:
    """安全地打开上传目标文件进行流式写入。
    
    上传目录可能被挂载到本地沙箱中。沙箱进程可以在未来的上传文件名处留下
    符号链接。普通的 ``Path.write_bytes`` 会跟随该链接并使用 gateway 权限
    覆盖 uploads 目录外的文件。此助手使用 ``O_NOFOLLOW`` 在 POSIX 上拒绝
    符号链接目标。在 Windows（缺少 ``O_NOFOLLOW``）上，它使用双重 ``lstat`` 
    检查和 ``open()`` 后的 ``fstat`` 验证来减少 TOCTOU 窗口；这不能消除所有
    竞态条件，但使利用变得显著困难。路径遍历验证在两种情况下都能防止逃离 *base_dir*。
    
    Args:
        base_dir: 上传目录的基路径
        filename: 要打开的文件名
        
    Returns:
        元组 (dest_path, file_handle)
        - dest_path: 目标文件的 Path 对象
        - file_handle: 已打开的文件句柄（二进制写模式）
        
    Raises:
        UnsafeUploadPathError: 如果目标不是常规文件或是符号链接
        PathTraversalError: 如果检测到路径遍历
        OSError: 如果打开文件失败（权限不足、磁盘满等）
        
    Note:
        POSIX 实现：
        - 使用 O_NOFOLLOW 标志，如果目标是符号链接则 open() 失败并返回 ELOOP
        - 使用 O_NONBLOCK 避免阻塞（针对设备文件等特殊情况）
        - 打开后使用 fstat 验证是常规文件且硬链接数为 1
        - 使用 ftruncate 清空文件内容
        
        Windows 实现：
        - 没有 O_NOFOLLOW 可用，使用两次 lstat 缩小 TOCTOU 窗口
        - 第一次 lstat 在 normalize_filename 后
        - 第二次 lstat 在 open() 前立即执行
        - open() 后使用 fstat 进一步防御
        - 注意：pre-open lstat 和 open() 之间仍存在狭窄的竞态窗口
        
        通用安全措施：
        - 文件权限设置为 0o600（仅所有者可读写）
        - 拒绝硬链接数 > 1 的文件（可能是硬链接攻击）
        - 始终使用 follow_symlinks=False 进行 stat 检查
    """
    safe_name = normalize_filename(filename)
    dest = base_dir / safe_name

    try:
        st = os.lstat(dest)
    except FileNotFoundError:
        st = None

    if st is not None and not stat.S_ISREG(st.st_mode):
        raise UnsafeUploadPathError(f"Upload destination is not a regular file: {safe_name}")

    validate_path_traversal(dest, base_dir)

    has_nofollow = hasattr(os, "O_NOFOLLOW")

    if has_nofollow:
        # POSIX: O_NOFOLLOW 使 open() 在 dest 是符号链接时以 ELOOP 失败。
        flags = os.O_WRONLY | os.O_CREAT | os.O_NOFOLLOW
        if hasattr(os, "O_NONBLOCK"):
            flags |= os.O_NONBLOCK

        try:
            fd = os.open(dest, flags, 0o600)
        except OSError as exc:
            if exc.errno in {errno.ELOOP, errno.EISDIR, errno.ENOTDIR, errno.ENXIO, errno.EAGAIN}:
                raise UnsafeUploadPathError(f"Unsafe upload destination: {safe_name}") from exc
            raise

        try:
            opened_stat = os.fstat(fd)
            if not stat.S_ISREG(opened_stat.st_mode) or opened_stat.st_nlink != 1:
                raise UnsafeUploadPathError(f"Upload destination is not an exclusive regular file: {safe_name}")
            os.ftruncate(fd, 0)
            fh = os.fdopen(fd, "wb")
            fd = -1
        finally:
            if fd >= 0:
                os.close(fd)
        return dest, fh

    # Windows: 没有 O_NOFOLLOW 可用。使用第二次 lstat 立即在 open() 之前
    # 来缩小 TOCTOU 窗口，然后在 open() 之后使用 fstat 作为进一步防御。
    # 注意：pre-open lstat 和 open() 之间仍存在狭窄的竞态窗口；
    # 路径遍历检查减轻了从 base_dir 逃逸但不能防止在检查后原子性地将 dest
    # 替换为符号链接的攻击者。
    if st is not None and st.st_nlink > 1:
        raise UnsafeUploadPathError(f"Upload destination has multiple links: {safe_name}")

    flags = os.O_WRONLY | os.O_CREAT
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY

    try:
        pre_open_st = os.lstat(dest)
    except FileNotFoundError:
        pre_open_st = None

    if pre_open_st is not None and not stat.S_ISREG(pre_open_st.st_mode):
        raise UnsafeUploadPathError(f"Upload destination is not a regular file: {safe_name}")
    if pre_open_st is not None and pre_open_st.st_nlink > 1:
        raise UnsafeUploadPathError(f"Upload destination has multiple links: {safe_name}")

    try:
        fd = os.open(dest, flags, 0o600)
    except OSError as exc:
        if exc.errno in {errno.EISDIR, errno.ENOTDIR, errno.ENXIO, errno.EAGAIN}:
            raise UnsafeUploadPathError(f"Unsafe upload destination: {safe_name}") from exc
        raise

    try:
        opened_stat = os.fstat(fd)
        if not stat.S_ISREG(opened_stat.st_mode) or opened_stat.st_nlink > 1:
            raise UnsafeUploadPathError(f"Upload destination is not an exclusive regular file: {safe_name}")
        os.ftruncate(fd, 0)
        fh = os.fdopen(fd, "wb")
        fd = -1
    finally:
        if fd >= 0:
            os.close(fd)
    return dest, fh


def write_upload_file_no_symlink(base_dir: Path, filename: str, data: bytes) -> Path:
    """写入上传字节而不跟随预先存在的目标符号链接。
    
    便捷函数，封装了 open_upload_file_no_symlink 和写入操作。
    
    Args:
        base_dir: 上传目录的基路径
        filename: 要写入的文件名
        data: 要写入的字节数据
        
    Returns:
        写入的文件路径
        
    Raises:
        UnsafeUploadPathError: 如果目标不安全
        PathTraversalError: 如果检测到路径遍历
        OSError: 如果写入失败
        
    Note:
        - 使用 with 语句确保文件句柄正确关闭
        - 内部调用 open_upload_file_no_symlink 保证安全性
    """
    dest, fh = open_upload_file_no_symlink(base_dir, filename)
    with fh:
        fh.write(data)
    return dest


def list_files_in_dir(directory: Path) -> dict:
    """列出 *directory* 中的文件（不包括目录）。
    
    扫描指定目录，返回所有常规文件的信息列表。
    
    Args:
        directory: 要扫描的目录
        
    Returns:
        包含 "files" 列表和 "count" 的字典
        - files: 按名称排序的文件列表，每个文件包含：
          - filename: 文件名
          - size: 文件大小（字节，int 类型）
          - path: 完整路径
          - extension: 文件扩展名（含点号，如 '.txt'）
          - modified: 修改时间（Unix 时间戳）
        - count: 文件总数
        
    Note:
        - 使用 os.scandir() 提高性能（比 os.listdir() 快）
        - 使用 follow_symlinks=False 避免跟随符号链接
        - 返回的 size 是 int 类型，调用 :func:`enrich_file_listing` 可将其字符串化
          并添加虚拟路径和 artifact URL
        - 如果目录不存在，返回空列表
    """
    if not directory.is_dir():
        return {"files": [], "count": 0}

    files = []
    with os.scandir(directory) as entries:
        for entry in sorted(entries, key=lambda e: e.name):
            if not entry.is_file(follow_symlinks=False):
                continue
            st = entry.stat(follow_symlinks=False)
            files.append(
                {
                    "filename": entry.name,
                    "size": st.st_size,
                    "path": entry.path,
                    "extension": Path(entry.name).suffix,
                    "modified": st.st_mtime,
                }
            )
    return {"files": files, "count": len(files)}


def delete_file_safe(base_dir: Path, filename: str, *, convertible_extensions: set[str] | None = None) -> dict:
    """在路径遍历验证后删除 *base_dir* 内的文件。
    
    安全地删除指定文件，支持清理伴随的 markdown 文件。
    
    Args:
        base_dir: 包含文件的目录
        filename: 要删除的文件名
        convertible_extensions: 小写扩展名集合（如 ``{".pdf", ".docx"}``），
            匹配时会同时删除伴随的 markdown 文件
        
    Returns:
        包含 success 和 message 的字典
        - success: True 表示删除成功
        - message: 操作结果消息
        
    Raises:
        FileNotFoundError: 如果文件不存在
        PathTraversalError: 如果检测到路径遍历
        
    Note:
        - 使用 resolve() 解析路径并进行遍历验证
        - 如果文件扩展名在 convertible_extensions 中，会同时删除同名的 .md 文件
          （用于清理上传转换过程中生成的 markdown）
        - 使用 missing_ok=True 删除 .md 文件，即使不存在也不会报错
        - 此函数不会检查文件是否是符号链接，依赖 validate_path_traversal 保证安全
    """
    file_path = (base_dir / filename).resolve()
    validate_path_traversal(file_path, base_dir)

    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {filename}")

    file_path.unlink()

    # 清理上传转换过程中生成的伴随 markdown 文件。
    if convertible_extensions and file_path.suffix.lower() in convertible_extensions:
        file_path.with_suffix(".md").unlink(missing_ok=True)

    return {"success": True, "message": f"Deleted {filename}"}


def upload_artifact_url(thread_id: str, filename: str) -> str:
    """为线程上传目录中的文件构建 artifact URL。
    
    *filename* 会被百分号编码，以确保空格、``#``、``?`` 等字符安全。
    
    Args:
        thread_id: 线程 ID
        filename: 文件名
        
    Returns:
        artifact URL 字符串
        
    Note:
        - 使用 urllib.parse.quote 对文件名进行 URL 编码
        - safe='' 表示所有特殊字符都会被编码
        - URL 格式：/api/threads/{thread_id}/artifacts{VIRTUAL_PATH_PREFIX}/uploads/{encoded_filename}
        - 此 URL 可用于前端直接访问上传的文件
    """
    return f"/api/threads/{thread_id}/artifacts{VIRTUAL_PATH_PREFIX}/uploads/{quote(filename, safe='')}"


def upload_virtual_path(filename: str) -> str:
    """为上传目录中的文件构建虚拟路径。
    
    Args:
        filename: 文件名（不应包含路径组件）
        
    Returns:
        虚拟路径字符串
        
    Note:
        - 虚拟路径用于内部标识和资源定位
        - 格式：{VIRTUAL_PATH_PREFIX}/uploads/{filename}
        - 与 artifact URL 不同，虚拟路径不进行 URL 编码
    """
    return f"{VIRTUAL_PATH_PREFIX}/uploads/{filename}"


def enrich_file_listing(result: dict, thread_id: str) -> dict:
    """在列表结果上添加虚拟路径、artifact URL，并将大小字符串化。
    
    原地修改 *result* 并返回它以方便链式调用。
    
    Args:
        result: list_files_in_dir 返回的字典
        thread_id: 线程 ID，用于构建 artifact URL
        
    Returns:
        增强后的 result 字典（原地修改）
        
    Note:
        - 为每个文件添加以下字段：
          - size: 从 int 转换为字符串
          - virtual_path: 虚拟路径（通过 upload_virtual_path）
          - artifact_url: artifact URL（通过 upload_artifact_url）
        - 原地修改输入字典，避免创建新对象
        - 通常在 API 响应前调用此函数以提供完整的文件信息
    """
    for f in result["files"]:
        filename = f["filename"]
        f["size"] = str(f["size"])
        f["virtual_path"] = upload_virtual_path(filename)
        f["artifact_url"] = upload_artifact_url(thread_id, filename)
    return result
