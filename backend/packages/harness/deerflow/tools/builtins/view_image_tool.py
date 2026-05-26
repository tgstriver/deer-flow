"""查看图片工具。

本模块提供 view_image_tool，用于读取和展示图片文件。
支持的安全特性：
- 路径验证：仅允许访问特定的虚拟路径（workspace、uploads、outputs）
- 文件大小限制：最大 20MB
- MIME 类型验证：通过文件头和内容双重验证
- 支持的格式：JPEG、PNG、WebP

图片数据以 base64 编码存储在状态中，供前端展示使用。
"""

import base64
import mimetypes
from pathlib import Path
from typing import Annotated

from langchain.tools import InjectedToolCallId, tool
from langchain_core.messages import ToolMessage
from langgraph.types import Command

from deerflow.agents.thread_state import ThreadDataState
from deerflow.config.paths import VIRTUAL_PATH_PREFIX
from deerflow.tools.types import Runtime

logger = None  # 避免未使用的导入警告

# 允许访问的图片虚拟路径前缀列表
# 这些路径对应于用户数据目录下的安全工作区
_ALLOWED_IMAGE_VIRTUAL_ROOTS = (
    f"{VIRTUAL_PATH_PREFIX}/workspace",  # 工作区目录
    f"{VIRTUAL_PATH_PREFIX}/uploads",  # 用户上传目录
    f"{VIRTUAL_PATH_PREFIX}/outputs",  # 输出文件目录
)
_ALLOWED_IMAGE_VIRTUAL_ROOTS_TEXT = ", ".join(_ALLOWED_IMAGE_VIRTUAL_ROOTS)

# 图片文件最大大小限制（20MB），防止内存溢出
_MAX_IMAGE_BYTES = 20 * 1024 * 1024

# 文件扩展名到 MIME 类型的映射表
# 仅支持这三种常见的 Web 图片格式
_EXTENSION_TO_MIME = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def _is_allowed_image_virtual_path(image_path: str) -> bool:
    return any(image_path == root or image_path.startswith(f"{root}/") for root in _ALLOWED_IMAGE_VIRTUAL_ROOTS)


def _detect_image_mime(image_data: bytes) -> str | None:
    """通过文件头（magic bytes）检测图片的真实 MIME 类型。

    这是比文件扩展名更可靠的检测方法，可以防止文件扩展名伪造。

    Args:
        image_data: 图片文件的原始字节数据

    Returns:
        检测到的 MIME 类型字符串，如果无法识别则返回 None

    Note:
        - JPEG: 以 \xff\xd8\xff 开头
        - PNG: 以 \x89PNG\r\n\x1a\n 开头（PNG 签名）
        - WebP: RIFF 容器格式，第 8-12 字节为 WEBP
    """
    # JPEG 文件签名
    if image_data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    # PNG 文件签名（8 字节）
    if image_data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    # WebP 文件签名（RIFF 容器 + WEBP 标识符）
    if len(image_data) >= 12 and image_data.startswith(b"RIFF") and image_data[8:12] == b"WEBP":
        return "image/webp"
    return None


def _sanitize_image_error(error: Exception, thread_data: ThreadDataState | None) -> str:
    """清理错误消息，移除可能泄露的本地文件系统路径信息。

    这是一个安全措施，防止错误消息中暴露服务器的内部路径结构。

    Args:
        error: 捕获的异常对象
        thread_data: 线程数据状态，用于路径掩码替换

    Returns:
        已清理的错误消息字符串，本地路径已被替换为占位符
    """
    from deerflow.sandbox.tools import mask_local_paths_in_output

    return mask_local_paths_in_output(f"{type(error).__name__}: {error}", thread_data)


@tool("view_image", parse_docstring=True)
def view_image_tool(
    runtime: Runtime,
    image_path: str,
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Read an image file.

    Use this tool to read an image file and make it available for display.

    When to use the view_image tool:
    - When you need to view an image file.

    When NOT to use the view_image tool:
    - For non-image files (use present_files instead)
    - For multiple files at once (use present_files instead)

    Args:
        image_path: Absolute /mnt/user-data virtual path to the image file. Common formats supported: jpg, jpeg, png, webp.
    """
    # 导入沙箱相关的验证和工具函数
    from deerflow.sandbox.exceptions import SandboxRuntimeError
    from deerflow.sandbox.tools import (
        get_thread_data,
        resolve_and_validate_user_data_path,
        validate_local_tool_path,
    )

    # 获取当前线程的数据状态（包含用户隔离信息、沙箱配置等）
    thread_data = get_thread_data(runtime)

    # 第一步：验证虚拟路径是否在允许的范围内
    # 这是第一道安全防线，防止访问系统敏感目录
    if not _is_allowed_image_virtual_path(image_path):
        return Command(
            update={
                "messages": [
                    ToolMessage(
                        f"Error: Only image paths under {_ALLOWED_IMAGE_VIRTUAL_ROOTS_TEXT} are allowed",
                        tool_call_id=tool_call_id,
                    )
                ]
            },
        )

    # 第二步：验证本地路径权限并解析为实际文件系统路径
    # validate_local_tool_path 检查路径遍历攻击和权限问题
    # resolve_and_validate_user_data_path 将虚拟路径映射到实际的用户数据目录
    try:
        validate_local_tool_path(image_path, thread_data, read_only=True)
        actual_path = resolve_and_validate_user_data_path(image_path, thread_data)
    except (PermissionError, SandboxRuntimeError) as e:
        # 路径验证失败，返回错误消息（已脱敏）
        return Command(
            update={"messages": [ToolMessage(f"Error: {str(e)}", tool_call_id=tool_call_id)]},
        )

    # 创建 Path 对象以便进行文件系统操作
    path = Path(actual_path)

    # 第三步：验证文件是否存在
    if not path.exists():
        return Command(
            update={"messages": [ToolMessage(f"Error: Image file not found: {image_path}", tool_call_id=tool_call_id)]},
        )

    # 第四步：验证路径指向的是文件而非目录
    if not path.is_file():
        return Command(
            update={"messages": [ToolMessage(f"Error: Path is not a file: {image_path}", tool_call_id=tool_call_id)]},
        )

    # 第五步：验证文件扩展名是否为支持的图片格式
    # 从预定义的映射表中查找期望的 MIME 类型
    expected_mime_type = _EXTENSION_TO_MIME.get(path.suffix.lower())
    if expected_mime_type is None:
        return Command(
            update={"messages": [ToolMessage(f"Error: Unsupported image format: {path.suffix}. Supported formats: {', '.join(_EXTENSION_TO_MIME)}", tool_call_id=tool_call_id)]},
        )

    # 第六步：通过 mimetypes 库检测 MIME 类型作为备用方案
    # 如果 mimetypes 无法识别，则使用基于扩展名的预期 MIME 类型
    mime_type, _ = mimetypes.guess_type(actual_path)
    if mime_type is None:
        mime_type = expected_mime_type

    # 第七步：检查文件大小，防止加载过大的图片导致内存问题
    try:
        image_size = path.stat().st_size
    except OSError as e:
        # 文件元数据读取失败（可能是权限问题或文件被删除）
        return Command(
            update={"messages": [ToolMessage(f"Error reading image metadata: {_sanitize_image_error(e, thread_data)}", tool_call_id=tool_call_id)]},
        )
    if image_size > _MAX_IMAGE_BYTES:
        return Command(
            update={"messages": [ToolMessage(f"Error: Image file is too large: {image_size} bytes. Maximum supported size is {_MAX_IMAGE_BYTES} bytes", tool_call_id=tool_call_id)]},
        )

    # 第八步：读取图片文件内容并转换为 base64 编码
    try:
        with open(actual_path, "rb") as f:
            image_data = f.read()
    except Exception as e:
        # 文件读取失败（IO 错误、权限问题等），错误消息已脱敏
        return Command(
            update={"messages": [ToolMessage(f"Error reading image file: {_sanitize_image_error(e, thread_data)}", tool_call_id=tool_call_id)]},
        )

    # 第九步：通过文件头内容验证图片的真实格式（防伪造检查）
    detected_mime_type = _detect_image_mime(image_data)
    if detected_mime_type is None:
        # 文件内容不是任何支持的图片格式
        return Command(
            update={"messages": [ToolMessage("Error: File contents do not match a supported image format", tool_call_id=tool_call_id)]},
        )
    if detected_mime_type != expected_mime_type:
        # 文件内容与扩展名不匹配，可能存在文件扩展名伪造
        return Command(
            update={"messages": [ToolMessage(f"Error: Image contents are {detected_mime_type}, but file extension indicates {expected_mime_type}", tool_call_id=tool_call_id)]},
        )
    # 使用检测到的真实 MIME 类型（更可靠）
    mime_type = detected_mime_type
    # 将图片数据编码为 base64 字符串，便于前端展示和传输
    image_base64 = base64.b64encode(image_data).decode("utf-8")

    # 第十步：构建新的 viewed_images 状态
    # 使用字典结构存储图片数据，键为虚拟路径，值为包含 base64 和 MIME 类型的对象
    # merge_viewed_images reducer 会自动处理与现有图片的合并逻辑
    new_viewed_images = {image_path: {"base64": image_base64, "mime_type": mime_type}}

    # 返回 Command 对象，更新线程状态中的 viewed_images 并发送成功消息
    # LangGraph 会根据此 Command 更新状态并继续执行后续节点
    return Command(
        update={"viewed_images": new_viewed_images, "messages": [ToolMessage("Successfully read image", tool_call_id=tool_call_id)]},
    )
