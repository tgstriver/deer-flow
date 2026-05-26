"""文件展示工具 — 将输出文件展示给用户查看和渲染。

本模块实现了 present_files 工具，用于将沙箱中的输出文件展示给用户，
使其在客户端界面中可见并可交互。仅支持 /mnt/user-data/outputs 目录下的文件。

核心功能：
- 路径规范化：将宿主机路径或虚拟沙箱路径统一转换为标准虚拟路径格式
- 安全校验：确保文件路径位于当前线程的 outputs 目录内，防止路径穿越攻击
- 状态更新：通过 Command 返回规范化路径列表，由 merge_artifacts reducer 处理合并与去重
"""

from pathlib import Path
from typing import Annotated

from langchain.tools import InjectedToolCallId, tool
from langchain_core.messages import ToolMessage
from langgraph.config import get_config
from langgraph.types import Command

from deerflow.config.paths import VIRTUAL_PATH_PREFIX, get_paths
from deerflow.runtime.user_context import get_effective_user_id
from deerflow.tools.types import Runtime

# 输出文件的虚拟路径前缀，只有此路径下的文件才能被展示给用户
OUTPUTS_VIRTUAL_PREFIX = f"{VIRTUAL_PATH_PREFIX}/outputs"


def _get_thread_id(runtime: Runtime) -> str | None:
    """从运行时上下文或 RunnableConfig 中解析当前线程 ID。

    依次从以下位置获取线程 ID：
    1. runtime.context 中的 "thread_id"
    2. runtime.config["configurable"] 中的 "thread_id"
    3. LangGraph get_config() 中的 "thread_id"（兜底方案）

    Args:
        runtime: 工具运行时对象，包含上下文和配置信息。

    Returns:
        线程 ID 字符串，若无法获取则返回 None。
    """
    # 首先尝试从运行时上下文中获取线程 ID
    thread_id = runtime.context.get("thread_id") if runtime.context else None
    if thread_id:
        return thread_id

    # 尝试从运行时配置的 configurable 字段获取
    runtime_config = getattr(runtime, "config", None) or {}
    thread_id = runtime_config.get("configurable", {}).get("thread_id")
    if thread_id:
        return thread_id

    # 最后尝试从 LangGraph 当前配置上下文获取（兜底）
    try:
        return get_config().get("configurable", {}).get("thread_id")
    except RuntimeError:
        return None


def _normalize_presented_filepath(
    runtime: Runtime,
    filepath: str,
) -> str:
    """将展示文件路径规范化为 /mnt/user-data/outputs/* 的标准格式。

    接受以下两种输入格式：
    - 虚拟沙箱路径，如 /mnt/user-data/outputs/report.md
    - 宿主机侧线程输出路径，如 /app/backend/.deer-flow/threads/<thread>/user-data/outputs/report.md

    Args:
        runtime: 工具运行时对象，包含线程状态和上下文信息。
        filepath: 待规范化的文件路径，可以是虚拟路径或宿主机路径。

    Returns:
        规范化后的虚拟路径字符串。

    Raises:
        ValueError: 如果运行时元数据缺失，或路径不在当前线程的 outputs 目录内。
    """
    # 检查运行时状态是否可用
    if runtime.state is None:
        raise ValueError("Thread runtime state is not available")

    # 获取线程 ID（必须存在才能继续校验）
    thread_id = _get_thread_id(runtime)
    if not thread_id:
        raise ValueError("Thread ID is not available in runtime context or runtime config")

    # 从运行时状态中获取线程数据和输出目录路径
    thread_data = runtime.state.get("thread_data") or {}
    outputs_path = thread_data.get("outputs_path")
    if not outputs_path:
        raise ValueError("Thread outputs path is not available in runtime state")

    outputs_dir = Path(outputs_path).resolve()
    stripped = filepath.lstrip("/")
    virtual_prefix = VIRTUAL_PATH_PREFIX.lstrip("/")

    # 如果输入已经是虚拟路径格式，则通过 resolve_virtual_path 解析到实际路径
    if stripped == virtual_prefix or stripped.startswith(virtual_prefix + "/"):
        try:
            actual_path = get_paths().resolve_virtual_path(thread_id, filepath, user_id=get_effective_user_id())
        except TypeError:
            actual_path = get_paths().resolve_virtual_path(thread_id, filepath)
    else:
        # 否则将输入视为宿主机路径，直接解析
        actual_path = Path(filepath).expanduser().resolve()

    # 校验实际路径必须在输出目录内（防止路径穿越）
    try:
        relative_path = actual_path.relative_to(outputs_dir)
    except ValueError as exc:
        raise ValueError(f"Only files in {OUTPUTS_VIRTUAL_PREFIX} can be presented: {filepath}") from exc

    # 构建标准化的虚拟输出路径
    return f"{OUTPUTS_VIRTUAL_PREFIX}/{relative_path.as_posix()}"


@tool("present_files", parse_docstring=True)
def present_file_tool(
    runtime: Runtime,
    filepaths: list[str],
    tool_call_id: Annotated[str, InjectedToolCallId],
) -> Command:
    """Make files visible to the user for viewing and rendering in the client interface.

    When to use the present_files tool:

    - Making any file available for the user to view, download, or interact with
    - Presenting multiple related files at once
    - After creating files that should be presented to the user

    When NOT to use the present_files tool:
    - When you only need to read file contents for your own processing
    - For temporary or intermediate files not meant for user viewing

    Notes:
    - You should call this tool after creating files and moving them to the `/mnt/user-data/outputs` directory.
    - This tool can be safely called in parallel with other tools. State updates are handled by a reducer to prevent conflicts.

    Args:
        filepaths: List of absolute file paths to present to the user. **Only** files in `/mnt/user-data/outputs` can be presented.
    """
    # 尝试将所有文件路径规范化为标准虚拟路径
    try:
        normalized_paths = [_normalize_presented_filepath(runtime, filepath) for filepath in filepaths]
    except ValueError as exc:
        # 路径校验失败时返回错误消息
        return Command(
            update={"messages": [ToolMessage(f"Error: {exc}", tool_call_id=tool_call_id)]},
        )

    # 返回状态更新命令：merge_artifacts reducer 会处理合并和去重
    return Command(
        update={
            "artifacts": normalized_paths,
            "messages": [ToolMessage("Successfully presented files", tool_call_id=tool_call_id)],
        },
    )
