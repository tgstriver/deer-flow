"""线程状态模块：定义 LangGraph Agent 的线程级状态结构。

本模块定义了 ThreadState 类，它在 AgentState 基础上扩展了以下状态字段：
- sandbox：沙箱执行环境信息
- thread_data：线程相关的路径数据（工作空间、上传目录、输出目录）
- title：对话线程的标题
- artifacts：线程产生的工作产物列表（带自动去重 reducer）
- todos：任务列表（用于计划模式）
- uploaded_files：已上传文件列表
- viewed_images：已查看的图像字典（带合并/清空 reducer）

此外还定义了辅助的 TypedDict 子类（SandboxState、ThreadDataState、ViewedImageData）
以及两个自定义 reducer 函数（merge_artifacts、merge_viewed_images），
用于在 LangGraph 状态更新时执行合并而非简单替换。
"""

from typing import Annotated, NotRequired, TypedDict

from langchain.agents import AgentState


class SandboxState(TypedDict):
    """沙箱状态：存储沙箱环境标识符。"""

    sandbox_id: NotRequired[str | None]


class ThreadDataState(TypedDict):
    """线程数据状态：存储线程关联的本地路径信息。"""

    workspace_path: NotRequired[str | None]  # 工作空间路径
    uploads_path: NotRequired[str | None]  # 上传文件目录路径
    outputs_path: NotRequired[str | None]  # 输出文件目录路径


class ViewedImageData(TypedDict):
    """已查看图像的数据结构：包含图像的 base64 编码和 MIME 类型。"""

    base64: str  # 图像的 base64 编码字符串
    mime_type: str  # 图像的 MIME 类型（如 image/png）


def merge_artifacts(existing: list[str] | None, new: list[str] | None) -> list[str]:
    """artifacts 列表的 reducer 函数：合并新旧产物列表并去重。

    LangGraph 在更新状态时，对 Annotated 字段调用此 reducer 函数，
    而不是简单地用新值覆盖旧值。这确保了不同轮次产生的产物
    能够累积保留，同时避免重复项。

    Args:
        existing: 当前已有的产物列表，可能为 None
        new: 本次新增的产物列表，可能为 None

    Returns:
        合并并去重后的产物列表
    """
    if existing is None:
        return new or []
    if new is None:
        return existing
    # 使用 dict.fromkeys 去重，同时保持元素的原始顺序
    return list(dict.fromkeys(existing + new))


def merge_viewed_images(existing: dict[str, ViewedImageData] | None, new: dict[str, ViewedImageData] | None) -> dict[str, ViewedImageData]:
    """viewed_images 字典的 reducer 函数：合并新旧图像字典。

    特殊处理：如果 new 是空字典 {}，则清空所有已查看的图像。
    这允许中间件在处理完毕后清空 viewed_images 状态，
    避免图像数据在后续轮次中持续占用内存。

    Args:
        existing: 当前已有的图像字典，键为图像路径，可能为 None
        new: 本次新增的图像字典，可能为 None。空字典表示清空所有图像

    Returns:
        合并后的图像字典；若 new 为空字典则返回空字典
    """
    if existing is None:
        return new or {}
    if new is None:
        return existing
    # 特殊情况：空字典表示清空所有已查看的图像
    if len(new) == 0:
        return {}
    # 合并字典，相同键的新值覆盖旧值
    return {**existing, **new}


class ThreadState(AgentState):
    """线程状态：在 LangGraph AgentState 基础上扩展的线程级状态类。

    ThreadState 继承自 AgentState，增加了以下与线程执行相关的状态字段：
    - sandbox：沙箱环境标识信息
    - thread_data：线程关联的路径数据（工作空间、上传目录、输出目录）
    - title：对话标题
    - artifacts：工作产物列表（使用 merge_artifacts reducer 自动合并去重）
    - todos：任务列表（计划模式使用）
    - uploaded_files：已上传文件信息列表
    - viewed_images：已查看图像字典（使用 merge_viewed_images reducer 合并/清空）

    Annotated 类型标注指定了自定义 reducer 函数，确保 LangGraph 状态更新时
    对这些字段执行合并操作而非简单替换。
    """

    sandbox: NotRequired[SandboxState | None]  # 沙箱环境信息
    thread_data: NotRequired[ThreadDataState | None]  # 线程路径数据
    title: NotRequired[str | None]  # 对话标题
    artifacts: Annotated[list[str], merge_artifacts]  # 工作产物列表（自动合并去重）
    todos: NotRequired[list | None]  # 任务列表（计划模式）
    uploaded_files: NotRequired[list[dict] | None]  # 已上传文件信息
    viewed_images: Annotated[dict[str, ViewedImageData], merge_viewed_images]  # 已查看图像字典（自动合并）
