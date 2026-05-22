"""上传文件中间件 —— 将已上传文件的信息注入到智能体上下文中。

本模块负责在智能体执行前，收集当前消息中新上传的文件以及历史消息中
已上传的文件，将其元数据（文件名、大小、路径）和文档大纲信息
格式化为 <uploaded_files> 标签块，并前置到最近一条 HumanMessage
的内容中，使模型能够感知可用的文件资源。

主要流程：
1. 从当前消息的 additional_kwargs.files 中提取新上传的文件信息；
2. 扫描线程的上传目录，收集历史文件（排除本次新上传的文件）；
3. 为每份文件提取文档大纲（或内容预览），帮助模型快速定位内容；
4. 将格式化后的文件信息注入到消息内容中。

依赖关系：
- 前端上传完成后，将文件元数据写入消息的 additional_kwargs.files 字段；
- 文件转换管线（markitdown）在上传时生成同名的 .md 文件；
- 本中间件读取 .md 文件以提取文档大纲。
"""

import logging
from pathlib import Path
from typing import NotRequired, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime

from deerflow.config.paths import Paths, get_paths
from deerflow.runtime.user_context import get_effective_user_id
from deerflow.utils.file_conversion import extract_outline

logger = logging.getLogger(__name__)


# 当文档大纲为空时，使用 .md 文件的前 N 行非空内容作为预览，
# 以便智能体在没有结构化标题的情况下仍能获取一些上下文信息
_OUTLINE_PREVIEW_LINES = 5


def _extract_outline_for_file(file_path: Path) -> tuple[list[dict], list[str]]:
    """从文件的转换结果中提取文档大纲和内容预览。

    上传管线会在文件同目录下生成 <stem>.md 文件（例如上传 report.xlsx
    后会生成 report.md）。本函数读取该 .md 文件，提取标题大纲；若大纲
    为空，则回退为读取前几行非空内容作为预览。

    参数:
        file_path: 上传文件的路径（不含 .md 后缀），函数会自动查找
                   同名的 .md 文件。

    返回:
        (outline, preview) 二元组：
        - outline: 由 ``{title, line}`` 字典组成的列表，每个条目对应
          .md 文件中的一个标题行。可能包含尾部哨兵条目（truncated=True）。
          当 .md 文件不存在或没有标题时，返回空列表。
        - preview: 当大纲非空时为空列表（无需回退）；当大纲为空时，
          返回 .md 文件前几行非空内容，使智能体在没有结构化标题的情况
          下仍有一些上下文参考。当 .md 文件不存在时，也返回空列表。
    """
    # 定位上传管线生成的同名 Markdown 文件
    md_path = file_path.with_suffix(".md")
    if not md_path.is_file():
        return [], []

    # 尝试从 .md 文件中提取结构化大纲
    outline = extract_outline(md_path)
    if outline:
        logger.debug("Extracted %d outline entries from %s", len(outline), file_path.name)
        # 大纲存在时不需要内容预览
        return outline, []

    # 大纲为空 —— 读取前几行非空内容作为预览锚点，
    # 让智能体在没有结构化标题时仍有内容可供参考
    preview: list[str] = []
    try:
        with md_path.open(encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    preview.append(stripped)
                if len(preview) >= _OUTLINE_PREVIEW_LINES:
                    break
    except Exception:
        logger.debug("Failed to read preview lines from %s", md_path, exc_info=True)
    # 返回空大纲 + 内容预览
    return [], preview


class UploadsMiddlewareState(AgentState):
    """上传文件中间件的状态模式。

    在标准 AgentState 基础上扩展了 uploaded_files 字段，
    用于在状态中保存本次消息新上传的文件信息列表。
    该字段为可选字段，仅在存在上传文件时才被填充。
    """

    uploaded_files: NotRequired[list[dict] | None]


class UploadsMiddleware(AgentMiddleware[UploadsMiddlewareState]):
    """上传文件中间件 —— 将已上传文件的信息注入到智能体上下文中。

    从当前消息的 additional_kwargs.files（前端上传完成后设置）
    读取文件元数据，并生成 <uploaded_files> 信息块前置到最后一条
    HumanMessage 的内容中，使模型知晓有哪些可用文件。

    新文件：从当前消息的 additional_kwargs.files 中提取；
    历史文件：扫描线程的上传目录，排除本次新上传的文件后收集剩余文件。
    对每份文件，会尝试提取文档大纲（或内容预览），帮助模型快速定位内容。

    状态输出:
        - uploaded_files: 本次消息新上传的文件列表
        - messages: 更新后的消息列表（最后一条 HumanMessage 内容被前置文件信息）
    """

    state_schema = UploadsMiddlewareState

    def __init__(self, base_dir: str | None = None):
        """初始化上传文件中间件。

        参数:
            base_dir: 线程数据的基础目录。若未指定，则使用 Paths 默认解析路径。
                      主要用于测试时指定自定义目录。
        """
        super().__init__()
        self._paths = Paths(base_dir) if base_dir else get_paths()

    def _format_file_entry(self, file: dict, lines: list[str]) -> None:
        """将单个文件条目格式化后追加到输出行列表。

        每个文件条目包含：文件名和大小、虚拟路径、文档大纲或内容预览。
        格式化结果直接追加到 lines 列表中。

        参数:
            file: 文件信息字典，包含 filename、size、path 等键，
                  以及可选的 outline 和 outline_preview 键。
            lines: 输出行列表，格式化内容将追加到该列表末尾。
        """
        # 将字节数转换为可读的大小表示（KB 或 MB）
        size_kb = file["size"] / 1024
        size_str = f"{size_kb:.1f} KB" if size_kb < 1024 else f"{size_kb / 1024:.1f} MB"
        lines.append(f"- {file['filename']} ({size_str})")
        lines.append(f"  Path: {file['path']}")

        # 尝试获取文档大纲，若无大纲则使用内容预览
        outline = file.get("outline") or []
        if outline:
            # 大纲存在时，检查是否被截断（标题过多时仅展示前部分）
            truncated = outline[-1].get("truncated", False)
            # 过滤掉截断哨兵条目，只保留可见的标题
            visible = [e for e in outline if not e.get("truncated")]
            lines.append("  Document outline (use `read_file` with line ranges to read sections):")
            for entry in visible:
                lines.append(f"    L{entry['line']}: {entry['title']}")
            if truncated:
                lines.append(f"    ... (showing first {len(visible)} headings; use `read_file` to explore further)")
        else:
            # 无结构化标题时，展示文档开头内容预览，帮助智能体了解文件大致内容
            preview = file.get("outline_preview") or []
            if preview:
                lines.append("  No structural headings detected. Document begins with:")
                for text in preview:
                    lines.append(f"    > {text}")
            # 提示智能体使用 grep 工具搜索关键词
            lines.append("  Use `grep` to search for keywords (e.g. `grep(pattern='keyword', path='/mnt/user-data/uploads/')`).")
        lines.append("")

    def _create_files_message(self, new_files: list[dict], historical_files: list[dict]) -> str:
        """创建格式化的文件信息消息。

        将新上传文件和历史文件分别列出，包含文件名、大小、路径、
        大纲或预览信息，并附上操作指引，全部包裹在 <uploaded_files> 标签中。

        参数:
            new_files: 本次消息中上传的文件列表。每个文件字典可能包含
                       可选的 ``outline`` 键 —— 一个由 ``{title, line}`` 字典
                       组成的列表，提取自转换后的 Markdown 文件。
            historical_files: 之前消息中上传的历史文件列表（仍然可用）。
                              格式同 new_files。

        返回:
            包裹在 <uploaded_files> 标签中的格式化字符串。
        """
        lines = ["<uploaded_files>"]

        # 列出本次消息新上传的文件
        lines.append("The following files were uploaded in this message:")
        lines.append("")
        if new_files:
            for file in new_files:
                self._format_file_entry(file, lines)
        else:
            lines.append("(empty)")
            lines.append("")

        # 列出历史消息中上传且仍然可用的文件
        if historical_files:
            lines.append("The following files were uploaded in previous messages and are still available:")
            lines.append("")
            for file in historical_files:
                self._format_file_entry(file, lines)

        # 附上操作指引，帮助模型高效利用文件资源
        lines.append("To work with these files:")
        lines.append("- Read from the file first — use the outline line numbers and `read_file` to locate relevant sections.")
        lines.append("- Use `grep` to search for keywords when you are not sure which section to look at")
        lines.append("  (e.g. `grep(pattern='revenue', path='/mnt/user-data/uploads/')`).")
        lines.append("- Use `glob` to find files by name pattern")
        lines.append("  (e.g. `glob(pattern='**/*.md', path='/mnt/user-data/uploads/')`).")
        lines.append("- Only fall back to web search if the file content is clearly insufficient to answer the question.")
        lines.append("</uploaded_files>")

        return "\n".join(lines)

    def _files_from_kwargs(self, message: HumanMessage, uploads_dir: Path | None = None) -> list[dict] | None:
        """从消息的 additional_kwargs.files 中提取文件信息。

        前端在上传文件成功后，将文件元数据写入消息的 additional_kwargs.files
        字段。每个条目包含：filename（文件名）、size（字节大小）、
        path（虚拟路径）、status（状态）。

        参数:
            message: 待检查的 HumanMessage 消息对象。
            uploads_dir: 上传文件的物理目录，用于验证文件是否实际存在。
                         当提供该参数时，物理目录中不存在的文件条目将被跳过，
                         以防止引用已删除的文件。

        返回:
            包含虚拟路径的文件字典列表；若字段不存在或为空则返回 None。
            每个文件字典包含：filename、size、path（虚拟路径）、extension。
        """
        kwargs_files = (message.additional_kwargs or {}).get("files")
        # 校验 additional_kwargs.files 必须是非空列表
        if not isinstance(kwargs_files, list) or not kwargs_files:
            return None

        files = []
        for f in kwargs_files:
            if not isinstance(f, dict):
                continue
            filename = f.get("filename") or ""
            # 安全校验：文件名不能为空，且不能包含路径分隔符（防止路径遍历攻击）
            if not filename or Path(filename).name != filename:
                continue
            # 若提供了物理上传目录，验证文件是否实际存在于磁盘上
            if uploads_dir is not None and not (uploads_dir / filename).is_file():
                continue
            files.append(
                {
                    "filename": filename,
                    "size": int(f.get("size") or 0),
                    # 构建虚拟路径，智能体通过该路径访问上传文件
                    "path": f"/mnt/user-data/uploads/{filename}",
                    "extension": Path(filename).suffix,
                }
            )
        return files if files else None

    @override
    def before_agent(self, state: UploadsMiddlewareState, runtime: Runtime) -> dict | None:
        """在智能体执行前注入已上传文件的信息。

        执行流程：
        1. 检查最后一条消息是否为 HumanMessage，若不是则跳过；
        2. 解析线程的上传目录路径（用于文件存在性校验和大纲提取）；
        3. 从当前消息的 additional_kwargs.files 提取新上传文件；
        4. 扫描上传目录收集历史文件（排除本次新上传的文件）；
        5. 为每份文件提取文档大纲或内容预览；
        6. 将格式化后的文件信息前置到 HumanMessage 内容中。

        新文件来自当前消息的 additional_kwargs.files，
        历史文件通过扫描线程的上传目录获得（排除新文件）。

        将 <uploaded_files> 上下文前置到最后一条 HumanMessage 内容中。
        原始的 additional_kwargs（包括 files 元数据）保留在更新后的消息上，
        以便前端从流式消息中读取结构化的文件信息。

        参数:
            state: 当前智能体状态。
            runtime: 包含 thread_id 的运行时上下文。

        返回:
            包含 uploaded_files 和 messages 更新的状态字典；
            若没有文件信息则返回 None。
        """
        messages = list(state.get("messages", []))
        if not messages:
            return None

        # 定位最后一条消息（即当前用户输入）
        last_message_index = len(messages) - 1
        last_message = messages[last_message_index]

        # 仅处理 HumanMessage，其他类型消息不注入文件信息
        if not isinstance(last_message, HumanMessage):
            return None

        # 解析上传目录路径，用于文件存在性校验和大纲提取
        # 优先从 runtime.context 获取 thread_id，若不可用则回退到 LangGraph 配置
        thread_id = (runtime.context or {}).get("thread_id")
        if thread_id is None:
            try:
                from langgraph.config import get_config

                thread_id = get_config().get("configurable", {}).get("thread_id")
            except RuntimeError:
                # get_config() 在可运行上下文之外（如单元测试）会抛出异常，忽略即可
                pass
        uploads_dir = self._paths.sandbox_uploads_dir(thread_id, user_id=get_effective_user_id()) if thread_id else None

        # 从当前消息的 additional_kwargs.files 中提取新上传的文件信息
        new_files = self._files_from_kwargs(last_message, uploads_dir) or []

        # 收集历史文件：扫描上传目录中的所有文件，排除本次新上传的文件
        # 这样做是为了让智能体知道之前上传过的文件仍然可用
        new_filenames = {f["filename"] for f in new_files}
        historical_files: list[dict] = []
        if uploads_dir and uploads_dir.exists():
            for file_path in sorted(uploads_dir.iterdir()):
                if file_path.is_file() and file_path.name not in new_filenames:
                    stat = file_path.stat()
                    # 为历史文件提取文档大纲或内容预览
                    outline, preview = _extract_outline_for_file(file_path)
                    historical_files.append(
                        {
                            "filename": file_path.name,
                            "size": stat.st_size,
                            "path": f"/mnt/user-data/uploads/{file_path.name}",
                            "extension": file_path.suffix,
                            "outline": outline,
                            "outline_preview": preview,
                        }
                    )

        # 为新上传的文件也提取文档大纲或内容预览，
        # 因为前端上传时不会携带大纲信息，需要在此处补充
        if uploads_dir:
            for file in new_files:
                phys_path = uploads_dir / file["filename"]
                outline, preview = _extract_outline_for_file(phys_path)
                file["outline"] = outline
                file["outline_preview"] = preview

        # 若没有任何文件（新文件和历史文件都为空），则无需注入信息
        if not new_files and not historical_files:
            return None

        logger.debug(f"New files: {[f['filename'] for f in new_files]}, historical: {[f['filename'] for f in historical_files]}")

        # 生成格式化的文件信息消息
        files_message = self._create_files_message(new_files, historical_files)

        # 提取原始消息内容 —— 需要处理字符串和列表两种格式
        # LangChain 的消息内容可能是纯字符串，也可能是多模态列表
        original_content = last_message.content
        if isinstance(original_content, str):
            # 简单情况：内容为纯字符串，直接前置文件信息
            updated_content = f"{files_message}\n\n{original_content}"
        elif isinstance(original_content, list):
            # 复杂情况：内容为列表格式（多模态消息，可能包含图片等）
            # 将文件信息作为第一个文本块前置，保留所有原始内容块（包括图片）
            files_block = {"type": "text", "text": f"{files_message}\n\n"}
            updated_content = [files_block, *original_content]
        else:
            # 其他类型的内容，保持原样不做修改
            updated_content = original_content

        # 创建包含合并内容的新消息。
        # 保留 additional_kwargs（包括 files 元数据），以便前端从流式消息中
        # 读取结构化的文件信息
        updated_message = HumanMessage(
            content=updated_content,
            id=last_message.id,
            name=last_message.name,
            additional_kwargs=last_message.additional_kwargs,
        )

        # 替换消息列表中的最后一条消息
        messages[last_message_index] = updated_message

        return {
            "uploaded_files": new_files,
            "messages": messages,
        }
