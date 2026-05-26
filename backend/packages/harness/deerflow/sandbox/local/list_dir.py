"""目录列表工具 — 递归遍历文件系统目录并返回结构化的路径列表。

本模块提供 list_dir 函数，用于在本地沙箱中列出指定目录下的文件和子目录。
支持深度限制、符号链接安全检查（防止通过符号链接逃逸到根目录外）、
以及基于 IGNORE_PATTERNS 的文件名过滤。
"""

from pathlib import Path

from deerflow.sandbox.search import should_ignore_name


def list_dir(path: str, max_depth: int = 2) -> list[str]:
    """递归列出目录内容，最多遍历到指定深度。

    安全特性：
    - 符号链接解析后检查是否仍在根目录范围内，防止通过符号链接逃逸
    - 使用 should_ignore_name 过滤掉版本控制目录、缓存目录等不应暴露的条目
    - 目录路径以 "/" 结尾标识，便于调用方区分文件和目录

    Args:
        path: 要列出的根目录路径。
        max_depth: 最大遍历深度（默认: 2）。
                   1 = 仅直接子项，2 = 子项 + 孙项，以此类推。

    Returns:
        排序后的绝对路径列表，目录路径以 "/" 结尾。
        排除匹配 IGNORE_PATTERNS 的条目。
    """
    result: list[str] = []
    root_path = Path(path).resolve()

    if not root_path.is_dir():
        return result

    def _is_within_root(candidate: Path) -> bool:
        """检查候选路径是否在根目录范围内 — 防止路径逃逸。"""
        try:
            candidate.relative_to(root_path)
            return True
        except ValueError:
            return False

    def _traverse(current_path: Path, current_depth: int) -> None:
        """递归遍历目录，直到达到最大深度。/ Recursively traverse directories up to max_depth."""
        if current_depth > max_depth:
            return

        try:
            for item in current_path.iterdir():
                # 跳过匹配忽略模式的条目（如 .git、node_modules 等）
                if should_ignore_name(item.name):
                    continue

                # 符号链接特殊处理：解析后检查是否在根目录范围内
                if item.is_symlink():
                    try:
                        item_resolved = item.resolve()
                        if not _is_within_root(item_resolved):
                            # 符号链接指向根目录外，跳过以防止目录遍历攻击
                            continue
                    except OSError:
                        continue
                    # 目录以 "/" 结尾标识
                    post_fix = "/" if item_resolved.is_dir() else ""
                    result.append(str(item_resolved) + post_fix)
                    continue

                item_resolved = item.resolve()
                if not _is_within_root(item_resolved):
                    # 解析后的路径超出根目录范围，跳过
                    continue

                post_fix = "/" if item.is_dir() else ""
                result.append(str(item_resolved) + post_fix)

                # Recurse into subdirectories if not at max depth
                # 仅在未达到最大深度时递归进入子目录
                if item.is_dir() and current_depth < max_depth:
                    _traverse(item, current_depth + 1)
        except PermissionError:
            # 权限不足时静默跳过，不中断遍历
            pass

    _traverse(root_path, 1)

    return sorted(result)
