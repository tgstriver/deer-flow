"""技能归档安装的共享业务逻辑模块。

Shared skill archive installation logic.

Pure business logic — no FastAPI/HTTP dependencies.
纯业务逻辑——不依赖 FastAPI/HTTP。

Both Gateway and Client delegate to these functions.
Gateway 和 Client 均委托调用本模块中的函数来完成技能的安装。
"""

import asyncio
import concurrent.futures
import logging
import posixpath
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath

from deerflow.skills.security_scanner import scan_skill_content

logger = logging.getLogger(__name__)

# 需要进行安全扫描的提示输入子目录（包含可被注入的文本内容）
_PROMPT_INPUT_DIRS = {"references", "templates"}
# 需要进行安全扫描的文件后缀（文本类文件，可能包含提示注入）
_PROMPT_INPUT_SUFFIXES = frozenset({".json", ".markdown", ".md", ".rst", ".txt", ".yaml", ".yml"})


class SkillAlreadyExistsError(ValueError):
    """当尝试安装同名技能时抛出的异常。Raised when a skill with the same name is already installed."""


class SkillSecurityScanError(ValueError):
    """当技能归档未通过安全扫描时抛出的异常。Raised when a skill archive fails security scanning."""


def is_unsafe_zip_member(info: zipfile.ZipInfo) -> bool:
    """判断 zip 成员路径是否不安全（绝对路径或目录遍历）。

    Return True if the zip member path is absolute or attempts directory traversal.
    如果 zip 成员路径是绝对路径或尝试目录遍历（..），则返回 True。

    Args:
        info: zip 文件中的条目信息对象。

    Returns:
        bool: 路径不安全返回 True，安全返回 False。
    """
    name = info.filename
    if not name:
        return False
    # 将反斜杠统一为正斜杠，处理 Windows 风格路径
    normalized = name.replace("\\", "/")
    # 检查是否以 / 开头（Unix 绝对路径）
    if normalized.startswith("/"):
        return True
    # 使用 PurePosixPath 检查 POSIX 绝对路径
    path = PurePosixPath(normalized)
    if path.is_absolute():
        return True
    # 检查 Windows 绝对路径（如 C:\...）
    if PureWindowsPath(name).is_absolute():
        return True
    # 检查路径中是否包含 .. 目录遍历
    if ".." in path.parts:
        return True
    return False


def is_symlink_member(info: zipfile.ZipInfo) -> bool:
    """检测 zip 条目是否为符号链接。

    Detect symlinks based on the external attributes stored in the ZipInfo.
    基于 ZipInfo 中存储的外部属性检测符号链接。

    Args:
        info: zip 文件中的条目信息对象。

    Returns:
        bool: 如果是符号链接返回 True，否则返回 False。
    """
    # external_attr 的高 16 位存储 UNIX 文件模式
    mode = info.external_attr >> 16
    return stat.S_ISLNK(mode)


def should_ignore_archive_entry(path: Path) -> bool:
    """判断归档条目是否应被忽略（macOS 元数据目录和隐藏文件）。

    Return True for macOS metadata dirs and dotfiles.
    对 macOS 元数据目录（__MACOSX）和以点开头的隐藏文件返回 True。

    Args:
        path: 归档中条目的路径。

    Returns:
        bool: 应忽略返回 True，否则返回 False。
    """
    return path.name.startswith(".") or path.name == "__MACOSX"


def resolve_skill_dir_from_archive(temp_path: Path) -> Path:
    """从解压后的归档内容中定位技能根目录。

    Locate the skill root directory from extracted archive contents.

    Filters out macOS metadata (__MACOSX) and dotfiles (.DS_Store).
    过滤掉 macOS 元数据（__MACOSX）和隐藏文件（.DS_Store）。

    Args:
        temp_path: 归档解压后的临时目录路径。

    Returns:
        Path: 技能目录的路径。

    Raises:
        ValueError: 如果归档在过滤后为空。
    """
    # 过滤掉应忽略的条目，只保留有效内容
    items = [p for p in temp_path.iterdir() if not should_ignore_archive_entry(p)]
    if not items:
        raise ValueError("Skill archive is empty")
    # 如果只有一个目录，则该目录即为技能根目录（常见的单层包裹结构）
    if len(items) == 1 and items[0].is_dir():
        return items[0]
    # 多个条目时，临时目录本身即为技能根目录
    return temp_path


def safe_extract_skill_archive(
    zip_ref: zipfile.ZipFile,
    dest_path: Path,
    max_total_size: int = 512 * 1024 * 1024,
) -> None:
    """安全解压技能归档，内置多层安全防护。

    Safely extract a skill archive with security protections.

    Protections / 安全防护:
    - Reject absolute paths and directory traversal (..).
      拒绝绝对路径和目录遍历（..）。
    - Skip symlink entries instead of materialising them.
      跳过符号链接条目，而不是将其物化。
    - Enforce a hard limit on total uncompressed size (zip bomb defence).
      对总解压大小执行硬性限制（防御 zip 炸弹）。

    Args:
        zip_ref: 已打开的 ZipFile 对象。
        dest_path: 解压目标路径。
        max_total_size: 允许的最大解压总大小，默认 512MB。

    Raises:
        ValueError: 如果包含不安全成员或超过大小限制。
    """
    # 获取目标路径的绝对路径，用于后续路径逃逸检测
    dest_root = dest_path.resolve()
    total_written = 0  # 已写入的总字节数

    for info in zip_ref.infolist():
        # 检查路径是否安全（绝对路径/目录遍历）
        if is_unsafe_zip_member(info):
            raise ValueError(f"Archive contains unsafe member path: {info.filename!r}")

        # 跳过符号链接，不进行物化
        if is_symlink_member(info):
            logger.warning("Skipping symlink entry in skill archive: %s", info.filename)
            continue

        # 规范化路径并构建目标路径
        normalized_name = posixpath.normpath(info.filename.replace("\\", "/"))
        member_path = dest_root.joinpath(*PurePosixPath(normalized_name).parts)
        # 二次校验：确保解析后的路径仍在目标目录内（防路径逃逸）
        if not member_path.resolve().is_relative_to(dest_root):
            raise ValueError(f"Zip entry escapes destination: {info.filename!r}")
        # 创建父目录
        member_path.parent.mkdir(parents=True, exist_ok=True)

        # 如果是目录，直接创建
        if info.is_dir():
            member_path.mkdir(parents=True, exist_ok=True)
            continue

        # 以 64KB 块读取文件内容，同时累计总大小
        with zip_ref.open(info) as src, member_path.open("wb") as dst:
            while chunk := src.read(65536):
                total_written += len(chunk)
                # 检查是否超过大小限制（zip 炸弹防护）
                if total_written > max_total_size:
                    raise ValueError("Skill archive is too large or appears highly compressed.")
                dst.write(chunk)


def _is_script_support_file(rel_path: Path) -> bool:
    """判断相对路径是否位于 scripts 子目录下（可执行脚本文件）。"""
    return bool(rel_path.parts) and rel_path.parts[0] == "scripts"


def _should_scan_support_file(rel_path: Path) -> bool:
    """判断相对路径是否需要进行安全扫描。

    扫描规则：scripts 目录下的所有文件，以及 references/templates 目录下
    的文本类文件（根据后缀判断）。
    """
    if _is_script_support_file(rel_path):
        return True
    return bool(rel_path.parts) and rel_path.parts[0] in _PROMPT_INPUT_DIRS and rel_path.suffix.lower() in _PROMPT_INPUT_SUFFIXES


def _move_staged_skill_into_reserved_target(staging_target: Path, target: Path) -> None:
    """将暂存目录中的技能文件原子性地移动到目标位置。

    采用"先预留目录，再移动内容"的策略，确保安装操作的原子性：
    如果移动过程中出现失败，会清理已预留的目标目录。

    Args:
        staging_target: 暂存目录中待移动的技能目录路径。
        target: 最终目标目录路径。

    Raises:
        SkillAlreadyExistsError: 当目标技能已存在时抛出。
    """
    installed = False  # 标记是否成功安装
    reserved = False  # 标记是否已预留目标目录
    try:
        # 尝试创建目标目录（mode=0o700 确保仅所有者可访问）
        target.mkdir(mode=0o700)
        reserved = True
        # 将暂存目录中的所有子项逐一移动到目标目录
        for child in staging_target.iterdir():
            shutil.move(str(child), target / child.name)
        installed = True
    except FileExistsError as e:
        raise SkillAlreadyExistsError(f"Skill '{target.name}' already exists") from e
    finally:
        # 如果目录已预留但安装未成功，清理预留的目录以保持一致性
        if reserved and not installed and target.exists():
            shutil.rmtree(target)


async def _scan_skill_file_or_raise(skill_dir: Path, path: Path, skill_name: str, *, executable: bool) -> None:
    """对单个技能文件执行安全扫描，如果未通过则抛出异常。

    读取文件内容，调用安全扫描器进行检测，根据扫描决策结果决定是否放行：
    - block: 直接拒绝
    - allow: 放行
    - warn: 警告但放行
    对于可执行文件，要求必须获得 allow 决策才能通过。

    Args:
        skill_dir: 技能根目录路径。
        path: 待扫描的文件路径。
        skill_name: 技能名称，用于错误信息。
        executable: 该文件是否为可执行文件（脚本类文件需要更严格的审核）。

    Raises:
        SkillSecurityScanError: 当安全扫描未通过时抛出。
    """
    # 计算文件相对于技能目录的路径，用于定位信息
    rel_path = path.relative_to(skill_dir).as_posix()
    location = f"{skill_name}/{rel_path}"
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise SkillSecurityScanError(f"Security scan failed for skill '{skill_name}': {location} must be valid UTF-8") from e

    # 调用安全扫描器进行内容检测
    try:
        result = await scan_skill_content(content, executable=executable, location=location)
    except Exception as e:
        raise SkillSecurityScanError(f"Security scan failed for {location}: {e}") from e

    # 解析扫描器的决策结果
    decision = getattr(result, "decision", None)
    reason = str(getattr(result, "reason", "") or "No reason provided.")
    # 决策为 block 时直接拒绝
    if decision == "block":
        if rel_path == "SKILL.md":
            raise SkillSecurityScanError(f"Security scan blocked skill '{skill_name}': {reason}")
        raise SkillSecurityScanError(f"Security scan blocked {location}: {reason}")
    # 可执行文件必须获得明确的 allow 决策
    if executable and decision != "allow":
        raise SkillSecurityScanError(f"Security scan rejected executable {location}: {reason}")
    # 既不是 allow 也不是 warn 的决策视为无效
    if decision not in {"allow", "warn"}:
        raise SkillSecurityScanError(f"Security scan failed for {location}: invalid scanner decision {decision!r}")


async def _scan_skill_archive_contents_or_raise(skill_dir: Path, skill_name: str) -> None:
    """对技能归档中的所有可安装文本和脚本文件执行安全扫描。

    Run the skill security scanner against all installable text and script files.
    运行技能安全扫描器，检测所有可安装的文本和脚本文件。

    扫描顺序：
    1. 先扫描 SKILL.md 主文件（非可执行）
    2. 再递归扫描 scripts/、references/、templates/ 下的相关文件

    Args:
        skill_dir: 技能根目录路径。
        skill_name: 技能名称，用于错误信息。

    Raises:
        SkillSecurityScanError: 当安全扫描未通过时抛出。
    """
    # 首先扫描 SKILL.md 主文件
    skill_md = skill_dir / "SKILL.md"
    await _scan_skill_file_or_raise(skill_dir, skill_md, skill_name, executable=False)

    # 递归遍历技能目录下的所有文件
    for path in sorted(skill_dir.rglob("*")):
        if not path.is_file():
            continue

        rel_path = path.relative_to(skill_dir)
        # 跳过已扫描的根 SKILL.md
        if rel_path == Path("SKILL.md"):
            continue
        # 禁止嵌套的 SKILL.md（会导致技能结构混乱）
        if path.name == "SKILL.md":
            raise SkillSecurityScanError(f"Security scan failed for skill '{skill_name}': nested SKILL.md is not allowed at {skill_name}/{rel_path.as_posix()}")
        # 仅扫描符合条件的服务文件
        if not _should_scan_support_file(rel_path):
            continue

        # scripts 目录下的文件按可执行模式扫描（更严格）
        await _scan_skill_file_or_raise(skill_dir, path, skill_name, executable=_is_script_support_file(rel_path))


def _run_async_install(coro):
    """在同步上下文中运行异步安装协程。

    如果当前已有运行中的事件循环（如在 FastAPI 请求处理中），
    则在新线程中运行；否则直接使用 asyncio.run()。

    Args:
        coro: 待运行的异步协程对象。

    Returns:
        协程的返回值。
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    # 如果当前已有运行中的事件循环，则需要在新线程中运行以避免冲突
    if loop is not None and loop.is_running():
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, coro).result()
    # 没有运行中的事件循环，直接运行
    return asyncio.run(coro)
