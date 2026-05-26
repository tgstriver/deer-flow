"""技能文件解析模块。

本模块负责解析 SKILL.md 文件，提取 YAML 前端元数据（frontmatter）中的
技能信息（名称、描述、许可证、允许的工具列表等），构建 Skill 数据对象。
"""

import logging
import re
from pathlib import Path

import yaml

from .types import SKILL_MD_FILE, Skill, SkillCategory

logger = logging.getLogger(__name__)


def parse_allowed_tools(raw: object, skill_file: Path) -> list[str] | None:
    """解析可选的 allowed-tools 前端元数据字段。

    Parse the optional allowed-tools frontmatter field.

    Returns None when the field is omitted. Returns a list when the field is a
    YAML sequence of strings, including an empty list for explicit no-tool
    skills. Raises ValueError for malformed values.
    字段省略时返回 None；字段为 YAML 字符串序列时返回列表（包括空列表，
    表示显式声明无工具）；格式错误时抛出 ValueError。

    Args:
        raw: allowed-tools 字段的原始值（来自 YAML 解析结果）。
        skill_file: SKILL.md 文件路径，用于错误信息。

    Returns:
        list[str] | None: 工具名称列表，或 None（字段省略时）。

    Raises:
        ValueError: 当字段值不是字符串列表或包含空工具名时抛出。
    """
    # 字段省略，返回 None 表示使用默认行为（允许所有工具）
    if raw is None:
        return None
    # 字段值必须是列表
    if not isinstance(raw, list):
        raise ValueError(f"allowed-tools in {skill_file} must be a list of strings")

    allowed_tools: list[str] = []
    for item in raw:
        # 列表中的每个元素必须是字符串
        if not isinstance(item, str):
            raise ValueError(f"allowed-tools in {skill_file} must contain only strings")
        tool_name = item.strip()
        # 不允许空工具名
        if not tool_name:
            raise ValueError(f"allowed-tools in {skill_file} cannot contain empty tool names")
        allowed_tools.append(tool_name)
    return allowed_tools


def parse_skill_file(skill_file: Path, category: SkillCategory, relative_path: Path | None = None) -> Skill | None:
    """解析 SKILL.md 文件并提取元数据。

    Parse a SKILL.md file and extract metadata.

    Args:
        skill_file: Path to the SKILL.md file.
            SKILL.md 文件的路径。
        category: Category of the skill.
            技能的分类（public 或 custom）。
        relative_path: Relative path from the category root to the skill
            directory.  Defaults to the skill directory name when omitted.
            从分类根目录到技能目录的相对路径。省略时默认使用技能目录名。

    Returns:
        Skill object if parsing succeeds, None otherwise.
        解析成功返回 Skill 对象，否则返回 None。
    """
    # 文件不存在或文件名不是 SKILL.md，直接返回 None
    if not skill_file.exists() or skill_file.name != SKILL_MD_FILE:
        return None

    try:
        content = skill_file.read_text(encoding="utf-8")

        # Extract YAML front-matter block between leading ``---`` fences.
        # 提取位于起始 ``---`` 围栏之间的 YAML 前端元数据块。
        front_matter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n", content, re.DOTALL)
        if not front_matter_match:
            return None

        front_matter_text = front_matter_match.group(1)

        try:
            metadata = yaml.safe_load(front_matter_text)
        except yaml.YAMLError as exc:
            logger.error("Invalid YAML front-matter in %s: %s", skill_file, exc)
            return None

        # 确保元数据是一个字典（YAML 映射）
        if not isinstance(metadata, dict):
            logger.error("Front-matter in %s is not a YAML mapping", skill_file)
            return None

        # Extract required fields.  Both must be non-empty strings.
        # 提取必填字段，两者都必须是非空字符串。
        name = metadata.get("name")
        description = metadata.get("description")

        if not name or not isinstance(name, str):
            return None
        if not description or not isinstance(description, str):
            return None

        # Normalise: strip surrounding whitespace that YAML may preserve.
        # 规范化：去除 YAML 可能保留的首尾空白字符。
        name = name.strip()
        description = description.strip()

        if not name or not description:
            return None

        # 解析可选的 license 字段
        license_text = metadata.get("license")
        if license_text is not None:
            license_text = str(license_text).strip() or None

        # 解析可选的 allowed-tools 字段
        try:
            allowed_tools = parse_allowed_tools(metadata.get("allowed-tools"), skill_file)
        except ValueError as exc:
            logger.error("Invalid allowed-tools in %s: %s", skill_file, exc)
            return None

        return Skill(
            name=name,
            description=description,
            license=license_text,
            skill_dir=skill_file.parent,
            skill_file=skill_file,
            relative_path=relative_path or Path(skill_file.parent.name),
            category=category,
            allowed_tools=allowed_tools,
            enabled=True,  # Actual state comes from the extensions config file.
            # 实际启用状态来自扩展配置文件。
        )

    except Exception:
        logger.exception("Unexpected error parsing skill file %s", skill_file)
        return None
