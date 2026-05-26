"""技能（Skills）子系统公共接口模块。

本模块作为 deerflow.skills 包的入口，对外暴露技能系统的核心类、异常和函数，
包括 Skill 数据模型、技能安装器、存储后端以及校验工具等。
"""

from __future__ import annotations

# 导入安装器中的异常类，供外部捕获
from .installer import SkillAlreadyExistsError, SkillSecurityScanError

# 导入存储相关类和工厂函数
from .storage import LocalSkillStorage, SkillStorage, get_or_new_skill_storage

# 导入技能数据模型
from .types import Skill

# 导入前端元数据校验相关
from .validation import ALLOWED_FRONTMATTER_PROPERTIES, _validate_skill_frontmatter

# 公开 API 列表，方便 from deerflow.skills import * 时控制导出范围
__all__ = [
    "Skill",
    "ALLOWED_FRONTMATTER_PROPERTIES",
    "_validate_skill_frontmatter",
    "SkillAlreadyExistsError",
    "SkillSecurityScanError",
    "SkillStorage",
    "LocalSkillStorage",
    "get_or_new_skill_storage",
]
