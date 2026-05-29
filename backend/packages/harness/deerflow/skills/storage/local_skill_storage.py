"""本地文件系统实现的 ``SkillStorage``。

本模块提供基于本地文件系统的技能存储后端，核心功能包括：
- 完整的 SkillStorage 抽象接口实现
- 技能的发现、加载和解析
- 自定义技能的读写操作（原子性写入）
- 从 .skill ZIP 归档安装技能
- 技能删除和版本历史管理
- 路径安全验证和防遍历攻击

目录布局::

    <root>/
    ├── public/              # 公共技能（只读）
    │   ├── academic-review/
    │   │   └── SKILL.md
    │   ├── data-analysis/
    │   │   └── SKILL.md
    │   └── ...
    └── custom/              # 自定义技能（可编辑）
        ├── my-skill/
        │   ├── SKILL.md
        │   ├── scripts/
        │   │   └── helper.py
        │   └── references/
        │       └── doc.md
        ├── another-skill/
        │   └── SKILL.md
        └── .history/        # 历史记录（集中存储）
            ├── my-skill.jsonl
            └── another-skill.jsonl

架构说明：
    - 继承自 SkillStorage 抽象基类
    - 实现所有标记为 @abstractmethod 的方法
    - 使用主机路径（host_path）进行实际文件操作
    - 使用容器路径（container_path）生成虚拟路径（用于沙箱挂载）

安全特性：
    - 路径遍历防护（validate_relative_path）
    - 原子性写入（临时文件 + 重命名）
    - 归档内容安全扫描
    - 符号链接跟随控制
"""

from __future__ import annotations

import errno
import json
import logging
import os
import shutil
import tempfile
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from deerflow.config.runtime_paths import resolve_path
from deerflow.skills.storage.skill_storage import SKILL_MD_FILE, SkillStorage
from deerflow.skills.types import SkillCategory

logger = logging.getLogger(__name__)

# 默认的技能容器路径（Docker 容器内的挂载点）
DEFAULT_SKILLS_CONTAINER_PATH = "/mnt/skills"


class LocalSkillStorage(SkillStorage):
    """基于本地文件系统的技能存储后端。
    
    将技能数据持久化到本地文件系统，支持公共技能和自定义技能的完整
    生命周期管理。
    
    Attributes:
        _host_root: 主机文件系统上的技能根目录
                   （如 /home/user/deer-flow/skills）
        
    Note:
        - 公共技能位于 <root>/public/<name>/SKILL.md
        - 自定义技能位于 <root>/custom/<name>/SKILL.md
        - 历史记录位于 <root>/custom/.history/<name>.jsonl
        - 支持 Docker 环境中的主机-容器路径映射
    """

    def __init__(
        self,
        host_path: str | None = None,
        container_path: str = DEFAULT_SKILLS_CONTAINER_PATH,
        app_config=None,
    ) -> None:
        """初始化 LocalSkillStorage。
        
        Args:
            host_path: 主机文件系统上的技能根目录
                      如果为 None，则从应用配置中读取
            container_path: 容器内的虚拟路径（默认 /mnt/skills）
                           用于生成沙箱挂载的虚拟路径
            app_config: 可选的应用配置对象
                       如果为 None，则调用 get_app_config()
        
        Note:
            - host_path 是实际的文件系统路径，用于所有文件操作
            - container_path 是容器视角的路径，用于 Docker 卷挂载
            - 这种分离支持主机和容器之间的路径映射
            - 如果未指定 host_path，会从 config.skills.get_skills_path() 读取
        """
        super().__init__(container_path=container_path)
        if host_path is None:
            from deerflow.config import get_app_config

            config = app_config or get_app_config()
            self._host_root: Path = config.skills.get_skills_path()
        else:
            self._host_root = resolve_path(host_path)

    # ------------------------------------------------------------------
    # 抽象操作实现
    # ------------------------------------------------------------------

    def get_skills_root_path(self) -> Path:
        """获取技能根目录的主机路径。
        
        Returns:
            主机文件系统上的技能根目录（如 /home/user/deer-flow/skills）
            
        Note:
            - 这是实际的文件系统路径，用于所有文件操作
            - 与容器路径形成映射关系（通过 Docker 卷挂载）
        """
        return self._host_root

    def custom_skill_exists(self, name: str) -> bool:
        """检查自定义技能是否存在。
        
        Args:
            name: 技能名称
            
        Returns:
            True 如果 SKILL.md 文件存在，False 否则
            
        Note:
            - 通过检查 custom/<name>/SKILL.md 文件存在性判断
            - 不验证技能名称格式（由调用者负责）
        """
        return self.get_custom_skill_file(name).exists()

    def public_skill_exists(self, name: str) -> bool:
        """检查公共技能是否存在。
        
        Args:
            name: 技能名称
            
        Returns:
            True 如果 SKILL.md 文件存在，False 否则
            
        Note:
            - 通过检查 public/<name>/SKILL.md 文件存在性判断
            - 自动规范化技能名称
        """
        normalized_name = self.validate_skill_name(name)
        return (self._host_root / SkillCategory.PUBLIC.value / normalized_name / SKILL_MD_FILE).exists()

    def _iter_skill_files(self) -> Iterable[tuple[SkillCategory, Path, Path]]:
        """遍历所有技能文件，生成 (category, category_root, skill_md_path)。
        
        递归扫描 public 和 custom 目录，找到所有的 SKILL.md 文件。
        
        Yields:
            三元组元组：
            - category: 技能类别（PUBLIC 或 CUSTOM）
            - category_root: 类别根目录（如 /skills/public）
            - skill_md_path: SKILL.md 文件的完整路径
            
        Note:
            - 如果根目录不存在，直接返回（不抛出异常）
            - 跳过以 '.' 开头的隐藏目录
            - 按字母顺序排序子目录（确保确定性遍历）
            - 使用 followlinks=True 允许符号链接跟随
            - 每个技能目录只能有一个 SKILL.md（在根级别）
        """
        if not self._host_root.exists():
            return
        for category in SkillCategory:
            category_path = self._host_root / category.value
            if not category_path.exists() or not category_path.is_dir():
                continue
            for current_root, dir_names, file_names in os.walk(category_path, followlinks=True):
                dir_names[:] = sorted(name for name in dir_names if not name.startswith("."))
                if SKILL_MD_FILE not in file_names:
                    continue
                yield category, category_path, Path(current_root) / SKILL_MD_FILE

    def read_custom_skill(self, name: str) -> str:
        """读取自定义技能的 SKILL.md 内容。
        
        Args:
            name: 技能名称
            
        Returns:
            SKILL.md 的完整文本内容（包括 frontmatter）
            
        Raises:
            FileNotFoundError: 如果技能不存在
            
        Note:
            - 使用 UTF-8 编码读取
            - 不验证技能名称格式（由 get_custom_skill_dir 内部验证）
        """
        if not self.custom_skill_exists(name):
            raise FileNotFoundError(f"Custom skill '{name}' not found.")
        return (self.get_custom_skill_dir(name) / SKILL_MD_FILE).read_text(encoding="utf-8")

    def write_custom_skill(self, name: str, relative_path: str, content: str) -> None:
        """原子性写入 custom/<name>/<relative_path> 下的文本文件。
        
        使用临时文件 + 重命名的方式确保写入操作的原子性，避免
        部分写入导致的数据损坏。
        
        Args:
            name: 技能名称
            relative_path: 相对于技能目录的路径（如 SKILL.md 或 scripts/helper.py）
            content: 要写入的文本内容
            
        Raises:
            ValueError: 如果 relative_path 不安全（路径遍历攻击）
            OSError: 如果写入失败（权限不足、磁盘满等）
            
        Note:
            - 使用 tempfile.NamedTemporaryFile + Path.replace() 实现原子写入
            - 临时文件创建在目标文件的同一目录下（确保同文件系统）
            - 自动创建父目录（parents=True, exist_ok=True）
            - 使用 UTF-8 编码写入
            - delete=False 防止临时文件在关闭时自动删除
        """
        target = self.validate_relative_path(relative_path, self.get_custom_skill_dir(name))
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            dir=str(target.parent),
        ) as tmp_file:
            tmp_file.write(content)
            tmp_path = Path(tmp_file.name)
        tmp_path.replace(target)

    async def ainstall_skill_from_archive(self, archive_path: str | Path) -> dict:
        """从 .skill ZIP 归档异步安装技能。
        
        完整的技能安装流程：
        1. 验证归档文件格式和完整性
        2. 安全解压到临时目录
        3. 解析并验证 frontmatter
        4. 检查技能是否已存在
        5. 扫描归档内容的安全性
        6. 原子性地移动到目标位置（两阶段提交）
        
        Args:
            archive_path: .skill 文件的路径
            
        Returns:
            包含安装结果的字典：
            - success: True 表示成功
            - skill_name: 安装的 skill 名称
            - message: 人类可读的消息
            
        Raises:
            FileNotFoundError: 如果归档文件不存在
            ValueError: 如果文件格式无效、frontmatter 错误或技能名不安全
            SkillAlreadyExistsError: 如果技能已存在
            
        Note:
            - 使用两阶段提交：先解压到 staging 目录，再原子性移动
            - staging 目录使用特殊前缀 .installing-{name}- 便于识别和清理
            - 安全扫描包括：路径遍历检查、可执行文件检测、符号链接验证
            - 归档文件必须具有 .skill 扩展名
            - 技能名称不能包含 '/'、'\\' 或 '..'
            - 安装成功后记录日志
            
        Example:
            >>> result = await storage.ainstall_skill_from_archive("my-skill.skill")
            >>> print(result)
            {'success': True, 'skill_name': 'my-skill', 'message': "Skill 'my-skill' installed successfully"}
        """
        import zipfile

        from deerflow.skills.installer import (
            SkillAlreadyExistsError,
            _move_staged_skill_into_reserved_target,
            _scan_skill_archive_contents_or_raise,
            resolve_skill_dir_from_archive,
            safe_extract_skill_archive,
        )
        from deerflow.skills.validation import _validate_skill_frontmatter

        logger.info("Installing skill from %s", archive_path)
        path = Path(archive_path)
        if not path.is_file():
            if not path.exists():
                raise FileNotFoundError(f"Skill file not found: {archive_path}")
            raise ValueError(f"Path is not a file: {archive_path}")
        if path.suffix != ".skill":
            raise ValueError("File must have .skill extension")

        custom_dir = self._host_root / "custom"
        custom_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            try:
                zf = zipfile.ZipFile(path, "r")
            except FileNotFoundError:
                raise FileNotFoundError(f"Skill file not found: {archive_path}") from None
            except (zipfile.BadZipFile, IsADirectoryError):
                raise ValueError("File is not a valid ZIP archive") from None

            with zf:
                safe_extract_skill_archive(zf, tmp_path)

            skill_dir = resolve_skill_dir_from_archive(tmp_path)

            is_valid, message, skill_name = _validate_skill_frontmatter(skill_dir)
            if not is_valid:
                raise ValueError(f"Invalid skill: {message}")
            if not skill_name or "/" in skill_name or "\\" in skill_name or ".." in skill_name:
                raise ValueError(f"Invalid skill name: {skill_name}")

            target = custom_dir / skill_name
            if target.exists():
                raise SkillAlreadyExistsError(f"Skill '{skill_name}' already exists")

            await _scan_skill_archive_contents_or_raise(skill_dir, skill_name)

            with tempfile.TemporaryDirectory(prefix=f".installing-{skill_name}-", dir=custom_dir) as staging_root:
                staging_target = Path(staging_root) / skill_name
                shutil.copytree(skill_dir, staging_target)
                _move_staged_skill_into_reserved_target(staging_target, target)
            logger.info("Skill %r installed to %s", skill_name, target)

        return {
            "success": True,
            "skill_name": skill_name,
            "message": f"Skill '{skill_name}' installed successfully",
        }

    def delete_custom_skill(self, name: str, *, history_meta: dict | None = None) -> None:
        """删除自定义技能（验证 + 可选历史记录 + 目录移除）。
        
        完整的删除流程：
        1. 验证技能名称格式
        2. 检查技能是否存在且可编辑
        3. 如果提供 history_meta，记录删除前的内容到历史
        4. 递归删除技能目录及其所有内容
        
        Args:
            name: 技能名称
            history_meta: 可选的历史记录元数据，包含：
                         - action: 操作类型（如 "human_delete"）
                         - author: 执行者信息
                         - reason: 删除原因
                         
        Raises:
            ValueError: 如果技能名称无效
            FileNotFoundError: 如果技能不存在或是公共技能
            OSError: 如果删除失败（权限不足、文件被占用等）
            
        Note:
            - 删除操作不可逆，请谨慎使用
            - 历史记录写入失败不会阻止目录删除（仅记录警告）
            - 使用 shutil.rmtree() 递归删除整个目录
            - 只读文件系统上的历史记录写入失败会被优雅处理
        """
        self.validate_skill_name(name)
        self.ensure_custom_skill_is_editable(name)
        target = self.get_custom_skill_dir(name)
        if history_meta is not None:
            prev_content = self.read_custom_skill(name)
            try:
                self.append_history(name, {**history_meta, "prev_content": prev_content})
            except OSError as e:
                if not isinstance(e, PermissionError) and e.errno not in {errno.EACCES, errno.EPERM, errno.EROFS}:
                    raise
                logger.warning(
                    "Skipping delete history write for custom skill %s due to readonly/permission failure; continuing with skill directory removal: %s",
                    name,
                    e,
                )
        if target.exists():
            shutil.rmtree(target)

    def append_history(self, name: str, record: dict) -> None:
        """为技能追加 JSONL 历史记录条目。
        
        将变更记录追加到技能的历史文件中，每条记录占一行。
        
        Args:
            name: 技能名称
            record: 历史记录字典，应包含：
                   - action: 操作类型（edit, rollback, human_delete 等）
                   - 其他任意元数据（如 prev_content, scanner_decision 等）
                   
        Raises:
            OSError: 如果写入失败（权限不足、磁盘满等）
            
        Note:
            - 自动添加 ts 字段（UTC ISO 格式时间戳）
            - 历史记录文件位于 custom/.history/<name>.jsonl
            - 追加模式打开，不会覆盖现有记录
            - 自动创建父目录（parents=True, exist_ok=True）
            - 使用 ensure_ascii=False 支持 Unicode 字符
        """
        self.validate_skill_name(name)
        payload = {"ts": datetime.now(UTC).isoformat(), **record}
        history_path = self.get_skill_history_file(name)
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False))
            f.write("\n")

    def read_history(self, name: str) -> list[dict]:
        """读取技能的所有历史记录，按时间从旧到新排序。
        
        Args:
            name: 技能名称
            
        Returns:
            历史记录列表，每个元素是一个字典
            - 空列表表示没有历史记录或历史文件不存在
            - 按时间戳升序排列（最早的在前，因为追加顺序就是时间顺序）
            
        Note:
            - 跳过空行（容错处理）
            - 如果历史文件不存在，返回空列表而非抛出异常
            - 使用 UTF-8 编码读取
            - 每行解析为一个 JSON 对象
        """
        self.validate_skill_name(name)
        history_path = self.get_skill_history_file(name)
        if not history_path.exists():
            return []
        records: list[dict] = []
        for line in history_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            records.append(json.loads(line))
        return records
