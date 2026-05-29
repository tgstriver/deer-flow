"""SkillStorage 抽象基类，提供模板方法流程。

本模块定义了技能存储的抽象接口和通用逻辑：
- SkillStorage 抽象基类：定义存储后端的统一接口
- 模板方法模式：子类实现原子操作，基类组合成完整流程
- 协议级辅助函数：名称验证、路径安全检查、Markdown 内容校验
- 路径遍历防护：确保所有文件操作在允许的目录内

架构设计：
    - 采用模板方法模式（Template Method Pattern）
    - 子类负责存储介质特定的原子操作（如文件系统读写、数据库操作）
    - 基类提供跨介质的通用逻辑（加载、验证、历史记录序列化）
    - 这种分离使得可以轻松切换不同的存储后端

使用场景：
    - LocalSkillStorage：基于本地文件系统的实现
    - 未来可扩展：DatabaseSkillStorage、CloudSkillStorage 等
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path

from deerflow.skills.types import SKILL_MD_FILE, Skill, SkillCategory  # noqa: F401

logger = logging.getLogger(__name__)

# 技能名称验证正则表达式：仅允许小写字母、数字和连字符
# 格式示例：academic-review, data-analysis, chart-visualization
_SKILL_NAME_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class SkillStorage(ABC):
    """技能存储后端的抽象基类。
    
    子类实现一组存储介质特定的原子操作；此基类提供最终的模板方法流程
    （load_skills、历史序列化、路径助手、验证），将它们与协议级助手组合。
    
    核心设计理念：
        - 模板方法模式：基类定义算法骨架，子类填充具体步骤
        - 关注点分离：存储特定逻辑 vs 通用业务逻辑
        - 开闭原则：对扩展开放（新存储后端），对修改关闭（基类不变）
    
    Attributes:
        _container_root: 容器内的根路径，默认为 /mnt/skills
    
    Note:
        - 所有公共方法都是线程安全的（除非另有说明）
        - 子类必须实现所有标记为 @abstractmethod 的方法
        - 基类方法不应被子类重写（final template methods）
    """

    def __init__(self, container_path: str = "/mnt/skills") -> None:
        """初始化 SkillStorage。
        
        Args:
            container_path: 容器内的根路径，用于生成虚拟路径
                          默认为 /mnt/skills（Docker 容器内的挂载点）
        
        Note:
            - container_path 是容器视角的路径，不是主机路径
            - 主机路径由子类通过 get_skills_root_path() 提供
            - 这种分离支持 Docker 环境中的路径映射
        """
        self._container_root = container_path

    # ------------------------------------------------------------------
    # 静态协议助手（非存储特定）
    # ------------------------------------------------------------------

    @staticmethod
    def validate_skill_name(name: str) -> str:
        """验证并规范化技能名称；返回规范化形式。
        
        技能名称必须符合严格的命名规范，以确保文件系统兼容性和 URL 安全性。
        
        Args:
            name: 待验证的技能名称
            
        Returns:
            去除首尾空格后的规范化名称
            
        Raises:
            ValueError: 如果名称不符合规范或超过长度限制
            
        Note:
            - 仅允许小写字母 (a-z)、数字 (0-9) 和连字符 (-)
            - 不能以连字符开头或结尾
            - 不能有连续的连字符
            - 最大长度 64 个字符
            - 示例有效名称：academic-review, data-analysis, chart-viz
            - 示例无效名称：My_Skill, review_tool, a-b-c-d-e-f-g-h-i-j-k-l-m-n-o-p-q-r-s-t-u-v-w-x-y-z-a-b-c-d-e-f-g-h-i-j-k-l-m-n-o-p-q-r-s-t-u-v-w-x-y-z-too-long
        """
        normalized = name.strip()
        if not _SKILL_NAME_PATTERN.fullmatch(normalized):
            raise ValueError("Skill name must be hyphen-case using lowercase letters, digits, and hyphens only.")
        if len(normalized) > 64:
            raise ValueError("Skill name must be 64 characters or fewer.")
        return normalized

    @staticmethod
    def validate_relative_path(relative_path: str, base_dir: Path) -> Path:
        """验证 *relative_path* 相对于 *base_dir* 并返回解析后的目标路径。
        
        检查 *relative_path* 非空，然后将其与 *base_dir* 连接并解析结果
       （跟随符号链接）。如果解析后的目标不在 *base_dir* 内，则抛出
        ``ValueError``。
        
        Args:
            relative_path: 相对路径字符串（如 scripts/helper.py）
            base_dir: 基准目录，解析后的路径必须在此目录内
            
        Returns:
            解析后的绝对路径
            
        Raises:
            ValueError: 如果 relative_path 为空或解析后超出 base_dir
            
        Note:
            - 使用 resolve() 解析所有符号链接和 '..' 组件
            - 使用 relative_to() 验证路径是否在基准目录内
            - 这是防止路径遍历攻击的关键安全检查
            - 必须在所有文件写入操作之前调用
            
        Example:
            >>> base = Path("/skills/custom/my-skill")
            >>> validate_relative_path("scripts/run.py", base)
            PosixPath('/skills/custom/my-skill/scripts/run.py')
        """
        if not relative_path:
            raise ValueError("relative_path must not be empty.")
        resolved_base = base_dir.resolve()
        target = (resolved_base / relative_path).resolve()
        try:
            target.relative_to(resolved_base)
        except ValueError as exc:
            raise ValueError("relative_path must resolve within the skill directory.") from exc
        return target

    @staticmethod
    def validate_skill_markdown_content(name: str, content: str) -> None:
        """验证 SKILL.md 内容：解析 frontmatter 并检查名称匹配。
        
        确保技能的 Markdown 内容包含有效的 YAML frontmatter，且其中的
        name 字段与请求的技能名称一致。
        
        Args:
            name: 请求的技能名称（用于与 frontmatter 中的 name 对比）
            content: SKILL.md 的完整文本内容（包括 frontmatter）
            
        Raises:
            ValueError: 如果 frontmatter 无效或名称不匹配
            
        Note:
            - 使用临时目录避免污染文件系统
            - 调用 _validate_skill_frontmatter 进行详细验证
            - frontmatter 必须包含 name、description、license 字段
            - 名称不匹配会导致错误，防止意外覆盖其他技能
            
        Example:
            >>> content = '''---
            ... name: my-skill
            ... description: My test skill
            ... license: MIT
            ... ---
            ... # Content here
            ... '''
            >>> validate_skill_markdown_content("my-skill", content)  # OK
            >>> validate_skill_markdown_content("other-skill", content)  # ValueError
        """
        import tempfile

        from deerflow.skills.validation import _validate_skill_frontmatter

        with tempfile.TemporaryDirectory() as tmp_dir:
            temp_skill_dir = Path(tmp_dir) / SkillStorage.validate_skill_name(name)
            temp_skill_dir.mkdir(parents=True, exist_ok=True)
            (temp_skill_dir / SKILL_MD_FILE).write_text(content, encoding="utf-8")
            is_valid, message, parsed_name = _validate_skill_frontmatter(temp_skill_dir)
            if not is_valid:
                raise ValueError(message)
            if parsed_name != name:
                raise ValueError(f"Frontmatter name '{parsed_name}' must match requested skill name '{name}'.")

    def ensure_safe_support_path(self, name: str, relative_path: str) -> Path:
        """验证并返回支持文件的解析后绝对路径。
        
        支持文件包括脚本、模板、参考资料和资产文件，必须存放在指定的
        子目录下以防止混乱和安全问题。
        
        Args:
            name: 技能名称
            relative_path: 相对于技能目录的路径（如 scripts/helper.py）
            
        Returns:
            解析后的绝对路径
            
        Raises:
            ValueError: 如果路径不安全或不在允许的子目录内
            
        Note:
            - 允许的支持文件子目录：references, templates, scripts, assets
            - 路径必须是相对的，不能包含 '..' 或空部分
            - 路径必须包含文件名（不能以 '/' 结尾）
            - 解析后的路径必须在对应的子目录内
            - 这是防止访问技能目录外文件的关键安全检查
            
        Example:
            >>> storage.ensure_safe_support_path("my-skill", "scripts/run.py")
            PosixPath('/skills/custom/my-skill/scripts/run.py')
            >>> storage.ensure_safe_support_path("my-skill", "../etc/passwd")  # ValueError
        """
        _ALLOWED_SUPPORT_SUBDIRS = {"references", "templates", "scripts", "assets"}
        skill_dir = self.get_custom_skill_dir(self.validate_skill_name(name)).resolve()
        if not relative_path or relative_path.endswith("/"):
            raise ValueError("Supporting file path must include a filename.")
        relative = Path(relative_path)
        if relative.is_absolute():
            raise ValueError("Supporting file path must be relative.")
        if any(part in {"..", ""} for part in relative.parts):
            raise ValueError("Supporting file path must not contain parent-directory traversal.")
        top_level = relative.parts[0] if relative.parts else ""
        if top_level not in _ALLOWED_SUPPORT_SUBDIRS:
            raise ValueError(f"Supporting files must live under one of: {', '.join(sorted(_ALLOWED_SUPPORT_SUBDIRS))}.")
        target = (skill_dir / relative).resolve()
        allowed_root = (skill_dir / top_level).resolve()
        try:
            target.relative_to(allowed_root)
        except ValueError as exc:
            raise ValueError("Supporting file path must stay within the selected support directory.") from exc
        return target

    # ------------------------------------------------------------------
    # 抽象原子操作（存储介质特定）
    # ------------------------------------------------------------------

    @abstractmethod
    def get_skills_root_path(self) -> Path:
        """技能根目录的绝对主机路径，用于沙箱挂载。
        
        返回主机文件系统上的实际路径，而不是容器内的虚拟路径。
        这对于 Docker 环境中的卷挂载至关重要。
        
        Returns:
            技能根目录的绝对路径（如 /home/user/deer-flow/skills/public + custom）
            
        Note:
            - 这是主机视角的路径，不是容器视角
            - 与 get_container_root() 形成映射关系
            - 来源：deerflow.skills.loader.get_skills_root_path
        """

    @abstractmethod
    def _iter_skill_files(self) -> Iterable[tuple[SkillCategory, Path, Path]]:
        """为每个 SKILL.md 生成 ``(category, category_root, skill_md_path)``。
        
        遍历所有技能目录（public 和 custom），找到所有的 SKILL.md 文件。
        
        Yields:
            三元组元组：
            - category: 技能类别（PUBLIC 或 CUSTOM）
            - category_root: 类别根目录（如 /skills/public 或 /skills/custom）
            - skill_md_path: SKILL.md 文件的完整路径
            
        Note:
            - 必须按深度优先顺序遍历子目录
            - 跳过以 '.' 开头的隐藏目录
            - 每个技能目录只能有一个 SKILL.md（在根级别）
            - 来源：从 deerflow.skills.loader.load_skills 的目录遍历逻辑中提取
        """

    @abstractmethod
    def read_custom_skill(self, name: str) -> str:
        """读取自定义技能的 SKILL.md 内容。
        
        Args:
            name: 技能名称
            
        Returns:
            SKILL.md 的完整文本内容（包括 frontmatter）
            
        Raises:
            FileNotFoundError: 如果技能不存在
            
        Note:
            - 仅适用于自定义技能（custom 类别）
            - 返回 UTF-8 编码的文本
            - 来源：deerflow.skills.manager.read_custom_skill_content
        """

    @abstractmethod
    def write_custom_skill(self, name: str, relative_path: str, content: str) -> None:
        """原子性写入 ``custom/<name>/<relative_path>`` 下的文本文件。
        
        使用临时文件 + 重命名的方式确保写入操作的原子性，避免
        部分写入导致的数据损坏。
        
        Args:
            name: 技能名称
            relative_path: 相对于技能目录的路径（如 SKILL.md 或 scripts/helper.py）
            content: 要写入的文本内容
            
        Raises:
            ValueError: 如果 relative_path 不安全
            OSError: 如果写入失败（权限不足、磁盘满等）
            
        Note:
            - 使用 tempfile.NamedTemporaryFile + Path.replace() 实现原子写入
            - 自动创建父目录（parents=True, exist_ok=True）
            - 来源：deerflow.skills.manager.atomic_write
        """

    @abstractmethod
    async def ainstall_skill_from_archive(self, archive_path: str | Path) -> dict:
        """从 ``.skill`` ZIP 归档异步安装技能。
        
        完整的技能安装流程：
        1. 验证归档文件格式和完整性
        2. 安全解压到临时目录
        3. 解析并验证 frontmatter
        4. 检查技能是否已存在
        5. 扫描归档内容的安全性
        6. 原子性地移动到目标位置
        
        Args:
            archive_path: .skill 文件的路径
            
        Returns:
            包含安装结果的字典：
            - success: True 表示成功
            - skill_name: 安装的 skill 名称
            - message: 人类可读的消息
            
        Raises:
            FileNotFoundError: 如果归档文件不存在
            ValueError: 如果文件格式无效或 frontmatter 错误
            SkillAlreadyExistsError: 如果技能已存在
            
        Note:
            - 使用两阶段提交：先解压到 staging 目录，再原子性移动
            -  staging 目录使用特殊前缀 .installing-{name}- 便于识别
            - 来源：deerflow.skills.installer.ainstall_skill_from_archive
        """

    def install_skill_from_archive(self, archive_path: str | Path) -> dict:
        """同步包装器 —— 委托给 :meth:`ainstall_skill_from_archive`。
        
        为不支持异步的代码提供同步接口，内部使用事件循环运行异步方法。
        
        Args:
            archive_path: .skill 文件的路径
            
        Returns:
            与 ainstall_skill_from_archive 相同的安装结果字典
            
        Note:
            - 使用 _run_async_install 处理事件循环
            - 如果已有运行中的事件循环，则复用；否则创建新的
            - 这是为了向后兼容同步调用场景
        """
        from deerflow.skills.installer import _run_async_install

        return _run_async_install(self.ainstall_skill_from_archive(archive_path))

    @abstractmethod
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
            - 来源：app.gateway.routers.skills.delete_custom_skill + skill_manage_tool
        """

    @abstractmethod
    def custom_skill_exists(self, name: str) -> bool:
        """检查自定义技能是否存在。
        
        Args:
            name: 技能名称
            
        Returns:
            True 如果 SKILL.md 文件存在，False 否则
            
        Note:
            - 仅检查 custom 类别的技能
            - 通过检查 SKILL.md 文件存在性判断
            - 来源：deerflow.skills.manager.custom_skill_exists
        """

    @abstractmethod
    def public_skill_exists(self, name: str) -> bool:
        """检查公共技能是否存在。
        
        Args:
            name: 技能名称
            
        Returns:
            True 如果 SKILL.md 文件存在，False 否则
            
        Note:
            - 仅检查 public 类别的技能
            - 公共技能是只读的，不能修改或删除
            - 来源：deerflow.skills.manager.public_skill_exists
        """

    @abstractmethod
    def append_history(self, name: str, record: dict) -> None:
        """为 ``name`` 追加 JSONL 历史记录条目。
        
        将变更记录追加到技能的历史文件中，每条记录占一行。
        
        Args:
            name: 技能名称
            record: 历史记录字典，应包含：
                   - action: 操作类型（edit, rollback, human_delete 等）
                   - timestamp: 时间戳（由实现自动添加）
                   - 其他任意元数据（如 prev_content, scanner_decision 等）
                   
        Raises:
            OSError: 如果写入失败（权限不足、磁盘满等）
            
        Note:
            - 使用 JSONL 格式（每行一个 JSON 对象）
            - 自动添加 ts 字段（UTC ISO 格式时间戳）
            - 历史记录文件位于 custom/.history/<name>.jsonl
            - 追加模式打开，不会覆盖现有记录
            - 来源：deerflow.skills.manager.append_history
        """

    @abstractmethod
    def read_history(self, name: str) -> list[dict]:
        """返回 ``name`` 的所有历史记录，按时间从旧到新排序。
        
        Args:
            name: 技能名称
            
        Returns:
            历史记录列表，每个元素是一个字典
            - 空列表表示没有历史记录或历史文件不存在
            - 按时间戳升序排列（最早的在前）
            
        Note:
            - 跳过空行（容错处理）
            - 如果历史文件不存在，返回空列表而非抛出异常
            - 来源：deerflow.skills.manager.read_history
        """

    # ------------------------------------------------------------------
    # 具体路径助手（布局是 SKILL.md 协议的一部分）
    # ------------------------------------------------------------------

    def get_container_root(self) -> str:
        """获取容器内的根路径。
        
        Returns:
            容器视角的根路径（如 /mnt/skills）
            
        Note:
            - 这是容器内的虚拟路径，不是主机路径
            - 用于生成沙箱挂载的虚拟路径
            - 来源：deerflow.config.skills_config.SkillsConfig.container_path 访问器
        """
        return self._container_root

    def get_custom_skill_dir(self, name: str) -> Path:
        """获取 ``custom/<name>`` 的路径。不创建目录。
        
        Args:
            name: 技能名称
            
        Returns:
            自定义技能目录的绝对路径（如 /skills/custom/my-skill）
            
        Note:
            - 仅返回路径，不检查目录是否存在
            - 自动规范化技能名称
            - 来源：deerflow.skills.manager.get_custom_skill_dir
        """
        normalized_name = self.validate_skill_name(name)
        return self.get_skills_root_path() / SkillCategory.CUSTOM.value / normalized_name

    def get_custom_skill_file(self, name: str) -> Path:
        """获取 ``custom/<name>/SKILL.md`` 的路径。
        
        Args:
            name: 技能名称
            
        Returns:
            SKILL.md 文件的绝对路径（如 /skills/custom/my-skill/SKILL.md）
            
        Note:
            - 仅返回路径，不检查文件是否存在
            - 自动规范化技能名称
            - 来源：deerflow.skills.manager.get_custom_skill_file
        """
        normalized_name = self.validate_skill_name(name)
        return self.get_custom_skill_dir(normalized_name) / SKILL_MD_FILE

    def get_skill_history_file(self, name: str) -> Path:
        """获取 ``custom/.history/<name>.jsonl`` 的路径。不创建父目录。
        
        Args:
            name: 技能名称
            
        Returns:
            历史记录文件的绝对路径（如 /skills/custom/.history/my-skill.jsonl）
            
        Note:
            - 仅返回路径，不检查文件是否存在
            - 历史文件使用 JSONL 格式（每行一个 JSON 对象）
            - 所有技能的历史记录集中在 .history 子目录
            - 来源：deerflow.skills.manager.get_skill_history_file
        """
        normalized_name = self.validate_skill_name(name)
        return self.get_skills_root_path() / SkillCategory.CUSTOM.value / ".history" / f"{normalized_name}.jsonl"

    # ------------------------------------------------------------------
    # 最终模板方法流程
    # ------------------------------------------------------------------

    def load_skills(self, *, enabled_only: bool = False) -> list[Skill]:
        """发现所有技能，合并启用状态，排序并可选过滤。
        
        完整的技能加载流程：
        1. 遍历所有技能文件（public + custom）
        2. 解析每个 SKILL.md 文件
        3. 从 extensions_config.json 读取启用状态
        4. 按名称排序
        5. 如果 enabled_only=True，仅返回启用的技能
        
        Args:
            enabled_only: 如果为 True，仅返回启用的技能；否则返回所有技能
            
        Returns:
            Skill 对象列表，按名称升序排列
            
        Note:
            - 每次都从磁盘重新读取 extensions_config.json（不使用缓存）
              这确保另一个进程的配置变更能立即生效
            - 解析失败的技能会被跳过（parse_skill_file 返回 None）
            - 如果加载 extensions_config 失败，技能保持默认启用状态
            - 来源：deerflow.skills.loader.load_skills
            
        Example:
            >>> storage.load_skills()  # 返回所有技能
            >>> storage.load_skills(enabled_only=True)  # 仅返回启用的技能
        """
        from deerflow.skills.parser import parse_skill_file

        skills_by_name: dict[str, Skill] = {}
        for category, category_root, md_path in self._iter_skill_files():
            skill = parse_skill_file(
                md_path,
                category=category,
                relative_path=md_path.parent.relative_to(category_root),
            )
            if skill:
                skills_by_name[skill.name] = skill

        skills = list(skills_by_name.values())

        # 从 extensions config 合并启用状态（每次调用都重新读取，以便
        # 立即捕获另一个进程所做的更改）。
        try:
            from deerflow.config.extensions_config import ExtensionsConfig

            extensions_config = ExtensionsConfig.from_file()
            for skill in skills:
                skill.enabled = extensions_config.is_skill_enabled(skill.name, skill.category)
        except Exception as e:
            logger.warning("Failed to load extensions config: %s", e)

        if enabled_only:
            skills = [s for s in skills if s.enabled]

        skills.sort(key=lambda s: s.name)
        return skills

    def ensure_custom_skill_is_editable(self, name: str) -> None:
        """确保自定义技能可编辑。
        
        验证技能是否存在且属于自定义类别，防止修改公共技能。
        
        Args:
            name: 技能名称
            
        Raises:
            ValueError: 如果技能是公共技能（建议创建同名自定义技能覆盖）
            FileNotFoundError: 如果技能不存在
            
        Note:
            - 公共技能是只读的，不能直接修改
            - 如需自定义公共技能，应在 custom 目录下创建同名技能
            - 此方法在编辑、删除、回滚操作前调用
            - 来源：deerflow.skills.manager.ensure_custom_skill_is_editable
            
        Example:
            >>> storage.ensure_custom_skill_is_editable("my-custom-skill")  # OK
            >>> storage.ensure_custom_skill_is_editable("public-skill")  # ValueError
            >>> storage.ensure_custom_skill_is_editable("nonexistent")  # FileNotFoundError
        """
        if self.custom_skill_exists(name):
            return
        if self.public_skill_exists(name):
            raise ValueError(f"'{name}' is a built-in skill. To customise it, create a new skill with the same name under skills/custom/.")
        raise FileNotFoundError(f"Custom skill '{name}' not found.")
