"""Skills 管理路由器。

本模块提供技能（Skills）的完整管理接口，核心功能包括：
- 列出所有可用技能（公共技能和自定义技能）
- 从 .skill ZIP 归档安装新技能
- 查看、编辑和删除自定义技能
- 技能启用/禁用状态管理
- 自定义技能版本历史和回滚
- 安全扫描和内容验证

架构说明：
    - 使用 SkillStorage 管理技能的持久化存储
    - 支持两种技能类别：public（公共）和 custom（自定义）
    - 自定义技能存储在用户可访问的目录，支持在线编辑
    - 所有修改操作都会触发系统提示词缓存刷新
    - 编辑和回滚操作会经过安全扫描器检查

安全特性：
    - 路径遍历防护（通过 resolve_thread_virtual_path）
    - 技能内容安全扫描（scan_skill_content）
    - 防止覆盖已存在的技能（SkillAlreadyExistsError）
    - 版本历史记录支持审计和回滚
"""
import json
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.gateway.deps import get_config
from app.gateway.path_utils import resolve_thread_virtual_path
from deerflow.agents.lead_agent.prompt import refresh_skills_system_prompt_cache_async
from deerflow.config.app_config import AppConfig
from deerflow.config.extensions_config import ExtensionsConfig, SkillStateConfig, get_extensions_config, reload_extensions_config
from deerflow.skills import Skill
from deerflow.skills.installer import SkillAlreadyExistsError
from deerflow.skills.security_scanner import scan_skill_content
from deerflow.skills.storage import get_or_new_skill_storage
from deerflow.skills.types import SKILL_MD_FILE, SkillCategory

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["skills"])


class SkillResponse(BaseModel):
    """技能信息的响应模型。
    
    用于返回单个技能的详细信息。
    
    Attributes:
        name: 技能名称
        description: 技能功能描述
        license: 许可证信息（可选）
        category: 技能类别（public 或 custom）
        enabled: 是否启用此技能，默认为 True
    """

    name: str = Field(..., description="Name of the skill")
    description: str = Field(..., description="Description of what the skill does")
    license: str | None = Field(None, description="License information")
    category: SkillCategory = Field(..., description="Category of the skill (public or custom)")
    enabled: bool = Field(default=True, description="Whether this skill is enabled")


class SkillsListResponse(BaseModel):
    """技能列表的响应模型。
    
    用于返回多个技能的集合。
    
    Attributes:
        skills: 技能响应对象列表
    """

    skills: list[SkillResponse]


class SkillUpdateRequest(BaseModel):
    """更新技能的请求模型。
    
    用于启用或禁用技能。
    
    Attributes:
        enabled: 是否启用或禁用技能
    """

    enabled: bool = Field(..., description="Whether to enable or disable the skill")


class SkillInstallRequest(BaseModel):
    """从 .skill 文件安装技能的请求模型。
    
    用于从线程的用户数据目录中的 .skill ZIP 归档安装技能。
    
    Attributes:
        thread_id: .skill 文件所在的线程 ID
        path: .skill 文件的虚拟路径（如 mnt/user-data/outputs/my-skill.skill）
    """

    thread_id: str = Field(..., description="The thread ID where the .skill file is located")
    path: str = Field(..., description="Virtual path to the .skill file (e.g., mnt/user-data/outputs/my-skill.skill)")


class SkillInstallResponse(BaseModel):
    """技能安装的响应模型。
    
    返回技能安装的结果信息。
    
    Attributes:
        success: 安装是否成功
        skill_name: 已安装的技能名称
        message: 安装结果消息
    """

    success: bool = Field(..., description="Whether the installation was successful")
    skill_name: str = Field(..., description="Name of the installed skill")
    message: str = Field(..., description="Installation result message")


class CustomSkillContentResponse(SkillResponse):
    """自定义技能内容的响应模型。
    
    继承自 SkillResponse，额外包含原始 SKILL.md 内容。
    
    Attributes:
        content: 原始 SKILL.md 文件内容
    """
    content: str = Field(..., description="Raw SKILL.md content")


class CustomSkillUpdateRequest(BaseModel):
    """更新自定义技能的请求模型。
    
    用于替换 SKILL.md 文件的内容。
    
    Attributes:
        content: 新的 SKILL.md 内容
    """
    content: str = Field(..., description="Replacement SKILL.md content")


class CustomSkillHistoryResponse(BaseModel):
    """自定义技能历史的响应模型。
    
    返回技能的所有历史变更记录。
    
    Attributes:
        history: 历史记录列表，每个记录是一个字典
    """
    history: list[dict]


class SkillRollbackRequest(BaseModel):
    """技能回滚的请求模型。
    
    用于将技能回滚到历史版本。
    
    Attributes:
        history_index: 要恢复的历史条目索引，默认为 -1（最新变更）
    """
    history_index: int = Field(default=-1, description="History entry index to restore from, defaulting to the latest change.")


def _skill_to_response(skill: Skill) -> SkillResponse:
    """将 Skill 对象转换为 SkillResponse。
    
    便捷函数，用于将内部的 Skill 数据类转换为 API 响应模型。
    
    Args:
        skill: Skill 对象
        
    Returns:
        SkillResponse 对象
        
    Note:
        - 直接映射所有字段，不进行额外处理
        - 用于统一技能信息的响应格式
    """
    return SkillResponse(
        name=skill.name,
        description=skill.description,
        license=skill.license,
        category=skill.category,
        enabled=skill.enabled,
    )


@router.get(
    "/skills",
    response_model=SkillsListResponse,
    summary="列出所有技能",
    description="从公共和自定义目录中检索所有可用技能的列表。",
)
async def list_skills(config: AppConfig = Depends(get_config)) -> SkillsListResponse:
    """获取所有可用技能的列表。
    
    从技能存储中加载所有技能（包括已禁用的），并返回它们的详细信息。
    
    Args:
        config: 应用配置（通过依赖注入获取）
        
    Returns:
        SkillsListResponse 包含所有技能的列表
        
    Raises:
        HTTPException: 500 如果加载技能失败
        
    Note:
        - 使用 enabled_only=False 加载所有技能，包括已禁用的
        - 每个技能都通过 _skill_to_response 转换为响应模型
        - 异常会记录详细日志并返回 500 错误
    """
    try:
        skills = get_or_new_skill_storage(app_config=config).load_skills(enabled_only=False)
        return SkillsListResponse(skills=[_skill_to_response(skill) for skill in skills])
    except Exception as e:
        logger.error(f"Failed to load skills: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to load skills: {str(e)}")


@router.post(
    "/skills/install",
    response_model=SkillInstallResponse,
    summary="安装技能",
    description="从位于线程用户数据目录中的 .skill 文件（ZIP 归档）安装技能。",
)
async def install_skill(request: SkillInstallRequest, config: AppConfig = Depends(get_config)) -> SkillInstallResponse:
    """从 .skill ZIP 归档安装新技能。
    
    解析请求中的虚拟路径，提取 ZIP 归档，并将其安装到自定义技能目录。
    
    Args:
        request: 安装请求，包含 thread_id 和 path
        config: 应用配置（通过依赖注入获取）
        
    Returns:
        SkillInstallResponse 包含安装结果
        
    Raises:
        HTTPException:
            - 404 如果 .skill 文件未找到
            - 409 如果技能已存在（SkillAlreadyExistsError）
            - 400 如果请求参数无效（ValueError）
            - 500 如果安装过程中发生其他错误
        
    Note:
        - 使用 resolve_thread_virtual_path 解析虚拟路径到实际文件系统路径
        - 调用 ainstall_skill_from_archive 异步安装技能
        - 安装成功后刷新系统提示词缓存
        - 异常处理区分了多种错误类型，返回不同的 HTTP 状态码
    """
    try:
        skill_file_path = resolve_thread_virtual_path(request.thread_id, request.path)
        result = await get_or_new_skill_storage(app_config=config).ainstall_skill_from_archive(skill_file_path)
        await refresh_skills_system_prompt_cache_async()
        return SkillInstallResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except SkillAlreadyExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to install skill: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to install skill: {str(e)}")


@router.get("/skills/custom", response_model=SkillsListResponse, summary="列出自定义技能")
async def list_custom_skills(config: AppConfig = Depends(get_config)) -> SkillsListResponse:
    """获取所有自定义技能的列表。
    
    仅返回 category 为 CUSTOM 的技能，不包括公共技能。
    
    Args:
        config: 应用配置（通过依赖注入获取）
        
    Returns:
        SkillsListResponse 包含所有自定义技能的列表
        
    Raises:
        HTTPException: 500 如果加载技能失败
        
    Note:
        - 从所有技能中过滤出 category == SkillCategory.CUSTOM 的技能
        - 使用 enabled_only=False 包括已禁用的自定义技能
    """
    try:
        skills = [skill for skill in get_or_new_skill_storage(app_config=config).load_skills(enabled_only=False) if skill.category == SkillCategory.CUSTOM]
        return SkillsListResponse(skills=[_skill_to_response(skill) for skill in skills])
    except Exception as e:
        logger.error("Failed to list custom skills: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to list custom skills: {str(e)}")


@router.get("/skills/custom/{skill_name}", response_model=CustomSkillContentResponse, summary="获取自定义技能内容")
async def get_custom_skill(skill_name: str, config: AppConfig = Depends(get_config)) -> CustomSkillContentResponse:
    """获取指定自定义技能的详细信息和原始内容。
    
    返回技能的基本信息和 SKILL.md 文件的完整内容。
    
    Args:
        skill_name: 技能名称
        config: 应用配置（通过依赖注入获取）
        
    Returns:
        CustomSkillContentResponse 包含技能信息和原始内容
        
    Raises:
        HTTPException:
            - 404 如果自定义技能未找到
            - 500 如果读取技能失败
        
    Note:
        - 清理 skill_name 中的换行符以防止注入攻击
        - 仅返回 category 为 CUSTOM 的技能
        - 使用 read_custom_skill 读取 SKILL.md 文件内容
    """
    try:
        skill_name = skill_name.replace("\r\n", "").replace("\n", "")
        skills = get_or_new_skill_storage(app_config=config).load_skills(enabled_only=False)
        skill = next((s for s in skills if s.name == skill_name and s.category == SkillCategory.CUSTOM), None)
        if skill is None:
            raise HTTPException(status_code=404, detail=f"Custom skill '{skill_name}' not found")
        return CustomSkillContentResponse(**_skill_to_response(skill).model_dump(), content=get_or_new_skill_storage(app_config=config).read_custom_skill(skill_name))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get custom skill %s: %s", skill_name, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get custom skill: {str(e)}")


@router.put("/skills/custom/{skill_name}", response_model=CustomSkillContentResponse, summary="编辑自定义技能")
async def update_custom_skill(skill_name: str, request: CustomSkillUpdateRequest, config: AppConfig = Depends(get_config)) -> CustomSkillContentResponse:
    """更新自定义技能的 SKILL.md 内容。
    
    替换指定技能的完整内容，并进行安全扫描和版本历史记录。
    
    Args:
        skill_name: 技能名称
        request: 包含新内容的请求对象
        config: 应用配置（通过依赖注入获取）
        
    Returns:
        CustomSkillContentResponse 包含更新后的技能信息
        
    Raises:
        HTTPException:
            - 400 如果内容无效或安全扫描阻止
            - 404 如果技能未找到
            - 500 如果更新失败
        
    Note:
        - 清理 skill_name 中的换行符
        - 使用 ensure_custom_skill_is_editable 验证技能可编辑性
        - 使用 validate_skill_markdown_content 验证 Markdown 格式
        - 调用 scan_skill_content 进行安全扫描，executable=False 禁止可执行内容
        - 记录 prev_content 和 new_content 到历史
        - 记录扫描器的决策和原因
        - 更新成功后刷新系统提示词缓存
    """
    try:
        skill_name = skill_name.replace("\r\n", "").replace("\n", "")
        storage = get_or_new_skill_storage(app_config=config)
        storage.ensure_custom_skill_is_editable(skill_name)
        storage.validate_skill_markdown_content(skill_name, request.content)
        scan = await scan_skill_content(request.content, executable=False, location=f"{skill_name}/{SKILL_MD_FILE}", app_config=config)
        if scan.decision == "block":
            raise HTTPException(status_code=400, detail=f"Security scan blocked the edit: {scan.reason}")
        prev_content = storage.read_custom_skill(skill_name)
        storage.write_custom_skill(skill_name, SKILL_MD_FILE, request.content)
        storage.append_history(
            skill_name,
            {
                "action": "human_edit",
                "author": "human",
                "thread_id": None,
                "file_path": SKILL_MD_FILE,
                "prev_content": prev_content,
                "new_content": request.content,
                "scanner": {"decision": scan.decision, "reason": scan.reason},
            },
        )
        await refresh_skills_system_prompt_cache_async()
        return await get_custom_skill(skill_name, config)
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to update custom skill %s: %s", skill_name, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update custom skill: {str(e)}")


@router.delete("/skills/custom/{skill_name}", summary="删除自定义技能")
async def delete_custom_skill(skill_name: str, config: AppConfig = Depends(get_config)) -> dict[str, bool]:
    """删除指定的自定义技能。
    
    永久删除技能文件和相关数据，并记录删除操作到历史。
    
    Args:
        skill_name: 要删除的技能名称
        config: 应用配置（通过依赖注入获取）
        
    Returns:
        包含 success: True 的字典
        
    Raises:
        HTTPException:
            - 400 如果技能名称无效
            - 404 如果技能未找到
            - 500 如果删除失败
        
    Note:
        - 清理 skill_name 中的换行符
        - 记录删除操作到历史（action: "human_delete"）
        - 删除后刷新系统提示词缓存
        - 此操作不可逆
    """
    try:
        skill_name = skill_name.replace("\r\n", "").replace("\n", "")
        storage = get_or_new_skill_storage(app_config=config)
        storage.delete_custom_skill(
            skill_name,
            history_meta={
                "action": "human_delete",
                "author": "human",
                "thread_id": None,
                "file_path": SKILL_MD_FILE,
                "prev_content": None,
                "new_content": None,
                "scanner": {"decision": "allow", "reason": "Deletion requested."},
            },
        )
        await refresh_skills_system_prompt_cache_async()
        return {"success": True}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to delete custom skill %s: %s", skill_name, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to delete custom skill: {str(e)}")


@router.get("/skills/custom/{skill_name}/history", response_model=CustomSkillHistoryResponse, summary="获取自定义技能历史")
async def get_custom_skill_history(skill_name: str, config: AppConfig = Depends(get_config)) -> CustomSkillHistoryResponse:
    """获取自定义技能的所有历史变更记录。
    
    返回技能的所有编辑、删除和回滚操作的详细记录。
    
    Args:
        skill_name: 技能名称
        config: 应用配置（通过依赖注入获取）
        
    Returns:
        CustomSkillHistoryResponse 包含历史记录列表
        
    Raises:
        HTTPException:
            - 404 如果技能未找到且没有历史文件
            - 500 如果读取历史失败
        
    Note:
        - 清理 skill_name 中的换行符
        - 检查技能是否存在或是否有历史文件
        - 使用 read_history 读取所有历史记录
        - 历史记录包含 action、author、timestamp、prev_content、new_content 等字段
    """
    try:
        skill_name = skill_name.replace("\r\n", "").replace("\n", "")
        storage = get_or_new_skill_storage(app_config=config)
        if not storage.custom_skill_exists(skill_name) and not storage.get_skill_history_file(skill_name).exists():
            raise HTTPException(status_code=404, detail=f"Custom skill '{skill_name}' not found")
        return CustomSkillHistoryResponse(history=storage.read_history(skill_name))
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to read history for %s: %s", skill_name, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to read history: {str(e)}")


@router.post("/skills/custom/{skill_name}/rollback", response_model=CustomSkillContentResponse, summary="回滚自定义技能")
async def rollback_custom_skill(skill_name: str, request: SkillRollbackRequest, config: AppConfig = Depends(get_config)) -> CustomSkillContentResponse:
    """将自定义技能回滚到历史版本。
    
    根据历史记录索引恢复技能的先前内容，并进行安全扫描。
    
    Args:
        skill_name: 技能名称
        request: 包含 history_index 的请求对象
        config: 应用配置（通过依赖注入获取）
        
    Returns:
        CustomSkillContentResponse 包含回滚后的技能信息
        
    Raises:
        HTTPException:
            - 400 如果历史为空、索引越界或选定的条目没有 prev_content
            - 400 如果安全扫描阻止回滚
            - 404 如果技能未找到
            - 500 如果回滚失败
        
    Note:
        - 清理 skill_name 中的换行符
        - 验证技能和历史文件存在
        - 检查历史记录不为空
        - 从历史记录中获取 target_content（prev_content）
        - 验证目标内容的 Markdown 格式
        - 调用 scan_skill_content 进行安全扫描
        - 即使扫描阻止也会记录回滚尝试到历史
        - 如果扫描允许，写入目标内容并记录成功回滚
        - 回滚后刷新系统提示词缓存
    """
    try:
        storage = get_or_new_skill_storage(app_config=config)
        if not storage.custom_skill_exists(skill_name) and not storage.get_skill_history_file(skill_name).exists():
            raise HTTPException(status_code=404, detail=f"Custom skill '{skill_name}' not found")
        history = storage.read_history(skill_name)
        if not history:
            raise HTTPException(status_code=400, detail=f"Custom skill '{skill_name}' has no history")
        record = history[request.history_index]
        target_content = record.get("prev_content")
        if target_content is None:
            raise HTTPException(status_code=400, detail="Selected history entry has no previous content to roll back to")
        storage.validate_skill_markdown_content(skill_name, target_content)
        scan = await scan_skill_content(target_content, executable=False, location=f"{skill_name}/{SKILL_MD_FILE}", app_config=config)
        skill_file = storage.get_custom_skill_file(skill_name)
        current_content = skill_file.read_text(encoding="utf-8") if skill_file.exists() else None
        history_entry = {
            "action": "rollback",
            "author": "human",
            "thread_id": None,
            "file_path": SKILL_MD_FILE,
            "prev_content": current_content,
            "new_content": target_content,
            "rollback_from_ts": record.get("ts"),
            "scanner": {"decision": scan.decision, "reason": scan.reason},
        }
        if scan.decision == "block":
            # 即使阻止也记录回滚尝试
            storage.append_history(skill_name, history_entry)
            raise HTTPException(status_code=400, detail=f"Rollback blocked by security scanner: {scan.reason}")
        storage.write_custom_skill(skill_name, SKILL_MD_FILE, target_content)
        storage.append_history(skill_name, history_entry)
        await refresh_skills_system_prompt_cache_async()
        return await get_custom_skill(skill_name, config)
    except HTTPException:
        raise
    except IndexError:
        raise HTTPException(status_code=400, detail="history_index is out of range")
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Failed to roll back custom skill %s: %s", skill_name, e, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to roll back custom skill: {str(e)}")


@router.get(
    "/skills/{skill_name}",
    response_model=SkillResponse,
    summary="获取技能详情",
    description="根据名称检索特定技能的详细信息。",
)
async def get_skill(skill_name: str, config: AppConfig = Depends(get_config)) -> SkillResponse:
    """获取指定技能的详细信息。
    
    根据技能名称查找并返回技能的基本信息（不包括内容）。
    
    Args:
        skill_name: 技能名称
        config: 应用配置（通过依赖注入获取）
        
    Returns:
        SkillResponse 包含技能信息
        
    Raises:
        HTTPException:
            - 404 如果技能未找到
            - 500 如果获取技能失败
        
    Note:
        - 清理 skill_name 中的换行符以防止注入攻击
        - 从所有技能中查找（包括公共和自定义）
        - 使用 enabled_only=False 包括已禁用的技能
    """
    try:
        skill_name = skill_name.replace("\r\n", "").replace("\n", "")
        skills = get_or_new_skill_storage(app_config=config).load_skills(enabled_only=False)
        skill = next((s for s in skills if s.name == skill_name), None)

        if skill is None:
            raise HTTPException(status_code=404, detail=f"Skill '{skill_name}' not found")

        return _skill_to_response(skill)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get skill {skill_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get skill: {str(e)}")


@router.put(
    "/skills/{skill_name}",
    response_model=SkillResponse,
    summary="更新技能",
    description="通过修改 extensions_config.json 文件来更新技能的启用状态。",
)
async def update_skill(skill_name: str, request: SkillUpdateRequest, config: AppConfig = Depends(get_config)) -> SkillResponse:
    """更新技能的启用/禁用状态。
    
    修改 extensions_config.json 配置文件中的技能状态，并重新加载配置。
    
    Args:
        skill_name: 技能名称
        request: 包含 enabled 状态的请求对象
        config: 应用配置（通过依赖注入获取）
        
    Returns:
        SkillResponse 包含更新后的技能信息
        
    Raises:
        HTTPException:
            - 404 如果技能未找到
            - 500 如果更新或重新加载失败
        
    Note:
        - 清理 skill_name 中的换行符
        - 验证技能存在
        - 解析配置文件路径，如果不存在则创建新文件
        - 更新 extensions_config.skills 字典中的 SkillStateConfig
        - 构建包含 mcpServers 和 skills 的配置数据
        - 将配置数据写入 JSON 文件（indent=2 格式化）
        - 调用 reload_extensions_config 重新加载配置
        - 刷新系统提示词缓存
        - 重新加载技能以验证更新成功
    """
    try:
        skill_name = skill_name.replace("\r\n", "").replace("\n", "")
        skills = get_or_new_skill_storage(app_config=config).load_skills(enabled_only=False)
        skill = next((s for s in skills if s.name == skill_name), None)

        if skill is None:
            raise HTTPException(status_code=404, detail=f"Skill '{skill_name}' not found")

        config_path = ExtensionsConfig.resolve_config_path()
        if config_path is None:
            config_path = Path.cwd().parent / "extensions_config.json"
            logger.info(f"No existing extensions config found. Creating new config at: {config_path}")

        extensions_config = get_extensions_config()
        extensions_config.skills[skill_name] = SkillStateConfig(enabled=request.enabled)

        config_data = {
            "mcpServers": {name: server.model_dump() for name, server in extensions_config.mcp_servers.items()},
            "skills": {name: {"enabled": skill_config.enabled} for name, skill_config in extensions_config.skills.items()},
        }

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)

        logger.info(f"Skills configuration updated and saved to: {config_path}")
        reload_extensions_config()
        await refresh_skills_system_prompt_cache_async()

        skills = get_or_new_skill_storage(app_config=config).load_skills(enabled_only=False)
        updated_skill = next((s for s in skills if s.name == skill_name), None)

        if updated_skill is None:
            raise HTTPException(status_code=500, detail=f"Failed to reload skill '{skill_name}' after update")

        logger.info(f"Skill '{skill_name}' enabled status updated to {request.enabled}")
        return _skill_to_response(updated_skill)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update skill {skill_name}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to update skill: {str(e)}")
