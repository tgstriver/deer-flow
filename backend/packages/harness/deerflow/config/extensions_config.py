"""MCP 服务器和技能的统一扩展配置。

本模块提供：
- MCP 服务器的完整配置管理（包括 OAuth 认证）
- 技能启用状态的配置
- 环境变量解析支持
- 单例模式的全局配置访问
- 配置文件热重载功能

支持的 MCP 传输类型：
    - stdio: 标准输入输出，通过命令启动子进程
    - sse: Server-Sent Events，基于 HTTP 的流式传输
    - http: 普通 HTTP 请求

配置文件搜索优先级：
    1. 显式指定的 config_path 参数
    2. DEER_FLOW_EXTENSIONS_CONFIG_PATH 环境变量
    3. 项目根目录下的 extensions_config.json 或 mcp_config.json
    4. backend 目录和仓库根目录（向后兼容）
"""

import json
import os
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from deerflow.config.runtime_paths import existing_project_file


class McpOAuthConfig(BaseModel):
    """MCP 服务器（HTTP/SSE 传输）的 OAuth 配置。
    
    用于配置 OAuth 2.0 认证流程，支持客户端凭证模式和刷新令牌模式。
    
    Attributes:
        enabled: 是否启用 OAuth 令牌注入，默认 True
        token_url: OAuth 令牌端点 URL，用于获取访问令牌
        grant_type: OAuth 授权类型，支持 'client_credentials' 或 'refresh_token'
        client_id: OAuth 客户端 ID（可选）
        client_secret: OAuth 客户端密钥（可选）
        refresh_token: OAuth 刷新令牌（仅用于 refresh_token 授权类型）
        scope: OAuth 作用域，指定请求的权限范围（可选）
        audience: OAuth 受众，特定于提供商的参数（可选）
        token_field: 令牌响应中包含访问令牌的字段名，默认 'access_token'
        token_type_field: 令牌响应中包含令牌类型的字段名，默认 'token_type'
        expires_in_field: 令牌响应中包含过期时间（秒）的字段名，默认 'expires_in'
        default_token_type: 当令牌响应中缺少令牌类型时的默认值，默认 'Bearer'
        refresh_skew_seconds: 在过期前多少秒开始刷新令牌，默认 60 秒
        extra_token_params: 发送到令牌端点的额外表单参数，默认为空字典
        
    Note:
        - model_config 设置 extra="allow" 允许额外的未知字段
        - 所有敏感信息（client_secret、refresh_token）应通过环境变量注入
        - refresh_skew_seconds 用于避免在请求过程中令牌过期
        
    Example:
        >>> # 客户端凭证模式配置
        >>> oauth = McpOAuthConfig(
        ...     token_url="https://auth.example.com/oauth/token",
        ...     grant_type="client_credentials",
        ...     client_id="my-client-id",
        ...     client_secret="$CLIENT_SECRET",  # 从环境变量读取
        ...     scope="read write"
        ... )
        >>> 
        >>> # 刷新令牌模式配置
        >>> oauth = McpOAuthConfig(
        ...     token_url="https://auth.example.com/oauth/token",
        ...     grant_type="refresh_token",
        ...     refresh_token="$REFRESH_TOKEN",
        ...     client_id="my-client-id"
        ... )
    """

    enabled: bool = Field(default=True, description="是否启用 OAuth 令牌注入")
    token_url: str = Field(description="OAuth 令牌端点 URL")
    grant_type: Literal["client_credentials", "refresh_token"] = Field(
        default="client_credentials",
        description="OAuth 授权类型",
    )
    client_id: str | None = Field(default=None, description="OAuth 客户端 ID")
    client_secret: str | None = Field(default=None, description="OAuth 客户端密钥")
    refresh_token: str | None = Field(default=None, description="OAuth 刷新令牌（用于 refresh_token 授权类型）")
    scope: str | None = Field(default=None, description="OAuth 作用域")
    audience: str | None = Field(default=None, description="OAuth 受众（特定于提供商）")
    token_field: str = Field(default="access_token", description="令牌响应中包含访问令牌的字段名")
    token_type_field: str = Field(default="token_type", description="令牌响应中包含令牌类型的字段名")
    expires_in_field: str = Field(default="expires_in", description="令牌响应中包含过期时间（秒）的字段名")
    default_token_type: str = Field(default="Bearer", description="当令牌响应中缺少令牌类型时的默认值")
    refresh_skew_seconds: int = Field(default=60, description="在过期前多少秒开始刷新令牌")
    extra_token_params: dict[str, str] = Field(default_factory=dict, description="发送到令牌端点的额外表单参数")
    model_config = ConfigDict(extra="allow")


class McpServerConfig(BaseModel):
    """单个 MCP 服务器的配置。
    
    定义如何连接和与 MCP 服务器通信，支持三种传输类型。
    
    Attributes:
        enabled: 是否启用此 MCP 服务器，默认 True
        type: 传输类型：'stdio'（标准输入输出）、'sse'（Server-Sent Events）或 'http'（HTTP），默认 'stdio'
        command: 启动 MCP 服务器的命令（仅用于 stdio 类型）
        args: 传递给命令的参数列表（仅用于 stdio 类型）
        env: MCP 服务器的环境变量字典
        url: MCP 服务器的 URL（仅用于 sse 或 http 类型）
        headers: HTTP 请求头字典（仅用于 sse 或 http 类型）
        oauth: OAuth 配置对象（仅用于 sse 或 http 类型），如果为 None 则不使用 OAuth
        description: 人类可读的描述，说明此 MCP 服务器提供什么功能
        
    Note:
        - stdio 类型需要 command 和可选的 args
        - sse/http 类型需要 url
        - env 中的值支持环境变量解析（以 $ 开头）
        - model_config 设置 extra="allow" 允许额外的未知字段
        
    Example:
        >>> # stdio 类型配置
        >>> server = McpServerConfig(
        ...     type="stdio",
        ...     command="npx",
        ...     args=["-y", "@modelcontextprotocol/server-filesystem"],
        ...     env={"ALLOWED_DIRS": "/home/user"},
        ...     description="文件系统访问服务器"
        ... )
        >>> 
        >>> # SSE 类型配置（带 OAuth）
        >>> server = McpServerConfig(
        ...     type="sse",
        ...     url="https://mcp.example.com/sse",
        ...     headers={"X-Custom-Header": "value"},
        ...     oauth=McpOAuthConfig(...),
        ...     description="远程 MCP 服务器"
        ... )
    """

    enabled: bool = Field(default=True, description="是否启用此 MCP 服务器")
    type: str = Field(default="stdio", description="传输类型：'stdio'、'sse' 或 'http'")
    command: str | None = Field(default=None, description="启动 MCP 服务器的命令（用于 stdio 类型）")
    args: list[str] = Field(default_factory=list, description="传递给命令的参数（用于 stdio 类型）")
    env: dict[str, str] = Field(default_factory=dict, description="MCP 服务器的环境变量")
    url: str | None = Field(default=None, description="MCP 服务器的 URL（用于 sse 或 http 类型）")
    headers: dict[str, str] = Field(default_factory=dict, description="HTTP 请求头（用于 sse 或 http 类型）")
    oauth: McpOAuthConfig | None = Field(default=None, description="OAuth 配置（用于 sse 或 http 类型）")
    description: str = Field(default="", description="人类可读的描述，说明此 MCP 服务器提供什么功能")
    model_config = ConfigDict(extra="allow")


class SkillStateConfig(BaseModel):
    """单个技能的状态配置。
    
    控制技能的启用/禁用状态。
    
    Attributes:
        enabled: 是否启用此技能，默认 True
        
    Note:
        - 如果技能未在配置中出现，公共技能和自定义技能默认启用
        - 可通过此配置显式禁用某些技能
    """

    enabled: bool = Field(default=True, description="是否启用此技能")


class ExtensionsConfig(BaseModel):
    """MCP 服务器和技能的统一配置。
    
    作为 DeerFlow 扩展系统的核心配置类，管理所有 MCP 服务器和技能的配置。
    
    Attributes:
        mcp_servers: MCP 服务器名称到配置的映射字典，别名 'mcpServers'
        skills: 技能名称到状态配置的映射字典
        
    Note:
        - model_config 设置 extra="allow" 允许额外的未知字段
        - populate_by_name=True 允许使用字段名或别名进行反序列化
        - mcp_servers 使用别名 'mcpServers' 以兼容 JSON 命名规范
        
    Example:
        >>> # 从文件加载配置
        >>> config = ExtensionsConfig.from_file()
        >>> 
        >>> # 获取启用的 MCP 服务器
        >>> enabled_servers = config.get_enabled_mcp_servers()
        >>> 
        >>> # 检查技能是否启用
        >>> if config.is_skill_enabled("web-search", "public"):
        ...     print("Web search skill is enabled")
    """

    mcp_servers: dict[str, McpServerConfig] = Field(
        default_factory=dict,
        description="MCP 服务器名称到配置的映射",
        alias="mcpServers",
    )
    skills: dict[str, SkillStateConfig] = Field(
        default_factory=dict,
        description="技能名称到状态配置的映射",
    )
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    @classmethod
    def resolve_config_path(cls, config_path: str | None = None) -> Path | None:
        """解析扩展配置文件路径。
        
        按照以下优先级查找配置文件：
        1. 如果提供了 `config_path` 参数，使用它
        2. 如果提供了 `DEER_FLOW_EXTENSIONS_CONFIG_PATH` 环境变量，使用它
        3. 否则，在调用者项目根目录中搜索 `extensions_config.json`，然后是 `mcp_config.json`
        4. 为了向后兼容，还搜索 legacy backend/仓库根目录默认位置
        5. 如果未找到，返回 None（扩展是可选的）
        
        Args:
            config_path: 可选的扩展配置文件路径
            
        Returns:
            如果找到则返回扩展配置文件的路径，否则返回 None
            
        Raises:
            FileNotFoundError: 如果显式指定的路径（通过参数或环境变量）不存在
            
        Note:
            - 步骤 1-2 是严格模式：如果指定了路径但不存在则抛出异常
            - 步骤 3-4 是宽松模式：如果找不到则返回 None
            - existing_project_file 会向上遍历目录树查找配置文件
            - 向后兼容旧版本的 mcp_config.json 文件名
            
        Resolution order:
            1. 如果提供了 `config_path` 参数，使用它
            2. 如果提供了 `DEER_FLOW_EXTENSIONS_CONFIG_PATH` 环境变量，使用它
            3. 否则，在调用者项目根目录中搜索 `extensions_config.json`，然后是 legacy `mcp_config.json`
            4. 最后，搜索 backend/仓库根目录默认位置以实现 monorepo 兼容性
        """
        if config_path:
            # 优先级1：显式参数
            path = Path(config_path)
            if not path.exists():
                raise FileNotFoundError(f"Extensions config file specified by param `config_path` not found at {path}")
            return path
        elif os.getenv("DEER_FLOW_EXTENSIONS_CONFIG_PATH"):
            # 优先级2：环境变量
            path = Path(os.getenv("DEER_FLOW_EXTENSIONS_CONFIG_PATH"))
            if not path.exists():
                raise FileNotFoundError(f"Extensions config file specified by environment variable `DEER_FLOW_EXTENSIONS_CONFIG_PATH` not found at {path}")
            return path
        else:
            # 优先级3：项目根目录
            project_config = existing_project_file(("extensions_config.json", "mcp_config.json"))
            if project_config is not None:
                return project_config

            # 优先级4：backend 和仓库根目录（向后兼容）
            backend_dir = Path(__file__).resolve().parents[4]
            repo_root = backend_dir.parent
            for path in (
                backend_dir / "extensions_config.json",
                repo_root / "extensions_config.json",
                backend_dir / "mcp_config.json",
                repo_root / "mcp_config.json",
            ):
                if path.exists():
                    return path

            # 扩展是可选的，如果未找到则返回 None
            return None

    @classmethod
    def from_file(cls, config_path: str | None = None) -> "ExtensionsConfig":
        """从 JSON 文件加载扩展配置。
        
        读取并解析配置文件，自动解析环境变量占位符。
        
        Args:
            config_path: 扩展配置文件的路径，如果为 None 则使用默认解析策略
            
        Returns:
            加载的配置对象，如果文件未找到则返回空配置
            
        Raises:
            ValueError: 如果 JSON 文件格式无效
            RuntimeError: 如果加载过程中发生其他错误
            
        Note:
            - 调用 resolve_config_path 确定文件路径
            - 自动解析以 $ 开头的环境变量（如 $OPENAI_API_KEY）
            - 如果文件不存在，返回空的 mcp_servers 和 skills 字典
            - 使用 model_validate 进行 Pydantic 验证
            
        See Also:
            resolve_config_path: 了解路径解析的详细信息
        """
        resolved_path = cls.resolve_config_path(config_path)
        if resolved_path is None:
            # 如果未找到扩展配置文件，返回空配置
            return cls(mcp_servers={}, skills={})

        try:
            with open(resolved_path, encoding="utf-8") as f:
                config_data = json.load(f)
            # 递归解析环境变量
            cls.resolve_env_variables(config_data)
            return cls.model_validate(config_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Extensions config file at {resolved_path} is not valid JSON: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Failed to load extensions config from {resolved_path}: {e}") from e

    @classmethod
    def resolve_env_variables(cls, config: dict[str, Any]) -> dict[str, Any]:
        """递归解析配置中的环境变量。
        
        使用 `os.getenv` 函数解析环境变量。示例：$OPENAI_API_KEY
        
        Args:
            config: 要解析环境变量的配置字典
            
        Returns:
            已解析环境变量的配置字典
            
        Note:
            - 只处理字符串值中以 $ 开头的占位符
            - 如果环境变量未设置，替换为空字符串而不是保留 "$VAR" 字面量
            - 递归处理嵌套的字典和列表
            - 原地修改传入的 config 字典
            
        Example:
            >>> config = {
            ...     "api_key": "$OPENAI_API_KEY",
            ...     "url": "https://api.example.com",
            ...     "nested": {"token": "$TOKEN"}
            ... }
            >>> # 假设 OPENAI_API_KEY=sk-xxx，TOKEN 未设置
            >>> ExtensionsConfig.resolve_env_variables(config)
            >>> # config 变为：
            >>> # {
            >>> #     "api_key": "sk-xxx",
            >>> #     "url": "https://api.example.com",
            >>> #     "nested": {"token": ""}
            >>> # }
        """
        for key, value in config.items():
            if isinstance(value, str):
                if value.startswith("$"):
                    # 环境变量占位符：$VAR_NAME
                    env_value = os.getenv(value[1:])
                    if env_value is None:
                        # 未解析的占位符 — 存储空字符串，以便下游消费者
                        # （例如 MCP 服务器）不会收到字面量的 "$VAR" 作为实际的环境值
                        config[key] = ""
                    else:
                        config[key] = env_value
                else:
                    config[key] = value
            elif isinstance(value, dict):
                # 递归处理嵌套字典
                config[key] = cls.resolve_env_variables(value)
            elif isinstance(value, list):
                # 递归处理列表中的字典项
                config[key] = [cls.resolve_env_variables(item) if isinstance(item, dict) else item for item in value]
        return config

    def get_enabled_mcp_servers(self) -> dict[str, McpServerConfig]:
        """获取仅启用的 MCP 服务器。
        
        过滤出所有 enabled=True 的 MCP 服务器配置。
        
        Returns:
            启用的 MCP 服务器字典，键为服务器名称，值为配置对象
            
        Note:
            - 只返回 enabled 字段为 True 的服务器
            - 可用于初始化时只连接需要的服务器
            
        Example:
            >>> config = ExtensionsConfig.from_file()
            >>> enabled = config.get_enabled_mcp_servers()
            >>> for name, server_config in enabled.items():
            ...     print(f"Connecting to {name}: {server_config.type}")
        """
        return {name: config for name, config in self.mcp_servers.items() if config.enabled}

    def is_skill_enabled(self, skill_name: str, skill_category: str) -> bool:
        """检查技能是否启用。
        
        根据配置和技能类别判断技能是否应该启用。
        
        Args:
            skill_name: 技能名称
            skill_category: 技能类别（'public'、'custom' 或其他）
            
        Returns:
            如果启用返回 True，否则返回 False
            
        Note:
            - 如果技能在配置中存在，使用配置的 enabled 值
            - 如果技能不在配置中：
              - public 和 custom 类别的技能默认启用
              - 其他类别的技能默认禁用
            - 这种设计允许选择性禁用某些技能而不影响其他技能
            
        Example:
            >>> config = ExtensionsConfig.from_file()
            >>> # 检查公共技能
            >>> if config.is_skill_enabled("web-search", "public"):
            ...     print("Web search is available")
            >>> 
            >>> # 检查自定义技能（即使未在配置中也默认启用）
            >>> if config.is_skill_enabled("my-custom-skill", "custom"):
            ...     print("Custom skill is available")
        """
        skill_config = self.skills.get(skill_name)
        if skill_config is None:
            # 公共技能和自定义技能默认启用
            return skill_category in ("public", "custom")
        return skill_config.enabled


_extensions_config: ExtensionsConfig | None = None


def get_extensions_config() -> ExtensionsConfig:
    """获取扩展配置实例。
    
    返回缓存的单例实例。使用 `reload_extensions_config()` 从文件重新加载，
    或使用 `reset_extensions_config()` 清除缓存。
    
    Returns:
        缓存的 ExtensionsConfig 实例
        
    Note:
        - 首次调用时从文件加载配置并缓存
        - 后续调用直接返回缓存实例，提高性能
        - 线程不安全，如需线程安全请使用锁
        - 适用于运行时不需要动态更新配置的场景
        
    Example:
        >>> config = get_extensions_config()
        >>> servers = config.get_enabled_mcp_servers()
    """
    global _extensions_config
    if _extensions_config is None:
        _extensions_config = ExtensionsConfig.from_file()
    return _extensions_config


def reload_extensions_config(config_path: str | None = None) -> ExtensionsConfig:
    """从文件重新加载扩展配置并更新缓存实例。
    
    当配置文件被修改后，可以在不重启应用程序的情况下获取最新配置。
    
    Args:
        config_path: 可选的扩展配置文件路径。如果未提供，使用默认解析策略
        
    Returns:
        新加载的 ExtensionsConfig 实例
        
    Note:
        - 强制从文件重新读取配置，忽略缓存
        - 更新全局单例缓存，后续调用 get_extensions_config() 将返回新配置
        - 适用于配置文件热重载场景
        - 如果文件不存在或格式错误，会抛出异常
        
    Example:
        >>> # 配置文件被修改后
        >>> new_config = reload_extensions_config()
        >>> print(f"Loaded {len(new_config.mcp_servers)} MCP servers")
    """
    global _extensions_config
    _extensions_config = ExtensionsConfig.from_file(config_path)
    return _extensions_config


def reset_extensions_config() -> None:
    """重置缓存的扩展配置实例。
    
    清除单例缓存，使下一次调用 `get_extensions_config()` 时从文件重新加载。
    用于测试或在不同配置之间切换。
    
    Note:
        - 将全局变量设置为 None
        - 下次调用 get_extensions_config() 时会重新从文件加载
        - 主要用于单元测试，避免测试之间的状态污染
        - 生产环境中通常不需要调用此函数
        
    Example:
        >>> # 测试场景
        >>> reset_extensions_config()
        >>> set_extensions_config(mock_config)
        >>> # 运行测试...
        >>> reset_extensions_config()  # 清理
    """
    global _extensions_config
    _extensions_config = None


def set_extensions_config(config: ExtensionsConfig) -> None:
    """设置自定义扩展配置实例。
    
    允许注入自定义或模拟配置用于测试目的。
    
    Args:
        config: 要使用的 ExtensionsConfig 实例
        
    Note:
        - 直接覆盖全局单例缓存
        - 主要用于单元测试，可以注入 mock 配置
        - 也可以用于动态构建配置而不需要从文件加载
        - 谨慎在生产环境中使用，可能导致配置不一致
        
    Example:
        >>> # 测试场景：注入 mock 配置
        >>> mock_config = ExtensionsConfig(
        ...     mcp_servers={"test-server": McpServerConfig(...)},
        ...     skills={}
        ... )
        >>> set_extensions_config(mock_config)
        >>> # 运行测试...
        >>> reset_extensions_config()  # 清理
    """
    global _extensions_config
    _extensions_config = config
