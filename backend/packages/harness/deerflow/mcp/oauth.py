"""MCP HTTP/SSE 服务器的 OAuth 令牌支持。

本模块提供：
- OAuth 令牌的获取、缓存和刷新功能
- 支持多种授权类型（client_credentials、refresh_token）
- 线程安全的并发访问控制
- 自动过期检测和提前刷新机制
- MCP 工具拦截器，自动注入 Authorization 头

核心功能：
    - OAuthTokenManager: 管理 OAuth 令牌的生命周期
    - build_oauth_tool_interceptor: 构建工具拦截器，自动注入认证头
    - get_initial_oauth_headers: 获取初始 OAuth 认证头用于 MCP 服务器连接

支持的授权类型：
    - client_credentials: 客户端凭证模式
    - refresh_token: 刷新令牌模式
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from deerflow.config.extensions_config import ExtensionsConfig, McpOAuthConfig

logger = logging.getLogger(__name__)


@dataclass
class _OAuthToken:
    """缓存的 OAuth 令牌数据。
    
    封装从 OAuth 服务器获取的令牌信息，包括访问令牌、类型和过期时间。
    
    Attributes:
        access_token: 访问令牌字符串，用于 API 认证
        token_type: 令牌类型（如 'Bearer'、'bearer'）
        expires_at: 令牌过期时间（UTC），用于判断是否需要刷新
        
    Note:
        - 这是内部使用的数据类，不对外暴露
        - expires_at 使用 UTC 时区，避免时区问题
    """

    access_token: str
    token_type: str
    expires_at: datetime


class OAuthTokenManager:
    """为 MCP 服务器获取、缓存和刷新 OAuth 令牌。
    
    管理多个 MCP 服务器的 OAuth 认证，提供自动刷新和缓存功能。
    
    主要特性：
        - 按服务器名称隔离令牌存储
        - 异步锁保证并发安全
        - 智能过期检测（考虑刷新偏移量）
        - 支持两种 OAuth 授权流程
        
    工作流程：
        1. 检查缓存中是否有有效令牌
        2. 如果令牌即将过期，使用锁防止重复刷新
        3. 调用 OAuth 端点获取新令牌
        4. 更新缓存并返回 Authorization 头
        
    Attributes:
        _oauth_by_server: 服务器名称到 OAuth 配置的映射
        _tokens: 服务器名称到缓存令牌的映射
        _locks: 服务器名称到异步锁的映射，防止并发刷新
        
    Example:
        >>> from deerflow.config.extensions_config import ExtensionsConfig
        >>> config = ExtensionsConfig.load()
        >>> manager = OAuthTokenManager.from_extensions_config(config)
        >>> header = await manager.get_authorization_header("my-mcp-server")
    """

    def __init__(self, oauth_by_server: dict[str, McpOAuthConfig]):
        """初始化 OAuth 令牌管理器。
        
        Args:
            oauth_by_server: 服务器名称到 OAuth 配置的映射字典
            
        Note:
            - 为每个启用了 OAuth 的服务器创建独立的异步锁
            - 初始时令牌缓存为空，按需获取
        """
        self._oauth_by_server = oauth_by_server
        self._tokens: dict[str, _OAuthToken] = {}
        self._locks: dict[str, asyncio.Lock] = {name: asyncio.Lock() for name in oauth_by_server}

    @classmethod
    def from_extensions_config(cls, extensions_config: ExtensionsConfig) -> OAuthTokenManager:
        """从扩展配置创建 OAuth 令牌管理器。
        
        遍历所有启用的 MCP 服务器配置，提取启用了 OAuth 的服务器。
        
        Args:
            extensions_config: 扩展配置对象，包含所有 MCP 服务器配置
            
        Returns:
            配置好的 OAuthTokenManager 实例
            
        Note:
            - 只处理同时满足两个条件的服务器：
              1. 服务器已启用（enabled=True）
              2. OAuth 配置存在且已启用（oauth.enabled=True）
            - 未启用 OAuth 的服务器会被忽略
        """
        oauth_by_server: dict[str, McpOAuthConfig] = {}
        for server_name, server_config in extensions_config.get_enabled_mcp_servers().items():
            if server_config.oauth and server_config.oauth.enabled:
                oauth_by_server[server_name] = server_config.oauth
        return cls(oauth_by_server)

    def has_oauth_servers(self) -> bool:
        """检查是否存在需要 OAuth 认证的服务器。
        
        Returns:
            True 如果有至少一个启用了 OAuth 的服务器，否则 False
            
        Note:
            - 用于快速判断是否需要初始化 OAuth 相关组件
        """
        return bool(self._oauth_by_server)

    def oauth_server_names(self) -> list[str]:
        """获取所有需要 OAuth 认证的服务器名称列表。
        
        Returns:
            服务器名称列表，顺序与配置中的顺序一致
            
        Note:
            - 返回的是字典键的列表副本
            - 可用于遍历所有 OAuth 服务器
        """
        return list(self._oauth_by_server.keys())

    async def get_authorization_header(self, server_name: str) -> str | None:
        """获取指定服务器的 Authorization 请求头。
        
        这是主要的公共接口，负责：
        1. 检查缓存中的令牌是否有效
        2. 如果即将过期，自动刷新令牌
        3. 使用异步锁防止并发刷新
        4. 返回格式化的 Authorization 头
        
        Args:
            server_name: MCP 服务器名称，必须在配置中存在
            
        Returns:
            Authorization 头字符串（如 'Bearer xxx'），如果服务器未配置 OAuth 则返回 None
            
        Note:
            - 使用双重检查锁定模式（Double-Check Locking）
            - 第一次检查无锁，快速返回有效令牌
            - 第二次检查在锁内，防止重复刷新
            - 令牌将在过期前 refresh_skew_seconds 秒开始刷新
            
        Example:
            >>> header = await manager.get_authorization_header("my-server")
            >>> # header = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
        """
        oauth = self._oauth_by_server.get(server_name)
        if not oauth:
            return None

        # 第一次检查：无锁快速路径
        token = self._tokens.get(server_name)
        if token and not self._is_expiring(token, oauth):
            return f"{token.token_type} {token.access_token}"

        # 第二次检查：加锁防止并发刷新
        lock = self._locks[server_name]
        async with lock:
            # 在锁内再次检查，可能其他协程已经刷新了令牌
            token = self._tokens.get(server_name)
            if token and not self._is_expiring(token, oauth):
                return f"{token.token_type} {token.access_token}"

            # 获取新令牌
            fresh = await self._fetch_token(oauth)
            self._tokens[server_name] = fresh
            logger.info(f"Refreshed OAuth access token for MCP server: {server_name}")
            return f"{fresh.token_type} {fresh.access_token}"

    @staticmethod
    def _is_expiring(token: _OAuthToken, oauth: McpOAuthConfig) -> bool:
        """检查令牌是否即将过期。
        
        判断令牌是否在刷新偏移时间内到期，如果是则认为需要刷新。
        
        Args:
            token: 当前缓存的令牌
            oauth: OAuth 配置，包含 refresh_skew_seconds
            
        Returns:
            True 如果令牌即将过期或已过期，需要刷新；False 如果仍然有效
            
        Note:
            - 使用 UTC 时间进行比较，避免时区问题
            - refresh_skew_seconds 允许提前刷新，避免在请求过程中过期
            - 默认偏移量为 0，但建议设置为 30-60 秒以提高可靠性
            
        Example:
            >>> # 如果令牌在 60 秒内过期，且 refresh_skew_seconds=30
            >>> # 则从现在开始 30 秒后就会认为即将过期
        """
        now = datetime.now(UTC)
        return token.expires_at <= now + timedelta(seconds=max(oauth.refresh_skew_seconds, 0))

    async def _fetch_token(self, oauth: McpOAuthConfig) -> _OAuthToken:
        """从 OAuth 服务器获取新的访问令牌。
        
        根据配置的授权类型（grant_type）发送相应的请求，解析响应并创建令牌对象。
        
        Args:
            oauth: OAuth 配置对象，包含端点 URL、凭证、参数等
            
        Returns:
            新的 _OAuthToken 对象，包含访问令牌、类型和过期时间
            
        Raises:
            ValueError: 如果缺少必需的凭证或收到无效的响应
            httpx.HTTPStatusError: 如果 HTTP 请求失败
            
        Note:
            - 支持两种授权类型：
              1. client_credentials: 客户端凭证模式，需要 client_id 和 client_secret
              2. refresh_token: 刷新令牌模式，需要 refresh_token
            - 请求超时设置为 15 秒
            - 令牌字段名可自定义（通过 token_field、token_type_field、expires_in_field）
            - 如果 expires_in 无效，默认使用 3600 秒（1 小时）
            
        Example:
            请求示例（client_credentials）：
                POST /oauth/token
                Content-Type: application/x-www-form-urlencoded
                
                grant_type=client_credentials&
                client_id=xxx&
                client_secret=yyy&
                scope=read write
                
            响应示例：
                {
                    "access_token": "eyJhbGc...",
                    "token_type": "Bearer",
                    "expires_in": 3600
                }
        """
        import httpx  # pyright: ignore[reportMissingImports]

        # 构建请求数据，包含 grant_type 和额外参数
        data: dict[str, str] = {
            "grant_type": oauth.grant_type,
            **oauth.extra_token_params,
        }

        # 可选参数：scope 和 audience
        if oauth.scope:
            data["scope"] = oauth.scope
        if oauth.audience:
            data["audience"] = oauth.audience

        # 根据授权类型添加相应的凭证
        if oauth.grant_type == "client_credentials":
            # 客户端凭证模式：必须提供 client_id 和 client_secret
            if not oauth.client_id or not oauth.client_secret:
                raise ValueError("OAuth client_credentials requires client_id and client_secret")
            data["client_id"] = oauth.client_id
            data["client_secret"] = oauth.client_secret
        elif oauth.grant_type == "refresh_token":
            # 刷新令牌模式：必须提供 refresh_token
            if not oauth.refresh_token:
                raise ValueError("OAuth refresh_token grant requires refresh_token")
            data["refresh_token"] = oauth.refresh_token
            if oauth.client_id:
                data["client_id"] = oauth.client_id
            if oauth.client_secret:
                data["client_secret"] = oauth.client_secret
        else:
            raise ValueError(f"Unsupported OAuth grant type: {oauth.grant_type}")

        # 发送 POST 请求获取令牌
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(oauth.token_url, data=data)
            response.raise_for_status()
            payload = response.json()

        # 解析响应，使用可配置的字段名
        access_token = payload.get(oauth.token_field)
        if not access_token:
            raise ValueError(f"OAuth token response missing '{oauth.token_field}'")

        token_type = str(payload.get(oauth.token_type_field, oauth.default_token_type) or oauth.default_token_type)

        # 解析过期时间，默认为 3600 秒
        expires_in_raw = payload.get(oauth.expires_in_field, 3600)
        try:
            expires_in = int(expires_in_raw)
        except (TypeError, ValueError):
            expires_in = 3600

        # 计算绝对过期时间（UTC）
        expires_at = datetime.now(UTC) + timedelta(seconds=max(expires_in, 1))
        return _OAuthToken(access_token=access_token, token_type=token_type, expires_at=expires_at)


def build_oauth_tool_interceptor(extensions_config: ExtensionsConfig) -> Any | None:
    """构建工具拦截器，自动注入 OAuth Authorization 头。
    
    创建一个异步拦截器函数，用于 MCP 工具调用时自动添加认证头。
    
    Args:
        extensions_config: 扩展配置对象，用于提取 OAuth 配置
        
    Returns:
        异步拦截器函数，如果没有启用 OAuth 的服务器则返回 None
        
    Note:
        - 拦截器签名：async def interceptor(request, handler) -> response
        - 从 request.server_name 获取目标服务器名称
        - 使用 request.override(headers=...) 添加 Authorization 头
        - 如果服务器不需要 OAuth，直接透传请求
        
    Example:
        >>> interceptor = build_oauth_tool_interceptor(config)
        >>> if interceptor:
        ...     mcp_client = Client(interceptors=[interceptor])
    """
    token_manager = OAuthTokenManager.from_extensions_config(extensions_config)
    if not token_manager.has_oauth_servers():
        return None

    async def oauth_interceptor(request: Any, handler: Any) -> Any:
        """OAuth 拦截器，为请求添加 Authorization 头。
        
        Args:
            request: MCP 请求对象，包含 server_name 和 headers
            handler: 下一个处理器或中间件
            
        Returns:
            处理后的响应对象
        """
        header = await token_manager.get_authorization_header(request.server_name)
        if not header:
            return await handler(request)

        updated_headers = dict(request.headers or {})
        updated_headers["Authorization"] = header
        return await handler(request.override(headers=updated_headers))

    return oauth_interceptor


async def get_initial_oauth_headers(extensions_config: ExtensionsConfig) -> dict[str, str]:
    """获取 MCP 服务器连接的初始 OAuth Authorization 头。
    
    为所有启用 OAuth 的服务器预取令牌，返回服务器名称到 Authorization 头的映射。
    
    Args:
        extensions_config: 扩展配置对象，用于提取 OAuth 配置
        
    Returns:
        字典，键为服务器名称，值为 Authorization 头字符串
        如果没有启用 OAuth 的服务器，返回空字典
        
    Note:
        - 在建立 MCP 连接时调用，预先获取所有需要的令牌
        - 过滤掉空值，只返回成功获取令牌的服务器
        - 可用于批量初始化多个服务器的认证
        
    Example:
        >>> headers = await get_initial_oauth_headers(config)
        >>> # headers = {
        >>> #     "server-a": "Bearer token1...",
        >>> #     "server-b": "Bearer token2..."
        >>> # }
    """
    token_manager = OAuthTokenManager.from_extensions_config(extensions_config)
    if not token_manager.has_oauth_servers():
        return {}

    headers: dict[str, str] = {}
    for server_name in token_manager.oauth_server_names():
        headers[server_name] = await token_manager.get_authorization_header(server_name) or ""

    # 过滤掉空值，只保留有效的 Authorization 头
    return {name: value for name, value in headers.items() if value}
