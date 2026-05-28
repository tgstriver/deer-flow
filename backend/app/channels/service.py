"""ChannelService — 管理所有 IM 渠道的生命周期。

本模块负责：
- 从配置文件中读取 IM 渠道配置
- 实例化并启动已启用的渠道
- 管理渠道的启动、停止和重启
- 提供渠道状态查询接口
- 维护全局单例服务实例

支持的渠道类型：
- 钉钉 (DingTalk)
- Discord
- 飞书 (Feishu)
- Slack
- Telegram
- 微信 (WeChat)
- 企业微信 (WeCom)
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from app.channels.base import Channel
from app.channels.manager import DEFAULT_GATEWAY_URL, DEFAULT_LANGGRAPH_URL, ChannelManager
from app.channels.message_bus import MessageBus
from app.channels.store import ChannelStore

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from deerflow.config.app_config import AppConfig

# 渠道名称 → 导入路径映射表，用于延迟加载
_CHANNEL_REGISTRY: dict[str, str] = {
    "dingtalk": "app.channels.dingtalk:DingTalkChannel",
    "discord": "app.channels.discord:DiscordChannel",
    "feishu": "app.channels.feishu:FeishuChannel",
    "slack": "app.channels.slack:SlackChannel",
    "telegram": "app.channels.telegram:TelegramChannel",
    "wechat": "app.channels.wechat:WechatChannel",
    "wecom": "app.channels.wecom:WeComChannel",
}

# 标识用户已为渠道配置凭据的键名列表
_CHANNEL_CREDENTIAL_KEYS: dict[str, list[str]] = {
    "dingtalk": ["client_id", "client_secret"],
    "discord": ["bot_token"],
    "feishu": ["app_id", "app_secret"],
    "slack": ["bot_token", "app_token"],
    "telegram": ["bot_token"],
    "wecom": ["bot_id", "bot_secret"],
    "wechat": ["bot_token"],
}

# 环境变量键名
_CHANNELS_LANGGRAPH_URL_ENV = "DEER_FLOW_CHANNELS_LANGGRAPH_URL"
_CHANNELS_GATEWAY_URL_ENV = "DEER_FLOW_CHANNELS_GATEWAY_URL"


def _resolve_service_url(config: dict[str, Any], config_key: str, env_key: str, default: str) -> str:
    """解析服务 URL，优先级：配置 > 环境变量 > 默认值。
    
    Args:
        config: 配置字典
        config_key: 配置键名
        env_key: 环境变量键名
        default: 默认值
        
    Returns:
        解析后的 URL 字符串
        
    Note:
        - 从配置中弹出值（避免重复使用）
        - 如果配置值为空字符串，则回退到环境变量
        - 最后回退到默认值
    """
    value = config.pop(config_key, None)
    if isinstance(value, str) and value.strip():
        return value
    env_value = os.getenv(env_key, "").strip()
    if env_value:
        return env_value
    return default


class ChannelService:
    """管理所有已配置 IM 渠道的生命周期。
    
    从 ``config.yaml`` 的 ``channels`` 键读取配置，
    实例化已启用的渠道，并启动 ChannelManager 调度器。
    
    Attributes:
        bus: 消息总线，用于渠道间通信
        store: 渠道存储，用于持久化渠道状态
        manager: 渠道管理器，负责消息分发和会话管理
        _channels: 运行中的渠道实例字典（名称 → 实例）
        _config: 原始配置字典
        _running: 服务是否正在运行
    """

    def __init__(self, channels_config: dict[str, Any] | None = None) -> None:
        """初始化 ChannelService。
        
        Args:
            channels_config: 渠道配置字典，如果为 None 则使用空配置
            
        Note:
            - 创建消息总线和渠道存储
            - 解析 LangGraph 和 Gateway URL（支持配置、环境变量、默认值）
            - 初始化 ChannelManager
            - 保存原始配置供后续启动使用
        """
        self.bus = MessageBus()
        self.store = ChannelStore()
        config = dict(channels_config or {})
        langgraph_url = _resolve_service_url(config, "langgraph_url", _CHANNELS_LANGGRAPH_URL_ENV, DEFAULT_LANGGRAPH_URL)
        gateway_url = _resolve_service_url(config, "gateway_url", _CHANNELS_GATEWAY_URL_ENV, DEFAULT_GATEWAY_URL)
        default_session = config.pop("session", None)
        channel_sessions = {name: channel_config.get("session") for name, channel_config in config.items() if isinstance(channel_config, dict)}
        self.manager = ChannelManager(
            bus=self.bus,
            store=self.store,
            langgraph_url=langgraph_url,
            gateway_url=gateway_url,
            default_session=default_session if isinstance(default_session, dict) else None,
            channel_sessions=channel_sessions,
        )
        self._channels: dict[str, Any] = {}  # name -> Channel instance
        self._config = config
        self._running = False

    @classmethod
    def from_app_config(cls, app_config: AppConfig | None = None) -> ChannelService:
        """从应用配置创建 ChannelService。
        
        Args:
            app_config: 应用配置对象，如果为 None 则从全局获取
            
        Returns:
            新创建的 ChannelService 实例
            
        Note:
            - 从 app_config.model_extra 中提取 channels 配置
            - AppConfig 允许额外字段 (extra="allow")
        """
        if app_config is None:
            from deerflow.config.app_config import get_app_config

            app_config = get_app_config()
        channels_config = {}
        # extra fields are allowed by AppConfig (extra="allow")
        extra = app_config.model_extra or {}
        if "channels" in extra:
            channels_config = extra["channels"]
        return cls(channels_config=channels_config)

    async def start(self) -> None:
        """启动管理器和所有已启用的渠道。
        
        遍历配置中的所有渠道，检查是否启用并启动它们。
        如果渠道有凭据但未启用，会发出警告。
        
        Note:
            - 如果服务已在运行，则直接返回
            - 先启动 ChannelManager
            - 然后逐个启动已配置的渠道
            - 记录所有启动的渠道名称
        """
        if self._running:
            return

        await self.manager.start()

        for name, channel_config in self._config.items():
            if not isinstance(channel_config, dict):
                continue
            if not channel_config.get("enabled", False):
                # 检查是否有凭据配置但未启用
                cred_keys = _CHANNEL_CREDENTIAL_KEYS.get(name, [])
                has_creds = any(not isinstance(channel_config.get(k), bool) and channel_config.get(k) is not None and str(channel_config[k]).strip() for k in cred_keys)
                if has_creds:
                    logger.warning(
                        "Channel '%s' has credentials configured but is disabled. Set enabled: true under channels.%s in config.yaml to activate it.",
                        name,
                        name,
                    )
                else:
                    logger.info("Channel %s is disabled, skipping", name)
                continue

            await self._start_channel(name, channel_config)

        self._running = True
        logger.info("ChannelService started with channels: %s", list(self._channels.keys()))

    async def stop(self) -> None:
        """停止所有渠道和管理器。
        
        按顺序停止所有运行中的渠道，然后停止管理器。
        即使某个渠道停止失败，也会继续停止其他渠道。
        
        Note:
            - 遍历所有渠道并调用 stop()
            - 捕获异常以确保所有渠道都被尝试停止
            - 清空渠道字典
            - 停止 ChannelManager
            - 更新运行状态
        """
        for name, channel in list(self._channels.items()):
            try:
                await channel.stop()
                logger.info("Channel %s stopped", name)
            except Exception:
                logger.exception("Error stopping channel %s", name)
        self._channels.clear()

        await self.manager.stop()
        self._running = False
        logger.info("ChannelService stopped")

    async def restart_channel(self, name: str) -> bool:
        """重启指定的渠道。返回是否成功。
        
        Args:
            name: 渠道名称
            
        Returns:
            True 如果重启成功，False 否则
            
        Note:
            - 先停止现有渠道实例（如果有）
            - 从配置中重新加载并启动
            - 如果配置不存在则返回 False
        """
        if name in self._channels:
            try:
                await self._channels[name].stop()
            except Exception:
                logger.exception("Error stopping channel %s for restart", name)
            del self._channels[name]

        config = self._config.get(name)
        if not config or not isinstance(config, dict):
            logger.warning("No config for channel %s", name)
            return False

        return await self._start_channel(name, config)

    async def _start_channel(self, name: str, config: dict[str, Any]) -> bool:
        """实例化并启动单个渠道。
        
        Args:
            name: 渠道名称
            config: 渠道配置字典
            
        Returns:
            True 如果启动成功，False 否则
            
        Note:
            - 从注册表中查找渠道类的导入路径
            - 使用 deerflow.reflection.resolve_class 动态加载类
            - 注入 channel_store 到配置中
            - 调用渠道的 start() 方法
            - 验证渠道是否进入运行状态
            - 失败时从字典中移除渠道
        """
        import_path = _CHANNEL_REGISTRY.get(name)
        if not import_path:
            logger.warning("Unknown channel type: %s", name)
            return False

        try:
            from deerflow.reflection import resolve_class

            channel_cls = resolve_class(import_path, base_class=None)
        except Exception:
            logger.exception("Failed to import channel class for %s", name)
            return False

        try:
            config = dict(config)
            config["channel_store"] = self.store
            channel = channel_cls(bus=self.bus, config=config)
            self._channels[name] = channel
            await channel.start()
            if not channel.is_running:
                self._channels.pop(name, None)
                logger.error("Channel %s did not enter a running state after start()", name)
                return False
            logger.info("Channel %s started", name)
            return True
        except Exception:
            self._channels.pop(name, None)
            logger.exception("Failed to start channel %s", name)
            return False

    def get_status(self) -> dict[str, Any]:
        """返回所有渠道的状态信息。
        
        Returns:
            包含服务运行状态和所有渠道状态的字典
            
        Note:
            - 遍历注册表中的所有渠道类型
            - 检查每个渠道的启用状态和运行状态
            - 即使未配置的渠道也会返回（enabled=False, running=False）
        """
        channels_status = {}
        for name in _CHANNEL_REGISTRY:
            config = self._config.get(name, {})
            enabled = isinstance(config, dict) and config.get("enabled", False)
            running = name in self._channels and self._channels[name].is_running
            channels_status[name] = {
                "enabled": enabled,
                "running": running,
            }
        return {
            "service_running": self._running,
            "channels": channels_status,
        }

    def get_channel(self, name: str) -> Channel | None:
        """按名称返回运行中的渠道实例（如果可用）。
        
        Args:
            name: 渠道名称
            
        Returns:
            渠道实例或 None（如果不存在或未运行）
        """
        return self._channels.get(name)


# -- singleton access -------------------------------------------------------

# 全局单例实例
_channel_service: ChannelService | None = None


def get_channel_service() -> ChannelService | None:
    """获取全局 ChannelService 单例实例（如果已启动）。
    
    Returns:
        ChannelService 实例或 None（如果未启动）
        
    Note:
        - 只读访问，不会创建新实例
        - 在 start_channel_service 之前调用会返回 None
    """
    return _channel_service


async def start_channel_service(app_config: AppConfig | None = None) -> ChannelService:
    """从应用配置创建并启动全局 ChannelService。
    
    Args:
        app_config: 应用配置对象，如果为 None 则从全局获取
        
    Returns:
        启动后的 ChannelService 实例
        
    Note:
        - 如果服务已存在，直接返回现有实例
        - 使用 from_app_config 创建新实例
        - 调用 start() 启动服务和所有渠道
        - 设置为全局单例
    """
    global _channel_service
    if _channel_service is not None:
        return _channel_service
    _channel_service = ChannelService.from_app_config(app_config)
    await _channel_service.start()
    return _channel_service


async def stop_channel_service() -> None:
    """停止全局 ChannelService。
    
    Note:
        - 如果服务存在，调用 stop() 停止
        - 清空全局引用
        - 再次调用 start_channel_service 会创建新实例
    """
    global _channel_service
    if _channel_service is not None:
        await _channel_service.stop()
        _channel_service = None
