"""模型工厂模块 —— 根据配置动态创建 LLM 聊天模型实例。

本模块是 DeerFlow 模型系统的核心入口，通过反射机制从 config.yaml 中指定的
模型类路径（如 ``langchain_openai:ChatOpenAI``）动态加载并实例化聊天模型。

核心功能：
  - 根据 thinking_enabled 标志动态切换思考/推理模式配置
  - 深度合并配置字典，支持嵌套覆盖而不修改原始数据
  - 为 OpenAI 兼容网关默认启用 stream_usage，确保令牌用量追踪可用
  - 对 Codex Responses API 模型自动映射思考模式到 reasoning_effort
  - 对 vLLM/Qwen 思考模型自动生成禁用参数（chat_template_kwargs）
"""

import logging

from langchain.chat_models import BaseChatModel

from deerflow.config import get_app_config
from deerflow.config.app_config import AppConfig
from deerflow.reflection import resolve_class
from deerflow.tracing import build_tracing_callbacks

logger = logging.getLogger(__name__)


def _deep_merge_dicts(base: dict | None, override: dict) -> dict:
    """递归合并两个字典，不修改原始输入。

    当 override 中的值也是字典且 base 中对应键也是字典时，递归合并两者；
    否则 override 的值直接覆盖 base 的值。

    Args:
        base: 基础字典（可为 None，视为空字典）
        override: 覆盖字典，其值优先级更高

    Returns:
        dict: 合并后的新字典（原始输入不会被修改）
    """
    merged = dict(base or {})
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            # 两个值均为字典时递归合并
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            # 非字典值直接覆盖
            merged[key] = value
    return merged


def _vllm_disable_chat_template_kwargs(chat_template_kwargs: dict) -> dict:
    """构建用于禁用 vLLM/Qwen 思考模式的 chat_template_kwargs 载荷。

    vLLM 的 Qwen 推理模型通过 chat_template_kwargs 中的 thinking 或
    enable_thinking 字段控制思考模式开关。本函数根据输入中存在的字段
    生成对应的禁用参数（值为 False）。

    Args:
        chat_template_kwargs: 当前启用的聊天模板参数字典

    Returns:
        dict: 包含禁用标志的参数字典（仅包含输入中已有的字段）
    """
    disable_kwargs: dict[str, bool] = {}
    if "thinking" in chat_template_kwargs:
        disable_kwargs["thinking"] = False
    if "enable_thinking" in chat_template_kwargs:
        disable_kwargs["enable_thinking"] = False
    return disable_kwargs


def _enable_stream_usage_by_default(model_use_path: str, model_settings_from_config: dict) -> None:
    """为 OpenAI 兼容模型默认启用 stream_usage，确保令牌用量追踪可用。

    LangChain 仅在使用标准 OpenAI 端点（无自定义 base_url）时自动启用
    stream_usage。DeerFlow 经常使用 OpenAI 兼容网关（如豆包、DeepSeek），
    若不显式启用 stream_usage，令牌用量追踪将为空，TokenUsageMiddleware
    将无法记录用量数据。

    仅在以下条件全部满足时生效：
      - 模型类为 langchain_openai:ChatOpenAI
      - 用户未显式配置 stream_usage
      - 配置中包含 base_url 或 openai_api_base（说明使用了兼容网关）

    Args:
        model_use_path: 模型类的完整路径（如 ``langchain_openai:ChatOpenAI``）
        model_settings_from_config: 从配置中提取的模型参数字典（将被就地修改）
    """
    if model_use_path != "langchain_openai:ChatOpenAI":
        return
    if "stream_usage" in model_settings_from_config:
        return
    # 仅在配置了自定义端点时才强制启用，避免标准 OpenAI 模型重复设置
    if "base_url" in model_settings_from_config or "openai_api_base" in model_settings_from_config:
        model_settings_from_config["stream_usage"] = True


def create_chat_model(name: str | None = None, thinking_enabled: bool = False, *, app_config: AppConfig | None = None, **kwargs) -> BaseChatModel:
    """根据配置创建聊天模型实例，支持思考模式切换和令牌用量追踪。

    从 config.yaml 中查找指定名称的模型配置，通过反射加载模型类，
    并根据 thinking_enabled 标志动态调整模型参数（如思考开关、推理力度等）。

    Args:
        name: 模型名称。若为 None，则使用配置中的第一个模型。
        thinking_enabled: 是否启用思考/推理模式。为 True 时应用
            when_thinking_enabled 配置；为 False 时应用 when_thinking_disabled
            配置或自动生成禁用参数。
        app_config: 应用配置实例。若为 None，则通过 get_app_config() 加载。
        **kwargs: 传递给模型构造函数的额外参数（如 reasoning_effort）。

    Returns:
        BaseChatModel: 配置完成的聊天模型实例，附带追踪回调。

    Raises:
        ValueError: 模型名称不存在于配置中，或思考模式不被模型支持时抛出。
    """
    # 加载应用配置（优先使用传入的实例，否则从文件加载）
    config = app_config or get_app_config()
    # 若未指定模型名称，默认使用配置中的第一个模型
    if name is None:
        name = config.models[0].name
    # 获取指定名称的模型配置
    model_config = config.get_model_config(name)
    if model_config is None:
        raise ValueError(f"Model {name} not found in config") from None
    # 通过反射加载模型类，确保其继承自 BaseChatModel
    model_class = resolve_class(model_config.use, BaseChatModel)
    # 序列化模型配置为字典，排除元信息字段
    model_settings_from_config = model_config.model_dump(
        exclude_none=True,
        exclude={
            "use",  # 类路径（已通过 resolve_class 使用）
            "name",  # 模型名称（已通过参数传入）
            "display_name",  # 显示名称（前端用，不影响模型构造）
            "description",  # 模型描述（前端用）
            "supports_thinking",  # 思考能力标记（仅用于配置判断）
            "supports_reasoning_effort",  # 推理力度标记（仅用于配置判断）
            "when_thinking_enabled",  # 思考启用时的参数覆盖（单独处理）
            "when_thinking_disabled",  # 思考禁用时的参数覆盖（单独处理）
            "thinking",  # 思考快捷配置（单独处理）
            "supports_vision",  # 视觉能力标记（仅用于配置判断）
        },
    )
    # 计算有效的思考启用参数：合并 when_thinking_enabled 和 thinking 快捷字段
    # thinking 快捷字段等同于设置 when_thinking_enabled["thinking"]
    has_thinking_settings = (model_config.when_thinking_enabled is not None) or (model_config.thinking is not None)
    # 以 when_thinking_enabled 为基础，合并 thinking 字段
    effective_wte: dict = dict(model_config.when_thinking_enabled) if model_config.when_thinking_enabled else {}
    if model_config.thinking is not None:
        # 深度合并 thinking 字段到 effective_wte 中已有的 thinking 子字典
        merged_thinking = {**(effective_wte.get("thinking") or {}), **model_config.thinking}
        effective_wte = {**effective_wte, "thinking": merged_thinking}

    # ===== 思考模式启用时的参数处理 =====
    if thinking_enabled and has_thinking_settings:
        # 模型不支持思考模式时抛出错误
        if not model_config.supports_thinking:
            raise ValueError(f"Model {name} does not support thinking. Set `supports_thinking` to true in the `config.yaml` to enable thinking.") from None
        # 将有效的思考启用参数合并到模型设置中
        if effective_wte:
            model_settings_from_config.update(effective_wte)

    # ===== 思考模式禁用时的参数处理 =====
    if not thinking_enabled:
        # 优先使用用户显式提供的禁用配置
        if model_config.when_thinking_disabled is not None:
            model_settings_from_config.update(model_config.when_thinking_disabled)
        # OpenAI 兼容网关：thinking 嵌套在 extra_body 中
        elif has_thinking_settings and effective_wte.get("extra_body", {}).get("thinking", {}).get("type"):
            model_settings_from_config["extra_body"] = _deep_merge_dicts(
                model_settings_from_config.get("extra_body"),
                {"thinking": {"type": "disabled"}},
            )
            # 禁用思考时将推理力度设置为最低
            model_settings_from_config["reasoning_effort"] = "minimal"
        # vLLM 使用 chat_template_kwargs 切换思考开关
        elif has_thinking_settings and (disable_chat_template_kwargs := _vllm_disable_chat_template_kwargs(effective_wte.get("extra_body", {}).get("chat_template_kwargs") or {})):
            model_settings_from_config["extra_body"] = _deep_merge_dicts(
                model_settings_from_config.get("extra_body"),
                {"chat_template_kwargs": disable_chat_template_kwargs},
            )
        # 原生 langchain_anthropic：thinking 是直接构造参数
        elif has_thinking_settings and effective_wte.get("thinking", {}).get("type"):
            model_settings_from_config["thinking"] = {"type": "disabled"}

    # 若模型不支持 reasoning_effort，移除该参数
    if not model_config.supports_reasoning_effort:
        kwargs.pop("reasoning_effort", None)
        model_settings_from_config.pop("reasoning_effort", None)

    # 为 OpenAI 兼容网关默认启用 stream_usage（确保令牌追踪可用）
    _enable_stream_usage_by_default(model_config.use, model_settings_from_config)

    # ===== Codex Responses API 模型特殊处理 =====
    # 将思考模式映射到 reasoning_effort 参数
    from deerflow.models.openai_codex_provider import CodexChatModel

    if issubclass(model_class, CodexChatModel):
        # ChatGPT Codex 端点拒绝 max_tokens/max_output_tokens 参数，需移除
        model_settings_from_config.pop("max_tokens", None)

        # 推理力度优先级：前端显式传入 > 配置文件 > 默认值 medium
        explicit_effort = kwargs.pop("reasoning_effort", None)
        if not thinking_enabled:
            # 禁用思考模式时，推理力度设为 none
            model_settings_from_config["reasoning_effort"] = "none"
        elif explicit_effort and explicit_effort in ("low", "medium", "high", "xhigh"):
            # 使用前端显式传入的推理力度
            model_settings_from_config["reasoning_effort"] = explicit_effort
        elif "reasoning_effort" not in model_settings_from_config:
            # 默认推理力度为 medium
            model_settings_from_config["reasoning_effort"] = "medium"

    # ===== MindIE 模型特殊处理 =====
    # 强制保守的重试默认值，防止超时级联。超时标准化由 MindIEChatModel 内部处理。
    if getattr(model_class, "__name__", "") == "MindIEChatModel":
        # 强制 max_retries 约束，防止超时级联
        model_settings_from_config["max_retries"] = model_settings_from_config.get("max_retries", 1)

    # ===== stream_usage 默认启用 =====
    # 确保流式响应中包含令牌用量元数据。
    # LangChain 的 BaseChatOpenAI 仅在无自定义 base_url/api_base 时默认启用
    # stream_usage=True，因此使用第三方端点（如豆包、DeepSeek）时会丢失用量数据。
    # 我们默认启用 stream_usage，除非用户显式配置了其他值。
    if "stream_usage" not in model_settings_from_config and "stream_usage" not in kwargs:
        # 仅当模型类支持 stream_usage 字段时才设置
        if "stream_usage" in getattr(model_class, "model_fields", {}):
            model_settings_from_config["stream_usage"] = True

    # 实例化模型，合并 kwargs 和配置参数
    model_instance = model_class(**kwargs, **model_settings_from_config)

    # 附加追踪回调（如 LangSmith 等可观测性提供商）
    callbacks = build_tracing_callbacks()
    if callbacks:
        existing_callbacks = model_instance.callbacks or []
        model_instance.callbacks = [*existing_callbacks, *callbacks]
        logger.debug(f"Tracing attached to model '{name}' with providers={len(callbacks)}")
    return model_instance
