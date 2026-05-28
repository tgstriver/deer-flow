from pydantic import BaseModel, ConfigDict, Field


class ModelConfig(BaseModel):
    """Config section for a model"""

    name: str = Field(..., description="DeerFlow 内部使用的模型别名")
    display_name: str | None = Field(..., default_factory=lambda: None, description="前端展示名")
    description: str | None = Field(..., default_factory=lambda: None, description="模型说明")
    use: str = Field(
        ...,
        description="Class path of the model provider(e.g. langchain_openai.ChatOpenAI)",
    )
    model: str = Field(..., description="provider 实际模型名")
    model_config = ConfigDict(extra="allow")
    use_responses_api: bool | None = Field(
        default=None,
        description="Whether to route OpenAI ChatOpenAI calls through the /v1/responses API",
    )
    output_version: str | None = Field(
        default=None,
        description="Structured output version for OpenAI responses content, e.g. responses/v1",
    )
    supports_thinking: bool = Field(default_factory=lambda: False, description="是否支持 thinking 模式")
    supports_reasoning_effort: bool = Field(default_factory=lambda: False, description="是否支持推理强度。某个模型可以声明支持reasoning effort，于是前端才会展示推理强度选择")
    when_thinking_enabled: dict | None = Field(
        default_factory=lambda: None,
        description="某个模型可以声明when_thinking_enabled，于是thinking模式开启时，工厂会把额外参数合并进provider",
    )
    when_thinking_disabled: dict | None = Field(
        default_factory=lambda: None,
        description="Extra settings to be passed to the model when thinking is disabled",
    )
    supports_vision: bool = Field(default_factory=lambda: False, description="是否支持图像输入。某个模型可以声明支持vision，于是图片查看工具view_image_tool可以进入工具列表")
    thinking: dict | None = Field(
        default_factory=lambda: None,
        description=(
            "Thinking settings for the model. If provided, these settings will be passed to the model when thinking is enabled. "
            "This is a shortcut for `when_thinking_enabled` and will be merged with `when_thinking_enabled` if both are provided."
        ),
    )
