"""视图图像中间件 —— 在 LLM 调用前将已查看的图像详情注入对话。

当代理通过 view_image 工具加载了图像后，该中间件负责：
1. 在每次 LLM 调用前检测上一轮对话是否包含 view_image 工具调用
2. 确认这些工具调用均已完成（即存在对应的 ToolMessage）
3. 如果满足条件，将已查看的图像（包含 base64 数据）以 HumanMessage 的形式注入状态
4. 使得 LLM 无需用户显式提示即可"看到"并分析图像

该中间件是中间件链中的第 14 个组件，仅在模型支持视觉能力时生效。
"""

import logging
from typing import override

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.runtime import Runtime

from deerflow.agents.thread_state import ThreadState

logger = logging.getLogger(__name__)


class ViewImageMiddlewareState(ThreadState):
    """视图图像中间件的状态模式。

    复用 ThreadState，使由 reducer 支持的键保留其注解定义，
    例如 viewed_images 使用 merge_viewed_images 归约器来合并或清除图像数据。
    """


class ViewImageMiddleware(AgentMiddleware[ViewImageMiddlewareState]):
    """视图图像中间件 —— 在 LLM 调用前注入图像详情。

    工作流程：
    1. 每次模型调用前执行
    2. 检查最后一条助手消息是否包含 view_image 工具调用
    3. 验证该消息中的所有工具调用是否已完成（是否均有对应的 ToolMessage）
    4. 若条件满足，创建一条包含所有已查看图像详情（含 base64 数据）的 HumanMessage
    5. 将该消息添加到状态中，使 LLM 可以查看并分析图像

    这使得 LLM 能够自动接收并分析通过 view_image 工具加载的图像，
    而无需用户显式请求描述图像内容。
    """

    state_schema = ViewImageMiddlewareState

    def _get_last_assistant_message(self, messages: list) -> AIMessage | None:
        """从消息列表中获取最后一条助手消息。

        从消息列表末尾向前遍历，返回第一条类型为 AIMessage 的消息，
        因为最后一条助手消息才是当前轮次需要检测的目标。

        Args:
            messages: 消息列表

        Returns:
            最后一条 AIMessage，若未找到则返回 None
        """
        # 反向遍历，效率更高：最近的助手消息通常在列表尾部
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                return msg
        return None

    def _has_view_image_tool(self, message: AIMessage) -> bool:
        """检查助手消息是否包含 view_image 工具调用。

        这是注入条件的第一步检测：只有当助手调用了 view_image 工具时，
        才有可能需要注入图像详情消息。

        Args:
            message: 待检查的助手消息

        Returns:
            若消息包含 view_image 工具调用则返回 True
        """
        # 防御性检查：消息可能没有 tool_calls 属性，或属性为空
        if not hasattr(message, "tool_calls") or not message.tool_calls:
            return False

        # 遍历所有工具调用，检查是否存在名称为 "view_image" 的调用
        return any(tool_call.get("name") == "view_image" for tool_call in message.tool_calls)

    def _all_tools_completed(self, messages: list, assistant_msg: AIMessage) -> bool:
        """检查助手消息中的所有工具调用是否已完成。

        这是注入条件的第二步检测：只有当 view_image 及同轮次的其它工具调用
        都已返回结果（即存在对应的 ToolMessage）时，才能安全地注入图像消息，
        否则可能会在工具尚未返回结果时就注入，导致信息不完整。

        Args:
            messages: 所有消息的列表
            assistant_msg: 包含工具调用的助手消息

        Returns:
            若所有工具调用均有对应的 ToolMessage 则返回 True
        """
        # 防御性检查：消息可能没有 tool_calls 属性，或属性为空
        if not hasattr(assistant_msg, "tool_calls") or not assistant_msg.tool_calls:
            return False

        # 收集助手消息中所有工具调用的 ID，用于后续匹配对应的 ToolMessage
        tool_call_ids = {tool_call.get("id") for tool_call in assistant_msg.tool_calls if tool_call.get("id")}

        # 定位助手消息在消息列表中的位置，以便只检查其后的 ToolMessage
        try:
            assistant_idx = messages.index(assistant_msg)
        except ValueError:
            # 消息不在列表中（不应发生，防御性处理）
            return False

        # 遍历助手消息之后的所有消息，收集已完成工具调用的 ID
        completed_tool_ids = set()
        for msg in messages[assistant_idx + 1 :]:
            if isinstance(msg, ToolMessage) and msg.tool_call_id:
                completed_tool_ids.add(msg.tool_call_id)

        # 判断助手消息中的所有工具调用 ID 是否都被已完成的工具 ID 集合包含
        return tool_call_ids.issubset(completed_tool_ids)

    def _create_image_details_message(self, state: ViewImageMiddlewareState) -> list[str | dict]:
        """创建包含所有已查看图像详情的格式化消息。

        从状态中的 viewed_images 字典提取图像路径、MIME 类型和 base64 数据，
        构建包含文本描述和图像 URL 内容块的消息列表。
        图像 URL 使用 data URI 方案将 base64 数据内联，供多模态 LLM 直接消费。

        Args:
            state: 当前状态，包含 viewed_images 图像数据

        Returns:
            内容块列表（文本块和图像块），用于构建 HumanMessage
        """
        # 从状态中获取已查看的图像字典：{图像路径: {mime_type, base64}}
        viewed_images = state.get("viewed_images", {})
        if not viewed_images:
            # 没有已查看的图像时，返回格式化的文本提示块（而非纯字符串数组）
            return [{"type": "text", "text": "No images have been viewed."}]

        # 构建消息内容：开头是引导文本
        content_blocks: list[str | dict] = [{"type": "text", "text": "Here are the images you've viewed:"}]

        for image_path, image_data in viewed_images.items():
            mime_type = image_data.get("mime_type", "unknown")
            base64_data = image_data.get("base64", "")

            # 添加文本描述块，标注图像路径和 MIME 类型
            content_blocks.append({"type": "text", "text": f"\n- **{image_path}** ({mime_type})"})

            # 添加实际的图像数据块，使用 data URI 方案将 base64 数据内联
            # 这样多模态 LLM 可以直接"看到"图像内容
            if base64_data:
                content_blocks.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{base64_data}"},
                    }
                )

        return content_blocks

    def _should_inject_image_message(self, state: ViewImageMiddlewareState) -> bool:
        """判断是否应该注入图像详情消息。

        注入必须同时满足以下条件：
        1. 消息列表非空
        2. 存在最后一条助手消息
        3. 该助手消息包含 view_image 工具调用
        4. 该助手消息中所有工具调用均已完成
        5. 尚未在助手消息之后注入过图像详情消息（避免重复注入）

        Args:
            state: 当前状态

        Returns:
            若应注入消息则返回 True
        """
        messages = state.get("messages", [])
        if not messages:
            # 条件 1：消息列表为空则无需注入
            return False

        # 条件 2：获取最后一条助手消息
        last_assistant_msg = self._get_last_assistant_message(messages)
        if not last_assistant_msg:
            return False

        # 条件 3：检查是否包含 view_image 工具调用
        if not self._has_view_image_tool(last_assistant_msg):
            return False

        # 条件 4：检查所有工具调用是否已完成
        if not self._all_tools_completed(messages, last_assistant_msg):
            return False

        # 条件 5：检查是否已经注入过图像详情消息（防止重复注入）
        # 在助手消息之后查找是否已存在包含图像详情关键词的 HumanMessage
        assistant_idx = messages.index(last_assistant_msg)
        for msg in messages[assistant_idx + 1 :]:
            if isinstance(msg, HumanMessage):
                content_str = str(msg.content)
                # 同时检查新旧两种提示文本，兼容历史版本
                if "Here are the images you've viewed" in content_str or "Here are the details of the images you've viewed" in content_str:
                    # 已注入过，不再重复
                    return False

        return True

    def _inject_image_message(self, state: ViewImageMiddlewareState) -> dict | None:
        """注入图像详情消息的内部辅助方法。

        若满足注入条件，则创建包含图像详情的 HumanMessage 并返回状态更新。
        返回的字典遵循 LangGraph 状态更新协议，messages 键的值将被
        对应的 reducer（add_messages）合并到现有消息列表中。

        Args:
            state: 当前状态

        Returns:
            包含新增 HumanMessage 的状态更新字典，若无需更新则返回 None
        """
        if not self._should_inject_image_message(state):
            return None

        # 创建包含文本和图像内容块的图像详情消息
        image_content = self._create_image_details_message(state)

        # 构建混合内容（文本 + 图像）的 HumanMessage
        human_msg = HumanMessage(content=image_content)

        logger.debug("Injecting image details message with images before LLM call")

        # 返回状态更新，messages 键由 add_messages 归约器自动合并
        return {"messages": [human_msg]}

    @override
    def before_model(self, state: ViewImageMiddlewareState, runtime: Runtime) -> dict | None:
        """在 LLM 调用前注入图像详情消息（同步版本）。

        该方法在每次模型调用前执行，检查上一轮是否包含已完成 view_image 工具调用。
        若满足条件，则注入包含图像详情的 HumanMessage，使 LLM 可以查看并分析图像。

        Args:
            state: 当前状态
            runtime: 运行时上下文（接口要求但未使用）

        Returns:
            包含新增 HumanMessage 的状态更新字典，若无需更新则返回 None
        """
        return self._inject_image_message(state)

    @override
    async def abefore_model(self, state: ViewImageMiddlewareState, runtime: Runtime) -> dict | None:
        """在 LLM 调用前注入图像详情消息（异步版本）。

        该方法在每次模型调用前执行，检查上一轮是否包含已完成 view_image 工具调用。
        若满足条件，则注入包含图像详情的 HumanMessage，使 LLM 可以查看并分析图像。
        异步版本与同步版本逻辑完全一致，均委托给 _inject_image_message。

        Args:
            state: 当前状态
            runtime: 运行时上下文（接口要求但未使用）

        Returns:
            包含新增 HumanMessage 的状态更新字典，若无需更新则返回 None
        """
        return self._inject_image_message(state)
