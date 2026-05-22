"""Token 使用量日志记录与步骤归因标注中间件。

本模块实现了 TokenUsageMiddleware，负责两件核心工作：

1. **Token 使用量日志**：当模型返回 AIMessage 时，提取 usage_metadata 并记录日志，
   包括 input_tokens、output_tokens、total_tokens 以及详细的 token 明细。

2. **步骤归因标注**：为每次 AI 响应构建结构化的归因信息（attribution），描述该步骤
   产生的具体动作（todo 操作、工具调用、子代理派发等），以便前端精确展示每一步
   消耗了多少 token。归因数据写入 AIMessage.additional_kwargs 中的专用 key。

此外，中间件还负责将子代理（subagent）执行完毕后的 token 使用量**回写合并**到
派发该子代理的原始 AIMessage 上，确保 token 统计的归属准确。

子模块辅助函数说明：
- _string_arg: 安全地将任意值转换为非空字符串或 None
- _normalize_todos: 规范化 todo 列表，过滤非法条目
- _todo_action_kind: 根据前后 todo 状态推断动作类型（新增/开始/完成/更新）
- _build_todo_actions: 对比前后 todo 列表，生成精确的变更动作列表（单一事实来源）
- _describe_tool_call: 将工具调用描述为结构化动作，write_todos 展开为 todo 动作，
  task 展开为子代理动作，搜索类工具展开为搜索动作等
- _infer_step_kind: 根据消息内容和动作列表推断步骤类型
- _build_attribution: 为一条 AIMessage 构建完整的归因字典
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.todo import Todo
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.runtime import Runtime

logger = logging.getLogger(__name__)

# 归因数据在 AIMessage.additional_kwargs 中存储时使用的键名
TOKEN_USAGE_ATTRIBUTION_KEY = "token_usage_attribution"


def _string_arg(value: Any) -> str | None:
    """将任意值安全地转换为非空字符串，或返回 None。

    如果值是字符串类型，去除首尾空白后返回；若空白后为空则返回 None。
    非字符串类型一律返回 None。此函数用于从可能类型不一致的字典中
    提取字符串参数，例如工具调用的 args 中的字段值。

    Args:
        value: 待转换的任意值。

    Returns:
        去除空白后的非空字符串，或 None。
    """
    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None
    return None


def _normalize_todos(value: Any) -> list[Todo]:
    """将原始 todo 数据规范化为 Todo 列表。

    输入通常来自工具调用参数（如 write_todos 的 args.todos），
    可能包含不合法的条目。本函数过滤掉非字典项，并仅保留
    合法的 content（非空字符串）和 status（pending/in_progress/completed）字段。

    Args:
        value: 原始 todo 数据，期望为字典列表。

    Returns:
        规范化后的 Todo 列表，每个 Todo 是仅包含合法字段的字典。
    """
    if not isinstance(value, list):
        return []

    normalized: list[Todo] = []
    for item in value:
        if not isinstance(item, dict):
            continue

        todo: Todo = {}
        content = _string_arg(item.get("content"))
        status = item.get("status")

        if content is not None:
            todo["content"] = content
        # 仅接受三种合法的 todo 状态
        if status in {"pending", "in_progress", "completed"}:
            todo["status"] = status

        normalized.append(todo)

    return normalized


def _todo_action_kind(previous: Todo | None, current: Todo) -> str:
    """根据前后 todo 状态推断动作类型。

    动作类型包括：
    - "todo_complete": todo 已完成（状态为 completed）
    - "todo_start": todo 已开始（状态为 in_progress）
    - "todo_update": todo 已更新（内容变更或其它状态变化）

    当 previous 为 None 时，表示这是一个新增的 todo 条目。
    当 previous 的 content 与 current 不同时，一律视为更新操作。

    Args:
        previous: 前一次的 todo 条目，None 表示新增。
        current: 当前 todo 条目。

    Returns:
        动作类型字符串。
    """
    status = current.get("status")
    previous_content = previous.get("content") if previous else None
    current_content = current.get("content")

    # 新增的 todo：没有前序条目，根据当前状态推断动作
    if previous is None:
        if status == "completed":
            return "todo_complete"
        if status == "in_progress":
            return "todo_start"
        return "todo_update"

    # 内容发生变更，统一归为更新
    if previous_content != current_content:
        return "todo_update"

    # 内容相同，根据状态变化推断动作
    if status == "completed":
        return "todo_complete"
    if status == "in_progress":
        return "todo_start"
    return "todo_update"


def _build_todo_actions(previous_todos: list[Todo], next_todos: list[Todo]) -> list[dict[str, Any]]:
    """对比前后 todo 列表，生成精确的变更动作列表。

    这是 write_todos 工具 token 归因的**单一事实来源**。前端在缺少此元数据时
    会回退到通用的"更新待办列表"标签。

    匹配策略采用两阶段：
    1. **按内容匹配**：同一 content 的前后 todo 配对，支持重复 content
       （使用有序列表逐个匹配，已匹配的索引被记录以防重复配对）。
    2. **按位置回退匹配**：若内容匹配未成功，且当前位置的前序 todo 未被匹配，
       则按索引位置配对。

    匹配后，仅当状态或内容实际发生变更时才生成动作。未被匹配的前序 todo
    视为已删除，生成 "todo_remove" 动作。

    Args:
        previous_todos: 变更前的 todo 列表。
        next_todos: 变更后的 todo 列表。

    Returns:
        变更动作列表，每个动作包含 kind 和 content 字段。
    """
    # 按内容构建前序 todo 的索引映射，支持同一内容出现多次
    previous_by_content: dict[str, list[tuple[int, Todo]]] = defaultdict(list)
    matched_previous_indices: set[int] = set()

    for index, todo in enumerate(previous_todos):
        content = todo.get("content")
        if isinstance(content, str) and content:
            previous_by_content[content].append((index, todo))

    actions: list[dict[str, Any]] = []

    for index, todo in enumerate(next_todos):
        content = todo.get("content")
        if not isinstance(content, str) or not content:
            continue

        # 第一阶段：按内容匹配前序 todo
        previous_match: Todo | None = None
        content_matches = previous_by_content.get(content)
        if content_matches:
            # 跳过已被匹配的前序 todo，确保一对一配对
            while content_matches and content_matches[0][0] in matched_previous_indices:
                content_matches.pop(0)
            if content_matches:
                previous_index, previous_match = content_matches.pop(0)
                matched_previous_indices.add(previous_index)

        # 第二阶段：按位置回退匹配
        if previous_match is None and index < len(previous_todos) and index not in matched_previous_indices:
            previous_match = previous_todos[index]
            matched_previous_indices.add(index)

        # 若前序匹配的 todo 内容和状态均未变化，则跳过（无实际变更）
        if previous_match is not None:
            previous_content = previous_match.get("content")
            previous_status = previous_match.get("status")
            if previous_content == content and previous_status == todo.get("status"):
                continue

        # 记录变更动作
        actions.append(
            {
                "kind": _todo_action_kind(previous_match, todo),
                "content": content,
            }
        )

    # 遍历未匹配的前序 todo，视为已删除
    for index, todo in enumerate(previous_todos):
        if index in matched_previous_indices:
            continue

        content = todo.get("content")
        if not isinstance(content, str) or not content:
            continue

        actions.append(
            {
                "kind": "todo_remove",
                "content": content,
            }
        )

    return actions


def _describe_tool_call(tool_call: dict[str, Any], todos: list[Todo]) -> list[dict[str, Any]]:
    """将工具调用描述为结构化动作列表。

    根据工具名称，将调用展开为不同类型的动作：
    - **write_todos**: 展开为具体的 todo 变更动作（todo_start/todo_complete/todo_update/todo_remove）。
      若无实际变更，则退化为通用 tool 动作。
    - **task**: 展开为子代理派发动作（subagent），包含描述和子代理类型。
    - **web_search / image_search**: 展开为搜索动作（search），包含查询关键词。
    - **present_files**: 展开为文件展示动作（present_files）。
    - **ask_clarification**: 展开为澄清动作（clarification）。
    - 其它工具: 通用工具动作（tool），包含工具名称和可选描述。

    对于 write_todos 工具，使用当前 todos 作为前序状态来计算变更，
    因为同一轮模型响应中可能有多次 write_todos 调用，后续调用需要
    基于前面调用产生的新 todo 列表来对比。

    Args:
        tool_call: 工具调用字典，包含 name、args、id 等字段。
        todos: 当前的 todo 列表，作为 write_todos 变更对比的前序状态。

    Returns:
        结构化动作列表。
    """
    name = _string_arg(tool_call.get("name")) or "unknown"
    args = tool_call.get("args") if isinstance(tool_call.get("args"), dict) else {}
    tool_call_id = _string_arg(tool_call.get("id"))

    # write_todos 工具：展开为具体的 todo 变更动作
    if name == "write_todos":
        next_todos = _normalize_todos(args.get("todos"))
        actions = _build_todo_actions(todos, next_todos)
        if not actions:
            # 无实际变更时，退化为通用 tool 动作
            return [
                {
                    "kind": "tool",
                    "tool_name": name,
                    "tool_call_id": tool_call_id,
                }
            ]
        # 将 todo 动作与 tool_call_id 关联
        return [
            {
                **action,
                "tool_call_id": tool_call_id,
            }
            for action in actions
        ]

    # task 工具（子代理派发）：记录子代理类型和描述
    if name == "task":
        return [
            {
                "kind": "subagent",
                "description": _string_arg(args.get("description")),
                "subagent_type": _string_arg(args.get("subagent_type")),
                "tool_call_id": tool_call_id,
            }
        ]

    # 搜索类工具：记录查询关键词
    if name in {"web_search", "image_search"}:
        query = _string_arg(args.get("query"))
        return [
            {
                "kind": "search",
                "tool_name": name,
                "query": query,
                "tool_call_id": tool_call_id,
            }
        ]

    # 文件展示工具
    if name == "present_files":
        return [
            {
                "kind": "present_files",
                "tool_call_id": tool_call_id,
            }
        ]

    # 澄清请求工具
    if name == "ask_clarification":
        return [
            {
                "kind": "clarification",
                "tool_call_id": tool_call_id,
            }
        ]

    # 其它通用工具：记录工具名称和可选描述
    return [
        {
            "kind": "tool",
            "tool_name": name,
            "description": _string_arg(args.get("description")),
            "tool_call_id": tool_call_id,
        }
    ]


def _infer_step_kind(message: AIMessage, actions: list[dict[str, Any]]) -> str:
    """根据消息内容和动作列表推断步骤类型。

    步骤类型包括：
    - "todo_update": 仅包含单个 todo 相关动作（start/complete/update/remove）
    - "subagent_dispatch": 仅包含单个子代理派发动作
    - "tool_batch": 包含多个动作的批量工具调用
    - "final_answer": 无工具调用但有文本内容（最终回答）
    - "thinking": 无工具调用也无文本内容（纯思考/推理步骤）

    推断优先级：有动作时先判断动作类型，无动作时根据内容判断。

    Args:
        message: AI 消息对象。
        actions: 由 _describe_tool_call 生成的动作列表。

    Returns:
        步骤类型字符串。
    """
    if actions:
        first_kind = actions[0].get("kind")
        # 单一 todo 动作 → todo_update 步骤
        if len(actions) == 1 and first_kind in {"todo_start", "todo_complete", "todo_update", "todo_remove"}:
            return "todo_update"
        # 单一子代理派发 → subagent_dispatch 步骤
        if len(actions) == 1 and first_kind == "subagent":
            return "subagent_dispatch"
        # 多个动作 → tool_batch 步骤
        return "tool_batch"

    # 无动作：有文本内容为最终回答，无内容为纯思考
    if message.content:
        return "final_answer"
    return "thinking"


def _has_tool_call(message: AIMessage, tool_call_id: str) -> bool:
    """判断 AIMessage 是否包含指定 id 的工具调用。

    兼容两种 tool_calls 格式：字典格式和对象格式（带 .id 属性）。
    用于在子代理 token 回写时，从 ToolMessage 反向查找派发它的 AIMessage。

    Args:
        message: 待检查的 AI 消息。
        tool_call_id: 要查找的工具调用 id。

    Returns:
        若消息中包含该 id 的工具调用则返回 True，否则 False。
    """
    for tc in message.tool_calls or []:
        if isinstance(tc, dict):
            if tc.get("id") == tool_call_id:
                return True
        elif hasattr(tc, "id") and tc.id == tool_call_id:
            return True
    return False


def _build_attribution(message: AIMessage, todos: list[Todo]) -> dict[str, Any]:
    """为一条 AIMessage 构建完整的 token 使用归因字典。

    遍历消息中的所有工具调用，逐个描述为结构化动作，并按顺序累加。
    对于 write_todos 工具调用，更新 current_todos 以便后续的 write_todos
    调用能基于最新的 todo 列表做变更对比。

    归因字典的 schema 设计遵循**只增不改**原则：新增字段时旧版前端可以安全忽略
    未知字段并优雅降级。

    Args:
        message: AI 消息对象，包含 tool_calls。
        todos: 当前状态中的 todo 列表，作为变更对比基准。

    Returns:
        归因字典，包含 version、kind、shared_attribution、tool_call_ids、actions。
    """
    tool_calls = getattr(message, "tool_calls", None) or []
    actions: list[dict[str, Any]] = []
    # 当前 todo 列表，会在遇到 write_todos 调用时更新
    current_todos = list(todos)

    for raw_tool_call in tool_calls:
        if not isinstance(raw_tool_call, dict):
            continue

        # 描述工具调用，展开为具体动作
        described_actions = _describe_tool_call(raw_tool_call, current_todos)
        actions.extend(described_actions)

        # 若为 write_todos 调用，更新当前 todo 列表作为后续对比基准
        if raw_tool_call.get("name") == "write_todos":
            args = raw_tool_call.get("args") if isinstance(raw_tool_call.get("args"), dict) else {}
            current_todos = _normalize_todos(args.get("todos"))

    # 收集所有工具调用的 id 列表
    tool_call_ids: list[str] = []
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue

        tool_call_id = _string_arg(tool_call.get("id"))
        if tool_call_id is not None:
            tool_call_ids.append(tool_call_id)

    return {
        # Schema 变更应尽量保持只增不改，以便旧版前端能安全忽略未知字段并优雅降级
        "version": 1,
        "kind": _infer_step_kind(message, actions),
        # 是否为共享归因（多个动作共享同一轮模型调用的 token）
        "shared_attribution": len(actions) > 1,
        "tool_call_ids": tool_call_ids,
        "actions": actions,
    }


class TokenUsageMiddleware(AgentMiddleware):
    """Token 使用量日志记录与步骤归因标注中间件。

    在模型响应之后执行（after_model / aafter_model），完成两件核心工作：
    1. 将子代理的 token 使用量回写合并到派发它的 AIMessage 上。
    2. 为最新的 AIMessage 构建归因信息并写入 additional_kwargs。

    子代理 token 回写机制：
    当 task 工具执行完毕后，其 token 使用量会被缓存（按 tool_call_id 索引）。
    本中间件在检测到 ToolMessage 时，从缓存中取出子代理的使用量，反向查找
    派发它的 AIMessage，并将 token 数量合并到该消息的 usage_metadata 中。
    同一轮模型响应可能派发多个 task 工具调用，因此需要合并到同一个 AIMessage。
    """

    def _apply(self, state: AgentState) -> dict | None:
        """执行中间件的核心逻辑。

        处理流程：
        1. 从消息列表末尾向前扫描连续的 ToolMessage，提取子代理 token 使用量
           并回写合并到对应的派发 AIMessage。
        2. 若最新消息是 AIMessage，记录 token 使用日志并构建归因信息。
        3. 返回需要更新的消息列表。

        Args:
            state: 当前代理状态，包含 messages 和 todos 等字段。

        Returns:
            包含更新消息的字典，或 None（无需更新时）。
        """
        messages = state.get("messages", [])
        if not messages:
            return None

        # ---- 子代理 token 使用量回写合并 ----
        # 当 task 工具完成时，其使用量按 tool_call_id 被缓存。
        # 检测到 ToolMessage 后，向前搜索对应的 AIMessage 并合并 token。
        # 从最新消息的前一条开始，向前遍历连续的 ToolMessage，
        # 使得同一轮模型响应派发的多个并发 task 工具调用都能将子代理 token
        # 回写到同一条派发消息（合并为一次更新）。
        state_updates: dict[int, AIMessage] = {}
        if len(messages) >= 2:
            from deerflow.tools.builtins.task_tool import pop_cached_subagent_usage

            idx = len(messages) - 2
            while idx >= 0:
                tool_msg = messages[idx]
                # 遇到非 ToolMessage 或无 tool_call_id 时停止向前扫描
                if not isinstance(tool_msg, ToolMessage) or not tool_msg.tool_call_id:
                    break

                subagent_usage = pop_cached_subagent_usage(tool_msg.tool_call_id)
                if subagent_usage:
                    # 从 ToolMessage 向前搜索派发它的 AIMessage。
                    # 一次模型响应可能派发多个 task 工具调用，因此不能假设固定偏移。
                    dispatch_idx = idx - 1
                    while dispatch_idx >= 0:
                        candidate = messages[dispatch_idx]
                        if isinstance(candidate, AIMessage) and _has_tool_call(candidate, tool_msg.tool_call_id):
                            # 合并到已有的更新（同一 AIMessage 可能已有其它 task 的 token 回写），
                            # 或从原始消息的 usage_metadata 开始合并。
                            existing_update = state_updates.get(dispatch_idx)
                            prev = existing_update.usage_metadata if existing_update else (getattr(candidate, "usage_metadata", None) or {})
                            merged = {
                                **prev,
                                "input_tokens": prev.get("input_tokens", 0) + subagent_usage["input_tokens"],
                                "output_tokens": prev.get("output_tokens", 0) + subagent_usage["output_tokens"],
                                "total_tokens": prev.get("total_tokens", 0) + subagent_usage["total_tokens"],
                            }
                            state_updates[dispatch_idx] = candidate.model_copy(update={"usage_metadata": merged})
                            break
                        dispatch_idx -= 1
                idx -= 1

        # ---- 处理最新的 AIMessage ----
        last = messages[-1]
        if not isinstance(last, AIMessage):
            # 最新消息不是 AIMessage（可能是 ToolMessage），
            # 仅返回子代理 token 回写的更新（如有）
            if state_updates:
                return {"messages": [state_updates[idx] for idx in sorted(state_updates)]}
            return None

        # 记录 token 使用量日志
        usage = getattr(last, "usage_metadata", None)
        if usage:
            input_token_details = usage.get("input_token_details") or {}
            output_token_details = usage.get("output_token_details") or {}
            detail_parts = []
            if input_token_details:
                detail_parts.append(f"input_token_details={input_token_details}")
            if output_token_details:
                detail_parts.append(f"output_token_details={output_token_details}")
            detail_suffix = f" {' '.join(detail_parts)}" if detail_parts else ""
            logger.info(
                "LLM token usage: input=%s output=%s total=%s%s",
                usage.get("input_tokens", "?"),
                usage.get("output_tokens", "?"),
                usage.get("total_tokens", "?"),
                detail_suffix,
            )

        # 构建归因信息并写入 additional_kwargs
        todos = state.get("todos") or []
        attribution = _build_attribution(last, todos if isinstance(todos, list) else [])
        additional_kwargs = dict(getattr(last, "additional_kwargs", {}) or {})

        # 若归因信息未变化，跳过更新（避免不必要的消息复制）
        if additional_kwargs.get(TOKEN_USAGE_ATTRIBUTION_KEY) == attribution:
            return {"messages": [state_updates[idx] for idx in sorted(state_updates)]} if state_updates else None

        # 写入归因信息并创建更新后的消息
        additional_kwargs[TOKEN_USAGE_ATTRIBUTION_KEY] = attribution
        updated_msg = last.model_copy(update={"additional_kwargs": additional_kwargs})
        # 将最新 AIMessage 的更新也加入 state_updates，按索引排序后统一返回
        state_updates[len(messages) - 1] = updated_msg
        return {"messages": [state_updates[idx] for idx in sorted(state_updates)]}

    @override
    def after_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        """同步版本的模型后处理钩子。

        Args:
            state: 当前代理状态。
            runtime: LangGraph 运行时。

        Returns:
            状态更新字典或 None。
        """
        return self._apply(state)

    @override
    async def aafter_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        """异步版本的模型后处理钩子。

        异步版本与同步版本共享相同的逻辑（_apply），因为内部无异步 I/O 操作。

        Args:
            state: 当前代理状态。
            runtime: LangGraph 运行时。

        Returns:
            状态更新字典或 None。
        """
        return self._apply(state)
