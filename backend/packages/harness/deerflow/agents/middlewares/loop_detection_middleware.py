"""循环检测中间件 —— 检测并打断重复的工具调用循环。

P0级安全机制：防止 Agent 无限期地使用相同参数反复调用同一工具，直到递归上限强制终止运行。

双重检测策略：

  第一层 —— 基于哈希的检测（Hash-based）：
    1. 每次 Agent 输出响应后，将本轮所有工具调用（名称 + 参数）归一化后计算哈希。
    2. 在滑动窗口内追踪最近的哈希值。
    3. 若同一哈希出现次数 >= warn_threshold，向 Agent 注入一条
       "你正在重复自己 —— 请收尾" 的系统消息（每个哈希仅注入一次）。
    4. 若同一哈希出现次数 >= hard_limit，直接剥离响应中的所有 tool_calls，
       强制 Agent 产出纯文本最终答案。
    适用场景：完全相同的工具调用组合被反复执行（参数一致）。

  第二层 —— 基于频率的检测（Frequency-based）：
    1. 统计同一工具类型（不区分参数）在会话中的调用次数。
    2. 若某工具调用次数 >= tool_freq_warn，注入频率警告消息（每个工具仅一次）。
    3. 若某工具调用次数 >= tool_freq_hard_limit，强制停止。
    适用场景：跨文件读取循环等参数不断变化但行为重复的情况，
    这是第一层哈希检测无法捕获的盲区。

  两层检测协同工作，从"完全重复"到"行为重复"两个维度全面覆盖循环检测。
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import TYPE_CHECKING, override

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langgraph.runtime import Runtime

if TYPE_CHECKING:
    from deerflow.config.loop_detection_config import LoopDetectionConfig

logger = logging.getLogger(__name__)

# 默认阈值 —— 可通过构造函数覆盖
_DEFAULT_WARN_THRESHOLD = 3  # 相同工具调用组合出现 3 次后注入警告
_DEFAULT_HARD_LIMIT = 5  # 相同工具调用组合出现 5 次后强制停止
_DEFAULT_WINDOW_SIZE = 20  # 滑动窗口大小，追踪最近 N 次工具调用的哈希
_DEFAULT_MAX_TRACKED_THREADS = 100  # 线程追踪的 LRU 驱逐上限
_DEFAULT_TOOL_FREQ_WARN = 30  # 同一工具类型调用 30 次后注入频率警告
_DEFAULT_TOOL_FREQ_HARD_LIMIT = 50  # 同一工具类型调用 50 次后强制停止


def _normalize_tool_call_args(raw_args: object) -> tuple[dict, str | None]:
    """将工具调用的 args 归一化为字典 + 可选的备用键。

    部分模型提供商会将 args 序列化为 JSON 字符串而非字典。
    此函数防御性地处理这些情况，确保循环检测不会因类型不匹配而崩溃，
    同时为非字典类型的载荷保留一个稳定的备用键用于后续哈希计算。

    Args:
        raw_args: 工具调用中的原始 args 字段，可能是 dict、str、None 或其他类型。

    Returns:
        (归一化后的字典, 备用键或None)
        - 如果 raw_args 是字典，直接返回 (raw_args, None)
        - 如果 raw_args 是 JSON 字符串且可解析为字典，返回 (parsed_dict, None)
        - 如果 raw_args 是 JSON 字符串但解析结果非字典，返回 ({}, 序列化字符串)
        - 如果 raw_args 是 None，返回 ({}, None)
        - 其他类型，返回 ({}, 序列化字符串)
    """
    if isinstance(raw_args, dict):
        return raw_args, None

    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
        except (TypeError, ValueError, json.JSONDecodeError):
            # JSON 解析失败，用原始字符串作为备用键
            return {}, raw_args

        if isinstance(parsed, dict):
            return parsed, None
        # 解析成功但结果不是字典（例如列表或基本类型），序列化为备用键
        return {}, json.dumps(parsed, sort_keys=True, default=str)

    if raw_args is None:
        return {}, None

    # 其他非标准类型（如列表、数字等），序列化为备用键
    return {}, json.dumps(raw_args, sort_keys=True, default=str)


def _stable_tool_key(name: str, args: dict, fallback_key: str | None) -> str:
    """从工具名称和关键参数派生一个稳定的键，避免对噪声字段过度拟合。

    不同工具类型采用不同的键生成策略：
    - read_file：按路径 + 行号分桶（200行一桶），忽略微小行号差异，
      避免读取相邻行时产生不同键而漏检。
    - write_file / str_replace：对完整参数做哈希，因为同一路径可能
      写入不同内容，仅用路径会合并不同的调用导致误判。
    - 其他工具：提取显著字段（path, url, query, command 等），
      若无显著字段则回退到完整参数序列化或 fallback_key。

    Args:
        name: 工具名称。
        args: 归一化后的参数字典。
        fallback_key: 来自 _normalize_tool_call_args 的备用键。

    Returns:
        用于哈希计算的稳定键字符串。
    """
    # read_file 特殊处理：按行号分桶，忽略微小偏移
    if name == "read_file" and fallback_key is None:
        path = args.get("path") or ""
        start_line = args.get("start_line")
        end_line = args.get("end_line")

        # 桶大小：200 行内的差异视为同一读取区域
        bucket_size = 200
        try:
            start_line = int(start_line) if start_line is not None else 1
        except (TypeError, ValueError):
            start_line = 1
        try:
            end_line = int(end_line) if end_line is not None else start_line
        except (TypeError, ValueError):
            end_line = start_line

        # 确保起止行有序
        start_line, end_line = sorted((start_line, end_line))
        # 计算桶索引（向下取整）
        bucket_start = max(start_line, 1)
        bucket_end = max(end_line, 1)
        bucket_start = (bucket_start - 1) // bucket_size
        bucket_end = (bucket_end - 1) // bucket_size
        return f"{path}:{bucket_start}-{bucket_end}"

    # write_file / str_replace 是内容敏感型：同一路径可能在迭代中被更新为不同内容。
    # 仅使用路径等显著字段会错误合并不同的调用（假阳性），因此对完整参数做哈希。
    if name in {"write_file", "str_replace"}:
        if fallback_key is not None:
            return fallback_key
        return json.dumps(args, sort_keys=True, default=str)

    # 通用策略：提取显著字段作为键，忽略无关的噪声字段
    salient_fields = ("path", "url", "query", "command", "pattern", "glob", "cmd")
    stable_args = {field: args[field] for field in salient_fields if args.get(field) is not None}
    if stable_args:
        return json.dumps(stable_args, sort_keys=True, default=str)

    # 无显著字段时回退到备用键或完整参数序列化
    if fallback_key is not None:
        return fallback_key

    return json.dumps(args, sort_keys=True, default=str)


def _hash_tool_calls(tool_calls: list[dict]) -> str:
    """对一组工具调用集合计算确定性哈希（名称 + 稳定键）。

    此哈希设计为与顺序无关：相同的工具调用多重集无论输入顺序如何，
    都应产生相同的哈希值。这是通过排序归一化后的 (name, key) 列表实现的。

    Args:
        tool_calls: 模型返回的工具调用列表。

    Returns:
        12 字符的 MD5 十六进制前缀，作为该组工具调用的指纹。
    """
    # 将每个工具调用归一化为稳定的 (name, key) 结构
    normalized: list[str] = []
    for tc in tool_calls:
        name = tc.get("name", "")
        args, fallback_key = _normalize_tool_call_args(tc.get("args", {}))
        key = _stable_tool_key(name, args, fallback_key)

        normalized.append(f"{name}:{key}")

    # 排序以确保同一多重集的不同排列产生相同的哈希
    normalized.sort()
    blob = json.dumps(normalized, sort_keys=True, default=str)
    return hashlib.md5(blob.encode()).hexdigest()[:12]


# 警告与强制停止消息模板
_WARNING_MSG = "[LOOP DETECTED] You are repeating the same tool calls. Stop calling tools and produce your final answer now. If you cannot complete the task, summarize what you accomplished so far."

_TOOL_FREQ_WARNING_MSG = (
    "[LOOP DETECTED] You have called {tool_name} {count} times without producing a final answer. Stop calling tools and produce your final answer now. If you cannot complete the task, summarize what you accomplished so far."
)

_HARD_STOP_MSG = "[FORCED STOP] Repeated tool calls exceeded the safety limit. Producing final answer with results collected so far."

_TOOL_FREQ_HARD_STOP_MSG = "[FORCED STOP] Tool {tool_name} called {count} times — exceeded the per-tool safety limit. Producing final answer with results collected so far."


class LoopDetectionMiddleware(AgentMiddleware[AgentState]):
    """循环检测中间件 —— 检测并打断重复的工具调用循环。

    采用两层检测策略：
      - 第一层（哈希检测）：检测完全相同的工具调用组合被反复执行。
      - 第二层（频率检测）：检测同一工具类型被高频调用（参数可能不同）。

    阈值参数由上游的 LoopDetectionConfig 进行 Pydantic 校验；
    推荐通过 from_config() 方法构造实例以确保参数通过验证。

    Args:
        warn_threshold: 相同工具调用集合出现多少次后注入警告消息。默认: 3。
        hard_limit: 相同工具调用集合出现多少次后剥离所有 tool_calls。默认: 5。
        window_size: 滑动窗口大小，追踪最近 N 次工具调用的哈希。默认: 20。
        max_tracked_threads: 最大追踪线程数，超出后驱逐最久未使用的线程。默认: 100。
        tool_freq_warn: 同一工具类型（不区分参数）调用多少次后注入频率警告。
            捕获哈希检测无法发现的跨文件读取循环。默认: 30。
        tool_freq_hard_limit: 同一工具类型调用多少次后强制停止。默认: 50。
        tool_freq_overrides: 按工具名称覆盖频率阈值的字典。
            值为 (warn, hard_limit) 元组，替换全局的 tool_freq_warn / tool_freq_hard_limit。
            未列出的工具使用全局阈值。适用于提高某些预期高频工具（如 bash 批处理管道）
            的限制，而不削弱其他工具的保护。默认: None（无覆盖）。
    """

    def __init__(
        self,
        warn_threshold: int = _DEFAULT_WARN_THRESHOLD,
        hard_limit: int = _DEFAULT_HARD_LIMIT,
        window_size: int = _DEFAULT_WINDOW_SIZE,
        max_tracked_threads: int = _DEFAULT_MAX_TRACKED_THREADS,
        tool_freq_warn: int = _DEFAULT_TOOL_FREQ_WARN,
        tool_freq_hard_limit: int = _DEFAULT_TOOL_FREQ_HARD_LIMIT,
        tool_freq_overrides: dict[str, tuple[int, int]] | None = None,
    ):
        super().__init__()
        self.warn_threshold = warn_threshold
        self.hard_limit = hard_limit
        self.window_size = window_size
        self.max_tracked_threads = max_tracked_threads
        self.tool_freq_warn = tool_freq_warn
        self.tool_freq_hard_limit = tool_freq_hard_limit
        # 按工具名称覆盖的 (warn, hard_limit) 元组
        self._tool_freq_overrides: dict[str, tuple[int, int]] = tool_freq_overrides or {}
        # 线程安全的锁，保护所有共享状态
        self._lock = threading.Lock()
        # 每个线程的哈希历史（OrderedDict 实现 LRU 驱逐）
        self._history: OrderedDict[str, list[str]] = OrderedDict()
        # 每个线程中已警告过的哈希集合（避免重复注入警告）
        self._warned: dict[str, set[str]] = defaultdict(set)
        # 每个线程中每个工具的调用频率计数器
        self._tool_freq: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # 每个线程中已发出频率警告的工具名称集合
        self._tool_freq_warned: dict[str, set[str]] = defaultdict(set)

    @classmethod
    def from_config(cls, config: LoopDetectionConfig) -> LoopDetectionMiddleware:
        """从 Pydantic 校验过的配置对象构造实例，信任其已完成的验证。

        Args:
            config: 已通过 Pydantic 校验的 LoopDetectionConfig 实例。

        Returns:
            使用配置参数初始化的 LoopDetectionMiddleware 实例。
        """
        return cls(
            warn_threshold=config.warn_threshold,
            hard_limit=config.hard_limit,
            window_size=config.window_size,
            max_tracked_threads=config.max_tracked_threads,
            tool_freq_warn=config.tool_freq_warn,
            tool_freq_hard_limit=config.tool_freq_hard_limit,
            tool_freq_overrides={name: (o.warn, o.hard_limit) for name, o in config.tool_freq_overrides.items()},
        )

    def _get_thread_id(self, runtime: Runtime) -> str:
        """从运行时上下文中提取线程 ID，用于按线程隔离追踪。

        Args:
            runtime: LangGraph 运行时实例。

        Returns:
            线程 ID 字符串，若无法获取则回退到 "default"。
        """
        thread_id = runtime.context.get("thread_id") if runtime.context else None
        if thread_id:
            return thread_id
        return "default"

    def _evict_if_needed(self) -> None:
        """若追踪线程数超过上限，驱逐最久未使用的线程。

        必须在持有 self._lock 的情况下调用。
        驱逐时会同时清理该线程的所有追踪状态（历史、警告记录、频率计数）。
        """
        while len(self._history) > self.max_tracked_threads:
            # popitem(last=False) 移除 OrderedDict 中最早插入（最久未使用）的条目
            evicted_id, _ = self._history.popitem(last=False)
            self._warned.pop(evicted_id, None)
            self._tool_freq.pop(evicted_id, None)
            self._tool_freq_warned.pop(evicted_id, None)
            logger.debug("Evicted loop tracking for thread %s (LRU)", evicted_id)

    def _track_and_check(self, state: AgentState, runtime: Runtime) -> tuple[str | None, bool]:
        """追踪工具调用并检测循环。

        两层检测逻辑：
          1. **基于哈希的检测**：检测完全相同的工具调用组合被反复执行。
          2. **基于频率的检测**：检测同一工具类型被高频调用（参数可能不同），
             例如 read_file 对 40 个不同文件的循环读取。

        Args:
            state: 当前 Agent 状态，包含消息历史。
            runtime: LangGraph 运行时实例，用于获取线程 ID。

        Returns:
            (警告消息或None, 是否应强制停止)
            - (None, False): 无循环，正常继续
            - (警告消息, False): 检测到循环，注入警告但允许继续
            - (强制停止消息, True): 循环严重，强制剥离 tool_calls
        """
        messages = state.get("messages", [])
        if not messages:
            return None, False

        # 只对 AI 消息（即模型输出）进行检测
        last_msg = messages[-1]
        if getattr(last_msg, "type", None) != "ai":
            return None, False

        # 仅当消息包含工具调用时才需要检测
        tool_calls = getattr(last_msg, "tool_calls", None)
        if not tool_calls:
            return None, False

        # 获取线程 ID 并计算当前工具调用组合的哈希
        thread_id = self._get_thread_id(runtime)
        call_hash = _hash_tool_calls(tool_calls)

        with self._lock:
            # 更新或创建线程条目（移到末尾以维护 LRU 顺序）
            if thread_id in self._history:
                self._history.move_to_end(thread_id)
            else:
                self._history[thread_id] = []
                self._evict_if_needed()

            # 将当前哈希追加到滑动窗口，并裁剪超出窗口的部分
            history = self._history[thread_id]
            history.append(call_hash)
            if len(history) > self.window_size:
                history[:] = history[-self.window_size :]

            # 统计当前哈希在窗口中出现的次数
            count = history.count(call_hash)
            tool_names = [tc.get("name", "?") for tc in tool_calls]

            # --- 第一层：基于哈希的检测（完全相同的工具调用组合） ---
            if count >= self.hard_limit:
                # 达到硬性上限，强制停止：将剥离 tool_calls，迫使 Agent 输出纯文本
                logger.error(
                    "Loop hard limit reached — forcing stop",
                    extra={
                        "thread_id": thread_id,
                        "call_hash": call_hash,
                        "count": count,
                        "tools": tool_names,
                    },
                )
                return _HARD_STOP_MSG, True

            if count >= self.warn_threshold:
                # 达到警告阈值，注入警告消息（同一哈希仅警告一次）
                warned = self._warned[thread_id]
                if call_hash not in warned:
                    warned.add(call_hash)
                    logger.warning(
                        "Repetitive tool calls detected — injecting warning",
                        extra={
                            "thread_id": thread_id,
                            "call_hash": call_hash,
                            "count": count,
                            "tools": tool_names,
                        },
                    )
                    return _WARNING_MSG, False

            # --- 第二层：基于工具类型频率的检测（参数可能不同的重复调用） ---
            freq = self._tool_freq[thread_id]
            for tc in tool_calls:
                name = tc.get("name", "")
                if not name:
                    continue
                # 递增该工具类型的调用计数
                freq[name] += 1
                tc_count = freq[name]

                # 获取该工具的有效阈值（优先使用覆盖配置，否则使用全局阈值）
                if name in self._tool_freq_overrides:
                    eff_warn, eff_hard = self._tool_freq_overrides[name]
                else:
                    eff_warn, eff_hard = self.tool_freq_warn, self.tool_freq_hard_limit

                if tc_count >= eff_hard:
                    # 达到频率硬性上限，强制停止
                    logger.error(
                        "Tool frequency hard limit reached — forcing stop",
                        extra={
                            "thread_id": thread_id,
                            "tool_name": name,
                            "count": tc_count,
                        },
                    )
                    return _TOOL_FREQ_HARD_STOP_MSG.format(tool_name=name, count=tc_count), True

                if tc_count >= eff_warn:
                    # 达到频率警告阈值，注入警告（同一工具仅警告一次）
                    warned = self._tool_freq_warned[thread_id]
                    if name not in warned:
                        warned.add(name)
                        logger.warning(
                            "Tool frequency warning — too many calls to same tool type",
                            extra={
                                "thread_id": thread_id,
                                "tool_name": name,
                                "count": tc_count,
                            },
                        )
                        return _TOOL_FREQ_WARNING_MSG.format(tool_name=name, count=tc_count), False

        # 无循环迹象，正常继续
        return None, False

    @staticmethod
    def _append_text(content: str | list | None, text: str) -> str | list:
        """将文本追加到 AIMessage 的 content 字段，兼容字符串、列表和 None 三种类型。

        当 content 是内容块列表时（例如 Anthropic 的 thinking 模式），
        追加一个新的 {"type": "text", ...} 块，而不是将字符串拼接到列表上
        （那样会引发 TypeError）。

        Args:
            content: 原始 content，可能是 str、list 或 None。
            text: 要追加的文本。

        Returns:
            追加文本后的 content，类型与输入一致（str 或 list）。
        """
        if content is None:
            return text
        if isinstance(content, list):
            # 列表模式：追加新的内容块
            return [*content, {"type": "text", "text": f"\n\n{text}"}]
        if isinstance(content, str):
            # 字符串模式：直接拼接
            return content + f"\n\n{text}"
        # 兜底：将非预期类型强制转为字符串后拼接，避免 TypeError
        return str(content) + f"\n\n{text}"

    @staticmethod
    def _build_hard_stop_update(last_msg, content: str | list) -> dict:
        """构建强制停止时的消息更新字典，清除工具调用元数据。

        强制停止时需要将 AIMessage 从"包含工具调用"变为"纯文本回复"，
        因此必须清理 tool_calls、additional_kwargs 中的工具调用信息，
        以及将 finish_reason 从 "tool_calls" 修正为 "stop"。

        Args:
            last_msg: 原始 AIMessage 对象。
            content: 更新后的 content 内容。

        Returns:
            用于 model_copy(update=...) 的更新字典。
        """
        # 清空 tool_calls 列表，使消息不再触发工具执行
        update = {
            "tool_calls": [],
            "content": content,
        }

        # 清除 additional_kwargs 中残留的工具调用信息
        additional_kwargs = dict(getattr(last_msg, "additional_kwargs", {}) or {})
        for key in ("tool_calls", "function_call"):
            additional_kwargs.pop(key, None)
        update["additional_kwargs"] = additional_kwargs

        # 将 finish_reason 从 "tool_calls" 修正为 "stop"，表示正常结束
        response_metadata = deepcopy(getattr(last_msg, "response_metadata", {}) or {})
        if response_metadata.get("finish_reason") == "tool_calls":
            response_metadata["finish_reason"] = "stop"
        update["response_metadata"] = response_metadata

        return update

    def _apply(self, state: AgentState, runtime: Runtime) -> dict | None:
        """执行循环检测并应用干预措施（警告注入或强制停止）。

        根据检测结果采取不同的干预措施：
        - 强制停止：剥离 AIMessage 的所有 tool_calls，将内容替换为停止消息，
          使 Agent 被迫输出纯文本最终答案。
        - 警告：将警告文本追加到 AIMessage 的 content 中，保留 tool_calls
          以便工具节点正常执行。

        Args:
            state: 当前 Agent 状态。
            runtime: LangGraph 运行时实例。

        Returns:
            更新字典（包含修改后的消息）或 None（无干预）。
        """
        warning, hard_stop = self._track_and_check(state, runtime)

        if hard_stop:
            # 强制停止：剥离最后一条 AIMessage 的 tool_calls，迫使 Agent 输出纯文本
            messages = state.get("messages", [])
            last_msg = messages[-1]
            content = self._append_text(last_msg.content, warning or _HARD_STOP_MSG)
            # 通过 model_copy 创建修改后的消息副本，清除所有工具调用相关字段
            stripped_msg = last_msg.model_copy(update=self._build_hard_stop_update(last_msg, content))
            return {"messages": [stripped_msg]}

        if warning:
            # v2.0-m1 临时方案 —— 参见 #2724。
            #
            # 将警告追加到 AIMessage 的 content 中，而不是插入一条独立的 HumanMessage。
            # 原因：在 AIMessage(tool_calls=...) 和其对应的 ToolMessage 响应之间
            # 插入任何非工具消息，会破坏 OpenAI/Moonshot 的严格配对验证
            # （"tool_call_ids did not have response messages"），因为在 after_model
            # 阶段工具节点尚未执行。
            # 保留 tool_calls，使工具节点仍能正常执行。
            #
            # 这是一个临时缓解措施：修改已有的 AIMessage 以携带框架生成的文本，
            # 会导致循环警告文本泄漏到下游消费者（MemoryMiddleware 事实提取、
            # TitleMiddleware、遥测、模型回放）中，仿佛是模型自己说出的。
            # 正确的修复方案是将警告注入时机从 after_model 推迟到 wrap_model_call，
            # 这样所有先前的 ToolMessage 都已存在于请求中 —— 参见 RFC #2517
            # （其验收标准包括"循环干预不会留下无效的 tool-call/tool-message 状态"）
            # 以及 fix/loop-detection-tool-call-pairing 分支上的原型实现。
            messages = state.get("messages", [])
            last_msg = messages[-1]
            patched_msg = last_msg.model_copy(update={"content": self._append_text(last_msg.content, warning)})
            return {"messages": [patched_msg]}

        # 无循环迹象，不干预
        return None

    @override
    def after_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        """同步钩子：模型输出后执行循环检测。

        在 Agent 中间件链中，此方法在每次模型生成响应后被调用，
        用于检查是否存在重复工具调用循环。
        """
        return self._apply(state, runtime)

    @override
    async def aafter_model(self, state: AgentState, runtime: Runtime) -> dict | None:
        """异步钩子：模型输出后执行循环检测。

        异步版本与同步版本逻辑完全一致，均委托给 _apply 方法。
        """
        return self._apply(state, runtime)

    def reset(self, thread_id: str | None = None) -> None:
        """重置追踪状态。若指定 thread_id，仅清除该线程的状态。

        Args:
            thread_id: 要重置的线程 ID。若为 None，清除所有线程的追踪状态。
        """
        with self._lock:
            if thread_id:
                # 仅重置指定线程的所有追踪状态
                self._history.pop(thread_id, None)
                self._warned.pop(thread_id, None)
                self._tool_freq.pop(thread_id, None)
                self._tool_freq_warned.pop(thread_id, None)
            else:
                # 重置所有线程的追踪状态
                self._history.clear()
                self._warned.clear()
                self._tool_freq.clear()
                self._tool_freq_warned.clear()
