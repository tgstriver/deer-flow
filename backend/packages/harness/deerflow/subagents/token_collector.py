"""子代理 Token 用量收集器 —— 回调处理器，用于收集子代理执行过程中的 LLM Token 用量。

每次子代理执行都会创建独立的收集器实例。子代理执行完成后，
收集到的用量记录会通过 :meth:`RunJournal.record_external_llm_usage_records`
转移到父代理的运行日志中，确保 Token 使用情况得到完整追踪。

核心设计要点：
- 基于 LangChain 的 BaseCallbackHandler 回调机制
- 通过 run_id 去重，避免同一 LLM 调用的 Token 被重复计算
- 收集 input_tokens、output_tokens 和 total_tokens 三项指标

Callback handler that collects LLM token usage within a subagent.

Each subagent execution creates its own collector. After the subagent
finishes, the collected records are transferred to the parent RunJournal
via :meth:`RunJournal.record_external_llm_usage_records`.
"""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


class SubagentTokenCollector(BaseCallbackHandler):
    """轻量级回调处理器，收集子代理中 LLM 调用的 Token 用量。

    该类通过 LangChain 的回调机制，在每次 LLM 调用结束时
    自动提取 usage_metadata 中的 Token 消耗数据并记录。

    去重机制：使用 _counted_run_ids 集合确保每个 run_id
    对应的 LLM 调用只被记录一次，防止重复计算。

    Lightweight callback handler that collects LLM token usage within a subagent.

    Attributes:
        caller: 调用方标识，格式为 "subagent:{name}"，用于区分不同子代理的用量。/ Caller identifier, format "subagent:{name}".
        _records: 已收集的用量记录列表。/ List of collected usage records.
        _counted_run_ids: 已计数的 run_id 集合，用于去重。/ Set of counted run IDs for deduplication.
    """

    def __init__(self, caller: str):
        """初始化 Token 收集器。

        Args:
            caller: 调用方标识字符串，通常为 "subagent:{agent_name}"。/ Caller identifier string, typically "subagent:{agent_name}".
        """
        super().__init__()
        self.caller = caller
        self._records: list[dict[str, int | str]] = []
        self._counted_run_ids: set[str] = set()

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: Any,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        rid = str(run_id)
        if rid in self._counted_run_ids:
            return

        for generation in response.generations:
            for gen in generation:
                if not hasattr(gen, "message"):
                    continue
                usage = getattr(gen.message, "usage_metadata", None)
                usage_dict = dict(usage) if usage else {}
                input_tk = usage_dict.get("input_tokens", 0) or 0
                output_tk = usage_dict.get("output_tokens", 0) or 0
                total_tk = usage_dict.get("total_tokens", 0) or 0
                if total_tk <= 0:
                    total_tk = input_tk + output_tk
                if total_tk <= 0:
                    continue
                self._counted_run_ids.add(rid)
                self._records.append(
                    {
                        "source_run_id": rid,
                        "caller": self.caller,
                        "input_tokens": input_tk,
                        "output_tokens": output_tk,
                        "total_tokens": total_tk,
                    }
                )
                return

    def snapshot_records(self) -> list[dict[str, int | str]]:
        """Return a copy of the accumulated usage records."""
        return list(self._records)
