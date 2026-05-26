"""声明式特性标志与中间件定位装饰器，用于 create_deerflow_agent。

纯数据类和装饰器 —— 无 I/O，无副作用。
"""
# Declarative feature flags and middleware positioning for create_deerflow_agent.
#
# Pure data classes and decorators — no I/O, no side effects.

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from langchain.agents.middleware import AgentMiddleware


@dataclass
class RuntimeFeatures:
    """``create_deerflow_agent`` 的声明式特性标志。

    大部分特性接受以下值：
    - ``True``：使用内置默认中间件
    - ``False``：禁用
    - ``AgentMiddleware`` 实例：使用此自定义实现替代

    ``summarization`` 和 ``guardrail`` 没有内置默认 —— 它们只接受
    ``False``（禁用）或 ``AgentMiddleware`` 实例（自定义）。

    特性概览：
    - sandbox:         沙箱基础设施（ThreadData → Uploads → Sandbox）
    - memory:          异步记忆更新
    - summarization:   上下文摘要压缩（需自定义实例）
    - subagent:        子代理委派与并发限制
    - vision:          图像理解能力
    - auto_title:      自动生成线程标题
    - guardrail:       工具调用护栏（需自定义实例）
    - loop_detection:  循环检测与打断
    """

    sandbox: bool | AgentMiddleware = True
    memory: bool | AgentMiddleware = False
    summarization: Literal[False] | AgentMiddleware = False
    subagent: bool | AgentMiddleware = False
    vision: bool | AgentMiddleware = False
    auto_title: bool | AgentMiddleware = False
    guardrail: Literal[False] | AgentMiddleware = False
    loop_detection: bool | AgentMiddleware = True


# ---------------------------------------------------------------------------
# Middleware positioning decorators
# 中间件定位装饰器
# ---------------------------------------------------------------------------


def Next(anchor: type[AgentMiddleware]):
    """声明此中间件应放置在 *anchor* 之后。

    用于自定义中间件的定位控制，与 ``create_deerflow_agent`` 的
    ``extra_middleware`` 参数配合使用。

    示例::

        @Next(ClarificationMiddleware)
        class MyMiddleware(AgentMiddleware):
            ...

    上述 MyMiddleware 将被插入到 ClarificationMiddleware 之后。

    Args:
        anchor: 锚点中间件类，本中间件将放置在其之后

    Returns:
        装饰器函数，在类上设置 ``_next_anchor`` 属性

    Raises:
        TypeError: 如果 anchor 不是 AgentMiddleware 的子类
    """
    if not (isinstance(anchor, type) and issubclass(anchor, AgentMiddleware)):
        raise TypeError(f"@Next expects an AgentMiddleware subclass, got {anchor!r}")

    def decorator(cls: type[AgentMiddleware]) -> type[AgentMiddleware]:
        cls._next_anchor = anchor  # type: ignore[attr-defined]
        return cls

    return decorator


def Prev(anchor: type[AgentMiddleware]):
    """声明此中间件应放置在 *anchor* 之前。

    用于自定义中间件的定位控制，与 ``create_deerflow_agent`` 的
    ``extra_middleware`` 参数配合使用。

    示例::

        @Prev(ClarificationMiddleware)
        class MyMiddleware(AgentMiddleware):
            ...

    上述 MyMiddleware 将被插入到 ClarificationMiddleware 之前。

    Args:
        anchor: 锚点中间件类，本中间件将放置在其之前

    Returns:
        装饰器函数，在类上设置 ``_prev_anchor`` 属性

    Raises:
        TypeError: 如果 anchor 不是 AgentMiddleware 的子类
    """
    if not (isinstance(anchor, type) and issubclass(anchor, AgentMiddleware)):
        raise TypeError(f"@Prev expects an AgentMiddleware subclass, got {anchor!r}")

    def decorator(cls: type[AgentMiddleware]) -> type[AgentMiddleware]:
        cls._prev_anchor = anchor  # type: ignore[attr-defined]
        return cls

    return decorator
