"""DeerFlow 代理系统包。

本包提供了 DeerFlow 代理系统的核心组件，包括：

- create_deerflow_agent: 纯参数代理工厂，SDK 级别的代理创建入口
- make_lead_agent: 配置驱动的 Lead Agent 应用工厂（LangGraph 图入口）
- RuntimeFeatures: 声明式特性标志，控制中间件的启用与组装
- Next / Prev: 中间件定位装饰器，用于控制自定义中间件在链中的插入位置
- ThreadState / SandboxState: 代理线程状态与沙箱状态定义

模块导入后自动预热已启用技能缓存（prime_enabled_skills_cache），
避免请求路径上的同步文件系统 I/O 阻塞。
"""

from .factory import create_deerflow_agent
from .features import Next, Prev, RuntimeFeatures
from .lead_agent import make_lead_agent
from .lead_agent.prompt import prime_enabled_skills_cache
from .thread_state import SandboxState, ThreadState

# LangGraph imports deerflow.agents when registering the graph. Prime the
# enabled-skills cache here so the request path can usually read a warm cache
# without forcing synchronous filesystem work during prompt module import.
# LangGraph 在注册图时会导入 deerflow.agents。在此处预热已启用技能缓存，
# 使得请求路径通常可以读取到已缓存的技能数据，而无需在 prompt 模块导入时
# 执行同步文件系统 I/O。
prime_enabled_skills_cache()

__all__ = [
    "create_deerflow_agent",
    "RuntimeFeatures",
    "Next",
    "Prev",
    "make_lead_agent",
    "SandboxState",
    "ThreadState",
]
