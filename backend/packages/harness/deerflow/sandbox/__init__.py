"""沙箱（Sandbox）模块 — 提供隔离的代码执行与文件操作环境。

本模块是 DeerFlow 沙箱子系统的顶层入口，定义了沙箱抽象接口和提供者模式，
支持本地文件系统沙箱和基于 Docker 的远程沙箱两种实现。

核心概念：
- Sandbox：沙箱抽象基类，定义执行命令、读写文件、目录列表等操作接口
- SandboxProvider：沙箱提供者抽象基类，负责沙箱的获取（acquire）、查询（get）和释放（release）生命周期管理
- 虚拟路径映射：Agent 看到的路径（如 /mnt/user-data/workspace）与实际物理路径之间的双向转换
"""

from .sandbox import Sandbox
from .sandbox_provider import SandboxProvider, get_sandbox_provider

__all__ = [
    "Sandbox",
    "SandboxProvider",
    "get_sandbox_provider",
]
