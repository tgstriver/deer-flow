"""AIO Sandbox 包（AIO Sandbox Package）。

本模块是 deerflow.community.aio_sandbox 包的入口，对外暴露
AIO 容器沙箱系统的核心组件，包括：

- AioSandbox: 基于 agent-infra/sandbox Docker 容器的沙箱实现
- AioSandboxProvider: 沙箱生命周期管理器（含本地/远程后端切换）
- LocalContainerBackend: 本地 Docker/Apple Container 后端
- RemoteSandboxBackend: 远程 K8s provisioner 后端
- SandboxBackend: 沙箱后端抽象基类
- SandboxInfo: 沙箱元信息（用于跨进程发现与状态持久化）

典型用法::

    from deerflow.community.aio_sandbox import AioSandboxProvider
"""

from .aio_sandbox import AioSandbox
from .aio_sandbox_provider import AioSandboxProvider
from .backend import SandboxBackend
from .local_backend import LocalContainerBackend
from .remote_backend import RemoteSandboxBackend
from .sandbox_info import SandboxInfo

__all__ = [
    "AioSandbox",
    "AioSandboxProvider",
    "LocalContainerBackend",
    "RemoteSandboxBackend",
    "SandboxBackend",
    "SandboxInfo",
]
