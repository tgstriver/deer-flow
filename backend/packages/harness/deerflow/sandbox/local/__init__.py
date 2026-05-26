"""本地沙箱实现模块 — 基于本地文件系统的沙箱提供者。

本模块提供了 LocalSandboxProvider，它是 SandboxProvider 的本地文件系统实现，
通过虚拟路径映射（PathMapping）机制将容器路径映射到宿主机本地路径，
在无需 Docker 的场景下实现与远程沙箱一致的路径语义。

核心组件：
- LocalSandbox：本地沙箱实例，实现命令执行、文件读写、目录列表等操作，
  并负责虚拟路径与物理路径之间的双向转换
- LocalSandboxProvider：本地沙箱提供者，管理沙箱的获取、缓存和释放，
  支持按线程（thread_id）隔离的沙箱实例和 LRU 缓存淘汰
- PathMapping：路径映射配置，定义容器路径到本地路径的映射关系及只读标志
- list_dir：目录列表工具函数，支持受限深度遍历和符号链接安全检查
"""

from .local_sandbox_provider import LocalSandboxProvider

__all__ = ["LocalSandboxProvider"]
