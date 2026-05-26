"""本地沙箱提供者 — 基于本地文件系统的沙箱实例管理与路径映射。

本模块实现了 LocalSandboxProvider，提供按线程隔离的本地沙箱实例管理：
- 每个线程（thread_id）拥有独立的 LocalSandbox，其路径映射解析 /mnt/user-data/... 到该线程的宿主目录
- 使用 LRU 缓存（默认上限 256 个）管理线程沙箱实例，防止长时间运行时内存无限增长
- 通过 threading.Lock 保证多线程并发安全
- 支持静态路径映射（skills 目录、自定义挂载）和按线程动态路径映射（user-data、acp-workspace）

关键概念：
- 虚拟路径前缀：/mnt/user-data、/mnt/acp-workspace、/mnt/skills 等对 Agent 暴露的统一路径
- 路径映射（PathMapping）：容器路径 -> 本地路径的映射，支持只读标志
- LRU 缓存淘汰：当缓存超出上限时，最久未使用的沙箱实例被驱逐
"""

import logging
import threading
from collections import OrderedDict
from pathlib import Path

from deerflow.sandbox.local.local_sandbox import LocalSandbox, PathMapping
from deerflow.sandbox.sandbox import Sandbox
from deerflow.sandbox.sandbox_provider import SandboxProvider

logger = logging.getLogger(__name__)

# Module-level alias kept for backward compatibility with older callers/tests
# that reach into ``local_sandbox_provider._singleton`` directly. New code reads
# the provider instance attributes (``_generic_sandbox`` / ``_thread_sandboxes``)
# instead.
# 模块级单例别名，仅为向后兼容旧调用方/测试而保留。
# 新代码应通过 provider 实例的 _generic_sandbox / _thread_sandboxes 属性访问。
_singleton: LocalSandbox | None = None

# Virtual prefixes that must be reserved by the per-thread mappings created in
# ``acquire`` — custom mounts from ``config.yaml`` may not overlap with these.
# 虚拟路径前缀 — 按线程创建的路径映射必须保留这些前缀，
# config.yaml 中的自定义挂载不可与这些前缀重叠。
_USER_DATA_VIRTUAL_PREFIX = "/mnt/user-data"
_ACP_WORKSPACE_VIRTUAL_PREFIX = "/mnt/acp-workspace"

# Default upper bound on per-thread LocalSandbox instances retained in memory.
# Each cached instance is cheap (a small Python object with a list of
# PathMapping and a set of agent-written paths used for reverse resolve), but
# in a long-running gateway the number of distinct thread_ids is unbounded.
# When the cap is exceeded the least-recently-used entry is dropped; the next
# ``acquire(thread_id)`` for that thread simply rebuilds the sandbox at the
# cost of losing its accumulated ``_agent_written_paths`` (read_file falls
# back to no reverse resolution, which is the same behaviour as a fresh run).
# 默认线程沙箱缓存上限。每个缓存实例开销很小（包含 PathMapping 列表和
# agent_written_paths 集合），但在长时间运行的网关中，不同 thread_id 数量
# 无上限。超出上限时淘汰最久未使用的条目；下次 acquire 时重新构建沙箱，
# 代价是丢失 _agent_written_paths（read_file 退化为不做反向路径解析）。
DEFAULT_MAX_CACHED_THREAD_SANDBOXES = 256


class LocalSandboxProvider(SandboxProvider):
    """本地文件系统沙箱提供者 — 支持按线程隔离的路径映射。

    早期版本返回一个进程级单例 LocalSandbox（id 为 "local"），无法满足
    /mnt/user-data/... 的按线程隔离语义，因为对应的宿主目录是按线程分配的
    （{base_dir}/users/{user_id}/threads/{thread_id}/user-data/）。

    当前实现为每个 thread_id 生成独立的 LocalSandbox，其 path_mappings 包含
    按线程分配的 /mnt/user-data/{workspace,uploads,outputs} 和 /mnt/acp-workspace
    映射，与 AioSandboxProvider 的 Docker 卷挂载方式对齐。
    传统的 acquire()/acquire(None) 调用仍返回通用单例（id "local"），
    供无线程上下文的调用方和测试使用。

    线程安全：acquire、get、reset 可能被多线程调用（网关工具分发、子代理
    工作池、后台内存更新器等），因此所有缓存状态变更通过 provider 级别的
    threading.Lock 串行化，与 AioSandboxProvider 的模式一致。

    内存控制：_thread_sandboxes 是一个 LRU 缓存，上限为 max_cached_threads
    （默认 DEFAULT_MAX_CACHED_THREAD_SANDBOXES=256）。超出上限时淘汰最久未
    使用的条目；被淘汰线程下次 acquire 时重建沙箱（仅丢失 _agent_written_paths
    反向解析提示，read_file 输出优雅降级）。

    Local-filesystem sandbox provider with per-thread path scoping.

    Earlier revisions of this provider returned a single process-wide
    ``LocalSandbox`` keyed by the literal id ``"local"``. That singleton could
    not honour the documented ``/mnt/user-data/...`` contract at the public
    ``Sandbox`` API boundary because the corresponding host directory is
    per-thread (``{base_dir}/users/{user_id}/threads/{thread_id}/user-data/``).

    The provider now produces a fresh ``LocalSandbox`` per ``thread_id`` whose
    ``path_mappings`` include thread-scoped entries for
    ``/mnt/user-data/{workspace,uploads,outputs}`` and ``/mnt/acp-workspace``,
    mirroring how :class:`AioSandboxProvider` bind-mounts those paths into its
    docker container. The legacy ``acquire()`` / ``acquire(None)`` call still
    returns a generic singleton with id ``"local"`` for callers (and tests)
    that do not have a thread context.

    Thread-safety: ``acquire``, ``get`` and ``reset`` may be invoked from
    multiple threads (Gateway tool dispatch, subagent worker pools, the
    background memory updater, …) so all cache state changes are serialised
    through a provider-wide :class:`threading.Lock`. This matches the pattern
    used by :class:`AioSandboxProvider`.

    Memory bound: ``_thread_sandboxes`` is an LRU cache capped at
    ``max_cached_threads`` (default :data:`DEFAULT_MAX_CACHED_THREAD_SANDBOXES`).
    When the cap is exceeded the least-recently-used entry is evicted on the
    next ``acquire``; the evicted thread's next ``acquire`` rebuilds a fresh
    sandbox (losing only its ``_agent_written_paths`` reverse-resolve hint,
    which gracefully degrades read_file output).
    """

    uses_thread_data_mounts = True  # 标识此提供者使用按线程的数据挂载

    def __init__(self, max_cached_threads: int = DEFAULT_MAX_CACHED_THREAD_SANDBOXES):
        """初始化本地沙箱提供者，建立静态路径映射。

        Args:
            max_cached_threads: LRU 缓存中保留的线程沙箱上限。
                超出上限时，最久未使用的条目在下次 acquire 时被淘汰。
        """
        self._path_mappings = self._setup_path_mappings()
        self._generic_sandbox: LocalSandbox | None = None  # 通用单例沙箱（id="local"），供无线程上下文的调用方使用
        self._thread_sandboxes: OrderedDict[str, LocalSandbox] = OrderedDict()  # 按线程 ID 缓存的沙箱实例（LRU 有序字典）
        self._max_cached_threads = max_cached_threads  # LRU 缓存上限
        self._lock = threading.Lock()  # 保护缓存状态的并发访问锁

    def _setup_path_mappings(self) -> list[PathMapping]:
        """建立静态路径映射 — 所有沙箱实例共享的映射配置。

        静态映射覆盖 skills 目录和 config.yaml 中的自定义挂载，
        这些都是进程级的，对所有线程相同。
        按线程的 /mnt/user-data/... 和 /mnt/acp-workspace 映射在 acquire() 中追加，
        因为它们依赖于 thread_id 和有效的 user_id。

        Setup static path mappings shared by every sandbox this provider yields.

        Static mappings cover the skills directory and any custom mounts from
        ``config.yaml`` — both are process-wide and identical for every thread.
        Per-thread ``/mnt/user-data/...`` and ``/mnt/acp-workspace`` mappings
        are appended inside :meth:`acquire` because they depend on
        ``thread_id`` and the effective ``user_id``.

        Returns:
            静态路径映射列表 / List of static path mappings
        """
        mappings: list[PathMapping] = []

        # Map skills container path to local skills directory
        # 将 skills 容器路径映射到本地 skills 目录
        try:
            from deerflow.config import get_app_config

            config = get_app_config()
            skills_path = config.skills.get_skills_path()
            container_path = config.skills.container_path

            # Only add mapping if skills directory exists
            # 仅当 skills 目录存在时添加映射
            if skills_path.exists():
                mappings.append(
                    PathMapping(
                        container_path=container_path,
                        local_path=str(skills_path),
                        read_only=True,  # Skills directory is always read-only / Skills 目录始终只读
                    )
                )

            # Map custom mounts from sandbox config
            # 从沙箱配置映射自定义挂载
            _RESERVED_CONTAINER_PREFIXES = [
                container_path,
                _ACP_WORKSPACE_VIRTUAL_PREFIX,
                _USER_DATA_VIRTUAL_PREFIX,
            ]
            sandbox_config = config.sandbox
            if sandbox_config and sandbox_config.mounts:
                for mount in sandbox_config.mounts:
                    host_path = Path(mount.host_path)
                    container_path = mount.container_path.rstrip("/") or "/"

                    if not host_path.is_absolute():
                        logger.warning(
                            "Mount host_path must be absolute, skipping: %s -> %s",
                            mount.host_path,
                            mount.container_path,
                        )
                        continue

                    if not container_path.startswith("/"):
                        logger.warning(
                            "Mount container_path must be absolute, skipping: %s -> %s",
                            mount.host_path,
                            mount.container_path,
                        )
                        continue

                    # Reject mounts that conflict with reserved container paths
                    # 拒绝与保留容器路径冲突的挂载（安全审计：防止覆盖虚拟路径前缀）
                    if any(container_path == p or container_path.startswith(p + "/") for p in _RESERVED_CONTAINER_PREFIXES):
                        logger.warning(
                            "Mount container_path conflicts with reserved prefix, skipping: %s",
                            mount.container_path,
                        )
                        continue
                    # Ensure the host path exists before adding mapping
                    # 确保宿主路径存在后再添加映射
                    if host_path.exists():
                        mappings.append(
                            PathMapping(
                                container_path=container_path,
                                local_path=str(host_path.resolve()),
                                read_only=mount.read_only,
                            )
                        )
                    else:
                        logger.warning(
                            "Mount host_path does not exist, skipping: %s -> %s",
                            mount.host_path,
                            mount.container_path,
                        )
        except Exception as e:
            # Log but don't fail if config loading fails
            logger.warning("Could not setup path mappings: %s", e, exc_info=True)

        return mappings

    @staticmethod
    def _build_thread_path_mappings(thread_id: str) -> list[PathMapping]:
        """Build per-thread path mappings for /mnt/user-data and /mnt/acp-workspace.

        Resolves ``user_id`` via :func:`get_effective_user_id` (the same path
        :class:`AioSandboxProvider` uses) and ensures the backing host
        directories exist before they are mapped into the sandbox view.
        """
        from deerflow.config.paths import get_paths
        from deerflow.runtime.user_context import get_effective_user_id

        paths = get_paths()
        user_id = get_effective_user_id()
        paths.ensure_thread_dirs(thread_id, user_id=user_id)

        return [
            # Aggregate parent mapping so ``ls /mnt/user-data`` and other
            # parent-level operations behave the same as inside AIO (where the
            # parent directory is real and contains the three subdirs). Longer
            # subpath mappings below still win for ``/mnt/user-data/workspace/...``
            # because ``_find_path_mapping`` sorts by container_path length.
            PathMapping(
                container_path=_USER_DATA_VIRTUAL_PREFIX,
                local_path=str(paths.sandbox_user_data_dir(thread_id, user_id=user_id)),
                read_only=False,
            ),
            PathMapping(
                container_path=f"{_USER_DATA_VIRTUAL_PREFIX}/workspace",
                local_path=str(paths.sandbox_work_dir(thread_id, user_id=user_id)),
                read_only=False,
            ),
            PathMapping(
                container_path=f"{_USER_DATA_VIRTUAL_PREFIX}/uploads",
                local_path=str(paths.sandbox_uploads_dir(thread_id, user_id=user_id)),
                read_only=False,
            ),
            PathMapping(
                container_path=f"{_USER_DATA_VIRTUAL_PREFIX}/outputs",
                local_path=str(paths.sandbox_outputs_dir(thread_id, user_id=user_id)),
                read_only=False,
            ),
            PathMapping(
                container_path=_ACP_WORKSPACE_VIRTUAL_PREFIX,
                local_path=str(paths.acp_workspace_dir(thread_id, user_id=user_id)),
                read_only=False,
            ),
        ]

    def acquire(self, thread_id: str | None = None) -> str:
        """Return a sandbox id scoped to *thread_id* (or the generic singleton).

        - ``thread_id=None`` keeps the legacy singleton with id ``"local"`` for
          callers that have no thread context (e.g. legacy tests, scripts).
        - ``thread_id="abc"`` yields a per-thread ``LocalSandbox`` with id
          ``"local:abc"`` whose ``path_mappings`` resolve ``/mnt/user-data/...``
          to that thread's host directories.

        Thread-safe under concurrent invocation: the cache check + insert is
        guarded by ``self._lock`` so two callers racing on the same
        ``thread_id`` always observe the same LocalSandbox instance.
        """
        global _singleton

        if thread_id is None:
            with self._lock:
                if self._generic_sandbox is None:
                    self._generic_sandbox = LocalSandbox("local", path_mappings=list(self._path_mappings))
                    _singleton = self._generic_sandbox
                return self._generic_sandbox.id

        # Fast path under lock.
        with self._lock:
            cached = self._thread_sandboxes.get(thread_id)
            if cached is not None:
                # Mark as most-recently used so frequently-touched threads
                # survive eviction.
                self._thread_sandboxes.move_to_end(thread_id)
                return cached.id

        # ``_build_thread_path_mappings`` touches the filesystem
        # (``ensure_thread_dirs``); release the lock during I/O.
        new_mappings = list(self._path_mappings) + self._build_thread_path_mappings(thread_id)

        with self._lock:
            # Re-check after the lock-free I/O: another caller may have
            # populated the cache while we were computing mappings.
            cached = self._thread_sandboxes.get(thread_id)
            if cached is None:
                cached = LocalSandbox(f"local:{thread_id}", path_mappings=new_mappings)
                self._thread_sandboxes[thread_id] = cached
                self._evict_until_within_cap_locked()
            else:
                self._thread_sandboxes.move_to_end(thread_id)
            return cached.id

    def _evict_until_within_cap_locked(self) -> None:
        """LRU-evict cached thread sandboxes once the cap is exceeded.

        Caller MUST hold ``self._lock``.
        """
        while len(self._thread_sandboxes) > self._max_cached_threads:
            evicted_thread_id, _ = self._thread_sandboxes.popitem(last=False)
            logger.info(
                "Evicting LocalSandbox cache entry for thread %s (cap=%d)",
                evicted_thread_id,
                self._max_cached_threads,
            )

    def get(self, sandbox_id: str) -> Sandbox | None:
        if sandbox_id == "local":
            with self._lock:
                generic = self._generic_sandbox
            if generic is None:
                self.acquire()
                with self._lock:
                    return self._generic_sandbox
            return generic
        if isinstance(sandbox_id, str) and sandbox_id.startswith("local:"):
            thread_id = sandbox_id[len("local:") :]
            with self._lock:
                cached = self._thread_sandboxes.get(thread_id)
                if cached is not None:
                    # Touching a thread via ``get`` (used by tools.py to look
                    # up the sandbox once per tool call) promotes it in LRU
                    # order so an active thread isn't evicted under load.
                    self._thread_sandboxes.move_to_end(thread_id)
                return cached
        return None

    def release(self, sandbox_id: str) -> None:
        # LocalSandbox has no resources to release; keep the cached instance so
        # that ``_agent_written_paths`` (used to reverse-resolve agent-authored
        # file contents on read) survives between turns. LRU eviction in
        # ``acquire`` and explicit ``reset()`` / ``shutdown()`` are the only
        # paths that drop cached entries.
        #
        # Note: This method is intentionally not called by SandboxMiddleware
        # to allow sandbox reuse across multiple turns in a thread.
        pass

    def reset(self) -> None:
        """Drop all cached LocalSandbox instances.

        ``reset_sandbox_provider()`` calls this to ensure config / mount
        changes take effect on the next ``acquire()``. We also reset the
        module-level ``_singleton`` alias so older callers/tests that reach
        into it see a fresh state.
        """
        global _singleton
        with self._lock:
            self._generic_sandbox = None
            self._thread_sandboxes.clear()
            _singleton = None

    def shutdown(self) -> None:
        # LocalSandboxProvider has no extra resources beyond the cached
        # ``LocalSandbox`` instances, so shutdown uses the same cleanup path
        # as ``reset``.
        self.reset()
