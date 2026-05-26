"""文件操作锁模块 — 为沙箱中的文件读写操作提供并发控制。

本模块通过细粒度的锁机制，防止同一沙箱中对同一文件路径的并发操作
（如 str_replace 和 write_file 同时操作同一文件）导致数据竞争。
使用 WeakValueDictionary 管理锁的生命周期，避免长时间运行进程中的内存泄漏。

锁的粒度为 (sandbox_id, path)，即同一沙箱内同一文件路径共享一把锁，
不同沙箱或不同文件路径互不阻塞。
"""

import threading
import weakref

from deerflow.sandbox.sandbox import Sandbox

# Use WeakValueDictionary to prevent memory leak in long-running processes.
# Locks are automatically removed when no longer referenced by any thread.
# 使用 WeakValueDictionary 防止长时间运行进程中的内存泄漏。
# 当锁不再被任何线程引用时，会自动被垃圾回收。
_LockKey = tuple[str, str]
_FILE_OPERATION_LOCKS: weakref.WeakValueDictionary[_LockKey, threading.Lock] = weakref.WeakValueDictionary()
# 保护 _FILE_OPERATION_LOCKS 字典本身并发访问的全局守卫锁
_FILE_OPERATION_LOCKS_GUARD = threading.Lock()


def get_file_operation_lock_key(sandbox: Sandbox, path: str) -> tuple[str, str]:
    """生成文件操作锁的键 — 由沙箱 ID 和文件路径组成。

    确保不同沙箱或不同文件路径使用不同的锁，实现细粒度并发控制。
    同一沙箱内同一文件路径共享同一把锁，避免数据竞争。

    Args:
        sandbox: 沙箱实例，用于获取沙箱标识符。
        path: 文件路径。

    Returns:
        由 (sandbox_id, path) 组成的元组，作为锁的键。
    """
    sandbox_id = getattr(sandbox, "id", None)
    if not sandbox_id:
        # 沙箱无 ID 时使用对象内存地址作为兜底标识
        sandbox_id = f"instance:{id(sandbox)}"
    return sandbox_id, path


def get_file_operation_lock(sandbox: Sandbox, path: str) -> threading.Lock:
    """获取指定沙箱和文件路径对应的操作锁。

    使用双重检查模式：先查找已有锁，不存在时再加锁创建新锁。
    确保并发场景下同一 (sandbox_id, path) 始终返回同一个 Lock 实例。

    Args:
        sandbox: 沙箱实例。
        path: 文件路径。

    Returns:
        与 (sandbox_id, path) 关联的 threading.Lock 实例。
    """
    lock_key = get_file_operation_lock_key(sandbox, path)
    with _FILE_OPERATION_LOCKS_GUARD:
        lock = _FILE_OPERATION_LOCKS.get(lock_key)
        if lock is None:
            lock = threading.Lock()
            _FILE_OPERATION_LOCKS[lock_key] = lock
        return lock
