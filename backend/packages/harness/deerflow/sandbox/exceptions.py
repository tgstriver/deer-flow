"""沙箱相关异常定义 — 提供结构化错误信息的异常层次体系。

本模块定义了沙箱操作中可能抛出的所有异常类型，采用层次化继承结构，
每类异常都携带结构化的详情信息（details），便于错误诊断和日志审计。

异常层次：
- SandboxError（基类）
  - SandboxNotFoundError（沙箱未找到）
  - SandboxRuntimeError（沙箱运行时不可用）
  - SandboxCommandError（命令执行失败）
  - SandboxFileError（文件操作失败）
    - SandboxPermissionError（权限不足）
    - SandboxFileNotFoundError（文件/目录不存在）
"""


class SandboxError(Exception):
    """沙箱异常基类 — 所有沙箱相关错误的公共父类。

    携带结构化详情信息，便于日志记录和错误诊断。

    Args:
        message: 错误描述信息。
        details: 可选的结构化详情字典，包含与错误相关的上下文数据。

    Attributes:
        message: 错误描述信息。
        details: 结构化详情字典。
    """

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        return self.message


class SandboxNotFoundError(SandboxError):
    """沙箱未找到异常 — 当请求的沙箱不存在或不可用时抛出。

    通常在通过 sandbox_id 查找沙箱实例失败时触发。

    Args:
        message: 错误描述信息，默认为 "Sandbox not found"。
        sandbox_id: 未找到的沙箱标识符。

    Attributes:
        sandbox_id: 未找到的沙箱标识符。
    """

    def __init__(self, message: str = "Sandbox not found", sandbox_id: str | None = None):
        details = {"sandbox_id": sandbox_id} if sandbox_id else None
        super().__init__(message, details)
        self.sandbox_id = sandbox_id


class SandboxRuntimeError(SandboxError):
    """沙箱运行时异常 — 当沙箱运行时不可用或配置错误时抛出。/ Raised when sandbox runtime is not available or misconfigured.

    例如：线程数据缺失、沙箱提供者初始化失败等场景。
    """

    pass


class SandboxCommandError(SandboxError):
    """沙箱命令执行异常 — 当在沙箱中执行命令失败时抛出。

    携带执行的命令内容和退出码信息，命令文本超过 100 字符时自动截断。

    Args:
        message: 错误描述信息。
        command: 执行失败的命令文本（超过 100 字符将被截断）。
        exit_code: 命令的退出码。

    Attributes:
        command: 执行失败的命令文本（完整版，未截断）。
        exit_code: 命令的退出码。
    """

    def __init__(self, message: str, command: str | None = None, exit_code: int | None = None):
        details = {}
        if command:
            # 命令文本超过 100 字符时截断，避免详情信息过长
            details["command"] = command[:100] + "..." if len(command) > 100 else command
        if exit_code is not None:
            details["exit_code"] = exit_code
        super().__init__(message, details)
        self.command = command
        self.exit_code = exit_code


class SandboxFileError(SandboxError):
    """沙箱文件操作异常 — 当在沙箱中执行文件操作失败时抛出。

    携带文件路径和操作类型信息，便于定位问题。

    Args:
        message: 错误描述信息。
        path: 操作涉及的文件路径。
        operation: 操作类型（如 "read"、"write"、"list" 等）。

    Attributes:
        path: 操作涉及的文件路径。
        operation: 操作类型。
    """

    def __init__(self, message: str, path: str | None = None, operation: str | None = None):
        details = {}
        if path:
            details["path"] = path
        if operation:
            details["operation"] = operation
        super().__init__(message, details)
        self.path = path
        self.operation = operation


class SandboxPermissionError(SandboxFileError):
    """沙箱权限异常 — 当文件操作因权限不足被拒绝时抛出。

    继承自 SandboxFileError，用于区分"权限拒绝"和"文件不存在"等不同错误类型。
    常见于：对只读挂载路径执行写操作、路径遍历攻击被拦截等安全审计场景。
    """

    pass


class SandboxFileNotFoundError(SandboxFileError):
    """沙箱文件未找到异常 — 当请求的文件或目录不存在时抛出。

    继承自 SandboxFileError，用于明确标识"路径不存在"这一特定错误场景。
    """

    pass
