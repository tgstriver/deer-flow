"""凭证自动加载模块，从 Claude Code CLI 和 Codex CLI 加载认证凭证。

本模块实现了两种凭证加载策略：
  1. Claude Code OAuth 令牌：从显式环境变量或导出的凭证文件加载
     - 使用 Authorization: Bearer 请求头（而非 x-api-key）
     - 需要 anthropic-beta: oauth-2025-04-20,claude-code-20250219
     - 支持 $CLAUDE_CODE_OAUTH_TOKEN、$CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR 和 $ANTHROPIC_AUTH_TOKEN
     - 可通过 $CLAUDE_CODE_CREDENTIALS_PATH 覆盖默认路径
  2. Codex CLI 令牌：从 ~/.codex/auth.json 加载
     - 使用 chatgpt.com/backend-api/codex/responses 端点
     - 支持旧版顶层令牌格式和当前嵌套令牌格式
     - 可通过 $CODEX_AUTH_PATH 覆盖默认路径
"""

import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Claude Code OAuth 令牌所需的 beta 请求头
OAUTH_ANTHROPIC_BETAS = "oauth-2025-04-20,claude-code-20250219,interleaved-thinking-2025-05-14"


def is_oauth_token(token: str) -> bool:
    """检查令牌是否为 Claude Code OAuth 令牌（而非标准 API 密钥）。

    通过检测令牌中是否包含 "sk-ant-oat" 前缀来判断。

    Args:
        token: 待检查的令牌字符串

    Returns:
        bool: 若为 OAuth 令牌返回 True，否则返回 False
    """
    return isinstance(token, str) and "sk-ant-oat" in token


@dataclass
class ClaudeCodeCredential:
    """Claude Code CLI OAuth 凭证数据类。

    存储从 Claude Code CLI 获取的 OAuth 凭证信息，包括访问令牌、
    刷新令牌、过期时间和来源标识。

    Attributes:
        access_token: OAuth 访问令牌（sk-ant-oat 前缀）
        refresh_token: OAuth 刷新令牌
        expires_at: 过期时间戳（毫秒级 Unix 时间戳）
        source: 凭证来源标识（如 "claude-cli-env"、"claude-cli-fd"、"claude-cli-file"）
    """

    access_token: str
    refresh_token: str = ""
    expires_at: int = 0
    source: str = ""

    @property
    def is_expired(self) -> bool:
        """判断 OAuth 令牌是否已过期。

        在实际过期时间前 1 分钟视为已过期，提供安全缓冲。

        Returns:
            bool: 若令牌已过期返回 True，否则返回 False
        """
        if self.expires_at <= 0:
            return False
        return time.time() * 1000 > self.expires_at - 60_000  # 1 分钟缓冲


@dataclass
class CodexCliCredential:
    """Codex CLI 凭证数据类。

    存储从 Codex CLI 获取的认证凭证，用于访问 ChatGPT Codex Responses API。

    Attributes:
        access_token: Codex 访问令牌
        account_id: 关联的 ChatGPT 账户 ID
        source: 凭证来源标识（如 "codex-cli"）
    """

    access_token: str
    account_id: str = ""
    source: str = ""


def _resolve_credential_path(env_var: str, default_relative_path: str) -> Path:
    """解析凭证文件路径，优先使用环境变量指定的路径。

    Args:
        env_var: 环境变量名，用于覆盖默认路径
        default_relative_path: 相对于用户主目录的默认路径

    Returns:
        Path: 解析后的凭证文件路径
    """
    configured_path = os.getenv(env_var)
    if configured_path:
        return Path(configured_path).expanduser()
    return _home_dir() / default_relative_path


def _home_dir() -> Path:
    """获取用户主目录路径。

    优先使用 HOME 环境变量，否则使用系统默认主目录。

    Returns:
        Path: 用户主目录路径
    """
    home = os.getenv("HOME")
    if home:
        return Path(home).expanduser()
    return Path.home()


def _load_json_file(path: Path, label: str) -> dict[str, Any] | None:
    """加载并解析 JSON 文件。

    Args:
        path: JSON 文件路径
        label: 用于日志标识的文件描述标签

    Returns:
        dict | None: 解析后的字典，若文件不存在或解析失败则返回 None
    """
    if not path.exists():
        logger.debug(f"{label} not found: {path}")
        return None
    if path.is_dir():
        logger.warning(f"{label} path is a directory, expected a file: {path}")
        return None

    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"Failed to read {label}: {e}")
        return None


def _read_secret_from_file_descriptor(env_var: str) -> str | None:
    """从文件描述符读取秘密值。

    某些安全场景下，令牌通过文件描述符传递而非环境变量明文。

    Args:
        env_var: 存储文件描述符数值的环境变量名

    Returns:
        str | None: 读取到的秘密值，失败则返回 None
    """
    fd_value = os.getenv(env_var)
    if not fd_value:
        return None

    try:
        fd = int(fd_value)
    except ValueError:
        logger.warning(f"{env_var} must be an integer file descriptor, got: {fd_value}")
        return None

    try:
        secret = os.read(fd, 1024 * 1024).decode().strip()
    except OSError as e:
        logger.warning(f"Failed to read {env_var}: {e}")
        return None

    return secret or None


def _credential_from_direct_token(access_token: str, source: str) -> ClaudeCodeCredential | None:
    """从直接的令牌字符串创建凭证对象。

    Args:
        access_token: 访问令牌字符串
        source: 凭证来源标识

    Returns:
        ClaudeCodeCredential | None: 凭证对象，令牌为空则返回 None
    """
    token = access_token.strip()
    if not token:
        return None
    return ClaudeCodeCredential(access_token=token, source=source)


def _iter_claude_code_credential_paths() -> list[Path]:
    """迭代 Claude Code 凭证文件的候选路径。

    优先使用环境变量指定的路径，其次使用默认路径。

    Returns:
        list[Path]: 凭证文件候选路径列表
    """
    paths: list[Path] = []
    override_path = os.getenv("CLAUDE_CODE_CREDENTIALS_PATH")
    if override_path:
        paths.append(Path(override_path).expanduser())

    default_path = _home_dir() / ".claude/.credentials.json"
    if not paths or paths[-1] != default_path:
        paths.append(default_path)

    return paths


def _extract_claude_code_credential(data: dict[str, Any], source: str) -> ClaudeCodeCredential | None:
    """从 JSON 数据中提取 Claude Code OAuth 凭证。

    Args:
        data: 从凭证文件解析的字典数据
        source: 凭证来源标识

    Returns:
        ClaudeCodeCredential | None: 凭证对象，若无有效令牌或已过期则返回 None
    """
    oauth = data.get("claudeAiOauth", {})
    access_token = oauth.get("accessToken", "")
    if not access_token:
        logger.debug("Claude Code credentials container exists but no accessToken found")
        return None

    cred = ClaudeCodeCredential(
        access_token=access_token,
        refresh_token=oauth.get("refreshToken", ""),
        expires_at=oauth.get("expiresAt", 0),
        source=source,
    )

    if cred.is_expired:
        logger.warning("Claude Code OAuth token is expired. Run 'claude' to refresh.")
        return None

    return cred


def load_claude_code_credential() -> ClaudeCodeCredential | None:
    """从 Claude Code 显式交接源加载 OAuth 凭证。

    按以下优先级顺序查找凭证：
      1. $CLAUDE_CODE_OAUTH_TOKEN 或 $ANTHROPIC_AUTH_TOKEN 环境变量
      2. $CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR 文件描述符
      3. $CLAUDE_CODE_CREDENTIALS_PATH 指定的凭证文件
      4. ~/.claude/.credentials.json 默认凭证文件

    导出的凭证文件格式：
    {
      "claudeAiOauth": {
        "accessToken": "sk-ant-oat01-...",
        "refreshToken": "sk-ant-ort01-...",
        "expiresAt": 1773430695128,
        "scopes": ["user:inference", ...],
        ...
      }
    }

    Returns:
        ClaudeCodeCredential | None: 成功加载返回凭证对象，否则返回 None
    """
    direct_token = os.getenv("CLAUDE_CODE_OAUTH_TOKEN") or os.getenv("ANTHROPIC_AUTH_TOKEN")
    if direct_token:
        cred = _credential_from_direct_token(direct_token, "claude-cli-env")
        if cred:
            logger.info("Loaded Claude Code OAuth credential from environment")
        return cred

    fd_token = _read_secret_from_file_descriptor("CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR")
    if fd_token:
        cred = _credential_from_direct_token(fd_token, "claude-cli-fd")
        if cred:
            logger.info("Loaded Claude Code OAuth credential from file descriptor")
        return cred

    override_path = os.getenv("CLAUDE_CODE_CREDENTIALS_PATH")
    override_path_obj = Path(override_path).expanduser() if override_path else None
    for cred_path in _iter_claude_code_credential_paths():
        data = _load_json_file(cred_path, "Claude Code credentials")
        if data is None:
            continue
        cred = _extract_claude_code_credential(data, "claude-cli-file")
        if cred:
            source_label = "override path" if override_path_obj is not None and cred_path == override_path_obj else "plaintext file"
            logger.info(f"Loaded Claude Code OAuth credential from {source_label} (expires_at={cred.expires_at})")
            return cred

    return None


def load_codex_cli_credential() -> CodexCliCredential | None:
    """从 Codex CLI 加载凭证（~/.codex/auth.json）。

    支持旧版顶层令牌格式和当前嵌套令牌格式：
    - 旧版：{"access_token": "...", "account_id": "..."}
    - 新版：{"tokens": {"access_token": "...", "account_id": "..."}}

    Returns:
        CodexCliCredential | None: 成功加载返回凭证对象，否则返回 None
    """
    cred_path = _resolve_credential_path("CODEX_AUTH_PATH", ".codex/auth.json")
    data = _load_json_file(cred_path, "Codex CLI credentials")
    if data is None:
        return None
    tokens = data.get("tokens", {})
    if not isinstance(tokens, dict):
        tokens = {}

    access_token = data.get("access_token") or data.get("token") or tokens.get("access_token", "")
    account_id = data.get("account_id") or tokens.get("account_id", "")
    if not access_token:
        logger.debug("Codex CLI credentials file exists but no token found")
        return None

    logger.info("Loaded Codex CLI credential")
    return CodexCliCredential(
        access_token=access_token,
        account_id=account_id,
        source="codex-cli",
    )
