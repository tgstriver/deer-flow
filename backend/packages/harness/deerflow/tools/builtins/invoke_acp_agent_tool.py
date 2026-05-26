"""Built-in tool for invoking external ACP-compatible agents.
ACP (Agent Client Protocol) 外部代理调用工具

本模块实现了 invoke_acp_agent 工具，用于调用外部兼容 ACP 协议的 Agent。
ACP 是一种标准化的 Agent 通信协议，允许 DeerFlow 与外部 Agent 进程交互。
主要功能包括：
- 构建 ACP 客户端并管理会话
- 处理权限请求的自动审批/拒绝
- 管理每个线程的独立工作空间
- 将 MCP 服务器配置转换为 ACP 格式
"""

import logging
import os
import shutil
from typing import Annotated, Any

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, InjectedToolArg, StructuredTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class _InvokeACPAgentInput(BaseModel):
    """ACP Agent 调用工具的输入参数模型。

    Attributes:
        agent: 要调用的 ACP Agent 名称。
        prompt: 发送给 Agent 的简洁任务提示。
    """

    agent: str = Field(description="Name of the ACP agent to invoke")
    prompt: str = Field(description="The concise task prompt to send to the agent")


def _get_work_dir(thread_id: str | None) -> str:
    """Get the per-thread ACP workspace directory.
    获取每个线程的 ACP 工作空间目录。

    Each thread gets an isolated workspace under
    ``{base_dir}/threads/{thread_id}/acp-workspace/`` so that concurrent
    sessions cannot read or overwrite each other's ACP agent outputs.
    每个线程在上述路径下拥有独立的工作空间，以防止并发会话互相读写对方的 ACP Agent 输出。

    Falls back to the legacy global ``{base_dir}/acp-workspace/`` when
    ``thread_id`` is not available (e.g. embedded / direct invocation).
    当 thread_id 不可用时（如嵌入式/直接调用），回退到全局工作空间。

    The directory is created automatically if it does not exist.
    如果目录不存在则自动创建。

    Args:
        thread_id: 线程 ID，可为 None。

    Returns:
        An absolute physical filesystem path to use as the working directory.
        用作工作目录的绝对物理文件系统路径。
    """
    from deerflow.config.paths import get_paths
    from deerflow.runtime.user_context import get_effective_user_id

    paths = get_paths()
    if thread_id:
        try:
            work_dir = paths.acp_workspace_dir(thread_id, user_id=get_effective_user_id())
        except ValueError:
            logger.warning("Invalid thread_id %r for ACP workspace, falling back to global", thread_id)
            work_dir = paths.base_dir / "acp-workspace"
    else:
        work_dir = paths.base_dir / "acp-workspace"

    work_dir.mkdir(parents=True, exist_ok=True)
    logger.info("ACP agent work_dir: %s", work_dir)
    return str(work_dir)


def _build_mcp_servers() -> dict[str, dict[str, Any]]:
    """Build ACP ``mcpServers`` config from DeerFlow's enabled MCP servers.
    从 DeerFlow 启用的 MCP 服务器构建 ACP mcpServers 配置。
    """
    from deerflow.config.extensions_config import ExtensionsConfig
    from deerflow.mcp.client import build_servers_config

    return build_servers_config(ExtensionsConfig.from_file())


def _build_acp_mcp_servers() -> list[dict[str, Any]]:
    """Build ACP ``mcpServers`` payload for ``new_session``.
    为 new_session 构建 ACP mcpServers 负载。

    The ACP client expects a list of server objects, while DeerFlow's MCP helper
    returns a name -> config mapping for the LangChain MCP adapter. This helper
    converts the enabled servers into the ACP wire format.
    ACP 客户端期望服务器对象列表，而 DeerFlow 的 MCP 辅助函数返回名称到配置的映射。
    此函数将启用的服务器转换为 ACP 传输格式。
    """
    from deerflow.config.extensions_config import ExtensionsConfig

    extensions_config = ExtensionsConfig.from_file()
    enabled_servers = extensions_config.get_enabled_mcp_servers()

    mcp_servers: list[dict[str, Any]] = []
    for name, server_config in enabled_servers.items():
        # 遍历所有已启用的 MCP 服务器，转换为 ACP 格式
        transport_type = server_config.type or "stdio"
        payload: dict[str, Any] = {"name": name, "type": transport_type}

        if transport_type == "stdio":
            # stdio 传输类型：需要 command 字段
            if not server_config.command:
                raise ValueError(f"MCP server '{name}' with stdio transport requires 'command' field")
            payload["command"] = server_config.command
            payload["args"] = server_config.args
            payload["env"] = [{"name": key, "value": value} for key, value in server_config.env.items()]
        elif transport_type in ("http", "sse"):
            # HTTP/SSE 传输类型：需要 url 字段
            if not server_config.url:
                raise ValueError(f"MCP server '{name}' with {transport_type} transport requires 'url' field")
            payload["url"] = server_config.url
            payload["headers"] = [{"name": key, "value": value} for key, value in server_config.headers.items()]
        else:
            raise ValueError(f"MCP server '{name}' has unsupported transport type: {transport_type}")

        mcp_servers.append(payload)

    return mcp_servers


def _build_permission_response(options: list[Any], *, auto_approve: bool) -> Any:
    """Build an ACP permission response.
    构建 ACP 权限响应。

    When ``auto_approve`` is True, selects the first ``allow_once`` (preferred)
    or ``allow_always`` option.  When False (the default), always cancels —
    permission requests must be handled by the ACP agent's own policy or the
    agent must be configured to operate without requesting permissions.
    当 auto_approve 为 True 时，选择第一个 allow_once（优先）或 allow_always 选项。
    当为 False（默认）时，始终取消——权限请求必须由 ACP Agent 自身的策略处理，
    或者 Agent 必须配置为不请求权限。
    """
    from acp import RequestPermissionResponse
    from acp.schema import AllowedOutcome, DeniedOutcome

    if auto_approve:
        for preferred_kind in ("allow_once", "allow_always"):
            for option in options:
                if getattr(option, "kind", None) != preferred_kind:
                    continue

                option_id = getattr(option, "option_id", None)
                if option_id is None:
                    option_id = getattr(option, "optionId", None)
                if option_id is None:
                    continue

                return RequestPermissionResponse(
                    outcome=AllowedOutcome(outcome="selected", optionId=option_id),
                )

    return RequestPermissionResponse(outcome=DeniedOutcome(outcome="cancelled"))


def _format_invocation_error(agent: str, cmd: str, exc: Exception) -> str:
    """Return a user-facing ACP invocation error with actionable remediation.
    返回面向用户的 ACP 调用错误信息，附带可操作的修复建议。
    """
    if not isinstance(exc, FileNotFoundError):
        return f"Error invoking ACP agent '{agent}': {exc}"

    message = f"Error invoking ACP agent '{agent}': Command '{cmd}' was not found on PATH."
    if cmd == "codex-acp" and shutil.which("codex"):
        return f"{message} The installed `codex` CLI does not speak ACP directly. Install a Codex ACP adapter (for example `npx @zed-industries/codex-acp`) or update `acp_agents.codex.command` and `args` in config.yaml."

    return f"{message} Install the agent binary or update `acp_agents.{agent}.command` in config.yaml."


def build_invoke_acp_agent_tool(agents: dict) -> BaseTool:
    """Create the ``invoke_acp_agent`` tool with a description generated from configured agents.
    根据配置的 Agent 列表创建 invoke_acp_agent 工具。

    The tool description includes the list of available agents so that the LLM
    knows which agents it can invoke without requiring hardcoded names.
    工具描述包含可用 Agent 列表，使 LLM 知道可以调用哪些 Agent，无需硬编码名称。

    Args:
        agents: Mapping of agent name -> ``ACPAgentConfig``.
                Agent 名称到 ACPAgentConfig 的映射。

    Returns:
        A LangChain ``BaseTool`` ready to be included in the tool list.
        可直接加入工具列表的 LangChain BaseTool。
    """
    agent_lines = "\n".join(f"- {name}: {cfg.description}" for name, cfg in agents.items())
    description = (
        "Invoke an external ACP-compatible agent and return its final response.\n\n"
        "Available agents:\n"
        f"{agent_lines}\n\n"
        "IMPORTANT: ACP agents operate in their own independent workspace. "
        "Do NOT include /mnt/user-data paths in the prompt. "
        "Give the agent a self-contained task description — it will produce results in its own workspace. "
        "After the agent completes, its output files are accessible at /mnt/acp-workspace/ (read-only)."
    )

    # Capture agents in closure so the function can reference it / 将 agents 捕获到闭包中，以便函数可以引用
    _agents = dict(agents)

    async def _invoke(agent: str, prompt: str, config: Annotated[RunnableConfig, InjectedToolArg] = None) -> str:
        """异步调用 ACP Agent 的核心实现。

        Args:
            agent: 要调用的 Agent 名称。
            prompt: 发送给 Agent 的任务提示。
            config: 运行时配置（自动注入）。

        Returns:
            Agent 的最终响应文本，或错误消息。
        """
        logger.info("Invoking ACP agent %s (prompt length: %d)", agent, len(prompt))
        logger.debug("Invoking ACP agent %s with prompt: %.200s%s", agent, prompt, "..." if len(prompt) > 200 else "")
        if agent not in _agents:
            available = ", ".join(_agents.keys())
            return f"Error: Unknown agent '{agent}'. Available: {available}"

        agent_config = _agents[agent]
        thread_id: str | None = ((config or {}).get("configurable") or {}).get("thread_id")

        try:
            from acp import PROTOCOL_VERSION, Client, text_block
            from acp.schema import ClientCapabilities, Implementation
        except ImportError:
            return "Error: agent-client-protocol package is not installed. Run `uv sync` to install project dependencies."

        class _CollectingClient(Client):
            """Minimal ACP Client that collects streamed text from session updates.
            最小化的 ACP 客户端，收集会话更新中的流式文本。
            """

            def __init__(self) -> None:
                self._chunks: list[str] = []  # 收集到的文本片段列表

            @property
            def collected_text(self) -> str:
                """返回已收集的所有文本片段拼接结果。"""
                return "".join(self._chunks)

            async def session_update(self, session_id: str, update, **kwargs) -> None:  # type: ignore[override]
                """处理 ACP 会话更新，收集文本内容块。"""
                try:
                    from acp.schema import TextContentBlock

                    if hasattr(update, "content") and isinstance(update.content, TextContentBlock):
                        self._chunks.append(update.content.text)
                except Exception:
                    pass

            async def request_permission(self, options, session_id: str, tool_call, **kwargs):  # type: ignore[override]
                """处理 ACP 权限请求，根据配置自动审批或拒绝。"""
                response = _build_permission_response(options, auto_approve=agent_config.auto_approve_permissions)
                outcome = response.outcome.outcome
                if outcome == "selected":
                    logger.info("ACP permission auto-approved for tool call %s in session %s", tool_call.tool_call_id, session_id)
                else:
                    logger.warning("ACP permission denied for tool call %s in session %s (set auto_approve_permissions: true in config.yaml to enable)", tool_call.tool_call_id, session_id)
                return response

        client = _CollectingClient()
        cmd = agent_config.command  # Agent 的启动命令
        args = agent_config.args or []  # Agent 的启动参数
        physical_cwd = _get_work_dir(thread_id)  # 获取工作目录
        try:
            mcp_servers = _build_acp_mcp_servers()
        except ValueError as exc:
            logger.warning(
                "Invalid MCP server configuration for ACP agent '%s'; continuing without MCP servers: %s",
                agent,
                exc,
            )
            mcp_servers = []
        agent_env: dict[str, str] | None = None
        if agent_config.env:
            # 解析环境变量：以 $ 开头的值从系统环境变量中获取
            agent_env = {k: (os.environ.get(v[1:], "") if v.startswith("$") else v) for k, v in agent_config.env.items()}

        try:
            from acp import spawn_agent_process

            async with spawn_agent_process(client, cmd, *args, env=agent_env, cwd=physical_cwd) as (conn, proc):
                # 启动 ACP Agent 进程并初始化连接
                logger.info("Spawning ACP agent '%s' with command '%s' and args %s in cwd %s", agent, cmd, args, physical_cwd)
                await conn.initialize(
                    protocol_version=PROTOCOL_VERSION,
                    client_capabilities=ClientCapabilities(),
                    client_info=Implementation(name="deerflow", title="DeerFlow", version="0.1.0"),
                )
                session_kwargs: dict[str, Any] = {"cwd": physical_cwd, "mcp_servers": mcp_servers}
                if agent_config.model:
                    session_kwargs["model"] = agent_config.model  # 如果配置了模型，传入模型名称
                session = await conn.new_session(**session_kwargs)
                await conn.prompt(
                    session_id=session.session_id,
                    prompt=[text_block(prompt)],
                )  # 向 Agent 发送提示
            result = client.collected_text  # 获取收集到的响应文本
            logger.info("ACP agent '%s' returned %s", agent, result[:1000])
            logger.info("ACP agent '%s' returned %d characters", agent, len(result))
            return result or "(no response)"
        except Exception as e:
            logger.error("ACP agent '%s' invocation failed: %s", agent, e)
            return _format_invocation_error(agent, cmd, e)

    return StructuredTool.from_function(
        name="invoke_acp_agent",
        description=description,
        coroutine=_invoke,
        args_schema=_InvokeACPAgentInput,
    )
