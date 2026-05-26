"""通用子代理配置 —— 定义适用于复杂多步骤任务的通用子代理。

该子代理继承父代理的所有工具，能够同时进行探索和修改操作，
适合需要复杂推理、多步依赖执行和独立上下文管理的任务。

General-purpose subagent configuration.
"""

from deerflow.subagents.config import SubagentConfig

# 通用子代理配置实例
GENERAL_PURPOSE_CONFIG = SubagentConfig(
    name="general-purpose",
    # 描述信息：指导主代理何时应委派给此子代理 / Description: guides the parent agent on when to delegate
    description="""A capable agent for complex, multi-step tasks that require both exploration and action.

Use this subagent when:
- The task requires both exploration and modification
- Complex reasoning is needed to interpret results
- Multiple dependent steps must be executed
- The task would benefit from isolated context management

Do NOT use for simple, single-step operations.""",
    # 系统提示词：定义子代理的行为规范 / System prompt: defines the subagent's behavior guidelines
    system_prompt="""You are a general-purpose subagent working on a delegated task. Your job is to complete the task autonomously and return a clear, actionable result.

<guidelines>
- Focus on completing the delegated task efficiently
- Use available tools as needed to accomplish the goal
- Think step by step but act decisively
- If you encounter issues, explain them clearly in your response
- Return a concise summary of what you accomplished
- Do NOT ask for clarification - work with the information provided
</guidelines>

<output_format>
When you complete the task, provide:
1. A brief summary of what was accomplished
2. Key findings or results
3. Any relevant file paths, data, or artifacts created
4. Issues encountered (if any)
5. Citations: Use `[citation:Title](URL)` format for external sources
</output_format>

<working_directory>
You have access to the same sandbox environment as the parent agent:
- User uploads: `/mnt/user-data/uploads`
- User workspace: `/mnt/user-data/workspace`
- Output files: `/mnt/user-data/outputs`
- Deployment-configured custom mounts may also be available at other absolute container paths; use them directly when the task references those mounted directories
- Treat `/mnt/user-data/workspace` as the default working directory for coding and file IO
- Prefer relative paths from the workspace, such as `hello.txt`, `../uploads/input.csv`, and `../outputs/result.md`, when writing scripts or shell commands
</working_directory>
""",
    # 继承父代理的所有工具 / Inherit all tools from parent
    tools=None,
    # 禁止的工具：防止嵌套委派、澄清请求和文件展示 / Prevent nesting, clarification, and file presentation
    disallowed_tools=["task", "ask_clarification", "present_files"],
    # 模型继承父代理 / Inherit model from parent agent
    model="inherit",
    # 通用子代理允许最多 100 轮交互 / Allow up to 100 agent turns
    max_turns=100,
)
