# langgraph.json 配置详解

本文件是 LangGraph Server 的核心配置文件，定义了代理运行时的启动参数。
LangGraph Server 在启动时读取此文件，据此发现图工厂函数、认证模块和检查点提供者。

## 配置字段说明

### `$schema`

```json
"$schema": "https://langgra.ph/schema.json"
```

JSON Schema 验证地址。用于 IDE 和验证工具检查配置格式是否正确。

---

### `python_version`

```json
"python_version": "3.12"
```

运行时要求的 Python 版本。LangGraph Server 会使用此版本创建运行环境。
DeerFlow 使用 Python 3.12+ 的特性（如类型提示语法 `X | None`）。

---

### `dependencies`

```json
"dependencies": ["."]
```

Python 包依赖列表。`"."` 表示依赖当前目录（backend/）下的包，
即通过 `pyproject.toml` 中定义的 `deerflow-harness` 包及其所有传递依赖。

LangGraph Server 在启动时会安装这些依赖到运行环境中。

---

### `env`

```json
"env": ".env"
```

环境变量文件路径。指向项目根目录下的 `.env` 文件，
其中存放 API 密钥（如 `OPENAI_API_KEY`）、数据库连接字符串等敏感配置。
config.yaml 中以 `$` 开头的值会从环境变量中解析。

---

### `graphs`

```json
"graphs": {
  "lead_agent": "deerflow.agents:make_lead_agent"
}
```

**图注册表** — LangGraph Server 最核心的配置项。

- **键** (`lead_agent`)：图的标识名称，用于 API 路由。
  客户端通过 `/threads/{thread_id}/runs` 等端点引用此图。
  也是 LangGraph Studio 中显示的图名称。

- **值** (`deerflow.agents:make_lead_agent`)：图工厂函数的导入路径，
  格式为 `模块路径:函数名`。LangGraph Server 调用此函数创建代理图实例。

  解析链路：
  1. `deerflow.agents` → `packages/harness/deerflow/agents/__init__.py`
  2. `__init__.py` 中的 `make_lead_agent` → 从 `deerflow.agents.lead_agent.agent` 导入
  3. `agent.py` 中的 `make_lead_agent(config)` → 解析配置并创建完整代理

  工厂函数签名必须接受 `RunnableConfig` 参数并返回 LangGraph 图实例，
  以保持与 LangGraph Server 的兼容性。

---

### `auth`

```json
"auth": {
  "path": "./app/gateway/langgraph_auth.py:auth"
}
```

认证模块路径。指向 Gateway 应用层的认证处理器：

- **模块**：`./app/gateway/langgraph_auth.py`
- **对象**：`auth`（该模块中导出的认证实例）

此认证处理器用于：
- 验证客户端请求的身份（API Key、JWT 等）
- 提供用户身份信息供 `get_effective_user_id()` 解析
- 在无认证模式下回退到默认用户 `"default"`

认证模块位于 `app/` 层而非 `deerflow/` 层，遵循 Harness/App 分层原则：
应用层处理认证，框架层只消费用户 ID。

---

### `checkpointer`

```json
"checkpointer": {
  "path": "./packages/harness/deerflow/runtime/checkpointer/async_provider.py:make_checkpointer"
}
```

检查点（状态持久化）提供者路径。用于保存和恢复代理的对话状态：

- **模块**：`./packages/harness/deerflow/runtime/checkpointer/async_provider.py`
- **函数**：`make_checkpointer`

检查点的作用：
- 每次代理执行后，将当前 ThreadState 保存到持久化存储
- 用户中断后恢复对话时，从检查点加载上次的状态
- 支持多种后端：内存（开发用）、PostgreSQL（生产用）、SQLite 等
- `make_checkpointer` 根据 `config.yaml` 中的配置动态选择后端

与 RunStore 的关系：
- Checkpointer 保存图的完整状态（消息、工具调用、待办事项等）
- RunStore 保存运行的元数据（状态、时间戳等）
- 两者互补，共同实现对话的持久化和恢复

---

## 与其他配置文件的关系

```
langgraph.json          ← LangGraph Server 启动配置（本文件）
config.yaml             ← DeerFlow 应用配置（模型、工具、沙箱、记忆等）
extensions_config.json  ← MCP 服务器和技能配置
.env                    ← 环境变量（API 密钥等敏感信息）
pyproject.toml          ← Python 包定义和依赖声明
```

## 启动流程

```
1. LangGraph Server 读取 langgraph.json
2. 安装 dependencies 中声明的 Python 包
3. 加载 .env 环境变量
4. 初始化 auth 认证模块
5. 初始化 checkpointer 检查点提供者
6. 调用 graphs 中注册的工厂函数 make_lead_agent(config)
7. 代理图就绪，开始接受客户端请求
```
