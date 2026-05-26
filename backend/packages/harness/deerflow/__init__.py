"""DeerFlow 核心框架包。

本包是 DeerFlow 超级代理系统的核心框架（harness），提供了完整的
LangGraph 代理构建、中间件组装、沙箱执行、子代理委派、技能加载等能力。

包结构概览：
- agents/         — 代理系统（工厂、Lead Agent、中间件、线程状态）
- sandbox/        — 沙箱执行系统（本地/远程文件操作隔离）
- subagents/      — 子代理委派系统
- tools/          — 内置工具与工具集成
- mcp/            — MCP（Model Context Protocol）集成
- models/         — 模型工厂（支持思考模式、视觉能力）
- skills/         — 技能发现、加载与解析
- config/         — 配置系统（应用、模型、沙箱等配置）
- community/      — 社区工具（Tavily、Jina AI、Firecrawl 等）
- reflection/     — 动态模块加载（resolve_variable、resolve_class）
- utils/          — 工具函数（网络、可读性等）
- client.py       — 嵌入式 Python 客户端（DeerFlowClient）
"""
