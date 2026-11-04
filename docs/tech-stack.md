# 技术选型

## 依赖

| 模块 | 选择 | 理由 |
|------|------|------|
| CLI 框架 | **Typer** | 类型提示驱动，JSON 输出友好，Pydantic 集成 |
| 数据模型 | **Pydantic** | 元数据建模、入参校验及 Schema 自动生成 |
| 搜索 | **BM25**（bm25s + jieba）+ **Embedding**（Jina/OpenAI，LangChain） | BM25 默认启用，Embedding 可选，RRF 混合排序 |
| MCP 客户端 | **mcp** | 官方 SDK，stdio / Streamable HTTP 传输 |
| 构建后端 | **hatchling** | 现代，与 uv 配合好 |

### 选型取舍

- **搜索**：BM25 提供关键词检索，Embedding 补充语义召回；RRF 按名次融合，无需统一原始分数尺度。启用范围、候选与摘要规则见[检索契约](output-format.md#search-检索)。
- **mcp 官方 vs FastMCP**：先用官方轻量 SDK，够用。FastMCP 功能更丰富但更重，后续需要再切。
- **Typer vs Click**：Typer 基于 Click 但更 Pythonic，类型提示自动生成参数，适合 JSON-first 的设计理念。
- **daemon IPC**：Unix domain socket + JSON 行协议，轻量、无额外依赖，macOS/Linux 原生支持。

## 模块与文档

目录、模块职责和内部生命周期统一见[架构设计](architecture.md#目录与职责)，不在本文件另列目录树。
命令操作见[使用指南](guide.md)，参数和返回契约见[输出字段定义与查看](output-format.md)，开发规范见[CLAUDE.md](../CLAUDE.md)。
