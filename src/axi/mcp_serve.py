"""将 axi 工具导出为 MCP server（stdio / streamable HTTP）。

两种暴露形态（互斥）：
- flat（平铺）：每个业务工具直接调用；含输出 Schema 时另提供按需 describe
- meta（元工具）：只暴露 search / grep / describe / run 四个固定工具，
  对端以渐进式披露的方式发现和调用，上下文占用恒定
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable, Collection, Mapping
from typing import Any

import mcp.types as types
from mcp.server.context import ServerRequestContext
from mcp.server.lowlevel import Server

from axi.config import app_config
from axi.context import set_request_env
from axi.daemon.client import fetch_tool_catalog, fetch_tool_summaries
from axi.daemon.protocol import LOG_PATH
from axi.executor import Executor
from axi.models import (
    DescribeOptions,
    RunResult,
    SearchResult,
    ToolDescription,
    ToolMeta,
)
from axi.output.registration import OUTPUT_SCHEMA_META_KEY
from axi.registry import Registry, ToolResolveError, split_names

logger = logging.getLogger(__name__)

_META_INSTRUCTIONS = (
    "Discover capabilities with search only when needed; describe supplies inputs, selected fields or the complete output schema. "
    "Reuse known contracts and call run directly when ready. matched_fields lists JSONPath: type; description strings. "
    "JSONPath $ refers to the complete successful business value, excluding AXI's envelope. "
    "Output definitions are documentation only. run returns {status, data, error}."
)


# ── Registry 构建 ─────────────────────────────────────────────


def build_registry(
    native_metas: list[ToolMeta], servers: str | None, flat: bool = False
) -> Registry:
    """构建 serve 进程的合并 Registry：native 工具 + daemon 的 MCP 工具。

    ``servers`` 为逗号分隔的 server 名，指定时只保留这些组；
    名字不存在时报错退出并列出可用 server。
    只选中 native 组时不启动 daemon；平铺模式不用搜索，不建 embedding。
    """
    names = split_names(servers) if servers else None
    metas = list(native_metas)

    need_mcp = app_config.mcp_servers and (
        names is None or any(n in app_config.mcp_servers for n in names)
    )
    if need_mcp:
        try:
            remote = fetch_tool_catalog(servers=names)
        except RuntimeError as exc:
            logger.warning("MCP tools unavailable: %s", exc)
        else:
            metas.extend(remote)

    if names:
        available = sorted({m.server for m in metas if m.server})
        missing = [n for n in names if n not in available]
        # 区分"配置了但没连上"和"根本不存在"，避免把 daemon 故障误报成配置错误
        unavailable = [n for n in missing if n in app_config.mcp_servers]
        if unavailable:
            raise SystemExit(
                f"Error: MCP server configured but not connected: "
                f"{', '.join(unavailable)}. Check daemon log: {LOG_PATH}"
            )
        if missing:
            if need_mcp:
                try:
                    summary = fetch_tool_summaries()
                except RuntimeError as exc:
                    logger.warning("MCP tools unavailable: %s", exc)
                else:
                    available = sorted(
                        set(available)
                        | {tool.server for tool in summary if tool.server}
                    )
            raise SystemExit(
                f"Error: server not found: {', '.join(missing)}. "
                f"Available: {', '.join(available)}"
            )
        metas = [m for m in metas if m.server in names]

    registry = Registry() if flat else Registry.from_search_config(app_config.search)
    for meta in metas:
        registry.register(meta)
    return registry


# ── 执行分发 ─────────────────────────────────────────────────


async def _execute(
    ctx: ServerRequestContext,
    registry: Registry,
    executor: Executor,
    name: str,
    params: dict[str, Any],
) -> RunResult:
    """限定导出的工具范围，注入请求变量，再交给共用执行入口。"""
    try:
        meta = registry.resolve(name)
    except ToolResolveError as e:
        return RunResult.fail(str(e))

    request = ctx.request
    set_request_env(request.headers if request else None)
    return await asyncio.to_thread(executor.run, meta.full_name, params)


def _result(data: Any, is_error: bool = False) -> types.CallToolResult:
    if not isinstance(data, str):
        data = json.dumps(data, ensure_ascii=False, default=str)
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=data)], is_error=is_error
    )


def _discover(
    search: Callable[..., list[SearchResult]],
    query: str,
    limit: int,
    *,
    tool: str | None = None,
) -> types.CallToolResult:
    """在同一个工作线程内检索并编码摘要，避免阻塞事件循环。"""
    try:
        results = (
            search(query, limit, tool=tool)
            if tool is not None
            else search(query, limit)
        )
    except ToolResolveError as exc:
        return _result({"error": str(exc)}, is_error=True)
    return _result([result.model_dump(exclude_none=True) for result in results])


class _DescribeParams(DescribeOptions):
    """flat 与 meta 两种导出模式共用的 describe 入参。"""

    name: str


_DESCRIBE_TOOL = types.Tool(
    name="describe",
    description="Inspect tool documentation. Set schema=true to read the complete registered output JSON Schema without executing the tool; omit fields or use null in this mode. "
    "Otherwise omit fields or use null for complete inputs without output docs. "
    'Use fields=["$"] for the output root directory, specific JSONPaths for local output structure, or [] for an empty output view. '
    "Selected output nodes expand one level; output views omit inputs. "
    "Use search or grep to locate output fields and reuse sufficient matches directly. "
    "Accepts comma-separated names for batch lookup. "
    "Explicit deep paths remain visible; overlapping selections are unioned. "
    "output_fields is a documentation string; "
    "combinations use anyOf(...), oneOf(...) and allOf(...). "
    "Expanded tuples retain [n] and [n:] position labels. "
    "Read these documents and use JSONPath to access business data.",
    input_schema=_DescribeParams.model_json_schema(),
)


def _describe(
    registry: Registry,
    names: list[str],
    arguments: Mapping[str, Any],
    *,
    exported_names: Mapping[str, str] | None = None,
) -> types.CallToolResult:
    """两种导出模式共用 Registry 查看流程，flat 结果改用实际导出名。"""
    items = registry.describe_many(
        names,
        fields=arguments.get("fields"),
        schema=arguments.get("schema", False),
    )
    descriptions = []
    for item in items:
        description = item.model_dump(exclude_unset=True)
        if exported_names is not None and isinstance(item, ToolDescription):
            description["name"] = exported_names.get(item.full_name, item.name)
            description.pop("server", None)
        descriptions.append(description)
    return _result(
        descriptions[0] if len(descriptions) == 1 else descriptions,
    )


_Dispatch = Callable[
    [ServerRequestContext, str, dict[str, Any]], Awaitable[types.CallToolResult]
]


def _build_server(
    tools: list[types.Tool], dispatch: _Dispatch, instructions: str | None = None
) -> Server:
    """两种模式共用的 Server 组装：静态工具列表 + call_tool 统一守卫。

    v2 不再把 handler 异常转 is_error 结果（会抛成对 agent 不可读的协议错误），
    缺参/坏正则等在此统一捕获。
    """
    listing = types.ListToolsResult(tools=tools)

    async def list_tools(
        ctx: ServerRequestContext, params: types.PaginatedRequestParams | None
    ) -> types.ListToolsResult:
        return listing

    async def call_tool(
        ctx: ServerRequestContext, params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        logger.info("call_tool %s", params.name)
        try:
            return await dispatch(ctx, params.name, params.arguments or {})
        except Exception as e:
            return _result(f"{type(e).__name__}: {e}", is_error=True)

    return Server(
        "axi",
        instructions=instructions,
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )


# ── flat 模式 ────────────────────────────────────────────────


def _sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def _unique_name(base: str, used: Collection[str]) -> str:
    """重名时为新工具追加数字后缀，保留已有导出名。"""
    name, index = base, 2
    while name in used:
        name, index = f"{base}_{index}", index + 1
    return name


def flat_name_map(metas: list[ToolMeta]) -> dict[str, str]:
    """MCP 工具名 → full_name。单 server 用裸名，多 server 加 ``server__`` 前缀。

    清洗后同名的工具按注册顺序追加 ``_2``/``_3`` 后缀，避免静默覆盖。
    """
    single = len({m.server for m in metas}) == 1
    mapping: dict[str, str] = {}
    for m in metas:
        base = _sanitize(m.name if single else f"{m.server}__{m.name}")
        mapping[_unique_name(base, mapping)] = m.full_name
    return mapping


def _tool_description(meta: ToolMeta) -> str:
    if meta.output_example is None:
        return meta.description
    example = json.dumps(meta.output_example, ensure_ascii=False, default=str)
    return f"{meta.description}\n\nOutput example: {example}"


def _export_tool(name: str, meta: ToolMeta) -> types.Tool:
    """通过 AXI 元数据导出字段文档，不为业务工具新增 MCP 输出契约。"""
    return types.Tool(
        name=name,
        description=_tool_description(meta),
        input_schema=meta.input_schema or {"type": "object"},
        meta={OUTPUT_SCHEMA_META_KEY: meta.output_schema}
        if meta.output_schema is not None
        else None,
    )


def _build_flat_server(registry: Registry, executor: Executor) -> Server:
    metas = registry.list_all()
    name_map = flat_name_map(metas)
    exported_names = {full_name: name for name, full_name in name_map.items()}
    # registry 在 serve 期间不可变，工具列表一次性预构建
    tools = [_export_tool(exported_names[meta.full_name], meta) for meta in metas]
    describe_name = (
        _unique_name("axi_describe", name_map)
        if any(meta.output_schema is not None for meta in metas)
        else None
    )
    if describe_name is not None:
        tools.append(
            _DESCRIBE_TOOL.model_copy(
                update={
                    "name": describe_name,
                    "description": "Inspect exported tools by their names from tools/list. "
                    + _DESCRIBE_TOOL.description,
                }
            )
        )

    async def dispatch(
        ctx: ServerRequestContext, name: str, arguments: dict[str, Any]
    ) -> types.CallToolResult:
        if describe_name is not None and name == describe_name:
            names = [
                name_map.get(item, item) for item in split_names(arguments["name"])
            ]
            return await asyncio.to_thread(
                _describe,
                registry,
                names,
                arguments,
                exported_names=exported_names,
            )
        full_name = name_map.get(name)
        if full_name is None:
            return _result(
                f"Unknown tool: {name}. Call tools/list to see the available tools.",
                is_error=True,
            )
        result = await _execute(ctx, registry, executor, full_name, arguments)
        if result.status == "error":
            return _result(result.error, is_error=True)
        return _result(result.data)

    return _build_server(tools, dispatch)


# ── meta 模式 ────────────────────────────────────────────────

_META_TOOLS = [
    types.Tool(
        name="search",
        description="Search tool names, descriptions and output field documentation. "
        "Search the combined native and MCP catalog and return up to top_k tools in total. "
        "matched_fields lists matching JSONPath: type; own description strings. "
        "JSONPath $ refers to the complete successful business value, excluding AXI's envelope. "
        "Use the paths to plan data access or pass them as a JSONPath array to describe.fields.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "tool": {
                    "type": "string",
                    "description": "Optional tool name; search only its output field documentation using the catalog's existing field ranking.",
                },
                "top_k": {
                    "type": "integer",
                    "default": 5,
                    "description": "Maximum tools from the combined catalog.",
                },
            },
            "required": ["query"],
        },
    ),
    types.Tool(
        name="grep",
        description="Case-insensitive full-text regex search over tool names, descriptions and output fields. "
        "Search the combined native and MCP catalog and return up to limit tools in registration order. "
        "Fields share search text: JSONPath: type; own description. Matching fields are stably ordered by data depth before maxFields truncation. "
        "Tool names and descriptions match independently; ^/$ anchor each complete text.",
        input_schema={
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "limit": {
                    "type": "integer",
                    "default": 10,
                    "description": "Maximum tools from the combined catalog.",
                },
            },
            "required": ["pattern"],
        },
    ),
    _DESCRIBE_TOOL,
    types.Tool(
        name="run",
        description="Execute a tool by name with a params object. "
        "Returns {status, data, error}.",
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "params": {"type": "object"},
            },
            "required": ["name"],
        },
    ),
]


def _build_meta_server(registry: Registry, executor: Executor) -> Server:
    async def dispatch(
        ctx: ServerRequestContext, name: str, arguments: dict[str, Any]
    ) -> types.CallToolResult:
        if name == "search":
            return await asyncio.to_thread(
                _discover,
                registry.search,
                arguments["query"],
                arguments.get("top_k", 5),
                tool=arguments.get("tool"),
            )
        if name == "grep":
            return await asyncio.to_thread(
                _discover,
                registry.grep,
                arguments["pattern"],
                arguments.get("limit", 10),
            )
        if name == "describe":
            return await asyncio.to_thread(
                _describe,
                registry,
                split_names(arguments["name"]),
                arguments,
            )
        if name == "run":
            result = await _execute(
                ctx,
                registry,
                executor,
                arguments["name"],
                arguments.get("params") or {},
            )
            return _result(result.model_dump(exclude_none=True))
        return _result(
            f"Unknown tool: {name}. Available: search, grep, describe, run.",
            is_error=True,
        )

    return _build_server(_META_TOOLS, dispatch, instructions=_META_INSTRUCTIONS)


# ── transport 与入口 ─────────────────────────────────────────


async def _run_stdio(server: Server) -> None:
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


def _setup_http_logging() -> None:
    """HTTP 模式把 axi 与 mcp 的日志打到 stderr（INFO），让连接/请求可见。

    只在 HTTP 模式调用：stdio 的 stdout 是协议通道不能污染，而未配置 handler 时
    Python 的 lastResort 只在 WARNING 以上输出，故 stdio/cli 的 info 日志天然静默。
    uvicorn 自身的 access log 由 Config(log_level="info") 负责，这里补齐 axi/mcp。
    """
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
    )
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


async def _run_http(server: Server, host: str, port: int) -> None:
    import uvicorn
    from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
    from starlette.applications import Starlette
    from starlette.routing import Mount

    manager = StreamableHTTPSessionManager(app=server)
    app = Starlette(
        routes=[Mount("/mcp", app=manager.handle_request)],
        lifespan=lambda app: manager.run(),
    )
    logger.info("axi MCP server listening on http://%s:%d/mcp", host, port)
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    await uvicorn.Server(config).serve()


def serve(
    native_metas: list[ToolMeta],
    transport: str,
    servers: str | None,
    flat: bool,
    host: str,
    port: int,
) -> None:
    """构建合并 Registry 并运行 MCP server，阻塞直到进程退出。

    stdio 模式下 stdout 是协议通道，日志只能走 stderr。
    """
    registry = build_registry(native_metas, servers, flat)
    if not registry.list_all():
        raise SystemExit(
            "Error: no tools to serve. Configure mcpServers or nativeTools "
            "in the axi.json pointed to by AXI_CONFIG."
        )

    executor = Executor(registry)
    if flat:
        server = _build_flat_server(registry, executor)
    else:
        server = _build_meta_server(registry, executor)

    if transport == "http":
        _setup_http_logging()
        asyncio.run(_run_http(server, host, port))
    else:
        asyncio.run(_run_stdio(server))
