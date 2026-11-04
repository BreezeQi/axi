"""axi CLI 入口：Typer app。"""

import json
import logging
import os
import signal
from enum import Enum
from typing import Any, Literal

import typer
from pydantic import BaseModel

from axi import runtime
from axi.config import CONFIG_PATH, app_config
from axi.daemon.client import (
    DAEMON_STOP_TIMEOUT,
    LOG_PATH,
    daemon_request,
    ensure_daemon,
    fetch_description,
    fetch_discovery,
    fetch_tool_summaries,
    is_daemon_running,
    read_live_pid,
    wait_for_exit,
    send_request,
)
from axi.daemon.protocol import DaemonRequest
from axi.models import (
    RunResult,
    ToolInfo,
    allowed_types,
)
from axi.output.selection import DescribeSelection
from axi.registry import (
    AmbiguousToolError,
    ToolNotFoundError,
    ToolResolveError,
    split_names,
)

logger = logging.getLogger(__name__)


class Transport(str, Enum):
    stdio = "stdio"
    http = "http"


app = typer.Typer(
    name="axi",
    help="Agent eXecution Interface - unified tool layer for AI Agents",
    no_args_is_help=True,
    rich_markup_mode="rich" if app_config.cli.rich else None,
)

daemon_app = typer.Typer(help="管理 axi daemon")
app.add_typer(daemon_app, name="daemon")


@app.callback(invoke_without_command=True)
def main_callback(ctx: typer.Context) -> None:
    """CLI 启动时加载 axi.json 中配置的原生工具模块。"""
    from axi.providers.native import load_native_tool_modules

    load_native_tool_modules()
    if ctx.invoked_subcommand is None:
        raise typer.Exit()


def _output_json(data: object) -> None:
    """统一 JSON 输出。"""
    if isinstance(data, BaseModel):
        d = data.model_dump(exclude_none=True)
    elif isinstance(data, list):
        d = [
            item.model_dump(exclude_none=True) if isinstance(item, BaseModel) else item
            for item in data
        ]
    else:
        d = data
    typer.echo(json.dumps(d, ensure_ascii=False))


# ── daemon 管理命令 ──────────────────────────────────────────────


@daemon_app.command("start")
def daemon_start() -> None:
    """启动 daemon。"""
    if is_daemon_running():
        typer.echo("Daemon is already running.")
        return

    if ensure_daemon():
        typer.echo("Daemon started.")
        return

    # PID 在但 socket 未建：daemon 还在连 MCP server，不是启动失败
    pid = read_live_pid()
    if pid is not None:
        typer.echo(
            f"Daemon is still initializing (pid {pid}). "
            f"Run `axi daemon status` to check progress; log: {LOG_PATH}",
            err=True,
        )
    else:
        typer.echo(f"Daemon failed to start. Check log: {LOG_PATH}", err=True)
    raise typer.Exit(code=1)


@daemon_app.command("stop")
def daemon_stop() -> None:
    """停止 daemon。"""
    pid = read_live_pid()
    if pid is None:
        typer.echo("Daemon is not running.")
        return
    if is_daemon_running():
        resp = send_request(DaemonRequest(method="shutdown"))
        if resp.status == "error":
            typer.echo(f"Failed to stop daemon: {resp.error}", err=True)
            raise typer.Exit(code=1)
    else:
        os.kill(pid, signal.SIGTERM)

    if not wait_for_exit(pid):
        typer.echo(
            f"Daemon did not exit within {DAEMON_STOP_TIMEOUT:.0f}s. Check log: {LOG_PATH}",
            err=True,
        )
        raise typer.Exit(code=1)
    typer.echo("Daemon stopped.")


@daemon_app.command("status")
def daemon_status() -> None:
    """查看 daemon 状态。"""
    if not is_daemon_running():
        pid = read_live_pid()
        # PID 在、socket 未建：daemon 正在连 MCP server
        _output_json(
            {"status": "starting", "pid": pid}
            if pid is not None
            else {"status": "stopped"}
        )
        return

    resp = send_request(DaemonRequest(method="status"))
    if resp.status == "error":
        _output_json({"status": "error", "error": resp.error})
        raise typer.Exit(code=1)

    result = {
        "status": "running",
        **resp.data,
        "native_tools": runtime.registry.server_tool_counts(),
    }
    _output_json(result)


# ── 核心命令 ──────────────────────────────────────────────


def _discover(
    method: Literal["search", "grep"],
    query: str,
    top_k: int,
    *,
    tool: str | None = None,
) -> list[dict[str, Any]]:
    """先检索 Native，再拼接 MCP 结果；远程失败时保留本地命中。"""
    if tool is not None:
        try:
            meta = runtime.registry.resolve(tool)
        except ToolNotFoundError:
            if not app_config.mcp_servers:
                raise
        else:
            return [
                item.model_dump(exclude_none=True)
                for item in runtime.registry.search(query, top_k, tool=meta.full_name)
            ]
        results = []
    else:
        retrieve = (
            runtime.registry.search if method == "search" else runtime.registry.grep
        )
        results = [
            item.model_dump(exclude_none=True) for item in retrieve(query, top_k=top_k)
        ]
    if top_k > 0 and app_config.mcp_servers:
        try:
            remote = fetch_discovery(
                DaemonRequest(
                    method=method,
                    query=query,
                    top_k=top_k,
                    tool_name=tool,
                )
            )
        except RuntimeError as exc:
            if tool is not None:
                raise
            typer.echo(f"Daemon tool discovery failed: {exc}", err=True)
        else:
            results.extend(remote)
    return results


@app.command()
def search(
    query: str = typer.Argument(help="搜索关键词"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="每种来源最多返回的工具数"),
    tool: str | None = typer.Option(None, "--tool", help="限定工具，只搜索其输出字段"),
) -> None:
    """各来源内双路检索、交错去重，Native 在前、MCP 在后，最多各 K 个。"""
    if top_k <= 0 or not query.strip():
        _output_json([])
        return
    try:
        results = _discover("search", query, top_k, tool=tool)
    except (ToolResolveError, RuntimeError) as exc:
        _output_json({"error": str(exc)})
        raise typer.Exit(code=1)
    _output_json(results)


@app.command()
def grep(
    pattern: str = typer.Argument(help="正则表达式"),
    limit: int = typer.Option(10, "--limit", "-l", help="每种来源最多返回的工具数"),
) -> None:
    """全文正则搜索工具名、描述及字段检索文本，忽略大小写。

    字段与 search 共用“JSONPath: 类型；自身说明”；工具名、描述分别匹配，^ / $ 锚定各自完整文本。
    工具内的命中字段按数据层级从浅到深稳定排序，再取 maxFields 条。
    """
    try:
        results = _discover("grep", pattern, limit)
    except ValueError as e:
        _output_json({"error": str(e)})
        raise typer.Exit(code=1)
    _output_json(results)


def _collect_tool_groups() -> dict[str | None, list[dict]]:
    """收集所有工具并按 server 分组。"""
    groups: dict[str | None, list[dict]] = {}

    # MCP 工具（通过 daemon）
    mcp_tools: list[ToolInfo] = []
    if app_config.mcp_servers:
        try:
            mcp_tools = fetch_tool_summaries()
        except RuntimeError as exc:
            typer.echo(f"Daemon tool listing failed: {exc}", err=True)

    for meta in [*runtime.registry.list_all(), *mcp_tools]:
        groups.setdefault(meta.server, []).append(
            {"name": meta.name, "description": meta.description}
        )

    # daemon 未返回工具时，至少从配置列出 server
    if not mcp_tools:
        for server in app_config.mcp_servers:
            groups.setdefault(server, [])

    return groups


def _filter_groups(
    groups: dict[str | None, list[dict]], server_name: str
) -> dict[str, list[dict]]:
    """按逗号分隔的 server 名过滤分组。未找到时抛 typer.Exit。"""
    names = split_names(server_name)
    filtered = {k: v for k, v in groups.items() if k in names}
    if not filtered:
        missing = [n for n in names if n not in groups]
        available = ", ".join(sorted(k for k in groups if k))
        _output_json(
            {"error": f"Server not found: {', '.join(missing)}. Available: {available}"}
        )
        raise typer.Exit(code=1)
    return filtered


@app.command("list")
def list_tools(
    server_name: str | None = typer.Argument(
        None, help="只列出指定 server 的工具（逗号分隔多个）"
    ),
) -> None:
    """列出所有 server 及其工具。"""
    groups = _collect_tool_groups()

    if server_name is not None:
        filtered = _filter_groups(groups, server_name)
        if len(filtered) == 1:
            key = next(iter(filtered))
            typer.echo(
                json.dumps({"server": key, "tools": filtered[key]}, ensure_ascii=False)
            )
        else:
            typer.echo(
                json.dumps(
                    [{"server": k, "tools": v} for k, v in filtered.items()],
                    ensure_ascii=False,
                )
            )
        return

    # 全部列出（只显示工具名）
    result = [
        {"server": key, "tools": [t["name"] for t in tools]}
        for key, tools in groups.items()
    ]
    typer.echo(json.dumps(result, ensure_ascii=False))


def _resolve_tools(
    names: list[str],
    selection: DescribeSelection,
) -> list[dict[str, Any]]:
    """按名称顺序解析工具，本地未找到时逐项请求远程说明。"""
    describer = runtime.registry.prepare_describe(selection)
    results: list[dict[str, Any]] = []
    for name in names:
        try:
            description = describer.describe(runtime.registry.resolve(name))
        except ToolNotFoundError as exc:
            if app_config.mcp_servers:
                result = fetch_description(name, selection)
            else:
                result = {"error": str(exc)}
        except (AmbiguousToolError, ValueError) as exc:
            result = {"error": str(exc)}
        else:
            result = description.model_dump(exclude_unset=True)
        results.append(result)
    return results


@app.command()
def describe(
    tool_name: str = typer.Argument(help="工具完整名称（逗号分隔多个）"),
    fields: str | None = typer.Option(
        None,
        "--fields",
        help='JSONPath 数组选择输出节点并展开一级，如 ["$.rows[*].id"]；省略/null=入参，["$"]=输出根，[]=空输出视图',
    ),
    schema: bool = typer.Option(
        False,
        "--schema",
        help="读取已登记的完整输出 JSON Schema；不能与非 null 的 --fields 同用，不执行业务工具",
    ),
) -> None:
    """默认查看入参；按需选择局部输出字段或完整输出 Schema。"""
    try:
        selection = DescribeSelection.prepare(
            json.loads(fields) if fields is not None else None,
            schema=schema,
        )
    except (ValueError, TypeError) as exc:
        _output_json({"error": f"Invalid describe selection: {exc}"})
        raise typer.Exit(code=1)

    names = split_names(tool_name)
    results = _resolve_tools(names, selection)
    _output_json(results[0] if len(names) == 1 else results)
    if len(names) == 1 and "error" in results[0]:
        raise typer.Exit(code=1)


@app.command(
    context_settings={"allow_extra_args": True, "allow_interspersed_args": False},
)
def run(
    ctx: typer.Context,
    tool_name: str = typer.Argument(help="工具完整名称"),
) -> None:
    """执行工具。参数支持 --key value 或 --json '{...}' 格式。"""
    args = ctx.args

    if "--help" in args or "-h" in args:
        typer.echo(ctx.get_help())
        raise typer.Exit()

    json_str, args = _extract_option(args, "--json", "-j")

    # 解析参数需要目标 schema；执行分发统一交给 Executor。
    try:
        meta = runtime.registry.resolve(tool_name)
    except AmbiguousToolError as e:
        _output_json({"error": str(e)})
        raise typer.Exit(code=1)
    except ToolNotFoundError:
        meta = None

    if json_str:
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError as e:
            _output_json(RunResult.fail(f"Invalid JSON argument: {e}"))
            raise typer.Exit(code=1)
    elif not args:
        parsed = {}
    else:
        # 按目标 schema 解析 --key value（string 字段保留原文）；MCP schema 问 daemon
        if meta:
            schema = meta.input_schema
        else:
            response = daemon_request(
                DaemonRequest(method="describe", tool_name=tool_name)
            )
            schema = (
                response.data.get("input_schema", {})
                if response.status == "success"
                else {}
            )
        parsed = _parse_params(args, schema)

    _output_json(runtime.executor.run(tool_name, parsed))


@app.command("mcp")
def mcp_command(
    transport: Transport = typer.Option(
        Transport.stdio, "--transport", help="传输方式"
    ),
    server: str | None = typer.Option(
        None, "--server", help="只暴露指定 server 的工具（逗号分隔多个）"
    ),
    flat: bool | None = typer.Option(
        None,
        "--flat/--meta",
        help="平铺每个工具 / 只暴露 search、grep、describe、run 元工具；"
        "缺省自动：指定了 --server 则平铺，否则元工具",
    ),
    port: int = typer.Option(8321, "--port", help="HTTP 端口"),
    host: str = typer.Option("127.0.0.1", "--host", help="HTTP 监听地址"),
) -> None:
    """将 axi 工具导出为 MCP server。"""
    from axi import mcp_serve

    if flat is None:
        flat = server is not None
    mcp_serve.serve(
        native_metas=runtime.registry.list_all(),
        transport=transport.value,
        servers=server,
        flat=flat,
        host=host,
        port=port,
    )


@app.command()
def doctor() -> None:
    """自检：配置、daemon、MCP 连接、embedding、native 工具来源。

    有问题时以非零码退出并在 ``issues`` 里给出可执行的下一步。
    配置了 MCP server 时会拉起 daemon 以验证连接。
    """
    import importlib.metadata

    from axi.providers.native import NATIVE_TOOLS_ENTRY_POINT_GROUP

    issues: list[str] = []

    # native 工具及其来源（entry_points 自动发现 = 任何已装包都可能注入工具，需可见）
    eps = importlib.metadata.entry_points(group=NATIVE_TOOLS_ENTRY_POINT_GROUP)
    server_counts = runtime.registry.server_tool_counts()
    native = {
        "total": sum(server_counts.values()),
        "servers": server_counts,
        "from_config": [e.module for e in app_config.native_tools],
        "from_entry_points": [{"name": ep.name, "value": ep.value} for ep in eps],
    }

    # Embedding 配置
    emb = app_config.search.embedding
    if emb.provider:
        embedding = {
            "provider": emb.provider,
            "api_key": "present" if emb.api_key else "missing",
            "model": emb.model,
        }
        if not emb.api_key:
            env_name = "JINA_API_KEY" if emb.provider == "jina" else "OPENAI_API_KEY"
            issues.append(
                f"Embedding provider '{emb.provider}' set but no API key. Put it in "
                f"axi.json search.embedding.apiKey or the {env_name} env var."
            )
    else:
        embedding = {"provider": None, "note": "BM25-only, no semantic search"}

    # daemon + MCP 连接（仅在配置了 MCP server 时拉起 daemon 验证）
    configured = set(app_config.mcp_servers)
    mcp_servers: list[dict] = []
    if configured:
        resp = daemon_request(DaemonRequest(method="status"))
        if resp.status == "success":
            connected = resp.data.get("server_tools", {})
            daemon = {
                "running": True,
                "pid": resp.data.get("pid"),
                "uptime_seconds": resp.data.get("uptime_seconds"),
            }
            for name in sorted(configured):
                if name in connected:
                    mcp_servers.append(
                        {
                            "server": name,
                            "status": "connected",
                            "tools": connected[name],
                        }
                    )
                else:
                    mcp_servers.append({"server": name, "status": "not_connected"})
                    issues.append(
                        f"MCP server '{name}' configured but not connected. "
                        f"Check daemon log: {LOG_PATH}"
                    )
        else:
            daemon = {"running": False, "error": resp.error}
            issues.append(f"Daemon unavailable: {resp.error}. Check log: {LOG_PATH}")
    else:
        daemon = {"running": is_daemon_running()}

    report = {
        "ok": not issues,
        "config": {"path": str(CONFIG_PATH), "exists": CONFIG_PATH.exists()},
        "daemon": daemon,
        "native_tools": native,
        "mcp_servers": mcp_servers,
        "embedding": embedding,
        "issues": issues,
    }
    _output_json(report)
    if issues:
        raise typer.Exit(code=1)


# ── 参数解析辅助函数 ──────────────────────────────────────────────


def _extract_option(args: list[str], *names: str) -> tuple[str | None, list[str]]:
    remaining = []
    value = None
    i = 0
    while i < len(args):
        if args[i] in names and i + 1 < len(args):
            value = args[i + 1]
            i += 2
        else:
            remaining.append(args[i])
            i += 1
    return value, remaining


def _schema_field_types(schema: dict, key: str) -> set[str]:
    """取字段在 schema 里声明的类型集合；字段不存在或无 schema 返回空集。"""
    props = schema.get("properties")
    if not isinstance(props, dict):
        return set()
    prop = props.get(key)
    return allowed_types(prop) if isinstance(prop, dict) else set()


def _coerce_cli_value(key: str, value: str, schema: dict) -> object:
    """按目标字段类型决定是否 JSON 解析。

    字段声明为 string → 原样保留，避免 ``true``/``42``/``null`` 被解析成
    非字符串类型（``--title false`` 应是字符串 "false"）。其余（含未知字段、
    无 schema）尝试 JSON 解析，让 ``--count 42``→42、``--flag true``→True、
    ``--data {...}``→dict 正常工作，失败落回原字符串。
    """
    if "string" in _schema_field_types(schema, key):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value


def _parse_params(params: list[str], schema: dict | None = None) -> dict:
    schema = schema or {}
    parsed: dict = {}
    i = 0
    while i < len(params):
        arg = params[i]
        if arg.startswith("--"):
            key = arg[2:]
            if i + 1 < len(params) and not params[i + 1].startswith("--"):
                parsed[key] = _coerce_cli_value(key, params[i + 1], schema)
                i += 2
            else:
                parsed[key] = True
                i += 1
        else:
            logger.warning("Ignoring unrecognized argument: %s", arg)
            i += 1
    return parsed
