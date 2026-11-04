"""验证 MCP 导出名称、目录合并及真实 stdio、HTTP 传输。"""

import asyncio
import json
import os
import re
import socket
import subprocess
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import httpx
import pytest
from mcp.types import CallToolResult
from mcp.types.version import LATEST_HANDSHAKE_VERSION

import axi.mcp_serve as mcp_serve
from axi.config import AxiConfig
from axi.daemon.protocol import STREAM_LIMIT, DaemonRequest, DaemonResponse
from axi.daemon.server import DaemonServer, _encode_response
from axi.executor import Executor
from axi.mcp_serve import (
    _build_flat_server,
    _build_meta_server,
    build_registry,
    flat_name_map,
)
from axi.models import SearchResult, ToolDescription, ToolMeta, ToolSource
from axi.providers.mcp import MCPConnection, MCPServerConfig
from axi.registry import Registry
from tests.support import make_tool, node, obj, arr, make_registry

RUN_CLI = "from axi.cli import app; app()"
TOOLS_PY = """
import axi
from axi import tool
@tool(description="Add integers")
def add_nums(a: int, b: int) -> int:
    return a + b
@tool(description="Greet")
def greet(name: str) -> str:
    return f"hi {name}"
@tool(description="Read variable A")
def read_var() -> str:
    return axi.env("A") or "unset"
"""


@pytest.mark.parametrize(
    "identities,expected",
    [
        (
            [("add", "demo"), ("greet", "demo")],
            {"add": "demo/add", "greet": "demo/greet"},
        ),
        ([("add", "a"), ("add", "b")], {"a__add": "a/add", "b__add": "b/add"}),
        (
            [("do.thing", "my.server"), ("x", "other")],
            {"my_server__do_thing": "my.server/do.thing", "other__x": "other/x"},
        ),
        (
            [("x", "my.server"), ("x", "my_server")],
            {"my_server__x": "my.server/x", "my_server__x_2": "my_server/x"},
        ),
    ],
)
def test_export_names(identities: list[tuple[str, str]], expected: dict) -> None:
    assert (
        flat_name_map(
            [make_tool(name=name, server=server) for name, server in identities]
        )
        == expected
    )


@pytest.mark.parametrize("flat", [False, True])
def test_merged_registry_uses_registration_order_and_filter_errors(
    monkeypatch: pytest.MonkeyPatch,
    flat: bool,
) -> None:
    source = make_registry(
        make_tool(
            obj({"speed": node("number", "车速")}),
            name="query",
            server="remote",
            description="native",
        )
    )
    native = source.resolve("query")
    remote = native.model_copy(
        update={"source": ToolSource.MCP, "description": "remote"}
    )
    monkeypatch.setattr(
        "axi.mcp_serve.app_config",
        AxiConfig.model_validate({"mcpServers": {"remote": {"command": "unused"}}}),
    )
    request = Mock(return_value=DaemonResponse.success([remote.model_dump()]))
    monkeypatch.setattr("axi.daemon.client.daemon_request", request)
    registry = build_registry(source.list_all(), "remote", flat=flat)
    assert registry.resolve("query").description == "remote"
    assert registry.list_names() == ["remote/query"]
    assert registry.search("remote")[0].source == ToolSource.MCP
    assert registry.describe("query", fields=["$"]).source == ToolSource.MCP
    execute = Mock(return_value=DaemonResponse.success("remote result"))
    monkeypatch.setattr("axi.executor.daemon_request", execute)
    assert Executor(registry).run("query", {}).data == "remote result"
    assert execute.call_args.args[0].tool_name == "remote/query"
    source.register(native.model_copy(update={"description": "updated"}))
    assert registry.describe("query", fields=["$"]).description == "remote"
    with pytest.raises(SystemExit, match="server not found"):
        build_registry(source.list_all(), "missing", flat=flat)
    assert request.call_count == 1  # 未配置的 server 名不会触发 daemon 查询。
    monkeypatch.setattr(
        "axi.daemon.client.daemon_request",
        lambda req: DaemonResponse.fail("daemon down"),
    )
    with pytest.raises(SystemExit, match="configured but not connected"):
        build_registry([], "remote", flat=flat)


@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize("summary_error", [False, True])
def test_export_filters_large_catalog_before_transport(
    monkeypatch: pytest.MonkeyPatch, flat: bool, summary_error: bool
) -> None:
    server = DaemonServer()
    server.registry.register(
        make_tool(
            obj({"value": node("number", "读数")}), server="selected", source="mcp"
        )
    )
    schema = obj({f"value_{i}": node("number", "x" * 1024) for i in range(1000)})
    for index in range(9):
        server.registry.register(
            make_tool(schema, server=f"other{index}", source="mcp")
        )
    assert (
        len(_encode_response(server._list_tools(DaemonRequest(method="list_tools"))))
        > STREAM_LIMIT
    )
    monkeypatch.setattr(
        mcp_serve,
        "app_config",
        AxiConfig.model_validate(
            {
                "mcpServers": {
                    name: {"command": "unused"} for name in ("selected", "offline")
                }
            }
        ),
    )

    requests = []

    def request_catalog(request: DaemonRequest) -> DaemonResponse:
        requests.append((request.catalog_view, request.catalog_servers))
        if request.catalog_view == "summary" and summary_error:
            return DaemonResponse.fail("summary unavailable")
        payload = _encode_response(server._list_tools(request))
        assert len(payload) < STREAM_LIMIT
        response = DaemonResponse.model_validate_json(payload)
        if request.catalog_view == "summary":
            assert all("output_schema" not in item for item in response.data)
        else:
            assert [item["server"] for item in response.data] == ["selected"]
        return response

    monkeypatch.setattr("axi.daemon.client.daemon_request", request_catalog)
    native = [make_tool(server="local")]
    registry = build_registry(native, "selected", flat=flat)
    assert registry.list_names() == ["selected/read"]
    assert (
        registry.describe("selected/read", fields=["$"]).output_fields
        == "{value: float；读数}"
    )
    assert requests == [("full", ["selected"])]

    with pytest.raises(SystemExit) as error:
        build_registry(native, "selected,missing", flat=flat)
    available = ["local", "selected"]
    if not summary_error:
        available = sorted([*available, *[f"other{i}" for i in range(9)]])
    assert str(error.value) == (
        f"Error: server not found: missing. Available: {', '.join(available)}"
    )
    assert requests[1:] == [("full", ["selected", "missing"]), ("summary", None)]


@pytest.fixture
def serve_env(tmp_path: Path) -> dict[str, str]:
    """用隔离配置和三个原生工具启动 CLI 子进程。"""
    tools = tmp_path / "demo.py"
    tools.write_text(TOOLS_PY)
    config = tmp_path / "axi.json"
    config.write_text(json.dumps({"nativeTools": [{"module": str(tools)}]}))
    return {
        "AXI_CONFIG": str(config),
        "HOME": str(tmp_path),
        "PATH": os.environ.get("PATH", ""),
        "A": "from-env",
    }


def _stdio_conn(env: dict[str, str], *cli_args: str) -> MCPConnection:
    return MCPConnection(
        MCPServerConfig(
            server="axi-under-test",
            command=sys.executable,
            args=["-c", RUN_CLI, "mcp", *cli_args],
            env=env,
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("flat", [False, True])
async def test_real_stdio_server(serve_env: dict[str, str], flat: bool) -> None:
    conn = _stdio_conn(serve_env, *(["--server", "demo"] if flat else []))
    try:
        await conn.connect()
        assert sorted(t.name for t in await conn.list_tools()) == (
            ["add_nums", "greet", "read_var"]
            if flat
            else ["describe", "grep", "run", "search"]
        )
        if flat:
            assert await conn.call_tool("add_nums", {"a": 3, "b": 5}) == 8
            assert await conn.call_tool("read_var", {}) == "from-env"
        else:
            assert (await conn.call_tool("grep", {"pattern": "add"}))[0][
                "name"
            ] == "demo/add_nums"
            assert (await conn.call_tool("describe", {"name": "add_nums"}))[
                "input_schema"
            ]["required"] == ["a", "b"]
            assert await conn.call_tool(
                "run", {"name": "add_nums", "params": {"a": 1, "b": 2}}
            ) == {"status": "success", "data": 3}
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_http_transport(serve_env: dict[str, str]) -> None:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            RUN_CLI,
            "mcp",
            "--transport",
            "http",
            "--server",
            "demo",
            "--port",
            str(port),
        ],
        env=serve_env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        async with httpx.AsyncClient() as client:
            for _ in range(100):
                try:
                    response = await client.post(
                        f"http://127.0.0.1:{port}/mcp/", json={}
                    )
                    if "uvicorn" in response.headers.get("server", ""):
                        break
                except httpx.TransportError:
                    pass
                await asyncio.sleep(0.2)
            else:
                pytest.fail("HTTP server did not start")
        for headers, expected in [
            (None, "from-env"),
            ({"Axi-A": "from-header"}, "from-header"),
        ]:
            conn = MCPConnection(
                MCPServerConfig(
                    server="axi-http",
                    url=f"http://127.0.0.1:{port}/mcp",
                    headers=headers or {},
                )
            )
            try:
                await conn.connect()
                assert await conn.call_tool("read_var", {}) == expected
            finally:
                await conn.close()
    finally:
        proc.terminate()
        proc.wait(timeout=5)


@pytest.mark.asyncio
@pytest.mark.parametrize("source", list(ToolSource))
@pytest.mark.parametrize("schema_view", [False, True])
@pytest.mark.parametrize(
    "collision,multi_server", [(False, False), (True, False), (True, True)]
)
async def test_flat_describe_names_and_selection(
    nested_meta: ToolMeta,
    source: ToolSource,
    schema_view: bool,
    collision: bool,
    multi_server: bool,
    mcp_call: Callable,
) -> None:
    registry = Registry()
    nested_meta.server = "demo"
    nested_meta.source = source
    registry.register(nested_meta)
    for name, server in (
        [("axi_describe", "demo"), ("axi_describe_2", "demo")] if collision else []
    ) + ([("other", "aux")] if multi_server else []):
        registry.register(make_tool(name=name, server=server))
    mapping = flat_name_map(registry.list_all())
    server = _build_flat_server(registry, Executor(registry))
    listing = await server._request_handlers["tools/list"].handler(
        SimpleNamespace(request=None, protocol_version=LATEST_HANDSHAKE_VERSION), None
    )
    helper = next(tool.name for tool in listing.tools if tool.name not in mapping)
    assert helper == (
        "axi_describe_3" if collision and not multi_server else "axi_describe"
    )
    exported = "demo__read" if multi_server else "read"
    result = await mcp_call(
        server,
        helper,
        {
            "name": f"{exported},missing,{exported}",
            **(
                {"schema": True}
                if schema_view
                else {"fields": ["$.summaryStatistic.forwardMileage"]}
            ),
        },
    )
    first, missing, repeated = json.loads(result.content[0].text)
    assert first == repeated and first["name"] == exported and "server" not in first
    description = ToolDescription.model_validate(first)
    assert description.model_dump(exclude_none=True) == first
    assert description.full_name == exported
    if source == ToolSource.MCP:
        with pytest.raises(ValueError, match="MCP tool must have a server name"):
            ToolMeta.model_validate(first)
    if schema_view:
        assert first["output_schema"] == nested_meta.output_schema
        assert "output_fields" not in first and "input_schema" not in first
    else:
        assert (
            first["output_fields"]
            == "{summaryStatistic: {forwardMileage: float；前进里程，单位米}}"
        )
    assert "Tool not found" in missing["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["search", "grep"])
@pytest.mark.parametrize("query", ["车速", "speed", "$['vehicles'][*]['speed']"])
async def test_meta_discovery_returns_jsonpath_documents(
    mcp_call: Callable[..., Awaitable[CallToolResult]],
    method: str,
    query: str,
) -> None:
    registry = make_registry(
        make_tool(
            obj({"vehicles": arr(obj({"speed": node("number", "车速")}))}),
            server="fleet",
        )
    )
    server = _build_meta_server(registry, Executor(registry))
    response = await mcp_call(
        server,
        method,
        {
            "query" if method == "search" else "pattern": re.escape(
                "$.vehicles[*].speed"
            )
            if method == "grep" and query.startswith("$")
            else query
        },
    )
    assert json.loads(response.content[0].text) == [
        {
            "name": "fleet/read",
            "description": "",
            "source": "native",
            "matched_fields": ["$.vehicles[*].speed: float；车速"],
        }
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["search", "grep"])
@pytest.mark.parametrize("stage", ["lookup", "dump", "encode"])
async def test_meta_discovery_keeps_event_loop_responsive(
    mcp_call: Callable[..., Awaitable[CallToolResult]],
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    stage: str,
) -> None:
    registry = make_registry(make_tool(obj({"speed": node("number", "车速")})))
    expected = [
        hit.model_dump(exclude_none=True) for hit in getattr(registry, method)("车速")
    ]
    owner, attribute = {
        "lookup": (registry, method),
        "dump": (SearchResult, "model_dump"),
        "encode": (mcp_serve, "_result"),
    }[stage]
    original = getattr(owner, attribute)
    loop, progressed = asyncio.get_running_loop(), Event()

    def blocked(*args: Any, **kwargs: Any) -> Any:
        loop.call_soon_threadsafe(progressed.set)
        assert progressed.wait(2), f"Discovery {stage} blocked the event loop"
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, attribute, blocked)
    response = await mcp_call(
        _build_meta_server(registry, Executor(registry)),
        method,
        {"query" if method == "search" else "pattern": "车速"},
    )
    assert not response.is_error
    assert json.loads(response.content[0].text) == expected
