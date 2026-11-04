"""验证 daemon 响应解码、大目录传输与入参处理。"""

import asyncio
import json
import os
import signal
import time
import threading
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
import typer
import axi.daemon.client as client
from typer.testing import CliRunner

import axi.cli as cli
from axi.config import AxiConfig, DaemonConfig
from axi.daemon.client import (
    _send,
    fetch_description,
    fetch_discovery,
    fetch_tool_catalog,
    fetch_tool_summaries,
    send_request,
)
from axi.daemon.protocol import (
    STREAM_LIMIT,
    DaemonRequest,
    DaemonResponse,
)
from axi.daemon.server import DaemonServer, _coerce_to_schema, _encode_response
from axi.models import DescribeOptions, ToolInfo, ToolMeta
from axi.providers.mcp import MCPServerConfig
from tests.support import make_tool, node, obj, ref, make_registry


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["catalog", "search", "grep", "describe", "encoding"])
async def test_status_can_respond_while_documentation_work_is_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    socket = str(tmp_path / "daemon.sock")
    monkeypatch.setattr("axi.daemon.client.SOCKET_PATH", socket)
    server = DaemonServer()
    server.registry.register(make_tool(obj({"value": node("number", "测量值")})))
    owner, method, request = {
        "catalog": (server.registry, "list_all", DaemonRequest(method="list_tools")),
        "search": (
            server.registry,
            "search",
            DaemonRequest(method="search", query="value"),
        ),
        "grep": (server.registry, "grep", DaemonRequest(method="grep", query="value")),
        "describe": (
            server.registry,
            "describe_many",
            DaemonRequest(method="describe", tool_name="read"),
        ),
        "encoding": (
            DaemonResponse,
            "model_dump_json",
            DaemonRequest(method="list_tools"),
        ),
    }[stage]
    original = getattr(owner, method)
    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()
    entered, release = asyncio.Event(), threading.Event()
    paused = False

    def pause_once(*args: Any, **kwargs: Any) -> Any:
        nonlocal paused
        if not paused:
            paused = True
            assert threading.get_ident() != loop_thread
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(timeout=5), "Documentation work was not released"
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, method, pause_once)
    offload = Mock(wraps=asyncio.to_thread)
    monkeypatch.setattr(asyncio, "to_thread", offload)
    listener = await asyncio.start_unix_server(server._handle_client, path=socket)
    async with listener:
        pending = asyncio.create_task(_send(request))
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
            status = await asyncio.wait_for(
                _send(DaemonRequest(method="status")), timeout=5
            )
            assert status.status == "success", status.error
            assert status.data["server_tools"] == {"unknown": 1}
            assert not pending.done()
        finally:
            release.set()
            response = await asyncio.wait_for(pending, timeout=5)
        assert response.status == "success", response.error
        assert offload.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_count", [0, 2])
async def test_startup_registers_imported_definitions_without_copying(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool_count: int
) -> None:
    import axi.daemon.server as daemon

    for name, value in {
        "SOCKET_DIR": str(tmp_path),
        "SOCKET_PATH": str(tmp_path / "daemon.sock"),
        "PID_PATH": str(tmp_path / "daemon.pid"),
    }.items():
        monkeypatch.setattr(daemon, name, value)
    loop = asyncio.get_running_loop()
    monkeypatch.setattr(loop, "add_signal_handler", Mock())
    monkeypatch.setattr(loop, "remove_signal_handler", Mock())
    originals: list[ToolMeta] = []

    async def connect_all(configs: list[MCPServerConfig]) -> list[ToolMeta]:
        tools = [
            make_tool(obj({"value": node("number", "测量值")}), name=f"read_{i}")
            for i in range(tool_count)
        ]
        originals.extend(tools)
        return tools

    ready = asyncio.Event()

    async def serve_forever() -> None:
        ready.set()
        await asyncio.Future()

    listener = AsyncMock()
    listener.__aenter__.return_value = listener
    listener.serve_forever.side_effect = serve_forever
    monkeypatch.setattr(
        daemon.asyncio, "start_unix_server", AsyncMock(return_value=listener)
    )
    server = DaemonServer()
    monkeypatch.setattr(
        server.mcp_provider,
        "load_config",
        lambda: [MCPServerConfig(server="demo", command="offline")],
    )
    monkeypatch.setattr(server.mcp_provider, "connect_all", connect_all)
    monkeypatch.setattr(server.mcp_provider, "close_all", AsyncMock())
    task = asyncio.create_task(server.start())
    try:
        await asyncio.wait_for(ready.wait(), timeout=5)
        assert len(originals) == tool_count
        assert all(server.registry.get(meta.full_name) is meta for meta in originals)
        assert len(server.registry.list_all()) == tool_count
        for i in range(tool_count):
            assert (
                server.registry.describe(f"read_{i}", fields=["$"]).output_fields
                == "{value: float；测量值}"
            )
    finally:
        server.stop()
        await task


@pytest.mark.asyncio
async def test_large_registered_catalog_round_trips_through_the_running_socket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import axi.daemon.server as daemon
    from axi.config import AxiConfig

    socket = str(tmp_path / "daemon.sock")
    monkeypatch.setattr(daemon, "app_config", AxiConfig())
    monkeypatch.setattr(daemon, "SOCKET_DIR", str(tmp_path))
    monkeypatch.setattr(daemon, "SOCKET_PATH", socket)
    monkeypatch.setattr(daemon, "PID_PATH", str(tmp_path / "daemon.pid"))
    monkeypatch.setattr("axi.daemon.client.SOCKET_PATH", socket)
    monkeypatch.setattr(asyncio.get_running_loop(), "add_signal_handler", Mock())
    server = daemon.DaemonServer()
    monkeypatch.setattr(server.mcp_provider, "load_config", lambda: [])
    meta = make_tool(
        name="large",
        server="api",
        source="mcp",
        description="record",
        input_schema=obj({"identifier": node("string")}),
        output_example={"value_0": 7},
        schema={
            "properties": {
                f"value_{i}": node("number", "measurement " * 10) for i in range(1000)
            }
        },
    )
    server.registry.register(meta)
    ready = asyncio.Event()
    listen = asyncio.start_unix_server

    async def listening(*args: Any, **kwargs: Any) -> asyncio.Server:
        listener = await listen(*args, **kwargs)
        ready.set()
        return listener

    monkeypatch.setattr(daemon.asyncio, "start_unix_server", listening)
    task = asyncio.create_task(server.start())
    try:
        await asyncio.wait_for(ready.wait(), timeout=5)
        for threaded in (False, True):
            for options in (
                {},
                {"catalog_view": "summary"},
                {"catalog_servers": ["api"]},
                {"catalog_view": "summary", "catalog_servers": ["api"]},
                {"catalog_servers": []},
                {"catalog_servers": ["missing"]},
            ):
                request = DaemonRequest(method="list_tools", **options)
                pending = (
                    asyncio.to_thread(send_request, request)
                    if threaded
                    else _send(request)
                )
                response = await asyncio.wait_for(pending, timeout=5)
                assert response.status == "success", response.error
                if request.catalog_servers in ([], ["missing"]):
                    assert response.data == []
                    continue
                item = response.data[0]
                assert {
                    key: item[key]
                    for key in ("name", "server", "description", "source")
                } == {
                    "name": "large",
                    "server": "api",
                    "description": "record",
                    "source": "mcp",
                }
                if request.catalog_view == "full":
                    assert item["input_schema"] == meta.input_schema
                    assert item["output_example"] == meta.output_example
                else:
                    assert "input_schema" not in item and "output_example" not in item
                if request.catalog_view == "summary":
                    assert set(item) == {"name", "server", "description", "source"}
                else:
                    assert item["output_schema"] == meta.output_schema
                    assert len(response.model_dump_json().encode()) > 64 * 1024
        response = await _send(
            DaemonRequest(method="grep", query=r"\$\.value_0:", top_k=1)
        )
        assert response.status == "success", response.error
        assert [item["name"] for item in response.data] == ["api/large"]
        assert response.data[0]["matched_fields"] == [
            "$.value_0: float；" + "measurement " * 10
        ]
        monkeypatch.setattr(
            "axi.output.describe.select_view",
            Mock(side_effect=AssertionError("Input lookup rendered output fields")),
        )
        inputs = await _send(DaemonRequest(method="describe", tool_name="api/large"))
        assert (
            inputs.status == "success"
            and inputs.data["input_schema"] == meta.input_schema
        )
        assert "output_fields" not in inputs.data
        full = await _send(
            DaemonRequest(method="describe", tool_name="api/large", schema=True)
        )
        assert full.status == "success", full.error
        assert full.data == {
            "name": "large",
            "server": "api",
            "description": meta.description,
            "source": "mcp",
            "output_schema_status": "registered",
            "output_schema": meta.output_schema,
        }
        assert server.registry.list_names() == ["api/large"]
    finally:
        server.stop()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("view", ["full", "summary"])
@pytest.mark.parametrize(
    "servers,expected",
    [(None, ["a", "b"]), (["b"], ["b"]), (["b", "a", "b"], ["a", "b"])],
)
def test_catalog_server_scope_preserves_registration_order(
    monkeypatch: pytest.MonkeyPatch,
    view: str,
    servers: list[str] | None,
    expected: list[str],
) -> None:
    server = DaemonServer()
    for name in ("a", "b"):
        server.registry.register(make_tool(obj({"value": node("number")}), server=name))

    def request_catalog(request: DaemonRequest) -> DaemonResponse:
        assert request.catalog_view == view and request.catalog_servers == servers
        return server._list_tools(request)

    monkeypatch.setattr("axi.daemon.client.daemon_request", request_catalog)
    tools = (
        fetch_tool_summaries(servers=servers)
        if view == "summary"
        else fetch_tool_catalog(servers=servers)
    )
    assert [tool.server for tool in tools] == expected
    assert all(
        type(tool) is (ToolInfo if view == "summary" else ToolMeta) for tool in tools
    )
    if view == "summary":
        assert all(
            set(tool.model_dump()) == {"name", "server", "description", "source"}
            for tool in tools
        )
    else:
        assert all(
            tool.output_schema == obj({"value": node("number")}) for tool in tools
        )


@pytest.mark.parametrize("value,expected", [("7", 7.0), ("bad", 45.0), (None, 45.0)])
def test_request_timeout(
    monkeypatch: pytest.MonkeyPatch, value: str | None, expected: float
) -> None:
    monkeypatch.delenv("AXI_REQUEST_TIMEOUT", raising=False)
    assert DaemonConfig().request_timeout == 120.0
    if value is not None:
        monkeypatch.setenv("AXI_REQUEST_TIMEOUT", value)
    assert (
        DaemonConfig.model_validate({"requestTimeout": 45}).request_timeout == expected
    )


@pytest.mark.asyncio
async def test_daemon_response_uses_existing_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = asyncio.StreamReader()
    payload = {"status": "success", "data": {"value": 7}}
    reader.feed_data(json.dumps(payload).encode() + b"\n")
    reader.feed_eof()
    writer = Mock(drain=AsyncMock(), wait_closed=AsyncMock())
    monkeypatch.setattr(
        "axi.daemon.client.asyncio.open_unix_connection",
        AsyncMock(return_value=(reader, writer)),
    )
    result = await _send(DaemonRequest(method="call_tool", tool_name="api/read"))
    assert result.model_dump(exclude_none=True) == payload
    writer.write.assert_called_once()
    writer.close.assert_called_once()


@pytest.mark.asyncio
async def test_daemon_connection_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("axi.daemon.client.SOCKET_PATH", str(tmp_path / "missing.sock"))
    result = await _send(DaemonRequest(method="list_tools"))
    assert result.status == "error" and "Cannot connect to daemon" in result.error


@pytest.fixture(params=["oversized", "json", "disconnect", "close"])
def failed_transport(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> tuple[Mock, type[Exception]]:
    """缩小响应行上限，用真实读取流程验证超限处理。"""
    failure = request.param
    writer = Mock(
        drain=AsyncMock(),
        wait_closed=AsyncMock(
            side_effect=ConnectionResetError("connection lost")
            if failure in ("disconnect", "close")
            else None
        ),
    )

    async def connect(path: str, *, limit: int) -> tuple[asyncio.StreamReader, Mock]:
        assert limit == STREAM_LIMIT
        reader = asyncio.StreamReader(limit=64)
        if failure == "disconnect":
            reader.set_exception(ConnectionResetError("connection lost"))
        else:
            reader.feed_data(
                {
                    "oversized": b"x" * 65 + b"\n",
                    "json": b"{bad}\n",
                    "close": b'{"status":"success","data":7}\n',
                }[failure]
            )
            reader.feed_eof()
        return reader, writer

    monkeypatch.setattr("axi.daemon.client.ensure_daemon", lambda: True)
    monkeypatch.setattr(
        "axi.daemon.client.asyncio.open_unix_connection",
        AsyncMock(side_effect=connect),
    )
    error_type = (
        ConnectionResetError if failure in ("disconnect", "close") else ValueError
    )
    return writer, error_type


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["list_tools", "call_tool"])
async def test_daemon_transport_errors_keep_original_propagation(
    failed_transport: tuple[Mock, type[Exception]], method: str
) -> None:
    writer, error_type = failed_transport
    with pytest.raises(error_type):
        await _send(DaemonRequest(method=method))
    writer.close.assert_called_once()


@pytest.mark.parametrize("method", ["list_tools", "search", "grep", "describe"])
def test_documentation_transport_failures_follow_each_callers_policy(
    failed_transport: tuple[Mock, type[Exception]], method: str
) -> None:
    writer, _ = failed_transport
    if method == "describe":
        names = ["first", "second", "first"]
        items = [fetch_description(name, DescribeOptions()) for name in names]
        assert len(items) == len(names)
        assert all("error" in item for item in items)
        assert all("Daemon describe request failed" in item["error"] for item in items)
    else:
        with pytest.raises(RuntimeError, match=f"Daemon {method} request failed"):
            if method == "list_tools":
                fetch_tool_catalog()
            else:
                fetch_discovery(DaemonRequest(method=method, query="temperature"))
    count = len(names) if method == "describe" else 1
    assert writer.write.call_count == writer.close.call_count == count


@pytest.mark.parametrize("command", ["search", "grep", "describe", "list"])
def test_cli_transport_failure_preserves_local_results(
    failed_transport: tuple[Mock, type[Exception]],
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    registry = make_registry(
        make_tool(name="local", server="native", description="temperature")
    )
    monkeypatch.setattr(cli.runtime, "registry", registry)
    monkeypatch.setattr(cli, "app_config", AxiConfig(mcpServers={"remote": {}}))
    monkeypatch.setattr("axi.providers.native.load_native_tool_modules", lambda: None)
    arguments = {
        "search": ["temperature"],
        "grep": ["temperature"],
        "describe": ["local,remote/read,local,remote/read"],
        "list": [],
    }
    result = CliRunner().invoke(cli.app, [command, *arguments[command]])
    assert result.exit_code == 0, result.output
    data = json.loads(result.stdout)
    if command == "describe":
        assert (
            data[0]
            == data[2]
            == registry.describe("local").model_dump(exclude_none=True)
        )
        assert data[1] == data[3]
        assert "Daemon describe request failed" in data[1]["error"]
    elif command == "list":
        assert data == [
            {"server": "native", "tools": ["local"]},
            {"server": "remote", "tools": []},
        ]
    else:
        assert [item["name"] for item in data] == ["native/local"]
        assert f"Daemon {command} request failed" in result.stderr
    writer, _ = failed_transport
    count = 2 if command == "describe" else 1
    assert writer.write.call_count == writer.close.call_count == count


@pytest.mark.parametrize("method", ["list_tools", "summary", "search", "grep"])
@pytest.mark.parametrize("data", [None, []])
def test_empty_catalog_and_discovery_responses(
    monkeypatch: pytest.MonkeyPatch, method: str, data: Any
) -> None:
    request = Mock(return_value=DaemonResponse.success(data))
    monkeypatch.setattr("axi.daemon.client.daemon_request", request)
    if method == "list_tools":
        results = fetch_tool_catalog()
    elif method == "summary":
        results = fetch_tool_summaries()
    else:
        results = fetch_discovery(DaemonRequest(method=method, query="value"))
    assert results == []
    assert request.call_args.args[0].method == (
        "list_tools" if method == "summary" else method
    )


@pytest.mark.parametrize("summary", [False, True])
def test_required_catalog_model_conversion_reports_its_original_error(
    monkeypatch: pytest.MonkeyPatch, summary: bool
) -> None:
    monkeypatch.setattr(
        "axi.daemon.client.daemon_request",
        Mock(return_value=DaemonResponse.success([{"name": "incomplete"}])),
    )
    fetch = fetch_tool_summaries if summary else fetch_tool_catalog
    with pytest.raises(ValueError) as error:
        fetch()
    assert "description" in str(error.value)
    assert "Restart" not in str(error.value)


@pytest.mark.parametrize("message", [None, "Tool not found: remote"])
def test_description_failure_keeps_the_requested_name(
    monkeypatch: pytest.MonkeyPatch, message: str | None
) -> None:
    request = Mock(return_value=DaemonResponse(status="error", error=message))
    monkeypatch.setattr("axi.daemon.client.daemon_request", request)
    assert fetch_description("remote", DescribeOptions()) == {
        "error": message or "Tool not found: remote"
    }
    request.assert_called_once()
    assert request.call_args.args[0].tool_name == "remote"


@pytest.mark.asyncio
async def test_search_uses_a_large_catalog_without_transferring_its_definitions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    socket = str(tmp_path / "daemon.sock")
    monkeypatch.setattr("axi.daemon.client.SOCKET_PATH", socket)
    monkeypatch.setattr("axi.daemon.server.app_config", AxiConfig())
    server = DaemonServer()
    schema = obj(
        {f"field{i}": node("number", "车辆运行统计指标说明" * 25) for i in range(100)}
    )
    for index in range(120):
        server.registry.register(
            make_tool(schema, f"read{index}", server="remote", source="mcp")
        )
    catalog = server._list_tools(DaemonRequest(method="list_tools"))
    assert len(_encode_response(catalog)) > STREAM_LIMIT
    listener = await asyncio.start_unix_server(server._handle_client, path=socket)
    async with listener:
        response = await _send(DaemonRequest(method="search", query="field99", top_k=1))
    assert response.status == "success", response.error
    assert len(response.data) == 1
    assert response.data[0]["matched_fields"] == [
        "$.field99: float；" + "车辆运行统计指标说明" * 25
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "definition,params,error",
    [
        (obj({"message": node("string")}, required=["message"]), {}, "Invalid params"),
        (obj({"message": node("string")}), {"message": 42}, "not connected"),
        (obj({"message": node("string")}), {"message": [1]}, "Invalid params"),
        (obj({"x": ref("missing")}), {"x": 1}, "not connected"),
    ],
)
async def test_daemon_input_validation(
    monkeypatch: pytest.MonkeyPatch, definition: dict, params: dict, error: str
) -> None:
    monkeypatch.setattr("axi.daemon.server.app_config", AxiConfig())
    server = DaemonServer()
    server.registry.register(
        make_tool(name="query", server="remote", source="mcp", input_schema=definition)
    )
    result = await server._handle_call_tool(
        DaemonRequest(method="call_tool", tool_name="remote/query", params=params)
    )
    assert result.status == "error" and error in result.error


def test_coercions_preserve_invalid_values_for_validation() -> None:
    schema = obj(
        {
            "msg": node("string"),
            "port": node("integer"),
            "rate": node("number"),
            "on": node("boolean"),
        }
    )
    assert _coerce_to_schema(
        {"msg": 42, "port": "08080", "rate": "1.5", "on": "true"}, schema
    ) == {"msg": "42", "port": 8080, "rate": 1.5, "on": True}
    assert _coerce_to_schema({"port": "abc", "msg": [1]}, schema) == {
        "port": "abc",
        "msg": [1],
    }


@pytest.mark.parametrize("pid", [None, 4_000_000, "current"])
def test_read_live_pid_and_socket_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pid: int | str | None
) -> None:
    current = os.getpid()
    value = current if pid == "current" else pid
    pid_path, sock_path = tmp_path / "daemon.pid", tmp_path / "daemon.sock"
    if value is not None:
        pid_path.write_text(str(value))
    monkeypatch.setattr(client, "PID_PATH", str(pid_path))
    monkeypatch.setattr(client, "SOCKET_PATH", str(sock_path))
    assert client.read_live_pid() == (current if pid == "current" else None)
    assert not client.is_daemon_running()
    sock_path.touch()
    assert client.is_daemon_running() is (pid == "current")


@pytest.mark.parametrize("exited", [False, True])
def test_spawn_waits_only_while_child_is_alive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exited: bool
) -> None:
    process = Mock()
    process.poll.return_value = 1 if exited else None
    monkeypatch.setattr(client, "LOG_PATH", str(tmp_path / "daemon.log"))
    monkeypatch.setattr(client, "is_daemon_running", lambda: False)
    monkeypatch.setattr(client.subprocess, "Popen", Mock(return_value=process))
    monkeypatch.setattr(client, "_DAEMON_START_TIMEOUT", 5.0 if exited else 0.3)
    started = time.monotonic()
    assert client._spawn_and_wait() is False
    elapsed = time.monotonic() - started
    assert elapsed < 1.0 if exited else elapsed >= 0.3


@pytest.mark.parametrize(
    "pid,expected",
    [(4242, {"status": "starting", "pid": 4242}), (None, {"status": "stopped"})],
)
def test_status_reports_initialization_state(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    pid: int | None,
    expected: dict,
) -> None:
    monkeypatch.setattr(cli, "is_daemon_running", lambda: False)
    monkeypatch.setattr(cli, "read_live_pid", lambda: pid)
    cli.daemon_status()
    assert json.loads(capsys.readouterr().out) == expected


def test_start_reports_initializing_not_failed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "is_daemon_running", lambda: False)
    monkeypatch.setattr(cli, "ensure_daemon", lambda: False)
    monkeypatch.setattr(cli, "read_live_pid", lambda: 4242)
    with pytest.raises(typer.Exit):
        cli.daemon_start()
    error = capsys.readouterr().err
    assert "still initializing" in error and "Failed" not in error


@pytest.mark.parametrize("pids", [[1234, 1234, None], [1234, 5678], [1234]])
def test_wait_for_exit_tracks_original_pid(
    monkeypatch: pytest.MonkeyPatch, pids: list[int | None]
) -> None:
    readings = iter(pids)
    monkeypatch.setattr(client, "read_live_pid", lambda: next(readings, pids[-1]))
    started = time.monotonic()
    assert client.wait_for_exit(1234, timeout=0.2) is (pids[-1] != 1234)
    if pids[-1] == 1234:
        assert time.monotonic() - started >= 0.2


@pytest.mark.parametrize("ready,exits", [(True, True), (False, True), (True, False)])
def test_stop_waits_for_original_process(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    ready: bool,
    exits: bool,
) -> None:
    monkeypatch.setattr(cli, "read_live_pid", lambda: 1234)
    monkeypatch.setattr(cli, "is_daemon_running", lambda: ready)

    def shutdown(request: DaemonRequest) -> DaemonResponse:
        assert request.method == "shutdown"
        monkeypatch.setattr(cli, "read_live_pid", lambda: 5678)
        return DaemonResponse.success(None)

    send, kill, wait = Mock(side_effect=shutdown), Mock(), Mock(return_value=exits)
    monkeypatch.setattr(cli, "send_request", send)
    monkeypatch.setattr(cli.os, "kill", kill)
    monkeypatch.setattr(cli, "wait_for_exit", wait)
    if exits:
        cli.daemon_stop()
        assert "Daemon stopped." in capsys.readouterr().out
    else:
        with pytest.raises(typer.Exit):
            cli.daemon_stop()
        assert "did not exit" in capsys.readouterr().err
    wait.assert_called_once_with(1234)
    if ready:
        send.assert_called_once()
        kill.assert_not_called()
    else:
        send.assert_not_called()
        kill.assert_called_once_with(1234, signal.SIGTERM)
