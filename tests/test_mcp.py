"""验证真实 stdio 导入与调用，以及远程输出 Schema 仅作文档的边界。"""

import asyncio
import os
import signal
import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock

import mcp.types as types
import pytest

import axi.mcp_serve as serve
from axi.config import AxiConfig
from axi.providers.mcp import MCPConnection, MCPProvider, MCPServerConfig
from tests.support import node, obj, make_registry

GOOD = obj({"answer": node("integer", "Answer")})


@pytest.mark.asyncio
async def test_real_mcp_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    config = {
        "mcpServers": {
            "test": {
                "command": sys.executable,
                "args": [str(Path(__file__).with_name("mock_mcp_server.py"))],
            }
        }
    }
    monkeypatch.setattr(
        "axi.providers.mcp.app_config", AxiConfig.model_validate(config)
    )
    provider = MCPProvider()
    try:
        tools = await provider.connect_all(provider.load_config())
        assert [tool.full_name for tool in tools] == [
            "test/echo",
            "test/add",
            "test/boom",
        ]
        assert tools[0].source == "mcp" and tools[0].input_schema["required"] == [
            "message"
        ]
        for name, params, expected in [
            ("echo", {"message": "hello axi"}, "hello axi"),
            ("add", {"a": 3, "b": 5}, 8),
        ]:
            result = await provider.call_tool("test", name, params)
            assert result.status == "success" and result.data == expected
        error = await provider.call_tool("test", "boom", {})
        assert error.status == "error" and "kaboom" in error.error
    finally:
        await provider.close_all()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "documentation,expected",
    [
        ({"output_schema": GOOD}, GOOD),
        ({"output_schema": {}}, {}),
        (
            {"output_schema": GOOD, "meta": {"axi/output_schema": node("string")}},
            GOOD,
        ),
        ({"meta": {"axi/output_schema": GOOD}}, GOOD),
        ({"output_schema": node("invalid")}, node("invalid")),
        (
            {"output_schema": obj({"speed": node("number", minimum="unused")})},
            obj({"speed": node("number", minimum="unused")}),
        ),
        ({"meta": {"axi/output_schema": ["x"]}}, None),
    ],
)
async def test_remote_output_definitions(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    documentation: dict,
    expected: dict | None,
) -> None:
    session = AsyncMock()
    input_schema = obj({"output_fields": node("string")})
    session.list_tools.side_effect = [
        types.ListToolsResult(
            tools=[
                types.Tool(name="query", input_schema=input_schema, **documentation)
            ],
            next_cursor="more",
        ),
        types.ListToolsResult(
            tools=[
                types.Tool(
                    name="other",
                    input_schema={},
                    meta={"axi/output_schema": GOOD},
                )
            ]
        ),
    ]
    session.call_tool.return_value = types.CallToolResult(
        content=[types.TextContent(type="text", text="Query succeeded")],
        structured_content={"answer": 7},
    )

    async def connect(connection: MCPConnection) -> None:
        connection.session = session

    monkeypatch.setattr(MCPConnection, "connect", connect)
    provider = MCPProvider()
    tools = await provider.connect_all(
        [MCPServerConfig(server="remote", command="unused")]
    )
    try:
        assert [tool.name for tool in tools] == ["query", "other"]
        assert tools[0].output_schema == expected
        assert tools[0].input_schema == input_schema
        assert tools[1].output_schema == GOOD
        registry = make_registry(tools[0])
        for discover in (registry.search, registry.grep):
            hits = discover("answer")
            assert ([hit.matched_fields for hit in hits] if hits else []) == (
                [["$.answer: int；Answer"]] if expected == GOOD else []
            )
        result = await provider.call_tool("remote", "query", {})
        assert result.status == "success" and result.data == {"answer": 7}
        exported = serve._export_tool("query", tools[0])
        assert exported.output_schema is None and exported.input_schema == input_schema
        if expected is not None:
            assert exported.meta == {"axi/output_schema": expected}
        else:
            assert "Ignoring output documentation" in caplog.text
    finally:
        await provider.close_all()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "texts,structured,is_error,expected",
    [
        ([], {"answer": 7}, False, {"answer": 7}),
        (["Query succeeded"], {"answer": 7}, False, {"answer": 7}),
        (['{"answer":8}'], {"answer": 7}, False, {"answer": 7}),
        (["Query succeeded"], {}, False, {}),
        (["Query succeeded"], [1, 2], False, [1, 2]),
        (["Query succeeded"], 7, False, 7),
        (["Query succeeded"], False, False, False),
        (["Query succeeded"], 0, False, 0),
        (["Query succeeded"], [], False, []),
        (["first", "second"], {"answer": 7}, False, {"answer": 7}),
        (['{"answer":7}'], None, False, {"answer": 7}),
        (["Query succeeded"], None, False, "Query succeeded"),
        (["first", "second"], None, False, ["first", "second"]),
        ([], None, False, []),
        (["Query failed"], {"answer": 7}, True, "Query failed"),
        ([], {}, True, "Unknown tool error"),
    ],
)
async def test_mcp_result_parsing(
    texts: list[str], structured: Any, is_error: bool, expected: Any
) -> None:
    conn = MCPConnection(MCPServerConfig(server="test", command="unused"))
    session = AsyncMock()
    response = types.CallToolResult(
        content=[types.TextContent(type="text", text=text) for text in texts],
        structured_content=structured,
        is_error=is_error,
    )
    session.call_tool.return_value = response
    conn.session = session
    provider = MCPProvider()
    provider._connections["test"] = conn

    result = await provider.call_tool("test", "query", {})
    if is_error:
        assert result.status == "error" and result.error == expected
    else:
        assert result.status == "success" and result.data == expected
        if structured is not None:
            assert result.data is response.structured_content
    session.call_tool.assert_awaited_once_with("query", {})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "structured,is_error,expected",
    [
        ({"answer": 7}, False, {"answer": 7}),
        ({}, False, {}),
        ({"answer": 7}, True, "Image content"),
        (None, False, "Image content"),
    ],
)
async def test_mcp_mixed_result_preserves_structured_data_and_error_priority(
    monkeypatch: pytest.MonkeyPatch,
    structured: dict[str, Any] | None,
    is_error: bool,
    expected: Any,
) -> None:
    render = Mock(return_value="Image content")
    monkeypatch.setattr(types.ImageContent, "__str__", render)
    conn = MCPConnection(MCPServerConfig(server="test", command="unused"))
    conn.session = AsyncMock()
    conn.session.call_tool.return_value = types.CallToolResult(
        content=[types.ImageContent(type="image", data="AAAA", mime_type="image/png")],
        structured_content=structured,
        is_error=is_error,
    )
    provider = MCPProvider()
    provider._connections["test"] = conn

    result = await provider.call_tool("test", "query", {})
    assert result.status == ("error" if is_error else "success")
    assert (result.error if is_error else result.data) == expected
    conn.session.call_tool.assert_awaited_once_with("query", {})


@pytest.fixture
def mock_config() -> MCPServerConfig:
    return MCPServerConfig(
        server="test-server",
        command=sys.executable,
        args=[str(Path(__file__).with_name("mock_mcp_server.py"))],
    )


@pytest.mark.asyncio
async def test_reconnect_from_concurrent_request_tasks(mock_config, tmp_path):
    pid_path = tmp_path / "worker.pid"
    config = mock_config.model_copy(
        update={"env": {"AXI_TEST_PID_PATH": str(pid_path)}}
    )
    provider = MCPProvider()
    try:
        await provider.connect_all([config])
        old_pid = int(pid_path.read_text())
        os.kill(old_pid, signal.SIGKILL)
        await asyncio.sleep(0.1)
        results = await asyncio.wait_for(
            asyncio.gather(
                provider.call_tool(config.server, "echo", {"message": "one"}),
                provider.call_tool(config.server, "echo", {"message": "two"}),
            ),
            timeout=10,
        )
        assert [r.status for r in results] == ["success", "success"]
        assert [r.data for r in results] == ["one", "two"]
        assert int(pid_path.read_text()) != old_pid
    finally:
        await provider.close_all()


@pytest.mark.asyncio
async def test_list_failure_is_cleaned_up_and_does_not_block_other_servers(
    mock_config, tmp_path
):
    pid_path = tmp_path / "failed-worker.pid"
    bad = mock_config.model_copy(
        update={
            "server": "bad",
            "env": {"AXI_TEST_LIST_ERROR": "1", "AXI_TEST_PID_PATH": str(pid_path)},
        }
    )
    provider = MCPProvider()
    try:
        tools = await asyncio.wait_for(provider.connect_all([bad, mock_config]), 10)
        assert len(tools) == 3
        assert {t.server for t in tools} == {mock_config.server}
        with pytest.raises(ProcessLookupError):
            os.kill(int(pid_path.read_text()), 0)
        result = await asyncio.create_task(
            provider.call_tool(mock_config.server, "echo", {"message": "healthy"})
        )
        assert result.data == "healthy"
    finally:
        await provider.close_all()


@pytest.mark.asyncio
async def test_list_tools_follows_pagination(mock_config):
    conn = MCPConnection(
        mock_config.model_copy(update={"env": {"AXI_TEST_PAGINATED": "1"}})
    )
    try:
        await conn.connect()
        tools = await conn.list_tools()
        assert [t.name for t in tools] == ["echo", "add", "boom"]
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_inflight_failure_does_not_reconnect_after_shutdown(
    mock_config, monkeypatch
):
    provider = MCPProvider()
    await provider.connect_all([mock_config])
    conn = provider._connections[mock_config.server]
    started = asyncio.Event()

    async def failed_call(name, params):
        started.set()
        raise ConnectionError("MCP disconnected")

    monkeypatch.setattr(conn, "call_tool", failed_call)
    async with conn.reconnect_lock:
        task = asyncio.create_task(
            provider.call_tool(mock_config.server, "echo", {"message": "late"})
        )
        await started.wait()
        shutdown = asyncio.create_task(provider.close_all())
        await asyncio.sleep(0)
    try:
        await shutdown
        result = await asyncio.wait_for(task, 10)
        assert result.status == "error"
        assert conn.session is None
    finally:
        await task
        await conn.close()
