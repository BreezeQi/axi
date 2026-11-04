"""验证注册、入参校验及原生、CLI、MCP 原有执行行为。"""

import asyncio
import inspect
import json
import sys
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from types import ModuleType
from typing import Annotated, Any, Literal

import anyio
import pytest
from mcp import ClientSession
from mcp.server.lowlevel import Server
from mcp.shared.memory import create_client_server_memory_streams
from pydantic import BaseModel, Field, RootModel, ValidationError, field_serializer
from typer.testing import CliRunner

import axi
import axi.cli as cli
from axi import runtime, tool
from axi.daemon.protocol import DaemonResponse
from axi.context import set_request_env
from axi.executor import Executor
from axi.mcp_serve import _build_flat_server, _build_meta_server
from axi.providers.native import _load_native_entry, get_native_function, register_tool
from axi.registry import Registry
from tests.support import describe_tool, make_tool, node, obj, make_registry


def test_public_tool_registration_and_lookup() -> None:
    @tool(description="向用户打招呼")
    def greet(name: str, greeting: str = "Hello") -> dict[str, str]:
        return {"message": f"{greeting}, {name}!"}

    registry = runtime.registry
    meta = registry.get("greet")
    assert meta.description == "向用户打招呼"
    assert meta.input_schema["required"] == ["name"]
    assert meta.input_schema["properties"]["greeting"]["default"] == "Hello"
    assert tool("greet")(name="PTC") == greet("PTC")
    assert runtime.executor.run("greet", {"name": "World"}).data == greet("World")
    assert registry.search("打招呼")[0].name == "greet"
    assert registry.grep("^greet")[0].name == "greet"


def test_annotated_input_schema() -> None:
    class Filter(BaseModel):
        min_price: float

    @tool(description="查询商品")
    def query(
        region: Annotated[Literal["cn", "jp"], Field(description="地区")],
        keywords: list[str] = [],
        filter: Filter | None = None,
        page: Annotated[int, Field(ge=1, le=100, description="页码")] = 1,
    ) -> None:
        pass

    schema = runtime.registry.get("query").input_schema
    props = schema["properties"]
    assert props["region"] == {
        "enum": ["cn", "jp"],
        "type": "string",
        "description": "地区",
    }
    assert props["keywords"]["items"] == {"type": "string"}
    assert schema["required"] == ["region"]
    assert props["filter"]["anyOf"][0] == {"$ref": "#/$defs/Filter"}
    assert schema["$defs"]["Filter"]["properties"]["min_price"]["type"] == "number"
    assert props["page"] == {
        "type": "integer",
        "default": 1,
        "minimum": 1,
        "maximum": 100,
        "description": "页码",
    }


class Range(BaseModel):
    start: int
    end: int


def compute(x: int, label: str = "default", span: Range | None = None) -> tuple:
    return x, label, span.end - span.start if span else None


@pytest.mark.parametrize(
    ("function", "params", "expected"),
    [
        (compute, {"x": "3", "label": 42}, (3, "42", None)),
        (compute, {"x": 1, "span": {"start": 1, "end": 5}}, (1, "default", 4)),
    ],
)
def test_native_input_conversion(
    function: Callable, params: dict, expected: Any
) -> None:
    registry = make_registry(register_tool(function))
    result = Executor(registry).run(function.__name__, params)
    assert result.status == "success" and result.data == expected


@pytest.mark.parametrize("params", [{}, {"x": "abc"}, {"x": 1, "typo": 2}])
def test_invalid_native_input(params: dict) -> None:
    registry = make_registry(register_tool(compute))
    result = Executor(registry).run("compute", params)
    assert result.status == "error" and "Invalid params" in result.error


def test_execution_errors() -> None:
    def broken() -> None:
        raise RuntimeError("boom")

    registry = make_registry(register_tool(broken))
    result = Executor(registry).run("broken", {})
    assert result.status == "error" and "RuntimeError: boom" in result.error


def test_same_name_modules_keep_separate_functions(tmp_path: Path) -> None:
    registry = runtime.registry
    loaded = {}
    modules = {"mod_a": "A", "mod_b": "B"}
    for module, result in modules.items():
        path = tmp_path / module / "tools.py"
        path.parent.mkdir()
        path.write_text(
            f"from axi import tool\n@tool()\ndef shadow_ping() -> tuple[str, str]:\n"
            f"    return __name__, '{result}'\n"
        )
        _load_native_entry(registry, str(path), module, loaded, "test")
    for module, result in modules.items():
        function = get_native_function(f"{module}/shadow_ping")
        assert function.__module__ == "tools"
        assert Executor(registry).run(f"{module}/shadow_ping", {}).data == (
            "tools",
            result,
        )


@pytest.mark.parametrize("filename", ["output_tools.py", "json.py"])
def test_file_output_models_resolve_postponed_annotations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename: str
) -> None:
    monkeypatch.setitem(sys.modules, "json", json)
    path = tmp_path / filename
    path.write_text(
        "from __future__ import annotations\n"
        "from json import dumps, loads\n"
        "from axi import tool\n"
        "from pydantic import BaseModel, Field\n"
        "class Row(BaseModel):\n"
        "    speed: float = Field(description='速度')\n"
        "class Report(BaseModel):\n"
        "    rows: list[Row]\n"
        "@tool(output_schema=Report)\n"
        "def read() -> dict:\n"
        "    return loads(dumps({'rows': [{'speed': 2.0}]}))\n"
    )
    registry = runtime.registry
    _load_native_entry(registry, str(path), "fleet", {}, "test")
    assert sys.modules["json"] is json
    expected = "{rows: [{speed: float；速度}]}"
    assert registry.describe("fleet/read", fields=["$.rows"]).output_fields == expected
    assert registry.search("$.rows[*].speed")[0].matched_fields == [
        "$.rows[*].speed: float；速度"
    ]
    assert registry.grep(r"rows\[\*\]\.speed")[0].matched_fields == [
        "$.rows[*].speed: float；速度"
    ]
    assert Executor(registry).run("fleet/read", {}).data == {"rows": [{"speed": 2.0}]}


def test_imported_output_model_uses_its_own_type_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models = ModuleType("output_models")
    monkeypatch.setitem(sys.modules, models.__name__, models)
    exec(
        "from __future__ import annotations\n"
        "from pydantic import BaseModel\n"
        "class Report(BaseModel):\n    item: Item\n"
        "class Item(BaseModel):\n    origin: int\n",
        models.__dict__,
    )
    caller = {"__name__": "tools"}
    exec(
        "from pydantic import BaseModel\n"
        "class Item(BaseModel):\n    other: str\n"
        "def read() -> dict:\n    return {}\n",
        caller,
    )
    meta = register_tool(caller["read"], output_schema=models.Report)
    assert (
        describe_tool(meta, fields=["$.item"]).output_fields == "{item: {origin: int}}"
    )


def test_file_generic_output_model_resolves_recursive_annotations(
    tmp_path: Path,
) -> None:
    path = tmp_path / "recursive.py"
    path.write_text(
        "from __future__ import annotations\n"
        "from axi import tool\n"
        "from pydantic import BaseModel, Field, RootModel\n"
        "class Child(BaseModel):\n    parent: Parent | None = None\n"
        "class Parent(BaseModel):\n"
        "    speed: float = Field(description='速度')\n    child: Child\n"
        "@tool(output_schema=RootModel[list[Parent]])\n"
        "def read() -> list[dict]:\n"
        "    return [{'speed': 2.0, 'child': {'parent': None}}]\n"
    )
    registry = runtime.registry
    _load_native_entry(registry, str(path), "recursive", {}, "test")
    assert registry.describe("recursive/read", fields=["$[*].speed"]).output_fields == (
        "[{speed: float；速度}]"
    )
    assert registry.search("$[*].speed")[0].matched_fields == [
        "$[*].speed: float；速度"
    ]
    assert Executor(registry).run("recursive/read", {}).data == [
        {"speed": 2.0, "child": {"parent": None}}
    ]


@pytest.mark.parametrize("with_schema", [False, True])
def test_output_documentation_preserves_execution(with_schema: bool) -> None:
    data = {
        "columns": ["value"],
        "values": [[7]],
        "constants": {"plate": "A"},
        "extra": 8,
    }

    def query(limit: int = 10) -> dict[str, Any]:
        return data

    meta = register_tool(
        query,
        output_schema=obj({"missing": node("integer", "必填值")}, required=["missing"])
        if with_schema
        else None,
    )
    assert get_native_function(meta.name) is query
    assert set(meta.input_schema["properties"]) == {"limit"}
    registry = make_registry(meta)
    result = Executor(registry).run(meta.name, {"limit": "2"})
    assert result.status == "success" and result.data is data
    assert describe_tool(meta).output_fields == (
        "{missing: int；必填值}" if with_schema else None
    )
    rejected = Executor(registry).run(meta.name, {"output_fields": ["missing"]})
    assert rejected.status == "error" and "output_fields" in rejected.error


def test_output_models_and_business_parameters_are_not_rewritten() -> None:
    class Actual(BaseModel):
        value: int = Field(serialization_alias="id")

    value = Actual(value=7)

    def query(output_fields: str, *args: Any, **kwargs: Any) -> Any:
        assert output_fields == "business"
        return value, args, kwargs

    meta = register_tool(query, output_schema=node("integer"))
    direct = get_native_function(meta.name)("business", "positional", flag=True)
    assert direct == (value, ("positional",), {"flag": True}) and direct[0] is value
    registry = make_registry(meta)
    result = Executor(registry).run(
        meta.name, {"output_fields": "business", "flag": True}
    )
    assert result.data == (value, (), {"flag": True}) and result.data[0] is value
    assert meta.input_schema["properties"]["output_fields"]["type"] == "string"


@pytest.mark.parametrize("mode", ["sync", "async", "awaitable"])
def test_decorator_keeps_function_identity_and_async_behavior(mode: str) -> None:
    value = {"value": 7, "extra": 8}

    async def async_query() -> dict:
        return value

    def sync_query() -> dict:
        return value

    def awaitable_query() -> Any:
        return async_query()

    original = {"sync": sync_query, "async": async_query, "awaitable": awaitable_query}[
        mode
    ]
    registered = tool(
        name="original_" + mode, description="Read", output_schema=node("object")
    )(original)
    assert registered is original
    result = registered()
    assert (asyncio.run(result) if inspect.isawaitable(result) else result) is value
    assert runtime.executor.run("original_" + mode, {}).data is value


@asynccontextmanager
async def _session(server: Server) -> AsyncIterator[ClientSession]:
    async with create_client_server_memory_streams() as (client, peer):
        async with anyio.create_task_group() as tasks:
            tasks.start_soon(server.run, *peer, server.create_initialization_options())
            try:
                async with ClientSession(*client) as session:
                    await session.initialize()
                    yield session
            finally:
                tasks.cancel_scope.cancel()


@pytest.mark.asyncio
@pytest.mark.parametrize("flat", [False, True])
@pytest.mark.parametrize(
    ("value", "with_schema"),
    [
        pytest.param(None, True, id="null"),
        pytest.param("42", True, id="text"),
        pytest.param(False, False, id="bool-without-schema"),
        pytest.param([1, None], True, id="array"),
        pytest.param({"forwardMileage": "未知", "extra": 8}, True, id="dict"),
    ],
)
async def test_mcp_output_documentation_does_not_change_results(
    flat: bool, with_schema: bool, value: Any
) -> None:
    calls = []

    def read() -> Any:
        calls.append(True)
        return value

    meta = register_tool(
        read,
        name="read",
        output_schema=RootModel[int] if with_schema else None,
    )
    registry = make_registry(meta)
    server = (_build_flat_server if flat else _build_meta_server)(
        registry, Executor(registry)
    )
    async with _session(server) as session:
        listing = await session.list_tools()
        assert all(tool.output_schema is None for tool in listing.tools)
        if flat and with_schema:
            exported = next(tool for tool in listing.tools if tool.name == meta.name)
            assert exported.meta == {"axi/output_schema": meta.output_schema}
        result = await session.call_tool(
            meta.name if flat else "run", {} if flat else {"name": meta.name}
        )
        assert not result.is_error
        assert "structured_content" not in result.model_fields_set
        if flat and isinstance(value, str):
            assert result.content[0].text == value
        else:
            expected = value if flat else {"status": "success"}
            if not flat and value is not None:
                expected["data"] = value
            assert json.loads(result.content[0].text) == expected
    assert calls == [True]


@pytest.mark.parametrize("with_schema", [False, True])
@pytest.mark.parametrize("null", [False, True])
def test_cli_legacy_dump_rules(
    monkeypatch: pytest.MonkeyPatch, with_schema: bool, null: bool
) -> None:
    class Row(BaseModel):
        identifier: int = Field(serialization_alias="id")
        label: str = "ready"
        optional: str | None = None

        @field_serializer("label", when_used="json")
        def uppercase(self, value: str) -> str:
            return value.upper()

    value = None if null else Row(identifier=7)
    meta = register_tool(
        lambda: value,
        name="read",
        output_schema=RootModel[str] if with_schema else None,
    )
    registry = make_registry(meta)
    monkeypatch.setattr(cli.runtime, "registry", registry)
    monkeypatch.setattr(cli.runtime, "executor", Executor(registry))
    result = CliRunner().invoke(cli.app, ["run", "read", "--json", "{}"])
    expected = {"status": "success"}
    if not null:
        expected["data"] = {"identifier": 7, "label": "ready"}
    assert result.exit_code == 0 and json.loads(result.stdout) == expected


@pytest.mark.parametrize("metadata", [{"name": ""}, {"name": "a/b"}, {"source": "mcp"}])
def test_invalid_tool_identity(metadata: dict) -> None:
    with pytest.raises(ValidationError):
        make_tool(**metadata)


def test_request_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AXI_TEST_VAR", "from-env")
    set_request_env(None)
    try:
        assert axi.env("AXI_TEST_VAR") == "from-env"
        assert axi.env("AXI_NO_SUCH_VAR", "fallback") == "fallback"
        set_request_env(
            {
                "axi-axi-test-var": "from-header",
                "axi-github-token": "token",
                "authorization": "hidden",
                "content-type": "hidden",
            }
        )
        assert axi.env("AXI_TEST_VAR") == "from-header"
        assert axi.env("GITHUB_TOKEN") == "token"
        assert axi.env("AUTHORIZATION") is None
        assert axi.env("CONTENT_TYPE") is None
    finally:
        set_request_env(None)


@pytest.mark.parametrize("with_schema", [False, True])
def test_named_decorator_without_description(with_schema: bool) -> None:
    @tool(name="named_only", output_schema=RootModel[int] if with_schema else None)
    def original(value: int) -> int:
        """Use the function docstring as the description."""
        return value * 2

    assert original(3) == 6
    meta = runtime.registry.get("named_only")
    assert meta.description == original.__doc__
    assert tool("named_only")(value=3) == 6


def test_positional_decorator_metadata_stays_supported():
    @tool("positional_metadata", "Legacy positional description")
    def original() -> str:
        return "ok"

    assert (
        runtime.registry.get("positional_metadata").description
        == "Legacy positional description"
    )
    assert original() == "ok"


@pytest.mark.parametrize("known", [False, True])
def test_mcp_execution_uses_daemon(monkeypatch, known):
    registry = Registry()
    if known:
        registry.register(
            make_tool(name="echo", server="srv", description="", source="mcp")
        )
    calls = []

    def request(req):
        calls.append(req)
        return DaemonResponse.success("remote result")

    monkeypatch.setattr("axi.executor.daemon_request", request)
    result = Executor(registry).run("srv/echo", {"message": "hi"})
    assert result.data == "remote result"
    assert calls[0].tool_name == "srv/echo"
    assert calls[0].params == {"message": "hi"}
