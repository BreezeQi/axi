"""验证 describe 的选择、批量、示例编码及各入口一致性。"""

import asyncio
import json
from collections.abc import Callable
from threading import Event
from typing import Any
from unittest.mock import Mock, patch

import jsonschema
import pytest
from pydantic import BaseModel
from typer.testing import CliRunner

import axi.cli as cli
from axi.config import AxiConfig
from axi.daemon.protocol import DaemonRequest, DaemonResponse
from axi.daemon.server import DaemonServer
from axi.mcp_serve import (
    _META_TOOLS,
    _build_flat_server,
    _build_meta_server,
    _tool_description,
)
from axi.models import ToolMeta, ToolSource
from axi.output.describe import ToolDescriber
from axi.output.selection import DescribeSelection
from axi.providers.native import register_tool
from axi.registry import Registry
from tests.support import composition, make_tool, node, obj, arr, make_registry
from tests.support import view_text


async def describe_entry(
    registry: Registry, transport: str, call: Callable, **options: Any
) -> Any:
    name = options.pop("name")
    if transport == "registry":
        results = [
            item.model_dump(exclude_none=True)
            for item in registry.describe_many(name.split(","), **options)
        ]
        return results[0] if "," not in name else results
    if transport == "cli":
        args = ["describe", name]
        if "fields" in options:
            args += ["--fields", json.dumps(options["fields"])]
        if options.get("schema"):
            args += ["--schema"]
        with patch.object(cli.runtime, "registry", registry):
            result = CliRunner().invoke(cli.app, args)
        assert result.exit_code == 0, result.output
        return json.loads(result.stdout)
    if transport == "daemon":
        server = DaemonServer()
        server.registry = registry
        result = DaemonResponse.model_validate_json(
            await server._dispatch(
                DaemonRequest(method="describe", tool_name=name, **options)
            )
        )
        assert result.status == "success", result.error
        return result.data
    server = (_build_flat_server if transport == "flat" else _build_meta_server)(
        registry, None
    )
    result = await call(
        server,
        "axi_describe" if transport == "flat" else "describe",
        {"name": name, **options},
    )
    assert not result.is_error
    return json.loads(result.content[0].text)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["registry", "cli", "daemon", "flat", "meta"])
@pytest.mark.parametrize("options", [{}, {"fields": None}, {"schema": False}])
async def test_describe_defaults_to_inputs_without_preparing_output_documentation(
    nested_meta: ToolMeta,
    transport: str,
    options: dict[str, Any],
    mcp_call: Callable,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nested_meta.input_schema = obj(
        {"identifier": node("string")}, required=["identifier"]
    )
    nested_meta.output_example = {"example": 7}
    registry = make_registry(nested_meta)
    monkeypatch.setattr(
        registry,
        "_documentation",
        Mock(side_effect=AssertionError("Inputs prepared output docs")),
    )
    result = await describe_entry(registry, transport, mcp_call, name="read", **options)
    assert result["name"] == "read"
    assert result["description"] == nested_meta.description
    assert result["input_schema"] == nested_meta.input_schema
    assert not any(key.startswith("output_") for key in result)
    schema = next(tool.input_schema for tool in _META_TOOLS if tool.name == "describe")
    assert set(schema["properties"]) == {"name", "fields", "schema"}
    assert not {"view", "depth"} & DaemonRequest.model_fields.keys()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["registry", "cli", "daemon", "flat", "meta"])
@pytest.mark.parametrize("source", list(ToolSource))
async def test_full_schema_is_read_without_preparing_documents_or_running_tools(
    nested_meta: ToolMeta,
    transport: str,
    source: ToolSource,
    mcp_call: Callable,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    schema = nested_meta.output_schema
    schema.update(
        {
            "required": ["summaryStatistic"],
            "additionalProperties": False,
            "x-vendor": {"labels": ["complete", "original"]},
        }
    )
    schema["properties"]["a.b"] = {
        "anyOf": [node("number", minimum=0), node("string", enum=["offline"])],
        "description": "Reading",
    }
    schema["$defs"]["WorkTime"]["properties"]["next"] = {
        "anyOf": [{"$ref": "#/$defs/WorkTime"}, node("null")]
    }
    nested_meta.source = source
    nested_meta.server = "demo"
    nested_meta.output_example = {"example": 7}
    registry = make_registry(nested_meta)
    forbidden = Mock(side_effect=AssertionError("Full schema lookup ran extra work"))
    monkeypatch.setattr(registry, "_documentation", forbidden)
    monkeypatch.setattr("axi.executor.Executor.run", forbidden)
    monkeypatch.setattr("axi.providers.mcp.MCPProvider.call_tool", forbidden)
    expected = {
        "name": "read",
        "description": nested_meta.description,
        "source": source,
        "output_schema_status": "registered",
        "output_schema": schema,
    }
    if transport != "flat":
        expected["server"] = "demo"
    assert (
        await describe_entry(registry, transport, mcp_call, name="read", schema=True)
        == expected
    )
    batch = await describe_entry(
        registry,
        transport,
        mcp_call,
        name="read,missing,read",
        schema=True,
        fields=None,
    )
    assert batch[0] == batch[2] == expected
    assert "Tool not found" in batch[1]["error"]
    forbidden.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["registry", "cli", "daemon", "flat", "meta"])
@pytest.mark.parametrize("definition", [None, {}])
async def test_full_schema_distinguishes_missing_and_empty_definitions(
    transport: str, definition: dict | None, mcp_call: Callable
) -> None:
    registry = make_registry(make_tool(definition), make_tool(obj({}), "documented"))
    result = await describe_entry(
        registry, transport, mcp_call, name="read", schema=True
    )
    expected = {"name": "read", "description": "", "source": "native"}
    expected["output_schema_status"] = (
        "not_registered" if definition is None else "registered"
    )
    if definition is not None:
        expected["output_schema"] = definition
    assert result == expected


def test_full_schema_responses_do_not_share_mutable_definitions() -> None:
    schema = obj({"value": node("number", "Original")})
    registry = make_registry(make_tool(schema))
    first, second = registry.describe_many(["read", "read"], schema=True)
    first.output_schema["properties"]["value"]["description"] = "Changed"
    assert second.output_schema == schema
    assert registry.resolve("read").output_schema == schema
    assert registry.describe("read", schema=True).output_schema == schema


@pytest.mark.parametrize("fields", [[], ["$"]])
def test_full_schema_rejects_field_selections(fields: list[str]) -> None:
    registry = make_registry(make_tool(obj({})))
    with pytest.raises(ValueError, match="cannot be combined"):
        registry.describe("read", schema=True, fields=fields)
    missing, first, repeated = registry.describe_many(
        ["missing", "read", "read"], schema=True, fields=fields
    )
    assert "Tool not found" in missing.error
    assert first == repeated and "cannot be combined" in first.error
    with patch.object(cli.runtime, "registry", registry):
        result = CliRunner().invoke(
            cli.app, ["describe", "read", "--schema", "--fields", json.dumps(fields)]
        )
    assert result.exit_code == 1
    assert "cannot be combined" in json.loads(result.stdout)["error"]
    daemon = DaemonServer()
    daemon.registry = registry
    response = daemon._describe(
        DaemonRequest(method="describe", tool_name="read", schema=True, fields=fields)
    )
    assert response.status == "error" and "cannot be combined" in response.error


@pytest.mark.parametrize("transport", ["registry", "mcp", "daemon"])
@pytest.mark.parametrize(
    ("options", "valid"),
    [
        ({}, True),
        ({"fields": None}, True),
        ({"schema": True}, True),
        ({"schema": False, "fields": []}, True),
        ({"schema": "yes"}, False),
        ({"fields": []}, True),
        ({"fields": ["$"]}, True),
        ({"fields": [7]}, False),
        ({"fields": ["$.statistics.events[*].distance"]}, True),
        ({"fields": ["$.statistics"]}, True),
        ({"fields": {}}, False),
        ({"fields": {"statistics": ["distance"]}}, False),
        ({"fields": {"distance": "number；距离"}}, False),
    ],
)
def test_describe_input_contract(options: dict, valid: bool, transport: str) -> None:
    if transport == "registry":
        if valid:
            DescribeSelection.prepare(**options)
        else:
            with pytest.raises(ValueError):
                DescribeSelection.prepare(**options)
        return
    if transport == "daemon":
        schema = DaemonRequest.model_json_schema()
        request = {"method": "describe", "tool_name": "demo/query", **options}
    else:
        schema = next(
            tool.input_schema for tool in _META_TOOLS if tool.name == "describe"
        )
        request = {"name": "demo/query", **options}
    validator = jsonschema.Draft202012Validator(schema)
    assert validator.is_valid(request) is valid
    assert not validator.is_valid({"fields": {}})


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["cli", "daemon", "flat", "meta"])
async def test_jsonpath_selection_and_unknown_hints_across_entries(
    nested_meta: ToolMeta,
    transport: str,
    mcp_call: Callable,
) -> None:
    registry = make_registry(nested_meta)
    fields = ["$.history[*].workTime.duration", "$.missing.leaf"]
    options = {"fields": fields}
    expected = registry.describe("read", **options).model_dump(exclude_none=True)
    assert expected["output_fields"] == view_text(
        {
            "history": composition(
                "anyOf", [{"workTime": {"duration": "float；作业时长，小时"}}]
            )
        }
    )
    assert expected["unknown_fields"] == ["$.missing.leaf"]
    assert "$.summaryStatistic" in expected["available_fields"]["$"]
    assert (
        await describe_entry(registry, transport, mcp_call, name="read", **options)
        == expected
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["registry", "cli", "daemon", "flat", "meta"])
@pytest.mark.parametrize("with_schema", [False, True])
async def test_batch_reuses_rendering_only_within_the_request(
    nested_meta: ToolMeta, transport: str, mcp_call: Callable, with_schema: bool
) -> None:
    registry = Registry()
    if not with_schema:
        # flat 模式至少有一个工具登记输出文档时，才提供 axi_describe。
        registry.register(nested_meta.model_copy(update={"name": "documented"}))
        nested_meta = nested_meta.model_copy(update={"output_schema": None})
    registry.register(nested_meta)
    options = {
        "fields": ["$.summaryStatistic.workTime.duration", "$.missing"],
    }
    expected = registry.describe("read", **options).model_dump(exclude_none=True)
    with patch.object(
        ToolDescriber,
        "_describe",
        autospec=True,
        side_effect=ToolDescriber._describe,
    ) as render:
        for request in (1, 2):
            results = await describe_entry(
                registry, transport, mcp_call, name="read,read,read", **options
            )
            assert results == [expected] * 3
            assert render.call_count == request


@pytest.mark.parametrize("with_schema", [False, True])
@pytest.mark.parametrize("fields", [None, ["$"]])
def test_batch_responses_are_independent_and_aliases_agree(
    nested_meta: ToolMeta, with_schema: bool, fields: Any
) -> None:
    registry = Registry()
    if not with_schema:
        nested_meta.output_schema = None
    nested_meta.input_schema = obj({"limit": node("integer")})
    nested_meta.output_example = {"ready": True}
    registry.register(nested_meta.model_copy(update={"server": "demo"}))
    unique = registry.describe_many(["read"], fields=fields)[0]
    first, second = registry.describe_many(["read", "demo/read"], fields=fields)
    assert unique == first == second
    before = second.model_dump()
    if fields is None:
        first.input_schema["properties"]["limit"]["type"] = "string"
    else:
        first.output_example["ready"] = False
    first.output_fields = "changed"
    assert second.model_dump() == before
    assert registry.describe("read", fields=fields).model_dump() == before


@pytest.mark.parametrize("with_schema", [False, True])
def test_reregistration_refreshes_the_next_batch(with_schema: bool) -> None:
    registry = Registry()
    meta = make_tool(
        obj({"value": node("number", "旧说明")}) if with_schema else None,
        description="旧工具说明",
    )
    registry.register(meta)
    for note in ("旧", "新"):
        meta.description = f"{note}工具说明"
        if with_schema:
            meta.output_schema["properties"]["value"]["description"] = f"{note}说明"
        registry.register(meta)
        results = registry.describe_many(["read", "read"], fields=["$"])
        assert [item.description for item in results] == [f"{note}工具说明"] * 2
        assert [item.output_fields for item in results] == (
            [f"{{value: float；{note}说明}}"] * 2 if with_schema else [None, None]
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("with_schema", [False, True])
async def test_examples_keep_original_serialization_at_each_entry(
    python_result: tuple[Any, type[BaseModel], Any],
    monkeypatch: pytest.MonkeyPatch,
    mcp_call: Callable,
    with_schema: bool,
) -> None:
    value, schema, expected = python_result

    def forbidden() -> None:
        raise AssertionError("Describe executed the business function")

    meta = register_tool(
        forbidden,
        name="example_result",
        output_schema=schema if with_schema else None,
        output_example=value,
    )
    registry = make_registry(meta)
    assert meta.output_example is value
    assert (
        registry.describe(meta.name, fields=["$"]).model_dump(exclude_unset=True)[
            "output_example"
        ]
        == expected
    )
    assert json.loads(
        _tool_description(meta).split("\n\nOutput example: ")[1]
    ) == json.loads(json.dumps(value, ensure_ascii=False, default=str))
    if not with_schema:
        # 至少一个导出工具有输出文档时，flat 模式才提供 describe 辅助工具。
        registry.register(make_tool(obj({}), "documented"))
    for transport in ("cli", "daemon", "flat", "meta"):
        view = await describe_entry(
            registry, transport, mcp_call, name=meta.name, fields=["$"]
        )
        assert view["output_example"] == expected
    monkeypatch.setattr(cli.runtime, "registry", Registry())
    monkeypatch.setattr(
        cli, "app_config", AxiConfig.model_validate({"mcpServers": {"remote": {}}})
    )
    monkeypatch.setattr(
        "axi.daemon.client.daemon_request", lambda req: DaemonResponse.success(view)
    )
    result = CliRunner().invoke(cli.app, ["describe", meta.name, "--fields", '["$"]'])
    assert (
        result.exit_code == 0
        and json.loads(result.stdout)["output_example"] == expected
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["daemon", "flat", "meta"])
@pytest.mark.parametrize("batch", [False, True])
async def test_batches_do_not_block_the_event_loop(
    nested_meta: ToolMeta,
    monkeypatch: pytest.MonkeyPatch,
    transport: str,
    mcp_call: Callable,
    batch: bool,
) -> None:
    registry = make_registry(nested_meta)
    options = {
        "fields": ["$.summaryStatistic.forwardMileage"] if batch else ["$"],
    }
    names = ["read", "missing", "read"] if batch else ["read"]
    expected = [
        item.model_dump(exclude_none=True)
        for item in registry.describe_many(names, **options)
    ]
    expected = expected if batch else expected[0]
    loop, progressed = asyncio.get_running_loop(), Event()
    render = registry.describe_many

    def blocked(*args: Any, **kwargs: Any) -> Any:
        loop.call_soon_threadsafe(progressed.set)
        assert progressed.wait(2), "Describe blocked the event loop"
        return render(*args, **kwargs)

    monkeypatch.setattr(registry, "describe_many", blocked)
    assert (
        await describe_entry(
            registry, transport, mcp_call, name=",".join(names), **options
        )
        == expected
    )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"fields": [7]}, "fields must be a JSONPath array or null."),
        ({"fields": "$"}, "fields must be a JSONPath array or null."),
    ],
)
def test_invalid_batch_retains_lookup_errors(
    options: dict[str, Any], message: str
) -> None:
    registry = Registry()
    for server in ("demo", "other"):
        registry.register(make_tool(name="query", server=server))
    names = ["missing", "query", "demo/query", "demo/query"]
    results = registry.describe_many(names, **options)
    assert len(results) == len(names)
    assert "Tool not found" in results[0].error and "Ambiguous" in results[1].error
    assert results[2].error == results[3].error == message
    assert registry.describe_many([], **options) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["flat", "meta"])
@pytest.mark.parametrize(
    ("target", "options", "message"),
    [
        ("missing", {}, "Tool not found"),
        ("query", {}, "Ambiguous"),
        ("demo/bad", {"fields": ["$"]}, "Check its registered output_schema"),
        ("demo/query", {"fields": [7]}, "fields must be a JSONPath array or null."),
        ("demo/query", {"schema": True, "fields": []}, "cannot be combined"),
        ("demo/query", {"schema": True, "fields": ["$"]}, "cannot be combined"),
    ],
)
async def test_mcp_describe_returns_errors_in_the_result_document(
    transport: str,
    target: str,
    options: dict[str, Any],
    message: str,
    mcp_call: Callable,
) -> None:
    registry = Registry()
    for server in ("demo", "other"):
        registry.register(make_tool(obj({}), "query", server=server))
    registry.register(
        make_tool({"properties": {"speed": "number"}}, "bad", server="demo")
    )
    flat = transport == "flat"
    server = (_build_flat_server if flat else _build_meta_server)(registry, None)
    result = await mcp_call(
        server,
        "axi_describe" if flat else "describe",
        {"name": target.replace("/", "__") if flat else target, **options},
    )
    assert not result.is_error
    payload = json.loads(result.content[0].text)
    assert set(payload) == {"error"}
    assert message in payload["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["registry", "cli", "daemon", "flat", "meta"])
@pytest.mark.parametrize(
    ("schema", "message"),
    [
        ({"properties": {"speed": "number"}}, "Check its registered output_schema"),
        (
            {"properties": {"speed": {"anyOf": None}}},
            "Check its registered output_schema",
        ),
    ],
)
async def test_output_rendering_error_only_fails_its_batch_item(
    nested_meta: ToolMeta,
    transport: str,
    mcp_call: Callable,
    schema: dict,
    message: str,
) -> None:
    registry = make_registry(nested_meta, make_tool(schema, "bad"))
    names = "read,bad,read"
    items = await describe_entry(
        registry, transport, mcp_call, name=names, fields=["$"]
    )
    assert items[0] == items[2]
    assert (
        items[0]["output_fields"]
        == registry.describe("read", fields=["$"]).output_fields
    )
    assert set(items[1]) == {"error"} and message in items[1]["error"]
    assert registry.describe("bad").input_schema == {}


@pytest.mark.parametrize("unavailable", [False, True])
@pytest.mark.parametrize("names", ["remote", "remote,read,missing,remote"])
@pytest.mark.parametrize(
    "options",
    [
        {"fields": None},
        {"fields": []},
        {"fields": ["$"]},
        {"fields": ["$.history[*].forwardMileage"]},
        {"schema": True},
    ],
)
def test_cli_remote_describe_preserves_data_order_and_options(
    nested_meta: ToolMeta,
    monkeypatch: pytest.MonkeyPatch,
    unavailable: bool,
    names: str,
    options: dict[str, Any],
) -> None:
    """逐工具传递查看参数，直接使用远程响应数据，不增加响应校验。"""
    registry = make_registry(nested_meta)
    local = registry.describe("read", **options).model_dump(exclude_none=True)
    remote = {
        "name": "remote",
        "extension": "preserved",
    }
    if options.get("schema"):
        remote.update(
            output_schema_status="registered",
            output_schema=obj({"value": node("number")}),
        )
    else:
        remote["output_fields"] = "{value: float；读数}"
    failure = {"error": "Tool not found: missing"}

    def respond(request: DaemonRequest) -> DaemonResponse:
        if unavailable:
            return DaemonResponse.fail("daemon unavailable")
        return (
            DaemonResponse.fail(failure["error"])
            if request.tool_name == "missing"
            else DaemonResponse.success(remote)
        )

    request = Mock(side_effect=respond)
    monkeypatch.setattr(cli.runtime, "registry", registry)
    monkeypatch.setattr(
        cli, "app_config", AxiConfig.model_validate({"mcpServers": {"remote": {}}})
    )
    monkeypatch.setattr("axi.daemon.client.daemon_request", request)
    requested = names.split(",")
    args = ["describe", names]
    if "fields" in options:
        args += ["--fields", json.dumps(options["fields"])]
    if options.get("schema"):
        args += ["--schema"]
    result = CliRunner().invoke(cli.app, args)
    assert result.exit_code == (1 if unavailable and len(requested) == 1 else 0), (
        result.output
    )
    assert [
        (
            call.args[0].tool_name,
            call.args[0].fields,
            call.args[0].full_schema,
        )
        for call in request.call_args_list
    ] == [
        (name, options.get("fields"), options.get("schema", False))
        for name in requested
        if name != "read"
    ]
    responses = {"remote": remote, "read": local, "missing": failure}
    if unavailable:
        responses.update(
            {name: {"error": "daemon unavailable"} for name in ("remote", "missing")}
        )
    expected = [responses[name] for name in requested]
    assert json.loads(result.stdout) == (
        expected[0] if len(expected) == 1 else expected
    )


@pytest.mark.parametrize("names", ["missing", "read,missing,read"])
def test_cli_missing_local_tool_without_mcp_does_not_contact_daemon(
    nested_meta: ToolMeta, monkeypatch: pytest.MonkeyPatch, names: str
) -> None:
    registry = make_registry(nested_meta)
    monkeypatch.setattr(cli.runtime, "registry", registry)
    monkeypatch.setattr(cli, "app_config", AxiConfig())
    request = Mock(return_value=DaemonResponse.fail("Unexpected daemon request"))
    monkeypatch.setattr("axi.daemon.client.daemon_request", request)
    result = CliRunner().invoke(cli.app, ["describe", names])
    request.assert_not_called()
    missing = {
        "error": "Tool not found: missing. Use search to discover available tools.",
    }
    if names == "missing":
        assert result.exit_code == 1
        assert json.loads(result.stdout) == missing
    else:
        assert result.exit_code == 0
        local = registry.describe("read").model_dump(exclude_none=True)
        assert json.loads(result.stdout) == [local, missing, local]


@pytest.mark.parametrize(
    "mode",
    ["ready", "empty_schema", "unavailable"],
)
def test_cli_input_schema_before_run(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    first = DaemonResponse.success(
        {
            "name": "query",
            "server": "remote",
            "description": "Read a message",
            "source": "mcp",
            "input_schema": {"properties": {"message": {"type": ["string", "null"]}}},
        }
    )
    if mode == "empty_schema":
        first.data["input_schema"] = None
    elif mode == "unavailable":
        first = DaemonResponse.fail("Daemon request timed out")
    value = "null" if mode == "ready" else None
    request = Mock(side_effect=[first, DaemonResponse.success({"message": value})])
    monkeypatch.setattr(cli.runtime, "registry", Registry())
    monkeypatch.setattr(cli, "daemon_request", request)
    monkeypatch.setattr("axi.executor.daemon_request", request)
    result = CliRunner().invoke(cli.app, ["run", "remote/query", "--message", "null"])
    inputs = request.call_args_list[0].args[0]
    assert inputs.method == "describe" and inputs.fields is None
    assert result.exit_code == 0 and request.call_count == 2
    assert json.loads(result.stdout) == {
        "status": "success",
        "data": {"message": value},
    }
    assert request.call_args.args[0].params == {"message": value}


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["registry", "cli", "daemon", "flat", "meta"])
@pytest.mark.parametrize(
    "fields",
    [
        None,
        [],
        ["$"],
        ["$.summaryStatistic"],
        ["$.summaryStatistic.forwardMileage"],
        ["$.missing.leaf"],
    ],
)
async def test_fields_select_inputs_or_output_documentation(
    nested_meta: ToolMeta, transport: str, fields: Any, mcp_call: Callable
) -> None:
    nested_meta.input_schema = obj(
        {"start": node("string", "起始日期")}, required=["start"]
    )
    nested_meta.output_example = {"example": 7}
    registry = make_registry(nested_meta)
    result = await describe_entry(
        registry, transport, mcp_call, name="read", fields=fields
    )
    assert result == registry.describe("read", fields=fields).model_dump(
        exclude_none=True
    )
    assert result["description"] == nested_meta.description
    if fields is None:
        assert result["input_schema"] == nested_meta.input_schema
        assert not any(key.startswith("output_") for key in result)
        return
    assert "input_schema" not in result
    assert result["output_schema_status"] == "registered"
    assert result["output_example"] == {"example": 7}
    assert isinstance(result["output_fields"], str)
    assert result["output_type"] == "dict"
    if fields == []:
        assert result["output_fields"] == "{}"
    elif fields == ["$"]:
        assert (
            result["output_fields"]
            == "{plateNumber: str；车牌/车辆编号, summaryStatistic: dict；查询时间窗内的车辆汇总统计, history: anyOf(list[dict], None)；历史统计}"
        )
    elif fields == ["$.summaryStatistic"]:
        assert (
            result["output_fields"]
            == "{summaryStatistic: {forwardMileage: float；前进里程，单位米, workTime: dict；作业时间明细, unexplained: dict}}"
        )
    elif fields == ["$.summaryStatistic.forwardMileage"]:
        assert (
            result["output_fields"]
            == "{summaryStatistic: {forwardMileage: float；前进里程，单位米}}"
        )
    else:
        assert result["unknown_fields"] == fields
        assert result["available_fields"] == {
            "$": ["$.plateNumber", "$.summaryStatistic", "$.history"]
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["registry", "cli", "daemon", "flat", "meta"])
async def test_missing_schema_is_distinct_from_unknown_paths(
    transport: str, mcp_call: Callable
) -> None:
    registry = make_registry(
        make_tool(name="legacy"), make_tool(obj({"id": node("integer")}))
    )
    result = await describe_entry(
        registry, transport, mcp_call, name="legacy,read", fields=["$.missing.leaf"]
    )
    assert result[0]["output_schema_status"] == "not_registered"
    assert "unknown_fields" not in result[0] and "output_fields" not in result[0]
    assert result[1]["output_schema_status"] == "registered"
    assert result[1]["unknown_fields"] == ["$.missing.leaf"]
    assert result[1]["available_fields"] == {"$": ["$.id"]}


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["registry", "cli", "daemon", "flat", "meta"])
@pytest.mark.parametrize(
    "root_combination", [False, True], ids=["containers", "root-branches"]
)
async def test_output_document_contract_across_entries(
    transport: str, root_combination: bool, mcp_call: Callable
) -> None:
    """核对完整表达式在各入口间正确传递；选择规则矩阵由 tests/output 覆盖。"""
    schema = obj(
        {
            "reading": {
                "anyOf": [
                    obj({"temperature": node("number", "温度")}),
                    arr(obj({"status": node("string", "状态")})),
                    node("null", "暂无读数"),
                ]
            },
            "position": node(
                "array",
                prefixItems=[node("number", "纬度"), node("number", "经度")],
                items=False,
            ),
            "samples": arr(node("number", "采样值")),
        }
    )
    expected = (
        "{reading: anyOf({temperature: float；温度}, [{status: str；状态}], None；暂无读数), "
        "position: [[0]: float；纬度, [1]: float；经度], samples: [float；采样值]}"
    )
    if root_combination:
        schema = {"oneOf": [schema, obj({"fallback": node("string", "备用")})]}
        expected = f"oneOf({expected}, {{fallback: str；备用}})"
    inputs = obj({"id": node("string")})
    registry = make_registry(make_tool(schema, input_schema=inputs))
    item = await describe_entry(
        registry,
        transport,
        mcp_call,
        name="read",
        fields=["$", "$.reading", "$.position", "$.samples"],
    )
    assert item["output_fields"] == expected
    assert item["output_schema_status"] == "registered"
    assert "unknown_fields" not in item and "input_schema" not in item
