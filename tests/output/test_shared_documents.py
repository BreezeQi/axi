"""验证注册文档复用释义与结构连接，各次遍历状态保持独立。"""

from copy import deepcopy
from typing import Any
from unittest.mock import Mock, patch

import pytest

from axi.output.explanation import SchemaFormatter
from axi.output.documents import SchemaDocument
from axi.output.schema import SchemaNode, SchemaShape, schema_pointer
from tests.support import (
    arr,
    composition,
    make_tool,
    node,
    obj,
    ref,
    make_registry,
)
from tests.support import view_text


def reading_schema() -> dict[str, Any]:
    return obj(
        {
            "reading": {
                "anyOf": [
                    obj({"temperature": node("number", "温度")}),
                    arr(obj({"status": node("string", "状态")})),
                ],
                "description": "本次读数",
            }
        }
    )


@pytest.mark.parametrize("first", ["search", "grep", "describe"])
def test_search_grep_and_describe_reuse_the_same_node_explanation(
    first: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = make_registry(make_tool(reading_schema()))
    reading = registry.resolve("read").output_schema["properties"]["reading"]
    original = SchemaFormatter._build
    builds = []

    def build(
        formatter: SchemaFormatter,
        definition: SchemaNode,
        descriptions: bool,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        if definition is reading and descriptions:
            builds.append(id(formatter))
        return original(formatter, definition, descriptions, *args, **kwargs)

    monkeypatch.setattr(SchemaFormatter, "_build", build)
    queries = {
        "search": lambda: registry.search("reading"),
        "grep": lambda: registry.grep("reading"),
        "describe": lambda: registry.describe("read", fields=["$"]),
    }
    queries[first]()
    assert len(builds) == 1
    for query in queries.values():
        query()
    assert len(builds) == 1
    assert registry.describe("read", fields=["$"]).output_fields == view_text(
        {"reading": "anyOf(dict, list[dict])；本次读数"}
    )
    document = registry._documentation(registry.resolve("read"))
    assert [field.text for field in document.fields.fields] == [
        "$.reading: anyOf(dict, list[dict])；本次读数",
        "$.reading.temperature: float；温度",
        "$.reading[*]: dict",
        "$.reading[*].status: str；状态",
    ]
    assert registry.describe("read", fields=["$.reading"]).output_fields == view_text(
        {
            "reading": composition(
                "anyOf", {"temperature": "float；温度"}, [{"status": "str；状态"}]
            )
        }
    )
    assert len(builds) == 1


@pytest.mark.parametrize("first", ["grep", "describe"])
def test_index_and_view_reuse_ordinary_connections(first: str) -> None:
    registry = make_registry(make_tool(obj({"detail": obj({"value": node("number")})})))
    schema = registry.resolve("read").output_schema
    containers = [schema, schema["properties"]["detail"]]
    queries = {
        "grep": lambda: registry.grep("value"),
        "describe": lambda: registry.describe("read", fields=["$.detail"]),
    }
    with patch.object(SchemaShape, "read", side_effect=SchemaShape.read) as read:
        queries[first]()
        for query in queries.values():
            query()
    for container in containers:
        assert sum(call.args[0] is container for call in read.call_args_list) == 1
    assert queries["grep"]()[0].matched_fields == ["$.detail.value: float"]
    assert queries["describe"]().output_fields == "{detail: {value: float}}"


@pytest.mark.parametrize("first", ["grep", "describe"])
@pytest.mark.parametrize("declaration", ["reference", "types"])
def test_shared_original_connections_are_cached_separately_from_views(
    first: str, declaration: str
) -> None:
    record = obj({"speed": node("number", "车速")})
    shared = (
        ref("Record")
        if declaration == "reference"
        else {**record, "type": ["object", "null"]}
    )
    names = [f"position{i}" for i in range(32)]
    registry = make_registry(
        make_tool(
            obj(
                {name: ref("Shared") for name in names},
                **{"$defs": {"Shared": shared, "Record": record}},
            )
        )
    )
    definition = registry.resolve("read").output_schema["$defs"]["Shared"]
    queries = {
        "grep": lambda: registry.grep("车速"),
        "describe": lambda: registry.describe(
            "read",
            fields=["$." + name for name in names],
        ),
    }
    with patch.object(SchemaShape, "read", wraps=SchemaShape.read) as read:
        queries[first]()
        hits, view = queries["grep"](), queries["describe"]()
        assert sum(call.args[0] is definition for call in read.call_args_list) == 1
    child = "{speed: float；车速}"
    if declaration == "types":
        child = f"anyOf({child}, None)"
    assert view.output_fields == view_text(dict.fromkeys(names, child))
    assert hits[0].matched_fields == [
        f"$.{name}.speed: float；车速" for name in names[:8]
    ]


def test_original_connections_are_shared_but_occurrence_paths_and_notes_are_distinct() -> (
    None
):
    schema = obj(
        {
            "first": {**ref("Reading"), "description": "前部读数"},
            "second": {**ref("Reading"), "description": "后部读数"},
        },
        **{"$defs": {"Reading": obj({"speed": node("number", "车速")}, "设备读数")}},
    )
    before = deepcopy(schema)
    registry = make_registry(make_tool(schema))
    lookup = Mock(wraps=schema_pointer)
    with patch("axi.output.explanation.schema_pointer", lookup):
        assert registry.grep("车速")[0].matched_fields == [
            "$.first.speed: float；车速",
            "$.second.speed: float；车速",
        ]
        assert lookup.call_count == 1
        collapsed = registry.describe("read", fields=["$"]).output_fields
        assert collapsed == view_text(
            {
                "first": "dict；设备读数；前部读数",
                "second": "dict；设备读数；后部读数",
            }
        )
        assert registry.describe(
            "read", fields=["$.first", "$.second"]
        ).output_fields == view_text(
            {key: {"speed": "float；车速"} for key in ("first", "second")}
        )
        assert lookup.call_count == 1
    assert schema == before


def test_selected_view_does_not_prepare_unselected_fields_or_process_inputs_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Unselected(dict):
        def get(self, key: str, default: Any = None) -> Any:
            raise AssertionError("Read an unselected field definition")

    registry = make_registry(
        make_tool(
            obj({"value": node("number", "数值"), "unused": Unselected(node("string"))})
        )
    )
    with monkeypatch.context() as inputs:
        inputs.setattr(
            registry,
            "_documentation",
            Mock(side_effect=AssertionError("Inputs prepared output docs")),
        )
        assert registry.describe("read").input_schema == {}
    assert registry.describe("read", fields=["$.value"]).output_fields == view_text(
        {"value": "float；数值"}
    )
    document = registry._documentation(registry.resolve("read"))
    assert document._fields is None


def test_prepared_definitions_survive_unrelated_registration_and_reset_on_reregistration() -> (
    None
):
    registry = make_registry(make_tool(obj({"speed": node("number", "旧车速")})))
    registry.grep("旧车速")
    old = registry._documentation(registry.resolve("read"))
    entries = old.fields
    registry.register(make_tool(obj({"other": node("string")}), "other"))
    assert registry._documentation(registry.resolve("read")) is old
    assert old.fields is entries
    replacement = registry.resolve("read").model_copy(deep=True)
    replacement.output_schema["properties"]["speed"]["description"] = "新车速"
    registry.register(replacement)
    current = registry._documentation(registry.resolve("read"))
    assert current is not old and current._fields is None
    assert registry.describe("read", fields=["$"]).output_fields == view_text(
        {"speed": "float；新车速"}
    )
    assert registry.grep("新车速")[0].matched_fields == ["$.speed: float；新车速"]
    assert [field.text for field in entries.fields] == ["$.speed: float；旧车速"]


def test_view_selection_can_change_after_recursive_automatic_stops() -> None:
    schema = {
        **ref("Node"),
        "$defs": {
            "Node": obj(
                {
                    "value": node("number", "数值"),
                    "next": {"anyOf": [ref("Node"), node("null")]},
                },
                "链表节点",
            )
        },
    }
    registry = make_registry(make_tool(schema))
    for _ in range(2):
        assert registry.describe("read", fields=["$"]).output_fields == view_text(
            {
                "value": "float；数值",
                "next": "anyOf(dict；链表节点, None)",
            }
        )
        result = registry.describe("read", fields=["$.next.next.value"])
        assert result.unknown_fields is None
        assert result.output_fields == view_text(
            {
                "next": composition(
                    "anyOf", {"next": composition("anyOf", {"value": "float；数值"})}
                )
            }
        )


def test_structure_reads_share_original_child_connections() -> None:
    properties = {f"field{i}": node("number") for i in range(1000)}
    branches = [obj({"status": node("string")}), arr(node("integer"))]
    schema = obj(properties, anyOf=branches)
    shape = SchemaDocument(schema).shape(schema)
    assert shape.properties is properties
    assert shape.groups["anyOf"] is branches
    array = node("array", prefixItems=branches, items=False)
    shape = SchemaDocument(array).shape(array)
    assert all(
        shape.items[f"[{index}]"] is branch for index, branch in enumerate(branches)
    )
