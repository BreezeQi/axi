"""验证引用保留路径、分支关系、说明及查看深度边界。"""

import json
import re
from copy import deepcopy
from typing import Any

import pytest
from pydantic import BaseModel, Field

from axi.output.explanation import SchemaFormatter
from axi.output.registration import normalize_schema
from axi.search.corpus import build_corpus
from tests.output.reference_cases import alias_definitions
from tests.support import (
    arr,
    composition,
    describe_tool,
    make_tool,
    node,
    obj,
    prepare_field_documents,
    ref,
    selected_paths,
    show,
    view_text,
    make_registry,
)


@pytest.mark.parametrize(
    ("local", "path", "explanation"),
    [
        (obj({"value": node("number", "读数")}), "$.value", "float；读数"),
        (arr(node("number", "读数")), "$[*]", "float；读数"),
        (
            {"anyOf": [obj({"value": node("number", "读数")}), arr(node("string"))]},
            "$.value",
            "float；读数",
        ),
    ],
)
def test_recursive_reference_does_not_hide_its_local_connections(
    local: dict[str, Any], path: str, explanation: str
) -> None:
    schema = {**ref("Node"), "$defs": {"Node": {**ref("Node"), **local}}}
    before = deepcopy(schema)
    registry = make_registry(make_tool(schema))
    assert registry.grep(re.escape(path) + ":")[0].matched_fields == [
        f"{path}: {explanation}"
    ]
    selected = registry.describe("read", fields=[path])
    assert selected.unknown_fields is None
    assert selected_paths(schema, [path]) == {(path, explanation)}
    assert (path, explanation) in selected_paths(schema)
    missing = registry.describe("read", fields=[path, "$.missing"])
    assert missing.unknown_fields == ["$.missing"]
    assert path in missing.available_fields["$"]
    assert schema == before


@pytest.mark.parametrize("reverse", [False, True])
def test_shared_subtrees_expand_from_each_selected_occurrence(reverse: bool) -> None:
    branches = [
        obj({"direct": ref("Shared")}),
        obj({"nested": obj({"direct": ref("Shared")})}),
    ]
    schema = {
        "anyOf": branches[::-1] if reverse else branches,
        "$defs": {
            "Shared": obj({"child": ref("Deep")}),
            "Deep": obj({"value": node("number")}),
        },
    }
    expected = [
        {"direct": {"child": {"value": "float"}}},
        {"nested": {"direct": {"child": "dict"}}},
    ]
    assert show(schema, fields=["$.direct.child", "$.nested.direct"]) == composition(
        "anyOf", *(expected[::-1] if reverse else expected)
    )


def test_recursive_terminal_description_survives_explicit_selections() -> None:
    class Node(BaseModel):
        """链表节点，记录下一环节信息"""

        value: int = Field(description="当前值")
        next: "Node | None"

    for nullable, described in [(True, True), (False, True), (True, False)]:
        schema = normalize_schema(Node)
        definition = schema["$defs"]["Node"]
        if not nullable:
            definition["properties"]["next"] = ref("Node")
        if not described:
            definition.pop("description")
        before = deepcopy(schema)
        meta = make_tool(schema)
        label = "dict" + ("；链表节点，记录下一环节信息" if described else "")
        if nullable:
            label = f"anyOf({label}, None)"
        view = describe_tool(meta)
        assert view.output_fields == view_text({"value": "int；当前值", "next": label})
        selected = describe_tool(meta, fields=["$.next.next.value"])
        assert selected_paths(schema, ["$.next.next.value"]) == {
            ("$.next.next.value", "int；当前值")
        }
        assert selected.unknown_fields is None
        assert schema == before


@pytest.mark.parametrize("closed", [False, True])
@pytest.mark.parametrize("array", [False, True])
def test_mutual_union_terminals_agree_with_search(closed: bool, array: bool) -> None:
    definitions = {
        side: obj(
            {
                "value": node("string", side + "值"),
                "next": node(
                    description="下一节点", anyOf=[ref("A"), ref("B"), node("null")]
                ),
            },
            **({"additionalProperties": False} if closed else {}),
        )
        for side in ("A", "B")
    }
    root = {"anyOf": [ref("A"), ref("B")]}
    schema = {**(arr(root) if array else root), "$defs": definitions}
    meta = make_tool(schema)
    own_terminal = "anyOf(dict, None)；下一节点"
    prefix = "$[*]" if array else "$"
    view = describe_tool(meta, fields=[prefix, prefix + ".next"]).output_fields
    assert selected_paths(schema, fields=[prefix, prefix + ".next"]) >= {
        (prefix + ".value", "str；A值"),
        (prefix + ".value", "str；B值"),
        (prefix + ".next.value", "str；A值"),
        (prefix + ".next.value", "str；B值"),
        (prefix + ".next.next", own_terminal),
    }
    assert view.startswith("[anyOf(" if array else "anyOf(")
    selected = describe_tool(
        meta, fields=[prefix + ".next.value", prefix + ".next.next"]
    )
    assert selected.unknown_fields is None
    assert selected_paths(schema, [prefix + ".next.value", prefix + ".next.next"]) >= {
        (prefix + ".next.value", "str；A值"),
        (prefix + ".next.value", "str；B值"),
    }
    corpus = build_corpus([meta], [prepare_field_documents(meta.output_schema)])
    assert prefix + ".next.next: " + own_terminal in corpus.output_texts
    registry = make_registry(meta)
    path = prefix + ".next.value"
    expected_hits = [path + ": str；A值", path + ": str；B值"]
    assert set(registry.search(path)[0].matched_fields) == set(expected_hits)
    assert registry.grep(re.escape(path) + ":")[0].matched_fields == expected_hits[::-1]


def test_nested_recursive_aliases_preserve_combined_contexts() -> None:
    def record(kind: str, description: str, target: str) -> dict:
        return obj(
            {
                "value": node(kind, description),
                "next": {"anyOf": [ref(target), node("null")]},
            }
        )

    schema = obj(
        {"head": ref("A")},
        **{
            "$defs": {
                "A": record("string", "A值", "B"),
                "B": {"anyOf": [record("number", "B值", "C"), ref("A")]},
                "C": record("string", "C值", "C"),
            }
        },
    )
    assert selected_paths(schema, ["$.head.next.next.value"]) == {
        ("$.head.next.next.value", label)
        for label in ("str；C值", "float；B值", "str；A值")
    }
    corpus = build_corpus([make_tool(schema)], [prepare_field_documents(schema)])
    assert "$.head.next.next.value: str；C值" in corpus.output_texts


def test_shared_references_in_other_branches_still_expand() -> None:
    schema = obj(
        {"record": {"anyOf": [ref("A"), ref("B")]}},
        **{
            "$defs": {
                "A": obj({"child": ref("B")}, additionalProperties=False),
                "B": obj({"child": ref("Value")}, additionalProperties=False),
                "Value": obj({"title": node("string")}, additionalProperties=False),
            }
        },
    )
    assert show(
        schema, fields=["$.record", "$.record.child", "$.record.child.child"]
    ) == view_text(
        {
            "record": composition(
                "anyOf",
                {"child": {"child": {"title": "str"}}},
                {"child": {"title": "str"}},
            )
        }
    )
    assert len(prepare_field_documents(schema).fields) == 5


@pytest.mark.parametrize("reverse", [False, True])
def test_shared_shapes_follow_long_aliases_in_either_branch_order(
    reverse: bool,
) -> None:
    definitions = alias_definitions(70, {"anyOf": [ref("Leaf")]})
    definitions["Leaf"] = obj({"value": node("number", "数值")})
    branches = [ref("Alias69"), ref("Value")]
    schema = {"$defs": definitions, "anyOf": branches[::-1] if reverse else branches}
    meta = make_tool(schema)
    for fields in (None, ["$.value"]):
        view = describe_tool(meta, fields=fields)
        assert ("$.value", "float；数值") in selected_paths(schema, fields)
        assert view.unknown_fields is None


def test_nested_oneof_structure_keeps_branch_grouping() -> None:
    number = obj({"value": node("number", "读数")})
    schema = {
        "oneOf": [{"oneOf": [number, number]}, obj({"value": node("string", "状态")})]
    }
    expected = composition(
        "oneOf",
        composition("oneOf", {"value": "float；读数"}, {"value": "float；读数"}),
        {"value": "str；状态"},
    )
    for definition in (schema, json.loads(json.dumps(schema))):
        assert show(definition) == view_text(expected)


@pytest.mark.parametrize("source", ["inline", "reference", "composition"])
@pytest.mark.parametrize("array", [False, True])
@pytest.mark.parametrize(
    "grouping,compact",
    [
        ("single", "oneOf(dict)"),
        ("duplicate", "oneOf(dict, dict)"),
        ("nested", "oneOf(oneOf(dict, dict))"),
    ],
)
def test_oneof_grouping_survives_views_indexing_and_json_round_trips(
    source: str, array: bool, grouping: str, compact: str
) -> None:
    record = obj({"value": node("number", "温度读数")})
    branch = record if source == "inline" else ref("Record")
    members = [branch] if grouping == "single" else [branch, branch]
    group = {"oneOf": [{"oneOf": members}] if grouping == "nested" else members}
    schema = obj(
        {"reading": arr(group) if array else group},
        **{
            "$defs": {
                "Record": {"allOf": [record]} if source == "composition" else record
            }
        },
    )
    branch_view = {"value": "float；温度读数"}
    group_view = composition("oneOf", *([branch_view] * len(members)))
    if grouping == "nested":
        group_view = composition("oneOf", group_view)
    expected = {"reading": [group_view] if array else group_view}
    path = "$.reading" + ("[*]" if array else "") + ".value"
    for definition in (schema, json.loads(json.dumps(schema))):
        before = deepcopy(definition)
        registry = make_registry(make_tool(definition))
        for fields in (["$.reading"], [path]):
            view = registry.describe(
                "read", fields=fields if fields is not None else ["$"]
            )
            assert view.output_fields == view_text(expected)
        assert [field.text for field in prepare_field_documents(definition).fields] == [
            "$.reading: " + (f"list[{compact}]" if array else compact),
            *([f"$.reading[*]: {compact}"] if array else []),
            f"{path}: float；温度读数",
        ]
        for discover in (registry.search, registry.grep):
            assert discover("温度读数")[0].matched_fields == [
                f"{path}: float；温度读数"
            ]
        assert definition == before


def test_same_name_recursive_fields_keep_both_branches() -> None:
    definitions = {
        side: obj(
            {
                "value" + side: node(kind),
                "next": {"anyOf": [ref(side), node("null")]},
            },
            side + "节点",
            additionalProperties=False,
        )
        for side, kind in [("A", "integer"), ("B", "string")]
    }
    schema = obj({"record": {"anyOf": [ref("A"), ref("B")]}}, **{"$defs": definitions})
    meta = make_tool(schema)
    expected = {
        "record": composition(
            "anyOf",
            {"valueA": "int", "next": "anyOf(dict；A节点, None)"},
            {"valueB": "str", "next": "anyOf(dict；B节点, None)"},
        )
    }
    assert describe_tool(meta, fields=["$.record"]).output_fields == view_text(expected)
    selected = describe_tool(
        meta, fields=["$.record.next.valueA", "$.record.next.valueB"]
    )
    assert selected_paths(schema, ["$.record.next.valueA", "$.record.next.valueB"]) == {
        ("$.record.next.valueA", "int"),
        ("$.record.next.valueB", "str"),
    }
    assert selected.unknown_fields is None
    assert len(prepare_field_documents(schema).fields) == 5


def test_recursive_object_array_union_keeps_nested_array_layers() -> None:
    schema = {
        "anyOf": [ref("Rows"), obj({"value": ref("Value")})],
        "$defs": {
            "Rows": arr(obj({"value": ref("Value")})),
            "Value": obj({"rows": ref("Rows")}),
        },
    }
    meta = make_tool(schema)
    view = describe_tool(meta, fields=["$", "$[*].value", "$.value.rows"])
    assert view.output_fields.startswith("anyOf(")
    assert selected_paths(schema, fields=["$", "$[*].value", "$.value.rows"]) == {
        ("$[*].value.rows", "list[dict]"),
        ("$.value.rows[*].value", "dict"),
    }
    selected = describe_tool(meta, fields=["$.value.rows[*].value"])
    assert selected_paths(schema, ["$.value.rows[*].value"]) == {
        ("$.value.rows[*].value.rows", "list[dict]")
    }
    assert selected.unknown_fields is None


def test_contradictory_container_declarations_remain_viewable() -> None:
    schema = {
        "$defs": {
            "Batch": arr(obj({"next": {"anyOf": [ref("Node"), node("null")]}})),
            "Node": {"allOf": [ref("Batch"), obj({"payload": node("number")})]},
        },
        **ref("Batch"),
    }
    meta = make_tool(schema)
    assert describe_tool(meta).output_fields == view_text(
        [{"next": "anyOf(allOf(list[dict], dict), None)"}]
    )
    assert ("$[*].next.payload", "float") in selected_paths(
        schema, fields=["$[*].next"]
    )
    view = describe_tool(meta, fields=["$[*].next.payload"])
    assert selected_paths(schema, ["$[*].next.payload"]) == {
        ("$[*].next.payload", "float")
    }
    assert view.unknown_fields is None
    assert "$[*].next.payload" in [
        f.path for f in prepare_field_documents(schema).fields
    ]


@pytest.mark.parametrize("kinds", [["array", "null"], ["array"]])
def test_recursive_array_types_terminate(kinds: list[str]) -> None:
    schema = obj(
        {"payload": ref("Node")}, **{"$defs": {"Node": node(kinds, items=ref("Node"))}}
    )
    meta = make_tool(schema)
    terminal = SchemaFormatter(schema).format(schema["properties"]["payload"])
    assert describe_tool(meta).output_fields == view_text({"payload": terminal})
    assert "list" in terminal and ("None" in terminal) == ("null" in kinds)
    assert len(describe_tool(meta).model_dump_json()) < 2000
    registry = make_registry(meta)
    assert registry.search("payload")[0].matched_fields == ["$.payload: " + terminal]


def test_cached_alias_targets_keep_each_uses_structural_siblings() -> None:
    schema = obj(
        {
            "first": {**ref("Record"), "properties": {"left": node("string")}},
            "second": {**ref("Record"), "properties": {"right": node("integer")}},
        },
        **{"$defs": {"Record": obj({"value": node("number", "温度")})}},
    )
    assert show(schema, fields=["$.first", "$.second"]) == view_text(
        {
            "first": composition("allOf", {"value": "float；温度"}, {"left": "str"}),
            "second": composition("allOf", {"value": "float；温度"}, {"right": "int"}),
        }
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_long_aliases_are_independent_of_cache_warming_order(reverse: bool) -> None:
    definitions = alias_definitions(
        70, obj({"value": node("number", "温度"), "next": ref("Alias69")}, "节点")
    )
    properties = {
        "deep": ref("Alias69"),
        "shallow": ref("Alias1"),
    }
    if reverse:
        properties = dict(reversed(properties.items()))
    schema = obj(properties, **{"$defs": definitions})
    expected = dict.fromkeys(properties, {"value": "float；温度", "next": "dict；节点"})
    assert show(schema, fields=["$." + name for name in properties]) == view_text(
        {name: expected[name] for name in properties}
    )
    assert selected_paths(schema, ["$.deep.next.value"]) == {
        ("$.deep.next.value", "float；温度")
    }


@pytest.mark.parametrize("declaration", ["alias", "anyOf", "oneOf", "allOf"])
@pytest.mark.parametrize("record", [False, True])
def test_schema_wrappers_do_not_consume_data_depth(
    declaration: str, record: bool
) -> None:
    leaf = node("number", "温度，单位摄氏度")
    definition = obj({"value": leaf}, "测量信息") if record else leaf
    folded = "dict；测量信息" if record else "float；温度，单位摄氏度"
    expanded = "{value: float；温度，单位摄氏度}" if record else folded
    definitions = {}
    if declaration == "alias":
        definitions = alias_definitions(70, definition)
        definition = ref("Alias69")
    else:
        for _ in range(70):
            definition = {declaration: [definition]}
        if declaration == "oneOf":
            folded = "oneOf(" * 70 + folded + ")" * 70
            expanded = "oneOf(" * 70 + expanded + ")" * 70
    schema = obj(
        {"reading": {**definition, "description": "当前读数"}},
        **{"$defs": definitions},
    )
    registry = make_registry(make_tool(schema))
    folded += "；当前读数"
    assert registry.describe("read", fields=["$"]).output_fields == view_text(
        {"reading": folded}
    )
    assert registry.describe("read", fields=["$.reading"]).output_fields == view_text(
        {"reading": expanded if record else folded}
    )
    path = "$.reading.value" if record else "$.reading"
    label = "float；温度，单位摄氏度" if record else folded
    selected = registry.describe("read", fields=[path])
    assert selected.unknown_fields is None
    assert label in selected.output_fields
    for discover in (registry.search, registry.grep):
        assert discover("摄氏度")[0].matched_fields == [f"{path}: {label}"]
    if record:
        unknown = registry.describe("read", fields=["$.reading.missing"])
        assert unknown.unknown_fields == ["$.reading.missing"]
        assert unknown.available_fields == {"$.reading": ["$.reading.value"]}


def test_recursive_compositions_do_not_reuse_another_branch_stop() -> None:
    definitions = {
        side: {"anyOf": [node(kind, side + "值"), ref(other)]}
        for side, kind, other in [("A", "string", "B"), ("B", "number", "A")]
    }
    for first, second in [("A", "B"), ("B", "A")]:
        schema = {"anyOf": [ref(first), ref(second)], "$defs": definitions}
        labels = {"A": "str；A值", "B": "float；B值"}
        assert SchemaFormatter(schema).format(schema) == (
            f"anyOf({labels[first]}, {labels[second]}, "
            f"$ref(#/$defs/{first}), $ref(#/$defs/{second}))"
        )
        schema = obj(
            {side: ref(side) for side in (first, second)}, **{"$defs": definitions}
        )
        expected = {
            side: f"anyOf({labels[side]}, {labels[other]}, $ref(#/$defs/{side}))"
            for side, other in ((first, second), (second, first))
        }
        assert [field.text for field in prepare_field_documents(schema).fields] == [
            f"$.{side}: {label}" for side, label in expected.items()
        ]
        assert describe_tool(make_tool(schema)).output_fields == view_text(expected)


def test_shared_recursion_keeps_order_and_branch_stops() -> None:
    shared = obj(
        {"next": node(description="下一节点", anyOf=[ref("A"), ref("B"), node("null")])}
    )
    definitions = {"Shared": {"allOf": [shared]}}
    for side, kind in [("A", "integer"), ("B", "string")]:
        definitions[side] = {
            "allOf": [
                ref("Shared"),
                {"properties": {"from" + side: node(kind, side + "来源")}},
            ]
        }
    schema = {"$defs": definitions, "anyOf": [ref("A"), ref("B")]}
    meta = make_tool(schema)
    view = describe_tool(meta, fields=["$", "$.next"]).output_fields
    assert view.startswith("anyOf(")
    assert selected_paths(schema, fields=["$", "$.next"]) >= {
        ("$.fromA", "int；A来源"),
        ("$.fromB", "str；B来源"),
        ("$.next.fromA", "int；A来源"),
        ("$.next.fromB", "str；B来源"),
    }
    assert [f.name for f in prepare_field_documents(schema).fields if f.depth == 2] == [
        "fromB",
        "fromA",
    ]
    selected = describe_tool(meta, fields=["$.next.next.fromA", "$.next.next.fromB"])
    assert selected_paths(schema, ["$.next.next.fromA", "$.next.next.fromB"]) == {
        ("$.next.next.fromA", "int；A来源"),
        ("$.next.next.fromB", "str；B来源"),
    }
    assert selected.unknown_fields is None


def test_reference_pointers_and_adjacent_declarations() -> None:
    schema = {
        "$defs": {
            "Count value": node("integer", "计数"),
            "Choice": {"anyOf": [node("number", "距离"), node("null")]},
        },
        "properties": {
            "count": ref("Count%20value"),
            "distance": ref("Choice/anyOf/0"),
        },
    }
    assert show(schema) == view_text(
        {
            "count": "int；计数",
            "distance": "float；距离",
        }
    )
    modern = {
        "$defs": {"Record": obj({"distance": node("number")})},
        **ref("Record"),
        "properties": {"distance": {"minimum": 0}, "extra": node("string")},
    }
    assert show(modern) == composition(
        "allOf", {"distance": "float"}, {"distance": "Any", "extra": "str"}
    )
