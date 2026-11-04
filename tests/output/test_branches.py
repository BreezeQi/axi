"""验证 describe 保留声明的组合分支及其数据位置。"""

from copy import deepcopy
import json
from typing import Any

import pytest
from pydantic import BaseModel, RootModel

from tests.support import (
    arr,
    composition,
    describe_tool,
    make_tool,
    node,
    obj,
    make_registry,
    ref,
    show,
)
from tests.support import view_text


def test_empty_array_alternatives_keep_unknown_hints() -> None:
    class Row(BaseModel):
        values: list[float]

    meta = make_tool(RootModel[Row | list[Row]].model_json_schema())
    fields = ["$.values[0]", "$[*].values[0]"]
    result = describe_tool(meta, fields=fields)
    assert result.output_fields == composition(
        "anyOf", {"values": []}, [{"values": []}]
    )
    assert result.unknown_fields == fields
    assert result.available_fields == {
        "$.values": ["$.values[*]"],
        "$[*].values": ["$[*].values[*]"],
    }


@pytest.mark.parametrize("declaration", ["anyOf", "oneOf", "allOf", "type"])
def test_mixed_branches_keep_their_data_positions(declaration: str) -> None:
    properties = {"temperature": node("number", "温度")}
    items = obj({"status": node("string", "状态")})
    keyword = "anyOf" if declaration == "type" else declaration
    reading = (
        node(["object", "array"], properties=properties, items=items)
        if declaration == "type"
        else {keyword: [obj(properties), arr(items)]}
    )
    reading["description"] = "本次读数"
    schema = obj({"reading": reading})
    before = deepcopy(schema)
    registry = make_registry(make_tool(schema))
    label = (
        "dict | list[dict]" if declaration == "type" else f"{keyword}(dict, list[dict])"
    ) + "；本次读数"
    folded = view_text({"reading": label})
    view = registry.describe("read", fields=["$"])
    assert view.output_fields == folded
    expected = {
        "reading": composition(
            keyword, {"temperature": "float；温度"}, [{"status": "str；状态"}]
        )
    }
    view = registry.describe("read", fields=["$.reading"])
    assert view.output_fields == view_text(expected)
    paths = ["$.reading.temperature", "$.reading[*].status"]
    for fields in (paths, paths[::-1]):
        view = registry.describe("read", fields=fields if fields is not None else ["$"])
        assert view.output_fields == view_text(expected) and view.unknown_fields is None
    selected = registry.describe("read", fields=[paths[1]])
    assert selected.output_fields == view_text(
        {"reading": composition(keyword, [{"status": "str；状态"}])}
    )
    assert selected.unknown_fields is None
    assert registry.search("$.reading")[0].matched_fields == ["$.reading: " + label]
    assert registry.grep("温度")[0].matched_fields == [
        "$.reading.temperature: float；温度"
    ]
    assert schema == before


@pytest.mark.parametrize("array", [False, True])
@pytest.mark.parametrize("scalar,label", [("null", "None"), ("string", "str")])
def test_expanded_type_lists_keep_scalar_alternatives(
    array: bool, scalar: str, label: str
) -> None:
    record = obj({"value": node("number", "读数")})
    reading = arr(record) if array else record
    reading["type"] = [reading["type"], scalar]
    reading["description"] = "本次读数"
    schema = obj({"reading": reading})
    result = describe_tool(make_tool(schema), fields=["$.reading"])
    record_view = "[{value: float；读数}]" if array else "{value: float；读数}"
    expected = f"{{reading: anyOf({record_view}, {label})}}"
    assert result.output_fields == expected
    child = "$.reading[*].value" if array else "$.reading.value"
    paths = ["$.reading", child]
    for fields in ([paths[0]], paths, paths[::-1]):
        selected = describe_tool(make_tool(schema), fields=fields)
        assert selected.output_fields == expected
        assert selected.unknown_fields is None


@pytest.mark.parametrize("keyword", ["anyOf", "oneOf", "allOf"])
def test_same_object_branches_are_not_a_single_record(keyword: str) -> None:
    schema = {
        keyword: [
            obj({"kind": node("string", "温度记录"), "temperature": node("number")}),
            obj({"kind": node("string", "状态记录"), "status": node("string")}),
        ]
    }
    assert describe_tool(make_tool(schema)).output_fields == composition(
        keyword,
        {"kind": "str；温度记录", "temperature": "float"},
        {"kind": "str；状态记录", "status": "str"},
    )


def test_composition_names_are_literal_business_fields() -> None:
    record = obj(
        {
            name: node("string", name)
            for name in ("$composition", "branches", "anyOf", "oneOf", "allOf")
        }
    )
    expected = {name: f"str；{name}" for name in record["properties"]}
    assert describe_tool(make_tool(record)).output_fields == view_text(expected)
    schema = {"anyOf": [record, arr(record)]}
    assert describe_tool(make_tool(schema)).output_fields == composition(
        "anyOf", expected, [expected]
    )
    selected = describe_tool(make_tool(schema), fields=['$["$composition"]'])
    assert selected.output_fields == composition(
        "anyOf", {"$composition": "str；$composition"}
    )
    assert selected.unknown_fields is None


def test_nested_compositions_share_terms_without_requoting_inner_expressions() -> None:
    reading = {
        "anyOf": [
            obj({"temperature": node("number", "温度")}),
            arr(obj({"status": node("string", "状态")})),
        ]
    }
    schema = {
        "oneOf": [
            obj({"reading": reading}),
            obj({"fallback": node("string", "备用值")}),
        ]
    }
    result = describe_tool(make_tool(schema), fields=["$", "$.reading"])
    expected = (
        "oneOf({reading: anyOf({temperature: float；温度}, "
        "[{status: str；状态}])}, {fallback: str；备用值})"
    )
    assert result.output_fields == expected
    wire = json.loads(result.model_dump_json(exclude_none=True))
    assert wire["output_fields"] == expected
    assert describe_tool(
        make_tool(obj({"reading": reading}))
    ).output_fields == view_text({"reading": "anyOf(dict, list[dict])"})


def test_expression_names_and_terminal_strings_are_json_escaped() -> None:
    name = 'a.b"\\c\n'
    note = '温度"摄氏度"\\原值\n说明'
    schema = obj(
        {"reading": {"anyOf": [obj({name: node("number", note)}), node("null")]}}
    )
    registry = make_registry(make_tool(schema))
    result = registry.describe("read", fields=["$.reading"])
    escaped_name = r'["a.b\"\\c\n"]'
    expected = f"{{reading: anyOf({{{escaped_name}: float；{note}}}, None)}}"
    assert result.output_fields == expected
    assert json.loads(result.model_dump_json())["output_fields"] == expected
    path = "$.reading[" + json.dumps(name) + "]"
    selected = registry.describe("read", fields=[path])
    assert selected.output_fields == f"{{reading: {{{escaped_name}: float；{note}}}}}"
    assert selected.unknown_fields is None


@pytest.mark.parametrize("keyword", ["anyOf", "oneOf", "allOf"])
def test_group_depth_and_parent_child_union_do_not_depend_on_path_order(
    keyword: str,
) -> None:
    record = obj({"detail": obj({"value": node("number", "读数")}, "明细")})
    terminals = [
        node("null", "暂无读数"),
        node("string", "设备状态"),
        obj({}, "空记录"),
        node("array", "空序列", items=False),
        True,
        False,
    ]
    labels = [
        "None；暂无读数",
        "str；设备状态",
        "dict；空记录",
        "list[false]；空序列",
        "Any",
        "false",
    ]
    schema = obj({"reading": {keyword: [record, arr(record), *terminals]}})
    meta = make_tool(schema)
    expected = {
        "reading": composition(
            keyword, {"detail": "dict；明细"}, [{"detail": "dict；明细"}], *labels
        )
    }
    assert describe_tool(meta, fields=["$.reading"]).output_fields == view_text(
        expected
    )
    paths = ["$.reading", "$.reading[*].detail.value"]
    union = {
        "reading": composition(
            keyword,
            {"detail": "dict；明细"},
            [{"detail": {"value": "float；读数"}}],
            *labels,
        )
    }
    for fields in (paths, paths[::-1]):
        assert describe_tool(meta, fields=fields).output_fields == view_text(union)
    unknown = describe_tool(meta, fields=["$.reading", "$.reading.missing.leaf"])
    assert unknown.output_fields == view_text(expected)
    assert unknown.unknown_fields == ["$.reading.missing.leaf"]
    assert unknown.available_fields == {
        "$.reading": ["$.reading.detail", "$.reading[*]"]
    }


def test_unknown_paths_use_the_nearest_parent_across_applicable_branches() -> None:
    schema = obj(
        {
            "reading": {
                "anyOf": [
                    obj({"detail": obj({"temperature": node("number")})}),
                    arr(obj({"status": node("string")})),
                ]
            }
        }
    )
    result = describe_tool(
        make_tool(schema),
        fields=[
            "$.reading.detail.missing",
            "$.reading[*].status",
            "$.reading[*].other",
        ],
    )
    assert result.unknown_fields == ["$.reading.detail.missing", "$.reading[*].other"]
    assert result.available_fields == {
        "$.reading.detail": ["$.reading.detail.temperature"],
        "$.reading[*]": ["$.reading[*].status"],
    }


def test_automatic_tuple_views_keep_each_declared_position() -> None:
    schema = obj(
        {
            "position": node(
                "array",
                prefixItems=[
                    obj({"latitude": node("number", "纬度")}),
                    obj({"longitude": node("number", "经度")}),
                ],
                items=False,
            )
        }
    )
    assert describe_tool(make_tool(schema), fields=["$.position"]).output_fields == (
        "{position: [[0]: {latitude: float；纬度}, [1]: {longitude: float；经度}]}"
    )


def test_sparse_tuple_positions_keep_their_labels() -> None:
    schema = obj(
        {
            "position": node(
                "array",
                prefixItems=[
                    node("number", "纬度"),
                    node("number", "经度"),
                    node("integer", "高度"),
                ],
                items=False,
            )
        },
    )
    meta = make_tool(schema)
    expected = "{position: [[0]: float；纬度, [2]: int；高度]}"
    paths = ["$.position[2]", "$.position[0]"]
    for fields in (paths, paths[::-1]):
        result = describe_tool(meta, fields=fields)
        assert result.output_fields == expected
        assert result.unknown_fields is None


def test_expanded_scalar_array_keeps_element_branch_notes() -> None:
    schema = arr(
        {
            "anyOf": [node("number", "摄氏度"), node("string", "设备状态")],
            "description": "本次读数",
        },
        "采样序列",
    )
    registry = make_registry(make_tool(schema))
    for fields in (None, ["$"], ["$[*]"]):
        assert registry.describe(
            "read", fields=fields if fields is not None else ["$"]
        ).output_fields == ("[anyOf(float；摄氏度, str；设备状态)；本次读数]")
    for discover in (registry.search, registry.grep):
        assert discover("摄氏度")[0].matched_fields == [
            "$[*]: anyOf(float；摄氏度, str；设备状态)；本次读数"
        ]


@pytest.mark.parametrize(
    "alternative,label",
    [
        (True, "Any"),
        (False, "false"),
        (node("object"), "dict"),
        (node("null"), "None"),
        (node("object", additionalProperties=False), "dict"),
        (ref("Open"), "dict"),
    ],
)
def test_undeclared_fields_do_not_contribute_types(
    alternative: Any, label: str
) -> None:
    schema = node(
        "object",
        anyOf=[{"properties": {"value": node("integer", "编号")}}, alternative],
        **{"$defs": {"Open": node("object")}},
    )
    assert show(schema) == composition("anyOf", {"value": "int；编号"}, label)


@pytest.mark.parametrize("keyword", ["anyOf", "oneOf", "allOf"])
def test_combination_field_annotations(keyword: str) -> None:
    schema = obj(
        {"value": node("number", "测量值")},
        **{
            keyword: [
                {"properties": {"value": node(description=note)}}
                for note in ["常规区间", "扩展区间"]
            ]
        },
    )
    alternatives = [{"value": "Any；常规区间"}, {"value": "Any；扩展区间"}]
    assert show(schema) == composition(
        "allOf",
        {"value": "float；测量值"},
        *(
            alternatives
            if keyword == "allOf"
            else [composition(keyword, *alternatives)]
        ),
    )


def test_shared_object_array_fields_retain_both_definitions() -> None:
    schema = {
        "anyOf": [
            obj({"value": node("string", "标签"), "single": node("integer")}),
            arr(obj({"value": node("number", "重量"), "repeated": node("string")})),
        ]
    }
    registry = make_registry(make_tool(schema))
    assert registry.describe("read", fields=["$"]).output_fields == composition(
        "anyOf",
        {"value": "str；标签", "single": "int"},
        [{"value": "float；重量", "repeated": "str"}],
    )
    assert registry.describe(
        "read", fields=["$[*].repeated"]
    ).output_fields == view_text([{"repeated": "str"}])
    for query, text in [
        ("标签", "$.value: str；标签"),
        ("重量", "$[*].value: float；重量"),
    ]:
        assert registry.search(query)[0].matched_fields == [text]
