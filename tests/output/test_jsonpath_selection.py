"""验证检索返回的 JSONPath 可以直接用于输出定义查看。"""

import json
from copy import deepcopy
from typing import Any

import pytest
from pydantic import BaseModel, Field


from tests.support import prepare_field_documents, make_registry
from tests.support import composition, arr, describe_tool, make_tool, node, obj, ref
from tests.support import view_text


def test_discovered_array_paths_remain_known_in_array_and_tuple_branches() -> None:
    class Reading(BaseModel):
        speed: float = Field(description="车速")

    class Report(BaseModel):
        readings: list[Reading] | tuple[str, Reading]

    registry = make_registry(make_tool(Report.model_json_schema(mode="serialization")))
    assert "$.readings[*].speed: float；车速" in registry.grep("车速")[0].matched_fields
    result = registry.describe("read", fields=["$.readings[*].speed"])
    assert result.unknown_fields is result.available_fields is None
    assert result.output_fields == (
        "{readings: anyOf([{speed: float；车速}], [[0]: {}, [1]: {speed: float；车速}])}"
    )


@pytest.mark.parametrize(
    ("paths", "expected", "unknown"),
    [
        (["$.samples[*]"], {"samples": ["float；采样值"]}, None),
        (["$.position[1]"], "{position: [[1]: float；经度]}", None),
        (
            ["$.position[1]", "$.position[0]"],
            "{position: [[0]: float；纬度, [1]: float；经度]}",
            None,
        ),
        (["$.position[*]"], "{position: [[0]: float；纬度, [1]: float；经度]}", None),
        (["$.tail[1:]"], "{tail: [[1:]: str；补充项]}", None),
        (["$.records[*]"], {"records": [{"detail": "dict"}]}, None),
        (
            ["$.records[*].detail.value"],
            {"records": [{"detail": {"value": "int；数值"}}]},
            None,
        ),
        (["$.position[9]"], {"position": []}, ["$.position[9]"]),
        (["$.records.detail"], {"records": []}, ["$.records.detail"]),
        (["$.samples[0]"], {"samples": []}, ["$.samples[0]"]),
    ],
)
def test_declared_array_positions(
    paths: list[str], expected: Any, unknown: Any
) -> None:
    schema = obj(
        {
            "samples": arr(node("number", "采样值")),
            "position": node(
                "array",
                prefixItems=[node("number", "纬度"), node("number", "经度")],
                items=False,
            ),
            "tail": node(
                "array", prefixItems=[node("integer")], items=node("string", "补充项")
            ),
            "records": arr(
                obj({"detail": obj({"value": node("integer", "数值")})}, "单条记录")
            ),
        }
    )
    before = deepcopy(schema)
    view = describe_tool(make_tool(schema), fields=paths)
    assert view.output_fields == view_text(expected)
    assert view.unknown_fields == unknown
    assert schema == before


@pytest.mark.parametrize("prefix_keyword", ["items", "prefixItems"])
def test_referenced_tuple_siblings_keep_discovered_tail_paths(
    prefix_keyword: str,
) -> None:
    tail_keyword = "additionalItems" if prefix_keyword == "items" else "items"
    schema = obj(
        {
            "position": {
                **ref("Base"),
                "type": "array",
                prefix_keyword: [node("number", "固定值")],
                tail_keyword: node("string", "补充项"),
            }
        },
        **{"$defs": {"Base": {}}},
    )
    before = deepcopy(schema)
    meta = make_tool(schema)
    fields = prepare_field_documents(schema).fields
    assert fields[-1].text == "$.position[1:]: str；补充项"
    for field in fields:
        assert describe_tool(meta, fields=[field.path]).unknown_fields is None
    assert (
        describe_tool(meta, fields=[fields[-1].path]).output_fields
        == "{position: [[1:]: str；补充项]}"
    )
    assert schema == before


@pytest.mark.parametrize("segment", ["[0]", "[1:]"])
@pytest.mark.parametrize("reverse", [False, True])
def test_wildcard_and_specific_tuple_selections_are_combined(
    segment: str,
    reverse: bool,
) -> None:
    record = obj(
        {
            "a": node("number"),
            "b": node("string"),
            "detail": obj({"x": node("integer"), "y": node("boolean")}),
        }
    )
    schema = obj(
        {
            "position": node(
                "array",
                prefixItems=[ref("Record")],
                items=ref("Record"),
            )
        },
        **{"$defs": {"Record": {"anyOf": [record, node("null")]}}},
    )
    paths = [
        "$.position[*].a",
        f"$.position{segment}.b",
        "$.position[*].detail.x",
        f"$.position{segment}.detail.y",
    ]
    if reverse:
        paths.reverse()
    before = deepcopy((schema, paths))
    view = describe_tool(make_tool(schema), fields=paths)
    combined = {"a": "float", "b": "str", "detail": {"x": "int", "y": "bool"}}
    wildcard = {"a": "float", "detail": {"x": "int"}}
    first, tail = (combined, wildcard) if segment == "[0]" else (wildcard, combined)
    assert view.output_fields == (
        f"{{position: [[0]: {view_text(first)}, [1:]: {view_text(tail)}]}}"
    )
    assert view.unknown_fields is None
    assert (schema, paths) == before


def test_tuple_overlap_keeps_referenced_fields_and_unknowns() -> None:
    schema = node(
        "array",
        prefixItems=[obj({"a": ref("A"), "b": ref("B")})],
        items=False,
        **{"$defs": {"A": node("number"), "B": node("string")}},
    )
    paths = ["$[*].a", "$[*].missing", "$[0].b"]
    before = deepcopy(paths)
    view = describe_tool(make_tool(schema), fields=paths)
    assert view.output_fields == "[[0]: {a: float, b: str}]"
    assert view.unknown_fields == ["$[0].missing"]
    assert paths == before


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize(
    "paths,unknown",
    [
        (["$[*].detail.left"], None),
        (["$[*].detail.left", "$[1].detail.right"], None),
        (["$[*].detail.left", "$[1].detail.left"], ["$[1].detail.left"]),
        (["$[*].detail.left", "$[2:].detail.left"], ["$[2:].detail.left"]),
        (["$[1].detail.missing"], ["$[1].detail.missing"]),
        (
            ["$[*].detail.left", "$[*].detail.missing"],
            ["$[0].detail.missing", "$[1].detail.missing", "$[2:].detail.missing"],
        ),
    ],
)
def test_wildcard_unknowns_follow_the_original_selection(
    paths: list[str], unknown: list[str] | None, reverse: bool
) -> None:
    schema = node(
        "array",
        prefixItems=[
            obj({"detail": obj({"left": node("number")})}),
            obj({"detail": obj({"right": node("string")})}),
        ],
        items=obj({"detail": obj({"tail": node("boolean")})}),
    )
    if reverse:
        paths = paths[::-1]
    before = deepcopy((schema, paths))
    view = describe_tool(make_tool(schema), fields=paths)
    assert view.unknown_fields == unknown
    assert (schema, paths) == before


@pytest.mark.parametrize("leaf", ["value", "missing"])
def test_wildcard_unknowns_keep_the_deepest_existing_prefix(leaf: str) -> None:
    schema = node(
        "array",
        prefixItems=[node("string"), obj({"detail": obj({"value": node("number")})})],
        items=False,
    )
    view = describe_tool(make_tool(schema), fields=[f"$[*].detail.{leaf}"])
    assert view.unknown_fields == (
        ["$[1].detail.missing"] if leaf == "missing" else None
    )


@pytest.mark.parametrize("wildcard_is_parent", [False, True])
def test_overlapping_tuple_parents_keep_the_union(
    wildcard_is_parent: bool,
) -> None:
    record = obj({"detail": obj({"x": node("integer"), "y": node("boolean")}, "明细")})
    schema = node("array", prefixItems=[record], items=record)
    paths = (
        ["$[*].detail", "$[0].detail.y"]
        if wildcard_is_parent
        else ["$[*].detail.y", "$[0].detail"]
    )
    complete = {"detail": {"x": "int", "y": "bool"}}
    leaf = {"detail": {"y": "bool"}}
    view = describe_tool(make_tool(schema), fields=paths)
    assert view.output_fields == (
        f"[[0]: {view_text(complete)}, [1:]: {view_text(complete if wildcard_is_parent else leaf)}]"
    )
    assert view.unknown_fields is None


@pytest.mark.parametrize(
    "key", ["a.b", "vehicle name", "a_b", "a-b", r"a\b", "[*]", "", "温度", "a'b\"c\\d"]
)
def test_literal_names_can_be_copied_or_single_quoted(key: str) -> None:
    schema = obj({key: node("number", "读数"), "a": obj({"b": node("integer")})})
    meta = make_tool(schema)
    path = prepare_field_documents(schema).fields[0].path
    assert describe_tool(meta, fields=[path]).output_fields == view_text(
        {key: "float；读数"}
    )
    quoted = (
        json.dumps(key, ensure_ascii=False)[1:-1]
        .replace(r"\"", '"')
        .replace("'", r"\'")
    )
    assert describe_tool(meta, fields=[f"$['{quoted}']"]).output_fields == view_text(
        {key: "float；读数"}
    )
    assert describe_tool(meta, fields=["$.a.b"]).output_fields == view_text(
        {"a": {"b": "int"}}
    )


@pytest.mark.parametrize(
    "schema",
    [
        node("number", "读数"),
        arr(node("string", "元素")),
        obj({"a": obj({"b": node("integer")})}),
    ],
)
def test_explicit_root_expands_one_level(schema: dict) -> None:
    meta = make_tool(schema)
    assert describe_tool(meta, fields=[]).output_fields == view_text({})
    assert (
        describe_tool(meta, fields=["$"]).output_fields
        == describe_tool(meta).output_fields
    )
    assert (
        describe_tool(meta, fields=["$"]).output_fields
        == describe_tool(meta).output_fields
    )


@pytest.mark.parametrize(
    "paths,expected",
    [
        (["$.value"], composition("anyOf", {"value": "str；标签"})),
        (
            ["$[*].value"],
            [{"value": "float；重量"}],
        ),
        (
            ["$.value", "$[*].value"],
            composition("anyOf", {"value": "str；标签"}, [{"value": "float；重量"}]),
        ),
    ],
)
def test_object_array_alternatives_select_the_declared_position(
    paths: list[str], expected: Any
) -> None:
    meta = make_tool(
        {
            "anyOf": [
                obj({"value": node("string", "标签")}),
                arr(obj({"value": node("number", "重量")})),
            ]
        }
    )
    view = describe_tool(meta, fields=paths)
    assert view.output_fields == view_text(expected) and view.unknown_fields is None


@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"a": ["b"]},
        ["a"],
        "$.a",
        2,
        False,
        ["$.x", 3],
        ("$.x",),
        ["$..a"],
        ["$.a[?@.b]"],
        ["$.a[-1]"],
        ["$['a', 'b']"],
        ["$.a[0:2]"],
        [r'$["a\q"]'],
        ["$" + ".a" * 65],
    ],
)
def test_invalid_and_unsupported_selectors_are_rejected(fields: Any) -> None:
    with pytest.raises(ValueError):
        describe_tool(make_tool(obj({"a": node("integer")})), fields=fields)


def test_explicit_selection_expands_one_level_below_the_selected_node() -> None:
    schema = obj(
        {"outer": obj({"selected": ref("Selected")})},
        **{
            "$defs": {
                "Selected": obj({"child": ref("Child")}),
                "Child": obj({"loop": ref("Child")}),
            }
        },
    )
    assert (
        describe_tool(make_tool(schema), fields=["$.outer.selected"]).output_fields
        == "{outer: {selected: {child: dict}}}"
    )
