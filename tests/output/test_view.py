"""验证紧凑终点释义、字段选择和数组展开深度。"""

import json
from copy import deepcopy
from typing import Any, Literal
from unittest.mock import Mock

import pytest
from pydantic import BaseModel, Field

from axi.models import ToolMeta
from axi.output.paths import name_segment
from axi.output.registration import normalize_schema
from axi.output.selection import normalize_paths
from axi.output.documents import SchemaDocument
from axi.output.view import render_view, select_view
from tests.support import (
    composition,
    arr,
    describe_tool,
    make_tool,
    node,
    obj,
    ref,
    show,
)
from tests.support import view_text

PLATE = "str；车牌/车辆编号"
PARENT = "dict；查询时间窗内的车辆汇总统计"
DETAIL = {"duration": "float；作业时长，小时"}
SUMMARY = {
    "forwardMileage": "float；前进里程，单位米",
    "workTime": "dict；作业时间明细",
    "unexplained": "dict",
}
FULL = {**SUMMARY, "workTime": DETAIL, "unexplained": DETAIL}
SHALLOW = {
    "plateNumber": PLATE,
    "summaryStatistic": PARENT,
    "history": "anyOf(list[dict], None)；历史统计",
}


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        (None, SHALLOW),
        (
            ["$", "$.summaryStatistic", "$.history"],
            {
                "plateNumber": PLATE,
                "summaryStatistic": SUMMARY,
                "history": composition("anyOf", [SUMMARY], "None"),
            },
        ),
        (
            [
                "$",
                "$.summaryStatistic",
                "$.summaryStatistic.workTime",
                "$.summaryStatistic.unexplained",
                "$.history",
                "$.history[*].workTime",
                "$.history[*].unexplained",
            ],
            {
                "plateNumber": PLATE,
                "summaryStatistic": FULL,
                "history": composition("anyOf", [FULL], "None"),
            },
        ),
        (["$.summaryStatistic"], {"summaryStatistic": SUMMARY}),
        (
            [
                "$.summaryStatistic",
                "$.summaryStatistic.workTime",
                "$.summaryStatistic.unexplained",
            ],
            {"summaryStatistic": FULL},
        ),
        (
            ["$.summaryStatistic.workTime.duration"],
            {"summaryStatistic": {"workTime": DETAIL}},
        ),
        ([], {}),
    ],
)
def test_view_selection(nested_meta: ToolMeta, fields: Any, expected: Any) -> None:
    before = nested_meta.model_dump()
    nested_meta.output_example = {"example": 7}
    view = describe_tool(nested_meta, fields=fields)
    assert view.output_fields == view_text(expected)
    assert view.unknown_fields is None
    # 字段选择只影响查看范围，不隐藏输出类型或示例。
    assert view.output_type == "dict" and view.output_example == {"example": 7}
    assert "output_schema" not in view.model_dump(exclude_unset=True)
    nested_meta.output_example = before["output_example"]
    assert nested_meta.model_dump() == before


@pytest.mark.parametrize(
    ("fields", "expected", "unknown"),
    [
        (None, {"records": "list[list[dict]]；批次"}, None),
        (
            ["$.records"],
            {"records": [[{"id": "int；编号", "details": "dict；明细"}]]},
            None,
        ),
        (
            ["$.records", "$.records[*][*].details"],
            {"records": [[{"id": "int；编号", "details": {"value": "float；数值"}}]]},
            None,
        ),
        (
            ["$.records[*][*].details.value"],
            {"records": [[{"details": {"value": "float；数值"}}]]},
            None,
        ),
        (
            ["$.records[*][*].missing"],
            {"records": [[{}]]},
            ["$.records[*][*].missing"],
        ),
    ],
)
def test_nested_arrays(fields: Any, expected: Any, unknown: Any) -> None:
    record = obj(
        {
            "id": node("integer", "编号"),
            "details": obj({"value": node("number", "数值")}, "明细"),
        }
    )
    meta = make_tool(obj({"records": arr(arr(record), "批次")}))
    view = describe_tool(meta, fields=fields)
    assert view.output_fields == view_text(expected) and view.unknown_fields == unknown


@pytest.mark.parametrize("name", ["rows", "a.b", "[0]", "[*]", "[2:]"])
def test_literal_container_names_keep_property_depth(name: str) -> None:
    record = obj({"detail": obj({"value": node("number", "读数")}, "明细")})
    schema = obj({name: arr(record, "记录")})
    before = deepcopy(schema)
    meta = make_tool(schema)
    collapsed = {name: "list[dict]；记录"}
    shallow = describe_tool(meta)
    assert shallow.output_fields == view_text(collapsed)
    expanded = describe_tool(meta, fields=["$" + name_segment(name)])
    assert expanded.output_fields == view_text({name: [{"detail": "dict；明细"}]})
    path = f"$[{json.dumps(name)}][*].detail.value"
    selected = describe_tool(meta, fields=[path])
    assert selected.output_fields == view_text(
        {name: [{"detail": {"value": "float；读数"}}]}
    )
    assert selected.unknown_fields is None
    assert schema == before


@pytest.mark.parametrize(
    "schema", [arr(obj({"id": node("integer")})), obj({"id": node("integer")})]
)
def test_unknown_fields_offer_record_directory(schema: dict) -> None:
    view = describe_tool(
        make_tool(schema),
        fields=["$[*].missing"] if "items" in schema else ["$.missing"],
    )
    assert view.unknown_fields == (
        ["$[*].missing"] if "items" in schema else ["$.missing"]
    )
    assert view.available_fields == {
        "$[*]" if "items" in schema else "$": [
            "$[*].id" if "items" in schema else "$.id"
        ]
    }


def test_unknown_paths_offer_only_the_nearest_parent_directory() -> None:
    schema = obj(
        {
            "unrelated": obj({"large": node("string")}),
            "summary": obj({"a.b": node("number"), "rows": arr(node("integer"))}),
        }
    )
    result = describe_tool(
        make_tool(schema),
        fields=["$.summary.missing.leaf", "$.summary.rows[0].x"],
    )
    assert result.unknown_fields == ["$.summary.missing.leaf", "$.summary.rows[0].x"]
    assert result.available_fields == {
        "$.summary": ['$.summary["a.b"]', "$.summary.rows"],
        "$.summary.rows": ["$.summary.rows[*]"],
    }
    assert "unrelated" not in json.dumps(result.available_fields)


@pytest.mark.parametrize("key", ["target", "a.b", "车辆名称", "[*]", 'a"b'])
def test_single_selected_property_does_not_scan_its_siblings(key: str) -> None:
    class Properties(dict):
        def items(self) -> Any:
            raise AssertionError("A single property lookup scanned all siblings")

    schema = obj(
        Properties(
            {**{f"field{i}": node("number") for i in range(10000)}, key: node("string")}
        )
    )
    path = "$" + name_segment(key)
    assert show(schema, fields=[path]) == view_text({key: "str"})


def test_multiple_selected_properties_keep_schema_declaration_order() -> None:
    schema = obj({"z": node("number"), "a": node("integer"), "other": node("string")})
    assert show(schema, fields=["$.a", "$.z"]) == "{z: float, a: int}"


@pytest.mark.parametrize(
    "schema,available",
    [
        (
            {
                **ref("Base"),
                "properties": {"own": node("string")},
                "$defs": {"Base": obj({"base": node("number")})},
            },
            ["$.base", "$.own"],
        ),
        (
            {
                "anyOf": [
                    node("array", prefixItems=[True, False], items=True),
                    obj({"a.b": node("string"), "[0]": node("number")}),
                ]
            },
            ['$["a.b"]', '$["[0]"]', "$[0]", "$[1]", "$[2:]"],
        ),
        (
            {
                **ref("A"),
                "$defs": {
                    "A": {"anyOf": [ref("B"), obj({"a": True})]},
                    "B": {"anyOf": [ref("A"), obj({"b": True})]},
                },
            },
            ["$.b", "$.a"],
        ),
    ],
)
def test_unknown_directory_preserves_branch_order_and_declared_positions(
    schema: dict[str, Any], available: list[str]
) -> None:
    selected = select_view(SchemaDocument(schema), normalize_paths(["$.missing"]))
    assert selected.unknown == ["$.missing"]
    assert selected.available == {"$": available}


def test_selection_and_formatting_do_not_share_traversal_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = obj({"id": node("integer", "编号")}, "父级")
    schema = obj({"record": ref("Record")}, "根", **{"$defs": {"Record": definition}})
    before = deepcopy(schema)
    forbidden = Mock(
        side_effect=AssertionError("Crossed selection/formatting boundary")
    )
    with monkeypatch.context() as selection:
        selection.setattr(SchemaDocument, "format", forbidden)
        selection.setattr(SchemaDocument, "expression", forbidden)
        selected = select_view(SchemaDocument(schema), normalize_paths(["$.record.id"]))
    terminal = selected.fields["record"]["id"]
    assert (
        selected.document.root is schema
        and terminal.definition is definition["properties"]["id"]
    )
    with monkeypatch.context() as formatting:
        formatting.setattr(SchemaDocument, "shape", forbidden)
        assert render_view(selected) == view_text({"record": {"id": "int；编号"}})
    assert schema == before


def test_wide_view_keeps_all_declared_fields() -> None:
    record = {f"value{i}": node("number", "读数") for i in range(6)}
    schema = obj({f"group{i}": obj(record) for i in range(40)})
    assert render_view(
        select_view(
            SchemaDocument(schema), normalize_paths([f"$.group{i}" for i in range(40)])
        )
    ) == view_text(
        {f"group{i}": {name: "float；读数" for name in record} for i in range(40)}
    )


@pytest.mark.parametrize("keyword", ["anyOf", "oneOf", "allOf"])
def test_expanded_combinations_keep_nesting_and_terminal_annotations(
    keyword: str,
) -> None:
    left, right = node("number", "温度"), node("string", "状态")
    left_expected: Any = "float；温度"
    right_expected: Any = "str；状态"
    for _ in range(8):
        left, right = obj({"child": left}), obj({"child": right})
        left_expected, right_expected = (
            {"child": left_expected},
            {"child": right_expected},
        )
    selected = select_view(
        SchemaDocument({keyword: [left, right]}), normalize_paths(["$" + ".child" * 7])
    )
    assert render_view(selected) == composition(keyword, left_expected, right_expected)


def test_shared_structure_keeps_each_selected_range() -> None:
    complete = {"x": "int", "y": "str", "details": {"value": "float"}}
    shallow = {**complete, "details": "dict"}
    schema = obj(
        {
            "first": ref("Shared"),
            "second": ref("Shared"),
            "selected": ref("Shared"),
            "wrapper": obj({"value": ref("Shared")}),
        },
        **{
            "$defs": {
                "Shared": {
                    "anyOf": [
                        obj(
                            {
                                "x": node("integer"),
                                "y": node("string"),
                                "details": obj({"value": node("number")}),
                            }
                        ),
                        node("null"),
                    ]
                }
            }
        },
    )
    meta = make_tool(schema)
    assert describe_tool(
        meta,
        fields=[
            "$",
            "$.first",
            "$.second",
            "$.selected",
            "$.first.details",
            "$.second.details",
            "$.selected.details",
            "$.wrapper.value",
        ],
    ).output_fields == view_text(
        {
            "first": composition("anyOf", complete, "None"),
            "second": composition("anyOf", complete, "None"),
            "selected": composition("anyOf", complete, "None"),
            "wrapper": {"value": composition("anyOf", shallow, "None")},
        }
    )
    fields = ["$.first.x", "$.second.y", "$.selected", "$.wrapper.value"]
    assert describe_tool(meta, fields=fields).output_fields == view_text(
        {
            "first": composition("anyOf", {"x": "int"}),
            "second": composition("anyOf", {"y": "str"}),
            "selected": composition("anyOf", shallow, "None"),
            "wrapper": {"value": composition("anyOf", shallow, "None")},
        }
    )


@pytest.mark.parametrize("nullable", [False, True])
@pytest.mark.parametrize("description", [None, "响应容器"])
def test_empty_container_is_a_stable_terminal(
    nullable: bool, description: str | None
) -> None:
    schema = node(["object", "null"] if nullable else "object", description)
    meta = make_tool(schema)
    expected = ("dict | None" if nullable else "dict") + (
        "；" + description if description else ""
    )
    assert describe_tool(meta).output_fields == view_text(expected)
    view = describe_tool(meta, fields=["$.missing"])
    assert view.output_fields == view_text({}) and view.available_fields == {"$": []}
    assert view.unknown_fields == (
        ["$[*].missing"] if "items" in schema else ["$.missing"]
    )


def test_pydantic_leaf_descriptions_ignore_constraints_and_parent_context() -> None:
    class Vehicle(BaseModel):
        speed: float = Field(description="速度，km/h", ge=0, le=100)
        label: str = Field(description="名称", min_length=2, pattern="^[A-Z]+$")
        state: Literal["on", "off"] = Field(description="状态")

    class Output(BaseModel):
        vehicle: Vehicle | None = Field(description="父级车辆信息")
        history: list[Vehicle] = Field(description="父级历史记录", min_length=1)

    schema = normalize_schema(Output)
    original = deepcopy(schema)
    expected = {
        "speed": "float；速度，km/h",
        "label": "str；名称",
        "state": "str；状态",
    }
    meta = make_tool(description="读取", schema=schema)
    assert describe_tool(
        meta, fields=["$.vehicle", "$.history"]
    ).output_fields == view_text(
        {
            "vehicle": composition("anyOf", expected, "None"),
            "history": [expected],
        }
    )
    selected = describe_tool(meta, fields=["$.vehicle.speed"])
    assert selected.output_fields == view_text(
        {"vehicle": composition("anyOf", {"speed": "float；速度，km/h"})}
    )
    assert describe_tool(meta).output_fields == view_text(
        {
            "vehicle": "anyOf(dict, None)；父级车辆信息",
            "history": "list[dict]；父级历史记录",
        }
    )
    assert schema == original


def test_parent_child_paths_form_a_union_independent_of_input_order() -> None:
    paths = ["$.A", "$.A.x", "$.A.y", "$.B.C.z", "$.A.unknown_z", "$.A.unknown_a"]
    schema = obj(
        {
            "A": obj(
                {"x": node("integer"), "y": node("string"), "other": node("number")}
            ),
            "B": obj({"C": obj({"z": node("boolean")})}),
        }
    )
    expected = {
        "A": {"x": "int", "y": "str", "other": "float"},
        "B": {"C": {"z": "bool"}},
    }
    views = [
        describe_tool(make_tool(schema), fields=order) for order in (paths, paths[::-1])
    ]
    assert views[0] == views[1]
    assert views[0].output_fields == view_text(expected)
    assert views[0].unknown_fields == ["$.A.unknown_a", "$.A.unknown_z"]


def test_output_selection_requires_a_definition() -> None:
    meta = make_tool()
    for fields in (None, [], ["$.value"], ["$"]):
        view = describe_tool(meta, fields=fields)
        assert view.output_schema_status == "not_registered"
        assert (
            view.output_fields is view.unknown_fields is view.available_fields is None
        )


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("composed", [False, True])
def test_parent_range_preserves_siblings_and_branches(
    reverse: bool, composed: bool
) -> None:
    detail = obj({"x": node("integer"), "y": node("string")})
    children = (
        {"detail": {"anyOf": [detail, arr(detail), node("null")]}}
        if composed
        else {
            "detail": detail,
            "sibling": obj({"value": node("number")}),
        }
    )
    paths = ["$.A", "$.A.detail.x"]
    expected = {"x": "int"}
    if composed:
        expected = {
            "detail": composition(
                "anyOf",
                expected,
                "list[dict]",
                "None",
            )
        }
    else:
        expected = {
            "detail": expected,
            "sibling": "dict",
        }
    result = describe_tool(
        make_tool(obj({"A": obj(children), "B": node("boolean")})),
        fields=paths[::-1] if reverse else paths,
    )
    assert result.output_fields == view_text({"A": expected})
    assert result.unknown_fields is None


@pytest.mark.parametrize("levels", [16, 32, 64])
def test_deep_selection_preserves_original_paths(levels: int) -> None:
    schema, expected = node("integer"), "int"
    for _ in range(levels):
        schema, expected = obj({"child": schema}), {"child": expected}
    tree = normalize_paths(["$" + ".child" * levels])
    before = deepcopy(tree)
    assert render_view(select_view(SchemaDocument(schema), tree)) == view_text(expected)
    assert tree == before


def test_overlapping_selected_subtrees_preserve_selection_and_siblings() -> None:
    properties = {
        f"parent{i}": obj({"value": node("integer"), "sibling": node("string")})
        for i in range(8)
    }
    paths = ["$"] + [
        path for key in properties for path in (f"$.{key}", f"$.{key}.value")
    ]
    tree = normalize_paths(paths)
    before = deepcopy(tree)
    result = render_view(select_view(SchemaDocument(obj(properties)), tree))
    assert result == view_text(
        {key: {"value": "int", "sibling": "str"} for key in properties}
    )
    assert tree == before
