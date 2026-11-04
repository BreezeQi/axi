"""验证折叠类型、节点自身说明与引用目标已有说明。"""

from copy import deepcopy
from typing import Any

import pytest

from axi.output.explanation import SchemaFormatter
from tests.support import arr, make_registry, make_tool, node, obj, ref, show, view_text


@pytest.mark.parametrize(
    ("definition", "expected"),
    [
        (True, "Any"),
        (False, "false"),
        (node(description="数值", enum=[1]), "Any；数值"),
        (node("number", allOf=[node("integer")]), "allOf(float, int)"),
        (node(["number", "null"], allOf=[node("integer")]), "allOf(float | None, int)"),
        (
            node("number", anyOf=[node("integer"), node("string")]),
            "allOf(float, anyOf(int, str))",
        ),
        (node("string", allOf=[node("integer")]), "allOf(str, int)"),
        (
            node("array", allOf=[{"items": node("integer")}]),
            "allOf(list[Any], items(int))",
        ),
        ({"oneOf": [node("number"), node("string")]}, "oneOf(float, str)"),
        (
            {"oneOf": [node("number", minimum=0), node("number", maximum=100)]},
            "oneOf(float, float)",
        ),
        (node(["number", "integer"]), "float | int"),
        (node("string", **{"$ref": "#/missing"}), "allOf($ref(#/missing), str)"),
        ({"anyOf": [{}, node("number")]}, "anyOf(Any, float)"),
        (
            {"anyOf": [{"oneOf": [node("string"), node("integer")]}, node("number")]},
            "anyOf(oneOf(str, int), float)",
        ),
        (
            node("number", "值", minimum=0, maximum=10, enum=[1, 2], default=1),
            "float；值",
        ),
        (arr(node("number", "不继承元素说明"), "采样"), "list[float]；采样"),
        (arr(arr(node("number"), "内层")), "list[list[float]]"),
        (
            node("array", prefixItems=[node("string"), node("number")], items=False),
            "list[str | float]",
        ),
        (
            arr({"anyOf": [node("number"), node("number")]}),
            "list[float]",
        ),
        ({"anyOf": [node("number"), node("number")]}, "float"),
        (
            {"anyOf": [node("number", "温度"), node("number", "重量")]},
            "anyOf(float；温度, float；重量)",
        ),
        (
            {
                "anyOf": [
                    {"anyOf": [node("number"), node("string")]},
                    {"anyOf": [node("number"), node("string")]},
                ],
                "description": "设备温度",
            },
            "anyOf(float, str)；设备温度",
        ),
        (
            {"allOf": [node("number"), {"allOf": [node("number"), node("integer")]}]},
            "allOf(float, int)",
        ),
        (
            {"oneOf": [node("number"), {"oneOf": [node("number"), node("string")]}]},
            "oneOf(float, oneOf(float, str))",
        ),
        (
            {
                "anyOf": [
                    node("number", "重复 重复；原样"),
                    node("number", "重复 重复；原样"),
                ]
            },
            "float；重复 重复；原样",
        ),
    ],
)
def test_terminal_types_and_own_descriptions(definition: Any, expected: str) -> None:
    assert SchemaFormatter({}).format(definition) == expected


@pytest.mark.parametrize("keyword", ["anyOf", "allOf", "oneOf"])
@pytest.mark.parametrize("different_note", [False, True])
def test_reference_annotation_fragments_share_compact_explanations(
    keyword: str, different_note: bool
) -> None:
    unit = "单位华氏度" if different_note else "单位摄氏度"
    schema = obj(
        {
            "temperature": node(
                description="设备读数",
                **{
                    keyword: [
                        node("number", "温度；单位摄氏度"),
                        {**ref("Temperature"), "description": unit},
                    ]
                },
            )
        },
        **{"$defs": {"Temperature": node("number", "温度")}},
    )
    original = deepcopy(schema)
    label = "float；温度；单位摄氏度"
    if keyword == "oneOf" or different_note:
        label = f"{keyword}({label}, float；温度；{unit})"
    label += "；设备读数"
    assert SchemaFormatter(schema).format(schema["properties"]["temperature"]) == label
    assert schema == original


@pytest.mark.parametrize(
    "record,notes,expected",
    [
        (False, ("车速", "速度读数", "车速"), "车速；速度读数"),
        (False, ("车速；车速", "速度读数", "车速；车速"), "车速；车速；速度读数"),
        (False, ("车速；速度", "车速", "速度"), "车速；速度；车速；速度"),
        (True, ("车辆信息", "车辆信息"), "车辆信息"),
        (True, ("车辆信息", "本次车辆"), "车辆信息；本次车辆"),
        (True, ("重复 重复；原样", "重复 重复；原样"), "重复 重复；原样"),
    ],
)
def test_reference_annotations_keep_whole_prose(
    record: bool, notes: tuple[str, ...], expected: str
) -> None:
    target = (
        obj({"speed": node("number", "车速")}, notes[0])
        if record
        else node("number", notes[0])
    )
    definitions = {"Value": target}
    reference = ref("Value")
    for index, note in enumerate(notes[1:-1]):
        definitions[f"Alias{index}"] = {**reference, "description": note}
        reference = ref(f"Alias{index}")
    schema = obj(
        {"value": {**reference, "description": notes[-1]}}, **{"$defs": definitions}
    )
    before = deepcopy(schema)
    label = ("dict" if record else "float") + "；" + expected
    registry = make_registry(make_tool(schema))
    assert registry.describe("read", fields=["$"]).output_fields == view_text(
        {"value": label}
    )
    assert SchemaFormatter(schema).format(schema["properties"]["value"]) == label
    if record:
        assert (
            registry.describe("read", fields=["$.value"]).output_fields
            == "{value: {speed: float；车速}}"
        )
        for discover in (registry.search, registry.grep):
            assert discover(notes[-1])[0].matched_fields == [f"$.value: {label}"]
    assert schema == before


@pytest.mark.parametrize(
    "alternative,expected,expanded",
    [
        (True, "anyOf(list[int], Any)", "anyOf([int], Any)"),
        (node("array"), "anyOf(list[int], list[Any])", "anyOf([int], list[Any])"),
        (arr(node("string")), "anyOf(list[int], list[str])", "anyOf([int], [str])"),
        (arr(False), "anyOf(list[int], list[false])", "anyOf([int], list[false])"),
        (node("null"), "anyOf(list[int], None)", "anyOf([int], None)"),
    ],
)
def test_union_array_items(alternative: Any, expected: str, expanded: str) -> None:
    schema = {"anyOf": [arr(node("integer")), alternative], "description": "结果列表"}
    assert SchemaFormatter(schema).format(schema) == expected + "；结果列表"
    assert show(schema) == expanded


def test_legacy_tuple_and_recursive_alias_types() -> None:
    legacy = node(
        "array", items=[node("string"), node("number")], additionalItems=False
    )
    assert (
        SchemaFormatter(legacy).format(legacy, descriptions=False)
        == "list[str | float]"
    )
    alias = {"$defs": {"A": ref("B"), "B": ref("A")}, **ref("A")}
    assert SchemaFormatter(alias).format(alias, descriptions=False) == "$ref(#/$defs/A)"
    definition = {"items": node("number")}
    for _ in range(20):
        definition = {"allOf": [definition]}
    definition = node("array", allOf=[definition])
    for _ in range(60):
        definition = arr(definition)
    assert SchemaFormatter(definition).format(definition, descriptions=False) == (
        "list[" * 60 + "allOf(list[Any], " + "items(float)" + ")" + "]" * 60
    )


def test_composite_reference_siblings_keep_their_own_type_declarations() -> None:
    schema = obj(
        {"value": {"anyOf": [{**ref("Value"), "type": "number"}, node("null")]}},
        **{
            "$defs": {
                "Value": node(
                    description="编号", allOf=[node("string"), node("integer")]
                )
            }
        },
    )
    assert show(schema) == view_text(
        {"value": "anyOf(allOf(allOf(str, int)；编号, float), None)"}
    )
    schema["properties"]["value"]["anyOf"][0].pop("type")
    assert show(schema) == view_text({"value": "anyOf(allOf(str, int)；编号, None)"})
