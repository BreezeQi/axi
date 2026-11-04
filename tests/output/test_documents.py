"""验证字段文档、JSONPath 声明位置与共享终点释义。"""

import re
from collections.abc import ItemsView
from copy import deepcopy
from typing import Any

import pytest

from axi.output.explanation import SchemaFormatter
from axi.output.schema import MAX_DATA_DEPTH, SchemaNode
from axi.search.corpus import build_corpus
from tests.support import prepare_field_documents, make_registry
from tests.support import arr, describe_tool, make_tool, node, obj, ref
from tests.support import selected_paths, view_text


def test_wide_field_preparation_does_not_rescan_properties_for_each_path() -> None:
    visits = 0

    class Properties(dict[str, dict]):
        def items(self) -> ItemsView[str, dict]:
            nonlocal visits
            visits += len(self)
            return super().items()

    width = 100
    properties = Properties({f"f{i}": node("number") for i in range(width)})
    index = prepare_field_documents(obj(properties))
    assert len(index.fields) == width
    assert index.fields[-1].text == "$.f99: float"
    assert visits <= 2 * width


@pytest.mark.parametrize("keyword", ["allOf", "anyOf", "oneOf"])
@pytest.mark.parametrize("array", [False, True])
def test_composed_field_preparation_does_not_build_views(
    monkeypatch: pytest.MonkeyPatch, keyword: str, array: bool
) -> None:
    schema = {keyword: [obj({f"f{i}": node("number")}) for i in range(40)]}
    if array:
        schema = obj({"rows": arr(schema)})

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Search prepared a describe view")

    monkeypatch.setattr("axi.output.view.select_view", forbidden)
    index = prepare_field_documents(schema)
    path = "$.rows[*].f39" if array else "$.f39"
    assert path + ": float" in [field.text for field in index.fields]


def test_declared_positions_include_scalar_items_and_fixed_tuple_slots() -> None:
    schema = obj(
        {
            "temperatures": arr(
                node("number", "温度，单位摄氏度"), "温度采样序列", maxItems=0
            ),
            "position": node(
                "array",
                prefixItems=[node("number", "纬度"), node("number", "经度")],
                items=False,
            ),
            "groups": arr(arr(obj({"name": node("string", "名称")}))),
        }
    )
    fields = prepare_field_documents(schema).fields
    assert [(f.path, f.depth) for f in fields] == [
        ("$.temperatures", 1),
        ("$.temperatures[*]", 2),
        ("$.position", 1),
        ("$.position[0]", 2),
        ("$.position[1]", 2),
        ("$.groups", 1),
        ("$.groups[*]", 2),
        ("$.groups[*][*]", 3),
        ("$.groups[*][*].name", 4),
    ]
    assert [f.text for f in fields if f.path.startswith("$.position")] == [
        "$.position: list[float]",
        "$.position[0]: float；纬度",
        "$.position[1]: float；经度",
    ]
    registry = make_registry(make_tool(schema))
    for field in fields:
        assert registry.describe("read", fields=[field.path]).unknown_fields is None
    for discover in (registry.search, registry.grep):
        assert discover("经度")[0].matched_fields == ["$.position[1]: float；经度"]
        assert discover("摄氏度")[0].matched_fields == [
            "$.temperatures[*]: float；温度，单位摄氏度"
        ]


@pytest.mark.parametrize("levels", [35, 64])
@pytest.mark.parametrize("declaration", ["inline", "reference", "composition"])
def test_deep_declared_positions_agree_between_discovery_and_view(
    levels: int, declaration: str
) -> None:
    schema = node("number", "独立温度指标")
    definitions = {}
    segments = []
    for level in reversed(range(levels)):
        child = schema
        if declaration == "reference":
            definitions[f"L{level}"] = child
            child = ref(f"L{level}")
        if level % 3 == 0:
            schema, segment = obj({"child": child}), ".child"
        elif level % 3 == 1:
            schema, segment = arr(child), "[*]"
        else:
            schema, segment = node("array", prefixItems=[child], items=False), "[0]"
        if declaration == "composition":
            schema = {"anyOf": [schema, node("null")]}
        segments.insert(0, segment)
    if definitions:
        schema["$defs"] = definitions
    path = "$" + "".join(segments)
    expected = path + ": float；独立温度指标"
    registry = make_registry(make_tool(schema))
    from axi.output.paths import parse_path

    parent = "$" + "".join(parse_path(path)[:-1])
    for options in ({"fields": [path]}, {"fields": [parent]}):
        view = registry.describe("read", **options)
        assert view.unknown_fields is None
        assert "独立温度指标" in view.output_fields
    for query in (path, "独立温度指标"):
        assert registry.search(query)[0].matched_fields == [expected]
        assert registry.grep(re.escape(query))[0].matched_fields == [expected]


@pytest.mark.parametrize("levels", [63, 64, 65, 66])
@pytest.mark.parametrize("position", ["property", "array", "tuple"])
def test_discovery_selection_and_expansion_share_the_data_depth_limit(
    levels: int, position: str
) -> None:
    schema = node("number", "边界温度指标")
    segment = {"property": ".child", "array": "[*]", "tuple": "[0]"}[position]
    for _ in range(levels):
        if position == "property":
            schema = obj({"child": schema}, "容器")
        elif position == "array":
            schema = arr(schema, "容器")
        else:
            schema = node("array", "容器", prefixItems=[schema], items=False)
    registry = make_registry(make_tool(schema))
    fields = prepare_field_documents(schema).fields
    last = fields[-1]
    assert len(fields) == last.depth == min(levels, MAX_DATA_DEPTH)
    assert last.path == "$" + segment * last.depth
    assert selected_paths(schema, fields=[last.path]) == {
        (last.path, last.text.split(": ", 1)[1])
    }
    view = registry.describe("read", fields=[last.path])
    assert (
        view.output_fields
        == registry.describe("read", fields=[last.path]).output_fields
    )
    assert ("边界温度指标" in view.output_fields) is (levels <= MAX_DATA_DEPTH)
    for discover in (registry.search, registry.grep):
        assert [hit.matched_fields for hit in discover("边界温度指标")] == (
            [[last.text]] if levels <= MAX_DATA_DEPTH else []
        )
    if levels > MAX_DATA_DEPTH:
        with pytest.raises(ValueError, match="exceeds 64 levels"):
            registry.describe("read", fields=["$" + segment * levels])


def test_tuple_explicit_tail_and_undeclared_elements() -> None:
    schema = obj(
        {
            "tuple": node("array", prefixItems=[node("string")], items=node("number")),
            "open": node("array"),
            "closed": arr(False),
        }
    )
    assert [f.path for f in prepare_field_documents(schema).fields] == [
        "$.tuple",
        "$.tuple[0]",
        "$.tuple[1:]",
        "$.open",
        "$.closed",
    ]


@pytest.mark.parametrize("keyword", ["anyOf", "oneOf", "allOf"])
def test_branches_keep_paths_and_descriptions_with_index_time_dedup(
    keyword: str,
) -> None:
    schema = obj(
        {
            "payload": {
                keyword: [
                    obj({"value": node("string", "标签")}),
                    obj({"value": node("number", "重量 重量，单位千克")}),
                    obj({"value": node("number", "重量 重量，单位千克")}),
                    arr(obj({"value": node("number", "重量 重量，单位千克")})),
                ]
            }
        }
    )
    before = deepcopy(schema)
    index = prepare_field_documents(schema)
    fields = index.fields
    values = [f.text for f in fields if f.name == "value"]
    assert values == [
        "$.payload.value: str；标签",
        "$.payload.value: float；重量 重量，单位千克",
        "$.payload[*].value: float；重量 重量，单位千克",
    ]
    corpus = build_corpus(
        [make_tool(schema, "A"), make_tool(schema, "B")], [index, index]
    )
    for value in values:
        assert corpus.output_texts.count(value) == 2
    registry = make_registry(make_tool(schema))
    assert registry.grep("重量")[0].matched_fields == values[1:]
    assert registry.search("$.payload.value")[0].matched_fields == values[:2]
    assert schema == before


@pytest.mark.parametrize("reference_first", [False, True])
def test_repeated_positions_format_once_without_skipping_first_emission(
    monkeypatch: pytest.MonkeyPatch, reference_first: bool
) -> None:
    shared = node("number", "温度")
    branches = [obj({"value": shared}) for _ in range(100)]
    expected = ["$.value: float；温度"]
    if reference_first:
        branches.insert(0, obj({"value": {**ref("Value"), "description": "引用位置"}}))
        expected.insert(0, "$.value: float；温度；引用位置")
    schema = {"anyOf": branches, "$defs": {"Value": shared}}
    original, calls = SchemaFormatter.format, 0

    def format_node(
        formatter: SchemaFormatter, definition: SchemaNode, **options: Any
    ) -> str:
        nonlocal calls
        if definition is shared:
            calls += 1
        return original(formatter, definition, **options)

    monkeypatch.setattr(SchemaFormatter, "format", format_node)
    assert [field.text for field in prepare_field_documents(schema).fields] == expected
    assert calls == 1


@pytest.mark.parametrize("keyword", ["anyOf", "oneOf", "allOf"])
@pytest.mark.parametrize(
    "members,description,query,label",
    [
        (
            [node("string", "标签"), node("number", "重量，单位千克")],
            "测量值",
            "重量",
            "str；标签, float；重量，单位千克",
        ),
        ([node("string"), node("integer")], "业务编号", "业务编号", "str, int"),
    ],
)
def test_scalar_compositions_share_complete_documents(
    keyword: str, members: list[SchemaNode], description: str, query: str, label: str
) -> None:
    schema = obj({"value": {keyword: members, "description": description}})
    before = deepcopy(schema)
    expected = f"{keyword}({label})；{description}"
    registry = make_registry(make_tool(schema))
    view = registry.describe("read", fields=["$"])
    assert view.output_fields == view_text({"value": expected})
    assert registry.describe("read", fields=["$.value"]).output_fields == view_text(
        {"value": expected}
    )
    for discover in (registry.search, registry.grep):
        assert discover(query)[0].matched_fields == ["$.value: " + expected]
    assert schema == before


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize(
    "key,segment",
    [
        ("forwardMileage", ".forwardMileage"),
        ("forward_mileage", ".forward_mileage"),
        ("forward-mileage", '["forward-mileage"]'),
        ("temperature.value", '["temperature.value"]'),
        (r"temperature\value", r'["temperature\\value"]'),
        ("a.b", '["a.b"]'),
        ("vehicle name", '["vehicle name"]'),
        ("   ", '["   "]'),
        ('a"b', '["a\\"b"]'),
        ("a'b", '["a\'b"]'),
        (r"a\b", r'["a\\b"]'),
        ("", '[""]'),
        ("0", '["0"]'),
    ],
)
def test_literal_names_have_unambiguous_paths_and_exact_lookup(
    key: str, segment: str, nested: bool
) -> None:
    properties = {key: node("number", "前进里程，单位米")}
    if nested:
        properties = {"summaryStatistic": obj(properties, "车辆汇总统计")}
    schema = obj(properties, "查询时间窗 查询时间窗")
    index = prepare_field_documents(schema)
    field = index.fields[-1]
    path = ("$.summaryStatistic" if nested else "$") + segment
    expected = [path + ": float；前进里程，单位米"]
    assert field.path == path and field.depth == (2 if nested else 1)
    assert field.text == expected[0]
    assert index.root_text == "查询时间窗 查询时间窗"
    registry = make_registry(make_tool(schema, description="读取统计"))
    for query in (path, key) if key.strip() else (path,):
        hits = registry.search(query)
        assert len(hits) == 1 and hits[0].name == "read"
        assert hits[0].matched_fields == expected
    if key:
        raw_hits = registry.grep(re.escape(key))
        if "\\" in key or '"' in key:
            assert raw_hits == []  # 引号和反斜线已被 JSONPath 转义。
        else:
            assert raw_hits[0].matched_fields == expected
    assert registry.grep(re.escape(path))[0].matched_fields == expected


@pytest.mark.parametrize("keyword", ["anyOf", "oneOf", "allOf"])
def test_root_notes_keep_reference_order_without_child_prose(keyword: str) -> None:
    schema = {
        **ref("Shared"),
        "description": "根说明 根说明",
        "$defs": {
            "Shared": obj(
                {"value": node("number", "字段自身说明")}, "共享说明", **ref("Alias")
            ),
            "Alias": {**ref("Shared"), "description": "别名说明"},
        },
        keyword: [
            {**ref("Shared"), "description": "分支说明"},
            {"description": "共享说明"},
        ],
    }
    before = deepcopy(schema)
    index = prepare_field_documents(schema)
    assert index.root_text == "根说明 根说明 共享说明 别名说明 分支说明"
    assert [field.text for field in index.fields] == ["$.value: float；字段自身说明"]
    assert schema == before


def test_recursive_stops_and_shared_references_keep_their_own_prose() -> None:
    schema = {
        **ref("Node"),
        "$defs": {
            "Node": obj(
                {
                    "value": node("integer"),
                    "next": {"anyOf": [ref("Node"), node("null")]},
                },
                "链表节点，记录下一环节信息",
            )
        },
    }
    expected = "anyOf(dict；链表节点，记录下一环节信息, None)"
    fields = prepare_field_documents(schema).fields
    assert [f.text for f in fields] == ["$.value: int", "$.next: " + expected]
    assert describe_tool(make_tool(schema)).output_fields == view_text(
        {"value": "int", "next": expected}
    )
    reused = obj(
        {"origin": ref("Node"), "destination": ref("Node")},
        **{"$defs": schema["$defs"]},
    )
    assert {f.path for f in prepare_field_documents(reused).fields} >= {
        "$.origin.value",
        "$.destination.value",
    }


@pytest.mark.parametrize("key", ["data", "result", "record"])
def test_business_keys_remain_in_document_paths(key: str) -> None:
    registry = make_registry(
        make_tool(
            obj({key: obj({"speed": node("number", "车速")})}),
        )
    )
    for discover in (registry.search, registry.grep):
        hit = discover("车速")[0].model_dump()
        assert set(hit) == {"name", "description", "source", "matched_fields"}
        assert hit["matched_fields"] == [f"$.{key}.speed: float；车速"]
    assert registry.describe("read", fields=[f"$.{key}"]).output_fields == (
        f"{{{key}: {{speed: float；车速}}}}"
    )


def test_nullable_reference_keeps_nested_branch_descriptions_searchable() -> None:
    schema = obj(
        {"value": {"anyOf": [ref("Value"), node("null")]}},
        **{
            "$defs": {
                "Value": {
                    "anyOf": [node("string", "标签"), node("number", "重量，单位千克")]
                }
            }
        },
    )
    expected = "anyOf(str；标签, float；重量，单位千克, None)"
    registry = make_registry(make_tool(schema))
    assert registry.describe("read", fields=["$"]).output_fields == view_text(
        {"value": expected}
    )
    for discover in (registry.search, registry.grep):
        assert discover("重量")[0].matched_fields == ["$.value: " + expected]


@pytest.mark.parametrize("keyword", ["anyOf", "oneOf", "allOf"])
@pytest.mark.parametrize("recursive", [False, True])
def test_shared_composite_references_expand_in_each_branch(
    keyword: str, recursive: bool
) -> None:
    value = {
        keyword: [
            {**ref("Value"), "description": "来源一"},
            {**ref("Value"), "description": "来源二"},
        ]
    }
    definitions = {
        "Value": {"anyOf": [node("string", "标签"), node("number", "重量，单位千克")]}
    }
    if recursive:
        definitions["Value"]["anyOf"].append(ref("Value"))
    schema = obj({"value": value}, **{"$defs": definitions})
    before = deepcopy(schema)
    cycle = ", $ref(#/$defs/Value)" if recursive else ""
    label = f"anyOf(str；标签, float；重量，单位千克{cycle})"
    expected = f"{keyword}({label}；来源一, {label}；来源二)"
    registry = make_registry(make_tool(schema))
    view = registry.describe("read", fields=["$"])
    assert view.output_fields == view_text({"value": expected})
    for discover in (registry.search, registry.grep):
        assert discover("重量")[0].matched_fields == ["$.value: " + expected]
    assert schema == before


def test_long_distinct_branch_notes_remain_complete() -> None:
    notes = [f"采样位置{index}：" + "说明内容" * 16 for index in range(128)]
    schema = obj({"temperature": {"anyOf": [node("number", note) for note in notes]}})
    expected = "anyOf(" + ", ".join("float；" + note for note in notes) + ")"
    assert len(expected) > 8192
    assert [f.text for f in prepare_field_documents(schema).fields] == [
        "$.temperature: " + expected
    ]
    assert describe_tool(make_tool(schema)).output_fields == view_text(
        {"temperature": expected}
    )


@pytest.mark.parametrize("wrapper", ["direct", "allOf", "ref"])
@pytest.mark.parametrize(
    "kind,label",
    [("string", "str"), (["string", "null"], "str | None"), ("array", "list[float]")],
)
def test_non_object_properties_are_ignored(wrapper: str, kind: Any, label: str) -> None:
    definition = node(
        kind, "原始值", properties={"ghost": node("integer")}, items=node("number")
    )
    schema = (
        definition
        if wrapper == "direct"
        else {"allOf": [definition]}
        if wrapper == "allOf"
        else {"$defs": {"Value": definition}, **ref("Value")}
    )
    meta = make_tool(schema)
    description = describe_tool(meta)
    assert description.output_type == label
    assert description.output_fields == (
        "[float]" if kind == "array" else label + "；原始值"
    )
    assert [f.text for f in prepare_field_documents(schema).fields] == (
        ["$[*]: float"] if kind == "array" else []
    )


@pytest.mark.parametrize(
    "mapping", [None, False, "additionalProperties", "patternProperties"]
)
def test_dynamic_keys_are_not_enumerated(mapping: Any) -> None:
    schema = obj({"id": node("string", "车辆编号")})
    if mapping is False:
        schema["additionalProperties"] = False
    elif mapping:
        definition = obj({"temperature": node("number", "摄氏度")})
        schema[mapping] = (
            {"^device": definition} if mapping == "patternProperties" else definition
        )
    registry = make_registry(make_tool(schema))
    for discover in (registry.search, registry.grep):
        assert discover("摄氏度") == []
        assert discover("车辆编号")[0].matched_fields == ["$.id: str；车辆编号"]
    view = registry.describe("read", fields=['$["device.a"].temperature'])
    assert view.output_fields == view_text({}) and view.unknown_fields == [
        '$["device.a"].temperature'
    ]
    assert view.available_fields == {"$": ["$.id"]}
