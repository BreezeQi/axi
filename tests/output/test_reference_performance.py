"""验证引用准备、释义和结构选择不会反复遍历共享定义。

按数据规模增长检查遍历成本；仅对共享准备和定位契约断言精确次数，
不使用受机器性能影响的墙钟耗时上限。
"""

import json
from copy import deepcopy
from typing import Any
from unittest.mock import patch

import pytest

from axi.output.documents import SchemaDocument
from axi.output.explanation import SchemaFormatter
from axi.output.schema import SchemaShape
from axi.output.selection import normalize_paths
from axi.output.view import _SelectionResult, select_view
from tests.output.reference_cases import (
    CountingSchema,
    alias_definitions,
    count_pointer_lookups,
    counted_schema,
    shared_schema,
)
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
    terminal_paths,
    view_text,
    make_registry,
)


def test_shared_scalars_are_not_reread_for_different_ancestors() -> None:
    reads = []
    shared = {f"f{i}": CountingSchema(node("number")) for i in range(32)}
    for count in (1, 2):
        names = ("A", "B")[:count]
        schema = obj(
            {name: ref(name) for name in names},
            **{
                "$defs": {
                    "Shared": obj(shared),
                    **{name: obj({"shared": ref("Shared")}) for name in names},
                }
            },
        )
        registry = make_registry(make_tool(schema))
        CountingSchema.reads = 0
        view = registry.describe(
            "read",
            fields=[f"$.{name}.shared" for name in names],
        )
        assert view.output_fields == view_text(
            {name: {"shared": {key: "float" for key in shared}} for name in names}
        )
        reads.append(CountingSchema.reads)
    assert reads[0] > 0 and reads[1] == reads[0]


def test_shared_recursive_views_bound_dependency_work() -> None:
    schema = obj(
        {"a": ref("A"), "b": ref("B")},
        **{"$defs": {side: obj({"shared": ref("Shared")}) for side in ("A", "B")}},
    )
    schema["$defs"]["Shared"] = obj(
        {"value": node("number", "数值"), "next": ref("Shared")}, "链表节点"
    )
    registry = make_registry(make_tool(schema))
    with count_pointer_lookups() as lookup:
        view = registry.describe("read", fields=["$.a.shared", "$.b.shared"])
    assert view.output_fields == view_text(
        {
            side: {"shared": {"value": "float；数值", "next": "dict；链表节点"}}
            for side in ("a", "b")
        }
    )
    assert lookup.call_count <= 16


@pytest.mark.parametrize("keyword", ["allOf", "anyOf", "oneOf"])
@pytest.mark.parametrize("array", [False, True])
def test_shared_structure_traversal_is_bounded(keyword: str, array: bool) -> None:
    reads = []
    for levels in (6, 12):
        schema = json.loads(json.dumps(shared_schema(levels, keyword, array)))
        with count_pointer_lookups() as lookup:
            assert selected_paths(schema) == {
                (("$[*]" if array else "$") + ".value", "float；数值")
            }
            assert [
                f.text
                for f in prepare_field_documents(schema).fields
                if f.name == "value"
            ] == [("$[*]" if array else "$") + ".value: float；数值"]
        reads.append(lookup.call_count)
    assert reads[1] < 6 * reads[0]


@pytest.mark.parametrize("array", [False, True])
@pytest.mark.parametrize("requested", ["value", "missing", "both"])
def test_explicit_shared_structure_traversal_is_bounded(
    array: bool, requested: str
) -> None:
    counts = []
    diagnostic_counts = []
    parent = "$[*]" if array else "$"
    names = ["value", "missing"] if requested == "both" else [requested]
    fields = [parent + "." + name for name in names]
    for levels in (6, 12):
        registry = make_registry(make_tool(shared_schema(levels, "anyOf", array)))
        with (
            patch.object(
                SchemaShape,
                "read",
                side_effect=SchemaShape.read,
            ) as shape,
            patch.object(
                _SelectionResult,
                "add_unknown",
                autospec=True,
                side_effect=_SelectionResult.add_unknown,
            ) as diagnostics,
        ):
            view = registry.describe(
                "read", fields=fields if fields is not None else ["$"]
            )
        expected = {"value": "float；数值"} if requested != "missing" else {}
        assert view.output_fields == view_text([expected] if array else expected)
        assert view.unknown_fields == (
            [parent + ".missing"] if requested != "value" else None
        )
        assert view.available_fields == (
            {parent: [parent + ".value"]} if requested != "value" else None
        )
        counts.append(shape.call_count)
        diagnostic_counts.append(diagnostics.call_count)
    assert counts[1] < 4 * counts[0]
    if requested != "value":
        assert diagnostic_counts[1] < 8 * diagnostic_counts[0]


@pytest.mark.parametrize("array", [False, True])
@pytest.mark.parametrize("leaf", ["value", "missing"])
def test_parent_and_deep_selection_reuse_shared_expansion(
    array: bool, leaf: str
) -> None:
    counts = []
    for levels in (6, 12):
        definitions = {"L0": obj({"value": node("number", "读数")})}
        expected = {"value": "float；读数"}
        for level in range(1, levels + 1):
            definitions[f"L{level}"] = obj(
                {"child": {"anyOf": [ref(f"L{level - 1}"), ref(f"L{level - 1}")]}}
            )
            expected = {"child": expected}
        root = arr(ref(f"L{levels}")) if array else ref(f"L{levels}")
        registry = make_registry(make_tool({**root, "$defs": definitions}))
        parent = "$[*]" if array else "$"
        deep_parent = parent + ".child" * levels
        with patch.object(
            SchemaDocument, "shape", autospec=True, side_effect=SchemaDocument.shape
        ) as select:
            view = registry.describe(
                "read", fields=[parent, deep_parent, deep_parent + "." + leaf]
            )
        assert view.output_fields == view_text([expected] if array else expected)
        assert view.unknown_fields == (
            [deep_parent + ".missing"] if leaf == "missing" else None
        )
        assert view.available_fields == (
            {deep_parent: [deep_parent + ".value"]} if leaf == "missing" else None
        )
        counts.append(select.call_count)
    assert counts[1] < 3 * counts[0]


@pytest.mark.parametrize("size", [2, 32])
def test_cached_selected_positions_keep_their_own_unknown_paths(size: int) -> None:
    schema = node(
        "array",
        prefixItems=[ref("Record") for _ in range(size)],
        items=False,
        **{"$defs": {"Record": obj({"detail": obj({"value": node("number")})})}},
    )
    registry = make_registry(make_tool(schema))
    detail = registry.resolve("read").output_schema["$defs"]["Record"]["properties"][
        "detail"
    ]
    with (
        patch.object(SchemaShape, "read", side_effect=SchemaShape.read) as shape,
        patch.object(
            SchemaDocument,
            "child_segments",
            autospec=True,
            side_effect=SchemaDocument.child_segments,
        ) as directory,
    ):
        view = registry.describe(
            "read", fields=["$[*].detail.value", "$[*].detail.missing"]
        )
    assert sum(call.args[0] is detail for call in shape.call_args_list) == 1
    assert directory.call_count == 1
    assert view.output_fields == (
        "["
        + ", ".join(f"[{index}]: {{detail: {{value: float}}}}" for index in range(size))
        + "]"
    )
    assert view.unknown_fields == [
        f"$[{index}].detail.missing" for index in range(size)
    ]
    assert view.available_fields == {
        f"$[{index}].detail": [f"$[{index}].detail.value"] for index in range(size)
    }


def test_mutually_recursive_variants_reuse_field_coverage() -> None:
    reads = []
    for size in (4, 7):
        references = [ref(f"Node{i}") for i in range(size)]
        schema = {
            "anyOf": references,
            "$defs": {
                f"Node{i}": obj(
                    {
                        "value": node("number", "读数"),
                        "next": {"anyOf": [*references, node("null")]},
                    }
                )
                for i in range(size)
            },
        }
        schema = counted_schema(schema)
        CountingSchema.reads = 0
        fields = prepare_field_documents(schema).fields
        reads.append(CountingSchema.reads)
        assert [field.text for field in fields] == [
            text
            for level in range(size)
            for text in (
                "$" + ".next" * level + ".value: float；读数",
                "$" + ".next" * (level + 1) + ": anyOf(dict, None)",
            )
        ]
        terminals = selected_paths(
            schema, fields=["$" + ".next" * level for level in range(size)]
        )
        assert {
            ("$" + ".next" * level + ".value", "float；读数") for level in range(size)
        } <= terminals
    # 复用声明时不能枚举祖先引用的所有排列。
    assert reads[1] < 64 * reads[0]


@pytest.mark.parametrize("composed", [False, True])
def test_shared_structure_is_reused_only_within_one_view(composed: bool) -> None:
    reads = []
    definition_reads = []
    for count in (1, 100):
        definitions = (
            shared_schema(8, "anyOf", False)["$defs"]
            if composed
            else {"A8": obj({"value": node("number", "数值")})}
        )
        schema = obj(
            {f"reading_{i}": ref("A8") for i in range(count)},
            **{"$defs": definitions},
        )
        schema = counted_schema(schema)
        before = deepcopy(schema)
        expected = {f"reading_{i}": {"value": "float；数值"} for i in range(count)}
        CountingSchema.reads = 0
        with count_pointer_lookups() as lookup:
            assert selected_paths(
                schema, fields=[f"$.{name}" for name in expected]
            ) == terminal_paths(expected)
        reads.append(lookup.call_count)
        definition_reads.append(CountingSchema.reads)
        assert schema == before
    # 新位置可以定位自身的入口引用，但不能重新遍历整份共享定义。
    assert reads[1] <= reads[0] + 2 * len(expected)
    # 工作量可以随输出位置增加，共享定义的读取次数不能随之增加。
    assert definition_reads[1] <= 2 * definition_reads[0] + 10 * len(expected)
    definitions = schema["$defs"]
    for name in ("A0", "B0") if composed else ("A8",):
        definitions[name]["properties"]["value"]["description"] = "新数值"
    assert selected_paths(schema, fields=[f"$.{name}" for name in expected]) == {
        (f"$.{name}.value", "float；新数值") for name in expected
    }


@pytest.mark.parametrize("record", [False, True])
def test_shared_alias_chains_reuse_pointer_lookups(record: bool) -> None:
    value = node("number", "温度")
    definitions = alias_definitions(20, obj({"value": value}) if record else value)
    reads = []
    for count in (1, 100):
        schema = obj(
            {f"reading_{index}": ref("Alias19") for index in range(count)},
            **{"$defs": definitions},
        )
        expected = {
            name: {"value": "float；温度"} if record else "float；温度"
            for name in schema["properties"]
        }
        with count_pointer_lookups() as lookup:
            assert show(
                schema, fields=["$." + name for name in schema["properties"]]
            ) == view_text(expected)
        reads.append(lookup.call_count)
    assert 0 < reads[1] <= reads[0] <= len(definitions)
    value["description"] = "新温度"
    assert show(
        schema, fields=["$." + name for name in schema["properties"]]
    ) == view_text(
        {
            name: {"value": "float；新温度"} if record else "float；新温度"
            for name in expected
        }
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_alias_entry_points_reuse_reference_traversal(reverse: bool) -> None:
    reads = []
    for count in (16, 60):
        properties = {"value": node("number", "读数")}
        definitions = {
            name: CountingSchema(definition)
            for name, definition in alias_definitions(count, obj(properties)).items()
        }
        indices = list(range(count))
        if reverse:
            indices.reverse()
        schema = obj(
            {f"reading{i}": ref(f"Alias{i}") for i in indices},
            **{"$defs": definitions},
        )
        document = SchemaDocument(schema)
        CountingSchema.reads = 0
        for index, definition in zip(indices, schema["properties"].values()):
            shape = document.shape(definition)
            assert shape.properties is properties
            assert shape.references == {
                "#/$defs/Value",
                *(f"#/$defs/Alias{i}" for i in range(index + 1)),
            }
        reads.append(CountingSchema.reads)
    # 统计定义读取次数，避免引用定位缓存掩盖实际的重复遍历。
    assert 0 < reads[0] and reads[1] < 5 * reads[0]


@pytest.mark.parametrize("keyword", ["anyOf", "allOf", "oneOf"])
def test_shared_reference_with_siblings_reuses_its_structure(keyword: str) -> None:
    wrapper = {
        **ref("Record"),
        "properties": CountingSchema({"extra": node("string")}),
    }
    counts = []
    for copies in (1, 100):
        schema = {
            "$defs": {"Record": obj(CountingSchema({"value": node("number")}))},
            keyword: [wrapper] * copies,
        }
        branch = composition("allOf", {"value": "float"}, {"extra": "str"})
        expected = composition(
            keyword, *([branch] * (copies if keyword == "oneOf" else 1))
        )
        if copies > 1 and keyword != "oneOf":
            expected = branch
        CountingSchema.reads = 0
        assert show(schema) == view_text(expected)
        # oneOf 保留每个分支，标量检查可以随输出量增加。
        # 共享属性声明的读取次数不能随分支数量增加。
        counts.append(CountingSchema.reads)
    assert counts[1] <= 3 * counts[0]


@pytest.mark.parametrize("keyword", ["anyOf", "allOf"])
@pytest.mark.parametrize("root_sibling", [False, True])
def test_unequal_shared_branches_bound_traversal_and_rendering(
    keyword: str, root_sibling: bool
) -> None:
    reads = []
    sizes = []
    for levels in (12, 18):
        definitions = {"A0": obj({"value": node("number", "读数")}), "A1": ref("A0")}
        for level in range(2, levels + 1):
            definitions[f"A{level}"] = {
                keyword: [ref(f"A{level - 1}"), ref(f"A{level - 2}")]
            }
        schema = obj({"reading": ref(f"A{levels}")}, **{"$defs": definitions})
        if root_sibling:
            schema.update(ref(f"A{levels}"))
        schema = counted_schema(schema)
        before = deepcopy(schema)
        expected = {("$.reading.value", "float；读数")}
        if root_sibling:
            expected.add(("$.value", "float；读数"))
        CountingSchema.reads = 0
        with count_pointer_lookups() as lookup:
            assert selected_paths(schema, fields=["$", "$.reading"]) == expected
        # 引用定位缓存不能掩盖定义读取次数的指数增长。
        reads.append(lookup.call_count + CountingSchema.reads)
        CountingSchema.reads = 0
        result = describe_tool(make_tool(schema), fields=["$", "$.reading"])
        assert CountingSchema.reads <= 64 * levels**2
        assert result.output_fields.count("float；读数") == len(expected)
        sizes.append(len(result.model_dump_json(exclude_none=True)))
        assert schema == before
    assert reads[1] < 3 * reads[0]
    assert len(set(sizes)) == 1


def test_wide_shared_dependencies_do_not_repeat_reference_traversal() -> None:
    reads = []
    for width in (1, 100):
        definitions = shared_schema(8, "anyOf", False)["$defs"]
        definitions["Shared"] = {
            "anyOf": [obj({f"f{i}": ref("A8") for i in range(width)}), node("null")]
        }
        for name in ("A", "B"):
            definitions[name] = obj({"shared": ref("Shared")})
        schema = obj({name: ref(name) for name in ("A", "B")}, **{"$defs": definitions})
        with count_pointer_lookups() as lookup:
            shared = {f"f{i}": {"value": "float；数值"} for i in range(width)}
            terminals = selected_paths(
                schema,
                fields=[
                    f"$.{name}.shared.f{i}" for name in ("A", "B") for i in range(width)
                ],
            )
            expected = terminal_paths({name: {"shared": shared} for name in ("A", "B")})
            assert expected <= terminals
        reads.append(lookup.call_count)
    assert reads[1] <= 2 * reads[0]


@pytest.mark.parametrize("keyword", ["anyOf", "oneOf", "allOf"])
@pytest.mark.parametrize("referenced", [False, True])
def test_nested_compositions_do_not_rescan_descendants(
    keyword: str, referenced: bool
) -> None:
    reads = []
    sizes = []
    for levels in (8, 16):
        schema = CountingSchema(node("number", "数值"))
        definitions = {}
        expected: Any = "float；数值"
        for level in range(levels):
            schema = CountingSchema(
                {keyword: [CountingSchema(obj({"child": schema})), node("null")]}
            )
            if referenced:
                definitions[f"D{level}"] = schema
                schema = CountingSchema(ref(f"D{level}"))
            expected = composition(keyword, {"child": expected}, "None")
        schema["$defs"] = definitions
        CountingSchema.reads = 0
        actual = show(
            schema,
            fields=["$"] + ["$" + ".child" * index for index in range(1, levels)],
        )
        assert actual == expected
        sizes.append(len(actual))
        reads.append(CountingSchema.reads)
    assert reads[1] < 3 * reads[0]
    assert sizes[1] < 3 * sizes[0]


@pytest.mark.parametrize("keyword", ["anyOf", "allOf"])
@pytest.mark.parametrize("array", [False, True])
def test_shared_references_to_recursive_records_do_not_repeat_traversal(
    keyword: str, array: bool
) -> None:
    reads = []
    for levels in (6, 12):
        definitions = shared_schema(levels, keyword, False)["$defs"]
        definitions["Node"] = obj(
            {"value": node("number", "温度"), "next": ref("Node")}, "节点"
        )
        for side in ("A", "B"):
            definitions[side + "0"] = arr(ref("Node")) if array else ref("Node")
        schema = obj({"record": ref(f"A{levels}")}, **{"$defs": definitions})
        with count_pointer_lookups() as lookup:
            fields = prepare_field_documents(schema).fields
        path = "$.record[*]" if array else "$.record"
        expected = [
            f"{path}: dict；节点",
            f"{path}.value: float；温度",
            f"{path}.next: dict；节点",
        ]
        assert [field.text for field in fields] == (
            ["$.record: list[dict]", *expected] if array else expected
        )
        reads.append(lookup.call_count)
    assert reads[1] < 3 * reads[0]


@pytest.mark.parametrize("keyword", ["anyOf", "allOf"])
def test_long_shared_reference_chains_keep_work_bounded(keyword: str) -> None:
    reads = []
    for levels in (4, 8):
        definitions = alias_definitions(60, node("number", "温度"))
        definitions["D0"] = ref("Alias59")
        for depth in range(1, levels + 1):
            definitions[f"D{depth}"] = {
                keyword: [ref(f"D{depth - 1}"), ref(f"D{depth - 1}")]
            }
        schema = obj({"temperature": ref(f"D{levels}")}, **{"$defs": definitions})
        with count_pointer_lookups() as lookup:
            assert [field.text for field in prepare_field_documents(schema).fields] == [
                "$.temperature: float；温度"
            ]
            formatter = SchemaFormatter(schema)
            assert (
                formatter.format(schema["properties"]["temperature"]) == "float；温度"
            )
            assert formatter.format(ref("Alias0")) == "float；温度"
        reads.append(lookup.call_count)
    assert reads[1] < 3 * reads[0]


@pytest.mark.parametrize("keyword", ["anyOf", "allOf"])
@pytest.mark.parametrize("recursive", [False, True])
def test_repeated_compositions_keep_text_and_work_bounded(
    keyword: str, recursive: bool
) -> None:
    reads = []
    label = (
        "anyOf(float；温度，单位摄氏度, "
        + ("list[$ref(#/$defs/Value)]" if recursive else "str；离线等状态文本")
        + ")"
    )
    expected = label + "；设备温度"
    for levels in (8, 12, 14):
        definitions = shared_schema(levels, keyword, False)["$defs"]
        definitions["Value"] = {
            "anyOf": [
                node("number", "温度，单位摄氏度"),
                arr(ref("Value")) if recursive else node("string", "离线等状态文本"),
            ]
        }
        for side in ("A", "B"):
            definitions[side + "0"] = ref("Value")
        schema = obj(
            {"temperature": {**ref(f"A{levels}"), "description": "设备温度"}},
            **{"$defs": definitions},
        )
        with count_pointer_lookups() as lookup:
            assert [f.text for f in prepare_field_documents(schema).fields] == [
                "$.temperature: " + expected
            ] + (["$.temperature[*]: " + label] if recursive else [])
        reads.append(lookup.call_count)
        assert show(schema) == view_text({"temperature": expected})
    assert reads[-1] < 3 * reads[0]


@pytest.mark.parametrize("entry", ["search", "describe"])
def test_references_are_reused_across_fields_only_within_one_preparation(
    entry: str,
) -> None:
    def prepare(schema: dict) -> str:
        if entry == "search":
            labels = [
                field.text.split(": ", 1)[1]
                for field in prepare_field_documents(schema).fields
            ]
            return view_text(dict(zip(schema["properties"], labels, strict=True)))
        return describe_tool(make_tool(schema)).output_fields

    reads = []
    for count in (1, 100):
        definitions = shared_schema(8, "anyOf", False)["$defs"]
        for side in ("A", "B"):
            definitions[side + "0"] = {
                "anyOf": [node("number", "温度"), node("string", "状态")]
            }
        schema = obj(
            {
                f"reading_{index}": {**ref("A8"), "description": f"设备{index}读数"}
                for index in range(count)
            },
            **{"$defs": definitions},
        )
        expected = [
            f"anyOf(float；温度, str；状态)；设备{index}读数" for index in range(count)
        ]
        before = deepcopy(schema)
        with count_pointer_lookups() as lookup:
            assert prepare(schema) == view_text(
                dict(zip(schema["properties"], expected, strict=True))
            )
        reads.append(lookup.call_count)
        assert schema == before
    assert 0 < reads[1] <= reads[0]
    for side in ("A", "B"):
        definitions[side + "0"]["anyOf"][0]["description"] = "新温度"
    assert prepare(schema) == view_text(
        dict(
            zip(
                schema["properties"],
                [label.replace("温度", "新温度") for label in expected],
                strict=True,
            )
        )
    )


def test_describe_reuses_formatter_for_reference_chains() -> None:
    schema = obj(
        {"reading": ref("Alias")},
        **{
            "$defs": {"Alias": ref("Value"), "Value": node("number", "温度")},
        },
    )
    with count_pointer_lookups() as lookup:
        view = describe_tool(make_tool(schema), fields=["$.reading", "$.missing"])
    assert view.output_fields == view_text({"reading": "float；温度"})
    assert view.available_fields == {"$": ["$.reading"]}
    assert view.output_type == "dict" and view.unknown_fields == ["$.missing"]
    assert 0 < lookup.call_count <= len(schema["$defs"])


@pytest.mark.parametrize("aliased", [False, True])
@pytest.mark.parametrize("missing_count", [1, 32])
def test_unknown_directory_reads_do_not_grow_with_sibling_count(
    aliased: bool,
    missing_count: int,
) -> None:
    counts = []
    for width in (16, 256):
        group = obj({f"field{i}": ref("Value") for i in range(width)})
        schema = obj(
            {"group": ref("Group") if aliased else group},
            **{"$defs": {"Group": group, "Value": node("number")}},
        )
        paths = [f"$.group.missing{i}" for i in range(missing_count)]
        with (
            patch.object(
                SchemaShape,
                "read",
                side_effect=SchemaShape.read,
            ) as reads,
            patch.object(
                SchemaDocument,
                "child_segments",
                autospec=True,
                side_effect=SchemaDocument.child_segments,
            ) as directory,
        ):
            selected = select_view(SchemaDocument(schema), normalize_paths(paths))
        assert selected.unknown == sorted(paths)
        assert selected.available == {
            "$.group": [f"$.group.field{i}" for i in range(width)]
        }
        assert directory.call_count == 1
        counts.append(reads.call_count)
    assert counts[0] == counts[1]


@pytest.mark.parametrize("arrays", [False, True])
def test_unknown_directory_does_not_read_descendant_references(arrays: bool) -> None:
    def container(definition: dict[str, str]) -> dict[str, Any]:
        return arr(definition) if arrays else obj({"child": definition})

    reads = []
    for levels in (8, 32):
        definitions = {
            "Shared": container(ref("H0")),
            "AliasA": ref("Shared"),
            "AliasB": ref("Shared"),
            **{f"H{i}": container(ref(f"H{i + 1}")) for i in range(levels)},
            f"H{levels}": node("integer"),
        }
        schema = {"$defs": definitions, "anyOf": [ref("AliasA"), ref("AliasB")]}
        before = deepcopy(schema)
        with count_pointer_lookups() as lookup:
            selected = select_view(
                SchemaDocument(schema), normalize_paths(["$.missing"])
            )
        assert selected.unknown == ["$.missing"]
        assert selected.available == {"$": ["$[*]" if arrays else "$.child"]}
        assert schema == before
        reads.append(lookup.call_count)
    assert reads[0] == reads[1]
