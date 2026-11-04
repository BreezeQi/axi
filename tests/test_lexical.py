"""验证关键词召回、字面路径及各类全文检索文本的独立边界。"""

import re
from typing import Annotated
from unittest.mock import Mock

import bm25s
import pytest
from pydantic import BaseModel, Field

from axi.output.registration import normalize_schema
from axi.registry import Registry
from axi.search.bm25 import BM25Search, tokenize_query
from tests.support import prepare_field_documents, make_registry
from axi.search.regex import RegexSearch, compile_pattern
from axi.search.tokenize import preprocess
from tests.support import arr, build_search, make_tool, node, obj
from tests.support import view_text


@pytest.mark.parametrize(
    "query,text",
    [
        ("天气", "get_weather 获取指定城市的天气信息"),
        ("read", "read_file Read contents of a file"),
        ("calculator", "calculator A simple math tool"),
        ("图", "image_viewer 看图说话工具"),
    ],
)
def test_bm25_recall(query: str, text: str) -> None:
    search = BM25Search()
    search.build_documents([text, "echo Repeat input", "news 最新新闻"])
    tokens = tokenize_query(query)
    assert search.search_tokens(tokens)[0][0] == 0
    assert search.search_tokens(tokenize_query("zzzzxyz")) == []
    assert search.search_tokens(tokenize_query("")) == []
    search.build_documents([])
    assert search.search_tokens(tokens) == []


@pytest.mark.parametrize("text", ["", "A", "++", "..", "##", "&&"])
def test_bm25_tokenless_documents_keep_positions_and_clear_old_index(
    text: str,
) -> None:
    search = BM25Search()
    search.build_documents([text, "speed"])
    tokens = tokenize_query("speed")
    assert [index for index, _ in search.search_tokens(tokens)] == [1]
    search.build_documents([text])
    assert search.search_tokens(tokens) == []


def test_bm25_scores_and_tied_cutoffs_match_the_original_library_retrieval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    documents = ["shared " * (i % 5) + "record " * (5 - i % 5) for i in range(1000)]
    reference = bm25s.BM25()
    reference.index(
        bm25s.tokenize([preprocess(document) for document in documents], stopwords=None)
    )
    prepare = Mock(wraps=preprocess)
    monkeypatch.setattr("axi.search.bm25.preprocess", prepare)
    search = BM25Search()
    search.build_documents(documents)
    assert prepare.call_count == len(set(documents))
    tokens = tokenize_query("shared")
    for limit in (-1, 0, 1, 5, 1000, 1001):
        expected = []
        if limit > 0:
            indices, scores = reference.retrieve(
                bm25s.tokenize("shared"),
                k=min(limit, len(documents)),
                show_progress=False,
            )
            expected = [
                (int(index), float(score))
                for index, score in zip(indices[0], scores[0])
                if score > 0
            ]
        assert search.search_tokens(tokens, top_k=limit) == expected
        assert search.search_tokens(tokens, top_k=limit) == expected
    prepare.reset_mock()
    search.build_documents(documents[:2])
    assert prepare.call_count == 2
    assert [index for index, _ in search.search_tokens(tokens)] == [1]


def test_root_scalar_prose_and_root_array_elements_are_discoverable() -> None:
    for schema, expected in [
        (node("number", "含税金额，单位分"), []),
        (arr(node("number", "含税金额，单位分")), ["$[*]: float；含税金额，单位分"]),
    ]:
        registry = make_registry(make_tool(description="读取记录", schema=schema))
        for discover in (registry.search, registry.grep):
            hit = discover("含税金额")[0]
            assert hit.name == "read" and hit.matched_fields == expected


@pytest.mark.parametrize("kind", ["array", "tuple"])
@pytest.mark.parametrize("nullable", [False, True])
def test_element_prose_is_indexed_at_its_own_position(
    kind: str, nullable: bool
) -> None:
    values_type = list[Annotated[float, Field(description="温度，单位摄氏度", ge=-40)]]
    elements = [("摄氏度", "$.values[*]: float；温度，单位摄氏度")]
    label = "list[float]"
    if kind == "tuple":
        values_type = tuple[
            Annotated[float, Field(description="纬度，单位度")],
            Annotated[float, Field(description="经度，单位度")],
        ]
        elements = [
            ("纬度", "$.values[0]: float；纬度，单位度"),
            ("经度", "$.values[1]: float；经度，单位度"),
        ]

    class Samples(BaseModel):
        values: values_type | None if nullable else values_type

    meta = make_tool(description="读取记录", schema=normalize_schema(Samples))
    label = f"anyOf({label}, None)" if nullable else label
    registry = make_registry(meta)
    assert registry.describe("read", fields=["$"]).output_fields == view_text(
        {"values": label}
    )
    assert registry.search("values")[0].matched_fields == ["$.values: " + label]
    for query, expected in elements:
        assert registry.search(query)[0].matched_fields == [expected]
        assert registry.grep(query)[0].matched_fields == [expected]


@pytest.mark.parametrize("method", ["search", "grep"])
def test_tool_parent_and_leaf_documents_match_independently(method: str) -> None:
    class WorkTime(BaseModel):
        duration: float = Field(description="时长，小时")

    class Summary(BaseModel):
        """车辆汇总统计"""

        forwardMileage: float = Field(description="前进里程，单位米", ge=0)
        workTime: WorkTime = Field(description="作业时间明细")

    class Report(BaseModel):
        """rootcontext rootcontext"""

        summaryStatistic: Summary
        unexplained: bool

    registry = make_registry(
        make_tool(normalize_schema(Report), description="capabilitylookup")
    )
    discover = getattr(registry, method)
    parent = "$.summaryStatistic: dict；车辆汇总统计"
    mileage = "$.summaryStatistic.forwardMileage: float；前进里程，单位米"
    work = "$.summaryStatistic.workTime: dict；作业时间明细"
    duration = "$.summaryStatistic.workTime.duration: float；时长，小时"
    for query, expected in [
        ("capabilitylookup", []),
        ("rootcontext", []),
        ("汇总", [parent]),
        ("前进里程", [mileage]),
        ("明细", [work]),
        ("时长", [duration]),
        ("bool", ["$.unexplained: bool"]),
        ("float", [mileage, duration]),
        ("汇总 前进" if method == "search" else "汇总|前进", [parent, mileage]),
    ]:
        hits = discover(query)
        assert len(hits) == 1 and hits[0].name == "read", query
        assert set(hits[0].matched_fields) == set(expected), query
    if method == "grep":
        assert discover("summaryStatistic")[0].matched_fields == [
            parent,
            mileage,
            work,
            duration,
        ]


def test_grep_preserves_independent_tool_and_field_text_boundaries() -> None:
    registry = make_registry(
        make_tool(
            obj({"speed": node("number", "速度")}, "schema root"),
            "query_vehicle",
            server="fleet",
            description="Read vehicle report",
        )
    )
    for pattern in ("^fleet/query_vehicle$", "^read vehicle report$", "^schema root$"):
        hit = registry.grep(pattern)[0]
        assert hit.name == "fleet/query_vehicle" and hit.matched_fields == []
    assert registry.grep("^fleet/query_vehicle Read vehicle report$") == []
    assert registry.grep("Read vehicle report schema root") == []
    assert registry.grep(r"^\$\.speed: float；速度$")[0].matched_fields == [
        "$.speed: float；速度"
    ]


@pytest.mark.parametrize("limit", [-1, 0, 1, 10])
def test_grep_compiles_once_and_limits_the_whole_catalog(
    monkeypatch: pytest.MonkeyPatch, limit: int
) -> None:
    registry = Registry()
    for source in ("native", "mcp"):
        registry.register(make_tool(name="read", server=source, source=source))
    compile = Mock(wraps=compile_pattern)
    monkeypatch.setattr("axi.search.regex.compile_pattern", compile)
    assert [hit.name for hit in registry.grep("READ$", top_k=limit)] == (
        ["native/read", "mcp/read"][: max(0, limit)]
    )
    compile.assert_called_once_with("READ$")


@pytest.mark.parametrize(
    "location,checks", [("tool", 0), (0, 1), (99, 100), (None, 100)]
)
def test_grep_without_summaries_short_circuits_field_matches(
    monkeypatch: pytest.MonkeyPatch, location: str | int | None, checks: int
) -> None:
    schema = obj(
        {
            f"value_{index}": node("number", "needle" if index == location else "other")
            for index in range(100)
        }
    )
    search = RegexSearch(max_fields=0)
    build_search(
        search,
        [
            make_tool(
                schema, description="needle" if location == "tool" else "capability"
            )
        ],
    )
    compiled = Mock(wraps=re.compile("needle"))
    monkeypatch.setattr("axi.search.regex.compile_pattern", lambda _: compiled)
    hits = search.search("needle")
    assert [(hit.name, hit.matched_fields) for hit in hits] == (
        [] if location is None else [("read", [])]
    )
    assert (
        sum(call.args[0].startswith("$.") for call in compiled.search.call_args_list)
        == checks
    )


@pytest.mark.parametrize("cap", [0, 1, 8])
def test_grep_stops_at_field_limit_without_reordering_shared_documents(
    monkeypatch: pytest.MonkeyPatch, cap: int
) -> None:
    meta = make_tool(
        obj(
            {
                "nested": obj({f"f{i}": node("number") for i in range(100)}),
                "last": node("string"),
            }
        )
    )
    index = prepare_field_documents(meta.output_schema)
    declared = list(index.fields)
    search = RegexSearch(max_fields=cap)
    search.build([meta], field_documents=[index])
    compiled = Mock(wraps=re.compile(r"^\$\."))
    monkeypatch.setattr("axi.search.regex.compile_pattern", lambda _: compiled)

    hits = search.search(r"^\$\.")

    assert hits[0].matched_fields == [
        field.text for field in sorted(declared, key=lambda field: field.depth)[:cap]
    ]
    assert index.fields == declared
    assert sum(
        call.args[0].startswith("$.") for call in compiled.search.call_args_list
    ) == (cap or 1)


@pytest.mark.parametrize("cap", [0, 1, 2, 3, 8])
def test_grep_sorts_all_actual_matches_by_depth_before_capping(cap: int) -> None:
    schema = obj(
        {
            "summaryStatistic": obj({"driver_info": obj({"name": node("string")})}),
            "driver_info": obj({"name": node("string")}),
            "unused": obj({"name": node("string")}),
        }
    )
    search = RegexSearch(max_fields=cap)
    build_search(search, [make_tool(schema, "Z"), make_tool(schema, "A")])
    hits = search.search("driver_info", top_k=1)
    assert [hit.name for hit in hits] == ["Z"]
    assert (
        hits[0].matched_fields
        == [
            "$.driver_info: dict",
            "$.summaryStatistic.driver_info: dict",
            "$.driver_info.name: str",
            "$.summaryStatistic.driver_info.name: str",
        ][:cap]
    )
    assert [hit.name for hit in search.search("driver_info")] == ["Z", "A"]


def test_grep_depth_counts_properties_and_elements_not_path_punctuation() -> None:
    schema = obj(
        {
            "rows": arr(obj({"a.b": node("number", "目标")})),
            "a.b.c.d": node("number", "目标"),
            "long_property_name": node("number", "目标"),
        }
    )
    search = RegexSearch(max_fields=2)
    build_search(search, [make_tool(schema)])
    assert search.search("目标")[0].matched_fields == [
        '$["a.b.c.d"]: float；目标',
        "$.long_property_name: float；目标",
    ]
