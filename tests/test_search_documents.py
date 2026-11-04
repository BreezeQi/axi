"""验证双路独立检索、交错推荐、字段摘要及离线向量检索。"""

import re
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from axi.config import SearchConfig
from axi.registry import Registry, ToolResolveError
from axi.search.cache import EmbeddingCache
from axi.search.corpus import build_corpus
from axi.search.hybrid import HybridSearch
from tests.support import prepare_field_documents, make_registry
from tests.support import arr, build_search, make_tool, node, obj


def routes(
    search: HybridSearch,
    lexical: list[tuple[int, str]],
    semantic: list[tuple[int, str]] | None = None,
) -> None:
    """为两路提供可控排名，隔离评分与结果整理的测试职责。"""
    if semantic is not None:
        search._embedding_provider = Mock()
        search._embedding_provider.embed_query.return_value = [1.0]
    tool_ids = {(i, ""): i for i in range(len(search._tools))}
    field_ids = {
        (doc.tool, doc.field.path.removeprefix("$.") if doc.field else ""): i
        for i, doc in enumerate(search._field_documents)
    }
    for engine, ids in (
        (search._tool_search, tool_ids),
        (search._field_search, field_ids),
    ):

        def scores(hits: list[tuple[int, str]]) -> list[tuple[int, float]]:
            return [
                (ids[hit], float(len(hits) - i))
                for i, hit in enumerate(hits)
                if hit in ids
            ]

        ranked = scores(lexical)
        engine._bm25.search_tokens = Mock(
            side_effect=lambda tokens, top_k, ranked=ranked: ranked[:top_k]
        )
        engine._embedding = None
        if semantic is not None:
            ranked = scores(semantic)
            engine._embedding = Mock()
            engine._embedding.search.side_effect = (
                lambda query_vector, top_k, ranked=ranked: ranked[:top_k]
            )


@pytest.mark.parametrize("cap", [0, 1, 2, 8])
@pytest.mark.parametrize("semantic", [False, True])
def test_interleaving_keeps_all_tools_field_summaries(cap: int, semantic: bool) -> None:
    search = HybridSearch(max_fields=cap)
    build_search(
        search,
        [
            make_tool(
                obj({key: node("number") for key in keys}),
                name,
                description="query records",
            )
            for name, keys in [
                ("X", ["x", "y", "z"]),
                ("A", ["x", "y"]),
                ("B", ["x"]),
                ("C", ["x"]),
            ]
        ],
    )
    ranked = [
        (0, ""),
        (1, ""),
        (2, ""),
        (1, "x"),
        (2, "x"),
        (1, "y"),
        (3, "x"),
        (0, "x"),
        (0, "y"),
        (0, "z"),
    ]
    routes(search, ranked, ranked if semantic else None)
    hits = search.search("general readings", top_k=3)
    assert [hit.name for hit in hits] == ["X", "A", "B"]
    assert [hit.matched_fields for hit in hits] == [
        ["$.x: float", "$.y: float", "$.z: float"][:cap],
        ["$.x: float", "$.y: float"][:cap],
        ["$.x: float"][:cap],
    ]
    for engine in (search._tool_search, search._field_search):
        engine._bm25.search_tokens.assert_called_once()
        if semantic:
            engine._embedding.search.assert_called_once()
    if semantic:
        search._embedding_provider.embed_query.assert_called_once_with(
            "general readings"
        )


@pytest.mark.parametrize(
    "tool_order,field_order,expected",
    [
        ([], [], []),
        ([0, 1], [], [0, 1]),
        ([], [1, 0], [1, 0]),
        ([0, 1], [0, 1], [0, 1]),
        ([0, 1, 2], [1, 2, 3], [0, 1, 2, 3]),
        ([2], [0, 1], [2, 0, 1]),
    ],
)
@pytest.mark.parametrize("top_k", [-1, 0, 1, 2, 3, 8])
def test_interleave_empty_overlap_and_k_boundaries(
    tool_order: list[int], field_order: list[int], expected: list[int], top_k: int
) -> None:
    search = HybridSearch()
    build_search(
        search,
        [make_tool(obj({"speed": node("number")}), f"tool{i}") for i in range(4)],
    )
    routes(search, [(i, "") for i in tool_order] + [(i, "speed") for i in field_order])
    assert [hit.name for hit in search.search("readings", top_k=top_k)] == [
        f"tool{i}" for i in expected[: max(0, top_k)]
    ]


@pytest.mark.parametrize("count", [1, 100])
@pytest.mark.parametrize("semantic", [False, True])
def test_field_counts_do_not_accumulate_tool_scores(count: int, semantic: bool) -> None:
    search = HybridSearch()
    build_search(
        search,
        [
            make_tool(obj({f"x{i}": node("number") for i in range(count)}), "A"),
            make_tool(obj({"value": node("number")}), "B"),
        ],
    )
    lexical = [(0, f"x{i}") for i in range(count)] + [(1, "value")]
    routes(search, lexical, [(1, "value")] if semantic else None)
    assert [hit.name for hit in search.search("general readings", top_k=2)] == (
        ["B", "A"] if semantic else ["A", "B"]
    )


def test_fields_are_fused_before_per_tool_capping() -> None:
    names = (
        [f"lexical{i}" for i in range(8)]
        + ["shared"]
        + [f"semantic{i}" for i in range(8)]
    )
    search = HybridSearch()
    build_search(
        search, [make_tool(obj({name: node("number") for name in names}), "winner")]
    )
    routes(
        search,
        [(0, name) for name in names[:9]],
        [(0, f"semantic{i}") for i in range(8)] + [(0, "shared")],
    )
    result = search.search("general measurements", top_k=1)[0]
    assert (
        result.matched_fields[0] == "$.shared: float"
        and len(result.matched_fields) == 8
    )


@pytest.mark.parametrize("cap", [0, 1, 2])
@pytest.mark.parametrize("keys", [("x", "y"), ("y", "x")])
@pytest.mark.parametrize(
    "weights,expected",
    [
        ((0.3, 0.7), ["y", "x"]),
        ((0.7, 0.3), ["x", "y"]),
        ((0.5, 0.5), ["x", "y"]),
    ],
)
def test_field_summaries_preserve_relevance_within_one_tool(
    cap: int, keys: tuple[str, str], weights: tuple[float, float], expected: list[str]
) -> None:
    search = HybridSearch(
        max_fields=cap, weight_bm25=weights[0], weight_embedding=weights[1]
    )
    build_search(search, [make_tool(obj({key: node("number") for key in keys}))])
    routes(search, [(0, "x"), (0, "y")], [(0, "y"), (0, "x")])
    result = search.search("general measurements", top_k=1)[0]
    assert result.name == "read"
    assert result.matched_fields == [f"$.{key}: float" for key in expected[:cap]]
    search._field_search._bm25.search_tokens.assert_called_once()
    search._field_search._embedding.search.assert_called_once()


@pytest.mark.parametrize("count", [0, 100])
@pytest.mark.parametrize("top_k", [1, 2])
def test_field_summaries_follow_the_same_global_text_ranking(
    count: int, top_k: int
) -> None:
    search = HybridSearch(weight_bm25=0.7, weight_embedding=0.3, max_fields=1)
    build_search(
        search,
        [
            make_tool(obj({key: node("number") for key in ("x", "y")}), "X"),
            make_tool(obj({f"f{i}": node("number") for i in range(count)}), "other"),
        ],
    )
    routes(
        search,
        [(0, "x"), (0, "y")],
        [(0, "y")] + [(1, f"f{i}") for i in range(count)] + [(0, "x")],
    )
    hits = search.search("general measurements", top_k=top_k)
    assert hits[0].name == "X"
    assert hits[0].matched_fields == [f"$.{'y' if count else 'x'}: float"]


@pytest.mark.parametrize(
    "ranked,expected",
    [
        (["summary"], ["$.summary: dict；汇总"]),
        (["summary.distance"], ["$.summary.distance: float；里程"]),
        (
            ["summary.distance", "summary"],
            ["$.summary.distance: float；里程", "$.summary: dict；汇总"],
        ),
    ],
)
def test_field_summaries_only_include_ranked_nodes_in_relevance_order(
    ranked: list[str], expected: list[str]
) -> None:
    search = HybridSearch()
    build_search(
        search,
        [
            make_tool(
                obj({"summary": obj({"distance": node("number", "里程")}, "汇总")})
            )
        ],
    )
    routes(search, [(0, path) for path in ranked])
    assert search.search("general measurements")[0].matched_fields == expected


@pytest.mark.parametrize("method", ["search", "grep"])
@pytest.mark.parametrize("cap", [0, 1, 8, 12])
def test_config_caps_individual_fields_without_parent_folding(
    method: str, cap: int
) -> None:
    registry = Registry.from_search_config(
        SearchConfig.model_validate({"maxFields": cap})
    )
    registry.register(
        make_tool(
            obj(
                {
                    "summary": obj(
                        {f"f{i}": node("number", "measurement") for i in range(12)},
                        "汇总",
                    )
                }
            )
        )
    )
    result = getattr(registry, method)("measurement")[0]
    assert result.name == "read"
    assert len(result.matched_fields) == cap
    assert set(result.matched_fields) <= {
        f"$.summary.f{i}: float；measurement" for i in range(12)
    }


def test_field_limit_validation() -> None:
    assert SearchConfig().max_fields == 8
    with pytest.raises(ValidationError):
        SearchConfig(maxFields=-1)


def test_corpus_text_and_literal_index_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    meta = make_tool(
        obj(
            {"summary": obj({"temperature": node(["number", "null"], "温度")}, "汇总")}
        ),
        description="读取记录",
    )
    index = prepare_field_documents(meta.output_schema)
    corpus = build_corpus([meta], [index])
    assert corpus.tool_texts == ["read 读取记录"]
    assert [
        doc.field.path if doc.field is not None else ""
        for doc in corpus.output_documents
    ] == [
        "$.summary",
        "$.summary.temperature",
    ]
    assert corpus.output_texts == [
        "$.summary: dict；汇总",
        "$.summary.temperature: float | None；温度",
    ]
    search = HybridSearch()
    search.build([meta], field_documents=[index])

    assert search.search("TEMPERATURE")[0].matched_fields == [
        "$.summary.temperature: float | None；温度"
    ]
    assert search.search("unknown field") == []
    build_search(search, [make_tool(obj({"new": node("number"), "": node("string")}))])
    assert (
        search.search("temperature") == search.search("") == search.search("   ") == []
    )
    assert search.search("NEW")[0].matched_fields == ["$.new: float"]


@pytest.mark.parametrize("semantic", [False, True])
@pytest.mark.parametrize("cap", [0, 1, 8])
@pytest.mark.parametrize(
    "query", ["TEMPERATURE", "$.left.temperature", "$['left']['temperature']"]
)
def test_exact_matches_keep_each_tool_and_distinct_declarations(
    semantic: bool, cap: int, query: str
) -> None:
    schema = obj(
        {
            "left": {
                "anyOf": [
                    obj({"temperature": node("number", "摄氏度")}),
                    obj({"temperature": node("string", "设备状态")}),
                ]
            },
            "right": obj({"temperature": node("number", "华氏度")}),
            "noise": node("string", "温度概览"),
        }
    )
    search = HybridSearch(max_fields=cap)
    build_search(
        search,
        [
            make_tool(schema, "first"),
            make_tool(schema, "second"),
            make_tool(name="guide"),
        ],
    )
    ranked = [(2, ""), (0, "noise"), (1, "noise")]
    routes(search, ranked, ranked if semantic else None)
    hits = search.search(query, top_k=3)
    assert [hit.name for hit in hits] == ["guide", "first", "second"]
    expected = [
        "$.left.temperature: float；摄氏度",
        "$.left.temperature: str；设备状态",
    ]
    if query == "TEMPERATURE":
        expected.append("$.right.temperature: float；华氏度")
    assert hits[0].matched_fields == []
    assert hits[1].matched_fields == hits[2].matched_fields == expected[:cap]


@pytest.mark.parametrize("query", ["speed", "$.speed"])
def test_exact_field_summaries_do_not_override_tool_route_priority(query: str) -> None:
    search = HybridSearch()
    build_search(
        search,
        [
            make_tool(
                obj(
                    {
                        "speed": node("number", "车速"),
                        "mileage": node("number", "用于计算平均速度的里程"),
                    }
                ),
                "vehicle",
            ),
            make_tool(name="guide", description="speed reference"),
        ],
    )
    routes(search, [(1, ""), (0, "speed")], [(0, "mileage")])
    hits = search.search(query, top_k=2)
    assert [hit.name for hit in hits] == ["guide", "vehicle"]
    assert hits[1].matched_fields == ["$.speed: float；车速"]
    assert search.search(query, top_k=1)[0].name == "guide"


@pytest.mark.parametrize(
    "query,expected_path",
    [
        ('$["reading.value"]', '$["reading.value"]'),
        ("$['reading.value']", '$["reading.value"]'),
        (r'$["reading\u002evalue"]', '$["reading.value"]'),
        ('$["reading"]["value"]', "$.reading.value"),
        ("$['rows'][*]['value']", "$.rows[*].value"),
        (r"$['a\'b']", '$["a\'b"]'),
        (r"$['a\\b']", r'$["a\\b"]'),
    ],
)
def test_equivalent_jsonpaths_share_exact_field_selection(
    query: str, expected_path: str
) -> None:
    registry = make_registry(
        make_tool(
            obj(
                {
                    "reading.value": node("number", "设备读数"),
                    "reference": node(
                        "string", "reading value reading value reading value"
                    ),
                    "reading": obj({"value": node("number", "嵌套读数")}),
                    "rows": arr(obj({"value": node("number", "记录值")})),
                    "a'b": node("number"),
                    "a\\b": node("number"),
                }
            )
        ),
        max_fields=1,
    )
    assert registry.describe("read", fields=[query]).unknown_fields is None
    fields = registry.search(query)[0].matched_fields
    assert len(fields) == 1 and fields[0].startswith(expected_path + ": ")
    assert registry.search(query) == registry.search(expected_path)


def test_unsupported_jsonpath_remains_a_full_text_query() -> None:
    registry = make_registry(make_tool(obj({"speed": node("number", "车速")})))
    query = "$.speed[?(@ > 10)]"
    assert registry.search(query)[0].matched_fields == ["$.speed: float；车速"]


@pytest.mark.parametrize("method", ["search", "grep"])
def test_queries_reuse_prepared_explanations_and_return_independent_views(
    monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    schema = arr(
        obj({"rows": arr(arr(obj({"speed": node(["number", "null"], "速度")})))})
    )
    registry = make_registry(make_tool(schema))
    discover = getattr(registry, method)
    path = "$[*].rows[*][*].speed"
    query = path if method == "search" else re.escape(path)
    expected = ["$[*].rows[*][*].speed: float | None；速度"]
    assert discover(query)[0].matched_fields == expected

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("A query reparsed registered output documentation")

    for name in (
        "axi.output.documents.SchemaDocument._prepare_fields",
        "axi.output.documents.SchemaFormatter",
        "axi.output.view.select_view",
    ):
        monkeypatch.setattr(name, forbidden)
    hit = discover(query)[0]
    hit.matched_fields[0] = "changed response"
    assert discover(query)[0].matched_fields == expected


def test_semantic_fields_cache_shared_texts_and_keep_owners(tmp_path: Path) -> None:
    provider = Mock()
    provider.embed_documents.side_effect = lambda texts: [
        [1.0, 0.0] if "temperature" in text else [0.0, 1.0] for text in texts
    ]
    provider.embed_query.return_value = [1.0, 0.0]
    tools = [
        make_tool(obj({"temperature": node("number", "温度")}), name)
        for name in ("A", "B")
    ]
    path = tmp_path / "vectors.json"

    def build() -> HybridSearch:
        search = HybridSearch(provider, EmbeddingCache(path))
        build_search(search, tools)
        return search

    search = build()
    assert [call.args[0] for call in provider.embed_documents.call_args_list] == [
        ["A ", "B "],
        ["$.temperature: float；温度"],
    ]
    for _ in range(2):
        hits = search.search("heat")
        assert {hit.name for hit in hits} == {"A", "B"}
        assert all(hit.matched_fields == ["$.temperature: float；温度"] for hit in hits)
    assert provider.embed_query.call_count == 2
    build()
    assert provider.embed_documents.call_count == 2
    tools[0].output_schema["properties"]["temperature"]["description"] = "实时温度"
    build()
    provider.embed_documents.assert_called_with(["$.temperature: float；实时温度"])


@pytest.mark.parametrize("semantic", [False, True])
@pytest.mark.parametrize("cap", [0, 1, 2, 8])
def test_scoped_search_filters_before_tool_and_summary_limits(
    semantic: bool, cap: int
) -> None:
    search = HybridSearch(max_fields=cap)
    build_search(
        search,
        [
            make_tool(obj({key: node("number") for key in ("x", "y", "z")}), name)
            for name in ("A", "B", "X")
        ],
    )
    ranked = [(0, ""), (0, "x"), (1, "x"), (2, "y"), (2, "x"), (2, "z")]
    routes(search, ranked, ranked if semantic else None)
    hit = search.search("measurements", top_k=1, tool="X")[0]
    assert hit.name == "X"
    assert hit.matched_fields == ["$.y: float", "$.x: float", "$.z: float"][:cap]
    search._tool_search._bm25.search_tokens.assert_not_called()
    search._field_search._bm25.search_tokens.assert_called_once()
    if semantic:
        search._tool_search._embedding.search.assert_not_called()
        search._field_search._embedding.search.assert_called_once()
        search._embedding_provider.embed_query.assert_called_once_with("measurements")
    for limit in (0, -1):
        assert search.search("measurements", top_k=limit, tool="X") == []


@pytest.mark.parametrize(
    "query,expected",
    [
        ("$.summary", ["$.summary: dict；车辆统计"]),
        ("里程", ["$.summary.distance: float；里程"]),
        ("distance", ["$.summary.distance: float；里程"]),
        ("$['a.b']", ['$["a.b"]: str；标签']),
        ("onlydescription", []),
        ("notpresent", []),
    ],
)
def test_registry_scoped_search_keeps_exact_paths_and_own_descriptions(
    query: str, expected: list[str]
) -> None:
    schema = obj(
        {
            "summary": obj({"distance": node("number", "里程")}, "车辆统计"),
            "a.b": node("string", "标签"),
        }
    )
    meta = make_tool(
        schema,
        server="fleet",
        description="onlydescription",
    )
    registry = make_registry(meta, make_tool(schema, "other"))
    hits = registry.search(query, tool="read")
    assert [hit.name for hit in hits] == (["fleet/read"] if expected else [])
    if hits:
        assert hits[0].matched_fields == expected
    assert registry.search(query, tool="fleet/read") == hits


def test_scoped_search_resolves_names_and_refreshes_existing_indexes() -> None:
    with pytest.raises(ToolResolveError, match="Tool not found"):
        Registry().search("temperature", tool="missing")
    registry = make_registry(make_tool(name="legacy", description="temperature"))
    assert registry.search("temperature", tool="legacy") == []
    with pytest.raises(ToolResolveError, match="Tool not found"):
        registry.search("temperature", tool="missing")
    registry.register(
        make_tool(obj({"speed": node("number", "旧车速")}), "read", server="fleet")
    )
    assert registry.search("旧车速", tool="read")[0].matched_fields == [
        "$.speed: float；旧车速"
    ]
    engine = registry._hybrid
    meta = registry.resolve("read")
    meta.output_schema["properties"]["speed"]["description"] = "新车速"
    registry.register(meta)
    assert registry.search("新车速", tool="read")[0].matched_fields == [
        "$.speed: float；新车速"
    ]
    assert registry._hybrid is engine
    registry.register(make_tool(name="read", server="other"))
    with pytest.raises(ToolResolveError, match="Ambiguous"):
        registry.search("车速", tool="read")
