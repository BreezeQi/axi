"""验证工具说明与字段独立排名、稳定融合及双路并行检索。"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event
from unittest.mock import Mock

import pytest

import axi.search.bm25 as bm25_module
from axi.output.documents import OutputField
from axi.search.corpus import SearchDocument
from axi.search.cache import EmbeddingCache
from axi.search.hybrid import HybridSearch, ScoredHits, _rrf_fuse
from tests.support import build_search, make_tool, node, obj


def test_rrf_uses_text_ranks_and_stable_ties() -> None:
    lexical = [(index, 10.0 - index) for index in range(5)]
    semantic = [(3, 1.0), (4, 0.9), (0, 0.8)]
    results = _rrf_fuse([lexical, semantic], [0.3, 0.7], 5)
    assert dict(results) == pytest.approx(
        {
            0: 0.3 / 61 + 0.7 / 63,
            1: 0.3 / 62,
            2: 0.3 / 63,
            3: 0.3 / 64 + 0.7 / 61,
            4: 0.3 / 65 + 0.7 / 62,
        }
    )
    assert [index for index, _ in results] == [3, 0, 4, 1, 2]
    assert [index for index, _ in _rrf_fuse([lexical], [1.0], 5)] == list(range(5))
    assert _rrf_fuse([[(2, 1.0)], [(1, 1.0)]], [0.5, 0.5], 1)[0][0] == 2


@pytest.mark.parametrize("selected,last_read", [({0, 1}, 6), ({0, 1, 3}, 8)])
def test_summary_collection_stops_only_after_every_selected_tool_is_full(
    selected: set[int], last_read: int
) -> None:
    reads: list[int] = []

    class Documents(list[SearchDocument]):
        def __getitem__(self, index: int) -> SearchDocument:
            reads.append(index)
            return super().__getitem__(index)

    search = HybridSearch(max_fields=2)
    search._field_documents = Documents(
        SearchDocument(
            tool=owner,
            field=OutputField(name=str(i), path=f"$[{i}]", depth=1, text=str(i)),
        )
        for i, owner in enumerate([0, 2, 0, 0, 1, 0, 1, 0, 1])
    )
    hits = [(i, float(9 - i)) for i in range(9)]
    assert search._collect_matched_fields(hits, [], selected) == {
        owner: {0: ["0", "2"], 1: ["4", "6"]}.get(owner, []) for owner in selected
    }
    assert max(reads) == last_read


@pytest.mark.parametrize("reordered", [False, True])
@pytest.mark.parametrize(
    ("top_k", "names"),
    [
        (1, ["C"]),
        (2, ["B", "C"]),
        (3, ["B", "D", "C"]),
        (5, ["B", "C", "D", "A", "E"]),
        (10, ["B", "C", "D", "A", "E"]),
    ],
)
def test_description_route_preserves_legacy_candidate_fusion(
    monkeypatch: pytest.MonkeyPatch, top_k: int, names: list[str], reordered: bool
) -> None:
    search = HybridSearch()
    tools = [
        make_tool(name=name, description="query records")
        for name in ("A", "B", "D", "E", "C")
    ]
    indices = list(range(5))
    if reordered:
        tools.reverse()
        indices.reverse()
    build_search(search, tools)
    engine = search._tool_search
    engine._bm25.search_tokens = Mock(
        side_effect=lambda tokens, top_k: list(zip(indices, [5, 4, 3, 2, 1]))[:top_k]
    )
    engine._embedding = Mock()
    search._embedding_provider = Mock()
    search._embedding_provider.embed_query.return_value = [1.0]
    engine._embedding.search.side_effect = lambda query_vector, top_k: list(
        zip([indices[i] for i in [4, 1, 2, 3, 0]], [5, 4, 3, 2, 1])
    )[:top_k]
    monkeypatch.setattr(
        "axi.search.hybrid.ThreadPoolExecutor",
        Mock(side_effect=AssertionError("An empty field corpus started workers")),
    )
    hits = search.search("general discovery", top_k=top_k)
    assert [hit.name for hit in hits] == names
    assert all(hit.matched_fields == [] for hit in hits)
    engine._bm25.search_tokens.assert_called_once_with(
        ["general", "discovery"], top_k=min(top_k, 5)
    )
    engine._embedding.search.assert_called_once_with([1.0], top_k=min(top_k, 5))


@pytest.mark.parametrize("last_route", ["tool", "field"])
def test_routes_run_concurrently_with_fixed_interleave_order(last_route: str) -> None:
    search = HybridSearch()
    build_search(
        search, [make_tool(obj({"speed": node("number")}), name) for name in ("A", "B")]
    )
    rendezvous = Barrier(2)
    first_done = Event()
    completion_order: list[str] = []

    def rank(route: str) -> ScoredHits:
        rendezvous.wait(timeout=5)
        if route == last_route:
            assert first_done.wait(timeout=5)
        completion_order.append(route)
        first_done.set()
        ranked = [(0 if route == "tool" else 1, 1.0)]
        return ranked

    search._tool_search.search = Mock(side_effect=lambda *args, **kwargs: rank("tool"))
    search._field_search.search = Mock(
        side_effect=lambda *args, **kwargs: rank("field")
    )
    assert [hit.name for hit in search.search("readings", top_k=2)] == ["A", "B"]
    assert completion_order[-1] == last_route
    search._tool_search.search.assert_called_once()
    search._field_search.search.assert_called_once()


def test_concurrent_queries_reuse_only_their_own_embedding(tmp_path: Path) -> None:
    provider = Mock()
    provider.embed_documents.side_effect = lambda texts: [
        [1.0, 0.0] if text.startswith("A ") or text.endswith("热") else [0.0, 1.0]
        for text in texts
    ]
    rendezvous = Barrier(2)

    def embed(query: str) -> list[float]:
        rendezvous.wait(timeout=5)
        return [1.0, 0.0] if query == "hot" else [0.0, 1.0]

    provider.embed_query.side_effect = embed
    search = HybridSearch(provider, EmbeddingCache(tmp_path / "vectors.json"))
    build_search(
        search,
        [
            make_tool(obj({"value": node("number", note)}), name)
            for name, note in [("A", "热"), ("B", "冷")]
        ],
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        queries = [pool.submit(search.search, query, 1) for query in ("hot", "cold")]
        hits = [query.result(timeout=5)[0] for query in queries]
    assert [hit.name for hit in hits] == ["A", "B"]
    assert [hit.matched_fields for hit in hits] == [
        ["$.value: float；热"],
        ["$.value: float；冷"],
    ]
    assert provider.embed_query.call_count == 2


@pytest.mark.parametrize(
    "query,top_k,empty_catalog",
    [
        ("", 5, False),
        (" \t\n", 5, False),
        ("speed", 0, False),
        ("speed", -1, False),
        ("speed", 5, True),
    ],
)
def test_empty_search_skips_both_routes(
    monkeypatch: pytest.MonkeyPatch, query: str, top_k: int, empty_catalog: bool
) -> None:
    search = HybridSearch()
    build_search(
        search, [] if empty_catalog else [make_tool(obj({"speed": node("number")}))]
    )
    for engine in (search._tool_search, search._field_search):
        engine.search = Mock()
    monkeypatch.setattr(
        "axi.search.hybrid.tokenize_query",
        Mock(side_effect=AssertionError("An empty search prepared a query")),
    )
    assert search.search(query, top_k=top_k) == []
    search._tool_search.search.assert_not_called()
    search._field_search.search.assert_not_called()


@pytest.mark.parametrize("large_schema", [False, True])
def test_unrelated_fields_preserve_capability_recall(large_schema: bool) -> None:
    schema = (
        obj({f"sensor{i}": node("number", "传感器采样") for i in range(200)})
        if large_schema
        else None
    )
    search = HybridSearch()
    build_search(
        search,
        [
            make_tool(name="orders", description="按地区查询订单及付款状态"),
            make_tool(schema, "weather", description="查询天气预报"),
        ],
    )
    for query, name in [("订单付款", "orders"), ("天气预报", "weather")]:
        hit = search.search(query)[0]
        assert hit.name == name and hit.matched_fields == []


def test_tool_and_field_hits_keep_capability_separate() -> None:
    search = HybridSearch()
    schema = obj({"temperature": node("number", "温度，单位摄氏度")})
    build_search(
        search,
        [
            make_tool(schema, "pump", description="查询液压泵的运行状态"),
            make_tool(schema, "motor", description="查询电机的运行状态"),
        ],
    )
    hit = search.search("电机 温度")[0]
    assert hit.name == "motor" and hit.matched_fields == [
        "$.temperature: float；温度，单位摄氏度"
    ]
    assert search.search("液压泵")[0].matched_fields == []
    assert not hasattr(hit, "score")


def test_schema_root_prose_stays_out_of_description_recommendations() -> None:
    search = HybridSearch()
    build_search(
        search,
        [
            make_tool(node("number", "温度"), "documented", description="查询测量记录"),
            make_tool(name="capability", description="查询温度"),
        ],
    )
    assert [hit.name for hit in search.search("温度", top_k=1)] == ["capability"]
    hits = search.search("温度", top_k=2)
    assert [hit.name for hit in hits] == ["capability", "documented"]
    assert all(hit.matched_fields == [] for hit in hits)


@pytest.mark.parametrize("description", ["", "++ .. ## &&"])
@pytest.mark.parametrize("cap", [0, 8])
def test_tokenless_tool_text_does_not_disable_field_recall(
    description: str, cap: int
) -> None:
    search = HybridSearch(max_fields=cap)
    build_search(
        search, [make_tool(obj({"x": node("number")}), "A", description=description)]
    )
    for query in ("x", "float"):
        hit = search.search(query)[0]
        assert hit.name == "A" and hit.matched_fields == ["$.x: float"][:cap]
    assert search.search("unknown") == []


@pytest.mark.parametrize("query", ["温度", "temperature", "$.temperature", "???"])
def test_query_is_tokenized_once_for_both_corpora(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], query: str
) -> None:
    search = HybridSearch()
    build_search(
        search,
        [
            make_tool(
                obj({"temperature": node("number", "温度")}),
                description="查询设备温度",
            ),
            make_tool(name="orders", description="读取订单状态"),
        ],
    )
    preprocess = Mock(wraps=bm25_module.preprocess)
    tokenize = Mock(wraps=bm25_module.bm25s.tokenize)
    monkeypatch.setattr(bm25_module, "preprocess", preprocess)
    monkeypatch.setattr(bm25_module.bm25s, "tokenize", tokenize)
    capsys.readouterr()
    hits = search.search(query)
    assert capsys.readouterr().err == ""
    preprocess.assert_called_once_with(query)
    if query == "???":
        assert hits == []
        tokenize.assert_not_called()
    else:
        assert hits[0].name == "read"
        assert hits[0].matched_fields == ["$.temperature: float；温度"]
        tokenize.assert_called_once()
