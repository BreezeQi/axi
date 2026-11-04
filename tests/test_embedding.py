"""离线验证余弦评分、向量缓存持久化及文档归属。"""

import math
import random
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
import pytest
from openai import APIConnectionError, OpenAIError, RateLimitError

from axi.config import SearchConfig
from axi.registry import Registry
from axi.search.hybrid import HybridSearch
from tests.support import make_tool, build_search, node, obj

from axi.search.cache import EmbeddingCache, content_hash
from axi.search.embedding import EmbeddingSearch, _cosine_similarity


def test_cached_vectors_keep_owners_and_cosine_scores(tmp_path: Path) -> None:
    cache = EmbeddingCache(tmp_path / "vectors.json")
    assert cache.get("missing") is None
    cache.set(content_hash("warm"), [1.0, 0.0])
    assert cache.get(content_hash("warm")) == [1.0, 0.0]
    provider = Mock()
    provider.embed_documents.return_value = [[2.0, 5.0], [0.0, 0.0], [-3.0, -4.0]]
    search = EmbeddingSearch(provider, cache)
    with (
        patch.object(cache, "get", wraps=cache.get) as cache_read,
        patch("axi.search.embedding.math.sqrt", wraps=math.sqrt) as norms,
    ):
        search.build_documents(["shared", "warm", "shared", "zero", "opposite"])
    assert cache_read.call_count == 4
    assert norms.call_count == 4
    provider.embed_documents.assert_called_once_with(["shared", "zero", "opposite"])

    expected = 26.0 / (5.0 * 29.0**0.5)
    with (
        patch(
            "axi.search.embedding._cosine_similarity", wraps=_cosine_similarity
        ) as cosine,
        patch("axi.search.embedding.math.sqrt", wraps=math.sqrt) as norms,
    ):
        assert search.search([3.0, 4.0], top_k=5) == [
            (0, expected),
            (2, expected),
            (1, 0.6),
        ]
        assert cosine.call_count == 4
        assert norms.call_count == 1
        assert search.search([1.0, 0.0], top_k=2) == [
            (1, 1.0),
            (0, 2.0 / 29.0**0.5),
        ]
        assert cosine.call_count == 8
        assert norms.call_count == 2
        assert search.search([0.0, 0.0]) == []
        assert cosine.call_count == 12
        assert norms.call_count == 3
        assert search.search([1.0, 0.0], top_k=0) == []
        assert cosine.call_count == 12
        assert norms.call_count == 3
    # 准备索引不改写缓存中保存的原始向量。
    assert cache.get(content_hash("shared")) == [2.0, 5.0]
    provider.embed_query.assert_not_called()


@pytest.mark.parametrize("dimension", [0, 1, 3, 64, 512])
def test_prepared_lengths_preserve_original_scores_and_order(
    tmp_path: Path, dimension: int
) -> None:
    rng = random.Random(19)
    vectors = [[value] * dimension for value in (0.0, 1.0, -1.0)] + [
        [rng.uniform(-1, 1) for _ in range(dimension)] for _ in range(30)
    ]
    provider = Mock()
    provider.embed_documents.return_value = vectors
    search = EmbeddingSearch(provider, EmbeddingCache(tmp_path / "vectors.json"))
    search.build_documents([f"document-{index}" for index in range(len(vectors))])

    def original_score(query: list[float], document: list[float]) -> float:
        """独立复现 0.0.13 的余弦公式，不依赖当前索引准备结果。"""
        dot = sum(x * y for x, y in zip(query, document))
        norm_query = math.sqrt(sum(x * x for x in query))
        norm_document = math.sqrt(sum(x * x for x in document))
        if norm_query == 0 or norm_document == 0:
            return 0.0
        return dot / (norm_query * norm_document)

    for query in vectors:
        ranked = sorted(
            (
                (index, original_score(query, vector))
                for index, vector in enumerate(vectors)
            ),
            key=lambda hit: hit[1],
            reverse=True,
        )
        for top_k in (1, len(vectors)):
            assert search.search(query, top_k=top_k) == [
                (index, score) for index, score in ranked[:top_k] if score > 0
            ]


def test_importing_axi_keeps_numerical_search_dependencies_lazy() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import axi; import sys; "
            "assert 'numpy' not in sys.modules; assert 'bm25s' not in sys.modules",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_rebuilding_vector_index_discards_previous_document_mapping(
    tmp_path: Path,
) -> None:
    provider = Mock()
    provider.embed_documents.return_value = [[1.0, 0.0]]
    search = EmbeddingSearch(provider, EmbeddingCache(tmp_path / "vectors.json"))
    search.build_documents(["old", "old"])
    provider.embed_documents.return_value = [[0.0, 2.0]]
    search.build_documents(["new"])
    assert search.search([0.0, 1.0]) == [(0, 1.0)]
    search.build_documents([])
    assert search.search([0.0, 1.0]) == []
    provider.embed_query.assert_not_called()


def test_cosine_ties_keep_the_0013_tool_order(tmp_path: Path) -> None:
    query = [0.7778282797757714, 0.8976035129913674, 0.5562439808087439]
    first = [0.7620485378990902, 0.40024042866305687, 1.0182774762589422]
    provider = Mock()
    provider.embed_documents.return_value = [first, [3 * value for value in first]]
    provider.embed_query.return_value = query
    search = HybridSearch(provider, EmbeddingCache(tmp_path / "vectors.json"))
    build_search(search, [make_tool(name=name) for name in ("first", "second")])
    assert [hit.name for hit in search.search("semantic query", top_k=1)] == ["first"]
    provider.embed_query.assert_called_once_with("semantic query")


def test_rebuilds_reuse_persisted_vectors(
    tmp_path: Path,
) -> None:
    provider = Mock()
    provider.embed_documents.side_effect = lambda texts: [
        [float(index + 1), 1.0] for index in range(len(texts))
    ]
    provider.embed_query.return_value = [1.0, 0.0]
    cache = EmbeddingCache(tmp_path / "vectors.json", namespace="model")
    search = HybridSearch(provider, cache)
    tools = [
        make_tool(obj({"value": node("number", "数值")}), name)
        for name in ("first", "second")
    ]
    build_search(search, tools)
    expected = search.search("query")
    vectors = search._field_search._embedding._vectors
    assert vectors[0] is vectors[1]
    with patch.object(Path, "write_text") as write:
        build_search(search, tools)
    write.assert_not_called()
    assert search.search("query") == expected
    assert provider.embed_documents.call_count == 2
    assert cache.get(content_hash("first ")) == [1.0, 1.0]

    tools[0].description = "新工具说明"
    tools[0].output_schema["properties"]["value"]["description"] = "新字段说明"
    build_search(search, tools)
    assert provider.embed_documents.call_count == 4
    build_search(
        HybridSearch(provider, EmbeddingCache(cache._path, namespace="model")), tools
    )
    assert provider.embed_documents.call_count == 4


def test_cache_write_failure_uses_the_original_bm25_fallback(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    provider = Mock()
    provider.embed_documents.side_effect = lambda texts: [[1.0] for _ in texts]
    cache = EmbeddingCache(tmp_path / "vectors.json")
    search = HybridSearch(provider, cache)
    tools = [make_tool(obj({"speed": node("number", "车速")}), description="telemetry")]
    with patch.object(Path, "write_text", side_effect=OSError("cache unavailable")):
        build_search(search, tools)
    assert provider.embed_documents.call_count == 2
    assert "cache unavailable" in caplog.text
    assert "using BM25" in caplog.text
    assert search.search("velocity") == []
    assert [hit.name for hit in search.search("telemetry")] == ["read"]
    assert search.search("车速")[0].matched_fields == ["$.speed: float；车速"]
    provider.embed_query.assert_not_called()


@pytest.mark.parametrize("failed_route", ["tool", "field"])
def test_one_route_indexing_failure_keeps_the_other_route_semantic(
    tmp_path: Path, failed_route: str
) -> None:
    provider = Mock()

    def embed(texts: list[str]) -> list[list[float]]:
        if texts[0].startswith("$") == (failed_route == "field"):
            raise RuntimeError("One corpus failed to index")
        return [[1.0] for _ in texts]

    provider.embed_documents.side_effect = embed
    provider.embed_query.return_value = [1.0]
    cache = EmbeddingCache(tmp_path / "vectors.json")
    search = HybridSearch(provider, cache)
    build_search(search, [make_tool(obj({"value": node("number", "measurement")}))])
    hits = search.search("unmatched semantic query")
    assert [hit.name for hit in hits] == ["read"]
    assert hits[0].matched_fields == (
        [] if failed_route == "field" else ["$.value: float；measurement"]
    )
    provider.embed_query.assert_called_once_with("unmatched semantic query")


@pytest.mark.parametrize(
    "field,changed",
    [("model", "new-model"), ("baseUrl", "https://new.invalid"), ("provider", "jina")],
)
def test_registry_cache_separates_vector_spaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, changed: str
) -> None:
    from langchain_community.embeddings import JinaEmbeddings
    from langchain_openai import OpenAIEmbeddings

    cache_path = tmp_path / "vectors.json"
    calls = []
    original = {
        "provider": "openai",
        "model": "old-model",
        "baseUrl": "https://old.invalid",
        "apiKey": "not-for-cache",
    }

    def embed(
        provider: OpenAIEmbeddings | JinaEmbeddings, texts: list[str]
    ) -> list[list[float]]:
        switched = (
            isinstance(provider, JinaEmbeddings)
            or provider.model == "new-model"
            or "new.invalid" in str(provider.client._client.base_url)
        )
        calls.append(switched)
        return [[0.0, 1.0], [1.0, 0.0]] if switched else [[1.0, 0.0], [0.0, 1.0]]

    for provider in (OpenAIEmbeddings, JinaEmbeddings):
        monkeypatch.setattr(provider, "embed_documents", embed)
        monkeypatch.setattr(provider, "embed_query", lambda self, text: [1.0, 0.0])
    monkeypatch.setattr(
        "axi.registry.EmbeddingCache",
        lambda *, namespace: EmbeddingCache(cache_path, namespace=namespace),
    )
    for config, expected in (
        (original, "demo/alpha"),
        ({**original, field: changed}, "demo/beta"),
        ({**original, field: changed}, "demo/beta"),
    ):
        registry = Registry.from_search_config(
            SearchConfig.model_validate({"embedding": config})
        )
        for tool in [
            make_tool(name=name, server="demo", description=name, source="mcp")
            for name in ("alpha", "beta")
        ]:
            registry.register(tool)
        assert registry.search("unmatched")[0].name == expected
    assert calls == [False, True]
    assert "not-for-cache" not in cache_path.read_text()


@pytest.mark.parametrize("phase", ["build", "query"])
@pytest.mark.parametrize(
    "error_type", [RuntimeError, OpenAIError, APIConnectionError, RateLimitError]
)
def test_embedding_failure_preserves_lexical_discovery(
    tmp_path: Path, phase: str, error_type: type[Exception]
) -> None:
    request = httpx.Request("POST", "https://example.invalid/embeddings")
    if error_type is APIConnectionError:
        error = APIConnectionError(request=request)
    elif error_type is RateLimitError:
        error = RateLimitError(
            "rate limit", response=httpx.Response(429, request=request), body={}
        )
    else:
        error = error_type("offline")
    provider = Mock()
    provider.embed_documents.side_effect = lambda texts: [[1.0] for _ in texts]
    (
        provider.embed_documents if phase == "build" else provider.embed_query
    ).side_effect = error
    search = HybridSearch(provider, EmbeddingCache(tmp_path / "vectors.json"))
    build_search(
        search,
        [
            make_tool(obj({"x": node("number")}), "alpha", server="demo"),
            make_tool(name="beta", server="demo", description="news"),
        ],
    )
    assert search.search("alpha")[0].name == "demo/alpha"
    assert search.search("x")[0].matched_fields == ["$.x: float"]
    assert search.search("unmatched phrase") == search.search("x", top_k=0) == []
    if phase == "query":
        assert provider.embed_query.call_count == 3
        provider.embed_query.side_effect = None
        provider.embed_query.return_value = [1.0]
        assert search.search("unmatched phrase")[0].matched_fields == ["$.x: float"]
    else:
        provider.embed_query.assert_not_called()
