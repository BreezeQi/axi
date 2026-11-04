"""对照 0.0.13 的评分与候选截取，核对工具路结果。"""

from math import sqrt
import asyncio
import json
from collections.abc import Awaitable, Callable
from itertools import product
from unittest.mock import Mock

import bm25s
import pytest
from mcp.types import CallToolResult
from typer.testing import CliRunner

import axi.cli as cli
from axi.config import AxiConfig
from axi.executor import Executor
from axi.mcp_serve import _build_meta_server
from axi.models import ToolMeta
from axi.registry import Registry
from axi.search.bm25 import tokenize_query
from axi.search.cache import EmbeddingCache
from axi.search.embedding import EmbeddingProvider
from axi.search.tokenize import preprocess
from tests.support import make_registry, make_tool, node, obj


def vectors(texts: list[str]) -> list[list[float]]:
    """字段向量设为负相似度，工具向量覆盖正分、同分和未命中。"""
    choices = [[1.0, 0.0], [1.0, 1.0], [-1.0, 0.0]]
    return [
        [-1.0, 0.0]
        if text.startswith("$")
        else choices[int(text.split()[0].rsplit("read", 1)[1]) % 3]
        for text in texts
    ]


def legacy_names(
    tools: list[ToolMeta], query: str, k: int, semantic: bool
) -> list[str]:
    """复现 v0.0.13（ce069c3）：各评分路先取前 K，再做稳定的加权 RRF。"""
    texts = [f"{tool.full_name} {tool.description}" for tool in tools]
    tokenizer = {
        "token_pattern": r"(?u)\b\w+\b",
        "stopwords": None,
        "show_progress": False,
    }
    reference = bm25s.BM25()
    reference.index(
        bm25s.tokenize([preprocess(text) for text in texts], **tokenizer),
        show_progress=False,
    )
    indices, scores = reference.retrieve(
        bm25s.tokenize(preprocess(query), **tokenizer),
        k=min(k, len(tools)),
        show_progress=False,
    )
    lexical = [
        (int(index), float(score))
        for index, score in zip(indices[0], scores[0])
        if score > 0
    ]
    ranked = lexical
    if semantic:
        # 沿用原 Python 余弦评分、稳定排序及正分过滤。
        ranked = sorted(
            [
                (i, vec[0] / sqrt(sum(v * v for v in vec)))
                for i, vec in enumerate(vectors(texts))
            ],
            key=lambda hit: hit[1],
            reverse=True,
        )
        semantic_hits = [(i, score) for i, score in ranked[:k] if score > 0]
        fused: dict[int, float] = {}
        for hits, weight in ((lexical, 0.3), (semantic_hits, 0.7)):
            for rank, (index, _score) in enumerate(hits, start=1):
                fused[index] = fused.get(index, 0.0) + weight / (60 + rank)
        ranked = sorted(fused.items(), key=lambda hit: hit[1], reverse=True)[:k]
    return [tools[index].full_name for index, _score in ranked]


@pytest.fixture(
    scope="module",
    params=list(product(["registry", "cli", "meta"], [False, True], [None, 0, 1000])),
    ids=lambda case: f"{case[0]}-embedding={case[1]}-fields={case[2]}",
)
def legacy_catalogs(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> tuple[str, list[Registry]]:
    """不同查询和数量上限共用只读目录及索引，避免重复建库。"""
    entry, semantic, field_count = request.param
    schema = (
        obj(
            {
                f"unrelated{i}": node("boolean", "device status")
                for i in range(field_count)
            }
        )
        if field_count is not None
        else None
    )
    groups = [
        [
            make_tool(
                schema if index == 0 else None,
                f"read{index}",
                server=f"{source}{index % 2}",
                source=source,
                description="weather " * (index % 3 + 1) + "record",
            )
            for index in range(20)
        ]
        for source in ("native", "mcp")
    ]
    provider = Mock(spec=EmbeddingProvider) if semantic else None
    if provider:
        provider.embed_documents.side_effect = vectors
        provider.embed_query.return_value = [1.0, 0.0]
    cache = EmbeddingCache(tmp_path_factory.mktemp("legacy") / "vectors.json")
    if entry == "cli":
        catalogs = [
            make_registry(*groups[0]),
            make_registry(
                *groups[1], embedding_provider=provider, embedding_cache=cache
            ),
        ]
    else:
        catalogs = [
            make_registry(
                *(tool for pair in zip(*groups) for tool in reversed(pair)),
                embedding_provider=provider,
                embedding_cache=cache,
            )
        ]
    return entry, catalogs


@pytest.mark.parametrize("top_k", [1, 3, 50])
@pytest.mark.parametrize("query", ["weather", "unknown"])
def test_tool_results_match_0013_with_no_field_candidates(
    legacy_catalogs: tuple[str, list[Registry]],
    top_k: int,
    query: str,
    monkeypatch: pytest.MonkeyPatch,
    mcp_call: Callable[..., Awaitable[CallToolResult]],
) -> None:
    entry, catalogs = legacy_catalogs
    registry = catalogs[0]
    if entry == "cli":
        native, remote = catalogs
        monkeypatch.setattr(cli.runtime, "registry", native)
        monkeypatch.setattr(
            cli,
            "app_config",
            AxiConfig.model_validate({"mcpServers": {"mcp": {"command": "unused"}}}),
        )
        monkeypatch.setattr(
            "axi.providers.native.load_native_tool_modules", lambda: None
        )
        monkeypatch.setattr(
            cli,
            "fetch_discovery",
            lambda req: [
                hit.model_dump() for hit in remote.search(req.query, req.top_k)
            ],
        )
        response = CliRunner().invoke(cli.app, ["search", query, "-k", str(top_k)])
        assert response.exit_code == 0, response.output
        results = json.loads(response.stdout)
    elif entry == "meta":
        response = asyncio.run(
            mcp_call(
                _build_meta_server(registry, Executor(registry)),
                "search",
                {"query": query, "top_k": top_k},
            )
        )
        assert not response.is_error
        results = json.loads(response.content[0].text)
    else:
        results = [hit.model_dump() for hit in registry.search(query, top_k)]
    expected = [
        name
        for catalog in catalogs
        for name in legacy_names(
            catalog.list_all(),
            query,
            top_k,
            catalog._embedding_provider is not None,
        )
    ]
    assert [hit["name"] for hit in results] == expected
    assert all(hit["matched_fields"] == [] for hit in results)
    for catalog in catalogs:
        engine = catalog._hybrid
        # 直接检查字段路排名；摘要为空不能证明字段路没有候选。
        assert (
            engine._field_search.search(
                tokenize_query(query),
                len(engine._field_documents),
                query_vector=[1.0, 0.0] if catalog._embedding_provider else None,
            )
            == []
        )
