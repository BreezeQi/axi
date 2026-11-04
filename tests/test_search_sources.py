"""验证目录合并、共享评分与 daemon 注册刷新。

排名和摘要边界由搜索引擎测试覆盖；这里核对混合来源仍属于同一目录，
保持统一语料和注册顺序。
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from axi.models import ToolMeta
from axi.config import AxiConfig
from axi.daemon.protocol import DaemonRequest
from axi.daemon.server import DaemonServer
from axi.registry import Registry
from axi.search.bm25 import tokenize_query
from axi.search.cache import EmbeddingCache
from axi.search.embedding import EmbeddingProvider
from axi.search.hybrid import HybridSearch
from axi.search.regex import RegexSearch
from tests.support import build_search, make_tool, node, obj


@pytest.fixture
def catalogs() -> tuple[list[ToolMeta], list[ToolMeta]]:
    groups: list[list[ToolMeta]] = []
    for source in ("native", "mcp"):
        groups.append(
            [
                make_tool(
                    obj(
                        {
                            "sensor": obj(
                                {
                                    "temperature": node("number", "温度"),
                                    "status": node("string", "温度状态"),
                                },
                                "温度传感器",
                            )
                        }
                    ),
                    name=f"read{index}",
                    server=f"{source}{index % 2}",
                    source=source,
                    description="temperature " * (index + 1),
                )
                for index in range(4)
            ]
        )
    return groups[0], groups[1]


@pytest.mark.parametrize("semantic", [False, True])
@pytest.mark.parametrize("query", ["temperature 温度", "$.sensor.temperature"])
def test_search_matches_one_mixed_catalog(
    catalogs: tuple[list[ToolMeta], list[ToolMeta]],
    tmp_path: Path,
    semantic: bool,
    query: str,
) -> None:
    provider = Mock(spec=EmbeddingProvider) if semantic else None
    if provider:
        provider.embed_documents.side_effect = lambda texts: [
            [float(len(text) % 7 + 1), float(len(text) % 5 + 1)] for text in texts
        ]
        provider.embed_query.return_value = [1.0, 1.0]

    # 按注册顺序交替放入两种来源；同一目录使用统一语料评分。
    tools = [tool for pair in zip(catalogs[1], catalogs[0]) for tool in pair]
    baseline = HybridSearch(provider, EmbeddingCache(tmp_path / "baseline.json"))
    build_search(baseline, tools)
    expected = baseline.search(query, top_k=2)

    registry = Registry(provider, EmbeddingCache(tmp_path / "mixed.json"))
    for tool in tools:
        registry.register(tool)
    if provider:
        provider.reset_mock()
    hits = registry.search(query, top_k=2)
    assert hits == expected
    assert len(hits) == 2
    if provider:
        provider.embed_query.assert_called_once_with(query)

    # 直接检查两路语料，不能只验证最终结果的整体数量上限。
    tokens = tokenize_query(query)
    for route in ("_tool_search", "_field_search"):
        actual = getattr(registry._hybrid, route)
        standalone = getattr(baseline, route)
        assert actual._bm25.search_tokens(tokens, 100) == (
            standalone._bm25.search_tokens(tokens, 100)
        )


@pytest.mark.parametrize(
    "pattern", ["temperature", r"^\$\.sensor\.temperature:", "absent"]
)
def test_grep_matches_one_mixed_catalog_in_registration_order(
    catalogs: tuple[list[ToolMeta], list[ToolMeta]],
    pattern: str,
) -> None:
    tools = [tool for pair in zip(catalogs[1], catalogs[0]) for tool in pair]
    engine = RegexSearch()
    build_search(engine, tools)
    expected = engine.search(pattern, top_k=2)
    registry = Registry()
    for tool in tools:
        registry.register(tool)
    assert registry.grep(pattern, top_k=2) == expected
    assert [hit.name for hit in expected] == (
        [] if pattern == "absent" else [tool.full_name for tool in tools][:2]
    )


@pytest.mark.parametrize("method", ["search", "grep"])
@pytest.mark.parametrize("matching", ["native", "mcp", "neither"])
def test_empty_source_does_not_expand_the_other_source_limit(
    matching: str, method: str
) -> None:
    registry = Registry()
    for source in ("mcp", "native"):
        for index in range(3):
            registry.register(
                make_tool(
                    name=f"read{index}",
                    server=f"{source}{index}",
                    source=source,
                    description="temperature" if source == matching else "orders",
                )
            )
    hits = getattr(registry, method)("temperature", top_k=2)
    assert [hit.source for hit in hits] == (
        [] if matching == "neither" else [matching] * 2
    )


@pytest.mark.parametrize("max_fields", [0, 1, 2])
@pytest.mark.parametrize("method", ["search", "grep"])
def test_daemon_discovery_reuses_indexes_and_refreshes_registered_definitions(
    catalogs: tuple[list[ToolMeta], list[ToolMeta]],
    monkeypatch: pytest.MonkeyPatch,
    max_fields: int,
    method: str,
) -> None:
    config = AxiConfig.model_validate({"search": {"maxFields": max_fields}})
    monkeypatch.setattr("axi.daemon.server.app_config", config)
    server = DaemonServer()
    baseline = Registry.from_search_config(config.search)
    for meta in catalogs[1]:
        server.registry.register(meta)
        baseline.register(meta)
    engine_attr = "_hybrid" if method == "search" else "_regex"

    def discover(query: str) -> list[dict]:
        response = json.loads(
            asyncio.run(
                server._dispatch(DaemonRequest(method=method, query=query, top_k=2))
            )
        )
        assert response["status"] == "success", response["error"]
        return response["data"]

    query = "temperature|温度" if method == "grep" else "temperature 温度"
    expected = [hit.model_dump() for hit in getattr(baseline, method)(query, 2)]
    assert len(expected) == 2
    assert all(len(hit["matched_fields"]) == max_fields for hit in expected)
    assert discover(query) == expected
    engine = getattr(server.registry, engine_attr)
    assert discover(query) == expected
    assert getattr(server.registry, engine_attr) is engine
    replacement = server.registry.list_all()[0].model_copy(
        update={
            "description": "updatedtemperature",
            "output_schema": obj({"new": node("integer", "updatedtemperature")}),
        }
    )
    server.registry.register(replacement)
    hits = discover("updatedtemperature")
    assert [hit["name"] for hit in hits] == [replacement.full_name]
    assert hits[0]["matched_fields"] == (
        ["$.new: int；updatedtemperature"] if max_fields else []
    )
    assert getattr(server.registry, engine_attr) is engine
    assert server.registry.list_names() == baseline.list_names()
