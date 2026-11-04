"""验证首次并发查询共用准备结果，不涉及查询期间并发更新定义。"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event
from typing import Any
from unittest.mock import Mock

import pytest

from axi.output.documents import FieldDocuments, SchemaDocument
from axi.registry import Registry
from axi.search.cache import EmbeddingCache
from axi.search.embedding import EmbeddingProvider
from axi.search.hybrid import HybridSearch
from axi.search.regex import RegexSearch
from tests.support import make_registry, make_tool, node, obj


def test_concurrent_cold_queries_share_fields_and_build_each_index_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider = Mock(spec=EmbeddingProvider)
    provider.embed_documents.side_effect = lambda texts: [[1.0] for _ in texts]
    provider.embed_query.return_value = [1.0]
    registry = make_registry(
        make_tool(obj({"speed": node("number", "velocity")}), description="velocity"),
        embedding_provider=provider,
        embedding_cache=EmbeddingCache(tmp_path / "vectors.json"),
    )
    entered, release = Event(), Event()
    start = Barrier(5)
    original = SchemaDocument._prepare_fields
    preparations: list[SchemaDocument] = []

    def prepare(document: SchemaDocument) -> FieldDocuments:
        preparations.append(document)
        entered.set()
        assert release.wait(5), "The test did not release field preparation"
        return original(document)

    builds = {HybridSearch: 0, RegexSearch: 0}
    for engine_type in builds:
        build = engine_type.build

        def counted_build(
            engine: Any, *args: Any, original: Any = build, **kwargs: Any
        ) -> None:
            builds[type(engine)] += 1
            original(engine, *args, **kwargs)

        monkeypatch.setattr(engine_type, "build", counted_build)
    monkeypatch.setattr(SchemaDocument, "_prepare_fields", prepare)

    def discover(method: str) -> Any:
        start.wait(timeout=5)
        return getattr(registry, method)("velocity")

    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = [pool.submit(discover, method) for method in ("search", "grep") * 2]
        try:
            start.wait(timeout=5)
            assert entered.wait(5)
        finally:
            release.set()
        results = [query.result(timeout=5) for query in pending]
    assert all(
        hits[0].matched_fields == ["$.speed: float；velocity"] for hits in results
    )
    assert len(preparations) == 1
    assert builds == {HybridSearch: 1, RegexSearch: 1}
    assert provider.embed_documents.call_count == 2


@pytest.mark.parametrize("method", ["search", "grep"])
def test_failed_preparation_can_be_retried_on_the_next_query(
    monkeypatch: pytest.MonkeyPatch, measurement_registry: Registry, method: str
) -> None:
    original = SchemaDocument._prepare_fields
    attempts = 0

    def prepare(document: SchemaDocument) -> FieldDocuments:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("Field preparation failed")
        return original(document)

    monkeypatch.setattr(SchemaDocument, "_prepare_fields", prepare)
    discover = getattr(measurement_registry, method)
    with pytest.raises(RuntimeError, match="Field preparation failed"):
        discover("oldvelocity")
    for retrieve in (discover, measurement_registry.search, measurement_registry.grep):
        assert retrieve("oldvelocity")[0].matched_fields == [
            "$.speed: float；oldvelocity"
        ]
    assert attempts == 2
