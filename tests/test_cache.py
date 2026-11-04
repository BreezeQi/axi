"""Embedding 缓存测试。"""

from pathlib import Path

from axi.search.cache import EmbeddingCache


class TestEmbeddingCache:
    def test_persistence(self, tmp_path: Path) -> None:
        path = tmp_path / "test.json"
        cache1 = EmbeddingCache(path)
        cache1.set("k1", [1.0, 2.0])
        cache1.save()

        cache2 = EmbeddingCache(path)
        cache1.set("k2", [3.0, 4.0])
        cache1.save()
        assert cache2.get("k1") == [1.0, 2.0]
        assert cache2.get("k2") is None
        assert EmbeddingCache(path).get("k2") == [3.0, 4.0]

    def test_namespace_does_not_reuse_legacy_or_other_models(self, tmp_path):
        path = tmp_path / "cache.json"
        legacy = EmbeddingCache(path)
        legacy.set("same-text", [1.0])
        legacy.save()
        first = EmbeddingCache(path, namespace="model-one")
        assert first.get("same-text") is None
        first.set("same-text", [2.0])
        first.save()
        second = EmbeddingCache(path, namespace="model-two")
        assert second.get("same-text") is None
        assert EmbeddingCache(path, namespace="model-one").get("same-text") == [2.0]
