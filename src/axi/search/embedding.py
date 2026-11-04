"""Embedding 搜索：向量相似度匹配。"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel

from axi.search.cache import EmbeddingCache, content_hash

if TYPE_CHECKING:
    from axi.config import EmbeddingConfig


logger = logging.getLogger(__name__)


def warn_embedding_failure(phase: str, exc: Exception) -> None:
    """统一报告可选 Embedding 的失败原因，保留 BM25 检索。"""
    logger.warning(
        "Embedding %s failed; using BM25. "
        "Check search.embedding in axi.json and provider availability: %s",
        phase,
        exc,
    )


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Embedding provider 接口，兼容 LangChain Embeddings。"""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


def create_embedding_provider(
    embedding_config: EmbeddingConfig,
) -> EmbeddingProvider | None:
    """根据配置创建 embedding provider。"""
    if not embedding_config.provider:
        return None

    if embedding_config.provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        kwargs: dict[str, Any] = {}
        if embedding_config.api_key:
            kwargs["api_key"] = embedding_config.api_key
        if embedding_config.model:
            kwargs["model"] = embedding_config.model
        if embedding_config.base_url:
            kwargs["base_url"] = embedding_config.base_url
        return OpenAIEmbeddings(**kwargs)  # type: ignore[return-value]

    if embedding_config.provider == "jina":
        from langchain_community.embeddings import JinaEmbeddings

        kwargs = {}
        if embedding_config.api_key:
            kwargs["jina_api_key"] = embedding_config.api_key
        if embedding_config.model:
            kwargs["model_name"] = embedding_config.model
        return JinaEmbeddings(**kwargs)  # type: ignore[return-value]

    raise ValueError(
        f"Unknown embedding provider: '{embedding_config.provider}'. "
        f"Supported: 'openai', 'jina'"
    )


class _IndexedVector(BaseModel):
    values: list[float]
    norm: float


def _cosine_similarity(
    a: list[float], b: list[float], norm_a: float, norm_b: float
) -> float:
    """使用预先计算的向量长度，余弦公式沿用原工具搜索实现。"""
    dot = sum(x * y for x, y in zip(a, b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddingSearch:
    """基于向量相似度的搜索。"""

    def __init__(self, provider: EmbeddingProvider, cache: EmbeddingCache) -> None:
        self._provider = provider
        self._cache = cache
        self._vectors: list[_IndexedVector] = []

    def build_documents(self, texts: list[str]) -> None:
        """优先复用缓存向量，批量请求并保存缺失向量。"""
        self._vectors = []
        if not texts:
            return

        hashes = [content_hash(t) for t in texts]

        vectors = {
            h: vector
            for h in dict.fromkeys(hashes)
            if (vector := self._cache.get(h)) is not None
        }
        # 相同字段文本共用一个向量，各文档仍保留独立序号和工具归属。
        missing = {h: text for h, text in zip(hashes, texts) if h not in vectors}
        if missing:
            new_vectors = self._provider.embed_documents(list(missing.values()))
            for h, vector in zip(missing, new_vectors):
                vectors[h] = vector
                self._cache.set(h, vector)
            self._cache.save()
        prepared = {
            h: _IndexedVector.model_construct(
                values=vector, norm=math.sqrt(sum(x * x for x in vector))
            )
            for h, vector in vectors.items()
        }
        self._vectors = [prepared[h] for h in hashes]

    def search(
        self, query_vector: list[float], top_k: int = 10
    ) -> list[tuple[int, float]]:
        """每次查询对共享向量只计算一次相似度，保留各文档序号及稳定排序。"""
        if not self._vectors or top_k <= 0:
            return []
        query_norm = math.sqrt(sum(x * x for x in query_vector))
        similarities: dict[int, float] = {}
        scored: list[tuple[int, float]] = []
        for index, vector in enumerate(self._vectors):
            identity = id(vector.values)
            score = similarities.get(identity)
            if score is None:
                score = _cosine_similarity(
                    query_vector, vector.values, query_norm, vector.norm
                )
                similarities[identity] = score
            scored.append((index, score))
        scored.sort(key=lambda hit: hit[1], reverse=True)
        return [(index, score) for index, score in scored[:top_k] if score > 0]
