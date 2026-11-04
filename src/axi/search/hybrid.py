"""工具说明与输出字段分别检索，按工具路优先交错去重取 K。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from itertools import chain, zip_longest

from axi.models import DEFAULT_MAX_FIELDS, SearchResult, ToolMeta
from axi.output.documents import FieldDocuments
from axi.search.bm25 import BM25Search, tokenize_query
from axi.search.cache import EmbeddingCache
from axi.search.corpus import (
    ExactFields,
    SearchDocument,
    build_corpus,
    find_exact_fields,
)
from axi.search.embedding import (
    EmbeddingProvider,
    EmbeddingSearch,
    warn_embedding_failure,
)

type ScoredHits = list[tuple[int, float]]

# RRF 平滑常数（标准值 60，值越大排名差异越平滑）
RRF_K = 60


def _rrf_fuse(
    ranked_lists: list[ScoredHits],
    weights: list[float],
    top_k: int,
) -> ScoredHits:
    """沿用加权 RRF 融合文本名次，工具归属不参与评分。"""
    scores: dict[int, float] = {}
    for ranked, weight in zip(ranked_lists, weights):
        for rank, (idx, _score) in enumerate(ranked, start=1):
            scores[idx] = scores.get(idx, 0.0) + weight / (RRF_K + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


class _TextSearch:
    """单份语料的文本搜索器，工具路与字段路共用 BM25、Embedding 和 RRF。"""

    def __init__(
        self,
        provider: EmbeddingProvider | None,
        cache: EmbeddingCache | None,
        weights: list[float],
    ) -> None:
        self._bm25 = BM25Search()
        self._embedding = (
            EmbeddingSearch(provider, cache) if provider and cache else None
        )
        self._weights = weights
        self._size = 0

    @property
    def has_embedding(self) -> bool:
        """当前检索路是否已有可用的文档向量。"""
        return self._embedding is not None and self._size > 0

    def build(self, texts: list[str]) -> None:
        """构建单路索引；调用方依次构建两路，共用同一份向量缓存。"""
        self._size = len(texts)
        self._bm25.build_documents(texts)
        if self._embedding:
            try:
                self._embedding.build_documents(texts)
            except Exception as exc:
                warn_embedding_failure("indexing", exc)
                self._embedding = None

    def search(
        self,
        query_tokens: list[str],
        top_k: int,
        exact: list[int] | None = None,
        *,
        query_vector: list[float] | None = None,
    ) -> ScoredHits:
        """按文本相关性排序；精确匹配补齐 BM25 遗漏的候选，再统一参与融合。"""
        if not self._size:
            return []
        count = min(top_k, self._size)
        ranked = self._bm25.search_tokens(query_tokens, top_k=count)
        if exact:
            scored = {index for index, _ in ranked}
            ranked += [(index, 0.0) for index in exact if index not in scored]
        if self._embedding and query_vector is not None:
            try:
                semantic = self._embedding.search(query_vector, top_k=count)
            except Exception as exc:
                warn_embedding_failure("query", exc)
            else:
                return _rrf_fuse(
                    [ranked, semantic],
                    self._weights,
                    count,
                )
        return ranked


class HybridSearch:
    """整理两路独立排名，工具路优先交错，字段摘要从已有命中独立收集。"""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        embedding_cache: EmbeddingCache | None = None,
        weight_bm25: float = 0.3,
        weight_embedding: float = 0.7,
        max_fields: int = DEFAULT_MAX_FIELDS,
    ) -> None:
        weights = [weight_bm25, weight_embedding]
        self._embedding_provider = embedding_provider
        self._tool_search = _TextSearch(embedding_provider, embedding_cache, weights)
        self._field_search = _TextSearch(embedding_provider, embedding_cache, weights)
        self._field_documents: list[SearchDocument] = []
        self._exact_fields: ExactFields = {}
        self._max_fields = max_fields
        self._tools: list[SearchResult] = []

    def build(
        self,
        tools: list[ToolMeta],
        *,
        field_documents: list[FieldDocuments],
    ) -> None:
        """复用已准备的字段文档，工具说明与字段语料分别建库。"""
        self._tools = [SearchResult.from_meta(tool) for tool in tools]
        corpus = build_corpus(tools, field_documents)
        self._exact_fields = corpus.exact_fields
        self._field_documents = corpus.output_documents
        self._tool_search.build(corpus.tool_texts)
        self._field_search.build(corpus.output_texts)

    def search(
        self, query: str, top_k: int = 10, *, tool: str | None = None
    ) -> list[SearchResult]:
        """交错两路推荐的前 K 个工具，去重后附带各工具已有的字段命中。"""
        if not self._tools or top_k <= 0 or not query.strip():
            return []
        exact = find_exact_fields(self._exact_fields, query)
        tool_hits, field_hits = self._rank_routes(
            query, top_k, exact, fields_only=tool is not None
        )
        if tool is not None:
            # 字段路保留全部命中，限定归属后再选工具与摘要，不受全局 Top-K 截断。
            field_hits = [
                (index, score)
                for index, score in field_hits
                if self._tools[self._field_documents[index].tool].name == tool
            ]
        field_tools: dict[int, None] = {}
        for index, _score in field_hits:
            field_tools.setdefault(self._field_documents[index].tool, None)
            if len(field_tools) == top_k:
                break
        selected: dict[str, int] = {}
        for owner in chain.from_iterable(
            zip_longest(
                (index for index, _score in tool_hits[:top_k]),
                field_tools,
            )
        ):
            if owner is not None:
                selected.setdefault(self._tools[owner].name, owner)
                if len(selected) == top_k:
                    break
        matched = self._collect_matched_fields(
            field_hits, exact, set(selected.values())
        )
        return [
            self._tools[owner].model_copy(
                update={"matched_fields": matched.get(owner, [])},
            )
            for owner in selected.values()
        ]

    def _rank_routes(
        self, query: str, top_k: int, exact: list[int], *, fields_only: bool
    ) -> tuple[ScoredHits, ScoredHits]:
        """共用一次分词和一个可选查询向量，两路分别评分和排名。"""
        query_tokens = tokenize_query(query)
        query_vector = None
        if self._embedding_provider and any(
            route.has_embedding
            for route in (
                (self._field_search,)
                if fields_only
                else (self._tool_search, self._field_search)
            )
        ):
            try:
                query_vector = self._embedding_provider.embed_query(query)
            except Exception as exc:
                warn_embedding_failure("query", exc)
        if fields_only:
            return [], self._field_search.search(
                query_tokens,
                len(self._field_documents),
                exact,
                query_vector=query_vector,
            )
        if not self._field_documents:
            return (
                self._tool_search.search(
                    query_tokens, top_k, query_vector=query_vector
                ),
                [],
            )
        # 字段路交给一个临时线程，工具路在当前线程执行，按固定路序收集结果。
        with ThreadPoolExecutor(max_workers=1) as pool:
            field_future = pool.submit(
                self._field_search.search,
                query_tokens,
                len(self._field_documents),
                exact,
                query_vector=query_vector,
            )
            tool_hits = self._tool_search.search(
                query_tokens, top_k, query_vector=query_vector
            )
            return tool_hits, field_future.result()

    def _collect_matched_fields(
        self, hits: ScoredHits, exact: list[int], selected: set[int]
    ) -> dict[int, list[str]]:
        """从已有字段排名中收集入选工具的摘要，不再次检索；精确查询只展示对应节点。"""
        if not self._max_fields or not selected:
            return {}
        documents = self._field_documents
        matched: dict[int, list[str]] = {owner: [] for owner in selected}
        exact_ids = set(exact)
        exact_tools = {documents[index].tool for index in exact_ids}
        remaining = len(selected)
        for index, _score in hits:
            document = documents[index]
            if (
                document.tool not in selected
                or document.field is None
                or document.tool in exact_tools
                and index not in exact_ids
            ):
                continue
            fields = matched[document.tool]
            if len(fields) < self._max_fields:
                fields.append(document.field.text)
                if len(fields) == self._max_fields:
                    remaining -= 1
                    if not remaining:
                        break
        return matched
