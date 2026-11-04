"""BM25 搜索：封装 bm25s 库。"""

from __future__ import annotations

import bm25s

from axi.search.tokenize import preprocess

# 默认 token_pattern 为 \w\w+ 会过滤单字 CJK token，改为允许单字符
_TOKEN_PATTERN = r"(?u)\b\w+\b"


def tokenize_query(query: str) -> list[str]:
    """查询只分词一次，生成不依赖索引词表的 token，供两路 BM25 共用。"""
    text = preprocess(query)
    if not text.strip():
        return []
    return bm25s.tokenize(
        text,
        token_pattern=_TOKEN_PATTERN,
        stopwords=None,
        return_ids=False,
        show_progress=False,
    )[0]


class BM25Search:
    """基于 bm25s 的关键词搜索。"""

    def __init__(self) -> None:
        self._retriever: bm25s.BM25 | None = None
        self._size = 0

    def build_documents(self, documents: list[str]) -> None:
        """按文本建 BM25 索引，文本所属工具由调用方记录。"""
        self._size = len(documents)
        if not documents:
            self._retriever = None
            return
        prepared = {text: preprocess(text) for text in dict.fromkeys(documents)}
        # 相同文本只预处理一次，但保留所有文档行，不改变 BM25 语料统计。
        corpus = [prepared[text] for text in documents]
        corpus_tokens = bm25s.tokenize(
            corpus, token_pattern=_TOKEN_PATTERN, stopwords=None, show_progress=False
        )
        if not corpus_tokens.vocab:
            self._retriever = None
            return
        self._retriever = bm25s.BM25()
        self._retriever.index(corpus_tokens, show_progress=False)

    def search_tokens(
        self, query_tokens: list[str], top_k: int = 10
    ) -> list[tuple[int, float]]:
        """用已分词的查询在当前索引中评分，返回正分文本序号及分数。"""
        if not self._retriever or top_k <= 0 or not query_tokens:
            return []
        indices, scores = self._retriever.retrieve(
            [query_tokens], k=min(top_k, self._size), show_progress=False
        )
        return [
            (int(index), float(score))
            for index, score in zip(indices[0], scores[0])
            if score > 0
        ]
