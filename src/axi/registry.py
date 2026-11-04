"""工具注册中心：管理所有工具元数据，提供搜索接口。"""

from __future__ import annotations

import json
from collections import Counter
from threading import Lock
from typing import TYPE_CHECKING

from axi.config import SearchConfig
from axi.models import (
    DEFAULT_MAX_FIELDS,
    SearchResult,
    FieldPaths,
    ToolDescription,
    ToolDescriptionError,
    ToolMeta,
)
from axi.search.cache import EmbeddingCache, content_hash
from axi.search.embedding import EmbeddingProvider, create_embedding_provider

if TYPE_CHECKING:
    from axi.output.describe import ToolDescriber
    from axi.output.documents import FieldDocuments, SchemaDocument
    from axi.output.selection import DescribeSelection
    from axi.search.hybrid import HybridSearch
    from axi.search.regex import RegexSearch


def split_names(value: str) -> list[str]:
    """解析逗号分隔的名称列表，忽略空段。"""
    return [n.strip() for n in value.split(",") if n.strip()]


class ToolResolveError(Exception):
    """工具名解析失败（未找到或存在歧义）。"""


class ToolNotFoundError(ToolResolveError):
    """工具未找到。"""


class AmbiguousToolError(ToolResolveError):
    """工具名存在歧义。"""


class Registry:
    """axi 的工具注册中心。"""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider | None = None,
        embedding_cache: EmbeddingCache | None = None,
        weight_bm25: float = 0.3,
        weight_embedding: float = 0.7,
        max_fields: int = DEFAULT_MAX_FIELDS,
    ) -> None:
        self._tools: dict[str, ToolMeta] = {}
        self._version = 0
        # 首次准备使用简单锁；search 与 grep 各自建库，互不触发对方重建。
        self._documents_lock = Lock()
        self._hybrid_lock = Lock()
        self._regex_lock = Lock()
        # 搜索引擎按需构建：bm25s/jieba/numpy 是重 import，describe/run/list
        # 完全用不到，延迟到 search()/grep() 首次调用时才付这笔冷启动成本。
        self._embedding_provider = embedding_provider
        self._embedding_cache = embedding_cache
        self._weight_bm25 = weight_bm25
        self._weight_embedding = weight_embedding
        self._max_fields = max_fields
        self._regex: RegexSearch | None = None
        self._hybrid: HybridSearch | None = None
        self._documents: dict[str, SchemaDocument] = {}
        self._regex_version = -1
        self._hybrid_version = -1

    @classmethod
    def from_search_config(cls, cfg: SearchConfig) -> Registry:
        """按搜索配置创建当前目录的检索能力，沿用向量缓存标识。"""
        provider = create_embedding_provider(cfg.embedding)
        namespace = content_hash(
            json.dumps(cfg.embedding.model_dump(exclude={"api_key"}), sort_keys=True)
        )
        return cls(
            embedding_provider=provider,
            embedding_cache=EmbeddingCache(namespace=namespace) if provider else None,
            weight_bm25=cfg.weights.bm25,
            weight_embedding=cfg.weights.embedding,
            max_fields=cfg.max_fields,
        )

    def register(self, meta: ToolMeta) -> None:
        """注册或替换工具定义；更新定义后须重新注册以刷新检索索引。"""
        self._tools[meta.full_name] = meta
        self._documents.pop(meta.full_name, None)
        self._version += 1

    def get(self, full_name: str) -> ToolMeta | None:
        """按完整名称获取工具元数据。"""
        return self._tools.get(full_name)

    def resolve(self, name: str) -> ToolMeta:
        """解析工具名：支持完整名称和短名称。

        - 含 ``/`` → 按 full_name 精确查找
        - 不含 ``/`` → 在所有工具的 ``.name`` 中精确匹配
          - 唯一匹配 → 返回
          - 多个匹配 → 抛 ToolResolveError，列出所有候选
          - 无匹配 → 抛 ToolResolveError
        """
        if "/" in name:
            meta = self.get(name)
            matches = [meta] if meta is not None else []
        else:
            matches = [t for t in self._tools.values() if t.name == name]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            candidates = ", ".join(t.full_name for t in matches)
            raise AmbiguousToolError(
                f"Ambiguous tool name '{name}'. Use a full name: {candidates}"
            )
        raise ToolNotFoundError(
            f"Tool not found: {name}. Use search to discover available tools."
        )

    def list_all(self) -> list[ToolMeta]:
        """列出所有工具。"""
        return list(self._tools.values())

    def prepare_describe(self, selection: DescribeSelection) -> ToolDescriber:
        """创建本次查看的详情组装器，底层文档由 Registry 共享。"""
        from axi.output.describe import ToolDescriber

        return ToolDescriber(selection, self._documentation)

    def describe(
        self,
        name: str,
        fields: FieldPaths = None,
        *,
        schema: bool = False,
    ) -> ToolDescription:
        """解析名称并查看单个工具，查找错误由调用方处理。"""
        from axi.output.selection import DescribeSelection

        meta = self.resolve(name)
        selection = DescribeSelection.prepare(fields, schema=schema)
        return self.prepare_describe(selection).describe(meta)

    def describe_many(
        self,
        names: list[str],
        fields: FieldPaths = None,
        *,
        schema: bool = False,
    ) -> list[ToolDescription | ToolDescriptionError]:
        """按名称顺序批量查看，保留重复项，单个工具失败不影响其余结果。"""
        if not names:
            return []
        from axi.output.selection import DescribeSelection

        describer: ToolDescriber | str
        try:
            describer = self.prepare_describe(
                DescribeSelection.prepare(fields, schema=schema)
            )
        except ValueError as exc:
            describer = str(exc)
        results: list[ToolDescription | ToolDescriptionError] = []
        for name in names:
            try:
                meta = self.resolve(name)
                if isinstance(describer, str):
                    result = ToolDescriptionError(error=describer)
                else:
                    result = describer.describe(meta)
            except (ToolResolveError, ValueError) as exc:
                result = ToolDescriptionError(error=str(exc))
            results.append(result)
        return results

    def list_names(self) -> list[str]:
        """列出所有工具的 full_name。"""
        return list(self._tools)

    def server_tool_counts(self) -> dict[str, int]:
        """按 server 统计工具数量（无 server 归入 "unknown"）。"""
        return dict(Counter(m.server or "unknown" for m in self._tools.values()))

    def set_server(self, full_name: str, server: str) -> None:
        """更新工具的 server 名，同时更新注册 key。"""
        meta = self._tools.pop(full_name)
        self._documents.pop(full_name, None)
        updated = meta.model_copy(update={"server": server})
        self.register(updated)

    def _documentation(self, meta: ToolMeta) -> SchemaDocument:
        """按工具共享按需准备的文档，重新注册时清除该工具的缓存。"""
        from axi.output.documents import SchemaDocument

        with self._documents_lock:
            if meta.full_name not in self._documents:
                self._documents[meta.full_name] = SchemaDocument(meta.output_schema)
            return self._documents[meta.full_name]

    def _search_catalog(self) -> tuple[list[ToolMeta], list[FieldDocuments]]:
        """为需要重建的索引取得工具目录及共享字段条目。"""
        from axi.output.documents import FieldDocuments

        tools = self.list_all()
        empty = FieldDocuments()
        return (
            tools,
            [
                self._documentation(tool).fields
                if tool.output_schema is not None
                else empty
                for tool in tools
            ],
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        tool: str | None = None,
    ) -> list[SearchResult]:
        """双路检索当前目录；指定工具时只检索它的字段文档。"""
        if top_k <= 0 or not query.strip():
            return []
        if tool is not None:
            tool = self.resolve(tool).full_name
        if not self._tools:
            return []
        with self._hybrid_lock:
            if self._hybrid is None:
                from axi.search.hybrid import HybridSearch

                self._hybrid = HybridSearch(
                    embedding_provider=self._embedding_provider,
                    embedding_cache=self._embedding_cache,
                    weight_bm25=self._weight_bm25,
                    weight_embedding=self._weight_embedding,
                    max_fields=self._max_fields,
                )
            if self._hybrid_version != self._version:
                tools, field_documents = self._search_catalog()
                self._hybrid.build(tools, field_documents=field_documents)
                self._hybrid_version = self._version
        return self._hybrid.search(query, top_k=top_k, tool=tool)

    def grep(
        self,
        pattern: str,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """对当前目录执行全文正则，按登记顺序最多返回 K 个工具。"""
        from axi.search.regex import RegexSearch, compile_pattern

        compiled = compile_pattern(pattern)
        if top_k <= 0:
            return []
        with self._regex_lock:
            if self._regex is None:
                self._regex = RegexSearch(max_fields=self._max_fields)
            if self._regex_version != self._version:
                tools, field_documents = self._search_catalog()
                self._regex.build(tools, field_documents=field_documents)
                self._regex_version = self._version
        return self._regex.search(compiled, top_k=top_k)
