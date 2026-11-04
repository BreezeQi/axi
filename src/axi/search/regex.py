"""正则表达式搜索实现。"""

import re
from itertools import islice

from pydantic import BaseModel

from axi.models import DEFAULT_MAX_FIELDS, SearchResult, ToolMeta
from axi.output.documents import FieldDocuments, OutputField


def compile_pattern(pattern: str) -> re.Pattern[str]:
    """独立编译并检查正则，目录为空时也能报告语法错误。"""
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        raise ValueError(f"Invalid regex pattern {pattern!r}: {exc}") from exc


class _ToolText(BaseModel):
    tool: SearchResult
    root: str
    fields: list[OutputField]


class RegexSearch:
    """基于正则表达式的工具搜索。"""

    def __init__(self, max_fields: int = DEFAULT_MAX_FIELDS) -> None:
        self._documents: list[_ToolText] = []
        self._max_fields = max_fields

    def build(
        self,
        tools: list[ToolMeta],
        *,
        field_documents: list[FieldDocuments],
    ) -> None:
        """按数据层级稳定排列字段，便于匹配后截取；隐藏摘要时直接沿用声明顺序。"""
        self._documents = [
            _ToolText.model_construct(
                tool=SearchResult.from_meta(tool),
                root=documentation.root_text,
                fields=sorted(documentation.fields, key=lambda field: field.depth)
                if self._max_fields
                else documentation.fields,
            )
            for tool, documentation in zip(tools, field_documents, strict=True)
        ]

    def search(
        self, pattern: str | re.Pattern[str], top_k: int = 10
    ) -> list[SearchResult]:
        """匹配工具与字段文本，复用目录已编译模式或接受独立正则查询。"""
        compiled = compile_pattern(pattern) if isinstance(pattern, str) else pattern
        if top_k <= 0:
            return []
        results = []
        for document in self._documents:
            tool = document.tool
            tool_matches = bool(
                compiled.search(tool.name)
                or compiled.search(tool.description)
                or (document.root and compiled.search(document.root))
            )
            matches = (
                field for field in document.fields if compiled.search(field.text)
            )
            fields = []
            if self._max_fields:
                fields = list(islice(matches, self._max_fields))
            elif not tool_matches:
                tool_matches = next(matches, None) is not None
            if tool_matches or fields:
                results.append(
                    tool.model_copy(
                        update={"matched_fields": [field.text for field in fields]},
                    )
                )
                if len(results) >= top_k:
                    break
        return results
