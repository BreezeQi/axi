"""用小型构造函数准备 Schema 和工具，使测试数据便于阅读。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from axi.models import (
    FieldPaths,
    ToolDescription,
    ToolMeta,
)
from axi.output.documents import FieldDocuments, SchemaDocument
from axi.output.schema import SchemaNode
from axi.output.selection import DescribeSelection
from axi.registry import Registry

if TYPE_CHECKING:
    from axi.search.hybrid import HybridSearch
    from axi.search.regex import RegexSearch


def node(
    kind: str | list[str] | None = None, description: str | None = None, **keywords: Any
) -> dict[str, Any]:
    result = dict(keywords)
    if kind is not None:
        result["type"] = kind
    if description is not None:
        result["description"] = description
    return result


def obj(
    properties: dict[str, SchemaNode], description: str | None = None, **keywords: Any
) -> dict[str, Any]:
    return node("object", description, properties=properties, **keywords)


def arr(
    items: SchemaNode, description: str | None = None, **keywords: Any
) -> dict[str, Any]:
    return node("array", description, items=items, **keywords)


def ref(name: str) -> dict[str, str]:
    return {"$ref": "#/$defs/" + name}


def make_tool(
    schema: dict[str, Any] | None = None, name: str = "read", **metadata: Any
) -> ToolMeta:
    return ToolMeta(
        name=name,
        output_schema=schema,
        **{"description": "", "source": "native", **metadata},
    )


def make_registry(*tools: ToolMeta, **search_options: Any) -> Registry:
    """通过公开 API 注册测试工具，不绕过注册和失效流程。"""
    registry = Registry(**search_options)
    for tool in tools:
        registry.register(tool)
    return registry


def prepare_field_documents(schema: dict[str, Any] | None) -> FieldDocuments:
    """为 Schema 单元测试准备独立的字段文档。"""
    return SchemaDocument(schema).fields


def build_search(search: HybridSearch | RegexSearch, tools: list[ToolMeta]) -> None:
    """先准备字段文档，再构建内部搜索引擎。"""
    search.build(
        tools,
        field_documents=[prepare_field_documents(tool.output_schema) for tool in tools],
    )


def describe_tool(
    meta: ToolMeta,
    *,
    fields: FieldPaths = None,
) -> ToolDescription:
    """生成输出视图，未指定字段时显式选择根节点。"""
    return make_registry(meta).describe(
        meta.full_name, fields=["$"] if fields is None else fields
    )


def show(schema: dict[str, Any], fields: FieldPaths = None) -> str | None:
    return describe_tool(make_tool(schema), fields=fields).output_fields


def view_text(value: Any) -> str:
    """将便于阅读的测试数据转为预期文档，不解析被测函数的返回文字。"""
    from axi.output.paths import name_segment

    if isinstance(value, dict):
        return (
            "{"
            + ", ".join(
                name_segment(key).removeprefix(".") + ": " + view_text(child)
                for key, child in value.items()
            )
            + "}"
        )
    if isinstance(value, list):
        return "[" + ", ".join(view_text(child) for child in value) + "]"
    return value


def composition(keyword: str, *branches: Any) -> str:
    """组装预期的组合表达式，不解析实际返回的表达式。"""
    if keyword != "oneOf" and len(branches) == 1:
        return view_text(branches[0])
    return f"{keyword}({', '.join(view_text(child) for child in branches)})"


def selected_paths(
    schema: dict[str, Any], fields: FieldPaths = None
) -> set[tuple[str, str]]:
    """直接检查选中的结构和终点，不解析文档表达式。"""
    from axi.output.view import select_view

    selection = DescribeSelection.prepare(fields)
    view = select_view(SchemaDocument(schema), selection.tree)
    return terminal_paths(view.fields, document=view.document)


def terminal_paths(
    view: Any, path: str = "$", *, document: SchemaDocument | None = None
) -> set[tuple[str, str]]:
    """从选择树或普通 JSON 测试数据中读取终点，不引入表达式解析器。"""
    from axi.output.paths import name_segment
    from axi.output.view import _Array, _Branches, _Terminal

    terminals = set()
    visited = set()

    def walk(value: Any, location: str) -> None:
        identity = id(value), location
        if identity in visited:
            return
        visited.add(identity)
        if isinstance(value, _Terminal):
            assert document is not None
            terminals.add((location, document.format(value.definition)))
        elif isinstance(value, _Branches):
            for child in value.members:
                walk(child, location)
        elif isinstance(value, _Array):
            for segment, child in value.items.items():
                walk(child, location + segment)
        elif isinstance(value, str):
            terminals.add((location, value))
        elif isinstance(value, list):
            for child in value:
                walk(child, location + "[*]")
        else:
            for name, child in value.items():
                walk(child, location + name_segment(name))

    walk(view, path)
    return terminals
