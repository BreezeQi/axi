"""读取 Schema 的本地引用和单层结构，不处理字段选择或展示深度。"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Self
from urllib.parse import unquote

from pydantic import BaseModel, Field

type SchemaNode = dict[str, Any] | bool

MAX_DATA_DEPTH = 64

_STRUCTURE_KEYS = {
    "type",
    "properties",
    "items",
    "prefixItems",
    "additionalItems",
    "anyOf",
    "oneOf",
    "allOf",
}
_TYPE_CHILDREN = {
    "object": ("properties",),
    "array": ("items", "prefixItems", "additionalItems"),
}


def schema_types(node: dict[str, Any]) -> list[str]:
    """读取节点声明的类型，不推导类型或校验约束。"""
    kinds = node.get("type", [])
    return kinds if isinstance(kinds, list) else [kinds]


def schema_pointer(root: dict[str, Any], ref: str) -> SchemaNode | None:
    """按 JSON Pointer 解析本地引用，不请求外部定义。"""
    if ref == "#":
        return root
    if not ref.startswith("#/"):
        return None
    node: Any = root
    for part in unquote(ref[2:]).split("/"):
        key = part.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and key in node:
            node = node[key]
        elif (
            isinstance(node, list)
            and key.isascii()
            and key.isdecimal()
            and str(int(key)) == key
            and int(key) < len(node)
        ):
            node = node[int(key)]
        else:
            return None
    return node if isinstance(node, (dict, bool)) else None


def array_parts(node: dict[str, Any]) -> tuple[list[SchemaNode], str]:
    """定位元组的固定位置和尾部定义，兼容 items 列表与 prefixItems 两种写法。"""
    items = node.get("items")
    return (
        (items, "additionalItems")
        if isinstance(items, list)
        else (node.get("prefixItems", []), "items")
    )


def array_positions(node: dict[str, Any]) -> Iterator[tuple[str, SchemaNode]]:
    """枚举声明的数组模板、元组固定位置及显式尾部，不生成实际元素索引。"""
    kinds = schema_types(node)
    if kinds and "array" not in kinds:
        return
    prefix, tail_key = array_parts(node)
    items = node.get(tail_key, False)
    for index, definition in enumerate(prefix):
        yield f"[{index}]", definition
    if items is not False:
        yield (f"[{len(prefix)}:]" if prefix else "[*]"), items


def view_structure(node: SchemaNode) -> SchemaNode:
    """将类型列表和引用旁的结构声明转为组合，子定义仍引用原 Schema。"""
    if isinstance(node, bool):
        return node
    ref = node.get("$ref")
    if isinstance(ref, str):
        siblings = {key: value for key, value in node.items() if key in _STRUCTURE_KEYS}
        return {"allOf": [{"$ref": ref}, siblings]} if siblings else node
    kinds = schema_types(node)
    if len(kinds) <= 1:
        return node
    alternatives = {
        "anyOf": [
            {
                "type": kind,
                **{
                    key: node[key]
                    for key in _TYPE_CHILDREN.get(kind, ())
                    if key in node
                },
            }
            for kind in kinds
        ]
    }
    groups = {key: node[key] for key in ("anyOf", "oneOf", "allOf") if key in node}
    return {"allOf": [alternatives, groups]} if groups else alternatives


class SchemaShape(BaseModel):
    """单个节点的属性、数组位置和组合分支连接，不归并不同分支。"""

    declares_object: bool = False
    declares_array: bool = False
    properties: dict[str, SchemaNode] = Field(default_factory=dict)
    items: dict[str, SchemaNode] = Field(default_factory=dict)
    groups: dict[str, list[SchemaNode]] = Field(default_factory=dict)
    references: frozenset[str] = frozenset()

    @classmethod
    def read(cls, node: SchemaNode, references: frozenset[str] = frozenset()) -> Self:
        """读取当前节点的结构；引用因循环停止时，本地声明仍须保留。"""
        if isinstance(node, bool):
            return cls(references=references)
        kinds = schema_types(node)
        properties = (
            node.get("properties", {}) if not kinds or "object" in kinds else {}
        )
        return cls.model_construct(
            references=references,
            declares_object="object" in kinds or not kinds and bool(properties),
            declares_array="array" in kinds
            or not kinds
            and ("items" in node or "prefixItems" in node),
            properties=properties,
            items=dict(array_positions(node)),
            groups={
                keyword: node[keyword]
                for keyword in ("anyOf", "oneOf", "allOf")
                if keyword in node
            },
        )
