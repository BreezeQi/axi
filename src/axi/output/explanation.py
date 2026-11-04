"""简化重复的类型组合，再生成类型与节点自身说明。"""

from __future__ import annotations

from functools import cached_property
from typing import Any

from pydantic import BaseModel

from axi.output.schema import (
    MAX_DATA_DEPTH,
    SchemaNode,
    array_parts,
    array_positions,
    schema_pointer,
    schema_types,
)

_TYPES = {
    "string": "str",
    "number": "float",
    "integer": "int",
    "boolean": "bool",
    "object": "dict",
    "array": "list",
    "null": "None",
}


class Expression(BaseModel, frozen=True):
    kind: str
    members: tuple[Expression, ...] = ()
    notes: tuple[str, ...] = ()

    @cached_property
    def text(self) -> str:
        if self.kind == "list":
            label = f"list[{self.members[0].text}]"
        elif self.kind == "|":
            label = " | ".join(member.text for member in self.members)
        elif self.kind in ("anyOf", "oneOf", "allOf", "items"):
            label = f"{self.kind}({', '.join(member.text for member in self.members)})"
        else:
            label = self.kind
        return "；".join((label, *self.notes))


def compose(kind: str, members: list[Expression]) -> Expression:
    """展开无独立说明的同类组合并去重；oneOf 保留原分支和嵌套。"""
    if kind != "oneOf":
        members = [
            child
            for member in members
            for child in (
                member.members
                if member.kind == kind and not member.notes
                else (member,)
            )
        ]
        if len(members) == 1:
            return members[0]
        unique: dict[str, Expression] = {}
        for member in members:
            unique.setdefault(member.text, member)
        members = list(unique.values())
        if len(members) == 1:
            return members[0]
    return Expression(kind=kind, members=tuple(members))


class SchemaFormatter:
    """生成并复用折叠释义，复用引用结果前核对当前引用链。"""

    def __init__(self, root: dict[str, Any]) -> None:
        self._root = root
        self._expressions: dict[tuple[int, bool], tuple[SchemaNode, Expression]] = {}
        self._targets: dict[str, SchemaNode | None] = {}
        # 记录会影响释义的引用；当前引用链与它们相交时，不能直接复用结果。
        self._memo: dict[tuple[str, bool, int], tuple[Expression, frozenset[str]]] = {}

    def reference(self, ref: str) -> SchemaNode | None:
        """缓存本地引用目标，供释义、字段枚举和查看共用。"""
        if ref not in self._targets:
            self._targets[ref] = schema_pointer(self._root, ref)
        return self._targets[ref]

    def format(self, node: SchemaNode, *, descriptions: bool = True) -> str:
        return self.expression(node, descriptions=descriptions).text

    def expression(self, node: SchemaNode, *, descriptions: bool = True) -> Expression:
        """复用已生成的释义表达式，不反向解析格式化后的文字。"""
        key = id(node), descriptions
        cached = self._expressions.get(key)
        if cached is None:
            expression, _ = self._build(node, descriptions)
            # 同时持有定义，防止临时组合被回收后，其对象 ID 被其他定义复用。
            cached = node, expression
            self._expressions[key] = cached
        return cached[1]

    def _reference(
        self,
        ref: str,
        descriptions: bool,
        active_refs: frozenset[str],
        depth: int,
    ) -> tuple[Expression, frozenset[str]]:
        if ref in active_refs:
            return Expression(kind=f"$ref({ref})"), frozenset({ref})
        key = (ref, descriptions, depth)
        cached = self._memo.get(key)
        if cached is not None and not cached[1] & active_refs:
            return cached
        target = self.reference(ref)
        if target is None:
            return Expression(kind=f"$ref({ref})"), frozenset({ref})
        expression, references = self._build(
            target, descriptions, active_refs | {ref}, depth
        )
        references = references | {ref}
        if not references & active_refs:
            self._memo[key] = expression, references
        return expression, references

    def _array_type(
        self, node: dict[str, Any], active_refs: frozenset[str], depth: int
    ) -> tuple[Expression, frozenset[str]]:
        elements = [definition for _, definition in array_positions(node)]
        if not elements:
            _, tail_key = array_parts(node)
            elements = [node.get(tail_key, True)]
        members = []
        references: frozenset[str] = frozenset()
        for element in elements:
            item, accessed = self._build(element, False, active_refs, depth + 1)
            members.append(item)
            references |= accessed
        return compose("|", members), references

    def _build(
        self,
        node: SchemaNode,
        descriptions: bool,
        active_refs: frozenset[str] = frozenset(),
        depth: int = 0,
    ) -> tuple[Expression, frozenset[str]]:
        """生成当前节点释义并记录引用依赖；这里只沿数组元素增加数据层级。"""
        if depth > MAX_DATA_DEPTH:
            return Expression(kind="Any"), frozenset()
        if isinstance(node, bool):
            return Expression(kind="Any" if node else "false"), frozenset()
        parts: list[Expression] = []
        references: frozenset[str] = frozenset()
        ref = node.get("$ref")
        if isinstance(ref, str):
            expression, references = self._reference(
                ref, descriptions, active_refs, depth
            )
            parts.append(expression)
        kinds = schema_types(node)
        has_items = "items" in node or "prefixItems" in node
        if "array" in kinds or not kinds and has_items:
            item_type, accessed = self._array_type(node, active_refs, depth)
            references |= accessed
        labels = [
            Expression(kind="list", members=(item_type,))
            if kind == "array"
            else Expression(kind=label)
            for kind, label in _TYPES.items()
            if kind in kinds
        ]
        if labels:
            parts.append(compose("|", labels))
        if not kinds and has_items:
            parts.append(Expression(kind="items", members=(item_type,)))
        for keyword in ("anyOf", "oneOf", "allOf"):
            if keyword not in node:
                continue
            members = []
            for branch in node[keyword]:
                member, accessed = self._build(branch, descriptions, active_refs, depth)
                members.append(member)
                references |= accessed
            parts.append(compose(keyword, members))
        expression = compose("allOf", parts) if parts else Expression(kind="Any")
        note = node.get("description")
        if (
            descriptions
            and isinstance(note, str)
            and note
            and note not in expression.notes
        ):
            # 按完整说明去重，不拆分其中的标点或重复业务词。
            expression = Expression(
                kind=expression.kind,
                members=expression.members,
                notes=(*expression.notes, note),
            )
        return expression, references
