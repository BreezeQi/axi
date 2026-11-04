"""按 Schema 声明准备共享字段文档，不参与检索评分或查看范围选择。"""

from __future__ import annotations

from itertools import chain
from threading import RLock
from typing import Any

from pydantic import BaseModel, Field

from axi.output.explanation import SchemaFormatter, Expression
from axi.output.paths import name_segment
from axi.output.schema import (
    MAX_DATA_DEPTH,
    SchemaNode,
    SchemaShape,
    schema_types,
    view_structure,
)


class OutputField(BaseModel):
    """字段文本及其名称、JSONPath 和数据层级，用于检索定位与摘要排序。"""

    name: str
    path: str
    depth: int
    text: str


class FieldDocuments(BaseModel):
    """按声明顺序保存字段文档，同一工具内按完整文本去重。"""

    fields: list[OutputField] = Field(default_factory=list)
    root_text: str = ""


class SchemaDocument:
    """注册定义的共享文档，按需准备字段文本并沿用原 Schema 连接。

    缓存节点释义和可复用的引用解析结果；选择、深度与递归停止状态属于本次查看，
    不随注册定义保存，也不预生成完整展开树。
    """

    def __init__(self, schema: dict[str, Any] | None) -> None:
        self.root = schema if schema is not None else {}
        self._formatter = SchemaFormatter(self.root)
        self._lock = RLock()
        self._fields: FieldDocuments | None = None
        self._aliases: dict[str, tuple[SchemaNode, frozenset[str]]] = {}
        self._shapes: dict[tuple[int, bool], tuple[SchemaNode, SchemaShape]] = {}

    def format(self, node: SchemaNode, *, descriptions: bool = True) -> str:
        with self._lock:
            return self._formatter.format(node, descriptions=descriptions)

    def expression(self, node: SchemaNode) -> Expression:
        with self._lock:
            return self._formatter.expression(node)

    def reference(self, ref: str) -> SchemaNode | None:
        with self._lock:
            return self._formatter.reference(ref)

    def shape(self, definition: SchemaNode, *, for_view: bool = True) -> SchemaShape:
        """复用单层连接；查看时规范化引用和类型列表，字段枚举读取原声明。"""
        with self._lock:
            normalized = (
                for_view
                and isinstance(definition, dict)
                and ("$ref" in definition or len(schema_types(definition)) > 1)
            )
            key = id(definition), normalized
            if key not in self._shapes:
                node, references = (
                    self._resolve(definition)
                    if normalized
                    else (definition, frozenset())
                )
                self._shapes[key] = (
                    definition,
                    SchemaShape.read(node, references),
                )
            return self._shapes[key][1]

    def child_segments(self, definition: SchemaNode) -> list[str]:
        """汇总各分支下一层的声明位置，用于未知路径提示，不枚举更深字段。"""
        properties: dict[str, None] = {}
        items: dict[str, None] = {}
        pending = [(definition, frozenset())]
        visited: set[int] = set()
        while pending:
            definition, ancestors = pending.pop()
            shape = self.shape(definition)
            if shape.references & ancestors or id(definition) in visited:
                continue
            visited.add(id(definition))
            properties.update(
                dict.fromkeys(name_segment(key) for key in shape.properties)
            )
            items.update(dict.fromkeys(shape.items))
            pending.extend(
                reversed(
                    [
                        (branch, ancestors | shape.references)
                        for branches in shape.groups.values()
                        for branch in branches
                    ]
                )
            )
        return [*properties, *items]

    def _resolve(
        self, node: SchemaNode, active_refs: frozenset[str] = frozenset()
    ) -> tuple[SchemaNode, frozenset[str]]:
        """复用引用链已解析的后半段，遇到当前链中的引用时仍停止解析。"""
        node = view_structure(node)
        if not isinstance(node, dict):
            return node, frozenset()
        ref = node.get("$ref")
        if not isinstance(ref, str):
            return node, frozenset()
        if ref in active_refs:
            return node, frozenset({ref})
        cached = self._aliases.get(ref)
        if cached is not None:
            _, references = cached
            if not references & active_refs:
                return cached
        target = self.reference(ref)
        if target is None:
            return node, frozenset()
        resolved, references = self._resolve(target, active_refs | {ref})
        references |= {ref}
        if not references & active_refs:
            self._aliases[ref] = resolved, references
        return resolved, references

    @property
    def fields(self) -> FieldDocuments:
        """在锁内首次准备字段条目，后续 search、grep 直接共用结果。"""
        with self._lock:
            if self._fields is None:
                self._fields = self._prepare_fields()
            return self._fields

    def _prepare_fields(self) -> FieldDocuments:
        """按声明的出现位置枚举字段，不构造查看用的展开树。"""
        documents: dict[str, OutputField] = {}
        root_notes: dict[str, None] = {}
        # 保留不同引用链下可复用的遍历结果，按首次遍历顺序检查。
        covered: dict[tuple[int, str], dict[frozenset[str], None]] = {}
        emitted: set[tuple[int, str]] = set()

        def record_field(
            node: SchemaNode, path: str, data_depth: int, name: str
        ) -> None:
            """同一定义在同一路径只生成一次文本，再按完整文本去重。"""
            identity = id(node), path
            if data_depth > MAX_DATA_DEPTH or identity in emitted:
                return
            emitted.add(identity)
            text = f"{path}: {self.format(node)}"
            if text not in documents:
                documents[text] = OutputField.model_construct(
                    name=name, path=path, depth=data_depth, text=text
                )

        def walk(
            node: SchemaNode,
            path: str = "$",
            data_depth: int = 0,
            active_refs: frozenset[str] = frozenset(),
        ) -> frozenset[str]:
            """返回因祖先引用而停止的引用集合，用于判断同一路径能否复用遍历结果。"""
            if data_depth > MAX_DATA_DEPTH or not isinstance(node, dict):
                return frozenset()
            if not data_depth and isinstance(note := node.get("description"), str):
                root_notes.setdefault(note)
            if not any(
                key in node
                for key in (
                    "$ref",
                    "properties",
                    "items",
                    "prefixItems",
                    "anyOf",
                    "oneOf",
                    "allOf",
                )
            ):
                # 没有子结构的节点无需继续遍历，也不用记录引用链下的遍历结果。
                return frozenset()
            coverage_key = id(node), path
            histories = covered.setdefault(coverage_key, {})
            if active_refs in histories:
                return active_refs
            for previous in histories:
                if previous <= active_refs:
                    return previous
            blocked: set[str] = set()
            ref = node.get("$ref")
            if isinstance(ref, str):
                if ref in active_refs:
                    blocked.add(ref)
                elif (target := self.reference(ref)) is not None:
                    blocked.update(walk(target, path, data_depth, active_refs | {ref}))
            shape = self.shape(node, for_view=False)
            properties = (
                (name_segment(name), name, child)
                for name, child in shape.properties.items()
            )
            items = ((segment, "", child) for segment, child in shape.items.items())
            for segment, child_name, child in chain(properties, items):
                # 属性和数组元素进入新的数据位置；引用与组合分支仍使用当前路径。
                child_path = path + segment
                record_field(child, child_path, data_depth + 1, child_name)
                blocked.update(walk(child, child_path, data_depth + 1, active_refs))
            for branches in shape.groups.values():
                for branch in branches:
                    blocked.update(walk(branch, path, data_depth, active_refs))
            # 子树内部的循环每次都会停止；只有来自祖先的停止条件会随当前引用链变化。
            # 记录这些条件，后续满足时即可复用同一路径已枚举的位置。
            blockers = frozenset(blocked & active_refs)
            histories.setdefault(blockers)
            return blockers

        walk(self.root)
        return FieldDocuments.model_construct(
            fields=list(documents.values()),
            root_text=" ".join(root_notes),
        )
