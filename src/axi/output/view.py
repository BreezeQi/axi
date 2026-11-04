"""先选择字段并展开一级，再独立渲染终点释义。"""

from __future__ import annotations

from pydantic import BaseModel, Field

from axi.output.documents import SchemaDocument
from axi.output.explanation import Expression, compose
from axi.output.paths import name_segment
from axi.output.schema import MAX_DATA_DEPTH, SchemaNode
from axi.output.selection import (
    SelectionNode,
    SelectionTrees,
    child_selection,
    node_selection,
    select_properties,
    selects_child,
)


class _Terminal(BaseModel):
    definition: SchemaNode


class _Branches(BaseModel):
    keyword: str
    members: list[_SelectedFields]


class _Array(BaseModel):
    items: dict[str, _SelectedFields] = Field(default_factory=dict)


type _SelectedFields = dict[str, _SelectedFields] | _Array | _Terminal | _Branches


class _SelectedView(BaseModel):
    model_config = {"arbitrary_types_allowed": True}

    document: SchemaDocument
    fields: _SelectedFields
    unknown: list[str] = Field(default_factory=list)
    available: dict[str, list[str]] = Field(default_factory=dict)


class _UnknownPath(BaseModel):
    edge: tuple[int, str]
    path: str
    parent: str
    definition: SchemaNode


class _SelectionResult(BaseModel):
    """一个选择片段，包含字段结构、相对路径诊断和选择时读取的引用。"""

    fields: _SelectedFields | None = None
    matched: set[tuple[int, str]] = Field(default_factory=set)
    unknown: dict[tuple[tuple[int, str], str, str, int], _UnknownPath] = Field(
        default_factory=dict
    )
    references: frozenset[str] = frozenset()

    def add_unknown(self, entry: _UnknownPath) -> None:
        key = entry.edge, entry.path, entry.parent, id(entry.definition)
        self.unknown.setdefault(key, entry)

    def include(self, child: _SelectionResult, segment: str = "") -> None:
        self.references |= child.references
        self.matched.update(child.matched)
        for entry in child.unknown.values():
            if segment:
                entry = entry.model_copy(
                    update={
                        "path": segment + entry.path,
                        "parent": segment + entry.parent,
                    }
                )
            self.add_unknown(entry)

    def record_unknowns(
        self,
        properties: dict[str, SchemaNode],
        items: dict[str, SchemaNode],
        sources: tuple[SelectionNode, ...],
        definition: SchemaNode,
    ) -> None:
        """记录未匹配的选择片段，未知路径相对于当前节点保存。"""
        known = {name_segment(key) for key in properties} | items.keys()
        if items:
            known.add("[*]")
        for source in sources:
            for key, child in source.children.items():
                identity = id(source), key
                if key in known:
                    self.matched.add(identity)
                else:
                    for suffix in _selected_suffixes(child):
                        self.add_unknown(
                            _UnknownPath.model_construct(
                                edge=identity,
                                path=key + suffix,
                                parent="",
                                definition=definition,
                            )
                        )

    def to_view(self, document: SchemaDocument) -> _SelectedView:
        """汇总适用分支的匹配结果，生成完整未知路径及最近父节点的提示。"""
        unknown = [
            entry for entry in self.unknown.values() if entry.edge not in self.matched
        ]
        available: dict[str, dict[str, None]] = {}
        seen: set[tuple[str, int]] = set()
        directories: dict[int, list[str]] = {}
        for entry in unknown:
            parent = "$" + entry.parent
            node_id = id(entry.definition)
            identity = parent, node_id
            if identity in seen:
                continue
            seen.add(identity)
            if node_id not in directories:
                directories[node_id] = document.child_segments(entry.definition)
            available.setdefault(parent, {}).update(
                (parent + segment, None) for segment in directories[node_id]
            )
        return _SelectedView.model_construct(
            document=document,
            fields=self.fields if self.fields is not None else {},
            unknown=list(dict.fromkeys("$" + entry.path for entry in unknown)),
            available={parent: list(choices) for parent, choices in available.items()},
        )


def _branches(keyword: str, members: list[_SelectedFields]) -> _Branches:
    """保留同一数据位置的组合关系，不将不同分支的字段拼成一个对象。"""
    return _Branches.model_construct(keyword=keyword, members=members)


class _FieldSelector:
    def __init__(self, document: SchemaDocument) -> None:
        self.document = document
        self._memo: dict[
            tuple[int, tuple[int, ...] | None, bool, bool, int],
            dict[frozenset[str], _SelectionResult],
        ] = {}

    def select(
        self,
        definition: SchemaNode,
        selection: SelectionTrees,
        *,
        expand: bool = True,
        included: bool = False,
        ancestor_refs: frozenset[str] = frozenset(),
        data_depth: int = 0,
    ) -> _SelectionResult:
        selection, expand, included = node_selection(selection, expand, included)
        if (not expand and selection is None) or data_depth >= MAX_DATA_DEPTH:
            return _SelectionResult.model_construct(fields=self._terminal(definition))
        # 诊断随选择片段保存为相对路径，片段可在不同出现位置复用。
        key = (
            id(definition),
            tuple(map(id, selection)) if selection is not None else None,
            included,
            expand,
            data_depth,
        )
        if key not in self._memo:
            self._memo[key] = {}
        cache = self._memo[key]
        for history, view in cache.items():
            # 只需比较本次读取的引用；其他祖先引用不会改变该片段的展开结果。
            if not (history ^ ancestor_refs) & view.references:
                return view
        result = self._select(
            definition,
            selection,
            ancestor_refs=ancestor_refs,
            expand=expand,
            included=included,
            data_depth=data_depth,
        )
        cache[ancestor_refs] = result
        return result

    def _terminal(self, definition: SchemaNode) -> _Terminal:
        return _Terminal.model_construct(definition=definition)

    def _select(
        self,
        definition: SchemaNode,
        selection: SelectionTrees,
        *,
        ancestor_refs: frozenset[str],
        expand: bool,
        included: bool,
        data_depth: int,
    ) -> _SelectionResult:
        shape = self.document.shape(definition)
        result = _SelectionResult(references=shape.references)
        if shape.references & ancestor_refs:
            if included:
                result.fields = self._terminal(definition)
            if selection is not None:
                result.record_unknowns({}, {}, selection, definition)
            return result
        ancestor_refs |= shape.references
        # 自动展开只到直接子字段；显式深层路径仍可继续进入。
        automatic = included and expand
        properties = select_properties(shape.properties, selection, automatic)
        items = {
            segment: child
            for segment, child in shape.items.items()
            if selects_child(selection, segment, automatic)
        }
        if selection is not None:
            result.record_unknowns(properties, items, selection, definition)

        def select_child(child: SchemaNode, segment: str) -> _SelectedFields:
            trees, child_expand, child_included = child_selection(
                selection, segment, expand, included
            )
            # 显式数据路径可以进入因递归保护而自动折叠的节点。
            selected = self.select(
                child,
                trees,
                expand=child_expand,
                included=child_included,
                ancestor_refs=frozenset() if trees is not None else ancestor_refs,
                data_depth=data_depth + 1,
            )
            result.include(selected, segment)
            return selected.fields if selected.fields is not None else {}

        def select_branch(child: SchemaNode) -> _SelectedFields | None:
            """组合分支不进入新的数据位置，沿用展开状态与引用链。"""
            selected = self.select(
                child,
                selection,
                expand=expand,
                included=included,
                ancestor_refs=ancestor_refs,
                data_depth=data_depth,
            )
            result.include(selected)
            member = selected.fields
            if not included and (
                member == {} or isinstance(member, _Array) and not member.items
            ):
                return None
            return member

        parts: list[_SelectedFields] = []
        if properties:
            parts.append(
                {
                    name: select_child(child, name_segment(name))
                    for name, child in properties.items()
                }
            )
        if items:
            parts.append(
                _Array.model_construct(
                    items={
                        segment: select_child(child, segment)
                        for segment, child in items.items()
                    }
                )
            )
        expanded = bool(parts)
        for keyword, branches in shape.groups.items():
            members = [
                member
                for child in branches
                if (member := select_branch(child)) is not None
            ]
            if members:
                expanded |= any(not isinstance(member, _Terminal) for member in members)
                parts.append(_branches(keyword, members))
        if expanded:
            result.fields = parts[0] if len(parts) == 1 else _branches("allOf", parts)
        elif included:
            result.fields = self._terminal(definition)
        elif shape.declares_object:
            result.fields = {}
        elif shape.declares_array:
            result.fields = _Array()
        return result


def _selected_suffixes(tree: SelectionNode) -> list[str]:
    return ([""] if tree.selected else []) + [
        segment + suffix
        for segment, child in tree.children.items()
        for suffix in _selected_suffixes(child)
    ]


def select_view(
    document: SchemaDocument,
    tree: SelectionNode | None,
) -> _SelectedView:
    if tree is not None and not (tree.selected or tree.children):
        return _SelectedView.model_construct(document=document, fields={})
    selector = _FieldSelector(document)
    selected = selector.select(
        document.root,
        (tree,) if tree is not None else None,
    )
    return selected.to_view(document)


def render_view(view: _SelectedView) -> str:
    """渲染已选结构并复用终点释义，不再处理选择或祖先状态。"""
    formatted: dict[int, Expression] = {}

    def format_node(node: _SelectedFields) -> Expression:
        identity = id(node)
        if identity in formatted:
            return formatted[identity]
        if isinstance(node, _Terminal):
            result = view.document.expression(node.definition)
        elif isinstance(node, _Branches):
            result = compose(
                node.keyword, [format_node(child) for child in node.members]
            )
        elif isinstance(node, _Array):
            result = Expression.model_construct(
                kind="["
                + ", ".join(
                    (segment + ": " if segment != "[*]" else "")
                    + format_node(child).text
                    for segment, child in node.items.items()
                )
                + "]",
            )
        else:
            result = Expression.model_construct(
                kind="{"
                + ", ".join(
                    name_segment(key).removeprefix(".") + ": " + format_node(child).text
                    for key, child in node.items()
                )
                + "}",
            )
        formatted[identity] = result
        return result

    return format_node(view.fields).text
