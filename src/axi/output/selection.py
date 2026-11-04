"""输出定义的 JSONPath 选择树与一级展开规则。"""

from __future__ import annotations

from pydantic import BaseModel, Field, ValidationError

from axi.models import DescribeOptions, FieldPaths
from axi.output.paths import name_segment, parse_path, segment_name
from axi.output.schema import SchemaNode


class SelectionNode(BaseModel):
    """记录是否选中当前子树，子选择按规范化路径片段保存。"""

    selected: bool = False
    children: dict[str, SelectionNode] = Field(default_factory=dict)


type SelectionTrees = tuple[SelectionNode, ...] | None


def normalize_paths(value: FieldPaths) -> SelectionNode | None:
    """将路径合成选择树，保留父子范围并集，结果不受路径顺序影响。"""
    if value is None:
        return None
    paths = [parse_path(path) for path in value]
    tree = SelectionNode()
    for path in sorted(paths):
        branch = tree
        for segment in path:
            if segment not in branch.children:
                branch.children[segment] = SelectionNode()
            branch = branch.children[segment]
        branch.selected = True
    return tree


class DescribeSelection(DescribeOptions):
    """单次 describe 共用的参数和选择树。"""

    tree: SelectionNode | None

    @classmethod
    def prepare(
        cls,
        fields: FieldPaths = None,
        *,
        schema: bool = False,
    ) -> DescribeSelection:
        try:
            selection = cls(fields=fields, schema=schema, tree=None)
        except ValidationError as exc:
            if exc.errors()[0]["loc"][0] == "schema":
                raise ValueError("schema must be a boolean.") from None
            raise ValueError("fields must be a JSONPath array or null.") from None
        if selection.full_schema and selection.fields is not None:
            raise ValueError(
                "schema=true cannot be combined with fields. "
                "Omit fields to view the complete output schema."
            )
        selection.tree = normalize_paths(selection.fields)
        return selection


def node_selection(
    trees: SelectionTrees,
    expand: bool,
    included: bool,
) -> tuple[SelectionTrees, bool, bool]:
    """显式选中节点展开一级，同时保留选中的深层路径。"""
    if trees is not None and any(tree.selected for tree in trees):
        expand = True
        included = True
    if included and trees is not None and not any(tree.children for tree in trees):
        trees = None
    return trees, expand, trees is None or included


def _child_keys(segment: str) -> tuple[str, ...]:
    """元组固定位置和尾部同时接受自身片段与 [*] 通配选择。"""
    return (
        ("[*]", segment)
        if segment.startswith("[") and segment[1:2].isdigit()
        else (segment,)
    )


def selects_child(trees: SelectionTrees, segment: str, included: bool) -> bool:
    """按规范化片段判断子节点是否被选中，区分带引号的键名与数组位置。"""
    return (
        trees is None
        or included
        or any(key in tree.children for tree in trees for key in _child_keys(segment))
    )


def select_properties(
    properties: dict[str, SchemaNode], trees: SelectionTrees, included: bool
) -> dict[str, SchemaNode]:
    """单属性选择直接定位，多属性选择保持 Schema 声明顺序。"""
    if trees is None or included:
        return properties
    keys = {key for tree in trees for key in tree.children}
    if len(keys) == 1:
        name = segment_name(next(iter(keys)))
        return (
            {name: properties[name]} if name is not None and name in properties else {}
        )
    return {
        name: definition
        for name, definition in properties.items()
        if name_segment(name) in keys
    }


def child_selection(
    trees: SelectionTrees,
    segment: str,
    expand: bool,
    included: bool,
) -> tuple[SelectionTrees, bool, bool]:
    """传递子选择；属性下的容器折叠，数组位置保留元素模板。"""
    child = child_trees(trees or (), segment)
    included = trees is None or included
    return child or None, expand and _child_keys(segment)[0] == "[*]", included


def child_trees(
    trees: tuple[SelectionNode, ...], segment: str
) -> tuple[SelectionNode, ...]:
    """汇集通配与固定位置的子选择，保留原选择节点，不复制或覆盖子树。"""
    keys = _child_keys(segment)
    return tuple(
        tree.children[key] for tree in trees for key in keys if key in tree.children
    )
