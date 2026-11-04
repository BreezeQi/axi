"""根据查看范围组装工具详情与输出字段文档。"""

from __future__ import annotations

from collections.abc import Callable

from axi.models import ToolDescription, ToolInfo, ToolMeta
from axi.output.documents import SchemaDocument
from axi.output.selection import DescribeSelection
from axi.output.view import render_view, select_view


class ToolDescriber:
    """同一批次复用详情结果，重复响应独立复制，不共享可修改的返回字段。"""

    def __init__(
        self,
        selection: DescribeSelection,
        documentation: Callable[[ToolMeta], SchemaDocument],
    ) -> None:
        self._selection = selection
        self._documentation = documentation
        self._descriptions: dict[str, ToolDescription] = {}

    def describe(self, meta: ToolMeta) -> ToolDescription:
        """生成单项详情，将文档异常统一交给各入口按工具隔离。"""
        try:
            cached = self._descriptions.get(meta.full_name)
            if cached is not None:
                return cached.model_copy(deep=True)
            description = self._describe(meta)
            self._descriptions[meta.full_name] = description
            return description
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(
                f"Cannot describe {meta.full_name}: {exc}. "
                "Check its registered output_schema."
            ) from exc

    def _describe(self, meta: ToolMeta) -> ToolDescription:
        """按视图读取完整定义或组装紧凑文档，默认仅返回入参。"""
        selection = self._selection
        # 显式选择导出字段；model_dump 复制 Schema 容器，避免响应污染注册定义。
        include = set(ToolInfo.model_fields)
        if selection.full_schema:
            include.add("output_schema")
        elif selection.fields is None:
            include.add("input_schema")
        else:
            include.add("output_example")
        out = ToolDescription.model_construct(
            **meta.model_dump(
                include=include,
                exclude_none=True,
            )
        )
        if selection.fields is None and not selection.full_schema:
            return out
        schema = meta.output_schema
        out.output_schema_status = (
            "registered" if schema is not None else "not_registered"
        )
        if schema is None or selection.full_schema:
            return out
        document = self._documentation(meta)
        view = select_view(document, selection.tree)
        out.output_fields = render_view(view)
        out.output_type = document.format(schema, descriptions=False)
        if view.unknown:
            out.unknown_fields = view.unknown
            out.available_fields = view.available
        return out
