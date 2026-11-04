"""引用回归共用的小型 Schema 和读取计数器。"""

import json
from collections.abc import ItemsView, Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import Mock, patch

import axi.output.schema as schema_module
from axi.output.schema import SchemaNode
from tests.support import arr, node, obj, ref


class CountingSchema(dict[str, Any]):
    """统计定义读取次数，不依赖被测实现的内部模型。"""

    reads = 0

    def get(self, key: str, default: Any = None) -> Any:
        CountingSchema.reads += 1
        return super().get(key, default)

    def items(self) -> ItemsView[str, Any]:
        CountingSchema.reads += 1
        return super().items()


def counted_schema(schema: dict[str, Any]) -> CountingSchema:
    """经 JSON 编解码消除 Python 对象共享，再包装节点以统计读取次数。"""
    return json.loads(json.dumps(schema), object_hook=CountingSchema)


@contextmanager
def count_pointer_lookups() -> Iterator[Mock]:
    """独立统计 JSON Pointer 定位次数，与后续定义读取次数分开验证。"""
    lookup = Mock(wraps=schema_module.schema_pointer)
    with patch("axi.output.explanation.schema_pointer", lookup):
        yield lookup


def shared_schema(levels: int, keyword: str, array: bool) -> dict:
    leaf = obj({"value": node("number", "数值")})
    definitions = {side + "0": arr(leaf) if array else leaf for side in ("A", "B")}
    for level in range(1, levels + 1):
        for side in ("A", "B"):
            definitions[f"{side}{level}"] = {
                keyword: [ref(f"A{level - 1}"), ref(f"B{level - 1}")]
            }
    return {"$defs": definitions, **ref(f"A{levels}")}


def alias_definitions(count: int, value: SchemaNode) -> dict[str, SchemaNode]:
    definitions = {"Value": value}
    for level in range(count):
        definitions[f"Alias{level}"] = ref(f"Alias{level - 1}" if level else "Value")
    return definitions
