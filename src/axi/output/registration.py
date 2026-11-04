"""将注册的输出模型转换为 Schema，定义由 Registry 持有。"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

OUTPUT_SCHEMA_META_KEY = "axi/output_schema"

type SchemaInput = type[BaseModel] | dict[str, Any]


def normalize_schema(value: SchemaInput | None) -> dict[str, Any] | None:
    """模型类转为 JSON Schema，字典直接沿用，不校验 Schema 约束。"""
    if value is None:
        return None
    if isinstance(value, type) and issubclass(value, BaseModel):
        return value.model_json_schema(mode="serialization")
    if isinstance(value, dict):
        return value
    raise TypeError("output_schema must be a Pydantic model class or JSON Schema dict")
