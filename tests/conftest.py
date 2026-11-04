"""发现、查看及各入口契约测试共用的输出定义。"""

from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any

import pytest
from mcp.types import CallToolResult
from pydantic import BaseModel, Field, RootModel, field_serializer

from axi.models import ToolMeta
from axi.output.registration import normalize_schema
from axi.registry import Registry
from tests.support import make_tool, node


@pytest.fixture
def measurement_registry() -> Registry:
    registry = Registry()
    registry.register(
        make_tool(
            description="Read measurements",
            schema={"properties": {"speed": node("number", "oldvelocity")}},
        )
    )
    return registry


@pytest.fixture
def nested_meta() -> ToolMeta:
    """提供三级字段与记录数组，验证展开深度和各入口展示一致性。"""

    class WorkTime(BaseModel):
        duration: float = Field(description="作业时长，小时")

    class Summary(BaseModel):
        forwardMileage: float = Field(description="前进里程，单位米")
        workTime: WorkTime = Field(description="作业时间明细")
        unexplained: WorkTime

    class Output(BaseModel):
        plateNumber: str = Field(description="车牌/车辆编号")
        summaryStatistic: Summary = Field(description="查询时间窗内的车辆汇总统计")
        history: list[Summary] | None = Field(description="历史统计")

    schema = normalize_schema(Output)
    return make_tool(schema=schema)


@pytest.fixture
def mcp_call() -> Callable[..., Awaitable[CallToolResult]]:
    """按 SDK 所需的请求上下文调用本地 MCP 处理器。"""
    from mcp.server.lowlevel import Server
    from mcp.types import CallToolRequestParams
    from mcp.types.version import LATEST_HANDSHAKE_VERSION

    async def call(
        server: Server, name: str, arguments: dict[str, Any]
    ) -> CallToolResult:
        return await server._request_handlers["tools/call"].handler(
            SimpleNamespace(request=None, protocol_version=LATEST_HANDSHAKE_VERSION),
            CallToolRequestParams(name=name, arguments=arguments),
        )

    return call


@pytest.fixture(params=["model", "nested", "array", "root", "dictionary", "null"])
def python_result(request: pytest.FixtureRequest) -> tuple[Any, type[BaseModel], Any]:
    """输出示例沿用 Python 模式：保留模型字段名，省略模型中的 None 字段。"""

    if request.param == "dictionary":
        value = {"identifier": 7, "optional": None}
        return value, RootModel[dict[str, int | None]], value
    if request.param == "null":
        return RootModel[None](None), RootModel[None], None

    class Record(BaseModel):
        identifier: int = Field(serialization_alias="id")
        label: str = "ready"
        optional: str | None = None

        @field_serializer("label", when_used="json")
        def serialize_label(self, value: str) -> str:
            return value.upper()

    row = Record(identifier=7)
    expected = {"identifier": 7, "label": "ready"}
    if request.param == "nested":

        class Nested(BaseModel):
            row: Record

        return {"row": row}, Nested, {"row": expected}
    if request.param == "array":
        return [row], RootModel[list[Record]], [expected]
    if request.param == "root":
        return RootModel[list[Record]]([row]), RootModel[list[Record]], [expected]
    return row, Record, expected
