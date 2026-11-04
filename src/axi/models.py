"""axi 核心数据模型。"""

from enum import Enum
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, Field, StrictStr, model_validator

type FieldPaths = Annotated[list[StrictStr], Field(strict=True)] | None
DEFAULT_MAX_FIELDS = 8


class DescribeOptions(BaseModel):
    """各入口共用的查看参数；完整定义与局部字段选择互斥。"""

    full_schema: bool = Field(
        default=False,
        alias="schema",
        strict=True,
        description="Return the complete registered output JSON Schema without executing the tool. "
        "Cannot be combined with non-null fields, including []. "
        "Default false preserves input or local output documentation views.",
    )

    fields: FieldPaths = Field(
        default=None,
        description='JSONPath array selecting output definitions, e.g. ["$.A.x", "$.rows[*].id"]. '
        "Supports $, .name, [\"literal name\"] or ['literal name'], [*], fixed tuple [n], "
        "and declared tuple tail [n:]. Selected containers show their direct fields; "
        "nested containers retain their compact type and description. "
        "Omitted/null shows tool documentation and complete inputs unless schema=true. "
        'Use ["$"] for the output root directory; [] returns an empty output view without inputs. '
        "This selects documentation only and never changes tool results.",
    )


def allowed_types(prop: dict) -> set[str]:
    """提取属性 schema 允许的原始类型，覆盖 type 为字符串/列表及 anyOf 分支（递归）。"""
    types: set[str] = set()
    t = prop.get("type")
    if isinstance(t, str):
        types.add(t)
    elif isinstance(t, list):
        types.update(x for x in t if isinstance(x, str))
    for branch in prop.get("anyOf", []):
        if isinstance(branch, dict):
            types |= allowed_types(branch)
    return types


class ToolSource(str, Enum):
    """工具来源类型。"""

    NATIVE = "native"
    MCP = "mcp"


class ToolInfo(BaseModel):
    """工具身份与业务说明，由目录摘要、注册定义和查看结果共用。"""

    name: str = Field(min_length=1, description="工具名称")
    server: str | None = Field(
        default=None, description="所属 server 名称（MCP server 或原生工具模块名）"
    )
    description: str = Field(description="工具描述")
    source: ToolSource = Field(description="工具来源")

    @property
    def full_name(self) -> str:
        """完整工具名，含 server 前缀。"""
        if self.server:
            return f"{self.server}/{self.name}"
        return self.name


class ToolMeta(ToolInfo):
    """注册的工具定义，保存入参 Schema 和仅作文档的输出 Schema。"""

    input_schema: dict[str, Any] = Field(
        default_factory=dict, description="输入参数 JSON Schema"
    )
    output_example: Any | None = Field(
        default=None, description="输出示例（原生工具可选提供）"
    )
    output_schema: dict[str, Any] | None = Field(
        default=None,
        description="完整正常成功业务值的 JSON Schema；仅作发现和文档，不校验返回值",
    )

    @model_validator(mode="after")
    def _validate_fields(self) -> Self:
        if "/" in self.name:
            raise ValueError(f"Tool name must not contain '/': {self.name!r}")
        if self.source == ToolSource.MCP and not self.server:
            raise ValueError("MCP tool must have a server name")
        return self


class ToolDescription(ToolInfo):
    """describe 返回的工具详情，输出字段以文档字符串展示。"""

    input_schema: dict[str, Any] | None = None
    output_example: Any | None = Field(
        default=None, description="输出示例（原生工具可选提供）"
    )
    output_schema_status: Literal["registered", "not_registered"] | None = None
    output_schema: dict[str, Any] | None = Field(
        default=None,
        description="Complete registered output JSON Schema, only for schema=true",
    )
    output_fields: str | None = Field(
        default=None, description="Selected output documentation text"
    )
    output_type: str | None = None
    unknown_fields: list[str] | None = None
    available_fields: dict[str, list[str]] | None = None


class ToolDescriptionError(BaseModel):
    """批量 describe 中单个工具的查找或文档错误。"""

    error: str


class ResultEnvelope(BaseModel):
    """统一结果信封基类。"""

    status: Literal["success", "error"] = Field(description="success 或 error")
    data: Any | None = Field(default=None, description="执行结果")
    error: str | None = Field(default=None, description="错误信息")

    @classmethod
    def success(cls, data: Any) -> Self:
        return cls(status="success", data=data)

    @classmethod
    def fail(cls, error: str) -> Self:
        return cls(status="error", error=error)


class RunResult(ResultEnvelope):
    """axi run 的统一输出信封。"""


class SearchResult(BaseModel):
    """搜索结果条目。

    结果按相关性排序，顺序本身即信号；内部 BM25、Embedding 和 RRF
    分数不属于返回契约，不向 Agent 暴露。
    """

    name: str = Field(description="工具完整名称")
    description: str = Field(description="工具描述")
    source: ToolSource = Field(description="工具来源")
    matched_fields: list[str] = Field(
        default_factory=list,
        description="命中字段文档：JSONPath: 类型；自身说明",
    )

    @classmethod
    def from_meta(cls, meta: ToolMeta) -> Self:
        """从工具定义生成发现摘要，不持有入参或输出 Schema。"""
        return cls(
            name=meta.full_name,
            description=meta.description,
            source=meta.source,
        )
