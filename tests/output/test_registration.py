"""验证 Pydantic 转换和注册后的字段文档保留原声明含义。"""

from copy import deepcopy
from typing import Annotated, Any

import pytest
from pydantic import BaseModel, Field, RootModel

from axi.executor import Executor
from axi.output.registration import normalize_schema
from axi.providers.native import register_tool
from tests.support import (
    composition,
    make_registry,
    make_tool,
    node,
    obj,
    prepare_field_documents,
    show,
    view_text,
)


def test_pydantic_conversion_and_instance_rejection() -> None:
    class Row(BaseModel):
        value: float = Field(description="数值")

    class Output(BaseModel):
        identifier: str = Field(alias="id")
        rows: list[Row]

    schema = normalize_schema(Output)
    assert show(schema, fields=["$", "$.rows"]) == view_text(
        {
            "id": "str",
            "rows": [{"value": "float；数值"}],
        }
    )
    assert "$.rows[*].value" in [
        field.path for field in prepare_field_documents(schema).fields
    ]
    with pytest.raises(TypeError):
        normalize_schema(Output.model_construct())


@pytest.mark.parametrize("definition", ["model", "schema"])
def test_columnar_schema_keeps_the_complete_result_and_nested_cell_paths(
    definition: str,
) -> None:
    class SummaryStatistic(BaseModel):
        forwardMileage: float = Field(description="前进里程，单位米")

    class Constants(BaseModel):
        modelName: str | None = Field(default=None, description="各行相同的车型名称")
        plateNumber: str | None = Field(default=None, description="各行相同的车牌")
        summaryStatistic: SummaryStatistic | None = None

    class Output(BaseModel):
        row_count: int = Field(description="本页记录数")
        constants: Constants
        columns: list[str] = Field(description="非常量列名，与各行单元格按位置对应")
        values: list[list[str | SummaryStatistic | None]]
        page: int
        page_size: int
        total: int | None
        has_more: bool | None

    raw = {
        "row_count": 2,
        "constants": {"modelName": "示例车型"},
        "columns": ["plateNumber", "summaryStatistic"],
        "values": [
            ["DEMO-001", {"forwardMileage": 1200}],
            ["DEMO-002", {"forwardMileage": 800}],
        ],
        "page": 1,
        "page_size": 100,
        "total": 2,
        "has_more": False,
    }
    schema = (
        Output
        if definition == "model"
        else Output.model_json_schema(mode="serialization")
    )
    meta = register_tool(lambda: raw, name="columnar_read", output_schema=schema)
    registry = make_registry(meta)
    assert Executor(registry).run(meta.name, {}).data is raw
    paths = {field.path for field in prepare_field_documents(meta.output_schema).fields}
    assert paths >= {
        "$.constants.modelName",
        "$.constants.summaryStatistic.forwardMileage",
        "$.values[*][*].forwardMileage",
        "$.row_count",
    }
    assert "$.summaryStatistic.forwardMileage" not in paths
    assert "$.values[*][*].forwardMileage: float；前进里程，单位米" in (
        registry.search("前进里程", tool=meta.name)[0].matched_fields
    )
    assert registry.grep(r"^\$\.values\[\*\]\[\*\]\.forwardMileage:")[
        0
    ].matched_fields == ["$.values[*][*].forwardMileage: float；前进里程，单位米"]
    assert (
        registry.describe(
            meta.name, fields=["$.values[*][*].forwardMileage"]
        ).output_fields
        == "{values: [[{forwardMileage: float；前进里程，单位米}]]}"
    )


def test_unused_constraints_do_not_prevent_registration_or_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("Output documentation must not validate constraints")

    monkeypatch.setattr("jsonschema.validators.validator_for", forbidden)
    schema = obj({"speed": node("number", "速度", minimum="unused by AXI")})
    meta = register_tool(lambda: {}, name="query", output_schema=schema)
    registry = make_registry(meta)
    assert registry.describe("query", fields=["$"]).output_fields == view_text(
        {"speed": "float；速度"}
    )
    for discover in (registry.search, registry.grep):
        assert discover("speed")[0].matched_fields == ["$.speed: float；速度"]
    schema["properties"]["speed"]["description"] = "更新速度"
    registry.register(meta)
    assert registry.describe("query", fields=["$"]).output_fields == view_text(
        {"speed": "float；更新速度"}
    )


@pytest.mark.parametrize("annotated", [False, True])
def test_nullable_any_keeps_its_unknown_type_in_every_view(annotated: bool) -> None:
    raw = Annotated[Any, Field(description="原始载荷")] if annotated else Any

    class Output(BaseModel):
        payload: raw | None = Field(description="业务附加信息")
        values: list[raw | None] = Field(description="原始采样值")

    schema = normalize_schema(Output)
    registry = make_registry(make_tool(schema))
    expected = {
        "payload": "anyOf(Any"
        + ("；原始载荷" if annotated else "")
        + ", None)；业务附加信息",
        "values": "list[anyOf(Any, None)]；原始采样值",
    }
    expanded = {
        **expected,
        "values": ["anyOf(Any" + ("；原始载荷" if annotated else "") + ", None)"],
    }
    assert registry.describe("read", fields=["$"]).output_fields == view_text(expected)
    assert registry.describe(
        "read", fields=["$", "$.values"]
    ).output_fields == view_text(expanded)
    for key, text in expected.items():
        assert registry.describe(
            "read", fields=["$." + key]
        ).output_fields == view_text({key: expanded[key]})
        assert registry.search(key)[0].matched_fields == [f"$.{key}: {text}"]
        assert registry.grep(r"^\$\." + key + ": ")[0].matched_fields == [
            f"$.{key}: {text}"
        ]


@pytest.mark.parametrize("marker", [None, "http://json-schema.org/draft-07/schema#"])
def test_reference_own_description_uses_pydantic2_rules(marker: str | None) -> None:
    class Mileage(RootModel[float]):
        pass

    class Output(BaseModel):
        forwardMileage: Mileage = Field(description="前进里程，单位米")

    schema = normalize_schema(Output)
    if marker:
        schema["$schema"] = marker
    original = deepcopy(schema)
    registry = make_registry(make_tool(schema))
    expected = {"forwardMileage": "float；前进里程，单位米"}
    view = registry.describe("read", fields=["$"])
    assert view.output_fields == view_text(expected)
    for discover in (registry.search, registry.grep):
        assert discover("前进里程")[0].matched_fields == [
            "$.forwardMileage: float；前进里程，单位米"
        ]
    assert schema == original


def test_annotated_nullable_references_keep_plain_types_and_own_prose() -> None:
    class Mileage(RootModel[float]):
        """里程值"""

    class Vehicle(BaseModel):
        speed: float = Field(description="速度")

    class Output(BaseModel):
        mileage: Annotated[Mileage, Field(description="前进里程", ge=0)] | None
        vehicle: Annotated[Vehicle, Field(description="车辆信息")] | None

    schema = normalize_schema(Output)
    original = deepcopy(schema)
    registry = make_registry(make_tool(schema))
    shallow = {
        "mileage": "anyOf(float；里程值；前进里程, None)",
        "vehicle": "anyOf(dict；车辆信息, None)",
    }
    for fields in (None, ["$", "$.vehicle"]):
        expected = (
            shallow
            if fields is None
            else {
                **shallow,
                "vehicle": composition("anyOf", {"speed": "float；速度"}, "None"),
            }
        )
        view = registry.describe("read", fields=fields if fields is not None else ["$"])
        assert view.output_fields == view_text(expected)
    for key, text in shallow.items():
        assert registry.describe(
            "read", fields=["$." + key]
        ).output_fields == view_text(
            {
                key: composition("anyOf", {"speed": "float；速度"}, "None")
                if key == "vehicle"
                else text
            }
        )
        assert registry.search(key)[0].matched_fields == [f"$.{key}: {text}"]
        assert registry.grep(r"^\$\." + key + ": ")[0].matched_fields == [
            f"$.{key}: {text}"
        ]
    assert any(
        text.startswith("$.mileage:")
        for text in registry.search("float")[0].matched_fields
    )
    assert schema == original


@pytest.mark.parametrize("nullable", [False, True])
def test_array_cardinality_keeps_declared_record_fields(nullable: bool) -> None:
    class Row(BaseModel):
        mileage: float = Field(description="累计里程")

    class Output(BaseModel):
        rows: list[Row] | None if nullable else list[Row] = Field(
            description="历史记录", max_length=0
        )

    schema = normalize_schema(Output)
    original = deepcopy(schema)
    registry = make_registry(make_tool(schema))
    shallow = {
        "rows": ("anyOf(list[dict], None)" if nullable else "list[dict]") + "；历史记录"
    }
    default = registry.describe("read", fields=["$"])
    assert default.output_fields == view_text(shallow)
    records = [{"mileage": "float；累计里程"}]
    expanded = {"rows": composition("anyOf", records, "None") if nullable else records}
    view = registry.describe("read", fields=["$.rows"])
    assert view.output_fields == view_text(expanded)
    selected = registry.describe("read", fields=["$.rows[*].mileage"])
    assert selected.output_fields == view_text(
        {"rows": composition("anyOf", records) if nullable else records}
    )
    assert selected.unknown_fields is None
    for discover in (registry.search, registry.grep):
        assert discover("mileage")[0].matched_fields == [
            "$.rows[*].mileage: float；累计里程"
        ]
    assert schema == original
