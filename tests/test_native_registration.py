"""原生工具的归组、定义和执行不依赖 Python 导入顺序。"""

import importlib
import json
import sys
from collections.abc import Iterator
from importlib.metadata import EntryPoint
from pathlib import Path

import pytest
from typer.testing import CliRunner

from axi import runtime, tool
from axi.config import NativeToolEntry
from axi.executor import Executor
from axi.providers import native
from axi.registry import Registry


@pytest.fixture
def native_registry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Iterator[Registry]:
    registry = Registry()
    monkeypatch.setattr(runtime, "registry", registry)
    monkeypatch.setattr(runtime, "executor", Executor(registry))
    monkeypatch.setattr(native, "_native_tools", {})
    monkeypatch.setattr(native, "_tool_definitions", {})
    monkeypatch.setattr(native, "_modules_loaded", False)
    monkeypatch.setattr(native.app_config, "native_tools", [])
    monkeypatch.setattr(native.importlib.metadata, "entry_points", lambda **_: [])
    monkeypatch.syspath_prepend(str(tmp_path))

    def no_daemon(*args: object) -> None:
        pytest.fail("A registered native tool must not fall back to daemon")

    monkeypatch.setattr("axi.executor.daemon_request", no_daemon)
    yield registry
    for name in list(sys.modules):
        if name.startswith("_axi_registration_"):
            sys.modules.pop(name)


@pytest.mark.parametrize("source", ["config", "entry_points", "both"])
@pytest.mark.parametrize("preimport", [False, True])
def test_discovery_keeps_same_named_tools_and_their_definitions(
    native_registry: Registry,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source: str,
    preimport: bool,
) -> None:
    modules = []
    for group, parameter in [("orders", "region: str"), ("metrics", "count: int")]:
        module = f"_axi_registration_{group}"
        modules.append(module)
        (tmp_path / f"{module}.py").write_text(
            "from axi import tool\n"
            f"@tool(description='source-{group}')\n"
            f"def query_orders({parameter}) -> dict:\n"
            f"    return {{'source': '{group}', 'value': {parameter.split(':')[0]}}}\n"
        )
        if preimport:
            imported = importlib.import_module(module)
            params = {"region": "cn"} if group == "orders" else {"count": 2}
            assert imported.query_orders(**params)["source"] == group
            assert importlib.import_module(module) is imported

    entries = [
        NativeToolEntry(module=module, name=group)
        for module, group in zip(modules, ("orders", "metrics"))
    ]
    if source != "entry_points":
        monkeypatch.setattr(native.app_config, "native_tools", entries)
    if source != "config":
        monkeypatch.setattr(
            native.importlib.metadata,
            "entry_points",
            lambda **_: [
                EntryPoint(
                    name=f"ignored-{entry.name}" if source == "both" else entry.name,
                    value=entry.module,
                    group="axi.native_tools",
                )
                for entry in entries
            ],
        )

    assert tool("orders/query_orders")(region="cn") == {
        "source": "orders",
        "value": "cn",
    }
    assert tool("metrics/query_orders")(count="2") == {"source": "metrics", "value": 2}
    expected = ["orders/query_orders", "metrics/query_orders"]
    assert native_registry.list_names() == expected
    for name, parameter in zip(expected, ("region", "count")):
        meta = native_registry.resolve(name)
        assert meta.input_schema["required"] == [parameter]
        assert native_registry.grep(meta.description)[0].name == name
    native.load_native_tool_modules()
    assert native_registry.list_names() == expected
    assert tool("orders/query_orders")(region="jp")["value"] == "jp"


@pytest.mark.parametrize("preimport", [False, True])
@pytest.mark.parametrize("entry", ["package", "export", "module_export"])
def test_package_discovery_includes_preimported_children_and_reexports(
    native_registry: Registry,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    preimport: bool,
    entry: str,
) -> None:
    package = tmp_path / "_axi_registration_package"
    package.mkdir()
    (package / "__init__.py").write_text("from . import implementation\n")
    implementation = package / "implementation"
    implementation.mkdir()
    (implementation / "__init__.py").write_text("from . import orders\n")
    (implementation / "orders.py").write_text(
        "from axi import tool\n@tool()\ndef query_orders() -> str:\n    return 'orders'\n"
    )
    # 普通模块也可以集中导出其他模块声明的工具。
    (tmp_path / "_axi_registration_export.py").write_text(
        "from _axi_registration_package.implementation.orders import query_orders\n"
    )
    (tmp_path / "_axi_registration_module_export.py").write_text(
        "from _axi_registration_package.implementation import orders\n"
    )
    if preimport:
        assert (
            importlib.import_module(
                "_axi_registration_package.implementation.orders"
            ).query_orders()
            == "orders"
        )
    monkeypatch.setattr(
        native.app_config,
        "native_tools",
        [NativeToolEntry(module=f"_axi_registration_{entry}", name="demo")],
    )
    assert tool("demo/query_orders")() == "orders"
    assert native_registry.list_names() == ["demo/query_orders"]


def test_file_modules_with_the_same_basename_keep_separate_tools(
    native_registry: Registry, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    entries = []
    for group in ("orders", "metrics"):
        directory = tmp_path / group
        directory.mkdir()
        path = directory / "tools.py"
        path.write_text(
            "from axi import tool\n@tool()\ndef query() -> str:\n"
            f"    return '{group}'\n"
        )
        entries.append(NativeToolEntry(module=str(path), name=group))
    monkeypatch.setattr(native.app_config, "native_tools", entries)
    assert tool("orders/query")() == "orders"
    assert tool("metrics/query")() == "metrics"
    assert native_registry.list_names() == ["orders/query", "metrics/query"]


def test_cli_uses_the_same_identity_for_preimported_tools(
    native_registry: Registry, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from axi.cli import app

    module = "_axi_registration_cli"
    (tmp_path / f"{module}.py").write_text(
        "from axi import tool\n@tool()\ndef query(region: str) -> str:\n"
        "    return region\n"
    )
    assert importlib.import_module(module).query("cn") == "cn"
    monkeypatch.setattr(
        native.app_config, "native_tools", [NativeToolEntry(module=module, name="demo")]
    )
    result = CliRunner().invoke(app, ["run", "demo/query", "--region", "cn"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {"status": "success", "data": "cn"}
    assert native_registry.list_names() == ["demo/query"]
