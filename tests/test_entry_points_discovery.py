"""验证配置与 entry points 发现、优先顺序及单模块导入失败隔离。"""

import sys
from importlib.metadata import EntryPoint
from unittest.mock import Mock

import pytest

import axi.cli as cli
from axi.config import NativeToolEntry
from axi.providers import native
from axi.registry import Registry


@pytest.mark.parametrize(
    ("configured", "ep_value", "ep_name", "names", "warning"),
    [
        (False, "fake_ep_tools", "pkg", ["pkg/ping_ep"], ""),
        (False, "fake_ep_tools:ping_ep", "pkg", ["pkg/ping_ep"], ""),
        (True, "fake_ep_tools", "pkg", ["shared/ping_ep"], ""),
        (
            True,
            "fake_ep_tools_alt",
            "shared",
            ["shared/ping_ep", "shared/pong_ep"],
            "Server name collision",
        ),
        (
            True,
            "not_a_real_module",
            "bad",
            ["shared/ping_ep"],
            "Failed to load native tool module",
        ),
    ],
)
def test_native_discovery(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    configured: bool,
    ep_value: str,
    ep_name: str,
    names: list[str],
    warning: str,
) -> None:
    for module in ("tests.fake_ep_tools", "tests.fake_ep_tools_alt"):
        monkeypatch.delitem(sys.modules, module, raising=False)
    registry = Registry()
    monkeypatch.setattr(cli.runtime, "registry", registry)
    monkeypatch.setattr(
        native.app_config,
        "native_tools",
        [NativeToolEntry(module="tests.fake_ep_tools", name="shared")]
        if configured
        else [],
    )
    entry_points = Mock(
        return_value=[
            EntryPoint(
                name=ep_name,
                value="tests." + ep_value,
                group="axi.native_tools",
            )
        ]
    )
    monkeypatch.setattr(native.importlib.metadata, "entry_points", entry_points)
    native.load_native_tool_modules()
    assert registry.list_names() == names
    if warning:
        assert warning in caplog.text
