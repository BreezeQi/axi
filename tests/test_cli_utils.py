"""验证 CLI 参数解析、配置校验与可操作的错误提示。"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import axi.cli as cli
from axi.config import AxiConfig, _warn_ignored_local_config, load_config
from axi.daemon.protocol import DaemonResponse
from axi.providers.mcp import MCPServerConfig
from tests.support import make_tool, node, make_registry


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ([], {}),
        (["--name", "alice"], {"name": "alice"}),
        (["--verbose"], {"verbose": True}),
        (["--data", '{"x":1}'], {"data": {"x": 1}}),
        (["--count", "42"], {"count": 42}),
        (["--verbose", "--name", "alice"], {"verbose": True, "name": "alice"}),
        (["--flag", "true"], {"flag": True}),
    ],
)
def test_parse_params(args: list[str], expected: dict) -> None:
    assert cli._parse_params(args) == expected


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        ("text", "true", "true"),
        ("text", "null", "null"),
        ("text", "42", "42"),
        ("opt", "false", "false"),
        ("count", "42", 42),
        ("active", "true", True),
        ("misc", "true", True),
    ],
)
def test_schema_aware_params(key: str, value: str, expected: object) -> None:
    schema = {
        "properties": {
            "text": node("string"),
            "opt": {"anyOf": [node("string"), node("null")]},
            "count": node("integer"),
            "active": node("boolean"),
        }
    }
    assert cli._parse_params(["--" + key, value], schema) == {key: expected}


@pytest.mark.parametrize(
    ("args", "value", "remaining"),
    [
        (["--json", '{"a":1}', "--verbose"], '{"a":1}', ["--verbose"]),
        (["-j", '{"a":1}'], '{"a":1}', []),
        (["--verbose", "--name", "x"], None, ["--verbose", "--name", "x"]),
    ],
)
def test_extract_option(
    args: list[str], value: str | None, remaining: list[str]
) -> None:
    assert cli._extract_option(args, "--json", "-j") == (value, remaining)


@pytest.mark.parametrize("missing_key", [False, True])
def test_doctor(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture, missing_key: bool
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = {"search": {"embedding": {"provider": "openai"}}} if missing_key else {}
    monkeypatch.setattr(cli, "app_config", AxiConfig.model_validate(config))
    if missing_key:
        with pytest.raises(cli.typer.Exit) as error:
            cli.doctor()
        assert error.value.exit_code == 1
    else:
        cli.doctor()
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is not missing_key
    assert report["embedding"].get("api_key") == ("missing" if missing_key else None)


@pytest.mark.parametrize("server", [None, "remote"])
@pytest.mark.parametrize("available", [False, True])
def test_listing_reports_remote_availability(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    server: str | None,
    available: bool,
) -> None:
    registry = make_registry(make_tool(server="local"))
    monkeypatch.setattr(cli.runtime, "registry", registry)
    response = (
        DaemonResponse.success([make_tool(server="remote", source="mcp").model_dump()])
        if available
        else DaemonResponse.fail("Restart: axi daemon stop && axi daemon start")
    )
    monkeypatch.setattr("axi.daemon.client.daemon_request", lambda req: response)
    monkeypatch.setattr(
        cli.app_config,
        "mcp_servers",
        {"remote": {"command": "unused"}},
    )
    cli.list_tools(server)
    captured = capsys.readouterr()
    assert ("axi daemon stop" in captured.err) is not available
    assert json.loads(captured.out) == (
        {
            "server": "remote",
            "tools": [{"name": "read", "description": ""}] if available else [],
        }
        if server
        else [
            {"server": "local", "tools": ["read"]},
            {"server": "remote", "tools": ["read"] if available else []},
        ]
    )


@pytest.mark.parametrize(
    "config", [{}, {"server": "test"}, {"server": "", "command": "python"}]
)
def test_invalid_mcp_config(config: dict) -> None:
    with pytest.raises(ValidationError):
        MCPServerConfig(**config)


@pytest.mark.parametrize(
    "transport", [{"command": "python"}, {"url": "http://localhost:8080"}]
)
def test_mcp_transport_config(transport: dict) -> None:
    assert MCPServerConfig(server="test", **transport).server == "test"


def test_config_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    path = tmp_path / "axi.json"
    assert load_config(path) == AxiConfig()
    path.write_text("{invalid json")
    with pytest.raises(SystemExit, match="Malformed config file"):
        load_config(path)
    path.write_text('{"mcpServers": {}}')
    assert load_config(path).mcp_servers == {}
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("AXI_CONFIG", raising=False)
    _warn_ignored_local_config()
    assert "ignored" in capsys.readouterr().err
