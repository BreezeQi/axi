"""验证发现入口的目录边界、参数传递和通信失败处理。

排名规则由搜索引擎测试覆盖；这里核对各入口实际使用的目录、数量上限
以及序列化后的摘要。
"""

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path
from runpy import run_path
from unittest.mock import Mock

import pytest
from mcp.types import CallToolResult
from typer.testing import CliRunner

import axi.cli as cli
from axi.config import AxiConfig, SearchConfig
from axi.daemon.protocol import DaemonResponse
from axi.daemon.server import DaemonServer
from axi.executor import Executor
from axi.registry import Registry
from axi.output.documents import SchemaDocument
from axi.search.bm25 import BM25Search
from axi.search.cache import EmbeddingCache
from axi.search.embedding import EmbeddingProvider
from tests.support import make_tool, node, obj, make_registry


@pytest.fixture
def catalog(monkeypatch: pytest.MonkeyPatch) -> Mock:
    local, remote = Registry(), Registry()
    for source, registry in [("native", local), ("mcp", remote)]:
        for index in range(3):
            registry.register(
                make_tool(
                    obj({"temperature": node("number", "温度")}),
                    f"read{index}",
                    server="mcp-other" if source == "mcp" and index == 1 else source,
                    source=source,
                    description="temperature",
                    input_schema=obj({"identifier": node("string")}),
                    output_example={"temperature": 7},
                )
            )
    monkeypatch.setattr(cli.runtime, "registry", local)
    monkeypatch.setattr(
        cli,
        "app_config",
        AxiConfig.model_validate(
            {
                "mcpServers": {
                    name: {"command": "unused"} for name in ("mcp", "mcp-other")
                }
            }
        ),
    )
    monkeypatch.setattr("axi.providers.native.load_native_tool_modules", lambda: None)
    server = DaemonServer()
    server.registry = remote
    daemon = Mock(
        side_effect=lambda req: DaemonResponse.model_validate_json(
            asyncio.run(server._dispatch(req))
        )
    )
    daemon.server = server
    monkeypatch.setattr("axi.daemon.client.daemon_request", daemon)
    return daemon


@pytest.mark.parametrize(
    "tool", ["native/read0", "mcp/read0", "mcp-other/read1", "read0"]
)
@pytest.mark.parametrize("cap", [0, 8])
def test_scoped_field_search_matches_registry_cli_daemon_and_meta(
    catalog: Mock,
    tool: str,
    cap: int,
    monkeypatch: pytest.MonkeyPatch,
    mcp_call: Callable,
) -> None:
    import axi.mcp_serve as mcp

    local = make_registry(*cli.runtime.registry.list_all(), max_fields=cap)
    remote = make_registry(*catalog.server.registry.list_all(), max_fields=cap)
    monkeypatch.setattr(cli.runtime, "registry", local)
    catalog.server.registry = remote
    owner = local if tool in ("native/read0", "read0") else remote
    resolved = owner.resolve(tool).full_name
    expected = [
        hit.model_dump(exclude_none=True)
        for hit in owner.search("temperature", top_k=1, tool=tool)
    ]
    result = CliRunner().invoke(
        cli.app, ["search", "temperature", "--tool", tool, "-k", "1"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == expected
    assert catalog.call_count == (owner is remote)
    if owner is remote:
        assert catalog.call_args.args[0].tool_name == tool
    assert expected[0]["name"] == resolved
    assert expected[0]["matched_fields"] == (
        ["$.temperature: float；温度"] if cap else []
    )
    merged = make_registry(*local.list_all(), *remote.list_all(), max_fields=cap)
    server = mcp._build_meta_server(merged, Executor(merged))
    response = asyncio.run(
        mcp_call(
            server, "search", {"query": "temperature", "tool": resolved, "top_k": 1}
        )
    )
    assert not response.is_error
    assert json.loads(response.content[0].text) == expected


def test_cli_scoped_search_reports_unknown_tool_and_does_not_fallback_after_a_local_miss(
    catalog: Mock,
) -> None:
    result = CliRunner().invoke(cli.app, ["search", "temperature", "--tool", "missing"])
    assert result.exit_code == 1
    assert "Tool not found" in json.loads(result.stdout)["error"]
    catalog.reset_mock()
    result = CliRunner().invoke(cli.app, ["search", "absent", "--tool", "native/read0"])
    assert result.exit_code == 0 and json.loads(result.stdout) == []
    catalog.assert_not_called()


@pytest.mark.parametrize(
    "top_k,max_fields,exact",
    [
        (-1, 2, False),
        (0, 2, False),
        (1, 2, False),
        (2, 2, False),
        (3, 2, False),
        (20, 2, False),
        (2, 0, False),
        (2, 2, True),
    ],
)
@pytest.mark.parametrize(
    "command,option,limit_name,field_query",
    [
        ("search", "--top-k", "top_k", "$['temperature']"),
        ("grep", "--limit", "limit", r"^\$\.temperature:"),
    ],
)
def test_cli_and_meta_discovery_respect_their_catalog_boundaries(
    catalog: Mock,
    monkeypatch: pytest.MonkeyPatch,
    mcp_call: Callable[..., Awaitable[CallToolResult]],
    top_k: int,
    exact: bool,
    field_query: str,
    command: str,
    option: str,
    limit_name: str,
    max_fields: int,
) -> None:
    import axi.mcp_serve as mcp

    query = field_query if exact else "temperature"
    config = SearchConfig.model_validate({"maxFields": max_fields})
    monkeypatch.setattr(cli.app_config, "search", config)
    monkeypatch.setattr(
        cli.runtime,
        "registry",
        make_registry(*cli.runtime.registry.list_all(), max_fields=max_fields),
    )
    catalog.server.registry = make_registry(
        *catalog.server.registry.list_all(), max_fields=max_fields
    )
    monkeypatch.setattr(mcp, "app_config", cli.app_config)
    registry = mcp.build_registry(cli.runtime.registry.list_all(), servers=None)
    for native in cli.runtime.registry.list_all():
        assert registry.resolve(native.full_name) is native
    remote = registry.resolve("mcp/read0")
    assert remote.input_schema == obj({"identifier": node("string")})
    assert remote.output_example == {"temperature": 7}
    expected_meta = [
        hit.model_dump(exclude_none=True)
        for hit in getattr(registry, command)(query, top_k=top_k)
    ]
    expected_cli = [
        hit.model_dump(exclude_none=True)
        for source in (cli.runtime.registry, catalog.server.registry)
        for hit in getattr(source, command)(query, top_k=top_k)
    ]
    catalog.reset_mock()
    result = CliRunner().invoke(cli.app, [command, query, option, str(top_k)])
    assert result.exit_code == 0, result.output
    cli_hits = json.loads(result.stdout)
    assert cli_hits == expected_cli
    count = max(0, min(top_k, 3))
    assert [item["source"] for item in cli_hits] == ["native"] * count + ["mcp"] * count
    assert catalog.call_count == (1 if top_k > 0 else 0)
    if top_k > 0:
        request = catalog.call_args.args[0]
        assert request.method == command and request.query == query
        assert request.top_k == top_k
    response = asyncio.run(
        mcp_call(
            mcp._build_meta_server(registry, Executor(registry)),
            command,
            {"query" if command == "search" else "pattern": query, limit_name: top_k},
        )
    )
    assert not response.is_error
    assert json.loads(response.content[0].text) == expected_meta
    assert len(expected_cli) == 2 * max(0, min(top_k, 3))
    assert len(expected_meta) == max(0, min(top_k, 6))
    for hit in cli_hits + json.loads(response.content[0].text):
        assert hit["matched_fields"] == (
            ["$.temperature: float；温度"] if max_fields else []
        )
        assert "score" not in hit
        assert set(hit) == {"name", "description", "source", "matched_fields"}


def test_cli_returns_remote_hits_when_native_has_none(
    catalog: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = make_registry(make_tool(name="orders", description="订单查询"))
    monkeypatch.setattr(cli.runtime, "registry", registry)
    result = CliRunner().invoke(cli.app, ["search", "temperature", "-k", "1"])
    assert result.exit_code == 0, result.output
    hits = json.loads(result.stdout)
    assert len(hits) == 1 and hits[0]["source"] == "mcp"
    assert catalog.call_args.args[0].method == "search"


@pytest.mark.parametrize(
    "command,query",
    [
        ("search", "temperature"),
        ("search", "$.temperature"),
        ("grep", "temperature"),
        ("grep", r"^\$\.temperature:"),
    ],
)
@pytest.mark.parametrize("native_matches", [False, True])
def test_cli_keeps_same_named_results_from_both_catalogs(
    catalog: Mock,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    query: str,
    native_matches: bool,
) -> None:
    schema = obj({"temperature": node("number", "温度")})
    registry = make_registry(
        make_tool(
            schema if native_matches else None,
            server="shared",
            description="temperature" if native_matches else "orders",
        )
    )
    monkeypatch.setattr(cli.runtime, "registry", registry)
    catalog.server.registry = make_registry(
        make_tool(schema, server="shared", source="mcp", description="temperature")
    )
    hits = cli._discover(command, query, top_k=1)
    assert [hit["source"] for hit in hits] == (
        ["native", "mcp"] if native_matches else ["mcp"]
    )
    assert [hit["name"] for hit in hits] == ["shared/read"] * len(hits)
    assert all(hit["matched_fields"] == ["$.temperature: float；温度"] for hit in hits)


def test_repeated_cli_searches_reuse_source_indexes(
    catalog: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepare = Mock(side_effect=SchemaDocument._prepare_fields)
    monkeypatch.setattr(
        SchemaDocument, "_prepare_fields", lambda document: prepare(document)
    )
    builds = []
    original = BM25Search.build_documents

    def build(engine: BM25Search, texts: list[str]) -> None:
        builds.append(len(texts))
        original(engine, texts)

    monkeypatch.setattr(BM25Search, "build_documents", build)
    expected = cli._discover("search", "temperature", 5)
    for _ in range(2):
        assert cli._discover("search", "temperature", 5) == expected
    assert prepare.call_count == 6 and builds == [3, 3, 3, 3]
    replacement = catalog.server.registry.resolve("mcp/read0").model_copy(
        update={"description": "updatedtemperature"}
    )
    catalog.server.registry.register(replacement)
    assert cli._discover("search", "updatedtemperature", 5)[0]["name"] == "mcp/read0"
    # 只有被替换的工具定义需要重新准备字段文档。
    assert prepare.call_count == 7 and builds == [3, 3, 3, 3, 3, 3]


def test_native_runtime_uses_bm25_with_embedding_configured(
    catalog: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = SearchConfig.model_validate(
        {"embedding": {"provider": "openai"}, "maxFields": 0}
    )
    monkeypatch.setattr(cli.runtime.app_config, "search", config)
    factory = Mock(side_effect=AssertionError("Native search initialized embeddings"))
    monkeypatch.setattr("axi.registry.create_embedding_provider", factory)
    registry = run_path(cli.runtime.__file__)["registry"]
    registry.register(make_tool(obj({"temperature": node("number", "温度")})))
    monkeypatch.setattr(cli.runtime, "registry", registry)
    result = CliRunner().invoke(cli.app, ["search", "temperature", "-k", "1"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)[0]["matched_fields"] == []
    factory.assert_not_called()


@pytest.mark.parametrize("phase", ["indexing", "query"])
def test_embedding_fallback_is_reported_without_changing_json_results(
    catalog: Mock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    phase: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = Mock(spec=EmbeddingProvider)
    provider.embed_documents.side_effect = lambda texts: [[1.0] for _ in texts]
    provider.embed_query.side_effect = RuntimeError("offline embedding")
    factory = Mock(return_value=provider)
    if phase == "indexing":
        provider.embed_documents.side_effect = RuntimeError("offline embedding")
    monkeypatch.setattr("axi.registry.create_embedding_provider", factory)
    monkeypatch.setattr(
        "axi.registry.EmbeddingCache",
        lambda **kwargs: EmbeddingCache(tmp_path / "vectors.json", **kwargs),
    )
    config = SearchConfig.model_validate({"embedding": {"provider": "openai"}})
    remote = catalog.server.registry.list_all()
    catalog.server.registry = Registry.from_search_config(config)
    for meta in remote:
        catalog.server.registry.register(meta)
    monkeypatch.setattr(
        cli.app_config,
        "search",
        SearchConfig.model_validate({"embedding": {"provider": "openai"}}),
    )
    # 这些警告由 daemon 产生，实际运行时写入 daemon 自己的日志。
    for _ in range(2 if phase == "query" else 1):
        caplog.clear()
        result = CliRunner().invoke(cli.app, ["search", "temperature"])
        assert result.exit_code == 0, result.output
        hits = json.loads(result.stdout)
        assert len(hits) == len({hit["name"] for hit in hits}) == 6
        assert (
            caplog.text.count(f"Embedding {phase} failed; using BM25.")
            == ({"indexing": 2, "query": 1}[phase])
        )
        assert (
            "offline embedding" in caplog.text
            and "Check search.embedding" in caplog.text
        )
    factory.assert_called_once()


@pytest.mark.parametrize("query", ["", " \t\n"])
def test_blank_search_does_not_request_daemon(catalog: Mock, query: str) -> None:
    result = CliRunner().invoke(cli.app, ["search", query])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == []
    catalog.assert_not_called()


def test_empty_grep_keeps_regex_semantics(catalog: Mock) -> None:
    result = CliRunner().invoke(cli.app, ["grep", ""])
    assert result.exit_code == 0, result.output
    assert len(json.loads(result.stdout)) == 6
    catalog.assert_called_once()


def test_native_listing_does_not_request_daemon(
    catalog: Mock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "app_config", AxiConfig())
    result = CliRunner().invoke(cli.app, ["list"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == [
        {"server": "native", "tools": ["read0", "read1", "read2"]}
    ]
    catalog.assert_not_called()


def test_listing_keeps_both_registered_entries_for_a_shared_name(catalog: Mock) -> None:
    cli.runtime.registry.register(
        make_tool(name="read0", server="mcp", description="local")
    )
    result = CliRunner().invoke(cli.app, ["list", "mcp"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {
        "server": "mcp",
        "tools": [
            {"name": "read0", "description": "local"},
            {"name": "read0", "description": "temperature"},
            {"name": "read2", "description": "temperature"},
        ],
    }


@pytest.mark.parametrize("command", ["search", "grep"])
@pytest.mark.parametrize("remote", ["unavailable", "absent", "collision"])
def test_partial_catalogs(
    catalog: Mock, monkeypatch: pytest.MonkeyPatch, command: str, remote: str
) -> None:
    if remote == "unavailable":
        catalog.side_effect = None
        catalog.return_value = DaemonResponse.fail(
            "Restart: axi daemon stop && axi daemon start"
        )
    elif remote == "absent":
        monkeypatch.setattr(cli, "app_config", AxiConfig())
    else:
        cli.runtime.registry.register(
            make_tool(name="read0", server="mcp", description="temperature")
        )
    result = CliRunner().invoke(
        cli.app, [command, "temperature", "-k" if command == "search" else "-l", "10"]
    )
    assert result.exit_code == 0, result.output
    hits = json.loads(result.stdout)
    if remote == "collision":
        assert len(hits) == 7
        assert [item["source"] for item in hits if item["name"] == "mcp/read0"] == [
            "native",
            "mcp",
        ]
        native_call = Mock(return_value="native result")
        monkeypatch.setattr(
            "axi.executor.get_native_function", lambda name: native_call
        )
        monkeypatch.setattr(
            "axi.executor.validate_native_params", lambda name, params: params
        )
        catalog.reset_mock()
        assert (
            Executor(cli.runtime.registry).run("mcp/read0", {}).data == "native result"
        )
        native_call.assert_called_once_with()
        catalog.assert_not_called()
    else:
        assert len(hits) == 3
        if remote == "absent":
            catalog.assert_not_called()
        else:
            assert "axi daemon stop" in result.stderr


@pytest.mark.parametrize("command", ["search", "grep"])
def test_discovery_consumes_response_data_without_prevalidation(
    catalog: Mock, command: str
) -> None:
    catalog.side_effect = None
    remote = {"name": "remote/read", "matched_fields": ["$.value: float；读数"]}
    catalog.return_value = DaemonResponse.success([remote])
    result = CliRunner().invoke(cli.app, [command, "temperature"])
    assert result.exit_code == 0, result.output
    hits = json.loads(result.stdout)
    assert [hit["source"] for hit in hits[:3]] == ["native"] * 3
    assert hits[3] == remote
    assert result.stderr == ""


@pytest.mark.parametrize("limit", [0, -1, 10])
def test_invalid_grep_precedes_daemon(catalog: Mock, limit: int) -> None:
    result = CliRunner().invoke(cli.app, ["grep", "[", "--limit", str(limit)])
    assert result.exit_code == 1 and "Invalid regex pattern" in result.stdout
    catalog.assert_not_called()
