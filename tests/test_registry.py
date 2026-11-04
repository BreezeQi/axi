"""验证工具名称解析、重新注册后的索引刷新及搜索配置。"""

import subprocess
import sys
from pathlib import Path
from textwrap import dedent
from unittest.mock import Mock

import pytest

from axi.config import SearchConfig
from axi.registry import (
    AmbiguousToolError,
    Registry,
    ToolNotFoundError,
    ToolResolveError,
)
from axi.search.cache import EmbeddingCache
from axi.search.embedding import EmbeddingProvider
from tests.support import make_tool, node, obj, make_registry


def test_empty_search_does_not_prepare_documents_or_import_the_search_stack() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            dedent("""
                import sys
                from axi.config import SearchConfig
                from axi.models import ToolMeta
                from axi.registry import Registry

                for registry in (
                    Registry(),
                    Registry.from_search_config(SearchConfig()),
                ):
                    for _ in range(2):
                        assert registry.search("temperature") == []
                    registry.register(ToolMeta(
                        name="read", description="temperature", source="native",
                        output_schema={"properties": {"temperature": {"type": "number"}}},
                    ))
                    for query, k in (("", 5), ("  ", 5), ("temperature", 0), ("temperature", -1)):
                        assert registry.search(query, top_k=k) == []
                    assert registry._hybrid is None and registry._documents == {}
                assert not {"bm25s", "numpy", "jieba"}.intersection(sys.modules)
            """),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_tools_without_output_schema_skip_document_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = make_registry(make_tool(description="capability"))
    for _ in range(2):
        with monkeypatch.context() as undocumented:
            undocumented.setattr(
                "axi.output.documents.SchemaDocument",
                Mock(side_effect=AssertionError("Prepared an absent output Schema")),
            )
            for discover in (registry.search, registry.grep):
                hits = discover("capability")
                assert [hit.name for hit in hits] == ["read"]
                assert hits[0].matched_fields == []
                assert discover("speed") == []
            assert (
                registry.describe("read", fields=["$"]).output_schema_status
                == "not_registered"
            )
        assert registry._documents == {}
        registry.register(
            make_tool(obj({"speed": node("number", "车速")}), description="capability")
        )
        for discover in (registry.search, registry.grep):
            assert discover("speed")[0].matched_fields == ["$.speed: float；车速"]
        registry.register(make_tool(description="capability"))


def test_registry_resolution_and_server_names() -> None:
    registry = Registry()
    for server, name in [("a", "echo"), ("b", "echo"), ("a", "greet")]:
        registry.register(make_tool(name=name, server=server, source="mcp"))
    assert registry.resolve("a/echo").server == "a"
    assert registry.resolve("greet").full_name == "a/greet"
    for name, error in [
        ("echo", AmbiguousToolError),
        ("missing", ToolNotFoundError),
        ("x/echo", ToolNotFoundError),
    ]:
        with pytest.raises(error) as failure:
            registry.resolve(name)
        assert isinstance(failure.value, ToolResolveError)
    assert registry.search("greet")[0].name == "a/greet"
    assert registry.grep("greet")[0].name == "a/greet"
    registry.set_server("a/greet", "renamed")
    with pytest.raises(KeyError, match="missing"):
        registry.set_server("missing", "unused")
    assert registry.list_names() == ["a/echo", "b/echo", "renamed/greet"]
    assert len(registry.grep("echo")) == 2
    assert registry.search("greet")[0].name == "renamed/greet"
    registry.register(make_tool(name="new", description="newdiscovery"))
    assert registry.search("newdiscovery")[0].name == "new"
    assert registry.grep("echo", top_k=0) == []
    with pytest.raises(ValueError, match="Invalid regex"):
        registry.grep("[invalid", top_k=0)


@pytest.mark.parametrize("method", ["search", "grep"])
def test_schema_refresh_requires_reregistration(
    method: str, measurement_registry: Registry, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = measurement_registry
    meta = registry.resolve("read")
    discover = getattr(registry, method)
    expected = discover("oldvelocity")
    cached = registry._hybrid if method == "search" else registry._regex
    meta.output_schema["properties"]["speed"]["description"] = "newvelocity"
    meta.input_schema = {"properties": {"ignored": node("integer")}}
    meta.output_example = object()
    with monkeypatch.context() as warm:
        catalog = Mock(
            side_effect=AssertionError("An unchanged index read the catalog")
        )
        warm.setattr(registry, "_search_catalog", catalog)
        assert discover("oldvelocity") == expected and discover("newvelocity") == []
        catalog.assert_not_called()
    assert (registry._hybrid if method == "search" else registry._regex) is cached
    registry.register(meta)
    assert discover("oldvelocity") == []
    assert (registry._hybrid if method == "search" else registry._regex) is cached
    assert discover("newvelocity")[0].matched_fields == ["$.speed: float；newvelocity"]
    assert (
        registry.describe("read", fields=["$"]).output_fields
        == "{speed: float；newvelocity}"
    )


@pytest.mark.parametrize("with_schema", [False, True])
def test_replacement_refreshes_tool_descriptions(with_schema: bool) -> None:
    registry = Registry()
    meta = make_tool(
        obj({"speed": node("number")}) if with_schema else None,
        description="oldcapability",
    )
    registry.register(meta)
    assert registry.get(meta.full_name) is meta
    for discover in (registry.search, registry.grep):
        assert discover("oldcapability")[0].description == "oldcapability"
        assert discover("newcapability") == []
    registry.register(meta.model_copy(update={"description": "newcapability"}))
    assert registry.describe("read", fields=["$"]).description == "newcapability"
    for discover in (registry.search, registry.grep):
        assert discover("oldcapability") == []
        assert discover("newcapability")[0].description == "newcapability"


@pytest.mark.parametrize("broken", [False, True])
@pytest.mark.parametrize("source", ["native", "mcp"])
def test_embedding_uses_original_initialization_and_is_reused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, broken: bool, source: str
) -> None:
    provider = Mock(spec=EmbeddingProvider)
    provider.embed_documents.side_effect = lambda texts: [[1.0] for _ in texts]
    provider.embed_query.return_value = [1.0]
    factory = Mock(
        return_value=provider,
        side_effect=ValueError("offline") if broken else None,
    )
    monkeypatch.setattr("axi.registry.create_embedding_provider", factory)
    monkeypatch.setattr(
        "axi.registry.EmbeddingCache",
        lambda **kwargs: EmbeddingCache(tmp_path / "vectors.json", **kwargs),
    )
    config = SearchConfig.model_validate(
        {"embedding": {"provider": "openai", "apiKey": "offline"}}
    )
    if broken:
        with pytest.raises(ValueError, match="offline"):
            Registry.from_search_config(config)
        factory.assert_called_once_with(config.embedding)
        return
    registry = Registry.from_search_config(config)
    factory.assert_called_once_with(config.embedding)
    provider.embed_documents.assert_not_called()
    assert registry.search("speed") == []
    meta = make_tool(
        {"properties": {"speed": node("number")}}, source=source, server="demo"
    )
    registry.register(meta)
    for query in ("", " \t\n"):
        assert registry.search(query) == []
    assert registry.list_names() == ["demo/read"]
    assert registry.describe("read", fields=["$"]).output_fields == "{speed: float}"
    assert registry.grep("speed")[0].name == "demo/read"
    assert registry.search("speed", top_k=0) == []
    provider.embed_documents.assert_not_called()
    provider.embed_query.assert_not_called()
    for _ in range(2):
        assert registry.search("speed")[0].matched_fields == ["$.speed: float"]
    calls = provider.mock_calls.copy()
    for query in ("", " \t\n"):
        assert registry.search(query) == []
    assert provider.mock_calls == calls
    registry.register(meta.model_copy(update={"description": "updated"}))
    assert registry.search("speed")[0].description == "updated"
    factory.assert_called_once_with(config.embedding)
