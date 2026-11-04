"""axi daemon 服务端：维持 MCP 连接，通过 Unix socket 接受请求。"""

import asyncio
import logging
import os
import signal
import time
from typing import Any

import jsonschema

from axi.config import CONFIG_PATH, app_config
from axi.daemon.protocol import (
    SOCKET_DIR,
    SOCKET_PATH,
    PID_PATH,
    STREAM_LIMIT,
    DaemonRequest,
    DaemonResponse,
    DaemonStatus,
)
from axi.models import ToolDescriptionError, ToolInfo, ToolSource, allowed_types
from axi.providers.mcp import MCPProvider
from axi.registry import Registry, ToolResolveError, split_names

logger = logging.getLogger(__name__)

_IDLE_EXEMPT_METHODS = frozenset({"status", "shutdown"})


def _encode_response(response: DaemonResponse) -> bytes:
    """按原序列化选项将响应编码为一行 JSON。"""
    return response.model_dump_json().encode() + b"\n"


def _coerce_to_schema(params: dict, schema: dict) -> dict:
    """按 schema 顶层类型做轻量转换，弥合 CLI 参数解析的类型猜测与严格校验的间隙。

    只处理无歧义的方向：数字 → string（``--message 42``），
    可解析的字符串 → integer/number/boolean（``--port 08080``）。
    目标类型本身就合法时不动（如 anyOf 同时允许 string 和 integer）。
    """
    props = schema.get("properties")
    if not isinstance(props, dict):
        return params
    coerced = dict(params)
    for key, value in params.items():
        prop = props.get(key)
        if not isinstance(prop, dict):
            continue
        allowed = allowed_types(prop)
        if (
            "string" in allowed
            and not ({"integer", "number"} & allowed)
            and isinstance(value, (int, float))
            and not isinstance(value, bool)
        ):
            coerced[key] = str(value)
        elif "string" not in allowed and isinstance(value, str):
            try:
                if "integer" in allowed:
                    coerced[key] = int(value)
                elif "number" in allowed:
                    coerced[key] = float(value)
                elif "boolean" in allowed and value.lower() in ("true", "false"):
                    coerced[key] = value.lower() == "true"
            except ValueError:
                pass
    return coerced


class DaemonServer:
    """axi daemon 主体。"""

    def __init__(self) -> None:
        self.mcp_provider = MCPProvider()
        self._start_time: float = time.monotonic()
        self._last_activity: float = time.monotonic()
        self._run_task: asyncio.Task | None = None
        self._stopping = False

        self._idle_timeout: float = app_config.daemon.idle_timeout_minutes * 60
        self.registry = Registry.from_search_config(app_config.search)
        # 按工具缓存编译后的 jsonschema validator；schema 不合法的缓存 None
        self._validators: dict[str, Any] = {}

    async def start(self) -> None:
        """启动 daemon：连接 MCP server，监听 Unix socket。"""
        self._start_time = time.monotonic()
        self._last_activity = time.monotonic()

        self._run_task = asyncio.current_task()
        self._stopping = False
        loop = asyncio.get_running_loop()
        watchdog: asyncio.Task | None = None
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self.stop)

        try:
            os.makedirs(SOCKET_DIR, exist_ok=True)
            # 先清理残留 socket，再发布新 PID，避免把初始化中的进程误判为就绪。
            if os.path.exists(SOCKET_PATH):
                os.unlink(SOCKET_PATH)
            with open(PID_PATH, "w") as f:
                f.write(str(os.getpid()))

            # 信号处理已就绪，连接 MCP 期间也能取消并回收子进程。
            await self._load_mcp_tools()

            server = await asyncio.start_unix_server(
                self._handle_client, path=SOCKET_PATH, limit=STREAM_LIMIT
            )
            async with server:
                logger.info(
                    "Daemon listening on %s (idle timeout: %ds)",
                    SOCKET_PATH,
                    int(self._idle_timeout),
                )
                watchdog = asyncio.create_task(self._idle_watchdog())
                await server.serve_forever()
        except asyncio.CancelledError:
            if not self._stopping:
                raise
        finally:
            # 每条 MCP 连接由自己的持有任务清理。
            try:
                if watchdog:
                    watchdog.cancel()
                await self.mcp_provider.close_all()
            finally:
                for path in (SOCKET_PATH, PID_PATH):
                    if os.path.exists(path):
                        os.unlink(path)
                for sig in (signal.SIGTERM, signal.SIGINT):
                    loop.remove_signal_handler(sig)
                self._run_task = None

    async def _load_mcp_tools(self) -> None:
        """直接登记导入的定义，返回后释放临时目录列表的引用。"""
        configs = self.mcp_provider.load_config()
        if configs:
            tools = await self.mcp_provider.connect_all(configs)
            for tool_meta in tools:
                self.registry.register(tool_meta)
            logger.info(
                "Loaded %d tools from %d MCP server(s)", len(tools), len(configs)
            )

    async def _idle_watchdog(self) -> None:
        """定期检查 idle 状态，超时则自动关闭 daemon。"""
        while True:
            await asyncio.sleep(60)
            idle = time.monotonic() - self._last_activity
            if idle > self._idle_timeout:
                logger.info("Idle timeout reached (%.0fs), shutting down", idle)
                self.stop()
                break

    def stop(self) -> None:
        """请求停止，由主任务取消初始化/监听并统一清理资源。"""
        if self._stopping:
            return
        self._stopping = True
        logger.info("Shutting down daemon...")
        if self._run_task:
            self._run_task.cancel()

    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """处理单个客户端连接。"""
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break

                req = None
                try:
                    req = DaemonRequest.model_validate_json(line)
                    if req.method not in _IDLE_EXEMPT_METHODS:
                        self._last_activity = time.monotonic()
                    payload = await self._dispatch(req)
                except Exception as e:
                    method = req.method if req else "<invalid>"
                    logger.exception("Error processing request: %s", method)
                    payload = _encode_response(
                        DaemonResponse.fail(f"{type(e).__name__}: {e}")
                    )
                writer.write(payload)
                await writer.drain()
        except (ConnectionResetError, BrokenPipeError):
            logger.debug("Client connection dropped")
        finally:
            writer.close()

    def _get_validator(self, full_name: str, schema: dict) -> Any | None:
        """获取（并缓存）工具的 jsonschema validator；schema 为空或不合法时返回 None。"""
        if full_name not in self._validators:
            validator = None
            if schema:
                try:
                    cls = jsonschema.validators.validator_for(schema)
                    cls.check_schema(schema)
                    validator = cls(schema)
                except Exception:
                    pass
            self._validators[full_name] = validator
        return self._validators[full_name]

    async def _dispatch(self, req: DaemonRequest) -> bytes:
        """文档生成与编码交给同一个工作线程，其他方法仍由异步处理器执行。"""
        if req.method in self._DOCUMENTATION_HANDLERS:
            return await asyncio.to_thread(self._encode_documentation, req)
        handler = self._HANDLERS.get(req.method)
        response = (
            await handler(self, req)
            if handler is not None
            else DaemonResponse.fail(f"Unknown method: {req.method}")
        )
        # shutdown 已请求取消服务任务，必须在下一次让出执行权之前完成确认响应的编码。
        return _encode_response(response)

    def _encode_documentation(self, req: DaemonRequest) -> bytes:
        """在同一个工作线程内生成文档并编码，避免阻塞请求事件循环。"""
        return _encode_response(self._DOCUMENTATION_HANDLERS[req.method](self, req))

    def _list_tools(self, req: DaemonRequest) -> DaemonResponse:
        tools = self.registry.list_all()
        if req.catalog_servers is not None:
            servers = set(req.catalog_servers)
            tools = [tool for tool in tools if tool.server in servers]
        include = None
        if req.catalog_view == "summary":
            include = set(ToolInfo.model_fields)
        return DaemonResponse.success(
            [t.model_dump(include=include, exclude_none=True) for t in tools]
        )

    def _search(self, req: DaemonRequest) -> DaemonResponse:
        results = self.registry.search(req.query or "", req.top_k, tool=req.tool_name)
        return DaemonResponse.success(results)

    def _grep(self, req: DaemonRequest) -> DaemonResponse:
        results = self.registry.grep(req.query or "", req.top_k)
        return DaemonResponse.success(results)

    def _describe(self, req: DaemonRequest) -> DaemonResponse:
        """处理单个或批量查看，保留名称顺序与逐工具错误隔离。"""
        if not req.tool_name:
            return DaemonResponse.fail("tool_name required")
        names = split_names(req.tool_name)
        descriptions = self.registry.describe_many(
            names, fields=req.fields, schema=req.full_schema
        )
        if len(names) == 1 and isinstance(descriptions[0], ToolDescriptionError):
            return DaemonResponse.fail(descriptions[0].error)
        results = [item.model_dump(exclude_unset=True) for item in descriptions]
        return DaemonResponse.success(results[0] if len(names) == 1 else results)

    async def _handle_call_tool(self, req: DaemonRequest) -> DaemonResponse:
        if not req.tool_name:
            return DaemonResponse.fail("tool_name required")
        try:
            meta = self.registry.resolve(req.tool_name)
        except ToolResolveError as e:
            return DaemonResponse.fail(str(e))

        if meta.source != ToolSource.MCP:
            return DaemonResponse.fail(
                "Native tools should be executed locally, not via daemon"
            )
        if not meta.server:
            return DaemonResponse.fail("MCP tool missing server")

        params = req.params or {}
        validator = self._get_validator(meta.full_name, meta.input_schema)
        if validator is not None:
            params = _coerce_to_schema(params, meta.input_schema)
            try:
                validator.validate(params)
            except jsonschema.ValidationError as e:
                return DaemonResponse.fail(f"Invalid params: {e.message}")
            except Exception:
                # schema 本身有问题（$ref 解析失败等），不拦调用
                pass

        result = await self.mcp_provider.call_tool(meta.server, meta.name, params)
        if result.status == "success":
            return DaemonResponse.success(result.data)
        return DaemonResponse.fail(result.error or "Unknown error")

    async def _handle_shutdown(self, req: DaemonRequest) -> DaemonResponse:
        self.stop()
        return DaemonResponse.success("Daemon shutting down")

    async def _handle_status(self, req: DaemonRequest) -> DaemonResponse:
        now = time.monotonic()
        uptime = now - self._start_time
        idle = now - self._last_activity
        idle_remaining = max(0.0, self._idle_timeout - idle)

        status = DaemonStatus(
            pid=os.getpid(),
            config_path=str(CONFIG_PATH),
            uptime_seconds=int(uptime),
            idle_seconds=int(idle),
            idle_timeout_seconds=int(self._idle_timeout),
            idle_remaining_seconds=int(idle_remaining),
            server_tools=self.registry.server_tool_counts(),
        )
        return DaemonResponse.success(status.model_dump())

    _DOCUMENTATION_HANDLERS = {
        "list_tools": _list_tools,
        "search": _search,
        "grep": _grep,
        "describe": _describe,
    }
    _HANDLERS = {
        "call_tool": _handle_call_tool,
        "shutdown": _handle_shutdown,
        "status": _handle_status,
    }


def run_daemon() -> None:
    """启动 daemon 进程。"""
    axi_logger = logging.getLogger("axi")
    axi_logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    )
    axi_logger.addHandler(handler)
    server = DaemonServer()
    asyncio.run(server.start())


if __name__ == "__main__":
    run_daemon()
