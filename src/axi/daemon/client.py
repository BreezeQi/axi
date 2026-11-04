"""daemon 客户端：CLI 侧通过 Unix socket 与 daemon 通信。"""

import asyncio
import fcntl
import logging
import os
import subprocess
import sys
import time
from typing import Any

from axi.config import CONFIG_PATH, app_config
from axi.daemon.protocol import (
    LOCK_PATH,
    LOG_PATH,
    SOCKET_DIR,
    SOCKET_PATH,
    PID_PATH,
    STREAM_LIMIT,
    DaemonRequest,
    DaemonResponse,
)
from axi.models import (
    DescribeOptions,
    ToolInfo,
    ToolMeta,
)

logger = logging.getLogger(__name__)

# 等待 daemon 就绪的上限。daemon 要先连上所有 MCP server 才创建 socket，
# 官方 MCP 由 uvx 拉起、首次还要下载依赖，冷启动远超旧的 3 秒窗口。
_DAEMON_START_TIMEOUT = 60.0  # 秒
_DAEMON_START_POLL_INTERVAL = 0.1  # 秒

# 等待 daemon 退出（关 MCP 连接、删 PID / socket）的上限
DAEMON_STOP_TIMEOUT = 60.0  # 秒


def read_live_pid() -> int | None:
    """PID 文件里的进程仍存活时返回 pid，否则 None。"""
    try:
        with open(PID_PATH) as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)
    except (OSError, ValueError):
        return None
    return pid


def is_daemon_running() -> bool:
    """daemon 已就绪（进程存活且 socket 已创建）时返回 True。"""
    return read_live_pid() is not None and os.path.exists(SOCKET_PATH)


def wait_for_exit(pid: int, timeout: float = DAEMON_STOP_TIMEOUT) -> bool:
    """等待目标 daemon 完成退出清理，超时返回 False。

    PID 文件消失或被新实例替换时，目标已释放运行时文件。
    固定目标 PID，避免并发 start 后转而等待新实例退出。
    """
    deadline = time.monotonic() + timeout
    while read_live_pid() == pid:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(_DAEMON_START_POLL_INTERVAL, remaining))
    return True


def _spawn_and_wait() -> bool:
    """拉起 daemon 子进程并轮询直到就绪。调用方需持有启动锁。"""
    with open(LOG_PATH, "a") as log_file:
        proc = subprocess.Popen(
            [sys.executable, "-m", "axi.daemon.server"],
            stdout=log_file,
            stderr=log_file,
            start_new_session=True,
            env={**os.environ, "AXI_CONFIG": str(CONFIG_PATH)},
        )

    return _wait_for_ready(proc)


def _wait_for_ready(proc: subprocess.Popen | None = None) -> bool:
    """等待新拉起或已在初始化的 daemon，不重复启动。"""
    deadline = time.monotonic() + _DAEMON_START_TIMEOUT
    while time.monotonic() < deadline:
        if is_daemon_running():
            return True
        # 子进程已退出：socket 不会再出现，别把超时时间白等完
        if proc is not None and proc.poll() is not None:
            logger.error(
                "Daemon exited with code %s during startup. Check log: %s",
                proc.returncode,
                LOG_PATH,
            )
            return False
        if proc is None and read_live_pid() is None:
            logger.error("Daemon exited during startup. Check log: %s", LOG_PATH)
            return False
        time.sleep(_DAEMON_START_POLL_INTERVAL)
    logger.error(
        "Daemon did not become ready within %.0fs. Check log: %s",
        _DAEMON_START_TIMEOUT,
        LOG_PATH,
    )
    return False


def ensure_daemon() -> bool:
    """确保 daemon 已启动。未运行时自动启动，返回是否就绪。

    并行的多个 axi 进程可能同时发现 daemon 未运行。用文件锁串行化启动：
    只有拿到锁的进程 spawn，其余进程阻塞在锁上，拿到锁后复查发现已就绪即返回。
    否则会多进程各拉起一个 daemon、互相 unlink socket、覆盖 PID 文件。
    """
    if is_daemon_running():
        return True

    os.makedirs(SOCKET_DIR, exist_ok=True)

    with open(LOCK_PATH, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # 锁内复查：可能刚才另一个进程已经把 daemon 拉起来了
        if is_daemon_running():
            return True
        if read_live_pid() is not None:
            return _wait_for_ready()
        return _spawn_and_wait()


def send_request(req: DaemonRequest) -> DaemonResponse:
    """向 daemon 发送请求并获取响应。"""
    return asyncio.run(_send(req))


def daemon_request(req: DaemonRequest) -> DaemonResponse:
    """确保 daemon 已启动后发送请求。所有入口（CLI/PTC/MCP serve）统一走这里。"""
    if not ensure_daemon():
        return DaemonResponse.fail(f"Failed to start axi daemon. Check log: {LOG_PATH}")
    return send_request(req)


def _documentation_request(request: DaemonRequest) -> Any:
    """统一处理发现与文档请求的通信失败，业务调用仍沿用原处理。"""
    try:
        response = daemon_request(request)
    except (OSError, ValueError) as exc:
        detail = str(exc).partition("\n")[0]
        raise RuntimeError(
            f"Daemon {request.method} request failed: {detail}. "
            f"Check daemon log: {LOG_PATH}"
        ) from exc
    if response.status == "error":
        raise RuntimeError(response.error or "")
    return response.data


def fetch_tool_catalog(*, servers: list[str] | None = None) -> list[ToolMeta]:
    """获取并转换完整工具定义；daemon 请求失败以 RuntimeError 交给调用方。"""
    data = _documentation_request(
        DaemonRequest(method="list_tools", catalog_servers=servers)
    )
    return [ToolMeta.model_validate(item) for item in data or []]


def fetch_tool_summaries(*, servers: list[str] | None = None) -> list[ToolInfo]:
    """仅获取工具身份和说明，不传输或补造 Schema。"""
    data = _documentation_request(
        DaemonRequest(
            method="list_tools", catalog_view="summary", catalog_servers=servers
        )
    )
    return [ToolInfo.model_validate(item) for item in data or []]


def fetch_discovery(request: DaemonRequest) -> list[dict[str, Any]]:
    """直接使用 search、grep 的响应数据，不额外校验或包装响应。"""
    return _documentation_request(request) or []


def fetch_description(name: str, options: DescribeOptions) -> dict[str, Any]:
    """逐工具获取说明，保留 daemon 返回的文档或对应错误。"""
    try:
        return _documentation_request(
            DaemonRequest(
                method="describe",
                tool_name=name,
                fields=options.fields,
                schema=options.full_schema,
            )
        )
    except RuntimeError as exc:
        return {"error": str(exc) or f"Tool not found: {name}"}


async def _send(req: DaemonRequest) -> DaemonResponse:
    try:
        reader, writer = await asyncio.open_unix_connection(
            SOCKET_PATH, limit=STREAM_LIMIT
        )
    except OSError as e:
        return DaemonResponse.fail(
            f"Cannot connect to daemon: {e}. Try: axi daemon stop && axi daemon start"
        )

    try:
        writer.write(req.model_dump_json(by_alias=True).encode() + b"\n")
        await writer.drain()

        timeout = app_config.daemon.request_timeout
        line = await asyncio.wait_for(reader.readline(), timeout=timeout)
        if not line:
            return DaemonResponse.fail("Daemon connection closed unexpectedly")

        return DaemonResponse.model_validate_json(line)
    except asyncio.TimeoutError:
        return DaemonResponse.fail(
            f"Daemon request timed out after {timeout}s. Raise it via "
            "AXI_REQUEST_TIMEOUT or axi.json daemon.requestTimeout."
        )
    finally:
        writer.close()
        await writer.wait_closed()
