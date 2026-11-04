"""独立 MCP 测试服务，用于验证真实 stdio 连接与调用。"""

import asyncio
import os
from pathlib import Path

from mcp.server import Server
from mcp.server.context import ServerRequestContext
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequestParams,
    CallToolResult,
    ListToolsResult,
    PaginatedRequestParams,
    TextContent,
    Tool,
)

TOOLS = [
    Tool(
        name=name,
        description=name,
        input_schema={
            "type": "object",
            "properties": properties,
            "required": list(properties),
        },
    )
    for name, properties in [
        ("echo", {"message": {"type": "string"}}),
        ("add", {"a": {"type": "number"}, "b": {"type": "number"}}),
        ("boom", {}),
    ]
]


async def list_tools(
    ctx: ServerRequestContext, params: PaginatedRequestParams | None
) -> ListToolsResult:
    if os.environ.get("AXI_TEST_LIST_ERROR"):
        raise RuntimeError("list tools failed")
    if os.environ.get("AXI_TEST_PAGINATED"):
        if params is None or params.cursor is None:
            return ListToolsResult(tools=TOOLS[:1], next_cursor="rest")
        return ListToolsResult(tools=TOOLS[1:])
    return ListToolsResult(tools=TOOLS)


async def call_tool(
    ctx: ServerRequestContext, params: CallToolRequestParams
) -> CallToolResult:
    arguments = params.arguments or {}
    if params.name == "echo":
        text = arguments["message"]
    elif params.name == "add":
        text = str(arguments["a"] + arguments["b"])
    elif params.name == "boom":
        text = "kaboom"
    else:
        text = f"Unknown tool: {params.name}"
    return CallToolResult(
        content=[TextContent(type="text", text=text)],
        is_error=params.name not in ("echo", "add"),
    )


server = Server("test-server", on_list_tools=list_tools, on_call_tool=call_tool)


async def main() -> None:
    if pid_path := os.environ.get("AXI_TEST_PID_PATH"):
        Path(pid_path).write_text(str(os.getpid()))
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
