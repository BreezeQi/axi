# axi 使用指南

axi 将 Python 函数和 MCP 工具提供为可发现、可搜索、可编程调用的接口。
命令参数以 `axi <command> --help` 为准，配置字段见[配置参考](configuration.md)。

## 安装与配置

```bash
# 安装 CLI
uv tool install axi-cli

# 在 Python 项目中使用 PTC
uv add axi-cli
```

从源码开发时，在仓库根目录执行 `uv sync`，通过 `uv run axi ...` 使用当前代码。
默认配置是 `~/.axi/axi.json`，不会自动读取当前目录的同名文件。项目配置须显式指定：

```bash
export AXI_CONFIG="$PWD/axi.json"
```

CLI 与 Python 使用同一份配置；Python 解释器还须能 `import axi`。
更改配置后重启对应 daemon 或 MCP 导出进程。

## 发现与查看

已知工具及契约时直接调用；缺少信息时按需使用以下命令：

```bash
axi list
axi list orders,inventory
axi search "查询订单" -k 3
axi grep '^orders/' -l 5
axi describe orders/query_orders
```

### list

不传名称时列出 server 和工具名；指定一个 server 时返回该组工具名称与说明，多个 server 返回分组数组。
`nativeTools`、已安装包的 `axi.native_tools` entry points 和配置的 MCP server 都可贡献工具。

### search 与 grep

search 检索工具名称、说明及输出字段，按相关性推荐；grep 对这些文本做忽略大小写的全文正则匹配，工具按登记顺序返回。
CLI 分别查询本地 Native 和 daemon MCP 目录，Native 在前直接拼接；`-k`、`-l` 分别限制每份目录，最多返回两倍上限。
多个 MCP server 共用 daemon 名额，目录之间不补足名额。MCP meta 与 Python Registry 对自身目录整体取上限。

| 命令 | 查询参数 | 数量参数 | 默认上限 |
| --- | --- | --- | --- |
| search | `query`，自然语言或字段位置 | `--top-k` / `-k` | 每份目录5个 |
| grep | `pattern`，正则表达式 | `--limit` / `-l` | 每份目录10个 |

结果包含工具名称、说明、来源、路径根与命中字段，例如：

```json
[{"name":"orders/query_orders","description":"按区域查询订单","source":"native","matched_fields":["$.orders[*].id: int；订单编号"]}]
```

CLI Native 使用 BM25；daemon MCP 和 MCP meta 合并目录可通过 `search.embedding` 配置语义检索。
`search.maxFields` 默认8，0只隐藏摘要，不影响字段召回。完整评分、交错推荐、精确路径及摘要规则见[检索规则](output-format.md#search-检索)。

限定已知工具的字段：`axi search "订单编号" --tool orders/query_orders`。复用已有字段排名，只返回该工具的命中摘要，数量仍由 maxFields 控制。

### describe

```bash
# 业务说明和完整入参；省略 fields 或传 null 均不处理输出文档
axi describe orders/query_orders

# 浏览输出根目录
axi describe orders/query_orders --fields '["$"]'

# 查看两个已知字段
axi describe orders/query_orders --fields '["$.orders[*].id","$.total"]'

# 浏览容器的直接子字段，下一层容器保留紧凑释义
axi describe orders/query_orders --fields '["$.orders"]'

# 需要完整输出结构、分支或约束时读取已登记的标准 Schema，不查询业务数据。
axi describe orders/query_orders --schema

# 空输出视图，不返回入参
axi describe orders/query_orders --fields '[]'

# 批量查看，保留名称顺序与逐项错误
axi describe orders/query_orders,inventory/get_stock
```

`--fields` 为输出定义的 JSONPath 数组；省略或 null 查看入参，非 null 数组只查看输出。
输出选择固定展开一级，显式深层路径仍可见，不提供单次查看全部层级的参数。
优先通过 search、grep 获取路径与完整说明；已知工具可用 `axi search "订单编号" --tool orders/query_orders` 定位字段，信息足够时直接调用。
需要同时查看入参和输出时，分别发出不带 fields 和 `--fields '["$"]'` 的请求，可以并行。

`output_fields` 是阅读用文档字符串，外层响应仍是 JSON；Agent 按 JSONPath 访问业务数据，不解析说明文本。
输出定义未登记时标记 `not_registered`；未知路径返回 `unknown_fields` 和最近有效父节点的 `available_fields`。
选择并集、数组、元组、组合及递归规则见[输出字段契约](output-format.md)。

使用当前入口发现的完整名称调用。CLI/PTC 优先本地；MCP meta 使用其合并目录的有效定义。
完整名称跨来源冲突时，不保证不同入口选中同一定义，切换入口前应重新确认。

## 调用

### CLI

```bash
axi run orders/query_orders --region cn --limit 5
axi run orders/query_orders --json '{"region":"cn","limit":5}'
```

`--key value` 按入参 Schema 解析：字符串字段保留原文，其他字段尝试 JSON 解码，失败则保留字符串。
`--json` / `-j` 用于完整对象或嵌套参数。

```json
{"status":"success","data":{"orders":[],"total":0}}
```

```json
{"status":"error","error":"Invalid params: ..."}
```

读取 `status` 判断执行结果；业务工具失败不一定产生非零退出码。搜索返回摘要数组，describe 返回文档，不能将所有命令输出都当作 run 信封。

### Python PTC

```python
from axi import tool

query = tool("orders/query_orders")
result = query(region="cn", limit=5)
# result 是业务数据本身；失败抛出 RuntimeError
```

沿用同一调用句柄进行批量查询，在脚本中关联、过滤或聚合完整业务结果：

```python
results = [query(region=region, limit=5) for region in ("cn", "jp")]
orders = [order for result in results for order in result["orders"]]
total_amount = sum(order["amount"] for order in orders)
```

句柄是同步函数。原生工具在当前进程执行，MCP 工具经 daemon 调用，无需为每次调用启动 CLI。
需要并发或接入已有 async 程序时，将同步句柄交给线程，并按上游限制控制并发：

```python
import asyncio
from axi import tool

query = tool("orders/query_orders")

async def read_orders():
    return await asyncio.to_thread(query, region="cn", limit=5)

result = asyncio.run(read_orders())
```

已有事件循环时直接在 async 函数中使用 `await asyncio.to_thread(...)`，不要嵌套 `asyncio.run()`。
返回结构明确时直接编写分析脚本；信息不足时查看定义，必要时对只读工具小规模试调。

## 接入工具

MCP 在当前配置的 `mcpServers` 中声明；Python 模块在 `nativeTools` 中声明，或由已安装包的 entry points 提供：

```json
{
  "mcpServers": {
    "retrieval": {"url":"http://127.0.0.1:8020/mcp"}
  },
  "nativeTools": [
    {"module":"my_project.tools"},
    {"module":"./scripts/tools.py","name":"scripts"}
  ]
}
```

原生函数通过 `@tool` 提供业务能力；可选 `output_schema` 接受 Pydantic 2 模型类或 JSON Schema 字典。
MCP 连接时读取上游 `outputSchema` 或 AXI 元数据。字段文档仅用于搜索和查看，由注册方保证与输出一致，不参与返回校验或裁剪。
接入步骤与完整配置见[README](../README.md)和[配置参考](configuration.md)。

## 导出为 MCP

```bash
# 四个元工具：search / grep / describe / run
axi mcp

# 平铺指定组；有输出定义时另提供 axi_describe
axi mcp --server orders

# 限定组但仍使用元工具
axi mcp --server orders --meta
```

flat 模式按 tools/list 中的导出名调用。meta 模式搜索合并目录，整体最多 K 个工具；与 CLI 的两份目录拼接不同。
传输选项见 `axi mcp --help`，内部职责见[架构](architecture.md)。

## Daemon 与排障

需要访问 MCP 时自动启动 daemon，纯本地发现、查看和调用不经过它。

```bash
axi daemon start
AXI_CONFIG=/absolute/path/to/axi.json axi daemon start
axi daemon status
axi daemon stop
axi doctor
```

每份配置文件路径对应独立实例，运行文件位于 `~/.axi/daemons/<hash>.sock`、`.pid`、`.log`、`.lock`。
daemon 默认空闲30分钟后退出，下次需要时自动启动；`stop` 等待目标完成清理后返回。

- `status` 为 starting 时仍在连接上游，避免重复启动。
- doctor 检查配置、工具来源、MCP 连接和 Embedding 配置；它可能启动 daemon，不逐个试调业务工具。
- 远程发现失败时可能只返回本地结果，同时检查 stderr；daemon 中的检索降级原因查看对应日志。
- 修改配置或升级后停止旧 daemon，并重启已有 MCP 导出进程。超时不表示上游操作未发生，不盲目重试有副作用的调用。
- 超时、空闲退出及 Embedding 配置详见[配置参考](configuration.md)。
