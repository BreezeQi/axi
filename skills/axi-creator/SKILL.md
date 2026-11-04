---
name: axi-creator
description: 将 Python 函数、HTTP API 或 SDK 封装为 axi 工具，接入已有 MCP server，或将 axi 工具导出为 MCP server。用于新增工具、注册 nativeTools、打包 axi 扩展或配置 axi mcp；封装后同时支持 CLI、Python PTC 和 MCP。仅查找和调用已有工具时使用 axi-use。
---

# axi 工具接入

把外部能力接入统一工具注册表，使 Agent 能按需搜索、查看 schema，并通过 CLI、Python PTC 或 MCP 调用。沿用用户现有配置和工具实现，接入时保留原有业务契约。

## 选择接入方式

| 起点与目标 | 做法 | 按需参考 |
|---|---|---|
| Python 函数、HTTP API、第三方 SDK | 用 `@tool` 包装，通过 `nativeTools` 加载 | 下方最小示例 |
| 需要嵌套输入、输出字段文档、参数约束或分发 Python 包 | 类型注解 / Pydantic，或 `axi.native_tools` entry points | [原生工具参考](references/native-tools.md) |
| 已有 MCP server，需要 CLI / PTC 入口 | 在当前配置中添加 `mcpServers` | [MCP 接入与导出](references/mcp.md) |
| 工具已经注册，需要 MCP 客户端访问 | `axi mcp` 元工具或平铺模式 | [MCP 接入与导出](references/mcp.md) |

只读取当前任务所需参考，不必为每次接入浏览全部配置或工具。

## 最小原生工具

创建 `tools.py`，这里用本地数据演示；真实工具可在函数体内调用 API 或 SDK：

```python
from axi import tool


@tool(
    name="query_orders",
    description="按区域查询订单金额，返回匹配订单和总数，支持 cn 区域",
    output_example={
        "orders": [{"id": 1, "region": "cn", "amount": 120}],
        "total": 1,
    },
)
def query_orders(region: str, limit: int = 10) -> dict:
    orders = [{"id": 1, "region": "cn", "amount": 120}]
    matched = [order for order in orders if order["region"] == region]
    return {"orders": matched[:limit], "total": len(matched)}
```

使用 `@tool(name="...")`，省略 description 时默认取 docstring。`tool("server/name")` 是获取调用句柄，不能当成只有名称参数的装饰器。

函数被 import 时才会注册。要让新 CLI / PTC 进程发现它，在任务选用的 `axi.json` 中合并：

```json
{
  "nativeTools": [
    {"module": "./tools.py", "name": "orders"}
  ]
}
```

```bash
export AXI_CONFIG="/absolute/path/to/axi.json"
```

默认配置是 `~/.axi/axi.json`，不自动读取当前目录。`.py` 相对路径以配置目录为基准；Python 模块路径必须在 axi 所在环境中可 import。省略 `name` 时，文件路径取 stem，模块路径取最后一段作为组名。配置与包依赖都应在实际执行工具的环境中可用。

## 让工具可发现、可组合

- 用明确的参数名和类型注解生成输入 schema；必填参数无默认值，可省略参数提供默认值。是否可传 `None` 与是否可省略是两回事。
- 工具的 description 写清能力、业务对象和适用条件，包含用户可能搜索的词。搜索索引使用完整工具名、工具级描述和已声明的输出字段文档；不索引输入参数描述。
- 参数含义和约束用类型、枚举及 `Field(description=...)` 表达，帮助 Agent 读取 schema 后正确传参。
- 按需通过 `@tool(output_schema=...)` 声明完整正常成功业务返回，接受 Pydantic 2 模型类或 JSON Schema 字典；只为固定属性、数组元素模板和元组位置提供路径、类型与自身说明。工具作者保证与实际输出一致，不合入错误返回分支。具体规则见 [原生工具参考](references/native-tools.md)。
- 返回稳定、可序列化的结构；可用 `output_example` 展示形状，减少 PTC 编写前的探测。它是示例，不是返回值校验器，返回类型注解也不会自动成为输出 schema。
- 工具成功返回业务数据，axi 负责 CLI / 元工具 run 的信封。输出定义只参与搜索和查看，不校验、转换或裁剪实际结果。不要额外套一层相同信封；执行失败抛出有意义的异常，避免把失败伪装为正常数据。
- 凭据从环境读取；需支持 HTTP 导出时的请求级变量，使用 `axi.env("API_KEY")`。保持 import 阶段只定义工具，把网络请求等执行逻辑留在函数调用中。

## 验证实际使用路径

在同一份 `AXI_CONFIG` 和目标 Python 环境中检查发现、schema 和结果。以下本地示例可以直接验证：

```bash
axi search "区域 订单"
axi describe orders/query_orders
axi run orders/query_orders --region cn --limit 5
```

为上面的工具登记 `output_schema` 后，再用 `axi describe orders/query_orders --fields '["$.orders"]'` 核对输出字段；仅提供 `output_example` 不会生成字段文档。

Python PTC 也应获得相同数据：

```python
from axi import tool

data = tool("orders/query_orders")(region="cn", limit=5)
assert data["orders"][0]["amount"] == 120
```

PTC 成功直接返回数据，失败抛 `RuntimeError`；CLI 检查 `status`，不能仅凭退出码判断工具执行成功。验证真实业务工具时使用只读或已有授权范围内的最小调用；写操作可先验证 schema 和模拟上游，不为测试额外产生业务变更。

为已接入工具新增 MCP 导出时，还应核对导出范围、模式与客户端配置，细节见 [MCP 参考](references/mcp.md)。MCP daemon 和已启动的导出进程不热加载配置，更新后按需要重启。

参数不确定时使用 `axi <command> --help`；需要查安装版本支持的配置模型时查看 `axi.config`。这些 skills 应连同各自的 `references/` 一起分发，无需依赖源码仓库中的其他文档。
