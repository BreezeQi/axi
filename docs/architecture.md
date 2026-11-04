# axi 架构

## 分层总览

```text
CLI / Python PTC / MCP 导出
             │
        Executor.run
         ┌───┴─────────────┐
         │                 │
     原生 Python        daemon client
      进程内执行          Unix socket
                           │
                       axi daemon
                     Registry + MCPProvider
                           │
                     MCP server 长连接
```

## 目录与职责

以下源码路径相对于 `src/axi/`；本页统一维护模块目录与内部职责。

| 路径 | 职责 |
| --- | --- |
| `__init__.py` | 公开 `@tool`、同步 PTC 句柄和 `env` |
| `cli.py` | CLI 参数解析、本地／daemon 目录编排和 JSON 输出 |
| `runtime.py` | CLI 与 Python PTC 共用的进程内 Registry、Executor |
| `registry.py` | 登记定义、按版本重建索引、共享输出文档和工具查询 |
| `models.py` | 工具定义、摘要、describe 参数与响应、执行信封 |
| `executor.py` | 统一原生与 MCP 执行分发 |
| `mcp_serve.py` | flat／meta 工具导出、名称映射与传输 |
| `config.py`、`context.py` | 进程配置及请求级环境变量 |
| `daemon/` | `client.py` 负责通信，`protocol.py` 定义协议，`server.py` 持有 MCP 目录与连接 |
| `providers/` | `native.py` 发现与注册 Python 工具，`mcp.py` 适配 MCP 定义和连接 |
| `search/` | `corpus.py` 准备语料与归属，BM25／Embedding 负责文本评分，`hybrid.py` 整理双路结果，`regex.py` 执行全文正则；分词和向量缓存分别由 `tokenize.py`、`cache.py` 提供 |
| `output/` | 注册转换、JSONPath、共享 Schema 文档、结构选择与文本渲染，详见[输出字段定义与查看](#输出字段定义与查看) |

仓库根目录的 `tests/` 保存离线回归；`docs/` 分开维护操作、外部契约与内部设计，
`skills/` 保存可独立分发的 Agent 使用文档及其 `references/`。
`tests/output/` 按注册转换、终点释义、字段枚举、查看选择和引用语义归组，复用与复杂度回归单独核验。
Schema 行为矩阵放在共享层测试；入口测试核对参数传递、返回协议、数量边界和错误隔离。

## 核心模块

### Daemon（后台守护进程）

维持所有 MCP server 的长连接，通过 Unix socket（`~/.axi/daemons/<hash>.sock`）接受 CLI 请求。

职责：
- 管理 MCP server 连接生命周期
- 托管 MCP 工具的 Registry
- 提供工具目录、search、grep 和 describe；通过 call_tool 调用对应 MCP server
- 支持需要长连接的 MCP server（如 browser MCP）

#### 自动启动

CLI 需要读取远程目录、查看或调用远程工具时，`client.py` 的 `ensure_daemon()` 会：
1. 检查当前配置对应的 PID 和 socket，已就绪则复用
2. 未就绪时获取配置专属的启动文件锁，锁内复查；PID 仍存活则继续等待现有初始化进程，否则通过 `python -m axi.daemon.server` 后台启动
3. 每 0.1 秒轮询就绪状态，最多等待 60 秒；进程提前退出时立即失败

#### 启动流程

`DaemonServer.start()` 依次执行：
1. 加载 `axi.json` 配置，初始化 MCPProvider，按配置创建 Registry 及可选 Embedding provider；各检索索引按需构建
2. 注册 SIGTERM / SIGINT 处理，确保初始化阶段也能停止
3. 创建 `~/.axi/daemons/`，清理当前配置的旧 socket，再写入 `<hash>.pid`
4. 连接 MCP server；每条连接由固定后台任务持有上下文，分页获取全部工具并注册到 Registry；单个 server 失败时关闭其连接并继续
5. 启动 Unix socket server（`~/.axi/daemons/<hash>.sock`），请求/响应读取上限均为 8 MiB
6. 启动 idle watchdog 任务

MCP 请求可以在不同请求任务中执行；重连和关闭通过连接的持有任务完成，避免跨任务退出 AnyIO 上下文。并发故障请求共用一次重连，关闭期间的失败请求不会重新拉起连接。

#### Idle Watchdog

daemon 内置空闲检测机制，防止无限占用资源：

- 每 **60 秒**检查一次空闲时间
- 每次处理请求时更新 `_last_activity` 时间戳（`status` 和 `shutdown` 请求除外，不重置计时器）
- 空闲超过 `idleTimeoutMinutes`（默认 30 分钟）后自动执行 `stop()`

#### 关闭流程

三种关闭方式，均执行相同的清理逻辑：

| 触发方式 | 说明 |
|----------|------|
| `axi daemon stop` | CLI 发送 `shutdown` 请求，daemon 优雅退出 |
| idle 超时 | watchdog 检测到空闲超时，自动关闭 |
| SIGTERM / SIGINT | 操作系统信号触发优雅关闭 |

清理步骤：关闭 socket listener → 取消 watchdog → 等各 MCP 连接的持有任务完成关闭 → 删除当前配置的 socket/PID → 注销信号处理。

CLI 的 `stop` 等待目标 PID 完成清理后才报告成功，上限 60 秒；PID 被并发启动的新实例替换时不会转而等待新实例。

#### 通信协议

CLI 和 daemon 通过 Unix socket 以 JSON Line 格式通信（每行一个完整 JSON 对象）：

- **请求**（`DaemonRequest`）：`method` + 可选参数（`tool_name`、`params`、`query`、`top_k`、`catalog_view`、`catalog_servers`、`fields`、`schema`）
- `search` / `grep` 在 daemon 的完整 MCP 目录内执行，只返回命中摘要。
- `list_tools.catalog_view` 区分完整定义和摘要：list 只读取工具身份和 description；MCP 导出读取完整定义。指定 server 时通过 `catalog_servers` 在 daemon 序列化前过滤，避免传输无关 Schema；null 表示全部，空数组表示空目录。`describe.fields` 省略或 null 查看入参，数组只查看选中输出
- 导出已请求 MCP 目录但仍有未知 server 时，补读全目录摘要以提示可用名称；未选中任何已配置的 MCP server 时不请求 daemon，提示只使用已加载的本地目录。摘要读取失败时沿用已知名称。
- **响应**（`DaemonResponse`）：`status`（"success" / "error"）+ `data` / `error`
- 客户端请求超时：默认 120 秒，可通过配置或 `AXI_REQUEST_TIMEOUT` 调整
- 客户端按成功／失败状态消费响应。`fetch_tool_catalog()` 逐项转为完整 `ToolMeta` 定义，`fetch_tool_summaries()` 逐项转为仅含身份与说明的 `ToolInfo`，不补造 Schema；search/grep 和单工具 describe 直接使用响应 data，不增加整批类型、数量检查或文档校验失败后的重启策略。
- 文档请求的连接中断、响应行超限和外层 JSON 解析失败也使用相同回退；不增加重试或提高传输上限，底层通信与业务执行请求保留原有异常行为。
- 目录、search、grep 和 describe 的构造及 JSON Line 编码在同一次工作线程调用内完成；search/grep 的摘要模型直接交给响应编码，不先构造中间字典。call_tool、status、shutdown 沿用直接编码，确保关闭确认在退出前写出。

### Registry（工具注册中心）

所有工具的索引，存储工具元数据（name, description, input_schema, provider 来源等），支持搜索。

职责：
- 维护工具元数据
- 提供搜索接口（默认 BM25 关键词搜索，可选 Embedding 语义搜索，支持正则匹配）
- 渐进式披露：search 返回能力摘要与命中字段释义，describe 返回入参与按需展开的输出字段

工具分别由两份 Registry 管理：
- `runtime.py` 的 Registry：由 CLI / Python PTC 共用，管理原生 Python 工具
- daemon 进程内的 Registry：管理 MCP 工具

CLI 在进程内 Registry 检索 Native，由 daemon 检索 MCP，只传回摘要；未配置 MCP 时不启动 daemon。
MCP meta 服务沿用 Native、MCP 依次注册的合并 Registry，完整名称相同时按原注册机制由后者替换；整体建库、评分和取 K。
Python Registry 同样对自身目录整体检索。CLI 两份目录直接拼接，同名结果可同时返回；实际调用沿用 Native 优先。
来源顺序及名额见[目录边界](output-format.md#目录边界)，排名见[检索规则](output-format.md#search-检索)。
daemon 对完整 MCP 目录建立索引，仅在注册版本变化后重建。
MCP 搜索配置在 daemon 启动时加载，修改后须重启 daemon。

Registry 管理以下状态：

| 状态 | 归属与生命周期 |
| --- | --- |
| 注册定义 | `register()` 直接登记元数据、清除该工具的字段文档缓存并递增版本；`set_server()` 沿用此机制 |
| 字段文档 | 已登记输出 Schema 按需准备 `SchemaDocument`，字段条目在文档锁内生成并缓存，供 search/grep 共用；未登记时使用空条目。失败后不保存结果，下次调用可重试 |
| 搜索引擎 | search 的双路引擎和 grep 的正则引擎各记录构建版本；下次使用时发现目录版本变化，再原地重建 |

调用方更新定义后须重新注册，Registry 不自动追踪字典原地修改。CLI、daemon 和 MCP 服务在启动时注册工具，
没有自动刷新工具清单的入口。索引只保留工具摘要、字段文档和归属，不持有原始 Schema。
MCP 导出组装合并目录时逐项调用 register()，各目录独立管理版本与索引，不复制新旧定义来隔离查询。
定义更新应与查询顺序执行，不保证运行中的查询与重新注册并行时的新旧版本隔离。
search 与 grep 使用独立构建锁，避免首次并发查询重复建库或请求 Embedding；字段准备使用普通缓存和文档锁，首次准备期间其他文档读取可能等待。

`Registry.from_search_config` 沿用原有流程，按配置初始化可选 Embedding，覆盖该 Registry 的整个目录；初始化失败直接报错。
CLI runtime 使用未配置 Embedding 的 Registry，Native 保持 BM25。daemon 与 MCP meta 分别把配置用于 MCP 目录与合并目录。
索引仍在首次检索时准备；Embedding 建库失败时报告原因并使用 BM25，查询失败只对本次请求降级。
daemon 与 MCP 服务将目录、检索、文档查看及这些响应的编码交给工作线程，避免阻塞请求事件循环。

Embedding 缓存沿用原有配置内容哈希，API key 不参与标识；使用原有文件读写流程。
两路独立建库和处理失败，共用文本向量缓存；各路仅在补充缺失向量后保存缓存。
缓存保存失败沿用建库失败的 BM25 降级。相同文本共享原始向量及建库时计算的长度，保留各自的文档归属，不另建矩阵副本或改写落盘向量。
每次查询共用一次分词和一个可选 Embedding 向量，各路独立排名；同一路中共享向量的文档在本次查询内复用余弦分数，仍保留各自位置和归属。
每路每次查询只计算一次查询向量长度；余弦公式和稳定排序沿用 0.0.13，查询准备与相似度结果不跨请求缓存。
BM25 建库复用相同文本的预处理结果，保留每份文档的统计贡献；空语料只影响对应检索路。

通用文本搜索器负责建库和评分，返回文本序号及分数；`HybridSearch._rank_routes` 共享查询准备并并行获取两路排名，`search` 负责交错推荐和字段归属。
工具路在调用线程执行，字段路使用一个临时工作线程，按固定路序收集结果；字段语料为空时不启动工作线程。
工具文本与工具摘要保持同序，直接用序号取结果；字段文档单独保存所属工具及释义。

候选选择、RRF、精确匹配、摘要数量和排序统一见[检索规则](output-format.md#search-检索)。
结果整理直接复用已有字段排名，不再次检索或计分。grep 共用字段文档，按目录登记顺序匹配，使用独立的正则引擎和构建锁。

### 输出字段定义与查看

原生装饰器和 MCP provider 统一将输出文档保存到 `ToolMeta.output_schema`。
MCP provider 分开处理分页获取与单个工具定义转换；输出文档错误只忽略该工具的输出定义，保留工具及后续分页。
`registration.py` 按 serialization 模式转换 Pydantic 模型类，JSON Schema 字典直接沿用；Registry 按原机制登记定义。
文件工具沿用文件名作为模块名；Pydantic 输出模型的延迟注解在注册时通过函数的名称空间解析，不改变模块导入或业务执行行为。
搜索、grep 和查看共享注册定义的字段文档信息：JSONPath＋折叠释义＋展开关系。展开关系直接沿原 Schema 的属性、元素、元组、组合和引用连接读取，不复制完整展开树；不改写 Schema 或影响真实调用。

| 模块 | 职责 |
| --- | --- |
| `output/registration.py` | 转换 Pydantic/JSON Schema 文档，定义 MCP 导入与导出共用的 AXI 元数据键；不检查完整 Schema 约束或保存校验记录 |
| `output/schema.py` | 本地引用、共享类型及单层连接读取；`view_structure` 规范化查看所需的类型列表和引用旁结构声明，字段枚举保留原声明 |
| `output/explanation.py` | 终点类型与数组元素读取，格式化类型和自身 description |
| `output/view.py` | 选择小型分支展示树；选择结果集中记录、合并相对诊断并生成提示，再独立格式化终点 |
| `output/describe.py` | 组装工具详情、输出示例及未知字段提示 |
| `output/documents.py` | 注册定义的 `SchemaDocument`：复用节点释义和单层连接；`shape(for_view=False)` 供字段枚举读取原声明，默认规范化引用与类型列表供查看；提供可用目录，按需准备字段条目，不参与检索打分 |
| `output/paths.py` | 检索与查看共用的 JSONPath 解析及名称转义 |
| `output/selection.py` | 选择树归一化、原始子选择连接及展开深度计算 |

`schema.py` 提供原始声明读取和查看所需的单层结构规范化，不解析引用链或生成释义。
`SchemaDocument` 持有注册定义的引用，管理别名解析和准备结果，不另复制 Schema。
Registry 按定义对象持有这个文档上下文，search、grep 和 describe 通过同一入口取得它。

```text
注册定义 → SchemaDocument
           ├─ 原 Schema 连接：properties / items / prefixItems / 组合 / $ref
           ├─ 已生成的折叠释义与引用目标
           └─ 按需准备一次 FieldDocuments → search / grep

describe（fields 省略时只看入参）→ fields 选择并展开一级的小型展示树 → 终点读取同一份折叠释义
```

1. **检索**：首次需要索引时，`SchemaDocument.fields` 按声明出现位置枚举 JSONPath，生成文本与数据层级。
   仅在属性或数组位置生成条目，引用和组合沿当前位置继续读取；同次遍历保留根说明索引范围，子字段只使用自身说明。`build_corpus` 沿用这些字段条目建立索引和精确查找；
   search/grep 命中后直接返回完整文本，不再解析 Schema 或按查询裁剪组合。search 指定 tool 时只执行现有字段路，在截取候选前限定归属，不另建单工具索引。
2. **查看**：`DescribeSelection.prepare` 统一 fields，`select_view` 通过 `SchemaDocument.shape` 沿原连接选择所需结构。
   结构读取只返回原 Schema 子节点连接，不逐字段包装上下文；字段筛选和祖先引用状态由选择器管理，未知路径提示只枚举父节点的直接声明。
   不归并不同组合分支或元组位置，数组连接保留 `[*]`、`[n]`、`[n:]`。查看结果统一持有一份共享文档上下文，终点只保留原定义，`render_view` 从该上下文读取折叠释义。
   只查看一个字段不会触发全部字段枚举；fields 省略或 null 时不取得输出文档上下文。
   终点和展开结果复用同一个组合器，整个所选视图直接渲染为文档字符串；不构造对外递归字典或解析已格式化文本。

复用与状态的边界：

| 状态 | 生命周期与约束 |
| --- | --- |
| 终点释义、本地引用目标及别名后缀 | 随注册定义复用；复用前检查当前引用链与数据层级预算，不保存祖先造成的停止状态 |
| 单层连接 | 普通节点共享原连接；引用和类型列表分别保存原始连接及查看连接，子定义直接引用原 Schema |
| 字段条目 | 首次检索时在文档锁内准备，search/grep 共用；不同出现位置保留各自路径，同路径不同有效释义不覆盖 |
| 选择、递归状态、展示片段和诊断 | 仅属于当前请求；缓存片段使用相对路径，并记录影响展开的引用依赖，支持显式选择复用 |

数据路径最多64层，每进入属性或数组位置增加一层；索引保留第64层条目，查看在该层显示终点释义。更深的显式路径由 JSONPath 解析拒绝。同位置的引用和组合包装不消耗层级，
不因别名链或组合包装超过64层而丢失类型、说明或后代字段。真实引用循环由当前引用链保护。
显式路径可进入自动折叠的节点；同位置的组合分支沿用当前引用链，不能把其他分支曾出现的引用当成循环。
父子选择并集分别传递范围和一级展开状态：显式子路径继续进入，选中节点的直接子容器保留终点释义。
`child_selection` 遇到对象属性后停止自动展开，数组模板、元组位置及尾部沿用当前状态。
`data_depth` 沿每个属性或数组位置递增，仅用于上述64层数据位置边界。
通配与固定位置使用原选择节点的连接，不复制选择树。未知路径汇总后，只读取最近有效父节点的下一层目录。

`prepare_describe()` 创建请求内 `ToolDescriber`，供 Registry 和 CLI 本地查看使用。
详情组装按 schema、fields 与登记状态选择响应字段：schema=true 直接导出完整定义，不取得字段文档；否则 fields 省略或 null 时只看入参，传数组时构造局部输出。Schema 容器随导出复制，同批次重复名称复用组装结果，响应独立复制。
文档锁同步释义与连接准备，不包含 Embedding I/O；首次字段枚举期间，其他读取同一文档的请求可能等待。
重新注册只清除对应工具的准备结果，并使检索索引按目录版本失效，不追踪字典原地修改。

CLI 按名称顺序逐项解析，本地未找到时请求 daemon describe，远程失败只影响对应项。
`ToolInfo` 共享身份字段，`ToolMeta` 保存定义，`ToolDescription` 保存查看结果，响应入参 Schema 与登记对象隔离。
示例沿用原有 Python 模式 dump；MCP flat 说明沿用原有 JSON 编码及文字回退。
CLI 解析 MCP 键值对入参时使用 `describe(name)`，失败则以空 Schema 继续原有解析，执行仍交给 Executor。

字段与路径语义、展示格式和支持边界见[输出字段契约](output-format.md)。

### Executor（统一执行入口）

CLI、Python PTC 和 MCP 导出统一调用 `Executor.run()`：
- 解析本地工具名称，原生工具按 Pydantic 模型校验后在进程内执行
- MCP 工具通过 `daemon_request()` 调用，统一转成 RunResult
- 未在本地找到的名称交给 daemon 解析；名称歧义直接返回错误

短名只在当前目录内判断歧义：CLI 的 describe/run 和 Python PTC 优先解析本地目录，
未命中才交给 daemon 的 MCP 目录；MCP 元工具在其导出目录内解析。两个来源分别存在
`local/read` 和 `remote/read` 时，CLI 的 `read` 可命中本地工具，而同时导出两者的
MCP 元工具会要求完整名称。使用当前入口发现结果中的完整 `name` 调用。完整名称跨来源冲突时，CLI/PTC 优先本地，MCP 合并目录由后注册的定义覆盖；切换入口时须核对其有效定义。

入口只处理自己的协议：CLI 解析参数和输出 JSON，PTC 解包结果，MCP 导出检查工具范围、注入请求变量并转换 MCP 响应。三者不再分别维护执行分发逻辑。

### Provider（工具来源适配器）

不同工具来源的适配器，将外部工具转换为 axi 内部统一表示。

已实现的 Provider：
- **MCP Provider**（`providers/mcp.py`）：连接 MCP server，读取 tool definition，管理连接池
- **Native Provider**（`providers/native.py`）：`@tool` 装饰器注册的原生 Python 函数

未来可扩展：
- OpenAPI Provider：从 OpenAPI spec 自动生成工具
- 其他

## 命令结构

```
axi
├── list                # 列出工具与 server
├── search              # BM25 / 可选 Embedding 混合搜索
├── grep                # 全文正则搜索
├── describe            # 查看入参与按需展开的输出定义
├── run                 # 执行工具
├── mcp                 # 导出为 MCP server
├── doctor              # 检查配置与运行状态
└── daemon
    ├── start           # 启动 daemon
    ├── stop            # 停止 daemon
    └── status          # 查看 daemon 状态
```

## 执行流程

```text
CLI 参数解析 / PTC 调用 / MCP 请求
                 │
           Executor.run(name, params)
                 │
        ┌────────┴─────────┐
        │                  │
  原生参数校验与执行    daemon_request
        │                  │
        │             MCPProvider.call_tool
        └────────┬─────────┘
              RunResult
```
