# axi — Agent eXecution Interface

AI Agent 与外部系统之间的统一工具层。通过 CLI 作为万能适配器，让任何工具（MCP server、Python 函数等）变成 Agent 可发现、可搜索、可编程调用的命令。

## 代码风格

代码应当具备可读性、简洁性和高效性。

- 架构采用分层设计：底层提供基本操作和数据结构，组合后具备充分的灵活性；高层提供开箱即用的 API，足以满足大多数使用场景。
- 偏好简洁明确的函数，每个函数专注于单一任务，输入和输出类型应明确指定。
- 用最少的代码解决问题，不要任何预设之外的东西，如果你写了 200 行，而 50 行足矣，推倒重写。
- 不为不可能发生的场景写错误处理。
- 实现复杂、甚至需要额外机制，且预期收益明显不足以支撑相应代码规模与维护成本时，才构成过度设计；改动规模本身不构成判断理由。

## 依赖管理

项目使用 uv 管理；项目依赖始终使用 uv 进行管理，而非直接改 `pyproject.toml`。

## 技术栈

- Python 3.12+
- 包管理：uv (pyproject.toml)
- CLI：Typer / 数据模型：Pydantic / MCP：mcp 官方 SDK
- 搜索：bm25s + jieba（BM25） / langchain-openai + langchain-community（Embedding）

## 快速参考

- **设计原则**：渐进式披露（按需 search／describe，信息足够时直接 run）
- **工具来源**：MCP 导入（axi.json mcpServers）/ Python `@tool` 装饰器原生注册（axi.json nativeTools，对象格式 `{"module": "...", "name": "..."}`，module 支持文件路径和模块路径，name 可选自动推导）
- **MCP 执行**：通过 daemon 长连接，支持有状态 MCP server（如 browser MCP）
- **搜索策略**：Registry 对自身目录整体检索，工具路与字段路分别建库，共用文本评分和原有 RRF，交错去重后最多 K 个。CLI 查询本地 Native 与 daemon MCP 两份目录，Native 在前拼接，最多 2K；MCP meta 查询合并目录，整体最多 K。多个 MCP server 共用 daemon 目录。grep 使用相同目录边界，目录内按登记顺序匹配并整体取 K。候选和摘要规则见 [检索规则](docs/output-format.md#search-检索)，内部评分不对外返回
- **搜索配置**：CLI Native 保持 BM25；daemon MCP 和 MCP meta 合并目录的两路共用 `axi.json` 的 `search.embedding` 段，Python Registry 使用自身配置；支持 `provider`（"jina"/"openai"）、`apiKey`（可选，不填从环境变量读 `JINA_API_KEY`/`OPENAI_API_KEY`）、`model`（可选）、`baseUrl`（可选，仅 openai；Jina 使用 SDK 固定端点）
- **工具内字段搜索**：search 的 tool 参数复用当前目录字段索引与排名，限定归属后截取摘要，不另建评分、索引或配置；describe 默认看入参，fields 数组选择输出并展开一级；schema=true 直接读取完整登记定义，不构造字段文档，且不能与非 null 的 fields 同用。
- **字段摘要**：`search.maxFields` 控制每个工具的命中字段上限，默认8，0为不附带字段；不自动合并父节点。定义更新须重新注册，索引按注册版本失效。
- **输出**：统一紧凑 JSON

## 代码规范

### 项目结构

目录和模块职责统一维护在[架构设计](docs/architecture.md#目录与职责)，不在本文件重复列目录树。
技术依赖与选型见[技术选型](docs/tech-stack.md)；本文件只维护开发规范。

### 编码原则

- **Pydantic 优先**：工具元数据、明确的参数与响应契约优先使用 Pydantic model；原始 JSON Schema 和简单容器沿用 dict/list，远程响应 data 直接消费，不额外包装模型或增加前置校验
- **类型提示**：所有函数必须有完整的 type hints
- **注释与文档字符串**：使用简洁中文说明职责、关键规则和原因；API、协议与类型标识保留原文。面向 Agent 的错误和工具提示仍使用英文；用作检索或断言数据的测试文本按测试语义保留
- **JSON-first**：CLI 输出默认 JSON，复用 `_output_json()`；模型按既有选项 dump，dict/list 直接交给 JSON 序列化，保留各入口已有编码语义
- **统一输出信封**：所有 `axi run` 结果包装为 `{"status": "success"|"error", "data": ..., "error": ...}`
- **输出定义只作文档**：`output_schema` 描述完整正常成功业务返回，包含实际的 constants/columns/values 等业务层级，JSONPath 的 $ 不包含 AXI CLI 信封，仅用于搜索和查看；注册方提供正常结果模型，无需覆盖错误返回。执行与错误处理沿用上游统一入口，不追加输出校验、转换、成功判定或自动纠错
- **定义生命周期**：沿用原有工具登记机制；重新注册递增目录版本并清除该工具的字段文档缓存，search/grep 各自按版本原地重建索引。保留首次准备的简单锁，不额外保障查询期间并发更新，不自动追踪原地修改或增加完整 Schema 校验
- **共享与按需准备**：search、grep、describe 共用“JSONPath＋折叠释义＋原 Schema 连接”；字段条目只准备一次，查看只处理所选范围，fields 省略或 null 时不处理输出文档。不复制完整展开树，输出所选节点固定展开一级，选择与递归停止状态仅属于本次查看
- **声明与展示**：保留类型、自身说明、组合分支、对象和数组层级及元组位置；不推断含义、传播祖先说明或推导约束。检索命中完整返回已有字段文本；describe 先选择结构，再渲染文档字符串，终点共用折叠释义
- **契约与实现分开记录**：字段支持范围、JSONPath、选择、一级展开和返回格式以[输出数据格式](docs/output-format.md)为准；模块职责、缓存、并发与诊断机制见[架构设计](docs/architecture.md#输出字段定义与查看)。调整实现时保留既定外部语义并验证受影响入口
- **MCP 走 daemon**：MCP 工具一律通过 daemon 执行，不在 CLI 进程内直连
- **原生走进程内**：`@tool` 注册的原生工具在 CLI 进程内直接执行
- **搜索复用**：目录边界由入口决定；工具与字段分别建库，共用原有文本评分能力，工具归属只负责结果整理。保留 v0.0.13 工具路的语料、候选与同分规则，不增加字段评分公式、权重或兼容模式
- **提示面向 Agent**：所有错误信息和输出提示的读者是模型，不是人。标准：一句话说清哪里错 + 给出可执行的下一步（如 "Tool not found: X. Use search to discover available tools."）；不堆栈、不啰嗦、保持英文

### 命名约定

- MCP 工具：`axi.json` 的 `mcpServers` key 名作为 server 名，调用格式 `server/tool_name`
- 原生工具：`nativeTools` 的 `name` 字段作为 server 名（省略时自动推导：文件路径取 stem，模块路径取最后一段），调用格式同样为 `server/tool_name`

### Skills 编写与同步

`skills/` 下的 SKILL.md 是项目对外的 Agent 使用文档，由仓库复制到 Agent 的技能目录，不包含在 Python wheel 中。复制时保留各自的 `references/`；质量和同步都是代码规范的一部分。

- **同步**：提交前检查 `git diff` 中的对外行为改动，逐项核对 `axi-use`、`axi-creator` 及其引用文件；参数与 CLI `--help`、MCP 工具定义及 Registry 公共接口一致。记录实际核对结果，仓库没有自动同步命令
- **文档职责**：README 提供最小示例，guide 说明命令操作，output-format 维护完整外部契约，architecture 说明内部职责和生命周期；技能保留 Agent 操作所需信息，changelog 只记录发布功能与行为变化，避免重复维护实现细节
- **Description 即触发器**：frontmatter 的 `description` 明确列"做什么 + 何时触发 + 反向 skip 条件"；表述略 pushy（Claude 倾向 undertrigger，过于保守反而用不到）
- **渐进式披露**：SKILL.md body 以"信息密度高、便于 Agent 速读"为目标，紧凑优先（参考现有两个 skill 的 ~120 行体量）；超出时拆到 `references/` 子目录按需加载，命令级参数交给 `--help`
- **祈使句 + 解释 why**：用"先 X 再 Y"而非"应该 / 可以"；说明原因而非堆 MUST；保持通用，别绑死到某个超窄示例
- **自包含**：不引用仓库内 `docs/*`（pip 分发后不可达），兜底用 `--help` 或指向源码

