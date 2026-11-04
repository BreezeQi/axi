# 字段说明与查看

先通过 search、grep 获取字段路径和完整说明；已足够时直接调用。缺少参数时 describe，仍需局部结构时再选择输出容器。
已知工具使用 `axi search "里程" --tool server/name`，仅搜索该工具字段，沿用现有排名、精确定位与 maxFields 上限。

## 读取命中信息

search、grep 的 `matched_fields` 使用同一份“JSONPath: 类型；自身说明”文本，例如：

```json
{
  "name": "fleet/query_vehicle",
  "matched_fields": [
    "$.vehicles[*].plateNumber: str；车辆编号",
    "$.vehicles[*].summaryStatistic.forwardMileage: float；前进里程，单位米"
  ]
}
```

按共同的 `vehicles[*]` 层级关联车牌和里程，编写脚本处理每条车辆记录。
JSONPath 的 `$` 固定指向完整正常成功业务值，CLI 信封先取 data。
业务自身的 data/result/record 字段保留在路径中，AXI 不自动剥离或解码。

| 位置 | JSONPath |
| --- | --- |
| 属性与对象数组字段 | `$.speed`、`$.vehicles[*].speed` |
| 标量元素／固定元组位置／声明的尾部 | `$[*]`、`$.position[1]`、`$.position[2:]` |
| 名称含点号、空格或引号 | `$["a.b"]`、`$['vehicle name']`；名称字符串中的引号和反斜线须转义 |

`$["a.b"]` 是一个键，`$.a.b` 是两个层级。路径只描述 Schema 声明的位置，不按实际数组长度枚举；AXI 不对真实数据执行 JSONPath。
父容器、元素和叶字段独立匹配，不继承祖先说明或自动展开后代；无说明时只有类型。
命中后完整返回条目，不按查询裁剪类型分支。同工具同完整文本去重，同路径不同释义分别保留，不承诺它们同时出现。

## 选择与展开

`describe(name, fields=None, schema=False)` 选择入参、局部输出或完整输出定义，不把这些视图同时返回。

| 当前缺口 | 操作 |
| --- | --- |
| 不知道工具 | `search(query)` 或 `grep(pattern)` |
| 不清楚入参 | `describe(name)`；省略 fields 或传 null |
| 浏览输出目录 | `describe(name, fields=["$"])` |
| 已知工具，寻找指标 | `search(query, tool=name)` |
| 已知路径，查看局部结构 | `describe(name, fields=[path])` |
| 需要完整输出结构、分支或约束 | `describe(name, schema=True)`；CLI 使用 `--schema` |
| 空输出视图 | `describe(name, fields=[])`，不返回入参 |

schema 默认 false，原有 fields 语义不变；true 时 fields 须省略或为 null，包括 `[]` 在内的其他值均冲突。
完整视图保留身份与业务说明，返回 output_schema_status 和完整的 output_schema JSON 对象，不同时附带入参、紧凑 output_fields、类型、路径提示或示例。
定义原样保留引用、分支和约束，不查询业务数据。未登记为 not_registered 且不附定义，显式空 Schema 为 registered 和 `{}`。需要完整定义时才使用，不作为每次调用的前置步骤。

例如列式分页工具：

```bash
axi search "车辆前进里程"
axi describe fleet/query_carrier_efficiency
axi search "后退里程" --tool fleet/query_carrier_efficiency
axi describe fleet/query_carrier_efficiency --fields '["$.constants","$.columns","$.values"]'
```

按当前缺口选择操作，不要求逐项执行。搜索已有足够的路径、类型和说明时直接编写调用与分析脚本。
同时缺入参和输出目录时，可并行请求 `describe(name)` 与 `describe(name, fields=["$"])`。
fields 直接复制路径，不带类型和说明。支持属性、带单／双引号的名称选择器、`[*]`、固定元组 `[n]` 和声明尾部 `[n:]`；不支持过滤、递归下降、属性通配或一般切片。
父子范围取并集，顺序无关；通配与固定位置的子选择也合并。普通数组选 `[*]`，不按实际索引选 `[0]`。
所选节点固定展开一级，子容器保持折叠，不提供可变深度或全部展开参数。
例如输出根目录为 `{rows: list[dict]}`，选择 `$.rows` 时为 `{rows: [{value: float；读数}]}`。祖先、数组模板与引用／组合不额外增加展开层级。
选择 `$` 仍只看根下一级。显式深层路径仍可见，并可进入自动递归折叠的分支。
JSONPath 最多64个属性／数组位置层级；第64层可检索和选择，查看在此折叠。引用与组合包装不占数据层级，真实循环仍会停止自动展开。

展开节点显示子结构，不重复父说明；终点使用自身类型、已有 description 及本地引用目标说明，递归停止不丢说明。
组合保持 anyOf/oneOf/allOf 与分支说明，例如 `anyOf(float；温度, str；设备状态)`；不推导交集或矛盾。anyOf/allOf 简化完全重复的释义和无独立说明的同类包装，oneOf 保留分支与嵌套。

output_fields 是阅读用字符串，外层仍为 JSON，例如：

```json
{"output_fields":"{reading: anyOf({temperature: float；温度}, [{status: str；状态}])}"}
```

文本不执行或解析。对象、数组和组合保留原层级；元组带位置标签，如 `[[0]: float；纬度, [1]: float；经度]`，局部选择不重编号或补位。
特殊名复用名称转义，如 `{["a.b"]: float}`；业务字段名 anyOf 等仍作为普通名称。仅选深层路径时保留适用分支；已选父范围内其余分支按一级规则展开或折叠。
已登记输出定义时，fields=[] 的 output_fields 为 `"{}"`，不隐藏输出类型或已有示例。

未知路径在 `unknown_fields` 中完整返回，`available_fields` 仅提示最近有效父节点的下一层位置；元组通配可具体化为固定位置或尾部。
通配路径在任一适用声明中存在即已知，显式固定位置独立判断。未登记 Schema 返回 `not_registered`，不将请求路径判成未知。
批量名称按顺序返回，错误只影响对应项；检查每项的 error，继续使用成功项。未配置 MCP 时，未知工具不启动 daemon。

## 检索数量与顺序

| 入口 | 目录与上限 |
| --- | --- |
| CLI | Native、daemon MCP 各最多 K 个，Native 在前拼接，不转移名额；所有 MCP server 共用 daemon 名额 |
| MCP meta | 合并目录整体最多 K 个 |
| Python Registry | 自身目录整体最多 K 个 |

每份目录中，工具说明和字段分别检索，按“工具第1、字段第1、工具第2、字段第2……”交错去重取 K；字段按最强命中推荐工具，不累计数量加分。
工具路为空时可仅靠字段召回。各入选工具独立附带已有命中中最相关的前 M 条，即使它未进入字段路前 K；M=`search.maxFields`，默认8，0仅隐藏摘要，不改变工具集合与顺序。
精确字段名／JSONPath 查询优先展示对应节点，不强制工具置顶；自然语言按普通相关性排列。单／双引号及等价转义定位同一声明。
grep 忽略大小写，对工具名、description、Schema 根说明分别匹配，`^` / `$` 锚定各自文本；JSONPath 的正则特殊字符须转义。
grep 工具按登记顺序，实际命中字段按数据层级从浅到深稳定排序后截取，同层保留声明顺序，不补父节点或展开后代。
根说明参与字段路召回，不占摘要名额；没有字段摘要或 M=0 不代表工具没有相关数据。

CLI Native 保持 BM25，daemon MCP 和 MCP meta 按配置使用 Embedding。建库或查询降级查看对应服务日志；远程发现失败时保留本地结果，CLI 同时在 stderr 提示。
定义更新须重新注册，修改配置或升级须重启对应进程。CLI 跨目录同名可同时返回，调用优先本地；MCP 合并目录同名由后注册的 MCP 覆盖，不承诺跨入口同一定义。

## 调用边界

output_schema 只作文档，由作者提供完整正常成功业务返回，保留 constants/columns/values/page 等真实层级，不用解码后的记录冒充实际返回；CLI 外层信封不属于业务返回。AXI 不增加返回校验、裁剪、自动解码、保存结果或脚本重试。
分析前可保存原始结果，结构不符时检查样本修正脚本，不将“没取到字段”直接当作业务零值；动态字典不枚举实际键名。
含输出定义的 flat MCP 提供 tools/list 中的 `axi_describe`（冲突时带数字后缀）；传入实际导出名，成功详情的 name 可直接调用，不附内部 server。
