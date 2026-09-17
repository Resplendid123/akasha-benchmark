# Akasha-Benchmark

评测 Akasha 的检索、引用归因与答案质量，提供数据集浏览、实验运行、指标报告和逐样本归因界面。
编译与查询通过 HTTP 访问 Akasha；Judge 和模型归因调用配置的模型端点。

## 启动

需要 Python 3.12+、uv、Node.js 和 npm。编译与查询还需要可用的 Akasha 服务及其后台 worker。

```bash
make sync          # uv sync + npm install
make serve         # 后端 :8848（包含建表）
make web           # 前端 :5173
```

## 界面

先在「配置」页填 Akasha 连接并测试通过，再按左栏顺序走：

| 页面 | 用途 |
| --- | --- |
| 1 数据集 | 下载并校验原始 JSON，浏览原始样例 |
| 2 归一化 | 将样本与语料写入 SQLite，检查 gold 文档是否齐全 |
| 3 编译 | 抽取子集、创建 Akasha 空间、导入编译并检查完整性 |
| 4 查询 | 在指定编译的空间中查询，查看检索、引用与生成响应 |
| 5 评测 | 选指标与 k 计算，可选执行 Judge |
| 6 归因 | 规则根因加可选的模型叙述，逐样本走完整链路 |
| 任务 | 查看各阶段任务：进度、日志、暂停、继续、清理 |
| 配置 | Akasha 连接与模型配置、Judge 与归因端点 |
| 测试 | 1–3 条样本跑完整链路并逐段校验契约 |

任务支持暂停和继续。新建任务会创建独立运行记录；已有名称不能用于续跑，请在任务页继续原任务。编译任务最多同时运行 3 个。
续跑使用原任务保存的参数与产物 ID。暂停在阶段检查点生效，已提交给 Akasha 的编译仍会继续执行。
清理编译、查询、评测或归因记录时，会级联删除关联的下游记录；清理任务仅删除任务记录。
远端 Akasha 空间保留，`audit_log` 日志只追加。

## 数据

`akasha_bench.db` 保存配置、归一化数据和运行结果，原始数据文件保存在 `dataset/`。
四组 HippoRAG_2 数据集由数据集页下载；`itfaq`（628 条中文 IT 支持问答、42 篇文档）
是本地数据集，不在下载源里，需手动运行`process.py`进 `dataset/`。
后端启动时按 `src/akasha_benchmark/store/schema.sql` 建表。
配置可在重建前从配置页导出，重建后导入。

一次编译对应一个 `run_id`，记录抽样配置与模型快照。查询前比对当前模型配置与快照：
embedding 不一致时拒绝执行；compiler、answer 或 image 不一致时记录警告。
评测通过 `page_id` 将响应中的 `sourcePageId` 映射回语料文档。

连接凭据与模型密钥由配置页写入数据库，以明文保存；默认数据库文件已加入 `.gitignore`。
平台设置读取 `AKASHA_PLATFORM_HOST`、`AKASHA_PLATFORM_PORT`、`AKASHA_PLATFORM_DB`
和 `AKASHA_PLATFORM_AUTH_TOKEN`。设置令牌后，API 请求须携带 `X-Auth-Token`；非回环地址必须配置令牌。
`make serve` 的监听端口由 Makefile 中的 Uvicorn 参数指定，修改监听地址或端口时需同步调整该命令与 Vite 代理。

## 指标口径

- 按数据集标注检查指标依赖。缺少依赖的指标省略并记录原因，不计为 0 分；例如 narrativeqa 与 itfaq 缺少 gold 文档标注，无法计算依赖它的检索指标。
- 检索指标按 `retrievedSources` 算，不用 `citations`（后者已被裁剪过）。
- 检索指标基于响应实际返回的 `retrievedSources`；列表为空时相应检索得分为 0。报告提供全样本与 `knowledge` 子集的均值，应结合回答模式分布和 HTTP 失败数解读，均值差不能直接表示拒答比例或根因。
- Judge 跳过或失败的条目不计入评单项 Judge 的失败率超过 10% 时，评测任务标记为失败。
- 编译会改写、也可能遗漏原文信息。这里测的是编译、检索与生成的共同结果，与公开基线比较前须对齐语料、样本、检索单位与输出口径。
- EM 要求归一化后整串相等，解释性长答案得分偏低；需结合 F1、引用证据与人工抽查解读。

## 文档与验证

- [代码架构](docs/architecture.md)

```bash
make test                          # 离线测试
npm --prefix web run typecheck
```
