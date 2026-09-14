# Akasha-Benchmark

评测 Akasha 的检索、引用归因与答案质量，提供数据集浏览、实验运行、指标报告和逐样本归因界面。
编译与查询通过 HTTP 访问 Akasha；归因的链路视图可选连接只读 PostgreSQL，不修改 Akasha 源码。

前后端分离：**所有操作都在前端点，后端执行**。项目不保留脚本或命令行入口。

## 启动

需要 Python 3.12+、uv、Node.js 和 npm。编译与查询还需要可用的 Akasha 服务及其后台 worker。

```bash
make sync          # uv sync + npm install
make serve         # 后端 :8848，建表在启动时自动完成
make web           # 另一个终端，前端 :5173
```

打开 `http://127.0.0.1:5173`，Vite 把 `/api` 代理到 `http://127.0.0.1:8848`。
`make build` 构建前端产物；FastAPI 只提供 API，不托管前端。

## 界面

先在「配置」页填 Akasha 连接并测试通过，再按左栏顺序走：

| 页面 | 用途 |
| --- | --- |
| 1 数据集 | 下载并校验原始 JSON，看未经解释的原始样例 |
| 2 归一化 | 经适配器入 SQLite（不写 Akasha），验收 gold 是否都在语料内 |
| 3 编译 | 抽子集、随机建一个 Akasha 空间、导入编译、过质量闸门 |
| 4 查询 | 选一次编译的空间跑 query，看逐条检索、引用与生成响应 |
| 5 评测 | 选指标与 k 计算，可选执行 Judge |
| 6 归因 | 规则根因加可选的模型叙述，逐样本走完整链路 |
| 任务 | 实时观测六层的任务：进度、日志、暂停、继续、清理 |
| 配置 | Akasha 连接、它那边的模型配置、judge 与归因端点 |
| 测试 | 1–3 条样本跑完整链路并逐段校验契约 |

每一层都可以**暂停、继续、清理**。暂停停在一个已落库的边界上，继续时接着跑；
清理删该层及其下游的数据库内容，远端的 Akasha 空间不删。
操作日志统一进 `audit_log`，**只追加**，清理任务记录不会删它。

## 数据

`akasha_bench.db` 是每一层的事实来源，schema 在 `src/akasha_benchmark/store/schema.sql`，
后端启动时自动建表。遇到重构前的旧库会先改名留档（`*.legacy-<时间戳>`）再建新表。

一次编译对应一个 `run_id`，它上面固化这次的抽样配置与模型快照。查询前拿现在的模型配置
与那份快照比对：embedding 变了拒绝执行（旧 chunk 永远召回不到），compiler 变了只警告。
`page_id` 是贯穿链路的钥匙 —— 评测靠它把 `sourcePageId` 反查回语料文档。

连接与模型密钥在界面配置并存进数据库。库含明文凭据，已 gitignore。
启动设置读 `AKASHA_PLATFORM_HOST` / `PORT` / `DB` / `AUTH_TOKEN`；
绑非回环地址时必须设访问令牌。

## 指标口径

- 数据集声明**拥有**什么标注，指标声明**需要**什么，闸门做集合比对。缺依赖时省略该指标并写明原因，**不伪造 0 分** —— 假分数会污染任何包含它的汇总。narrativeqa 没有 gold 文档，整族检索指标省略。
- 检索指标按 `retrievedSources` 算，不用 `citations`（后者已被裁剪过）。
- `no_match` 与 `general` 无条件返回空 `retrievedSources`，得分按定义为 0。报告同时给全样本与 `knowledge` 切片，差值就是生成端拒答的规模，不是检索失败。
- Judge 失败该条排除而不是记 0，另叠失败率闸门：排除得太多时那个均值已不代表整体。
- 编译会改写、也可能遗漏原文信息。这里测的是编译、检索与生成的共同结果，与公开基线比较前须对齐语料、样本、检索单位与输出口径。
- EM 要求归一化后整串相等，解释性长答案得分偏低；需结合 F1、引用证据与人工抽查解读。

## 文档与验证

- [原始数据集](docs/datasets.md) · [归一化字段](docs/normalized_datasets.md)
- [接口约定](docs/akasha_api.md) · [指标定义](docs/metrics.md) · [代码架构](docs/architecture.md)

```bash
make test                          # 离线测试
npm --prefix web run typecheck
```

端到端验证走界面的「测试」页：它会真实建空间、编译、查询、评测、归因，并逐段校验契约。
产物保留在平台里 —— 一次真实的链路记录比一份「通过」的报告有用。
