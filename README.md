# Akasha-Benchmark

评测 Akasha 的检索、引用归因与答案质量，提供数据集浏览、实验运行、报告和逐样本归因界面。
入库与查询通过 HTTP 访问 Akasha；审计与血缘分析可选连接只读 PostgreSQL，不修改 Akasha 源码。

## 启动

需要 Python 3.12+、uv、Node.js 和 npm。入库、编译和查询还需要可用的 Akasha 服务及其后台 worker。

```bash
uv sync
npm --prefix web ci
make setup                                # 初始化数据库
```

数据集下载在「数据集」页操作，下载完成后自动校验。
在两个终端分别运行 `make web` 和 `make serve`，前后端均启用热更新。
打开 `http://127.0.0.1:5173`；Vite 将 `/api` 请求代理到 `http://127.0.0.1:8848`。
在「归一化」页点击「开始归一化」，自动检查原始文件、归一化并验收产物。
前端构建使用 `npm --prefix web run build`，FastAPI 仅提供 API，不托管前端产物。

## 运行实验

在「配置」页填写 Akasha 连接、Akasha 模型配置、judge 和归因分析模型，再按界面顺序运行：

| 页面 | 用途 |
| --- | --- |
| 数据集、归一化 | 查看原始数据、统一样本和 gold 文档 |
| 编译层 | 抽子集、导入和编译语料，查看原文与编译结果的差异 |
| 查询层 | 选择编译批次和 Q 数量，查看逐条检索、引用与生成响应 |
| 评测层 | 选择查询记录与指标，计算并查看评测结果 |
| 归因层 | 查看链路、规则归因及可选的模型分析 |
| 任务 | 查看进度、日志，停止或清理任务 |
| 测试 | 运行小样本端到端链路测试 |

平台将阶段参数存入 `run_config`，子进程通过 `--run-config` 读取。阶段也保留命令行参数，
可用 `uv run python -m akasha_benchmark.<阶段> --help` 查看；指定 `--run-config` 时库中参数优先。

Akasha 连接在 `connection` 表中只有一份。workspace 由登录后的服务端响应确定，
入库时的连接快照和 workspace 保存在索引层。续跑会检查 workspace 和 space 身份；
embedding 配置漂移拒绝执行，compiler 漂移会警告。

要将已入库的层转到另一部署，先调用 `POST /api/layers/index/{id}/discard-ingest`（`confirm=true`）。
这会清除本地入库记录并保留子集，不删除远端 Space。

首次运行可在「测试」选择数据集和 1–5 条样本，自动完成抽样、入库编译、
响应契约与映射校验、续跑检查和评测。进度与检查结果在「任务」页查看，指标结果在「评测层」查看。
此流程会调用模型，独立创建的层与远端 Space 保留以便复查；不需要运行测试命令。

## 数据与配置

`akasha_bench.db` 是各阶段的事实来源。索引层保存子集和入库产物，查询层保存完整响应，
评测层保存指标与汇总。编译批次标签作为共同的 `run_id` 展示，查询和评测各有独立 ID，
分别关联编译批次与查询记录；查询开始前固定样本 ID，续跑使用同一批问题。索引层的 `subset_hash` 按实际文档内容计算，`config_hash` 在入库时加入模型配置。
入库和查询按库中已有记录跳过已处理条目，中断后可续跑。

连接和模型密钥在界面配置并存入数据库。库及备份包含明文凭据，已被 gitignore。
平台启动设置使用 `AKASHA_PLATFORM_HOST`、`PORT`、`DB`、`AUTH_TOKEN`
（各项均带 `AKASHA_PLATFORM_` 前缀）。host、port、db 也可通过命令行设置。
绑定非回环地址时必须设置访问令牌。

`migrations/001_initial.sql` 直接创建最终 schema。已完整应用旧版 001–006 的库会先核对
结构、备份为 `*.pre-baseline`，再合并迁移账本，业务数据不变。尚未完成旧迁移的库需要
先用旧版代码升级到 006。普通迁移同样在执行前备份，并检查已应用文件的校验和。

归一化、子集和评测加 `--export` 可输出文件快照：

```text
data/normalized/{dataset}/  samples.jsonl、corpus.jsonl、manifest.json
data/subsets/{label}/{ds}/  samples.jsonl、corpus/*.md、corpus_hashes.json
data/reports/{eval_label}/  metrics.json、per_sample.jsonl、report.md
```

常规阶段从库读取；`store.reindex` 用于从历史文件恢复产物。`make clean` 清理导出、
缓存、日志及前端产物，保留数据库和原始数据集。

## 指标口径

- 数据集声明已有标注，指标声明所需依赖。缺依赖时评测省略该指标并说明原因；narrativeqa 没有 gold 文档，不计算检索指标。
- 检索指标按 `retrievedSources` 计算。报告同时提供全样本和 `answerMode == "knowledge"` 切片；返回空来源的回落响应不能单凭零分判断为检索失败。
- 编译可能改写或遗漏原文信息。当前测量包含编译、检索与生成的共同影响，与公开基线比较前须对齐语料、样本、检索单位和输出口径。
- EM 要求归一化后的答案整串相等。解释性长答案通常得分低，但并非必然为零；需结合 F1、引用证据和人工检查解读。
- 审计诊断通过 `audit_join` 从 Akasha 的 `knowledge_query_audit` 复制到本地 `audit_record`，便于后续离线分析。

## 文档与验证

- [原始数据集](docs/datasets.md) · [归一化字段](docs/normalized_datasets.md)
- [接口约定](docs/akasha_api.md) · [指标定义](docs/metrics.md)
- [排查案例](docs/cases/README.md)
- [代码架构](docs/architecture.md)

```bash
make test
npm --prefix web run typecheck
AKASHA_LIVE=1 uv run pytest tests/test_live_lineage.py -v
```

在线用例默认跳过。单样本冒烟支持离线重放：

```bash
AKASHA_LIVE_REPLAY=data/smoke/<ts>-roundtrip.json uv run pytest tests/test_live_akasha.py
```
