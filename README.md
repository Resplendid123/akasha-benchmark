# Akasha-Benchmark

评测 Akasha 的多跳检索、引用归因与端到端答案质量。全部代码在本仓库，
**不修改 Akasha 主仓库** —— Akasha 侧只通过 HTTP 访问。

实施计划见 [PLAN.md](docs/PLAN.md)，数据集字段说明见 [docs/datasets.md](docs/datasets.md)，
接口请求／响应样例见 [docs/akasha_api.md](docs/akasha_api.md)，
评测指标见 [docs/metrics.md](docs/metrics.md)。

## 先读这一条

**Akasha 的稠密/词法召回跑在 LLM 生成的文本上，不是原文。**
`knowledge_chunks` 索引的是编译器产出的 `artifact.markdown`；原文在
`knowledge_source_chunks`，该表不参与召回，只在引用解析时提供证据窗口。

后果有两个方向：Recall@k 会系统性偏低且非调参可解；但多跳可能偏高，
因为实体被物化成独立 artifact 并跨文档合并。
**因此本仓库产出的数字与已发表 baseline 直接比较是无效的**，
只在 Akasha 自身的不同配置之间横向可比。真正的对照是 PLAN.md §9 的原文基线。

## 环境

```bash
uv sync                                   # Python 3.12，勿用 3.14
uv run python scripts/download_datasets.py         # 下载四组数据到 dataset/
uv run python scripts/download_datasets.py --check # 只校验
```

入库和查询需要 Akasha 跑在 docker（`db` + `redis` + `akasha`，BullMQ 必须要 redis）。
凭据走配置文件或环境变量，**不硬编码**：

```bash
cp akasha.config.example.json akasha.config.json   # 已 gitignore
# 或者用环境变量，环境变量优先：
export AKASHA_BASE_URL=http://localhost:3000
export AKASHA_EMAIL=eval@example.com
export AKASHA_PASSWORD=...
```

## 评测流程

归一化、子集、评测纯本地；入库和查询需要 Akasha 在线。

```bash
# 归一化：四组全量，不抽样
uv run python -m akasha_benchmark.normalize
uv run python scripts/validate_datasets.py          # 验收：四组全过

# 子集：先 QA 后 corpus，保证 gold 全覆盖
uv run python -m akasha_benchmark.subset --run-id run001

# 入库：导入 + 编译 + 完整性校验
uv run python -m akasha_benchmark.ingest --run-id run001

# 查询：逐条跑 query，存完整响应（并发 1，不算指标）
uv run python -m akasha_benchmark.run_queries --run-id run001

# 评测：离线算指标
uv run python -m akasha_benchmark.evaluate --run-id run001
uv run python -m akasha_benchmark.audit_join --run-id run001   # 可选，需要 psycopg + DB
```

入库和查询都**可恢复**：重跑会读已有的 `page_map.jsonl` / 响应 jsonl，跳过已完成的条目。

## 产出

```
data/
  normalized/{dataset}/     samples.jsonl  corpus.jsonl  manifest.json
  subsets/{run_id}/{ds}/    samples.jsonl  corpus/{doc_id}.md  manifest.json
  ingest/{run_id}/          page_map.jsonl  manifest.json
  responses/{run_id}/       {dataset}.jsonl  manifest.json
  reports/{run_id}/         metrics.json  per_sample.jsonl  report.md  audit_join.json
```

每个阶段都写 manifest，记录源文件 sha256、条数、随机种子、模型配置快照。
JSON 走「临时文件 + `os.replace`」原子写，中断不会留下截断文件。

## 几条贯穿全程的约束

- **显式适配器**：每个数据集一个类，按名字分派。禁止按字段存在性猜 schema。
- **严格模型**：`extra="forbid"` + `frozen=True`，多余字段报错而非忽略。
- **capability 声明**：narrativeqa 无 gold，不声明 `EVIDENCE_RECALL`；
  请求检索指标时**抛异常**，不返回 0.0 —— 假分数会污染汇总。
- **身份不推断**：口径收在 `datasets/models.py` 的 `CORPUS_ID_RULES` /
  `SAMPLE_ID_RULES` 一处。按 ID 匹配后再比一次 question 文本，不一致报错。
- **corpus 不去重**：musique 有 647 个重复 title 但它们是不同段落，
  去重会丢 gold。去重前后条数都进 manifest。

## 读指标时注意

`no_match` 和 `general` 两种 answerMode 会**无条件**返回空的 `retrievedSources`
（`ai-knowledge-chat.service.ts:641,667`），所以这些行的检索分数天然是 0，
与检索实际找到了什么无关。报告里每张检索表都出两份 —— 全样本、
以及只算 `answerMode == "knowledge"` 的切片，差值就是生成端拒答的规模。

`retrievalDiagnostics` 不在 HTTP 响应里（controller 解构排除了它），
分层归因必须走 `audit_join`（读 `knowledge_query_audit.metadata`）。
连接键是 `sha256:<hex>`，**带 `sha256:` 前缀**。

## 测试

```bash
uv run pytest tests/ -q
```

入库和查询的测试跑在 `httpx.MockTransport` 上，覆盖 multipart 字段名、
OWNER 闸门、质量闸门、断点续跑、失败照样落盘。

替身是照我们对接口的理解写的，所以理解错了它也照样绿。要验**服务端**的实际行为，
跑单样本在线冒烟 —— 一条样本从入库走到查询，逐条核对
[akasha_api.md](docs/akasha_api.md) 里的字段约定：

```bash
make smoke                 # 等价于 AKASHA_LIVE=1 uv run pytest tests/test_live_akasha.py -v
```

默认整个模块 skip，所以 `make test` 不需要 Akasha 在线。建的都是 `smoke` 前缀的
独立 Space（不碰 `bench` 那几个），跑完删掉。两组用例各管一件事：

| 用例 | 覆盖 | 成本 |
| --- | --- | --- |
| 单样本往返（23 项） | 刻意绕开阶段代码直接打 HTTP，逐条核对字段约定 | 约 5 分钟 |
| 批量三段（9 项，`-k pipeline`） | 走真实的 `ingest.run()` / `run_queries.run()` / `evaluate.run()`，覆盖批量、**续跑**与报告产出 | 约 9 分钟 |

续跑是入库和查询各跑两遍，验第二遍一条都不重发、产物行数不增长 —— 这类缺陷不报错，
只会让 `page_map` 出现重复行或白烧一遍 LLM 调用。

全过程写进 `data/smoke/`。单样本那趟可以离线重放，不必再烧 LLM 调用：

```bash
AKASHA_LIVE_REPLAY=data/smoke/<ts>-roundtrip.json uv run pytest tests/test_live_akasha.py
```

批量三段没有重放（它验的是阶段入口对真实服务的行为），但产物同样落盘，
含一份可直接阅读的 `<ts>-report.md`。

规模可调：`AKASHA_LIVE_PIPELINE_SAMPLES`（默认 3）、`AKASHA_LIVE_DISTRACTORS`（默认 3）、
`AKASHA_LIVE_KEEP=1` 保留 Space 不删。
