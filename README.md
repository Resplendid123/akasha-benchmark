# Akasha-Benchmark

评测 Akasha 的多跳检索、引用归因与端到端答案质量。全部代码在本仓库，
**不修改 Akasha 主仓库** —— Akasha 侧只通过 HTTP 访问。

实施计划见 [PLAN.md](PLAN.md)，数据集字段说明见 [docs/datasets.md](docs/datasets.md)。

## 先读这一条

**Akasha 的稠密/词法召回跑在 LLM 生成的文本上，不是原文。**
`knowledge_chunks` 索引的是编译器产出的 `artifact.markdown`；原文在
`knowledge_source_chunks`，该表不参与召回，只在引用解析时提供证据窗口。

后果有两个方向：Recall@k 会系统性偏低且非调参可解；但多跳可能偏高，
因为实体被物化成独立 artifact 并跨文档合并。
**因此本仓库产出的数字与已发表 baseline 直接比较是无效的**，
只在 Akasha 自身的不同配置之间横向可比。真正的对照是 PLAN.md §9 的第六轮原文基线。

## 环境

```bash
uv sync                                   # Python 3.12，勿用 3.14
uv run python download_dataset.py         # 下载四组数据到 dataset/
uv run python download_dataset.py --check # 只校验
```

轮次 3/4 需要 Akasha 跑在 docker（`db` + `redis` + `akasha`，BullMQ 必须要 redis）。
凭据走配置文件或环境变量，**不硬编码**：

```bash
cp akasha.config.example.json akasha.config.json   # 已 gitignore
# 或者用环境变量，环境变量优先：
export AKASHA_BASE_URL=http://localhost:3000
export AKASHA_EMAIL=eval@example.com
export AKASHA_PASSWORD=...
```

## 五轮流程

轮次 1、2、5 纯本地；3、4 需要 Akasha 在线。

```bash
# 轮次 1：四组归一化（全量，不抽样）
uv run python -m akasha_benchmark.normalize
uv run python scripts/validate_datasets.py          # 验收：四组全过

# 轮次 2：抽子集（先 QA 后 corpus，保证 gold 全覆盖）
uv run python -m akasha_benchmark.subset --run-id run001

# 轮次 3：入库 + 编译 + 完整性校验
uv run python -m akasha_benchmark.ingest --run-id run001

# 轮次 4：逐条跑 query，存完整响应（并发 1，不算指标）
uv run python -m akasha_benchmark.run_queries --run-id run001

# 轮次 5：离线算指标
uv run python -m akasha_benchmark.evaluate --run-id run001
uv run python -m akasha_benchmark.audit_join --run-id run001   # 可选，需要 psycopg + DB
```

轮次 3 和 4 都**可恢复**：重跑会读已有的 `page_map.jsonl` / 响应 jsonl，跳过已完成的条目。

## 产出

```
data/
  normalized/{dataset}/     samples.jsonl  corpus.jsonl  manifest.json
  subsets/{run_id}/{ds}/    samples.jsonl  corpus/{doc_id}.md  manifest.json
  ingest/{run_id}/          page_map.jsonl  manifest.json
  responses/{run_id}/       {dataset}.jsonl  manifest.json
  reports/{run_id}/         metrics.json  per_sample.jsonl  report.md  audit_join.json
```

每轮都写 manifest，记录源文件 sha256、条数、随机种子、模型配置快照。
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

轮次 3/4 的测试跑在 `httpx.MockTransport` 上，覆盖 multipart 字段名、
OWNER 闸门、质量闸门、断点续跑、失败照样落盘。真实运行仍需要在线的 Akasha。
