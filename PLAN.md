# Akasha-Benchmark 实施计划

评测 Akasha 的多跳检索、引用归因与端到端答案质量。全部代码在本仓库实现，
**不修改 Akasha 主仓库**。Akasha 侧只通过 HTTP 接口访问。

参考实现 `D:\VSCodeProjects\SAG-Benchmark`（下文 `SAG/` 前缀均指该处）只作设计参考，
不 import、不依赖其代码。

本文吸收并替代 `HANDOFF.md`。确认无遗漏后可删除该文件（它未被 git 跟踪，
删除不可恢复，请自行确认）。

---

## 0. 已核实的事实

以下全部由实际读文件测得，非推测。**其中三项推翻了 HANDOFF.md 的判断**，已标注。

### 0.1 数据现状

| 数据集 | QA 行 | corpus 行 | 原生 QA ID | corpus `idx` | gold 来源 |
| --- | --- | --- | --- | --- | --- |
| hotpotqa | 1000 | 9811 | `_id` (1000/1000 唯一) | **有**（int） | `supporting_facts` |
| 2wikimultihopqa | 1000 | 6119 | `_id` (1000/1000 唯一) | **无** ⚠️ | `supporting_facts` |
| musique | 1000 | 11656 | `id` (1000/1000 唯一) | **无** ⚠️ | `paragraphs[].is_supporting` |
| narrativeqa | 293 | 4111 | **无** | 有（str） | **无** |

⚠️ **修正 HANDOFF #1**：handoff 称「corpus 文件统一是 `{title, text}`（外加一个 `idx`）」。
实测 2wiki 与 musique 的 corpus **没有 `idx` 字段**，只有 `{title, text}`。
这两组必须在归一化阶段自行赋 id。

### 0.2 gold → corpus 对齐（HANDOFF 决定 A 的答案）

handoff 说这个检查从未跑过。已跑：

| 数据集 | gold title 全部命中 corpus | corpus 重复 title | 结论 |
| --- | --- | --- | --- |
| hotpotqa | **1000/1000 (100%)** | 0 | title 可唯一定位 |
| 2wikimultihopqa | **1000/1000 (100%)** | 0 | title 可唯一定位 |
| musique | **1000/1000 (100%)** | **647 个（涉 2465 行）** ⚠️ | title **不可**唯一定位 |

⚠️ **修正 HANDOFF #2**：handoff 担心「gold 文档字符串未必能在 corpus 精确命中」。
实测 title 级 100% 命中，这个担心在 title 层面不成立。

但 musique 有 647 个重复 title，**title 级匹配对 musique 是有歧义的**。
额外测得：musique 的 `paragraphs[].paragraph_text` 与 corpus `text`
**逐字节精确相等**（exact=2648 / whitespace_only=0 / MISMATCH=0）。
所以 musique 的对齐键是 `(title, text)` 而非 title。

**决定 A 定稿**：全链路走 id，零字符串匹配。各数据集的 corpus 行身份如下——

- hotpotqa：用原生 `idx`
- 2wiki：用 corpus 数组行号（归一化时赋予）
- musique：用 corpus 数组行号；`title → idx` 有歧义时用 `(title, text)` 消歧
- narrativeqa：用原生 `idx`（形如 `{document_id}_{chunk_seq}`）

### 0.3 重复 question 文本

| 数据集 | QA 行 | 唯一 question | 重复 |
| --- | --- | --- | --- |
| hotpotqa | 1000 | 1000 | **0** |
| 2wikimultihopqa | 1000 | 1000 | **0** |
| musique | 1000 | 999 | 1 |
| narrativeqa | 293 | 293 | 0 |

⚠️ **修正 HANDOFF #3**：handoff 称「hotpotqa 这类数据集确实存在重复 question 文本」。
本批快照实测 hotpotqa 重复为 0，四组合计仅 musique 有 1 例。

这条修正有实际后果：Akasha 审计表按 `queryHash = sha256(query)` 连接，
原本担心的哈希碰撞几乎不存在，**第五轮可以安全地 join 审计表拿分层归因指标**
（musique 那 1 例单独排除即可）。

### 0.4 其他实测细节

- **gold 数量分布**：hotpotqa `{2:650, 3:253, 4:81, 5:13, 6:1, 7:2}`；
  2wiki `{2:765, 4:234, 5:1}`；musique `{2:518, 3:316, 4:166}`
- **gold 内部重复**：hotpotqa 有 350 行、musique 有 47 行的 gold title 列表内部有重复
  （`supporting_facts` 是 `(title, 句子下标)` 对，同一 title 多个句子）。
  归一化时 **必须去重成集合**
- **musique 跳数**：`id` 前缀编码跳数 —— `2hop:518, 3hop1:243, 3hop2:73,
  4hop1:108, 4hop2:27, 4hop3:31`。可直接用于按跳数分层报告指标
- **2wiki 的 `evidences`** 是关系三元组（`[["Lothair II","mother","Ermengarde of Tours"]]`），
  **不是** gold 文档，禁止当检索 ground truth
- **narrativeqa**：`answer` 恒为 2 条人工参考的列表；10 篇文档 293 个问题；
  每篇切 152–926 个 chunk；`document.text` 内联约 210KB（QA 文件 94MB 的原因）
- **musique 独有字段**：`answer_aliases`、`answerable`、`question_decomposition`
- **2wiki 独有字段**：`answer_id`、`entity_ids`、`evidences_id`

### 0.5 Akasha 侧接口事实

- **导入**：`POST /api/pages/import`，multipart，字段 `file` + `spaceId`。
  接受 `.md/.html/.docx/.pdf`，单文件上限 30MB。**返回创建的 page 对象（含 `id`）**
- **title 规则**（[import.service.ts:106](../Akasha/apps/server/src/integrations/import/services/import.service.ts#L106)）：
  优先取 Markdown **首个 heading** 作为 title 并**从正文移除**；无 heading 才退回文件名
- **查询**：`POST /api/llm-wiki/query`，body `{query, spaceIds[], type?, scoreThreshold?, chatContext?}`
- **没有只检索不生成的 HTTP 端点**。`retrieveOnly` 存在于 service 但未暴露。
  每条 query 都会调生成模型
- **`retrievalDiagnostics` 不在 HTTP 响应里**。controller 用
  `const { retrievalDiagnostics, retrievalScope, ...response }` 解构排除了它，
  只写入 `knowledge_query_audit.metadata`
- **批量导入会 `skipKnowledgeCompile`**，但逐个 `pages/import` 不会 ——
  它走 `insertPage` → `PAGE_CREATED` → 1 小时静默期调度。
  用 `POST /api/llm-wiki/admin/compile-spaces` 可绕过静默期立即建 Run
- **静默期**：`KNOWLEDGE_PAGE_COMPILE_QUIET_PERIOD_MS = 3600000`，
  且每次编辑**重置**到期时间（`eligible_at = greatest(旧值, now+1h)`）

### 0.6 一个必须写进报告的架构事实

**Akasha 的稠密/词法召回跑在 LLM 生成的文本上，不是原文。**

`knowledge_chunks` 索引的是编译器产出的 `artifact.markdown`；原文存在
`knowledge_source_chunks`，该表**不参与召回**，只在引用解析时提供原文证据窗口。

后果：
1. Recall@k 会系统性偏低，且非调参可解 —— 这是架构决定的
2. 但多跳可能偏高，因为实体被物化成独立 artifact 并跨文档合并，
   两个 hop 可能被编译器直接连成一条 graph edge

**因此与已发表 baseline 直接比数字是无效的。** 唯一有意义的对照是第六轮
（见下），本计划的五轮只产出 Akasha 自身的横向可比数字。

---

## 1. 环境

- Python 3.12（`.python-version` 已 pin，**勿用 3.14**）
- Akasha 跑在 docker：`db`(pgvector/pg18) + `redis` + `akasha` 三个服务。
  参考主仓 `docker-compose.yml.bak`（当前 `docker-compose.yml` 只有 db，缺 redis，
  而 BullMQ 必须要 redis）
- 依赖尽量少。**不要引入 spacy** —— 本评测不需要，它是 SAG 里最大的体积约束
- 所有 Akasha 连接参数（base URL、账号、workspace/space id、API key）
  走配置文件或环境变量，**不硬编码**

新增依赖预期：`pydantic`（严格模型）、`httpx`（HTTP 客户端）、
`numpy`（指标聚合）。`psycopg[binary]` 仅第五轮 join 审计表时需要。

---

## 2. 目录结构

```
src/akasha_benchmark/
  datasets/
    models.py          规范化样本模型 + capability 枚举
    base.py            适配器 Protocol
    registry.py        名称/别名注册表
    resolver.py        数据集名 -> 文件路径 + 适配器
    common.py          hotpotqa/2wiki 共用 supporting_facts 解析
    hotpotqa.py  two_wiki.py  musique.py  narrativeqa.py
    corpus.py          corpus 加载 + 行身份赋予 + 去重口径
  normalize.py         轮次 1 入口
  subset.py            轮次 2 入口
  ingest.py            轮次 3 入口
  run_queries.py       轮次 4 入口
  metrics/
    retrieval.py       Recall@k / nDCG@k / MRR / Hit@k
    qa.py              EM / F1 + normalize_answer
    attribution.py     citation precision/recall
    multihop.py        graph-neighbor 贡献率 / 跳数覆盖
  evaluate.py          轮次 5 入口
  io_utils.py          原子写 JSON/JSONL + sha256
  config.py            配置加载

data/
  normalized/{dataset}/            轮次 1 产出（全量）
    samples.jsonl
    corpus.jsonl
    manifest.json
  subsets/{run_id}/{dataset}/      轮次 2 产出
    samples.jsonl
    corpus/{doc_id}.md
    manifest.json
  ingest/{run_id}/
    page_map.jsonl                 doc_id -> akasha_page_id
    manifest.json
  responses/{run_id}/{dataset}.jsonl   轮次 4 产出（原始响应全文）
  reports/{run_id}/                 轮次 5 产出

scripts/
  validate_datasets.py             轮次 1 的校验脚本
```

---

## 3. 贯穿全程的设计约束

这几条来自 SAG 的经验，建议照做：

1. **显式适配器，拒绝 schema 猜测。** 每个数据集一个适配器类，自己声明名字、
   别名、校验规则。禁止写 `if "supporting_facts" in row: ... elif "paragraphs" in row:`
   这种按字段存在性分派 —— 数据集换版多个字段就会静默走错分支
2. **规范化模型严格校验。** pydantic `ConfigDict(extra="forbid", frozen=True)`，
   多余字段报错而非忽略，构造后不可变
3. **capability 声明。** narrativeqa 不声明 `EVIDENCE_RECALL`；请求该指标时
   **显式抛异常**，不要返回空列表算出 0.0 —— 假分数会静默污染汇总
4. **身份不许推断。** 有原生 ID 用原生 ID；narrativeqa 用全量数据集行号字符串，
   且**这个口径必须收到一处常量**（SAG 在预处理与评测两侧各写一份靠注释互指，
   这点是脆弱的，别学）。重复 ID 直接报错。按 ID 匹配后**再比一次 question 文本**，
   不一致报错 —— 能抓到预测文件与数据集版本不匹配
5. **manifest + sha256。** 每轮产出都写 manifest，记源文件绝对路径、sha256、
   条数、limit、生成时刻。JSON 写入走「临时文件 + `os.replace`」原子写
6. **corpus 去重口径全链路统一。** SAG 有个不一致（`get_docs()` 去重、
   `save_as_markdown()` 不去重，导致两套数字不可比）——**不要复制**。
   本计划统一：**不去重**，因为 musique 有 647 个重复 title 但它们是不同段落，
   去重会丢失 gold。去重前后条数都记进 manifest

---

## 4. 轮次 1：四组归一化

**目标**：把 4 组原始数据转成统一的 `samples.jsonl` + `corpus.jsonl`，全量，不抽样。

### 4.1 规范化样本模型

```python
class Capability(StrEnum):
    EVIDENCE_RECALL = "evidence_recall"
    ANSWER_EM_F1 = "answer_em_f1"

class CanonicalSample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dataset: str
    sample_id: str            # 全局唯一: f"{dataset}:{dataset_sample_id}"
    dataset_sample_id: str    # 原生 ID，或 narrativeqa 的全量行号字符串
    question: str
    answers: tuple[str, ...]  # 统一成元组，单答案也是长度 1
    gold_doc_ids: tuple[str, ...]  # corpus 行身份；narrativeqa 为空
    metadata: dict[str, Any]  # hop 数、level、type 等，不参与校验
```

`answers` 统一成元组解决 handoff 提到的类型不一致
（hotpotqa/2wiki/musique 是 str，narrativeqa 是 list）。
**musique 的 `answer_aliases` 并入 `answers`**（去重后），
这样 EM/F1 取 max 时自动覆盖别名。

### 4.2 corpus 行身份

`corpus.py` 负责赋 `doc_id`，规则按 0.2 定稿：

- hotpotqa：`str(row["idx"])`
- 2wiki：`str(数组行号)`
- musique：`str(数组行号)`
- narrativeqa：`row["idx"]`

同时建两张反查表：
- `title -> [doc_id]`（可能一对多）
- `(title, text) -> doc_id`（唯一，用于 musique 消歧）

**建表时如果 `(title, text)` 仍有重复，直接报错**，不静默取一个。

### 4.3 各适配器的 gold 解析

- **hotpotqa / 2wiki**：`supporting_facts` 取 title，**去重成集合**，
  再经 `title -> doc_id` 映射。`common.py` 抽公共逻辑，
  句子拼接符作参数（hotpotqa `""`、2wiki `" "`）——
  虽然本计划用 id 匹配不需要拼接文本，但保留该参数供第六轮基线用
- **musique**：取 `is_supporting` 段落，用 `(title, paragraph_text) -> doc_id` 映射
  （不能只用 title，见 0.2）
- **narrativeqa**：不解析 gold，不声明 `EVIDENCE_RECALL`

### 4.4 产出

`data/normalized/{dataset}/`：
- `samples.jsonl` —— 每行一个 `CanonicalSample`
- `corpus.jsonl` —— 每行 `{doc_id, title, text}`
- `manifest.json` —— 源文件路径+sha256、QA 条数、corpus 条数、
  去重前后条数、适配器版本、capability 列表、生成时刻

### 4.5 校验脚本（`scripts/validate_datasets.py`）

**逐行过全量数据，不是只看 row 0。** 检查项：

- 每行能否通过适配器
- `dataset_sample_id` 缺失/重复
- gold 条数分布；「声明了 EVIDENCE_RECALL 却抽不出 gold」的行
- gold doc_id 是否都能在 corpus 中找到（预期 100%，见 0.2）
- 重复 question 文本计数（预期见 0.3）
- corpus `(title, text)` 唯一性

**验收标准**：四组全部通过，gold 解析率 100%，无重复 ID。

---

## 5. 轮次 2：各抽 100 篇子集

**目标**：每数据集抽出可独立评测的子集，放同一 `run_id` 目录下。

### 5.1 抽样顺序必须是「先 QA 后 corpus」

**不能随机抽 100 篇 corpus** —— gold 文档可能不在子集里，Recall 天然为 0。

正确顺序：

1. 固定随机种子，从 `samples.jsonl` 抽 **100 条 QA**
2. 取这 100 条的 gold doc_id **全集**（去重）作为 corpus 必选集
3. 若必选集不足目标 corpus 规模，从剩余 corpus 随机补负样本

按 0.4 的 gold 分布估算 100 条 query 的 gold 文档数：
hotpotqa ≈ 250、2wiki ≈ 280、musique ≈ 270。
**建议 corpus 规模取 gold 全集 + 等量负样本**，即每组约 500–560 篇。

> 成本提示：编译是每篇 2 次 LLM 调用。四组合计约 2000–2200 篇 ≈ 4000–4400 次调用。
> 若预算紧，先只做 hotpotqa 一组（约 500 篇 ≈ 1000 次调用）。

**musique 抽样时按跳数分层**（2hop/3hop/4hop 按 0.4 的比例），
否则 100 条随机抽样可能几乎全是 2hop，测不出多跳深度的影响。

narrativeqa：10 篇文档 4111 chunk，抽 100 条 QA 会牵出大量 chunk。
建议**只取 2 篇文档的全部 chunk**（约 300–900 篇）+ 对应的 QA，
它的价值在实体跨文档合并与 EM/F1，不在检索指标。

### 5.2 Markdown 生成

每篇 corpus 写成 `corpus/{doc_id}.md`：

```markdown
# {title}

{text}
```

**文件名用 `doc_id`，不用 title。** 理由：Akasha 优先取首个 heading 当 title
并从正文移除，所以 heading 负责 title、文件名负责身份，两者独立互不干扰。
这样即使 title 重复（musique 647 例）也不影响身份追踪。

### 5.3 产出

`data/subsets/{run_id}/{dataset}/`：`samples.jsonl`、`corpus/*.md`、
`manifest.json`（随机种子、抽样策略、QA 数、corpus 数、gold 覆盖率、
每篇 md 的 sha256）

**验收标准**：每条 sample 的所有 gold doc_id 都在该子集 corpus 内（覆盖率 100%）。

---

## 6. 轮次 3：写入库脚本

**目标**：把子集 corpus 灌进 Akasha 并完成编译，产出 `doc_id -> page_id` 映射。

### 6.1 环境准备

1. 起 docker（db + redis + akasha）
2. 首次用 `POST /api/auth/setup` 建 workspace 与首个用户；后续 `POST /api/auth/login` 拿 JWT
3. **评测用户必须是 OWNER** —— 否则第三道授权闸门会静默丢弃 chunk，
   你会误判为召回质量差
4. 每个数据集**建独立 Space**（`POST /api/spaces/create`），互不干扰。
   四个 Space 各自评测，避免跨数据集实体合并污染结果
5. 通过 `PUT /api/llm-wiki/admin/model-configs/:feature` 固定
   `compiler` / `embedding` / `answer` / `image` 四项配置，
   **把配置快照写进 manifest**。特别注意 embedding：
   换模型后旧 chunk 的 `embedding_profile` 对不上就永远召回不到

### 6.2 导入

逐个 `POST /api/pages/import`（multipart：`file` + `spaceId`），
**串行，并发 1**。记录返回的 `page.id`，写 `page_map.jsonl`：

```json
{"dataset": "hotpotqa", "doc_id": "42", "page_id": "uuid", "title": "...", "md_sha256": "..."}
```

**可恢复**：启动时读已有 `page_map.jsonl`，跳过已导入的 doc_id。

### 6.3 触发编译并等待

导入完成后 `POST /api/llm-wiki/admin/compile-spaces`（body `{spaceIds:[...]}`）
绕过 1 小时静默期立即建 Run。

轮询 `POST /api/llm-wiki/admin/diagnostics/summary` 直到 Run 全部终态。
注意区分 `succeeded` / `partial` / `failed`，**partial 也要记录**。

### 6.4 入库完整性校验（不可跳过）

`POST /api/llm-wiki/admin/diagnostics/quality` 拿质量报告，要求：
- `missing_chunk_page_count == 0`
- `missing_embedding_page_count == 0`
- `stale_*` 计数为 0

任何一项非 0 就**不要进入下一轮** —— 半成品库跑出的指标没有意义。
若有失败页，用 `POST /api/llm-wiki/admin/retry-pages` 重试。

### 6.5 产出

`data/ingest/{run_id}/`：`page_map.jsonl`、`manifest.json`
（workspace/space id、模型配置快照、Run 结果分布、质量报告、耗时、
Akasha 版本或 git commit）

**验收标准**：`page_map` 条数 == 子集 corpus 条数；质量报告全部 0。

---

## 7. 轮次 4：生成脚本（并发 1）

**目标**：逐条跑 query，把原始响应完整落盘。**这一轮不算任何指标。**

### 7.1 请求

`POST /api/llm-wiki/query`，串行，**并发 1，一次一个请求**：

```json
{"query": "<question>", "spaceIds": ["<该数据集的 space>"], "type": "user"}
```

不传 `chatContext`（本评测是单轮）。
不传 `scoreThreshold`（用默认 0.45；后续做敏感性分析时才扫这个参数）。

### 7.2 落盘：存全文，不要只存你现在想要的字段

`data/responses/{run_id}/{dataset}.jsonl`，每行：

```json
{
  "sample_id": "hotpotqa:5abe...",
  "question": "...",
  "requested_at": "2026-09-08T…",
  "latency_ms": 1234,
  "http_status": 200,
  "response": { …完整响应体原样… }
}
```

**存完整响应体**。第五轮要用的字段有：

| 字段 | 用途 |
| --- | --- |
| `answer` | EM / F1 |
| `answerMode` | `knowledge`/`no_match`/`general` 分布 |
| `retrievedSources[].sourcePageId` | **Recall@k / nDCG@k / MRR**（裁剪前） |
| `citations[].sourcePageId` | 归因 precision（裁剪后） |
| `citationEvidence[].excerpts[]` | span 级证据校验 |
| `snippets[].retrievalReasons[]` | **多跳归因：`graph-neighbor` 占比** |
| `snippets[].sourceWindows[]` | 证据窗口 |
| `warnings` / `budget` / `completenessNotice` | 上下文截断损失 |

`retrievalDiagnostics` **不在响应里**（见 0.5），第五轮从数据库取。

### 7.3 稳健性

- **可恢复**：启动时读已有 jsonl，跳过已完成的 `sample_id`
- 失败（非 2xx、超时）**照样写一行**，记 `http_status` 与错误体，
  不要静默跳过 —— 失败率本身是指标
- 每条之间留固定间隔，避免打满 LLM 配额
- 记录本轮开始时的模型配置快照（再拉一次 `GET /admin/model-configs`），
  与轮次 3 的快照比对，不一致则报错终止

### 7.4 产出

`data/responses/{run_id}/{dataset}.jsonl` + `manifest.json`
（模型配置、请求总数、失败数、总耗时、p50/p95 延迟）

**验收标准**：每个 `sample_id` 恰好一行；失败数已知且记录在案。

---

## 8. 轮次 5：评测指标脚本

**目标**：从落盘响应算指标，输出报告。**纯离线，不再碰 Akasha。**

### 8.1 检索指标（`metrics/retrieval.py`）

把 `retrievedSources[].sourcePageId` 经 `page_map` 反查回 `doc_id`，
与 `gold_doc_ids` 比：

- **Recall@k**（k = 2, 5, 10, 20）—— 主指标
- **nDCG@k** —— gold 全部同权（二元相关性）
- **MRR** —— 首个 gold 的倒数排名
- **Hit@k** —— 至少命中一个 gold

用 `retrievedSources` 而非 `citations`：前者是裁剪前的召回全集，
后者已被「被引 ∩ 有证据」交集裁剪过，用它算 Recall 会低估检索能力。

narrativeqa 无 gold —— **请求该指标时抛异常**，不返回 0.0。

### 8.2 多跳专项（`metrics/multihop.py`）

这是本次评测的核心，通用基准测不出：

- **`graph-neighbor` 贡献率**：`snippets[].retrievalReasons` 含 `graph-neighbor`
  的条目占比，以及**其中有多少是 gold**。这直接量化图扩展的净价值
- **gold 覆盖完整度**：多跳题需要全部 gold 才能答对，
  单独统计「gold 全命中」的比例（比 Recall 平均值更贴近多跳实际需求）
- **按跳数分层**：musique 用 `id` 前缀（2hop/3hop1/3hop2/4hop1/4hop2/4hop3），
  hotpotqa 用 `level`/`type`，2wiki 用 `type`。报告随跳数增加的衰减曲线
- **信号来源分布**：`semantic` / `lexical` / `exact-title` / `graph-neighbor`
  各自贡献的 gold 命中数

### 8.3 答案质量（`metrics/qa.py`）

**EM / F1 用标准口径，不要自创** —— 否则没法跟论文比：
小写、去标点、去冠词 `a/an/the`、合并空白；多参考答案取 max
（HippoRAG 2 / MRQA 口径）。

- musique：`answer_aliases` 已在轮次 1 并入 `answers`
- narrativeqa：2 条人工参考，取 max

同时报告 `answerMode` 分布 —— `no_match` 率与 `general` 兜底率
是「检索没喂够料」的直接信号。

### 8.4 引用归因（`metrics/attribution.py`）

- **citation precision**：`citations` 中命中 gold 的比例
- **citation recall**：gold 被 `citations` 覆盖的比例
- **裁剪损失**：`len(retrievedSources) - len(citations)`，
  以及被裁掉的里面有多少其实是 gold（裁剪过严的证据）
- **证据可验证率**：`citationEvidence[].excerpts` 非空的引用占比

### 8.5 分层归因（需 join 审计表）

按 0.3 的实测结论，重复 question 几乎不存在，可以安全 join：

```sql
SELECT query_hash, retrieval_mode, metadata
FROM knowledge_query_audit
WHERE workspace_id = $1 AND created_at BETWEEN $2 AND $3
```

Python 侧算 `sha256(question)` 匹配。**musique 那 1 例重复 question 单独排除。**

拿到 `metadata` 后做三段归因：

| 分段 | 判据 | 说明 |
| --- | --- | --- |
| 召回上限 | `candidateChunkCount` vs gold 命中 | gold 是否**进过**候选集 |
| 排序损失 | `rankedCandidateCount` vs `candidateChunkCount` | 进了候选但被 RRF/阈值刷掉 |
| 授权损失 | `filteredChunkCount` | 排序通过但被第三道闸门丢弃 |

没有这个分层，你只会得到「Recall@10 = 0.6」却不知道该调什么。

**另外必须按 `retrievalMode` 切分报告** ——
`high_completeness` 与 `high_completeness_fallback` 是两种召回口径，
混在一起平均会掩盖问题。`accessPolicyFallbackUsed` 占比本身就是要报告的指标。

### 8.6 产出

`data/reports/{run_id}/`：
- `metrics.json` —— 全部指标，机器可读
- `per_sample.jsonl` —— 每条 query 的逐项结果，便于定位坏样本
- `report.md` —— 人读摘要，含 0.6 那条架构说明与本轮模型配置

---

## 9. 轮次 6（本计划外，建议后续做）

**绕过编译的原文基线对照。**

同一批子集语料，另建一套只做原文分块的检索基线（不走 LLM 编译），
跑同一批 query。差值就是「LLM 预处理对多跳检索的净贡献」。

理由见 0.6：Akasha 的召回跑在生成文本上，与公开 baseline 的 qrels 存在
架构性错配。没有这个对照，五轮产出的绝对数字**无法回答
「这套设计是否值得」** —— 而这是这个项目最该知道的答案。

实现上不需要动 Akasha：直接读 `data/normalized/{dataset}/corpus.jsonl`，
本地建一个同嵌入模型的向量索引即可。

---

## 10. 遗留事项

来自 HANDOFF.md，需要你决定：

1. **SAG 那边复制了 8 个数据集文件到 `SAG/dataset/`**（约 136MB，与本仓重复）。
   `SAG/.gitignore:138` 的 `/dataset/*` 已覆盖，不会误提交。
   留着可作第六轮的对照 baseline；不需要就删，本仓的 `dataset/` 是原始副本。
   **注意 narrativeqa 内联了 Project Gutenberg 书籍与 IMSDb 剧本正文，
   按 `SAG/dataset/README.md` 不可再分发** —— 本仓 `.gitignore` 需确认已排除 `dataset/`
2. **`SAG/scripts/_validate_datasets.py`** 未跟踪且未被 gitignore，
   是 SAG 那边 `git status` 唯一未跟踪项。思路已吸收进本文 4.5，可删
3. **`HANDOFF.md`**（本仓）已被本文完全吸收，确认后可删。
   它未被 git 跟踪，删除不可恢复
4. **本仓零提交**（`master` 无 commit）。建议在轮次 1 开始前先做首次提交，
   确保后续工作可回溯

---

## 11. 各轮验收标准汇总

| 轮次 | 验收标准 |
| --- | --- |
| 1 归一化 | 四组全量通过适配器；gold 解析率 100%；无重复 ID；manifest 含 sha256 |
| 2 抽样 | 每条 sample 的 gold 全在子集 corpus 内（覆盖率 100%）；musique 按跳数分层 |
| 3 入库 | `page_map` 条数 == corpus 条数；质量报告 missing/stale 全为 0 |
| 4 生成 | 每个 sample_id 恰好一行；失败数已记录；模型配置与轮次 3 一致 |
| 5 指标 | narrativeqa 请求检索指标时抛异常而非返回 0；报告按 `retrievalMode` 切分 |
