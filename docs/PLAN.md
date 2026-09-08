# Akasha-Benchmark 实施计划

评测 Akasha 的多跳检索、引用归因与端到端答案质量。全部代码在本仓库实现，
**不修改 Akasha 主仓库**。入库和查询通过 HTTP；可选审计归因直接只读查询数据库。

## 当前进度（2026-09-08）

按当前工作区代码核对，轮次 1–5、HTTP 客户端、配置模块及可选审计归因均已实现。
本次运行 `uv run pytest tests/ -q`：**69 passed**。这些是本地测试，
轮次 3/4 使用 `httpx.MockTransport`，不代表已连通真实 Akasha。
当前仓库默认路径下没有 `dataset/`、`data/`，因此尚无本工作区的全量校验、
在线编译、响应和报告产物可供验收；自定义目录中的运行情况本次未核实。

| 阶段 | 实现状态 | 当前验收状态 |
| --- | --- | --- |
| 数据下载 | 脚本已迁至 `scripts/download_dataset.py` | 迁移后的输出路径待修复，见 §10 |
| 1 归一化 | 四组适配器、严格模型、全量校验脚本已实现 | 待下载数据后做全量验收 |
| 2 子集 | gold 覆盖、负样本、MuSiQue 分层、NarrativeQA 整篇抽样已实现 | 待生成固定 run_id 子集 |
| 3 入库 | OWNER、独立 Space、编译、质量检查、续跑已实现 | 多数据集映射问题待修复；在线验收待做 |
| 4 查询 | 串行请求、配置比对、响应落盘与续跑已实现 | 前置闸门和失败处理待补强；在线验收待做 |
| 5 评测 | 检索、答案、引用、多跳及报告已实现 | 待真实响应验收 |
| 可选审计 | 查询哈希关联、按 retrievalMode 汇总已实现 | 续跑时间窗待修复；数据库验收待做 |
| 6 原文基线 | 尚未实现 | 后续对照实验，见 §9 |

本文保留实施约束与验收目标；“已实现”与“待补强”分别标明当前行为和剩余工作。
SAG 仅为历史设计参考，不是运行依赖；旧 HANDOFF 的迁移和清理不再作为本项目任务。

---

## 0. 数据快照与接口依据

本节数据统计来自先前记录的固定快照，本次未重新下载或全量复测。
Akasha 服务端行为为既有接口约定及客户端代码所依赖的假设，真实部署仍需核验。

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

### 0.2 gold → corpus 对齐

历史快照检查记录：

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

**身份规则**：归一化时用 title 或 `(title, text)` 对齐；后续评测按 ID 匹配。各数据集的 corpus 行身份如下——

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

审计连接键为 `sha256:<hex>`（含前缀）。重复 question 是连接歧义，
不是密码学哈希碰撞。`audit_join.py` 排除所选子集中所有重复文本对应的样本，
并在 workspace 与时间窗内，对同一哈希取最后一条审计记录；续跑限制见 §10。

### 0.4 其他实测细节

- **历史原始标注数量分布（不可直接作为去重后的指标分母）**：hotpotqa `{2:650, 3:253, 4:81, 5:13, 6:1, 7:2}`；
  2wiki `{2:765, 4:234, 5:1}`；musique `{2:518, 3:316, 4:166}`
- **gold 内部重复**：hotpotqa 有 350 行、musique 有 47 行的 gold title 列表内部有重复
  （hotpotqa 的 `supporting_facts` 是 `(title, 句子下标)` 对；MuSiQue 按支撑段落解析）。
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
- **title 规则**（`import.service.ts:106`，历史服务端定位）：
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
- 在线运行需要 Akasha、数据库及 Redis（BullMQ 依赖）；具体镜像、数据库版本和
  compose 文件以实际部署为准，本工作区未核验外部主仓部署状态。
- 依赖尽量少。**不要引入 spacy** —— 本评测不需要，它是 SAG 里最大的体积约束
- 所有 Akasha 连接参数（base URL、账号、workspace/space id、API key）
  走配置文件或环境变量，**不硬编码**

`pyproject.toml` 已声明 `httpx`、`huggingface-hub`、`numpy`、`pydantic`，
开发依赖为 `pytest`。`psycopg[binary]` 尚未加入依赖，仅可选审计需要。
配置由根目录 `akasha.config.json` 或 `AKASHA_*` 环境变量提供，环境变量优先；
默认使用 email/password 登录，敏感配置在运行记录中脱敏。

下列命令从仓库根执行。下载前先完成 §10 的路径修复，或自行把数据准备到根目录 `dataset/`：

```bash
uv sync
uv run python scripts/download_dataset.py
uv run python scripts/download_dataset.py --check
uv run python -m akasha_benchmark.normalize
uv run python scripts/validate_datasets.py
uv run python -m akasha_benchmark.subset --run-id run001
# 配置 Akasha 后执行；确认轮次 3 质量通过才进入轮次 4
uv run python -m akasha_benchmark.ingest --run-id run001
uv run python -m akasha_benchmark.run_queries --run-id run001
uv run python -m akasha_benchmark.evaluate --run-id run001
# 可选：先完成 evaluate，并安装 psycopg、配置 database_url 和 workspace_id
uv run python -m akasha_benchmark.audit_join --run-id run001
```

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
  akasha_client.py      HTTP、cookie 登录、节流与接口封装
  audit_join.py         可选数据库审计归因

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

docs/
  PLAN.md                         本计划
  datasets.md                     数据集字段说明
tests/                            本地与 HTTP mock 测试
scripts/
  download_dataset.py             下载与文件大小校验（迁移后路径待修复）
  validate_datasets.py             轮次 1 的校验脚本
```

---

## 3. 贯穿全程的设计约束

以下约束已体现在当前实现中，验收仍需检查实际产物：

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
5. **运行记录与 sha256。** 轮次 1–4 写 manifest：归一化记录源文件及输出哈希，
   子集记录种子与 Markdown 哈希，在线轮次记录模型快照和运行统计。轮次 5
   写 metrics/per_sample/report，不另写 manifest。汇总文件原子替换；
   page_map 与响应 JSONL 逐条追加并 flush，截断尾行恢复仍待补强
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

### 4.5 校验脚本（`../scripts/validate_datasets.py`）

**逐行过全量数据，不是只看 row 0。** 检查项：

- 每行能否通过适配器
- `dataset_sample_id` 缺失/重复
- gold 条数分布；「声明了 EVIDENCE_RECALL 却抽不出 gold」的行
- gold doc_id 是否都能在 corpus 中找到（预期 100%，见 0.2）
- 重复 question 文本计数（预期见 0.3）
- corpus `(title, text)` 唯一性

**验收标准**：四组全部通过，gold 解析率 100%，无重复 ID。

---

## 5. 轮次 2：构建可复现子集

**目标**：每数据集抽出可独立评测的子集，放同一 `run_id` 目录下。

### 5.1 抽样顺序必须是「先 QA 后 corpus」

**不能随机抽 100 篇 corpus** —— gold 文档可能不在子集里，Recall 天然为 0。

正确顺序：

1. 固定随机种子，从 `samples.jsonl` 抽 **100 条 QA**
2. 取这 100 条的 gold doc_id **全集**（去重）作为 corpus 必选集
3. 按 `negatives_ratio` 从剩余 corpus 随机补负样本，默认与 gold 文档等量，受剩余语料数限制

默认 `qa_limit=100`、`negatives_ratio=1.0`。随机源包含 `run_id`、
数据集名与 seed；实际语料数以去重后的 gold 全集和子集 manifest 为准，
不按原始标注条数估算编译成本。

**MuSiQue 按 `hop_prefix` 六层分配名额**，使用最大余额法保留小层。
NarrativeQA 默认优先选择 chunk 最少的 2 篇文档，保留它们的全部 chunk，
再取对应 QA；QA 超过上限时继续抽样。它不参与检索相关指标。

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
2. 首次部署需先完成 workspace 与用户初始化；脚本用 `POST /api/auth/login`
   登录并保持 `authToken` cookie，不自动执行 setup
3. **评测用户必须是 OWNER** —— 否则第三道授权闸门会静默丢弃 chunk，
   你会误判为召回质量差
4. 每个数据集**建独立 Space**（`POST /api/spaces/create`），互不干扰。
   四个 Space 各自评测，避免跨数据集实体合并污染结果
5. 运行前在 Akasha 配好 `compiler` / `embedding` / `answer` / `image`。
   当前脚本只读取并记录模型配置，不主动修改服务端配置。特别注意 embedding：
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
- `missingChunkPageCount == 0`
- `missingEmbeddingPageCount == 0`
- `missingSourcePageCount == 0`
- `stalePageCount == 0`

实际读取 `report.summary` 的 camelCase 字段，缺字段也判失败。

任何一项非 0 就**不要进入下一轮** —— 半成品库跑出的指标没有意义。
若有失败页，需另行调用 `POST /api/llm-wiki/admin/retry-pages`；脚本不自动重试。
`--skip-compile` 是调试入口，不等于质量验收通过；超时和下一轮的自动阻断缺口见 §10。

### 6.5 产出

`data/ingest/{run_id}/`：`page_map.jsonl`、`manifest.json`
（workspace/space id、模型快照、导入统计、Run 状态、超时标记、质量报告和
`quality_passed`）。尚未记录 Akasha 版本或 commit、完整耗时，这些仍是追溯待办。

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

- **可恢复**：启动时读已有 jsonl，跳过所有已有 `sample_id`，包括失败行；
  当前没有自动重试失败样本的策略
- 非 2xx 响应会落盘；捕获的 `AkashaError` / `OSError` 记为状态 0。
  原生 `httpx.RequestError`（含超时）尚未被这层捕获，需补齐后才能保证失败均落盘
- 每条之间留固定间隔，避免打满 LLM 配额
- 记录本轮开始时的模型配置快照（再拉一次 `GET /admin/model-configs`），
  与轮次 3 的快照比对，默认不一致则终止；`--allow-config-drift` 可覆盖，
  manifest 会记录不一致，正式可比实验不应使用该覆盖

### 7.4 产出

`data/responses/{run_id}/{dataset}.jsonl` + `manifest.json`
（模型配置、当前调用的开始/生成时间、请求数、失败数、成功请求的 p50/p95/max 延迟）。
续跑后 manifest 只统计本次新请求，并覆盖原时间窗；完整运行统计尚待合并。

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

NarrativeQA 无 gold：`evaluate.py` 自动省略检索、归因和多跳指标，并写明原因；
显式能力检查 `require_evidence_capability` 则抛 `CapabilityError`，不伪造 0 分。

检索汇总同时输出全样本与 `answerMode == "knowledge"` 切片。`no_match` /
`general` 的空 `retrievedSources` 不能证明底层未召回。未知 page_id 保留排名位，
不计为 gold，并单列 `unmapped_page_ids`。

### 8.2 多跳专项（`metrics/multihop.py`）

这是本次评测的核心，通用基准测不出：

- **`graph-neighbor` 贡献率**：`snippets[].retrievalReasons` 含 `graph-neighbor`
  的条目占比，以及**其中有多少是 gold**。这直接量化图扩展的净价值
- **gold 覆盖完整度**：多跳题需要全部 gold 才能答对，
  单独统计「gold 全命中」的比例（比 Recall 平均值更贴近多跳实际需求）
- **当前分层报告**：MuSiQue 按 `hop_count`（2/3/4）汇总；HotpotQA 与
  2Wiki 按 `type` 汇总。抽样用六层 `hop_prefix`，报告不是六层，也未按 `level` 分层
- **信号来源分布**：`semantic` / `lexical` / `exact-title` / `graph-neighbor`
  各自贡献的 gold 命中数

### 8.3 答案质量（`metrics/qa.py`）

**EM / F1 使用常见答案归一化口径**（口径相同不代表实验可直接与论文比较）：
小写、去标点、去冠词 `a/an/the`、合并空白；多参考答案取 max
（HippoRAG 2 / MRQA 口径）。

- musique：`answer_aliases` 已在轮次 1 并入 `answers`
- narrativeqa：2 条人工参考，取 max

同时报告 `answerMode` 分布 —— `no_match` 率与 `general` 兜底率
用于观察回答模式；不能仅据该分布断定底层检索失败。

### 8.4 引用归因（`metrics/attribution.py`）

- **citation precision**：`citations` 中命中 gold 的比例
- **citation recall**：gold 被 `citations` 覆盖的比例
- **裁剪损失**：先映射、去重为 doc_id，计算 `retrieved_set - cited_set` 的大小，
  同时记录差集中的 gold 数；不是原始数组长度相减
- **证据可验证率**：`citationEvidence` 条目中 `excerpts` 非空的占比，
  只检查证据存在性，尚未校验 span 内容是否真实支持答案

当前引用归因会忽略无法映射的 page_id；未知引用对 precision 分母的影响待明确（§10）。

### 8.5 分层归因（需 join 审计表）

该模块是离线评测之外的可选步骤，先读取 `per_sample.jsonl`，再以只读 SQL 取审计：

```sql
SELECT query_hash, retrieval_mode, metadata
FROM knowledge_query_audit
WHERE workspace_id = $1 AND created_at BETWEEN $2 AND $3
```

Python 侧用 `"sha256:" + hashlib.sha256(question.encode("utf-8")).hexdigest()`
匹配。所选子集中的重复文本全部排除；同一哈希按 `created_at` 取窗口内最后一条。

拿到 `metadata` 后做三段归因：

| 分段 | 判据 | 说明 |
| --- | --- | --- |
| 候选为空 | `candidateChunkCount == 0` | 只证明候选为空，不能推断 gold 是否进过非空候选集 |
| 排序损失 | `max(candidateChunkCount - rankedCandidateCount, 0)` | 候选数量减少，不是逐 gold 的损失归因 |
| 授权损失 | `filteredChunkCount` | 排序通过但被第三道闸门丢弃 |

这些计数帮助定位损失阶段；当前 `gold_hit` 来自离线 `hit@10`，
不是候选集 gold 命中。报告不能把计数差解释为已证明的逐文档因果归因。

**另外必须按 `retrievalMode` 切分报告** ——
`high_completeness` 与 `high_completeness_fallback` 是两种召回口径，
混在一起平均会掩盖问题。`accessPolicyFallbackUsed` 占比本身就是要报告的指标。

### 8.6 产出

`data/reports/{run_id}/`：
- `metrics.json` —— 全部指标，机器可读
- `per_sample.jsonl` —— 每条 query 的逐项结果，便于定位坏样本
- `report.md` —— 人读摘要，含 0.6 那条架构说明与本轮模型配置
- `audit_join.json` —— 可选命令单独写入，含 `by_retrieval_mode` 与关联覆盖统计；
  当前不合并进 `report.md`

评测对重复 sample_id、未知 sample_id、question 不一致报错；缺响应文件的
数据集会跳过，部分缺失样本列入 `missing_responses`，不会自动补成失败行。
已有 HTTP 失败行参与指标统计。正式验收必须额外检查覆盖率，不能只看退出码。

---

## 9. 轮次 6（本计划外，建议后续做）

**绕过编译的原文基线对照。**

同一批子集语料，另建一套只做原文分块的检索基线（不走 LLM 编译），
跑同一批 query。差值就是「LLM 预处理对多跳检索的净贡献」。

理由见 0.6：Akasha 的召回跑在生成文本上，与公开 baseline 的 qrels 存在
架构性错配。没有这个对照，五轮产出的绝对数字**无法回答
「这套设计是否值得」** —— 而这是这个项目最该知道的答案。

实现无需修改 Akasha：读取同一 run_id 的子集语料建立原文索引。
需锁定语料、query、嵌入模型、分块、检索预算与生成配置；直接索引全量
normalized corpus 会改变候选范围，不适合作为同子集对照。向量基线与 Akasha
混合检索还存在机制差异，结果应说明这些差异，不能全部归因于 LLM 编译。

---

## 10. 剩余工作与执行顺序

### 10.1 正式在线运行前修复

- [ ] **下载路径迁移**：`scripts/download_dataset.py` 的 `DEST` 仍按脚本目录取
  `dataset/`，实际会写到 `scripts/dataset/`；resolver 读取根目录 `dataset/`。
  统一路径，并同步 README、脚本 docstring、resolver 的旧下载命令。
  `.gitignore` 已排除根目录 `/dataset/`，未覆盖错误的新输出位置。
- [ ] **跨数据集 page_map 身份**：`ingest._load_page_map` 仅以 `doc_id` 建字典，
  多组数字 ID 会互相覆盖，影响续跑过滤、条数统计。改为 `(dataset, doc_id)`
  并增加两组同 ID 的续跑测试，校验每组映射集合等于对应子集。
- [ ] **强制前置质量闸门**：轮次 4 目前只要求轮次 3 manifest 存在，未检查
  `quality_passed`、导入完整性或 `runs.timed_out`。轮次 3 的超时也未独立触发
  失败退出，`--skip-compile` 可返回成功。补上正式运行的自动阻断与测试。
- [ ] **网络异常与文件恢复**：明确捕获 `httpx.RequestError` 并将超时、断连落盘；
  为响应及 page_map 追加 JSONL 的截断尾行制定恢复策略。已有失败行当前会被
  续跑跳过，需要明确重试方式，避免简单追加造成重复 sample_id。
- [ ] **续跑统计与审计时间窗**：响应文件累积保留，manifest 却被本次请求统计
  覆盖，审计会漏掉早期请求。保存完整会话记录或从响应时间构造覆盖窗口，
  并加强同 query 多次运行的匹配；测试续跑后的累计条数与审计覆盖率。

### 10.2 真实数据与在线验收

- [ ] 下载固定快照，执行四组 normalize 与 validate，记录新的哈希及去重 gold 分布。
- [ ] 固定 seed/run_id 构建子集，确认每组 gold 覆盖及实际编译规模。
- [ ] 先用单数据集小规模走通在线入库、查询、离线报告，再执行完整四组实验。
- [ ] 记录 Akasha 版本/commit、模型配置与完整耗时；核对 HTTP 返回字段和质量诊断。
- [ ] 明确未映射引用的计分分母；核对答案失败计分、缺失响应、证据存在性与
  真正 span 校验的差异。NarrativeQA 在审计中没有 gold，其 gold_hit 不应混入
  有 gold 数据集的准确率结论。
- [ ] 可选审计配置 database_url/workspace_id 和 psycopg，核对 unmatched、
  excluded、retrievalMode 切片；再决定是否把审计摘要合并进人读报告。

### 10.3 后续工作

- [ ] 实现 §9 的原文基线与固定实验配置对照。

仓库已有提交（本次检查 HEAD 为 `6b74301`），不再保留“零提交、先首次提交”的
旧待办。外部 SAG 文件清理不在本项目范围。本文仅记录当前代码缺口，未代替代码修复。

---

## 11. 各轮验收标准汇总

下表是正式实验的验收目标；当前进度见文首，已知自动检查缺口见 §10。

| 轮次 | 验收标准 |
| --- | --- |
| 1 归一化 | 四组全量通过；有 gold 的三组解析率 100%；无重复 ID；manifest 哈希匹配 |
| 2 抽样 | gold 全在子集 corpus；MuSiQue 六层抽样；NarrativeQA 整篇保留；种子与哈希可追溯 |
| 3 入库 | 每组 page_map 身份集合等于子集；无未完成编译；四项 camelCase 质量计数均为 0 |
| 4 查询 | 全量 sample_id 恰好一行；失败均落盘；配置一致；续跑统计覆盖完整实验 |
| 5 指标 | 覆盖率与失败数明确；全样本/knowledge 检索切片；NarrativeQA 省略无定义指标；报告注明架构边界 |
| 可选审计 | 完整运行窗口；重复文本排除；未匹配数量明确；按 retrievalMode 汇总，计数归因不冒充候选 gold 验证 |

当前测试基线：`uv run pytest tests/ -q`，2026-09-08 本地运行 **69 passed**。
后续为 §10 的缺口补充有区分力的测试，并以真实产物完成各轮验收。
