# Akasha-Benchmark 实施计划

评测 Akasha 的多跳检索、引用归因与端到端答案质量。全部代码在本仓库实现，
**不修改 Akasha 主仓库**。入库和查询通过 HTTP；可选审计归因直接只读查询数据库。

本文只写实施约束、待办与验收目标。事实性说明各有归属，不在此重复：

| 内容 | 文档 |
| --- | --- |
| 两个接口的请求/响应形状、字段陷阱 | [akasha_api.md](akasha_api.md) |
| 每个指标怎么算、为什么这么算 | [metrics.md](metrics.md) |
| 原始数据字段与各组差异 | [datasets.md](datasets.md) |
| 归一化产物的字段与口径 | [normalized_datasets.md](normalized_datasets.md) |

## 当前进度（2026-09-09）

各阶段、HTTP 客户端、配置模块及可选审计归因均已实现。
`uv run pytest -q`：**86 passed, 23 skipped**。23 个 skip 是单样本在线冒烟
（`make smoke`），已在真实 Akasha 上跑过一次 **23 项全过**（2026-09-09，
hotpotqa 单样本，5 分 18 秒）。

那一趟抓出两个必须先修的问题，都已解决：

1. **全局响应信封**。Akasha 给所有路由挂了 `TransformHttpResponseInterceptor`，
   响应是 `{data, success, status}`，而客户端此前直接读顶层字段。后果不是报错而是
   静默读空：OWNER 闸门永远拒绝执行、每篇导入记成失败、**质量闸门四项计数全
   `None` 导致 `all(value == 0)` 假通过**。修在 `request()` 一处
   （`unwrap_envelope`），细节见 [akasha_api.md](akasha_api.md) §0。
   替身此前返回裸响应，所以 73 个测试全绿也没能拦住 —— 现在替身也套信封了。
2. **provider 的 `baseUrl` 少了 `/v1`**，编译 255ms 就 `provider_error`，
   `compiledPageCount` 恒为 0。属于部署配置问题，见 akasha_api.md §3。

离线链条已有产物：`dataset/` 四组原始数据、`data/normalized/` 四组归一化结果、
`data/subsets/run001/`（hotpotqa 100 问/400 篇、2wiki 100/490、musique 100/476、
narrativeqa 58/356，共 358 问 1722 篇，四组 `gold_coverage` 均为 1.0）。

在线阶段目前只有冒烟产物 `data/smoke/`：`data/ingest/`、`data/responses/`、
`data/reports/` 仍不存在，所以入库、查询、报告三段的**批量**验收还没做。
`akasha.config.json` 已配好并连通。

| 阶段 | 实现状态 | 当前验收状态 |
| --- | --- | --- |
| 数据下载 | `scripts/download_datasets.py` → 根目录 `dataset/` | 已下载，四组文件校验通过 |
| 归一化 | 四组适配器、严格模型、全量校验脚本 | 已产出，manifest 哈希在案 |
| 子集 | gold 覆盖、负样本、MuSiQue 分层、NarrativeQA 整篇 | `run001` 已产出，覆盖率 1.0 |
| 入库 | OWNER 闸门、独立 Space、编译、质量检查、续跑 | 接口约定已在线核过（冒烟）；批量与续跑的在线验收待做 |
| 查询 | 串行请求、配置比对、响应落盘与续跑 | 单条已在线跑通（冒烟）；批量在线验收待做 |
| 评测 | 检索、答案、引用、多跳及报告 | 待真实响应验收 |
| 可选审计 | 查询哈希关联、按 retrievalMode 汇总 | 续跑时间窗待修复；数据库验收待做 |
| 原文基线 | 尚未实现 | 后续对照实验，见 §9 |

---

## 0. 两条贯穿全程的事实

### 0.1 身份规则

各数据集的 corpus 行身份，以及为什么不能都用 title：

| 数据集 | doc_id | 重复 title |
| --- | --- | --- |
| hotpotqa | 原生 `idx` | 0（9811 行 9811 个 title） |
| 2wikimultihopqa | corpus 数组行号（原始数据**没有** `idx`） | 0（6119/6119） |
| musique | corpus 数组行号（同样没有 `idx`） | **2465 行落在重复 title 组里** |
| narrativeqa | 原生 `idx`（`{document_id}_{chunk_seq}`） | 4111 行仅 10 个 title（同篇切块） |

musique 的对齐键因此是 `(title, text)` 而非 title —— 实测
`paragraphs[].paragraph_text` 与 corpus `text` 逐字节相等，所以这个键是可靠的。
`(title, text)` 仍有重复时直接报错，不静默取一个。

gold title 在三组有 gold 的数据集里都 100% 命中 corpus，所以对齐只在
title 是否唯一这一层出问题，不在能否命中这一层。

### 0.2 去重后的 gold 篇数分布

`gold_count_distribution` 记的是**去重后**的篇数，与原始标注条数不同 ——
hotpotqa 的 `supporting_facts` 是 `(title, 句子下标)` 对，同一篇会出现多次：

| 数据集 | 去重后分布 |
| --- | --- |
| hotpotqa | `{2: 1000}`（原始标注最多到 7 条，去重后全是 2 篇） |
| 2wikimultihopqa | `{2: 765, 4: 235}` |
| musique | `{2: 518, 3: 316, 4: 166}` |
| narrativeqa | `{0: 293}`（无 gold） |

归一化时 gold **必须去重成集合**，否则 Recall 分母是错的。

### 0.3 一个必须写进报告的架构事实

**Akasha 的稠密/词法召回跑在 LLM 生成的文本上，不是原文。**

`knowledge_chunks` 索引的是编译器产出的 `artifact.markdown`；原文存在
`knowledge_source_chunks`，该表**不参与召回**，只在引用解析时提供证据窗口。

后果有两个方向：Recall@k 会系统性偏低且非调参可解；但多跳可能偏高，因为实体被
物化成独立 artifact 并跨文档合并，两个 hop 可能被编译器直接连成一条 graph edge。

**因此与已发表 baseline 直接比数字是无效的。** 唯一有意义的对照是 §9 的原文基线，
本计划各阶段只产出 Akasha 自身的横向可比数字。这条要写进 `report.md`。

---

## 1. 环境

- Python 3.12（`.python-version` 已 pin，**勿用 3.14**）
- 在线运行需要 Akasha、数据库及 Redis（BullMQ 依赖）。镜像与 compose 以实际部署为准
- 依赖尽量少。`pyproject.toml` 声明 `httpx`、`huggingface-hub`、`numpy`、`pydantic`，
  开发依赖 `pytest`。`psycopg[binary]` 未加入依赖，仅可选审计需要
- 所有 Akasha 连接参数走根目录 `akasha.config.json` 或 `AKASHA_*` 环境变量，
  环境变量优先，**不硬编码**。密钥在运行记录中脱敏

命令从仓库根执行。`Makefile` 把这套流程包了一层，`make help` 看目标：

```bash
uv sync
uv run python scripts/download_datasets.py            # ~137MB
uv run python scripts/download_datasets.py --check
uv run python -m akasha_benchmark.normalize
uv run python scripts/validate_datasets.py
uv run python -m akasha_benchmark.subset --run-id run001
# 以下需要 Akasha 在线；确认入库质量闸门通过才开始跑查询
uv run python -m akasha_benchmark.ingest --run-id run001
uv run python -m akasha_benchmark.run_queries --run-id run001
uv run python -m akasha_benchmark.evaluate --run-id run001
# 可选：需先完成 evaluate，并安装 psycopg、配置 database_url 与 workspace_id
uv run python -m akasha_benchmark.audit_join --run-id run001
```

---

## 2. 产物布局

```
data/
  normalized/{dataset}/            samples.jsonl  corpus.jsonl  manifest.json
  subsets/{run_id}/{dataset}/      samples.jsonl  corpus/{doc_id}.md
                                   corpus_hashes.json  manifest.json
  ingest/{run_id}/                 page_map.jsonl  manifest.json
  responses/{run_id}/              {dataset}.jsonl  manifest.json
  reports/{run_id}/                metrics.json  per_sample.jsonl  report.md
                                   audit_join.json（可选）
```

`page_map.jsonl` 是 `(dataset, doc_id) -> page_id` 的映射，评测靠它把响应里的
`sourcePageId` 反查回语料文档。它和响应 JSONL 都是逐条追加并 flush，中断可续跑。

---

## 3. 贯穿全程的设计约束

1. **显式适配器，拒绝 schema 猜测。** 每个数据集一个适配器类，自己声明名字、
   别名、校验规则。禁止 `if "supporting_facts" in row: ... elif "paragraphs" in row:`
   这种按字段存在性分派 —— 数据集换版多个字段就会静默走错分支
2. **规范化模型严格校验。** pydantic `ConfigDict(extra="forbid", frozen=True)`，
   多余字段报错而非忽略，构造后不可变
3. **capability 声明，不推断。** narrativeqa 不声明 `EVIDENCE_RECALL`；请求该指标时
   **显式抛异常**，不要返回空列表算出 0.0 —— 假分数会静默污染汇总
4. **身份不许推断。** 有原生 ID 用原生 ID；narrativeqa 用全量数据集行号字符串，
   且**这个口径必须收到一处常量**。重复 ID 直接报错。按 ID 匹配后**再比一次
   question 文本**，不一致报错 —— 能抓到预测文件与数据集版本不匹配
5. **运行记录与 sha256。** 各阶段写 manifest：归一化记源文件及输出哈希，子集记种子
   与每篇 md 的哈希，在线阶段记模型快照与运行统计。汇总文件原子替换
6. **corpus 不去重。** musique 有重复 title 但它们是不同段落，去重会丢 gold。
   去重前后条数都记进 manifest，口径全链路统一

---

## 4. 归一化

**目标**：四组原始数据转成统一的 `samples.jsonl` + `corpus.jsonl`，全量，不抽样。

`CanonicalSample` 的字段与各组 metadata 见
[normalized_datasets.md](normalized_datasets.md)。三个口径决定在这一层：

- `answers` 统一成元组，单答案也是长度 1，解决四组类型不一致
- **musique 的 `answer_aliases` 并入 `answers`**（去重后），答案 F1 取 max 时自动覆盖别名
- gold 去重成集合后再经身份表映射（见 §0.1、§0.2）

**2wiki 的 `evidences` 是关系三元组，不是 gold 文档，禁止当检索 ground truth。**

### 4.1 校验脚本（`../scripts/validate_datasets.py`）

**逐行过全量数据，不是只看 row 0。** 检查每行能否通过适配器、
`dataset_sample_id` 缺失或重复、gold 条数分布、「声明了 EVIDENCE_RECALL 却抽不出
gold」的行、gold doc_id 能否在 corpus 找到、重复 question 文本计数、
corpus `(title, text)` 唯一性。

**验收**：四组全部通过，gold 解析率 100%，无重复 ID。

---

## 5. 子集

**目标**：每数据集抽出可独立评测的子集，放同一 `run_id` 目录下。

### 5.1 抽样顺序必须是「先 QA 后 corpus」

**不能随机抽 100 篇 corpus** —— gold 文档可能不在子集里，Recall 天然为 0。

1. 固定随机种子，从 `samples.jsonl` 抽 100 条 QA
2. 取这 100 条的 gold doc_id **全集**（去重）作为 corpus 必选集
3. 按 `negatives_ratio` 从剩余 corpus 补负样本，默认与 gold 等量，受剩余语料数限制

随机源包含 `run_id`、数据集名与 seed。编译成本按子集 manifest 的实际语料数估算，
不按原始标注条数。

**MuSiQue 按 `hop_prefix` 六层分配名额**，最大余额法保留小层。
NarrativeQA 默认取 chunk 最少的 2 篇文档并保留全部 chunk，再取对应 QA；
它不参与检索相关指标。

### 5.2 Markdown 文件名用 doc_id，不用 title

每篇写成 `corpus/{doc_id}.md`，正文是 `# {title}\n\n{text}`。

Akasha 优先取首个 heading 当 title 并从正文移除，所以 heading 负责 title、
文件名负责身份，两者独立互不干扰。这样即使 title 重复也不影响身份追踪。

**验收**：每条 sample 的所有 gold doc_id 都在该子集 corpus 内（覆盖率 100%）。

---

## 6. 入库

**目标**：把子集 corpus 灌进 Akasha 并完成编译，产出 `doc_id -> page_id` 映射。

### 6.1 环境准备

1. 起 docker（db + redis + akasha），首次部署先完成 workspace 与用户初始化 ——
   脚本用 `POST /api/auth/login` 登录并保持 `authToken` cookie，不自动执行 setup
2. **评测用户必须是 OWNER** —— 否则第三道授权闸门会静默丢弃 chunk，
   症状看起来像召回质量差
3. 每个数据集**建独立 Space**，避免跨数据集实体合并污染结果
4. 在 Akasha 后台配好 `compiler` / `embedding` / `answer` / `image` 四项模型。
   脚本只读取并记录，不修改服务端配置。特别注意 embedding：换模型后旧 chunk 的
   `embedding_profile` 对不上就永远召回不到

### 6.2 导入

逐个 `POST /api/pages/import`，**串行，并发 1**。记录返回的 `page.id` 写
`page_map.jsonl`。启动时读已有映射跳过已导入的，可续跑。

**续跑的键是 `(dataset, doc_id)`。** doc_id 是各数据集内部的裸 ID，跨组会撞 ——
锁定子集上 hotpotqa×2wiki 撞 15 个、hotpotqa×musique 17 个、2wiki×musique 28 个。

导入前重算每篇 md 的 sha256 并与子集 manifest 比对，不符直接抛错：
语料在两次运行之间被改过，这份 page_map 就不能用了。

### 6.3 触发编译并等待

导入完成后 `POST /api/llm-wiki/admin/compile-spaces` 绕过 1 小时静默期立即建 Run
（静默期是 `KNOWLEDGE_PAGE_COMPILE_QUIET_PERIOD_MS = 3600000`，且每次编辑会
**重置**到期时间）。

轮询 `POST /api/llm-wiki/admin/diagnostics/summary` 直到 Run 全部终态。
区分 `succeeded` / `partial` / `failed`，**partial 也要记录** —— 它意味着一部分页
编译成功一部分没有，指标会因此偏低但不会报错。

### 6.4 入库完整性闸门（不可跳过）

`POST /api/llm-wiki/admin/diagnostics/quality` 拿质量报告，要求
`missingChunkPageCount`、`missingEmbeddingPageCount`、`missingSourcePageCount`、
`stalePageCount` **全部为 0**。

字段名是 camelCase，读 `report.summary`，缺字段也判失败 —— 取不到值时每一项都是
None，闸门会假通过，然后拿一个半成品库跑出一堆没意义的指标。

任何一项非 0 就**不要进入查询**。失败页需另行调
`POST /api/llm-wiki/admin/retry-pages`，脚本不自动重试。
`--skip-compile` 是调试入口，不等于质量验收通过。

**验收**：`page_map` 条数 == 子集 corpus 条数；质量报告全部 0。

---

## 7. 查询

**目标**：逐条跑 query，把原始响应完整落盘。**这一步不算任何指标。**

### 7.1 请求

`POST /api/llm-wiki/query`，**并发 1，一次一个请求**，请求体只有
`{query, spaceIds: [该数据集的 space], type: "user"}`。

不传 `chatContext`（本评测是单轮），不传 `scoreThreshold`（用服务端默认值；
做敏感性分析时才扫这个参数）。每条之间留固定间隔，避免打满 LLM 配额。

### 7.2 落盘存全文，不要只存现在想要的字段

`data/responses/{run_id}/{dataset}.jsonl` 每行一条，HTTP 响应体整个塞进
`response`。重跑一次要烧 LLM 调用，所以以后想到要看某个新字段时，
不该被迫重跑。各字段的用途见 [akasha_api.md](akasha_api.md)。

`retrievalDiagnostics` **不在响应里** —— controller 解构时排除了它，
只写进 `knowledge_query_audit.metadata`，所以三段归因必须走 §8.5 读数据库。

### 7.3 稳健性

- **可恢复**：启动时读已有 jsonl，跳过所有已有 `sample_id`，**包括失败行**；
  当前没有自动重试失败样本的策略
- 非 2xx 落盘记真实状态码；`AkashaError` / `httpx.RequestError` / `OSError`
  记状态 0。`httpx.RequestError` 不是 `OSError` 子类，漏掉它一次网络抖动就会
  让整个阶段带 traceback 崩掉且该行不落盘
- 本阶段开始时再拉一次模型配置快照与入库时比对，默认不一致则终止。
  `--allow-config-drift` 可覆盖并记进 manifest，正式可比实验不应使用

**验收**：每个 `sample_id` 恰好一行；失败数已知且记录在案。

---

## 8. 评测

**目标**：从落盘响应算指标，输出报告。**纯离线，不再碰 Akasha。**

指标定义、口径与边界全部在 [metrics.md](metrics.md)。这里只留计划层面的三条：

- **检索指标一律用 `retrievedSources`，不用 `citations`。** 后者已被
  「被引 ∩ 有证据」裁剪过，用它算 Recall 会低估检索能力
- **每个检索指标出两份**：全样本，以及 `answerMode == "knowledge"` 切片。
  `no_match` / `general` 无条件返回空 `retrievedSources`，不管检索实际找到什么，
  所以全样本那份把「生成端拒答」也算进了检索指标，两份的差值就是这个效应的规模
- **NarrativeQA 没有 gold**，检索、归因、多跳指标一律省略并写明原因，不伪造 0 分

### 8.5 分层归因（可选，需 join 审计表）

离线评测之外的可选步骤。先读 `per_sample.jsonl`，再以只读 SQL 取审计表，
按 workspace 与查询阶段自己的运行时间窗卡范围。

连接键是 `"sha256:" + sha256(question)`，**带前缀** —— 裸十六进制值一行都匹配不上。
子集中的重复 question 文本全部排除，同一哈希取窗口内最后一条。

`metadata` 拆三段：`candidateChunkCount == 0` 是候选为空、
`max(candidate - ranked, 0)` 是排序损失、`filteredChunkCount` 是授权损失。
这些是**计数**，帮助定位损失阶段，不是逐 gold 的因果归因 —— 报告不能把计数差
解释为已证明的逐文档归因。`gold_hit` 来自离线 `hit@10`，不是候选集 gold 命中。

**必须按 `retrievalMode` 切分报告**：`high_completeness` 与
`high_completeness_fallback` 是两种召回口径，混在一起平均会掩盖问题。
`accessPolicyFallbackUsed` 占比本身就是要报告的指标。

### 8.6 报告的硬性要求

评测对重复 sample_id、未知 sample_id、question 不一致报错。缺响应文件的数据集跳过，
部分缺失样本列入 `missing_responses`，不自动补成失败行。已有 HTTP 失败行参与统计。
`report.md` 必须带 §0.3 的架构说明与本次运行的模型配置。

正式验收要检查覆盖率，不能只看退出码。

---

## 9. 原文基线（本计划外，建议后续做）

同一批子集语料，另建一套只做原文分块的检索基线（不走 LLM 编译），跑同一批 query。
差值就是「LLM 预处理对多跳检索的净贡献」。

理由见 §0.3：Akasha 的召回跑在生成文本上，与公开 baseline 的 qrels 存在架构性错配。
没有这个对照，各阶段产出的绝对数字**无法回答「这套设计是否值得」** ——
而这是这个项目最该知道的答案。

实现无需修改 Akasha：读取同一 run_id 的子集语料建立原文索引。需锁定语料、query、
嵌入模型、分块、检索预算与生成配置；直接索引全量 normalized corpus 会改变候选范围，
不适合作为同子集对照。向量基线与 Akasha 混合检索还存在机制差异，
结果应说明这些差异，不能全部归因于 LLM 编译。

---

## 10. 剩余工作与执行顺序

### 10.1 正式在线运行前修复

- [x] **下载路径迁移**：`scripts/download_datasets.py` 的 `DEST` 取仓库根 `dataset/`，
  与 resolver 读取位置一致，`.gitignore` 的 `/dataset/` 覆盖到。
- [x] **跨数据集 page_map 身份**：`ingest._load_page_map` 已改为按
  `(dataset, doc_id)` 建字典。实测碰撞规模见 §6.2，共 60 个。原实现下后导入的组会
  覆盖前一组的行，续跑时前一组这些 doc 被误判成已导入，`page_map_rows` 也少算。
  `test_ingest_keeps_colliding_doc_ids_of_two_datasets_apart` 锁住这点；
  回滚修复后该测试以 `page_map_rows 4 != 6` 失败。
- [x] **查询的传输层异常**：`httpx.RequestError` 已捕获（`run_queries.py:88`）。
  它不是 `OSError` 子类，而 `query()` 用 `raise_for_status=False` 非 2xx 不抛，
  所以传输层异常原本是唯一能逃出来的东西。
  `test_run_queries_records_transport_errors_instead_of_crashing` 锁住这点。
- [ ] **强制前置质量闸门**：查询目前只要求入库 manifest 存在，未检查
  `quality_passed`、导入完整性或 `runs.timed_out`（`make query` 拦了一道，
  Python 侧没有）。入库超时也未独立触发失败退出，`--skip-compile` 可返回成功。
- [ ] **失败行恢复策略**：响应及 page_map 追加 JSONL 的截断尾行如何恢复未定；
  已有失败行会被续跑跳过，需要明确重试方式，避免简单追加造成重复 sample_id。
- [ ] **续跑统计与审计时间窗**：响应文件累积保留，manifest 却被本次请求统计覆盖，
  审计会漏掉早期请求。需保存完整会话记录或从响应时间构造覆盖窗口，
  并加强同 query 多次运行的匹配。

### 10.2 在线验收

- [x] 下载固定快照，四组 normalize 与 validate 通过，哈希与去重 gold 分布在案。
- [x] 固定 seed/run_id 构建子集，四组 gold 覆盖 1.0，实际编译规模 1722 篇。
- [x] 单样本在线冒烟已通过（`make smoke` / `tests/test_live_akasha.py`，23 项全过）：
  hotpotqa 一条样本从入库走到查询，`akasha_api.md` 的字段约定逐条核对，
  `answerMode=knowledge`、两篇 gold 排在干扰项之前、引用与证据各就各位、
  `sourcePageId` 能反查回 `doc_id`。跑完自建 Space 已删除。
- [x] 小规模走通完整的 ingest → query → report 三段（冒烟的 pipeline 用例，
  hotpotqa 3 问 9 篇，8 分 55 秒，9 项全过）：调的是三个阶段真实的 `run()`，
  入库与查询各跑两遍验续跑 —— 入库第二遍 `imported=0 / skipped=9`、查询第二遍
  `requested=0 / skipped=3`，`page_map` 9→9 行、响应 3→3 行都没长。报告侧
  `missing_responses` 与 `unmapped_page_ids` 均为空。
- [ ] 执行完整四组实验（1722 篇）。单篇编译约 40 秒是主要成本。
- [x] **EM 已移除。** 冒烟实测 `recall@10` / `full_coverage@10` / `mrr` 全 1.000、
  三条答案全部实质正确，EM 仍是 0.000 —— Akasha 返回解释性散文，参考答案是短跨度，
  整串相等不可能成立，所以 EM 恒为 0 且没有方差，零信息量。不报比报出来再附免责
  说明更诚实。F1 保留（那三条 0.054 / 0.087 / 0.143，随质量变化），但绝对值被
  解释性 token 稀释，只可同配置比较。`Capability.ANSWER_EM_F1` 一并改名
  `ANSWER_F1`，其值 `answer_f1` 会出现在 `metrics.json` 的 `capabilities` 里。
  依据见 [metrics.md](metrics.md)「为什么没有 Exact Match」。
- [ ] 决定要不要加 containment 类宽松指标（答案是否**包含**参考答案）。目前没有实现：
  它同样有偏，散文越长越容易蒙中。加不加取决于要回答什么问题。
- [ ] 记录 Akasha 版本/commit、模型配置与完整耗时；核对 HTTP 返回字段和质量诊断。
- [ ] 明确未映射引用的计分分母；核对答案失败计分、缺失响应、证据存在性与真正
  span 校验的差异。NarrativeQA 在审计中没有 gold，其 `gold_hit` 不应混入有 gold
  数据集的准确率结论。
- [ ] 可选审计配置 `database_url` / `workspace_id` 和 psycopg，核对 unmatched、
  excluded、retrievalMode 切片；再决定是否把审计摘要合并进人读报告。

### 10.3 后续

- [ ] 实现 §9 的原文基线与固定实验配置对照。

---

## 11. 各阶段验收标准汇总

| 阶段 | 验收标准 |
| --- | --- |
| 归一化 | 四组全量通过；有 gold 的三组解析率 100%；无重复 ID；manifest 哈希匹配 |
| 抽样 | gold 全在子集 corpus；MuSiQue 六层抽样；NarrativeQA 整篇保留；种子与哈希可追溯 |
| 入库 | 每组 page_map 身份集合等于子集；无未完成编译；四项 camelCase 质量计数均为 0 |
| 查询 | 全量 sample_id 恰好一行；失败均落盘；配置一致；续跑统计覆盖完整实验 |
| 指标 | 覆盖率与失败数明确；全样本/knowledge 检索切片；NarrativeQA 省略无定义指标；报告注明架构边界 |
| 可选审计 | 完整运行窗口；重复文本排除；未匹配数量明确；按 retrievalMode 汇总，计数归因不冒充候选 gold 验证 |

测试基线：`uv run pytest -q`，2026-09-09 本地 **86 passed, 23 skipped**。
23 个 skip 是 `tests/test_live_akasha.py` 的单样本在线冒烟，需要 `AKASHA_LIVE=1`
和在线的 Akasha 才跑（`make smoke`）；同日已在真实 Akasha 上 23 项全过。
后续为 §10.1 的缺口补充有区分力的测试，并以真实产物完成各阶段验收。

