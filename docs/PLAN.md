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
- [x] **EM 报出来，但按形态探针读。** 冒烟实测 `recall@10` / `full_coverage@10` /
  `mrr` 全 1.000、三条答案全部实质正确，EM 仍是 0.000 —— Akasha 返回解释性散文，
  参考答案是短跨度，整串相等不可能成立，所以 EM 预期恒为 0。它一度被移除（零信息量），
  现改为报出并在 `report.md` 表格上方直接写明「预期为 0」：熟悉 hotpotqa 的读者会
  主动去找这一列，少一列比多一列 0 更容易被误读。EM 变成非 0 的含义是生成端改了
  答案形态，不是答案变对了。F1 同时保留（那三条 0.054 / 0.087 / 0.143，随质量变化），
  绝对值被解释性 token 稀释，只可同配置比较。`Capability.ANSWER_F1`
  （值 `answer_f1`，出现在 `metrics.json` 的 `capabilities` 里）**未随之改名**，
  它标的是「这个数据集有参考答案可打分」，与报几个指标无关。
  两个指标的计算式、逐 token 分解与低分归因见 [metrics.md](metrics.md)「答案质量 QA」。
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

---

## 12. 评测平台（设计已定，尚未实现）

在本仓库与 Akasha 之上加一个 Web 评测平台：跑评测、看数据处理过程、
配指标与模型、追样本与归因。**本章只记已定的决策与理由，代码一行未写。**

Akasha 侧仍然只通过 HTTP（加一条只读 SQL），不修改 Akasha 主仓库 —— 这条不变。

### 12.1 定了的二十条

| # | 决策 | 理由要点 |
| --- | --- | --- |
| 1 | 观测与执行控制都做 | 顺序待定，见 §12.6 |
| 2 | **数据库当事实来源**，归一化产物入库 | 文件降级为可选导出；代价见 §12.2 |
| 3 | 评测端 SQLite，Akasha 链路走只读 Postgres | 方向单向：SQLite 可写权威，PG 只读外来 |
| 4 | Akasha 审计数据**抄进 SQLite 存档** | `knowledge_query_audit` 是运行时表，会随容器重建消失 |
| 5 | run 按成本边界**拆三层** | 索引层 / 查询层 / 评测层，见 §12.3 |
| 6 | 单活模型配置 + 硬闸门 | embedding 不匹配=拒绝执行；compiler 不匹配=警告 |
| 7 | 阶段任务走 **subprocess**，进度写库 | 15 小时任务不能与 Web 后端同生命周期 |
| 8 | 查询并发做成旋钮，默认 1 | 把死字段 `concurrency` 的语义补活 |
| 9 | **judge 是指标**，不是新的一层 | 与 F1/recall 同层，只是需要模型 |
| 10 | capability 从「指标名」改成「**数据依赖**」 | 见 §12.4 |
| 11 | judge provider 前端配、落库 | 但 `api_key` 不落库，见第 12 条 |
| 12 | 库只存非密字段，`api_key` 走环境变量 | db 文件不因此变成密钥文件 |
| 13 | judge 失败**该条排除**，另叠失败率闸门 | 记 0 会让限流伪装成质量差 |
| 14 | 标注三层，**只有样本层跨 run 继承** | 样本层是资产，另两层是笔记 |
| 15 | LLM 归因**只给原始材料**，不给已算指标 | 否则标签与指标的交叉验证变成空的 |
| 16 | 公开数据集与自有知识库**互相校准** | 四组公开数据是 judge 唯一的校准路径 |
| 17 | 标注与 ground truth 功能**预留** | schema 留字段，界面不做 |
| 18 | 后端 FastAPI + uvicorn | pydantic 2.13.5 已是硬依赖，直接复用 |
| 19 | 前端 Vite + React + TypeScript | 三个重交互视图；指标字段「可能不存在」需类型建模 |
| 20 | 同仓库平铺，手写 SQL 迁移 + 迁移前自动备份 | 平台与 benchmark 共享数据层，非单向依赖 |

### 12.2 库当事实来源要付的代价

现在整个项目的完整性保证**建立在文件上**。改成库权威，这四条要跟着搬，
而它们全都落在「错了不报错、只给出看着合理的假结果」的地带：

1. 各阶段 manifest 记的上游 sha256 链。
2. `ingest` 导入前重算每篇 md 的 sha256，与子集 manifest 不符直接抛错。
3. 续跑判据 —— 读已有 `page_map.jsonl` / 响应 jsonl 跳过已完成条目。
4. `evaluate` 两道闸门：`sample_id` 不在子集里报错、ID 对得上但 question 文本不一致报错。

**必须保住的性质**：`normalize` / `subset` / `evaluate` 目前完全不依赖 Akasha 在线，
改造后仍须如此（这也是选 SQLite 而非复用 Akasha 那个 Postgres 的主要理由）。

**写事务必须短、逐批提交**。否则 Web 端在 ingest 的 15 小时里读不到进度，
第 1 条决策要的观测就废了。

索引里不塞全文：narrativeqa 原始 QA 文件 94MB 是因为每行内联整篇正文
（约 210KB × 293 行，实际只有 10 篇不同文档）。存标识、指标、指针与短字段，
正文按需读；真要全文搜索再上 SQLite FTS，那是单独一件事。

### 12.3 三层的边界是 Akasha 的约束画出来的

| 层 | 由什么决定 | 成本 |
| --- | --- | --- |
| **索引层** | subset 配置（seed/qa-limit/negatives-ratio）+ compiler + embedding | 约 15 小时 / 1722 篇 |
| **查询层** | 挂某个索引层 + answer 模型 + `scoreThreshold` + 并发度 | 10–14 秒 × 每条 |
| **评测层** | 挂某个查询层 + 指标组 + k（+ judge 配置） | 确定性指标秒级；judge 有网络成本 |

分层依据是模型配置的重编译语义：compiler/embedding 改了必须重编译（索引层），
answer 改了不用（查询层）。`put_model_config` 已实现于
`akasha_client.py:349` 但**全仓库零调用**，answer 模型热切换只需接上它。

**embedding 是唯一会静默失效的那个**：换模型后旧 chunk 的 `embedding_profile`
对不上，那些 chunk 永远召回不到（`ingest.py:11`、§6.1），而评测会照常算出
一份「recall 低、拒答率高」的报告 —— 看起来像配置差，实际是索引与 embedding 错配。
所以闸门对 embedding 不匹配必须**拒绝执行**，不是警告。

真要做 embedding 对照实验，起第二个 Akasha 实例比切换全局配置更实际。

### 12.4 capability 反转成数据依赖

现有两个 capability 的**值**已经是数据依赖的意思，只是名字取的是指标名：

```
EVIDENCE_RECALL = "evidence_recall"   # 实际含义：有 gold 文档标注
ANSWER_F1       = "answer_f1"         # 实际含义：有参考答案
```

judge 指标的依赖与这两个对不齐 —— faithfulness / answer relevancy
**不需要任何标注**，对四组都成立，声明它们没有信息量。所以反转成
「指标声明依赖、数据集声明拥有、闸门做集合比对」，新增指标不再碰枚举，
而「拒绝计算而不是返回 0.0」的保护自动继承。

**一个具体收获**：narrativeqa 现在整组检索指标省略（无 gold），
而 faithfulness / context precision 不需要 gold 文档就能算 —— judge 能填上这个洞，
且它恰恰最需要（46% 的参考答案措辞在原文里根本不存在，F1 绝对值信息量最低）。

改造顺带清一笔账：磁盘上 `data/normalized/*/manifest.json` 与
`data/subsets/run001/*/manifest.json` 记的是 `answer_em_f1`，当前代码是
`answer_f1`（产物早于改名）。这个不一致已经存在，一起清掉。

### 12.5 judge、归因与标注是三件不同的事

别合。三者的产物、去向、汇总语义都不同：

| | 产物 | 去向 | 参与汇总 |
| --- | --- | --- | --- |
| **judge 指标** | 分数 | 指标层 | 是 |
| **LLM 归因** | 结构化标签 + 理由 | 标注表（作者=模型） | 否 |
| **人工标注** | 同上 | 标注表（作者=human） | 否 |

后两者同表、只差作者列，因此 judge-human 一致率是一个 `GROUP BY` 就能算出来的
免费产物 —— 而它是判断「这个 LLM 归因能不能信」的唯一办法。

**标签取值**从现有分析里提炼，它们本来就是 metrics.md 已区分开的归因类别：
`annotation_wording`（gold 全名 vs 模型通称）、`answer_form`（散文稀释）、
`gold_incomplete`、`retrieval_miss`、`citation_dropped`（对应 `truncated_gold`）、
`question_ambiguous`。有标签才能回答「这批低分里多少是标注问题、多少是真检索失败」。

**标签与指标可交叉验证**：标了 `citation_dropped` 的样本，`truncated_gold` 应 > 0。
不一致说明判断或指标有一个错了 —— 这本身是有用信号，也是
第 15 条决策（归因不看指标）存在的理由：看过指标再出标签，这条验证就变成空的。

judge 失败要分四类记（限流 / 超时 / 解析失败 / 模型拒答），处置完全不同。
重试直接复用 `akasha_client.py` 那套（`RETRYABLE_STATUSES = {429,502,503,504}`、
5 次指数退避带抖动、4xx 不重试），语义对 judge 适用，不必重写。

judge provider 配置进指标层的身份哈希时，**只能进 `base_url` + `model`，
绝不能进 api_key**。`redacted()` 建议改成白名单式（现在是黑名单，
漏写一个字段就泄露密钥）。

### 12.6 ground truth 的取巧办法与它的偏差（预留功能）

自有知识库上标 gold 文档的候选不必从全库找 —— 从**多次查询的
`retrievedSources` 并集**里找就够（几十篇而非几千篇），LLM 判候选、人确认。

**但这个取巧有必须写进报告的偏差**：从召回并集标出的 gold，天然排除了
「所有配置都没召回到的那些真 gold」。所以据此算出的 recall@k 是**偏高的上界**，
能回答「配置 A 比 B 好多少」，不能回答「绝对检索水平如何」。
性质与 §0.3 同类，处理方式也该一样 —— 写进报告，不留给读者自己发现。

参考答案**不该让 LLM 写**：judge 用 LLM 判答案对错，参考答案又是 LLM 写的，
等于自己出题自己判。标注记 `source`（`human` / `model` / `model_confirmed_by_human`）
与 `confidence`，这样任何指标结果都能追溯到「它依赖的 gold 有多少是人确认过的」。

### 12.7 目录与运维

```
src/akasha_benchmark/        现有，阶段代码 + 数据层（新增 store/）
src/akasha_platform/         FastAPI 后端
web/                         Vite + React 前端
migrations/                  SQLite schema 演进
```

开发时 Vite dev server + FastAPI 两进程；生产 `npm run build` 出静态文件由
FastAPI 挂 `StaticFiles`，单进程单端口。`[project.scripts]` 加 `akasha-platform`，
Makefile 加 `make serve`。

**迁移必须能在有数据的库上跑**，不能只在空库验证过 —— SQLite 的 `ALTER TABLE`
不能删列改类型，复杂改动走「建新表 → 拷数据 → 换名」，且要在真实数据的库副本上
先跑一遍。**迁移前自动备份**（`cp` 成 `akasha_bench.db.pre-{version}`），
20 行代码换掉一整类事故。

`reindex` 从文件重建产物索引时**必须显式绕开标注表与 judge 判决表** ——
这两张表不可重建，与产物表同库，一个粗心的 `DELETE FROM` 就没了。

### 12.8 设计期间查实的几条事实（会影响实现）

- **`concurrency` 是死字段**。只出现在定义、类型转换表、写进 manifest 三处，
  **没有任何代码读它控制行为**。真正生效的是 `request_interval_seconds`（默认 0.5s），
  `_throttle()` 在每个请求前 sleep。所谓「串行」不是并发度设成 1，是压根没写并发。
- **编译并发不在我们手里**。导入完成后调一次 `compile-spaces` 建 Run，之后只是轮询等；
  真正在编译的是 Akasha 的 BullMQ worker。观察到的约 40 秒/篇是那边的吞吐，
  客户端怎么调都改不了。ingest 的进度条本质是「帮我盯着别人干活」。
- **查询是唯一并行有收益的环节**：358 条 × 10–14 秒约 70–83 分钟。
  但 answer 模型走第三方 OpenAI 兼容端点（配额未知），且 `httpx.Client`
  并发要开多个实例或改 async。
- **导入并行收益很小**（编译才是那 15 小时），且写入端点 `import_page` / `create_space`
  在重试机制里是显式不重试的 —— 5xx 后结果有歧义，可能建出第二个 page 而
  page_map 里没有。并发会让这个歧义更难查。
- **`apiKeySet` 只是布尔量，不回传 key**。所以 judge 即便用同一个端点同一个模型，
  平台也得自己配一份凭据。这是平台第一次持有 LLM 密钥。
- **`numpy==2.5.3` 是死依赖**，全仓库零引用。加 web 依赖时一起删。
- **`docs/diagrams/*.html` 是静态文档产物**（717KB/721KB 自包含，支持 `?embed=1`），
  可以 iframe 嵌进前端当架构说明，但数字写死在 JSON spec 里，
  叠不上本次 run 的真实数据。真实数据的血缘视图另做，两件事别混。
- **Akasha 跑在 WSL 内部的 podman 里**，Windows 侧 `podman ps` 看不到它
  （`/api/health` 返回 200 但容器列表为空）。运维文档要写清从哪连、连不上时怎么查。
- 测试基线已是 **108 passed / 32 skipped**，§11 记的 86/23 已过期。

### 12.9 血缘视图：确切的查询链路（实测，非推断）

平台的核心能力。下面这条链路是 2026-09-10 在 run001 的真实库上逐跳走通的，
表名列名均已核对 —— **不是照 Akasha 文档抄的**。

```
pages.id                      ← page_map.jsonl 里的 page_id（导入接口返回）
  ↓ knowledge_page_sources.source_page_id
knowledge_pages.id            ← 编译产出的 artifact，不是原始 page
  ↓ knowledge_chunks.knowledge_page_id          参与召回的文本
  ↓ knowledge_graph_edges.from/to_knowledge_page_id   图边
  ↓ knowledge_claims.knowledge_page_id          抽出的断言
knowledge_source_chunks.source_page_id          原文，不参与召回
```

**两处与先前假设不符，已纠正**：`knowledge_chunks` 的外键是 `knowledge_page_id`
而非 `source_page_id`；`knowledge_page_sources` **直接带 `source_page_id`**，
不必经 `knowledge_sources` 中转。

关键列：`knowledge_pages` 有 `page_type`（`entity` / `source_summary`）、
`canonical_key`（实体合并的键）、`compile_scope`、`stale_at`；
`knowledge_chunks` 有 `chunk_role`、`retrieval_channel`、`embedding_profile`、
`search_tsv`；`knowledge_graph_edges` 有 `relation`（实测取值 `produced`、
`released_album`）。workspace 下 `knowledge_*` 表共 29 张。

#### 验收样例：一条 recall@5 = 0.5 的样本，六跳定位到根因

`hotpotqa:5ae4f2595542990ba0bbb1a8`（bridge / hard，gold 2 篇）。
gold `6369 Cyndi Lauper` 排第 1，gold `6365 Dee Does Broadway` **到 k=20 都没出现**。
`includedItemCount: 20` / `omittedItemCount: 0`，所以不是上下文预算挤掉的；
它压根没进候选集。

6365 编成 3 个 artifact（`Dee Does Broadway` / `Dee Snider` /
`Source Summary: …`），**全部图边只有 4 条，都在 Dee Snider ↔ Dee Does Broadway
之间**，到 `canonical_key = cyndi_lauper` 的边 **0 条** —— 嘉宾关系没升格成图边。

而更要紧的是编译把查询需要的短语删了：

```
原文（knowledge_source_chunks，不参与召回）：
  "Guests in the album include the Grammy and Emmy award winning Cyndi Lauper, …"
编译（knowledge_chunks，参与召回）：
  "…featuring vocal contributions from guest artists including Cyndi Lauper, …"
```

三个 artifact 的 chunk 正文全部不含 `Grammy` / `Emmy`。而问题问的正是
*"who won Grammy and Emmy award"*。于是三条召回路径同时断：词法（词已不在索引文本里）、
稠密（编译产物主题是「Dee Snider 的百老汇专辑」，与「歌手生日」语义远）、
图扩展（那条边不存在）。

**这是 §0.3 那条架构论断的具体实例，也是「非调参可解」的证据** ——
调 `scoreThreshold`、加大 k、换检索模式都救不回来。

完整案例（含三条召回路径为什么同时断、以及跨全库的系统性验证）见
[cases/recall-miss-compiled-away.md](cases/recall-miss-compiled-away.md)。
只读探查脚本在 [cases/scripts/](cases/scripts/)，可重跑。

**跨全库验证得到的三条编译器行为**（400+ 篇，非单例）：
编译**不是压缩而是扩写**（中位 2.19 倍，仅 2.7% 净压缩），所以丢修饰语是改写策略
而非空间不足；**图边极稀疏**（1454 artifact / 555 边，59% 的 entity 零出边）；
**relation 自由生成**（555 条边散在 377 种取值上，295 种只出现一次，
`createdBy` 与 `created_by` 并存、`father of` 带空格），
所以图遍历无法按关系类型做。

**顺带更正 §0.3 的一条预判**：那里担心「多跳可能偏高，因为实体跨文档合并」，
实测 925 个 entity artifact 里被多于一篇原文贡献的**只有 12 个（1.3%）**——
合并几乎没发生，方向反而是「桥接关系没建立、多跳靠图走不通」。
机制描述没错，只是这份语料上极少触发；语料更密时结论可能不同。

#### 由此确定的两个视图

1. **逐样本血缘**：失败样本 → gold doc_id → page_id → artifact 列表 → chunk 正文
   → 图边 → 原文。跨五张表六跳，必须一屏走完，否则归因就得像这次一样手写 SQL。
2. **原文 vs 编译产物 diff**：升为一等视图。这次的根因只有把两者并排才看得见。
   §9 的原文基线从整体上量这个效应，而**逐样本 diff 能直接指出丢了哪个词** ——
   对调参无用（改不了编译器），对「这个数字该怎么读」有决定性作用。

#### 顺带实测到的检索侧事实

`audit_join.json`（99/100 样本匹配上）：`mean_candidate_chunks` 恒为 **200.0**
（硬上限，不是自然分布）、`mean_ranking_loss` **152.04**、
`mean_authorization_loss` **28.07**、`recall_ceiling_miss_rate` **0.0**、
`gold_hit_rate` **0.9697**、`access_policy_fallback_rate` 0.0，
`retrieval_mode` 全部是 `high_completeness`。候选集从不为空，
损失全在排序段 —— 200 个候选里平均砍掉 152 个。

### 12.10 还没定的（下一轮）

- **第一版的先后**：观测层与执行控制哪个先上。
- **三层的身份键**：配置哈希（内容寻址）还是用户命名 + 自增 ID。
  决定「两个索引层什么时候算同一个」，也决定复用规则。
- **认证与绑定**：默认只绑 `127.0.0.1`。红线 —— 这个服务持有 Akasha 管理员凭据、
  只读数据库连接、以及启动长任务的能力，**任何时候暴露到 `0.0.0.0` 都必须先有认证**，
  不是「以后再说」的项。
- **前端信息架构**：血缘视图（§12.9）、对比视图、任务视图怎么组织。
  `snippets[].id` 是裸 UUID 不带类型前缀且 snippet 里没有 `kind`，
  光看响应分不出这条来自原文块还是编译产物 —— 现已确认可用
  `knowledge_chunks.id` 反查补上 `page_type` 与 `chunk_role`。
- **失败案例入口的默认切分**：四条 `recall@5 < 1.0` 里三条是 `answerMode: general`
  （生成端回落，`retrievedSources` 被无条件清空），只有一条是真的漏 gold ——
  而那一条答案还是对的。混在一起看会把 1 条检索问题读成 4 条。
  按 `answerMode` 分开必须是默认视图，不是可选筛选器。
- **`full_coverage@k` 该不该顶替 `recall@k` 当主指标**：hotpotqa 全部 2 篇 gold，
  recall 只有 0/0.5/1 三个取值，均值 0.965 的可读性不如 `full_coverage@5 = 0.96`。
- **judge 具体指标清单**：faithfulness / answer relevancy / context precision 等
  各自的 prompt、输出 schema 与依赖声明。
- **answer 模型热切换**要不要做进第一版（`put_model_config` 已就绪但零调用）。

