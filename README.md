# Akasha-Benchmark

评测 Akasha 的多跳检索、引用归因与端到端答案质量。全部代码在本仓库，
**不修改 Akasha 主仓库** —— Akasha 侧只通过 HTTP 访问。

实施计划见 [PLAN.md](docs/PLAN.md)，数据集字段说明见 [docs/datasets.md](docs/datasets.md)，
接口请求／响应样例见 [docs/akasha_api.md](docs/akasha_api.md)，
评测指标见 [docs/metrics.md](docs/metrics.md)，
逐样本追到根因的排查记录见 [docs/cases/](docs/cases/)。

## 先读这一条

**Akasha 的稠密/词法召回跑在 LLM 生成的文本上，不是原文。**
`knowledge_chunks` 索引的是编译器产出的 `artifact.markdown`；原文在
`knowledge_source_chunks`，该表不参与召回，只在引用解析时提供证据窗口。

直接后果是 Recall@k 系统性偏低，且非调参可解 —— 编译丢掉的词已经不在被索引的
文本里，调 `scoreThreshold`、加大 k、换检索模式都救不回来。

多跳方向曾预判为「可能偏高，因为实体跨文档合并」，**实测不成立**：925 个 entity
artifact 里被多于一篇原文贡献的只有 12 个（1.3%），合并几乎没发生；
方向反而是桥接关系没升格成图边、多跳靠图走不通。机制描述没错，
只是这份语料上极少触发（`docs/cases/recall-miss-compiled-away.md`）。

**因此本仓库产出的数字与已发表 baseline 直接比较是无效的**，
只在 Akasha 自身的不同配置之间横向可比。真正的对照是 PLAN.md §9 的原文基线。

## 环境

```bash
uv sync                                   # Python 3.12，勿用 3.14
uv run python scripts/download_datasets.py         # 下载四组数据到 dataset/
uv run python scripts/download_datasets.py --check # 只校验
```

平台的前端需要 Node（实测 24 / npm 11）。入库和查询需要 Akasha 跑在 docker
（`db` + `redis` + `akasha`，BullMQ 必须要 redis）。

## 跑起来

```bash
make setup    # sync -> download -> migrate -> normalize -> validate
make web      # 构建前端（npm install + build）
make serve    # 单进程单端口，默认只绑 127.0.0.1
```

然后在界面上填 Akasha 连接、勾指标、跑实验。**没有跑实验用的命令行参数** ——
执行控制在平台里，参数存进库里的 `run_config`，argv 只带它的 id。两处并存的话
迟早会漂，而症状是「界面上改了但跑的还是默认值」。

**配置只有一个来源：库。** 在界面的配置层里填，项目目录不留配置文件。曾经有过
两条旁路（`akasha.config.json`、`AKASHA_*` 环境变量覆盖），两条都删了 ——
同一份配置有多个来源时，「我改了但没生效」是查不出来的。

**代价明说**：`akasha_bench.db` 因此存着明文 Akasha 密码与模型端点的 api_key，
它是一个凭据文件（已 gitignore）。备份、拷贝、分享它等于分享凭据。

唯一仍走环境变量的是 `AKASHA_PLATFORM_*`（host / port / 库的位置 / 访问令牌），
因为它是 bootstrap：库的位置本身就在里面，不可能从库里读。前三项另有命令行参数
（`akasha-platform --host/--port/--db`）。

### 配置只有一处

配置层里四组，一处改完：Akasha 连接、**Akasha 那边的模型配置**（compiler /
embedding / answer / image）、judge 模型、归因分析模型。第二组原先在编译层 ——
挪过来是因为「改一个模型要去哪」不该取决于它属于哪一层。

**连接只有一份**，只能改，不能新增或删除（`CHECK (id = 1)` 写在 schema 里）。
历史记录不靠多行：`index_layer.connection_json` 与 `workspace_id` 存了**入库时**
那份配置的快照。配置改过之后层上显示的仍是历史真相 —— 显示当前配置会让人以为
那一层跑在新配置上。

改 `base_url` / `email` 可能落到另一个 workspace，而已入库的层的 `page_map` 只在
原来那个里有意义。两道闸门拦住那个静默失效，**都在登录之后判**：

| 闸门 | 拦的是 | 判据来自 |
| --- | --- | --- |
| workspace 不匹配 | 层入库在 workspace A，现在登录到 B | `users/me` |
| `ensure_space` 校验 space 身份 | 同 slug 解析到另一个 space | `list_spaces` |

**配置上没有 workspace_id 这一项。** workspace 由服务端决定（自建部署走
`workspaceRepo.findFirst()`），客户端选不了 —— 让用户填一个他决定不了的值，
填错时闸门会误报。

不拦的话：`_find_space` 按当前 workspace 过滤找不到同 slug 的 space →
`ensure_space` 建一个新的 → `set_space` 覆盖库里的 space_id → 那 1722 个 page_id
全部悬空。而查询照常跑完，每条都召回不到 —— 看起来像「这批语料检索效果差」。

真要换部署就先清掉那一层的入库产物（`POST /api/layers/index/{id}/discard-ingest`，
要 `confirm=true`）：清 page_map / space 绑定 / 质量闸门 / 编译记录，**子集保留**,
**远端的 space 不删**。响应里按约 40 秒/篇报出重新入库的代价。

代价是这道闸门不能离线判：`index_layer_readiness` 是纯库函数，登不了 Akasha。
所以它只报层记下的那个值，比对交给 ingest / query。拦截时机没变 ——
两者都在发出任何写入或查询之前判。

### 一个 Akasha 侧的约束

`/admin/model-configs` 的请求与响应里**都没有 workspaceId**，所以那份配置至少是
跨账号共享的。同一个部署上换账号**换不出**不同的编译模型；要那样得换部署
（另一个 base_url）。多人共用一个部署时它是共享可变状态，唯一的保护是入库/查询前
的快照比对：embedding 漂移拒绝执行，compiler 漂移警告。

judge 与归因分析模型各需要一份自己的凭据 —— Akasha 的 `/model-configs` 只回传
`apiKeySet` 布尔量，从不回传 key。在配置层填，或填一个环境变量名让它从那里读。

### 界面的八层

顺序即流水线顺序，每层都能看到这一段的任意样本：

| 层 | 看什么 | 配什么 |
| --- | --- | --- |
| 数据集 | 原始样例（直接读 `dataset/*.json`，不经适配器） | — |
| 归一化 | 适配器状态、归一化后的样本与 gold 正文 | — |
| 编译层 | 任意文档的原文 ↔ 编译产物 diff、六跳链路 | 抽样参数（条数/种子/负样本比）、起入库 |
| 评测层 | 问题的生成结果与完整响应体 | 勾指标、选编译数据、answer / judge 模型 |
| 报告层 | 本轮所选指标的结果与分层 | — |
| 归因层 | 原文→编译→检索→生成完整链路 + 自动根因 | 分析模型 |
| 任务 | 各阶段任务的进度与日志 | 启动 / 停止 / 清理 |
| 配置 | — | Akasha 连接、Akasha 的模型配置、judge、归因分析 |

指标不单列一层：它是评测层的可勾选配置，勾选范围由所选数据集**拥有的标注**
决定。勾了只对部分组成立的指标不会报错 —— 缺依赖的那组会省略它并写明原因，
而不是伪造 0 分。

归因分两段：**规则在前**，判据全部来自已有指标与链路，不需要任何模型配置就能
出结果；配了分析模型会在分类之上补一段因果叙述。规则给分类（哪一段断了），
模型给叙述（为什么断在那里）—— 把后者当分类用会得到一个听起来有道理、
但与指标对不上的结论。

**库是事实来源**：各阶段读写根目录的 `akasha_bench.db`。磁盘上的 jsonl/md 是
可选导出，不是任何阶段的输入。入库和查询都**可恢复**：重跑读库里已有的
`page_map` / `query_response`，跳过已完成的条目。所以任务的「停止」是可续跑的,
不是假装挂起一个进程。

开发时两个进程：`uv run akasha-platform`（API :8848）+
`npm --prefix web run dev`（Vite :5173，`/api` 代理过去）。

> 这个服务持有 Akasha 管理员凭据、只读数据库连接、以及启动长任务的能力。
> 默认只绑 `127.0.0.1`；绑非回环地址时若未设 `AKASHA_PLATFORM_AUTH_TOKEN`
> 会**拒绝启动**。

## 三层与它们的身份

成本边界画出了三层（`docs/PLAN.md` §12.3）。库里叫索引层／查询层／评测层，
界面上把索引层叫「编译层」—— 用户关心的是这批文档编译成什么样了，
而抽样与导入是达成它的手段：

| 层 | 由什么决定 | 成本 |
| --- | --- | --- |
| 索引层（编译层） | 子集配置 + compiler + embedding | 约 15 小时 / 1722 篇 |
| 查询层 | 挂某个索引层 + answer 模型 + `scoreThreshold` | 10–14 秒 × 每条 |
| 评测层 | 挂某个查询层 + 指标组 + k（+ judge 配置） | 确定性指标秒级 |

每层有自增 id 加人可读的 label，以及一个哈希判「两个层是不是同一个」。
索引层的 `subset_hash` 是**内容寻址的**（对实际文档集取哈希），不是配置寻址的 ——
配置寻址会说谎：随机源里只要有配置之外的东西，或者产物是 `reindex` 导进来的
历史数据，「配置相同」就不再等于「文档相同」。

`config_hash` 只在入库时才写得出来（那时才有模型配置），在那之前是 NULL ——
一个索引层在编译之前身份本就不完整。

## 产出

权威在 `akasha_bench.db`。加 `--export` 可另外落一份文件快照：

```
data/
  normalized/{dataset}/     samples.jsonl  corpus.jsonl  manifest.json
  subsets/{label}/{ds}/     samples.jsonl  corpus/{doc_id}.md  corpus_hashes.json
  reports/{eval_label}/     metrics.json  per_sample.jsonl  report.md
```

文件写入走「临时文件 + `os.replace`」原子写，中断不会留下截断文件。

库里有两类表，`reindex` 只准动第一类：**产物表**可从上游重算，会先清空再重建;
**不可重建表**是 `annotation` / `judge_verdict` / `judge_provider` /
`badcase_analysis` —— 它们是人和模型的判断，没有上游可重算。

## 几条贯穿全程的约束

- **显式适配器**：每个数据集一个类，按名字分派。禁止按字段存在性猜 schema。
- **严格模型**：`extra="forbid"` + `frozen=True`，多余字段报错而非忽略。
- **数据依赖声明**：数据集声明自己**拥有**什么标注（`DataDependency`），
  指标声明自己**需要**什么，闸门做集合比对。narrativeqa 无 gold 文档，
  请求检索指标时**抛异常**，不返回 0.0 —— 假分数会污染汇总。
  新增指标不必碰枚举，而 judge 类指标（`requires` 为空集）对四组都成立。
- **身份不推断**：口径收在 `datasets/models.py` 的 `CORPUS_ID_RULES` /
  `SAMPLE_ID_RULES` 一处。按 ID 匹配后再比一次 question 文本，不一致报错。
- **corpus 不去重**：musique 有 647 个重复 title 但它们是不同段落，
  去重会丢 gold。去重前后条数都进 manifest。

## 读指标时注意

`no_match` 和 `general` 两种 answerMode 会**无条件**返回空的 `retrievedSources`
（`ai-knowledge-chat.service.ts:641,667`），所以这些行的检索分数天然是 0，
与检索实际找到了什么无关。报告里每张检索表都出两份 —— 全样本、
以及只算 `answerMode == "knowledge"` 的切片，差值就是生成端拒答的规模。

平台里这个切分是**默认布局而不是筛选器**：run001 上四条 `recall@5 < 1.0` 里
三条是 `answerMode: general`，只有一条是真的漏 gold —— 而那一条答案还是对的。
混在一起看会把 1 条检索问题读成 4 条。

`retrievalDiagnostics` 不在 HTTP 响应里（controller 解构排除了它），
分层归因必须走 `audit_join`（读 `knowledge_query_audit.metadata`）。
连接键是 `sha256:<hex>`，**带 `sha256:` 前缀**。那张表是 Akasha 的运行时表、
会随容器重建消失，所以 `audit_join` 会把结果抄进 `audit_record` 存档。

Exact Match **预期恒为 0**：Akasha 返回解释性散文，参考答案是短跨度，
整串相等不可能成立。把它当答案**形态**的探针读，不当质量指标读。

## 测试

```bash
uv run pytest tests/ -q
```

**262 passed / 36 skipped**（skip 的是需要 Akasha 或只读数据库在线的那些）。

入库和查询的测试跑在 `httpx.MockTransport` 上，覆盖 multipart 字段名、
OWNER 闸门、质量闸门、断点续跑、失败照样落盘。

数据层那批用例每一条都对着一个具体的失效方式，不是为覆盖率写的 —— 库成了
事实来源之后，那四条完整性保证全落在「错了不报错、只给出看着合理的假结果」
的地带：

| 用例锁住的性质 | 不锁会怎样 |
| --- | --- |
| 续跑键含 dataset 维 | doc_id 跨组撞 60 个，前一组被误判成已导入而缺篇 |
| 质量闸门缺字段判失败 | `all(value == 0)` 对空值集合返回 True，假通过 |
| `subset_hash` 内容寻址 | 两个不同子集拿到同一哈希，UI 并列做对照 |
| judge 失败该条排除 | 一次 429 风暴看起来像模型突然变笨 |
| 迁移能在有数据的库上跑 | 只在空库验证过的迁移，第一次真用就炸 |
| 归因先判生成端回落 | 4 条低分里 3 条是拒答，判错顺序会读成 4 条检索问题 |
| 密钥字段空串表示不改 | 每次改 base_url 都顺手把密码清掉，下次登录才发现 |
| workspace 比对用服务端的值 | 用户填错一个他决定不了的字段，闸门就会误报 |
| 连接是单例且 CHECK 在 schema 里 | 插进第二行后「哪一行是真的」没有答案 |
| 层上显示入库时的身份 | 配置改过之后，显示当前配置等于说谎 |
| 已入库的层拒绝重抽子集 | subset_doc 换了、page_map 没换，检索指标全变 0 |
| readiness 按文档集合比而非条数 | 条数相等而集合不同时，那个状态看起来完全正常 |
| 拦截在任何写入之前 | ensure_space 跑过之后 space_id 已经被覆盖了 |
| 同 slug 不等于同一个 space | 语料导进另一个 workspace，库里记的映射全部失效 |
| 迁移回填不丢入库产物 | 那 1722 页要重挣约 19 小时编译 |
| run_config 白名单过滤 | 请求体能决定阶段进程读到什么参数，那是个远程执行面 |
| 参数名与阶段 argparse 对齐 | 界面上改了、跑的还是默认值，且没有任何地方报错 |

血缘视图另有一组在线用例，判据不是「接口返回 200」，而是**平台给出的结论与
当初手写六跳 SQL 的结论逐条相同**：

```bash
AKASHA_LIVE=1 uv run pytest tests/test_live_lineage.py -v
```

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
