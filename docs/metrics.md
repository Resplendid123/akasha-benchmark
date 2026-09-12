# 评测指标说明

指标实现在 [metrics/](../src/akasha_benchmark/metrics/)，
评估编排在 [evaluate.py](../src/akasha_benchmark/evaluate.py)。

```bash
uv run python -m akasha_benchmark.evaluate --query-label run001-query
uv run python -m akasha_benchmark.evaluate --query-label run001-query --k 5 --k 20   # 自定义 k
```

确定性评测默认写入数据库；加 `--export` 可在 `data/reports/{eval_label}/` 导出：

| 文件 | 内容 |
| --- | --- |
| `metrics.json` | 逐数据集汇总，机器读 |
| `per_sample.jsonl` | 逐样本明细，用于复查单条和自定义切片 |
| `report.md` | 中文报告，带架构说明和分层表 |

四组指标：

| 组 | 模块 | 回答什么 |
| --- | --- | --- |
| 答案质量 | [qa.py](../src/akasha_benchmark/metrics/qa.py) | 答得对不对 |
| 检索 | [retrieval.py](../src/akasha_benchmark/metrics/retrieval.py) | 该看的文档找到了没有 |
| 引用归因 | [attribution.py](../src/akasha_benchmark/metrics/attribution.py) | 找到了的有没有真的用上 |
| 多跳专项 | [multihop.py](../src/akasha_benchmark/metrics/multihop.py) | 图扩展有没有净贡献 |

## 比较前先对齐口径

Akasha 检索编译产物，编译可能遗漏或改写原文，也可能改变跨文档实体关系。
指标同时受编译、检索和答案形态影响；不能仅凭单个案例断言整体偏高或偏低。
与公开结果比较前需对齐语料、样本、检索单位和输出格式。原文检索基线可用于隔离编译的影响。

## 答案质量 QA

`normalize_answer` 照抄 SQuAD/MRQA 的标准口径，**顺序不能换**：
小写 → 去标点 → 去冠词 `a`/`an`/`the` → 合并空白。

```
"The Beatles!"            -> "beatles"
"  A  Hard   Day's Night" -> "hard days night"
"An apple, an orange."    -> "apple orange"
```

照抄而不是自创，是为了让数字至少在口径上能跟已发表结果对上。自创归一化会让
所有对外比较失去意义。

两个指标，都在归一化之后算：

| 指标 | 定义 | 怎么算 |
| --- | --- | --- |
| `em` | 整串完全相等才算 1 | `normalize_answer(pred) == normalize_answer(ref)` |
| `f1` | 词袋级 F1，重复词按出现次数取交集 | 见下式 |

F1 用 `Counter` 取交集，所以重复词按出现次数计：

```
shared    = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
precision = shared / len(pred_tokens)      # 分母是预测的长度
recall    = shared / len(ref_tokens)       # 分母是参考的长度
f1        = 2 * precision * recall / (precision + recall)
```

`precision` 的分母是**预测答案的 token 数** —— 这一条是后面所有低分的机制来源，
预测越长，分母越大，F1 越低，与答案对不对无关。

**两个指标各自独立对多条参考取 max**，不是先挑一条参考再算两个数
（后者会给出一个两边都不最优的组合，`test_metrics.py` 有一条钉这个）。
musique 数据集中的别名在归一化时已并入 `answers`，narrativeqa 本身带 2 条人工参考，
所以打分侧不用再写一遍别名逻辑。

退化情况与官方 SQuAD 脚本一致：预测和参考都空算完全匹配（`f1 = 1.0`），
预测空而参考非空算 0。

`answer_mode_distribution` 报各 `answerMode` 的占比，缺失值单独归到 `missing`
而不是并进别的桶。`no_match` 率和 `general` 兜底率是「检索没喂够料」的直接信号。

### EM 与答案长度

EM 要求归一化后的整段答案完全相等。解释性长答案即使包含正确答案也可能得零，
但输出与参考一致时仍得 1。结合 F1、引用证据和人工抽查解读，不能将 EM 零分直接等同于语义错误。

### 为什么 F1 也低：precision 塌了

F1 受同一效应影响但**仍然是变化的**，可用于观察词面重叠。同一批冒烟
逐 token 分解（用 `tokenize` 重算，mean F1 = 0.0946，与报告一致）：

| sample | \|pred\| | \|gold\| | shared | P | R | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| 5a710a1e55 | 36 | 1 | 1 | 0.0278 | 1.0000 | 0.0541 |
| 5a717d4c55 | 43 | 3 | 2 | 0.0465 | 0.6667 | 0.0870 |
| 5a71835755 | 23 | 5 | 2 | 0.0870 | 0.4000 | 0.1429 |
| **mean** | | | | **0.0537** | **0.6889** | **0.0946** |

**recall 0.69、precision 0.054，差 13 倍。** 该答出的词答出来了，分是被
precision 的分母吃掉的。两个独立效应，量级差很多：

**一、散文稀释（主因，只压 precision）。** 预测 23–43 token，参考 1–5 token。
最极端的一条 gold 只有一个词 `Flavivirus`：

```
Q:    What is the genus of the viral disease that has symptoms such as fever, chills,
      loss of appetite, nausea, muscle pains, and headaches, and has a chance of
      causing liver damage?
gold: Flavivirus                                                          (1 token)
pred: The disease described is yellow fever, which is caused by the yellow fever
      virus belonging to the genus **Flavivirus** . Yellow fever symptoms include
      fever, chills, loss of appetite, nausea, muscle pains, and headaches, and
      it can cause liver damage .                                        (36 token)
```

归一化后逐 token 归类（`^` 命中 gold、`q` 抄自问题、`.` 其他）：

```
disease[q] described[.] is[q] yellow[.] fever[q] which[.] is[q] caused[.] by[.]
yellow[.] fever[q] virus[.] belonging[.] to[.] genus[q] flavivirus[^] yellow[.]
fever[q] symptoms[q] include[.] fever[q] chills[q] loss[q] of[q] appetite[q]
nausea[q] muscle[q] pains[q] and[q] headaches[q] and[q] it[.] can[.] cause[.]
liver[q] damage[q]
```

| 类别 | n | 占比 |
| --- | --- | --- |
| 命中 gold | 1 | 2.8% |
| 抄自问题 | 21 | 58.3% |
| 其他（补全推理链） | 14 | 38.9% |

最大头是**复述问题的条件**：第二句把症状清单原样抄回来当作「症状对得上」的论证，
21 个 token 全部是问题里已有的词，对答案零信息量，但每一个都进 precision 的分母。
其余是显式写出中间实体（yellow fever → yellow fever virus → 属），多跳题上这是
可解释性的优点，打分时是纯负担。真正答题的只有 `flavivirus` 一个词，埋在第 16 位。

反事实：把 precision 修满、recall 保持不动，mean F1 会从 0.0946 升到 **0.7905**。
**绝大部分损失来自这一条。**

**二、gold 用全名、模型用通称（次因，只压 recall）。** hotpotqa 这批每条只有
1 个参考答案，没有别名，多参考取 max 在这里帮不上忙：

| gold | 模型答 | 丢掉的 token | R |
| --- | --- | --- | --- |
| `Pavel Sergeyevich Alexandrov` | `Pavel Alexandrov` | `sergeyevich` | 0.667 |
| `Petrus Josephus Hubertus (Pierre) Cuypers` | `Pierre Cuypers` | `petrus` `josephus` `hubertus` | 0.400 |

这是标注口径问题，不是答错。

**答案文风改不了。** `run_queries.py` 和 `akasha_client.py` 的请求体里只有 query
和 `scoreThreshold`，没有任何控制答案形态的参数（prompt / answerStyle / 长度约束
都没有），`budget.responseReserve` 也是 0。文风由 Akasha 服务端的 answer 生成逻辑
决定，评测侧无法干预。所以这个稀释在当前架构下是结构性的。

由此，F1 的用法：

- 只在**同配置之间**比较（比如调 `scoreThreshold` 前后），不与公开 baseline 比
- 判断答案对不对，看 `citation_precision` / `evidence_verifiable_rate` 配合人工抽查
- 原文基线有助于隔离编译影响 —— 它跑在同一个生成端上、啰嗦程度相当，
  所以两者的 F1 差值仍然可读

想让绝对值本身可读，需要加「答案是否包含参考答案」的宽松指标（containment）。
**目前没有实现**，因为它同样有偏 —— 散文越长越容易蒙中，且上面那两条全名样本
它同样判 0（整串包含不成立，实测 3 条只命中 1 条）。加不加取决于要回答什么问题，


### narrativeqa 的答案分数有个措辞造成的上限

原始 NarrativeQA 的标注者是**只看剧情摘要**写问题和答案的，所以参考答案用的是
摘要的措辞（Wikipedia 式概括），不是剧本／原著的措辞。而检索跑在全文切出来的块上，
拿到的是原文措辞的片段。实测这批参考答案（整串归一化后）能否原样找到：

| 在哪找 | 命中 |
| --- | --- |
| `document.summary.text` | 153 / 293（52.2%） |
| corpus 全文 | 105 / 293（35.8%） |
| 两边都找不到 | 135（46%） |

摘要只占全文 0.58%–2.51%，却比全文更容易命中答案 —— 这就是措辞错配。
有 46% 的题参考答案的措辞在检索得到的原文里根本不存在，只能靠词袋重叠拿部分分。
**分数低不代表系统坏了。** 判断 narrativeqa 表现只看同组内的相对变化，
不要跨数据集直接比 F1 绝对值。

这个上限与上一节的散文稀释是**两个独立的效应**，且叠加：narrativeqa 既有措辞
错配，答案又是散文。所以它的 F1 绝对值信息量最低，四组里最不该单独解读。

（`summary.text` 本身不进规范化数据，corpus 是从全文切的块，
见 [normalized_datasets.md](normalized_datasets.md)。）

## 检索质量 Retrieval

用 `retrievedSources` 算，**不用** `citations`。前者是裁剪前的召回全集，
后者已经被「被引 ∩ 有证据」的交集裁过一遍，拿它算 Recall 会低估检索能力。

相关性是二元的（所有 gold 同权），所以 nDCG 的理想排序就是把全部 gold 排在最前。

默认 k 是 `(2, 5, 10, 20)`，每个 k 出四个数：

| 指标 | 定义 | 何时用 |
| --- | --- | --- |
| `recall@k` | 前 k 个里命中的 gold 占全部 gold 的比例 | 主指标 |
| `hit@k` | 前 k 个里**至少**命中一个 gold 就算 1 | 宽松下界 |
| `ndcg@k` | 二元相关性下的 nDCG，理想排序是 gold 占最前 `min(len(gold), k)` 位 | 关心排序质量 |
| `full_coverage@k` | 前 k 个里**凑齐全部** gold 才算 1 | 多跳的真实需求 |

外加一个与 k 无关的 `mrr`：首个 gold 的倒数排名，没命中记 0。

nDCG 的折扣按排名取 log2，二元相关性下增益就是「是不是 gold」：

```
DCG@k  = Σ 1/log2(rank+1)        rank 从 1 数，只累加命中 gold 的位置
IDCG@k = Σ 1/log2(rank+1)        rank 从 1 到 min(|gold|, k)
nDCG@k = DCG@k / IDCG@k          IDCG 为 0 时返回 0
```

**汇总一律是逐样本算完再取算术平均**（`multihop` 的按信号计数项除外，见下）。
不是把全部样本的命中数和 gold 数各自累加再相除 —— 那是 micro 平均，
会让 gold 多的样本权重更大。

### full_coverage 为什么比 Recall 均值更贴切

多跳题少一跳就答不对。平均 Recall 0.5 既可能是「一半的题 gold 全齐」，
也可能是「每道题都差一篇」—— 后者的答案质量会崩，前者不会，
但两者的 Recall 均值一模一样。`full_coverage@k` 把这两种情况分开。

它也解释了为什么 `recall@k` 高而答案 F1 低不一定矛盾：查一下同一 k 的
`full_coverage@k`，多跳题少一跳就答不全。

但**两者都满分时 F1 仍然可能很低**，那就与检索无关了 —— 是散文稀释，
见上面「为什么 F1 也低：precision 塌了」。冒烟实测过这一组：`full_coverage@10`
为 1.000、三条答案全部实质正确，F1 仍只有 0.05–0.14，EM 全 0。
先排除这一种再去怀疑检索。

### 未反查到的 page 要占住名次

`ranked_doc_ids` 把 `sourcePageId` 按序反查成 `doc_id`，反查不到的**不丢弃**，
换成一个不可能等于任何 gold 的占位键 `__unmapped__:{page_id}` 继续占住排名位。

直接丢掉它们会让后面的结果整体前移，把所有对排名敏感的指标（MRR、nDCG）算高。
这些 page 由 `unmapped_page_ids` 单独统计，非空说明库里有本次子集之外的页，
`report.md` 会单独提示。同一个 page 重复出现只占一个名次。

### 没有 gold 就拒绝计算

`recall_at_k` 和 `ndcg_at_k` 在 gold 为空时**抛 ValueError**，分母无定义。
数据集依赖由 `DataDependency` 和指标 registry 校验。narrativeqa 没有 `gold_docs`，
`evaluate.py` 省略相应指标，并记录 `omitted_metrics` 和 `omission_reason`，避免把未定义值记为零分。

## 引用归因

`citations` 是 `resolveAnswerCitations` 从 `retrievedSources` 收窄成
「答案真的引了」且「有证据支撑」的部分。两个集合的**差集**才是有意思的量。

| 指标 | 定义 |
| --- | --- |
| `citation_precision` | 被引文档里是 gold 的比例 |
| `citation_recall` | gold 里被引到的比例 |
| `citation_count` / `retrieved_count` | 去重后的篇数 |
| `truncation_loss` | 被检索到但没进答案引用的文档数 |
| `truncated_gold` | 其中本来是 gold 的篇数 |
| `evidence_verifiable_rate` | `excerpts` 非空的引用占比 |
| `evidence_entries` | `citationEvidence` 条目数 |

`truncated_gold` 是这组里最该看的一个：它大于 0 说明**检索找到了 gold，
是引用过滤把它丢了**。这和「检索没找到」要调的地方完全不同 ——
一个调引用阈值，一个调召回。分不开这两者，就只能盲调。

`evidence_verifiable_rate` 回答「这条引用能不能被核验」。引用没有 excerpts
就是一个无法追溯到原文的断言。

## 多跳专项

这组是通用 RAG 基准测不出来的东西。`snippets[].retrievalReasons` 暴露了每个
snippet 是哪个信号产出的，图扩展的净贡献因此能直接量化。已知信号取值：
`semantic`、`lexical`、`exact-title`、`graph-neighbor`、`sidecar-prefiltered`。

snippet 对应哪些页走 `sourceWindows[].sourcePageId`，这是把「检索原因」和
「是否命中 gold」关联起来的唯一路径。

| 指标 | 定义 |
| --- | --- |
| `snippet_count` | snippet 总数 |
| `graph_neighbor_snippets` | 挂了 `graph-neighbor` 的 snippet 数 |
| `graph_neighbor_share` | 上者 / `snippet_count` |
| `graph_neighbor_gold_snippets` | 图扩展 snippet 里命中 gold 的条数 |
| `graph_neighbor_precision` | `graph_neighbor_gold_snippets / graph_neighbor_snippets` |
| `graph_exclusive_gold_count` | **只有**靠图扩展才拿到的 gold 篇数 |
| `graph_exclusive_gold_share` | 上者 / gold 总数 |
| `reason_counts` | 各信号产出的 snippet 数，dict |
| `reason_gold_counts` | 各信号产出且命中 gold 的 snippet 数，dict |
| `reason_gold_doc_counts` | 各信号命中的 gold **文档**数（去重后），dict |

`graph_exclusive_gold_*` 是这组的核心：它算的是 `图扩展找到的 gold − 其他信号
也找到的 gold`。两个信号都找到的那些**不算**图的净增量 —— 语义召回本来就能找到，
图只是重复了一遍。这个差集才是 graph edge 的增量价值。

一个 snippet 可能同时挂多个 `retrievalReasons`，按 set 去重后各自记一次。

汇总时前 7 个数值键取算术平均，但**按信号的计数项先累加再算比率**，
并且换成三个只在汇总层出现的键：

| 汇总键 | 定义 |
| --- | --- |
| `reason_totals` | `reason_counts` 跨样本累加 |
| `reason_gold_totals` | `reason_gold_counts` 跨样本累加 |
| `reason_gold_rate` | `reason_gold_totals[信号] / reason_totals[信号]` |

先算每样本比率再取均值会给样本量小的信号过高权重，所以这三个键走累加。
`reason_gold_doc_counts` 不进汇总（跨样本的 gold 文档集合没有意义）。

## 两份检索表：全样本 vs knowledge-only

`no_match` 和 `general` 两种 `answerMode` 会**无条件**返回空的
`retrievedSources`，不管检索实际找到了什么。这些行的检索分数天然是 0。

所以每个检索指标都出两份：

- `retrieval` —— 全样本。这份把「生成端拒答」也算进了检索指标里。
- `retrieval_knowledge_only` —— 只算 `answerMode == "knowledge"` 的切片。

**两份的差值就是生成端拒答的规模**，不是检索失败的规模。答案 F1 同样出
`f1_knowledge_only` 第二份。`knowledge_answer_count`
记参与 knowledge-only 那份的样本数，差得太多说明这份的样本量已经不够看。

## 分层报告

按数据集各自的 metadata 字段切，出随难度的衰减曲线：

| 数据集 | 分层键 | 分层 |
| --- | --- | --- |
| hotpotqa | `type` | `bridge` / `comparison` |
| 2wikimultihopqa | `type` | `compositional` / `comparison` / `bridge_comparison` / `inference` |
| musique | `hop_count` | 2 / 3 / 4 |
| narrativeqa | `kind` | `movie` / `gutenberg` |

只有 musique 有真正的跳数，其余三组只能按题型或文档类型切。
每个分层报 `count`、检索、多跳、答案 F1，以及 `knowledge_answer_share`。

`report.md` 的分层表固定用 `recall@10` 和 `full_coverage@10` 两列，
完整数据在 `metrics.json` 里。

## 失败行照样进统计

HTTP 失败的行**不跳过**，F1 记 0 照样参与统计，因为失败率本身是结果的一部分。

汇总层另有几个运行状态字段，不是质量指标但读数时要先看：

| 字段 | 含义 |
| --- | --- |
| `samples_in_subset` | 子集里的样本数，即分母应该是多少 |
| `responses_evaluated` | 实际参与计算的响应数 |
| `missing_responses` | 子集里有、响应文件里没有的 `sample_id` 列表 |
| `http_failures` | 非 2xx 的行数 |
| `latency_ms_mean` | 平均延迟，缺失值按 0 计入 |

`responses_evaluated` 小于 `samples_in_subset` 说明查询没跑完，
此时所有指标的分母都偏小，先补跑再读数。

`evaluate.py` 有两道一致性闸门，任一不过直接报错而不是出一份错的报告：

- 响应文件里的 `sample_id` 不在子集里 → 两份产物来自不同的 run。
- ID 对得上但 `question` 文本不一致 → 产物来自不同的数据快照。

`sample_id` 在响应文件里重复也报错。子集里有但响应文件里缺的样本记入
`missing_responses`。

某个数据集没跑过查询会被**跳过**（打印 skip，不算失败），
其余数据集正常出报告。

## 三段归因（可选）

`retrievalDiagnostics` 被 controller 从 HTTP 响应里解构排除了，只写进
`knowledge_query_audit.metadata`。所以想知道召回是**在哪一段**丢的，
只有走审计表这一条路：

```bash
uv run python -m akasha_benchmark.audit_join --query-label run001-query
```

逐样本三个原始计数直接取自 `metadata`，缺失按 0：
`candidate_chunk_count`、`ranked_candidate_count`、`filtered_chunk_count`，
外加 `access_policy_fallback_used`（bool）和 `gold_hit`
（该样本的 `hit@10` 是否为真，用来把分段损失和最终命中关联起来）。

由它们导出三段：

| 分段 | 判据 | 含义 |
| --- | --- | --- |
| 召回上限 | `recall_ceiling_miss`：`candidate_chunk_count == 0` | 候选集本身是空的，后面任何环节都救不回来 |
| 排序损失 | `ranking_loss`：`max(candidate − ranked, 0)` | 进了候选集，但没通过 RRF / 阈值 |
| 授权损失 | `authorization_loss`：`filtered_chunk_count` | 排序过了，被第三道授权闸门丢掉 |

汇总层（`overall` 与每个 `by_retrieval_mode`）：

| 汇总键 | 定义 |
| --- | --- |
| `count` | 该组的样本数 |
| `recall_ceiling_miss_rate` | `recall_ceiling_miss` 为真的比例 |
| `mean_candidate_chunks` | `candidate_chunk_count` 均值 |
| `mean_ranking_loss` | `ranking_loss` 均值 |
| `mean_authorization_loss` | `authorization_loss` 均值 |
| `access_policy_fallback_rate` | 用了授权兜底的比例 |
| `gold_hit_rate` | `gold_hit` 为真的比例 |

没有这个拆分，你只知道 `recall@10 = 0.6`，却不知道该调哪个旋钮。
`recall_ceiling_miss_rate` 高就是召回段的问题，它低而 `mean_ranking_loss` 大
就是排序段的问题，`mean_authorization_loss` 大则与检索质量无关。

连接键是 `sha256:<hex>`，**带 `sha256:` 前缀**。裸的十六进制值一行都匹配不上。
查询按其自身的运行时间窗卡范围，否则上一次运行的审计行会混进来。

连不上的样本计入 `unmatched_samples`（时间窗外、或那条 query 没留下审计记录）。
重复的 question 文本会让那一行没法安全 join，这些行**直接排除**而不是随便连一个，
计入 `excluded_ambiguous_question_text`：在锁定快照上是 musique 1 行、其余 0 行
（见 [normalized_datasets.md](normalized_datasets.md) 的 `unique_question_texts`）。
同一条 query 重试过会有多行审计记录，取最后一次。

结果按 `retrievalMode` 切开报告。`high_completeness` 和
`high_completeness_fallback` 是两种不同的召回口径，混在一起平均会把问题盖掉。

这个模块**可选**，需要 `psycopg` 和 `database_url`。缺依赖或缺配置时给明确提示
并返回 1，评测的其余指标不受影响。

## 测试

```bash
uv run pytest tests/test_metrics.py -q
```

[test_metrics.py](../tests/test_metrics.py) 里的期望值都是**手算**的，
不是「跑一遍看着差不多」。指标算错了不会报错，只会给出一个看着合理的假结果，
所以这里必须拿独立算出来的数字对。覆盖 k 的边界、nDCG 的理想排序、
未反查 page 的名次占位、gold 为空时报错、SQuAD 退化情况、
图扩展净增量、按信号汇总时的累加口径。
