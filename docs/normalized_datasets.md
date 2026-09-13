# 规范化数据字段说明

[datasets.md](datasets.md) 是四组数据集**下载下来的原始字段**，schema 互不相同；本文是被适配器抹平之后的**统一 schema**，
字段与指标的关联见 [metrics.md](metrics.md)。

```bash
uv run python -m akasha_benchmark.normalize                      # 四组全做
uv run python -m akasha_benchmark.normalize --dataset hotpotqa   # 只做一组
```

归一化默认写入数据库。加 `--export` 才会在 `data/normalized/{dataset}/` 导出三个文件：

| 文件 | 内容 |
| --- | --- |
| `samples.jsonl` | 每行一个 `CanonicalSample`，即一道题 |
| `corpus.jsonl` | 每行一个 `CorpusDoc`，即一篇待检索文档 |
| `manifest.json` | 源文件 sha256、条数、身份规则、provides、实测分布 |

体积实测：

| 数据集 | samples 行 | samples 体积 | corpus 行 | corpus 体积 |
| --- | --- | --- | --- | --- |
| hotpotqa | 1000 | 385 KB | 9811 | 5.8 MB |
| 2wikimultihopqa | 1000 | 431 KB | 6119 | 2.9 MB |
| musique | 1000 | 469 KB | 11656 | 6.1 MB |
| narrativeqa | 293 | 113 KB | 4111 | 2.7 MB |

注意 corpus 的块
**不是** `document.text` 的切片，见下方 [narrativeqa 的块经过加工](#narrativeqa-的块经过加工)。

## samples.jsonl

模型定义在 [models.py](../src/akasha_benchmark/datasets/models.py)，
`extra="forbid"` + `frozen=True`：上游多一个字段就在构造处报错，
校验通过之后下游谁也改不动它。

| 字段 | 类型 | 作用 |
| --- | --- | --- |
| `dataset` | str | 数据集名，四个取值之一。跨数据集汇总时用它分组 |
| `sample_id` | str | **全局唯一**行标识，格式 `{dataset}:{dataset_sample_id}` |
| `dataset_sample_id` | str | 数据集内的原生 ID，回溯原始文件时用 |
| `question` | str | 问题原文，一字不改地送给系统 |
| `answers` | list[str] | 参考答案，**至少一个**。答案 F1 对多参考取 max |
| `gold_doc_ids` | list[str] | gold 文档的 `doc_id`，指向同目录 `corpus.jsonl`。算 evidence recall 的分母 |
| `metadata` | dict | 分层报告用的附加信息，**逐数据集不同**，见下 |

以 2wiki 第一条为例：

```json
{
  "dataset": "2wikimultihopqa",
  "sample_id": "2wikimultihopqa:83bf3b5a0bd911eba7f7acde48001122",
  "dataset_sample_id": "83bf3b5a0bd911eba7f7acde48001122",
  "question": "When did Lothair Ii's mother die?",
  "answers": ["20 March 851"],
  "gold_doc_ids": ["4", "5"],
  "metadata": {
    "type": "compositional",
    "gold_count": 2,
    "supporting_fact_count": 2,
    "evidence_triple_count": 2,
    "answer_id": null
  }
}
```

对照原始行：

- `_id` 加了 `2wikimultihopqa:` 前缀成为 `sample_id`，同时原值留在 `dataset_sample_id`。
  加前缀是因为四组数据放一起汇总时，裸 ID 不保证不撞。
- 单个 `answer` 字串被包成单元素列表。四个数据集统一成列表，
  打分侧不必再分「一个参考」和「多个参考」两条路。
- `supporting_facts` 的 `[title, 句子下标]` **降级成了文档级**：
  title 去重后经 corpus 反查成 `doc_id`。句子下标丢弃 ——
  Akasha 返回的是文档，句子级 recall 没法算。
- `context` 那 10 篇候选**整个丢掉**了。检索从全量 corpus 做，
  不喂预先筛好的 10 篇候选，否则测不出检索能力。

### gold_doc_ids 是文档级的

四个数据集的原生 evidence 粒度不一样（2wiki/hotpotqa 句子级、musique 段落级、
narrativeqa 无标注），统一降到文档级。副作用是 `gold_count` 通常小于原生
evidence 条数，两个数都留在 `metadata` 里，差值即被折叠掉的冗余：


不去重会把 recall 的分母算大，所以去重是必须的，见
[common.py](../src/akasha_benchmark/datasets/common.py)。

降级到文档级不是选择，而是被 Akasha 的返回粒度决定的：一个 `doc_id` 上传成一个页面、
拿回一个 `page_id`，响应里只有 `sourcePageId`，句子级 recall 没有可比对的东西。

### metadata 各数据集字段

同一个 `metadata` 键在不同数据集里不一定存在，读之前先看 `dataset`。

**hotpotqa**

| 键 | 值 |
| --- | --- |
| `type` | `bridge` 811 / `comparison` 189 |
| `level` | 恒为 `hard` |
| `gold_count` | 恒为 2 |
| `supporting_fact_count` | 2: 650, 3: 253, 4: 81, 5: 13, 6: 1, 7: 2 |

**2wikimultihopqa**

| 键 | 值 |
| --- | --- |
| `type` | `compositional` 413 / `comparison` 244 / `bridge_comparison` 235 / `inference` 108 |
| `gold_count` | 2: 765, 4: 235 |
| `supporting_fact_count` | 2: 765, 4: 234, 5: 1 |
| `evidence_triple_count` | 推理链三元组条数，2: 749, 3: 3, 4: 241, 5: 6, 6: 1 |
| `answer_id` | 答案实体 QID，163 行为 `null`（答案是日期/数字时） |

`evidences` 三元组本身**没有**进规范化数据，只留了条数。它是知识图谱侧的符号表示，
不是文档，当检索 ground truth 会算出一个看着合理但没有意义的数。

**musique**

| 键 | 值 |
| --- | --- |
| `hop_prefix` | `2hop` 518 / `3hop1` 243 / `4hop1` 108 / `3hop2` 73 / `4hop3` 31 / `4hop2` 27 |
| `hop_count` | 2: 518, 3: 316, 4: 166。从前缀首字符取的整数 |
| `gold_count` | 同 `hop_count` |
| `answerable` | 全为 `true` |
| `alias_count` | 并入 `answers` 的别名个数，0: 724，最多 6 |
| `decomposition_steps` | 原生 `question_decomposition` 步数，同 `hop_count` |
| `gold_with_ambiguous_title` | 该行有几条 gold 的 title 在 corpus 里对应多行。0: 452，最多 4 |

`gold_with_ambiguous_title` 存在的意义是留证据：548 行至少有一条 gold 的 title 有歧义，
所以 musique 的 gold 必须按 `(title, paragraph_text)` 定位，只按 title 会命中错行。

**narrativeqa**

| 键 | 值 |
| --- | --- |
| `document_id` | 40 位 SHA1，只有 10 个唯一值。抽子集靠它整篇整篇地抽 |
| `kind` | `movie` 204 / `gutenberg` 89 |
| `reference_count` | 1: 20, 2: 273 |
| `summary_title` | 书名/片名，如 `All About Steve`。只作可读标签 |

原始 `document.summary` 里的 `text` / `tokens` / `url` **都不进规范化数据**，
只取了 `title`。corpus 是从 `document.text` 全文切的块，摘要全程没有消费方 ——
它是原始 NarrativeQA「摘要设定」的遗留，本 benchmark 走的是全文设定。

但摘要的存在解释了一件事：参考答案是标注者**只看摘要**写出来的，措辞与全文不同，
narrativeqa 的答案分数因此有个天然上限，见 [metrics.md](metrics.md)。

`reference_count` 有 20 行是 1 而不是 2，与 [datasets.md](datasets.md) 讲的
「原始数据全部 293 行都是 2 个参考」并不矛盾：原始确实都是 2 个，但其中 20 行
两个参考**逐字节相同**（如 `["Hartman Hughes", "Hartman Hughes"]`），
适配器保序去重后剩 1 个。对 max 取值的答案 F1 没有影响。

## corpus.jsonl

三个字段，四组数据集完全一致：

| 字段 | 类型 | 作用 |
| --- | --- | --- |
| `doc_id` | str | 行标识，`gold_doc_ids` 和检索结果都用它对齐 |
| `title` | str | 文档标题 |
| `text` | str | 正文 |

`doc_id` 一律是字符串，但**怎么来的分数据集**，规则登记在
[models.py](../src/akasha_benchmark/datasets/models.py) 的 `CORPUS_ID_RULES`：

| 数据集 | 规则 | 形如 |
| --- | --- | --- |
| hotpotqa | `native_id`，原生 `idx` 转 str | `"0"`, `"1"` |
| 2wikimultihopqa | `row_idx`，corpus 数组行号 | `"0"`, `"1"` |
| musique | `row_idx` | `"0"`, `"1"` |
| narrativeqa | `native_id`，原生 str idx | `"4b30ab…865_0"` |

2wiki 和 musique 用行号是因为它们的原始 corpus 根本没有 `idx`。
这条前提写进了断言：这两组的 corpus 行里**突然出现** `idx` 会直接报错，
因为那意味着数据换版了，行号身份不再可信，见
[corpus.py:111-120](../src/akasha_benchmark/datasets/corpus.py#L111-L120)。

`sample_id` 侧同理，规则在 `SAMPLE_ID_RULES`：hotpotqa / 2wiki 用原生 `_id`，
musique 用原生 `id`，narrativeqa 没有原生 ID，用它在**全量**文件里的行号。
narrativeqa 的 `document.id` 是文档级的（293 个问题只有 10 个值），当行身份会撞一片。

### 语料不去重

`corpus.jsonl` 与原始 corpus 行数一一对应，不做任何去重。musique 有 2465 行
落在重复 title 组里，但它们是同名文档的**不同段落**，去重会直接丢 gold。
narrativeqa 的 4111 行全部在重复 title 组里，因为 10 篇文档切块后共享书名。

manifest 的 `corpus_dedup_stats` 记着去重前后的数，`dedup_applied` 恒为 `false`，
这个选择随时可查：

| 数据集 | rows | unique_titles | unique (title,text) | 重复组内行数 |
| --- | --- | --- | --- | --- |
| hotpotqa | 9811 | 9811 | 9811 | 0 |
| 2wikimultihopqa | 6119 | 6119 | 6119 | 0 |
| musique | 11656 | 9838 | 11656 | 2465 |
| narrativeqa | 4111 | 10 | 4111 | 4111 |

四组的 `(title, text)` 都是唯一的。这是底线 —— 如果连这个都重复，
就没有任何键能定位到行，此时建索引直接报错而不是挑一个凑合。

### narrativeqa 数据集中的语料

10 篇文档切成 4111 块，`doc_id` 形如 `{document_id}_{chunk_seq}`，`chunk_seq`
严格 `0..n-1` 连续，同一篇的块共享 title。块长中位数 556 字符（91–936），
相邻块**没有 overlap**，
是硬切不是滑窗。

语料中的这些块**不是** QA 文件里 `document.text` 的切片：`document.text` 是带
`<html><head>` 的整页，含导航栏和 meta 标签，块里只有正文。体积因此对不上，

corpus 的 `text` 只用于
`(title, text)` 身份去重和 `to_markdown()` 渲染导入，没有拿它跟别的文本做字符串比对；narrativeqa 的确定性答案指标包括 EM 和 F1，比的是模型答案和 QA 文件里的参考答案，两边都不碰 corpus。


## manifest.json

导出清单记录数据来源与标注能力，字段由 `normalize.export_dataset` 生成：

| 字段 | 作用 |
| --- | --- |
| `stage` / `dataset` / `note` | 阶段、数据集和导出说明 |
| `adapter` / `adapter_version` | 适配器及版本 |
| `exported_at` | 导出时刻，UTC |
| `provides` | 已有标注：`reference_answers`，以及有 gold 时的 `gold_docs` |
| `identity_rules` | 样本和语料 ID 规则 |
| `sources` | QA、corpus 的源路径、sha256 和行数 |
| `corpus_dedup_stats` / `dedup_applied` | 去重统计；不执行去重 |
| `gold_count_distribution` | 去重后的 gold 篇数分布 |
| `unique_question_texts` | 不重复的问题文本数 |

前三组数据提供参考答案和 gold 文档；narrativeqa 只提供参考答案。
指标在 registry 中声明 `requires`，评测对缺少依赖的数据集省略相应指标，
并在 `omitted_metrics` 中记录原因。底层显式依赖校验失败时抛 `DependencyError`。

### unique_question_texts 为什么要记

musique 是 999 而不是 1000，有一行问题文本与另一行重复。审计表按
`sha256(query)` join，重复问题的那一行连不上。其余三组都是全数唯一。

## 校验

```bash
uv run python -m akasha_benchmark.normalize      # 生成
uv run python scripts/validate_datasets.py       # 逐行过全量
```

校验是**逐行过全量数据**的，不是只看 row 0：每行能否通过适配器、
`dataset_sample_id` 有无缺失或重复、gold 条数分布、声明了 `gold_docs`
却抽不出 gold 的行、gold doc_id 是否都在 corpus 里、重复 question 计数、
corpus `(title, text)` 唯一性。四组应全部通过，gold 解析率 100%，无重复 ID。

它**刻意不复用** `normalize.py` 的结果，而是从原始文件重新推导一遍再逐行比对，
并重算源文件 sha256 与数据库记录对账。这样写入侧的 bug 会表现为「对不上」，
而不是被自己的输出确认为正确。

写文件走「临时文件 + `os.replace`」原子替换，中断不会留下截断的 jsonl。
