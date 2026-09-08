# 数据集字段说明

四组多跳/长文档问答数据集，取自 [osunlp/HippoRAG_2](https://huggingface.co/datasets/osunlp/HippoRAG_2)，
由 [download_dataset.py](download_dataset.py) 下载到 [dataset/](dataset/)。

每组两个文件，职责分开：

- `<name>.json` —— **QA 文件**。问题、标准答案、以及标注好的 gold evidence。用来出题和打分。
- `<name>_corpus.json` —— **检索语料**。扁平的文档列表，喂给检索器建索引。

下表数字均为实际读文件统计：

| 数据集 | QA 行数 | corpus 行数 | 题型 | evidence 标注 |
| --- | --- | --- | --- | --- |
| hotpotqa | 1000 | 9811 | 2 跳 | 句子级 |
| 2wikimultihopqa | 1000 | 6119 | 2–4 跳 | 句子级 + 三元组 |
| musique | 1000 | 11656 | 2–4 跳 | 段落级 |
| narrativeqa | 293 | 4111 | 长文档 | 无 |

## corpus 文件

三个字段，但 `idx` 并不统一，写加载器时要注意：

| 字段 | 类型 | 作用 |
| --- | --- | --- |
| `title` | str | 文档标题。hotpotqa / 2wiki / musique 里它同时是 gold evidence 的关联键 |
| `text` | str | 正文，检索的实际内容 |
| `idx` | 见下 | 行标识，**只有两个数据集有** |

- hotpotqa：`idx` 是 int，0 起自增。
- narrativeqa：`idx` 是 str，格式 `<document.id>_<chunk 序号>`，例如
  `4b30ab1c49b62dc59b9773954958d9ac6807a865_0`。4111 个值全局唯一。
- 2wikimultihopqa / musique：**没有 `idx`**，只有 `title` + `text`。

`title` 能否当唯一键也分数据集：hotpotqa（9811）和 2wiki（6119）的 title 全局唯一；
musique 有 1818 行 title 重复（同名文档的不同段落）；narrativeqa 的 4111 个 chunk
只有 10 个不同 title，因为它是按文档切块的，title 是书/剧本名。

## hotpotqa

维基百科 2 跳问答。全部为 `level: hard`。

| 字段 | 类型 | 作用 |
| --- | --- | --- |
| `_id` | str | 24 位十六进制，行唯一标识 |
| `question` | str | 问题 |
| `answer` | str | 标准答案，EM/F1 的比对对象 |
| `context` | list | 10 个 `[title, [句子, ...]]`。每题的候选文档，2 个 gold + 8 个干扰 |
| `supporting_facts` | list | gold evidence，元素为 `[title, 句子下标]`，下标指向 `context` 里对应 title 的句子列表 |
| `type` | str | `bridge`(811) / `comparison`(189)，桥接推理还是比较 |
| `level` | str | 难度，本集恒为 `hard` |

## 2wikimultihopqa

维基百科 + Wikidata，2–4 跳。结构与 hotpotqa 基本对齐，多了知识图谱侧的标注。

| 字段 | 类型 | 作用 |
| --- | --- | --- |
| `_id` | str | 32 位十六进制，行唯一标识 |
| `question` | str | 问题 |
| `answer` | str | 标准答案 |
| `context` | list | 10 个 `[title, [句子, ...]]`，同 hotpotqa |
| `supporting_facts` | list | gold evidence，`[title, 句子下标]` |
| `evidences` | list | 三元组 `[主语, 关系, 宾语]`，如 `["Lothair II", "mother", "Ermengarde of Tours"]`。推理链的符号化表示 |
| `evidences_id` | list | 同上，但换成 Wikidata QID，如 `["Q298945", "mother", "Q235653"]` |
| `entity_ids` | str | 下划线连接的 QID，如 `Q298945_Q235653` |
| `type` | str | `compositional`(413) / `comparison`(244) / `bridge_comparison`(235) / `inference`(108) |
| `answer_id` | str \| null | 答案实体的 QID，163 行为 `null`（答案是日期、数字等非实体时） |

## musique

2–4 跳，带显式子问题分解。evidence 是段落级而非句子级。

| 字段 | 类型 | 作用 |
| --- | --- | --- |
| `id` | str | 行标识，前缀即跳数：`2hop`(518) / `3hop1`(243) / `3hop2`(73) / `4hop1`(108) / `4hop2`(27) / `4hop3`(31) |
| `question` | str | 问题 |
| `answer` | str | 标准答案 |
| `answer_aliases` | list | 答案别名，276 行非空。打 EM 时应并入候选，命中任一即算对 |
| `answerable` | bool | 是否可答，本集全为 `true` |
| `paragraphs` | list | 20 个候选段落，每个含 `idx`(int) / `title` / `paragraph_text` / `is_supporting`(bool) |
| `question_decomposition` | list | 推理链，每步含 `id` / `question` / `answer` / `paragraph_support_idx`（指向 `paragraphs[].idx`） |

gold evidence 取 `paragraphs` 里 `is_supporting == true` 的项，每题 2–4 个
（2 个:518 行，3 个:316 行，4 个:166 行）。

## narrativeqa

电影剧本和 Gutenberg 图书的长文档问答。**没有 evidence 标注**，只能评答案质量。

| 字段 | 类型 | 作用 |
| --- | --- | --- |
| `question` | str | 问题 |
| `answer` | list[str] | **2 个**人工参考答案，全部 293 行都是 2 个。评分需对多参考取最优 |
| `document` | dict | 源文档，见下 |

`document` 的字段：

| 字段 | 作用 |
| --- | --- |
| `id` | 40 位 SHA1，文档标识。注意只有 **10 个**唯一值 |
| `kind` | `movie`(204) / `gutenberg`(89) |
| `url` | 原文地址 |
| `file_size` / `word_count` | 原文体积 |
| `start` / `end` | 正文首尾片段，用于定位截断边界 |
| `summary` | 含 `text` / `tokens` / `url` / `title`，人工摘要 |
| `text` | **整篇原文全文**，166KB–505KB |

## 几个实测注意点

**QA 与 corpus 的关联**已逐行验证：

- hotpotqa：2468 条 gold evidence 的 title 全部命中 corpus，句子下标无越界。
- musique：19990 个内联段落（含 2648 条 supporting）的 `(title, text)` 全部能在 corpus 中找到。
- narrativeqa：10 个 `document.id` 全部对应上 corpus 的 `idx` 前缀。
- 2wiki：有两处小瑕疵 —— 9/2471 条 gold evidence 的句子下标越界或 title 不在
  该行 `context` 里；6120 个 context title 中有 1 个不在 corpus。写适配器时别假设 100%。

**句子拼接方式不一致。** 把 `context` 的句子列表还原成 corpus 里的 `text` 时，
hotpotqa 用空串 `""` 拼，2wiki 用空格 `" "` 拼。两边 `text` 都以 title 开头。
这意味着「gold evidence 字符串精确命中 corpus」的比例不会是 100%，
evidence recall 的比对口径需要容忍这个差异。

**narrativeqa 的 `document.id` 不能当行标识。** 293 个问题只有 10 个文档，
它是文档级 ID。需要行身份时用行号。

**narrativeqa QA 文件 94MB**，因为每行都内联了整篇 `document.text`（约 210KB），
10 篇原文被重复了 293 次。corpus 反而只有 2.9MB。按行流式处理或只取需要的字段，
不要无脑全量 load 进内存。

## 校验

```bash
uv run python download_dataset.py --check
```

逐个文件确认存在、能解析成 JSON，并打印体积和记录数。



