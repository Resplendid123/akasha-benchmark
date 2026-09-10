# 数据集字段说明

四组多跳/长文档问答数据集，取自 [osunlp/HippoRAG_2](https://huggingface.co/datasets/osunlp/HippoRAG_2)，
由 [download_datasets.py](../scripts/download_datasets.py) 下载到 [dataset/](../dataset/)。

每组两个文件，职责分开：

- `<name>.json` —— **QA 文件**。
- `<name>_corpus.json` —— **检索语料**。

本文只讲原始字段。抹平之后的统一 schema 见 [normalized_datasets.md](normalized_datasets.md)。

| 数据集 | QA 行数 | corpus 行数 | 题型 | evidence 标注 |
| --- | --- | --- | --- | --- |
| hotpotqa | 1000 | 9811 | 2 跳 | 句子级 |
| 2wikimultihopqa | 1000 | 6119 | 2–4 跳 | 句子级 + 三元组 |
| musique | 1000 | 11656 | 2–4 跳 | 段落级 |
| narrativeqa | 293 | 4111 | 长文档 | 无 |

## corpus 文件

| 字段 | 类型 | 作用 |
| --- | --- | --- |
| `title` | str | 文档标题。hotpotqa / 2wiki / musique 里它同时是 gold evidence 的关联键 |
| `text` | str | 正文，检索的实际内容 |
| `idx` | 见下 | 行标识，**只有两个数据集有** |

- hotpotqa：`idx` 是 int，0 起自增。
- narrativeqa：`idx` 是 str，格式 `<document.id>_<chunk 序号>`，例如
  `4b30ab1c49b62dc59b9773954958d9ac6807a865_0`。4111 个值全局唯一。
- 2wikimultihopqa / musique：**没有 `idx`**，只有 `title` + `text`。

## 2wikimultihopqa

维基百科 + Wikidata，2–4 跳。结构与 hotpotqa 基本对齐，多了知识图谱侧的标注。

| 字段 | 类型 | 作用 |
| --- | --- | --- |
| `_id` | str | 32 位十六进制，行唯一标识 |
| `question` | str | 问题 |
| `answer` | str | 标准答案 |
| `context` | list | 10 个 `[title, [句子, ...]]`，title同时是supporting_facts和corpus中的title关联键；每题的候选文档中 2 个 gold + 8 个干扰  |
| `supporting_facts` | list | gold evidence，`[title, 句子下标]` ，title再context中，句子下标是对应corpus中的第几个片段（细粒度）|
| `evidences` | list | 三元组 `[主语, 关系, 宾语]`，如 `["Lothair II", "mother", "Ermengarde of Tours"]`。推理链的符号化表示 |
| `evidences_id` | list | 同上，但换成 Wikidata QID，如 `["Q298945", "mother", "Q235653"]` |
| `entity_ids` | str | 上述中通过下划线连接QID的关系链表示，如 `Q298945_Q235653` |
| `type` | str | `compositional`(413) - 组合推理（评估多源汇聚） / `comparison`(244) - 比较推理（评估领域内优劣）/ `bridge_comparison`(235) - 桥接推理（评估跨领域类比） / `inference`(108) - 推断推理（评估因果链条） |
| `answer_id` | str \| null | 答案对应实体的 QID， `null`（答案是日期、数字等非实体时） |

## hotpotqa

维基百科 2 跳问答。全部为 `level: hard`。

| 字段 | 类型 | 作用 |
| --- | --- | --- |
| `_id` | str | 24 位十六进制，行唯一标识 |
| `question` | str | 问题 |
| `answer` | str | 标准答案 |
| `context` | list | 10 个 `[title, [句子, ...]]`。title同时是supporting_facts和corpus中的title关联键；每题的候选文档中 2 个 gold + 8 个干扰 |
| `supporting_facts` | list | gold evidence，元素为 `[title, 句子下标]`，下标指向 `context` 里对应 title 的句子列表 |
| `type` | str | `bridge`(811) - 桥接推理（评估跨领域类比）/ `comparison`(189) - 比较推理（评估领域内优劣）|
| `level` | str | 难度，本集恒为 `hard` |

## musique

2–4 跳，带显式子问题分解。evidence 是段落级而非句子级。

| 字段 | 类型 | 作用 |
| --- | --- | --- |
| `id` | str | 行标识，前缀即跳数：`2hop`(518) / `3hop1`(243) / `3hop2`(73) / `4hop1`(108) / `4hop2`(27) / `4hop3`(31) |
| `question` | str | 问题 |
| `answer` | str | 标准答案 |
| `answer_aliases` | list | 答案别名，276 行非空。打分时应并入候选，命中任一即算对 |
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
| `answer` | list[str] | **2 个**人工参考答案。 |
| `document` | dict | 源文档，见下 |

`document` 的字段：

| 字段 | 作用 |
| --- | --- |
| `id` | 40 位 SHA1，文档标识。 |
| `kind` | `movie`(204) / `gutenberg`(89) |
| `url` | 原文地址 |
| `file_size` / `word_count` | 原文体积 |
| `start` / `end` | 正文首尾片段，用于定位截断边界 |
| `summary` | 含 `text` / `tokens` / `url` / `title`，人工摘要 |
| `text` | **整篇原文全文**，166KB–505KB |

## 校验

```bash
uv run python scripts/download_datasets.py --check
```

逐个文件确认存在、能解析成 JSON，并打印体积和记录数。

## 附录：各数据集首行样本

每个数据集 QA 文件的 `[0]` 号样本，字段与上文表格对应。候选段落统一裁到 3 个、过长文本截断，省略处均有标注，其余原样保留。

### hotpotqa

10 个候选段落裁到 3 个：`supporting_facts` 指向的 2 个 gold 加 1 个干扰，保持原始顺序。

```json
{
  "_id": "5abe953b5542993f32c2a170",
  "answer": "superhero roles as the Marvel Comics",
  "question": "what is one of the stars of  The Newcomers known for",
  "supporting_facts": [
    [
      "The Newcomers (film)",
      0
    ],
    [
      "Chris Evans (actor)",
      1
    ]
  ],
  "context": [
    [
      "Vaada Poda Nanbargal",
      [
        "Vaada Poda Nanbargal is a 2011 Indian Tamil-language romantic comedy film directed by Manikai.",
        " P. Arumaichandran has produced this movie under the banner 8 Point Entertainments.",
        " The film stars newcomers Nanda, Sharran Kumar and Yashika in the lead roles.",
        " The lead actor Nanda happens to be one of the strong contender of a popular television series \"Yaar Adutha Prabhu Deva\" aired on Vijay TV."
      ]
    ],
    [
      "Chris Evans (actor)",
      [
        "Christopher Robert Evans (born June 13, 1981) is an American actor and filmmaker.",
        " Evans is known for his superhero roles as the Marvel Comics characters Steve Rogers / Captain America in the Marvel Cinematic Universe and Johnny Storm / Human Torch in \"Fantastic Four\" and ."
      ]
    ],
    [
      "The Newcomers (film)",
      [
        "The Newcomers is a 2000 American family drama film directed by James Allen Bradley and starring Christopher McCoy, Kate Bosworth, Paul Dano and Chris Evans.",
        " Christopher McCoy plays Sam Docherty, a boy who moves to Vermont with his family, hoping to make a fresh start away from the city.",
        " It was filmed in Vermont, and released by Artist View Entertainment and MTI Home Video."
      ]
    ],
    // ... 其余 7 个省略，结构相同
  ],
  "type": "bridge",
  "level": "hard"
}
```

### 2wikimultihopqa

同上，10 个裁到 3 个，2 gold + 1 干扰。

```json
{
  "_id": "83bf3b5a0bd911eba7f7acde48001122",
  "type": "compositional",
  "question": "When did Lothair Ii's mother die?",
  "context": [
    [
      "Teutberga",
      [
        "Teutberga( died 11 November 875) was a queen of Lotharingia by marriage to Lothair II.",
        "She was a daughter of Bosonid Boso the Elder and sister of Hucbert, the lay- abbot of St. Maurice's Abbey."
      ]
    ],
    [
      "Lothair II",
      [
        "Lothair II (835 –) was the king of Lotharingia from 855 until his death.",
        "He was the second son of Emperor Lothair I and Ermengarde of Tours.",
        "He was married to Teutberga (died 875), daughter of Boso the Elder."
      ]
    ],
    [
      "Ermengarde of Tours",
      [
        "Ermengarde of Tours (d. 20 March 851) was the daughter of Hugh of Tours, a member of the Etichonen family.",
        "In October 821 in Thionville, she married the Carolingian Emperor Lothair I of the Franks (795–855).",
        "In 849, two years before her death, she made a donation to the abbey Erstein in the Elsass, in which she is buried.",
        "Lothair and Ermengarde had eight children:"
      ]
    ],
    // ... 其余 7 个省略，结构相同
  ],
  "entity_ids": "Q298945_Q235653",
  "supporting_facts": [
    [
      "Lothair II",
      1
    ],
    [
      "Ermengarde of Tours",
      0
    ]
  ],
  "evidences": [
    [
      "Lothair II",
      "mother",
      "Ermengarde of Tours"
    ],
    [
      "Ermengarde of Tours",
      "date of death",
      "20 March 851"
    ]
  ],
  "answer": "20 March 851",
  "evidences_id": [
    [
      "Q298945",
      "mother",
      "Q235653"
    ],
    [
      "Q235653",
      "date of death",
      "date_information"
    ]
  ],
  "answer_id": null
}
```

### musique

20 个候选段落裁到 3 个，含 1 个 `is_supporting: true`。本题 gold 共 2 个，另一个在省略部分。

```json
{
  "id": "2hop__13548_13529",
  "paragraphs": [
    {
      "idx": 0,
      "title": "Lionel Messi",
      "paragraph_text": "After a year at Barcelona's youth academy, La Masia, Messi was finally enrolled in the Royal Spanish Football Federation (RFEF) in February 2002. Now playing in all competitions, he befriended his teammates, among whom were Cesc Fàbregas and Gerard Piqué. After completing his growth hormone treatment aged 14, Messi became an integral part of the ``Baby Dream Team '', Barcelona's greatest - ever youth side. During his first full season (2002 -- 03), he was top scorer with 36 goals in 30 games for the Cadetes A, who won an unprecedented treble of the league and both the Spanish and Catalan cups. The Copa Catalunya final, a 4 -- 1 victory over Espanyol, became known in club lore as the partido de la máscara, the final of the mask. A week after suffering a broken cheekbone during a league match, Messi was allowed to start the game on the condition that he wear a plastic protector; soon hindered by the mask, he took it off and scored two goals in 10 minutes before his substitution. At the close of the season, he received an offer to join Arsenal, his first from a foreign club, but while Fàbregas and Piqué soon left for England, he chose to remain in Barcelona.",
      "is_supporting": false
    },
    {
      "idx": 1,
      "title": "FC Barcelona",
      "paragraph_text": "Despite being the favourites and starting strongly, Barcelona finished the 2006–07 season without trophies. A pre-season US tour was later blamed for a string of injuries to key players, including leading scorer Eto'o and rising star Lionel Messi. There was open feuding as Eto'o publicly criticized coach Frank Rijkaard and Ronaldinho. Ronaldinho also admitted that a lack of fitness affected his form. In La Liga, Barcelona were in first place for much of the season, but inconsistency in the New Year saw Real Madrid overtake them to become champions. Barcelona advanced to the semi-finals of the Copa del Rey, winning the first leg against Getafe 5–2, with a goal from Messi bringing comparison to Diego Maradona's goal of the century, but then lost the second leg 4–0. They took part in the 2006 FIFA Club World Cup, but were beaten by a late goal in the final against Brazilian side Internacional. In the Champions League, Barcelona were knocked out of the competition in the last 16 by eventual runners-up Liverpool on away goals.",
      "is_supporting": true
    },
    {
      "idx": 2,
      "title": "FC Barcelona",
      "paragraph_text": "In June 1982, Diego Maradona was signed for a world record fee of £5 million from Boca Juniors. In the following season, under coach Luis, Barcelona won the Copa del Rey, beating Real Madrid. However, Maradona's time with Barcelona was short-lived and he soon left for Napoli. At the start of the 1984–85 season, Terry Venables was hired as manager and he won La Liga with noteworthy displays by German midfielder Bernd Schuster. The next season, he took the team to their second European Cup final, only to lose on penalties to Steaua Bucureşti during a dramatic evening in Seville.",
      "is_supporting": true
    },
    // ... 其余 17 个省略，结构相同
  ],
  "question": "When was the person who Messi's goals in Copa del Rey compared to get signed by Barcelona?",
  "question_decomposition": [
    {
      "id": 13548,
      "question": "To whom was Messi's goal in the first leg of the Copa del Rey compared?",
      "answer": "Diego Maradona",
      "paragraph_support_idx": 1
    },
    {
      "id": 13529,
      "question": "When was #1 signed by Barcelona?",
      "answer": "June 1982",
      "paragraph_support_idx": 2
    }
  ],
  "answer": "June 1982",
  "answer_aliases": [],
  "answerable": true
}
```

### narrativeqa

无候选段落列表，只有单篇 `document.text`（本例 210,625 字符）。该字段与 `summary.tokens` 均截断标注。

```json
{
  "document": {
    "id": "4b30ab1c49b62dc59b9773954958d9ac6807a865",
    "kind": "movie",
    "url": "http://www.imsdb.com/scripts/All-About-Steve.html",
    "file_size": 211827,
    "word_count": 28085,
    "start": "ALL ABOUT STEVE",
    "end": ". THE END",
    "summary": {
      "text": " Mary Horowitz, a crossword puzzle writer for the Sacramento Herald, is socially awkward and considers her pet hamster her only true friend.\nHer parents decide to set her up on a blind date. Mary's ex ...<截断>",
      "tokens": [
        "Mary",
        "Horowitz",
        ",",
        "a",
        "crossword",
        "puzzle",
        "...<截断，共 492 个 token>"
      ],
      "url": "http://en.wikipedia.org/wiki/All_About_Steve",
      "title": "All About Steve"
    },
    "text": "<html>\n<head><title>All About Steve Script at IMSDb.</title>\n<meta name=\"description\" content=\"All About Steve script at the Internet Movie Script Database.\">\n<meta name=\"keywords\" content=\"All About  ...<截断，全文共 210,625 字符>"
  },
  "question": "What is Mary Horowitz's job?",
  "answer": [
    "She is a crossword writer for the Sacramento Herald.",
    "She is a crossword puzzle writer."
  ]
}
```
