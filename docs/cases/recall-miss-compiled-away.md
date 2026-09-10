# 召回漏一篇 gold：修饰语被编译掉了

**样本** `hotpotqa:5ae4f2595542990ba0bbb1a8`（bridge / hard，gold 2 篇）
**run** `run001`，2026-09-10
**结论** 非调参可解。调 `scoreThreshold`、加大 k、换检索模式都救不回来。

---

## 一句话

原文里有一个能和查询逐字对应的短语，**它在编译产物里消失了**；
而编译产物才是被索引的文本。

## 表面现象

| 指标 | 值 | 读起来像 |
| --- | --- | --- |
| `recall@5` | 0.500 | 一半的 gold 没找到 |
| `recall@20` | 0.500 | 加大 k 也没用 |
| `mrr` | 1.000 | 但第一名就是 gold |
| `f1` | 0.136 | 答案质量很差 |
| `em` | 0.000 | 完全没答对 |

**而答案是对的。** 参考答案 `June 22, 1953`，模型答 "born on **June 22, 1953**"，
还主动声明了证据缺口：

> *(Note: The provided context details her 1985 Grammy Award win, though it does not
> explicitly mention an Emmy Award.)*

所以五个指标里有四个在误导，只有 `mrr` 是对的。

## 这道题的结构

```
Q: When was the American singer, songwriter, actress and LGBT rights activist
   born who won Grammy and Emmy award?
A: June 22, 1953
```

hotpotqa 的 bridge 题要两篇文档接力：

| gold | 内容 | 作用 |
| --- | --- | --- |
| `6369` Cyndi Lauper | "(born June 22, 1953) is an American singer, songwriter, actress and LGBT rights activist" | **答案在这里** |
| `6365` Dee Does Broadway | "Guests in the album include the **Grammy and Emmy award winning** Cyndi Lauper…" | **识别线索在这里** |

查询里的 `who won Grammy and Emmy award` 这一段，出处是 6365。
标注要求两篇都召回，因为按题目设计你得靠 6365 才能把「拿过 Grammy 和 Emmy 的那位」
锁定到 Cyndi Lauper。

实际召回 11 篇：

```
 1. 6369  Cyndi Lauper            <== GOLD，第一名
 2. 1682  Shirley Manson
 3. 6274  Amy (2015 film)
 4. 1556  To Know Him Is to Love Him
 5. 2660  Dolly Parton
 6. 9466  Sarah Kate Ellis
 7. 8607  Josephine Baker
 8. 1720  Anita Loos
 9. 7071  Gary Sinise
10. 4767  Kristy Lee Cook
11. 1307  Dianne Hiles
```

6365 **不在里面**，到 k=20 也没有。

## 先排除两个常见解释

**不是上下文预算挤掉的。** `budget` 里 `includedItemCount: 20`、
`omittedItemCount: 0` —— 一条都没被截断。

**不是候选集为空。** `audit_join.json` 的 `recall_ceiling_miss_rate` 是 0.0，
`mean_candidate_chunks` 恒为 200.0。候选集从来不空。

6365 是**压根没进候选集**，不是进了之后被丢掉。

---

## 根因：三条召回路径同时断

### 1. 词法断了 —— 关键短语被编译掉

`knowledge_chunks`（**参与召回**）与 `knowledge_source_chunks`（**不参与召回**）
的同一段内容对照：

```
原文（knowledge_source_chunks）：
  "Guests in the album include the Grammy and Emmy award winning Cyndi Lauper,
   Clay Aiken, Nick Adams and many others."

编译（knowledge_chunks）：
  "…featuring vocal contributions from guest artists including
   Cyndi Lauper, Clay Aiken, and Nick Adams."
```

6365 编出的三个 artifact，chunk 正文**全部** `含 Grammy: False` / `含 Emmy: False`。

拆开看丢了什么：

```
原文:  the Grammy and Emmy award winning    Cyndi Lauper
       └────────────┬────────────┘         └─────┬─────┘
                 修饰语                        实体
编译:                                       Cyndi Lauper
       修饰语没了                            实体名保留
```

**实体名活下来了，实体的头衔没有。** 而查询要靠头衔识别实体 ——
`won Grammy and Emmy award` 与原文的 `Grammy and Emmy award winning`
几乎逐字对应，词法召回本该直接命中。

### 2. 稠密断了 —— 改写后的主题偏离

编译产物的叙述中心是「Dee Snider 的百老汇翻唱专辑」。
查询问的是「某歌手的出生日期」。语义距离远，稠密召回抓不到。

原文里 Cyndi Lauper 是带头衔出现的，改写后她只是嘉宾名单的一项。

### 3. 图扩展断了 —— 那条边不存在

6365 编出 3 个 artifact，**全部图边只有 4 条**：

```
Dee Snider --[produced]-------> Dee Does Broadway
Dee Snider --[released_album]--> Dee Does Broadway
（反向各一条）
```

到 `canonical_key = cyndi_lauper` 的边：**0 条**。

而同一个 space 里确实存在那个 artifact，它自己的边是：

```
Cyndi Lauper --[released debut solo album]--> She's So Unusual
Cyndi Lauper --[released second record]-----> True Colors (Album)
```

两边各自成一小团，中间没有连接：

```
想要的:  6365[专辑] --嘉宾--> [Cyndi Lauper] <--同一实体--> 6369[Cyndi Lauper]
实际有:  6365[专辑] <--produced/released_album--> [Dee Snider]
         6369[Cyndi Lauper] --released_*--> [She's So Unusual] / [True Colors]
```

图扩展这次确实跑了（5 条 snippet 挂 `graph-neighbor`），连到的是
Cyndi Lauper 自己的专辑、以及别的女歌手/activist —— 全是「同类实体」，
而 6365 需要的是一条「嘉宾/参演」关系边。**不是走错路，是那条路不存在。**

---

## 这是个例还是系统性的

跨已入库的 hotpotqa 文档统计（[probe_extraction.py](scripts/probe_extraction.py)）。
三条结论，第一条与直觉相反。

> 下面的数字是 **2026-09-10 的快照**，当时编译已完成约 400–410 篇。
> 编译仍在推进，重跑脚本得到的绝对值会变（实测 401→410 篇时中位压缩率 2.19→2.20），
> 但三条结论的量级不受影响。

### 编译不是在压缩，是在扩写

| 编译后字符数 / 原文字符数 | |
| --- | --- |
| 中位数 | **2.19** |
| p25 / p75 | 1.81 / 2.65 |
| 最小 / 最大 | 0.67 / 6.62 |
| 净压缩（< 1.0）占比 | **2.7%** |

只有 2.7% 变短，中位数扩到 2.19 倍。
**所以「因为要压缩所以省略细节」不成立** —— 它写的字比原文多一倍。

丢修饰语是**改写策略**的结果，不是空间不够：改写成以主实体为中心的叙述时，
挂在次要实体身上的定语被剥掉。这篇的主实体是 Dee Snider 和专辑，
Cyndi Lauper 是嘉宾 —— 她的头衔被判定与本文主题无关。

### 图边极其稀疏，一半实体是孤点

| | |
| --- | --- |
| artifact 总数 | 1454（entity 925 / source_summary 401 / concept 116 / comparison 32） |
| 图边总数 | **555** |
| entity 中 0 条出边的 | **538 个（约 59%）** |

每篇抽 1–7 个 entity（中位 2–3），**没有硬上限**，所以不是「抽满就停」。
但边少得多：一半以上实体没有出边。
6365 那种「边只在本文档主实体之间」是典型而非例外。

### relation 是自由生成的，不是受控词表

| | |
| --- | --- |
| 边 555 条 | relation 取值 **377 种** |
| 只出现 1 次的 relation | **295 种** |
| 含空格的 relation | 82 条 |

同一语义多种写法并存：

```
directed_by(17) / directed(2) / directed_sequels_of / wrote and directed
created_by(8)   / created(3)  / createdBy(1)
produced_by(7)  / produced(1) / produced_and_played_on
co-founded / co_founded / co_founder_of / founded / founded_by / founding_member_of
father of（带空格）  …  served as head writer and executive producer of（一整句）
```

`createdBy` 与 `created_by` 并存、`father of` 带空格、还有一条 relation 是完整英文短语 ——
LLM 每篇自由发挥，没有 schema 约束或事后归一。
**后果是图遍历没法按关系类型做**，同一种关系有 N 个名字，
555 条边散在 377 种关系上（平均每种 1.5 条）。

---

## 顺带推翻 PLAN §0.3 的一条预判

PLAN.md §0.3 担心「**多跳可能偏高**，因为实体被物化成独立 artifact 并
**跨文档合并**，两个 hop 可能被编译器直接连成一条 graph edge」。

实测：925 个 entity artifact 里，**被多于一篇原文贡献的只有 12 个（1.3%）**。

```
Rijksmuseum / Ricky Skaggs / Isabella Bird / Al Gore / Kovno Ghetto / … 各 2 篇
```

`cyndi_lauper` 只被 6369 一篇贡献。**跨文档实体合并几乎没在发生。**

所以那条担忧在这份数据上方向是**反的**：合并没发生 → 桥接关系没建立 →
多跳靠图走不通。读多跳指标时要按这个方向理解，
`graph_exclusive_gold_*` 偏低是预期而非异常。

（这是对 §0.3 的实测更正，原文的机制描述没错 —— 编译器确有合并能力，
只是在这份 400 篇的语料上极少触发。语料更密、同一实体在多篇反复出现时，
结论可能不同。）

---

## 对指标口径的四条含义

**一、失败案例入口必须按 `answerMode` 默认切分。**
本 run 四条 `recall@5 < 1.0` 里，**三条是 `answerMode: general`**
（生成端回落，`retrievedSources` 被无条件清空，与检索找到什么无关），
只有这一条是真的漏 gold —— 而它答案还是对的。
混在一起看会把 1 条检索问题读成 4 条。全样本 `recall@5` 0.965、
knowledge-only 0.9948，差值 0.03 就是那三条。

**二、`full_coverage@k` 比 `recall@k` 更该当主指标。**
hotpotqa 全部 2 篇 gold，recall 只有 0/0.5/1 三个取值。
`recall@5 = 0.5` 看着像「一半没找到」，真实含义是「两篇差一篇」——
`full_coverage@5` 直接是 0，更准。

**三、这条样本是 judge 指标的典型靶子。**
答案对、`f1 = 0.136`、`recall@5 = 0.5`：三个确定性指标全给出「不好」的信号，
而它实质是好的（答案正确，且模型主动声明了证据缺口）。
reference-free 的 faithfulness / answer relevancy 才能识别这种情况。

**四、这是 §0.3「Recall@k 系统性偏低且非调参可解」的具体证据。**
要匹配的词已经不在被索引的文本里，任何检索侧参数都改不了。
唯一有意义的对照是 §9 的原文基线。

## 对平台设计的含义

排查这一条走了六跳、跨五张表，全靠手写 SQL。这正是 PLAN.md §12.9
「血缘视图」要实现的东西 —— 实现完之后拿这条样本走一遍，走不通就是没做到。

另外「**原文 vs 编译产物 diff**」应当是一等视图：本案的根因只有把两者并排才看得见。
§9 的原文基线从整体上量这个效应，而逐样本 diff 能直接指出丢了哪个词 ——
对调参无用（改不了编译器），对「这个数字该怎么读」有决定性作用。

---

## 复现

```bash
uv run python docs/cases/scripts/probe_lineage.py      # 本案的六跳链路
uv run python docs/cases/scripts/probe_extraction.py   # 全库的三条统计
```

链路（表名列名均已核对，非照文档抄）：

```
pages.id                    ← page_map.jsonl 的 page_id
  ↓ knowledge_page_sources.source_page_id
knowledge_pages.id          ← 编译产出的 artifact，不是原始 page
  ↓ knowledge_chunks.knowledge_page_id              参与召回的文本
  ↓ knowledge_graph_edges.from/to_knowledge_page_id 图边
knowledge_source_chunks.source_page_id              原文，不参与召回
```

两处与先前假设不符：`knowledge_chunks` 的外键是 `knowledge_page_id`
而非 `source_page_id`；`knowledge_page_sources` 直接带 `source_page_id`，
不必经 `knowledge_sources` 中转。

本案涉及的 ID：

| | |
| --- | --- |
| sample_id | `hotpotqa:5ae4f2595542990ba0bbb1a8` |
| gold doc_id | `6365`（Dee Does Broadway）、`6369`（Cyndi Lauper） |
| page_id 6365 | `01a089d1-fdc7-7e19-9d4b-05109dc38745` |
| page_id 6369 | `01a089d2-07d5-72d5-bf7e-a38f23fe2c51` |
