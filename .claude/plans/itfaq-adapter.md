# 接入 itfaq 适配器

基线：147 个测试全过。每步改完重跑 `uv run pytest -q` 与 `npm --prefix web run typecheck`。

## 数据形状（已实测，非推测）

| 文件 | 行数 | 字段 | 备注 |
| --- | --- | --- | --- |
| `itfaq.json` | 628 | `{id, question, answer}` | `id` 形如 `qa_001`；`answer` 是单串，不是列表 |
| `itfaq_corpus.json` | 42 | `{id, title, text}` | `id` 形如 `doc_001`；42 个 title 全唯一 |

- **无 gold 标注，也无「问题→文档」映射**。所以既不能算检索族指标，
  也不能像 narrativeqa 那样按 `document_id` 安全地抽文档子集。
- 42 篇 text 全部以 `# {title}` 开头，且与 title 精确相等。
- 8 条 question 文本重复（`id` 不同），不影响 `sample_id` 唯一性。
- 语料无 `idx` 字段，身份字段是 `id` —— 现有 `assign_doc_id` 两条规则都不认。

## 两个决策

1. **语料子集取全量 42 篇**（新策略 `full_corpus`）。没有「问题→文档」映射，
   抽子集就无法保证被抽到的问题还答得上，no_match 占比不可控，judge 指标失去意义。
   代价：一次编译 42 篇约 28 分钟（narrativeqa 默认子集 356 篇约 4 小时，反而更贵）。
2. **适配器声明 `subset_strategy`**，`compile` 按声明分派，删掉
   `"hop_prefix" in samples[0]["metadata"]` 这处字段嗅探 —— 它与 `base.py`
   开头「分派只按数据集名字，不按 row 里有没有某个字段猜」的约定相反。

## 一、`datasets/models.py`

`CORPUS_ID_RULES` 的值从规则代号改成**身份字段名**，`row_idx` 保留作哨兵：

```python
CORPUS_ID_RULES = {
    "hotpotqa": "idx", "narrativeqa": "idx",
    "2wikimultihopqa": "row_idx", "musique": "row_idx",
    "itfaq": "id",
}
```

去掉 `native_id` 这层间接，itfaq 的 `id` 因此不用新增规则分支。
`identity_rules()` 经 `/api/datasets` 出到前端，值从 `"native_id"` 变成 `"idx"`；
已 grep 确认无代码或测试比对这些字面量。

新增 `SubsetStrategy(StrEnum)`：`QA_THEN_GOLD` / `STRATIFIED_HOP` /
`WHOLE_DOCS` / `FULL_CORPUS`。放这里而不是 `base.py`，与 `DataDependency`
同一处 —— 两者都是 `datasets` 与 `stages` 共用的词汇表。

`SAMPLE_ID_RULES` 加 `"itfaq": "native_id"`（QA 行自带 `qa_001`）。

`CorpusDoc.to_markdown()` 加一处幂等判断：正文首行已经是 `# {title}` 时不再加标题。

```python
first = self.text.lstrip().splitlines()[0] if self.text.strip() else ""
if first == f"# {self.title}":
    return f"{self.text.rstrip()}\n"
return f"# {self.title}\n\n{self.text}\n"
```

比的是**首行精确相等**，不是 `startswith`。已实测：musique 有 2 行正文以 `# ` 开头，
但那是 markdown 表格片段，首行不等于 `# {title}`，所以四组现有数据集渲染结果不变；
只有 itfaq 的 42 篇受影响，避免标题重复两遍进 chunk 与 embedding。

## 二、`datasets/corpus.py`

`assign_doc_id` 按新口径改：`row_idx` 走行号（并保留「意外出现身份字段说明数据换版」
的检查），否则按字段名取值、缺字段就报错。模块 docstring 的「四组」改成「各组」。

## 三、`datasets/base.py`

`DatasetAdapter` 加两个 ClassVar：

- `subset_strategy: ClassVar[SubsetStrategy]` —— 编译抽子集走哪条路。
- `downloadable: ClassVar[bool] = True` —— itfaq 覆盖成 `False`。
  它不在 HippoRAG_2 仓库里，是本地数据集，下载阶段只校验存在性，不发 HTTP。

## 四、`datasets/itfaq.py`（新增）

```python
class ITFaqAdapter(DatasetAdapter):
    name = "itfaq"
    aliases = ("it_faq", "itfaq_zh")
    qa_filename = "itfaq.json"
    corpus_filename = "itfaq_corpus.json"
    provides = frozenset({DataDependency.REFERENCE_ANSWERS})   # 无 gold
    subset_strategy = SubsetStrategy.FULL_CORPUS
    downloadable = False

    def expected_qa_rows(self) -> int: return 628
```

`parse_row` 校验 `id` / `question` / `answer` 三个字段均为非空串，
`answers=(answer,)`，`gold_doc_ids=()`。metadata 存 `answer_chars`
（README 已写明解释性长答案的 EM 偏低，长度是读指标时的背景）。

## 五、`datasets/registry.py` 与 `datasets/__init__.py`

注册 `ITFaqAdapter`，docstring 从「四个适配器」改成「各适配器」。
`__init__` 导出 `SubsetStrategy`。

## 六、`stages/compile.py`

`build_subset` 按 `adapter.subset_strategy` 分派，四条策略各一段：

- `FULL_CORPUS`：`doc_ids = sorted(by_id)`，`gold_ids = negatives = []`，
  QA 按 seed 抽 `qa_limit` 条。**`negatives_ratio` 对这一组失效**，日志写明。
- `WHOLE_DOCS`：narrativeqa 现有逻辑原样搬过去。
- `STRATIFIED_HOP` / `QA_THEN_GOLD`：现有 gold 分支拆成这两条，
  gold + 负样本那段两者共用。

保留 `adapter.has(GOLD_DOCS)` 的前置检查：策略声明要 gold 而数据集没有时报错，
不静默产出空 gold 集。`DEFAULT_NARRATIVEQA_DOCS` 与 `narrativeqa_docs` 形参不动。

## 七、`stages/download.py`

`run()` 把选中的文件分成两拨：`downloadable` 适配器的文件去下载，
其余只进校验清单。全部选中项都是本地数据集时**跳过站点解析**，不发任何 HTTP。
docstring 与 `run()` 的「四组」改成按 `datasets` 参数描述。

## 八、`api/datasets.py` 与前端

- `/api/datasets` 每项加 `downloadable`。docstring「四组」改「各组」。
- `types.ts` 的 `DatasetEntry` 加 `downloadable: boolean`。
- `Datasets.tsx`：下载目标过滤掉 `downloadable === false` 的组；
  本地数据集缺文件时状态显示「缺失（本地数据集）」而不是让用户点下载。
  `allReady` 判断不变。

`Testing.tsx` 的 `DATASETS` 不动 —— 链路测试要测检索族指标，itfaq 没有 gold。
`stages/chain.py` 的 `DATASETS` 同理不动。

## 九、测试（`dataset/` 已 gitignore，用合成夹具，与 conftest 现有做法一致）

`tests/conftest.py` 加 `itfaq_dataset_dir` 夹具（3 条 QA、2 篇语料，
语料首行带 `# {title}`）。新增 `tests/test_itfaq.py`：

1. 归一化：`sample_id` 为 `itfaq:qa_001`，`doc_id` 取自 `id` 字段（`doc_001`），
   `answers` 单元素，`gold_doc_ids` 为空。
2. 无 gold：检索族与引用族指标全部落进 `registry.omitted`，
   `faithfulness` 不在其中。
3. `full_corpus`：`qa_limit` 小于总数时语料仍是全部，`gold` 与 `negatives` 为 0，
   且 `negatives_ratio=0.0` 不改变语料篇数。
4. `to_markdown` 幂等：首行已是 `# {title}` 时不重复加标题；
   另加一条 hotpotqa 形状的反例确认原行为不变。
5. `download.run` 只选 itfaq 时不发 HTTP（monkeypatch `_resolve_endpoint` 抛异常，
   仍应通过），文件缺失时报错。

## 十、文档

- `README.md`：数据集表述从四组改为五组；指标口径那条补一句 itfaq 与 narrativeqa
  同为无 gold 组，检索族指标省略。
- `docs/architecture.md`：编译层那条补 `full_corpus` 策略一句。
