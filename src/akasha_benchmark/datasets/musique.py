"""musique 适配器。1000 行，2-4 跳，带显式子问题分解。

本集有两处是特有的：

* gold 按 ``(title, paragraph_text)`` 定位，**不能只按 title**。
  2648 条 gold 段落里有 769 条的 title 对应多行 corpus；而 (title, text)
  在全部 11656 行上唯一。这是实测结论，不是推测。
* ``answer_aliases`` 并入 ``answers``，这样答案 F1 对多参考取 max 时
  自动覆盖别名，不用在打分侧再写一遍别名逻辑。

跳数编码在 id 前缀里（``2hop__…``、``3hop1__…``），抽子集靠它分层，
评测靠它出随跳数的衰减曲线。
"""

from __future__ import annotations

from typing import Any, ClassVar

from .base import DatasetAdapter
from .corpus import CorpusIndex
from .models import CanonicalSample, DataDependency, SubsetStrategy, make_sample_id

HOP_PREFIXES = ("2hop", "3hop1", "3hop2", "4hop1", "4hop2", "4hop3")


def hop_prefix(dataset_sample_id: str) -> str:
    """``"3hop1__9285_5188_23307"`` 取 ``"3hop1"``。前缀不认识就报错。

    注意分隔符是双下划线 ``__``，不是冒号。
    """
    prefix = dataset_sample_id.split("__", 1)[0]
    if prefix not in HOP_PREFIXES:
        raise ValueError(f"musique: unrecognized hop prefix {prefix!r} in id {dataset_sample_id!r}")
    return prefix


def hop_count(prefix: str) -> int:
    """``"4hop2"`` 取 4。"""
    return int(prefix[0])


class MusiqueAdapter(DatasetAdapter):
    name: ClassVar[str] = "musique"
    aliases: ClassVar[tuple[str, ...]] = ("musique_ans", "musique-ans")
    qa_filename: ClassVar[str] = "musique.json"
    corpus_filename: ClassVar[str] = "musique_corpus.json"
    provides: ClassVar[frozenset[DataDependency]] = frozenset(
        {DataDependency.GOLD_DOCS, DataDependency.REFERENCE_ANSWERS}
    )
    subset_strategy: ClassVar[SubsetStrategy] = SubsetStrategy.STRATIFIED_HOP

    def expected_qa_rows(self) -> int:
        return 1000

    def parse_row(
        self, row: dict[str, Any], row_index: int, corpus: CorpusIndex
    ) -> CanonicalSample:
        native_id = row.get("id")
        if not isinstance(native_id, str) or not native_id:
            raise ValueError(f"{self.name}: row {row_index} has no usable 'id'")

        answer = row.get("answer")
        if not isinstance(answer, str) or not answer:
            raise ValueError(f"{self.name}: row {row_index} answer is not a non-empty string")

        paragraphs = row.get("paragraphs")
        if not isinstance(paragraphs, list) or not paragraphs:
            raise ValueError(f"{self.name}: row {row_index} has empty or non-list paragraphs")

        gold_ids: dict[str, None] = {}
        ambiguous_titles = 0
        for para in paragraphs:
            if not para.get("is_supporting"):
                continue
            title, text = para["title"], para["paragraph_text"]
            try:
                doc_id = corpus.id_for_pair(title, text)
            except KeyError as exc:
                raise ValueError(f"{self.name}: row {row_index} gold unresolvable: {exc}") from None
            # 记下有多少 gold 的 title 本身是歧义的，作为「必须用 pair 消歧」的证据。
            if len(corpus.title_to_ids.get(title, ())) > 1:
                ambiguous_titles += 1
            gold_ids.setdefault(doc_id, None)

        if not gold_ids:
            raise ValueError(f"{self.name}: row {row_index} has no is_supporting paragraph")

        # 别名并入参考答案集；保序去重，主答案排在最前，便于逐样本结果好读。
        answers: dict[str, None] = {answer: None}
        for alias in row.get("answer_aliases") or []:
            if isinstance(alias, str) and alias:
                answers.setdefault(alias, None)

        prefix = hop_prefix(native_id)
        return CanonicalSample(
            dataset=self.name,
            sample_id=make_sample_id(self.name, native_id),
            dataset_sample_id=native_id,
            question=row["question"],
            answers=tuple(answers),
            gold_doc_ids=tuple(gold_ids),
            metadata={
                "hop_prefix": prefix,
                "hop_count": hop_count(prefix),
                "gold_count": len(gold_ids),
                "answerable": row.get("answerable"),
                "alias_count": len(answers) - 1,
                "decomposition_steps": len(row.get("question_decomposition") or []),
                "gold_with_ambiguous_title": ambiguous_titles,
            },
        )
