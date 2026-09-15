"""2wikimultihopqa 适配器。1000 行，2-4 跳，维基百科 + Wikidata。

``evidences`` 是关系三元组，形如 ``["Lothair II", "mother", "Ermengarde of Tours"]``。
它是推理链的符号化表示，**不是 gold 文档**，禁止当检索 ground truth ——
本适配器只用 ``supporting_facts`` 解析 gold，三元组仅存条数进 metadata
供分层报告用。
"""

from __future__ import annotations

from typing import Any, ClassVar

from .base import DatasetAdapter
from .common import gold_titles_from_supporting_facts, resolve_gold_doc_ids
from .corpus import CorpusIndex
from .models import CanonicalSample, DataDependency, SubsetStrategy, make_sample_id

# 本集的 context 句子用空格拼接（hotpotqa 是空串），差异见 common.py。
SENTENCE_JOINER = " "


class TwoWikiMultihopQAAdapter(DatasetAdapter):
    name: ClassVar[str] = "2wikimultihopqa"
    aliases: ClassVar[tuple[str, ...]] = ("2wiki", "twowiki", "2wikimultihop", "two_wiki")
    qa_filename: ClassVar[str] = "2wikimultihopqa.json"
    corpus_filename: ClassVar[str] = "2wikimultihopqa_corpus.json"
    provides: ClassVar[frozenset[DataDependency]] = frozenset(
        {DataDependency.GOLD_DOCS, DataDependency.REFERENCE_ANSWERS}
    )
    subset_strategy: ClassVar[SubsetStrategy] = SubsetStrategy.QA_THEN_GOLD

    def expected_qa_rows(self) -> int:
        return 1000

    def parse_row(
        self, row: dict[str, Any], row_index: int, corpus: CorpusIndex
    ) -> CanonicalSample:
        native_id = row.get("_id")
        if not isinstance(native_id, str) or not native_id:
            raise ValueError(f"{self.name}: row {row_index} has no usable '_id'")

        answer = row.get("answer")
        if not isinstance(answer, str) or not answer:
            raise ValueError(f"{self.name}: row {row_index} answer is not a non-empty string")

        titles = gold_titles_from_supporting_facts(row, row_index, self.name)
        gold_doc_ids = resolve_gold_doc_ids(titles, corpus, row_index, self.name)

        evidences = row.get("evidences") or []
        return CanonicalSample(
            dataset=self.name,
            sample_id=make_sample_id(self.name, native_id),
            dataset_sample_id=native_id,
            question=row["question"],
            answers=(answer,),
            gold_doc_ids=gold_doc_ids,
            metadata={
                "type": row.get("type"),
                "gold_count": len(gold_doc_ids),
                "supporting_fact_count": len(row["supporting_facts"]),
                # 推理链长度，仅供分层报告。不是 gold 文档。
                "evidence_triple_count": len(evidences),
                "answer_id": row.get("answer_id"),
            },
        )
