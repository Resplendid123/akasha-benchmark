"""hotpotqa 适配器。1000 行，维基百科 2 跳，句子级 evidence。"""

from __future__ import annotations

from typing import Any, ClassVar

from .base import DatasetAdapter
from .common import gold_titles_from_supporting_facts, resolve_gold_doc_ids
from .corpus import CorpusIndex
from .models import Capability, CanonicalSample, make_sample_id

# 本集的 context 句子用空串拼接才能还原成 corpus 的 text。原文基线要用。
SENTENCE_JOINER = ""


class HotpotQAAdapter(DatasetAdapter):
    name: ClassVar[str] = "hotpotqa"
    aliases: ClassVar[tuple[str, ...]] = ("hotpot", "hotpot_qa", "hotpotqa_dev")
    qa_filename: ClassVar[str] = "hotpotqa.json"
    corpus_filename: ClassVar[str] = "hotpotqa_corpus.json"
    capabilities: ClassVar[frozenset[Capability]] = frozenset(
        {Capability.EVIDENCE_RECALL, Capability.ANSWER_F1}
    )

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

        return CanonicalSample(
            dataset=self.name,
            sample_id=make_sample_id(self.name, native_id),
            dataset_sample_id=native_id,
            question=row["question"],
            answers=(answer,),
            gold_doc_ids=gold_doc_ids,
            metadata={
                "type": row.get("type"),
                "level": row.get("level"),
                # 去重后的 gold 篇数；本集实测恒为 2。
                "gold_count": len(gold_doc_ids),
                # 去重前的支撑句条数，与 gold_count 的差值即句子级冗余。
                "supporting_fact_count": len(row["supporting_facts"]),
            },
        )
