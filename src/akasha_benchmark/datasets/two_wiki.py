from __future__ import annotations

from typing import Any, ClassVar

from .base import DatasetAdapter
from .common import (
    gold_titles_from_supporting_facts,
    require_native_id,
    resolve_gold_doc_ids,
)
from .corpus import CorpusIndex
from .models import CanonicalSample, DataDependency, SubsetStrategy, make_sample_id


class TwoWikiMultihopQAAdapter(DatasetAdapter):
    name: ClassVar[str] = "2wikimultihopqa"
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
        native_id = require_native_id(row, row_index, self.name, "_id")

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

                "evidence_triple_count": len(evidences),
                "answer_id": row.get("answer_id"),
            },
        )
