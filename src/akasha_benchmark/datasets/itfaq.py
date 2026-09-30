from __future__ import annotations

from typing import Any, ClassVar

from .base import DatasetAdapter
from .common import require_native_id
from .corpus import CorpusIndex
from .models import CanonicalSample, DataDependency, SubsetStrategy, make_sample_id


class ITFaqAdapter(DatasetAdapter):
    name: ClassVar[str] = "itfaq"
    qa_filename: ClassVar[str] = "itfaq.json"
    corpus_filename: ClassVar[str] = "itfaq_corpus.json"
    provides: ClassVar[frozenset[DataDependency]] = frozenset({DataDependency.REFERENCE_ANSWERS})
    subset_strategy: ClassVar[SubsetStrategy] = SubsetStrategy.FULL_CORPUS
    downloadable: ClassVar[bool] = False

    def expected_qa_rows(self) -> int:
        return 628

    def parse_row(
        self, row: dict[str, Any], row_index: int, corpus: CorpusIndex
    ) -> CanonicalSample:
        native_id = require_native_id(row, row_index, self.name, "id")

        answer = row.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"{self.name}: row {row_index} answer is not a non-empty string")

        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"{self.name}: row {row_index} question is not a non-empty string")

        return CanonicalSample(
            dataset=self.name,
            sample_id=make_sample_id(self.name, native_id),
            dataset_sample_id=native_id,
            question=question,
            answers=(answer,),
            gold_doc_ids=(),
            metadata={

                "answer_chars": len(answer),
            },
        )
