"""narrativeqa 适配器。293 个问题、10 篇长文档，无 evidence 标注。

不声明 ``GOLD_DOCS``：检索指标在这组上无定义，请求它们会抛异常而不是记 0.0。
``dataset_sample_id`` 用全量文件里的行号字符串，口径见 :data:`SAMPLE_ID_RULES`。
解析时丢掉每行内联的 ``document.text``（同样内容 corpus 已切块）。
"""

from __future__ import annotations

from typing import Any, ClassVar

from .base import DatasetAdapter
from .corpus import CorpusIndex
from .models import CanonicalSample, DataDependency, SubsetStrategy, make_sample_id


class NarrativeQAAdapter(DatasetAdapter):
    name: ClassVar[str] = "narrativeqa"
    aliases: ClassVar[tuple[str, ...]] = (
        "narrative_qa",
        "narrativeqa_dev_10_doc",
        "narrativeqa_dev",
    )
    qa_filename: ClassVar[str] = "narrativeqa.json"
    corpus_filename: ClassVar[str] = "narrativeqa_corpus.json"
    provides: ClassVar[frozenset[DataDependency]] = frozenset({DataDependency.REFERENCE_ANSWERS})
    subset_strategy: ClassVar[SubsetStrategy] = SubsetStrategy.WHOLE_DOCS

    def expected_qa_rows(self) -> int:
        return 293

    def parse_row(
        self, row: dict[str, Any], row_index: int, corpus: CorpusIndex
    ) -> CanonicalSample:
        answers_raw = row.get("answer")
        if not isinstance(answers_raw, list) or not answers_raw:
            raise ValueError(
                f"{self.name}: row {row_index} answer is not a non-empty list of references"
            )
        answers: dict[str, None] = {}
        for ref in answers_raw:
            if not isinstance(ref, str) or not ref.strip():
                raise ValueError(f"{self.name}: row {row_index} has a blank reference answer")
            answers.setdefault(ref, None)

        document = row.get("document")
        if not isinstance(document, dict) or not document.get("id"):
            raise ValueError(f"{self.name}: row {row_index} has no document.id")
        document_id = document["id"]

        # document.id 是文档级的（只有 10 个不同值），不能当行身份。
        native_id = str(row_index)
        return CanonicalSample(
            dataset=self.name,
            sample_id=make_sample_id(self.name, native_id),
            dataset_sample_id=native_id,
            question=row["question"],
            answers=tuple(answers),
            gold_doc_ids=(),
            metadata={
                # 抽子集靠这个字段整篇整篇地抽文档。
                "document_id": document_id,
                "kind": document.get("kind"),
                "reference_count": len(answers),
                "summary_title": (document.get("summary") or {}).get("title"),
            },
        )
