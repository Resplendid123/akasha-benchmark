"""itfaq 适配器。628 条中文 IT 支持问答、42 篇 FAQ 文档，无 evidence 标注。

不是 HippoRAG_2 的一部分，是本地数据集，所以 ``downloadable`` 为假。

与 narrativeqa 一样不声明 ``GOLD_DOCS``：检索指标在这组上无定义，
请求它们会抛异常而不是记 0.0。但两者的抽子集策略不同 —— narrativeqa 的 QA 行带
``document.id``，抽掉一部分文档后还能筛出「属于这些文档」的问题；本集的 QA 行
只有 ``{id, question, answer}``，没有任何指回文档的字段，抽语料就无法保证被抽到的
问题还答得上。所以语料整份导入，见 :attr:`SubsetStrategy.FULL_CORPUS`。

语料正文自带 ``# {title}`` 首行，重复的那一遍由 ``CorpusDoc.to_markdown`` 去掉。
"""

from __future__ import annotations

from typing import Any, ClassVar

from .base import DatasetAdapter
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
        native_id = row.get("id")
        if not isinstance(native_id, str) or not native_id:
            raise ValueError(f"{self.name}: row {row_index} has no usable 'id'")

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
                # 本集答案长到 700 字符，读 EM/F1 时要拿它当背景。
                "answer_chars": len(answer),
            },
        )
