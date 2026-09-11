"""narrativeqa 适配器。293 个问题、10 篇长文档，无 evidence 标注。

**故意不声明** ``GOLD_DOCS``：这份数据没有 gold 文档，检索指标在它上面
是无定义的。对它请求检索指标会抛异常，而不是把 0.0 混进平均值。

身份：没有原生 QA ID，所以 ``dataset_sample_id`` 用它在**全量**数据文件里的
行号字符串。这条口径只在 :data:`SAMPLE_ID_RULES` 注册一次，预处理与评测
两侧都从那里读，所以抽子集不会让样本被重新编号。

QA 文件有 94MB，因为每一行都内联了整篇 ``document.text``
（约 210KB x 293 行，而实际只有 10 篇不同的文档）。解析时只留
``document.id`` / ``kind`` 并丢掉正文，加载成本才降下来；
同样的内容 corpus 文件里已经切好块了。
"""

from __future__ import annotations

from typing import Any, ClassVar

from .base import DatasetAdapter
from .corpus import CorpusIndex
from .models import CanonicalSample, DataDependency, make_sample_id


def document_id_of(doc_id: str) -> str:
    """``"4b30ab…865_17"`` 取 ``"4b30ab…865"``。corpus 的 idx 形如 ``{document_id}_{chunk_seq}``。"""
    return doc_id.rsplit("_", 1)[0]


class NarrativeQAAdapter(DatasetAdapter):
    name: ClassVar[str] = "narrativeqa"
    aliases: ClassVar[tuple[str, ...]] = (
        "narrative_qa",
        "narrativeqa_dev_10_doc",
        "narrativeqa_dev",
    )
    qa_filename: ClassVar[str] = "narrativeqa.json"
    corpus_filename: ClassVar[str] = "narrativeqa_corpus.json"
    # 不含 GOLD_DOCS，原因见模块 docstring。
    provides: ClassVar[frozenset[DataDependency]] = frozenset({DataDependency.REFERENCE_ANSWERS})

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

        # document.id 是文档级的（只有 10 个不同值），绝不能当行身份。
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
