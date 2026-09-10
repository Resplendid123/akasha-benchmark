"""规范化样本与语料模型，各阶段共用。

有两条规则是后续所有代码都依赖的：

* ``extra="forbid"`` 加 ``frozen=True`` —— 上游改字段名时在构造处直接报错，
  而不是静默归一成空值；校验通过后任何下游代码都改不动它。
* capability 是显式声明的，不靠推断。narrativeqa 没有 evidence 标注，
  所以它不声明 ``EVIDENCE_RECALL``；对它请求该指标会抛 :class:`CapabilityError`,
  而不是把一个假的 0.0 混进汇总。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

# doc_id / sample_id 的口径收在这里一处，避免预处理与评测两侧各写一份而漂移。
# PLAN.md 0.1 定稿的各数据集 corpus 行身份：
#   hotpotqa     用原生 "idx"（int）转 str
#   2wiki        用 corpus 数组行号转 str
#   musique      用 corpus 数组行号转 str，title 有歧义时用 (title, text) 消歧
#   narrativeqa  用原生 "idx"（str，形如 "{document_id}_{chunk_seq}"）
CORPUS_ID_RULES: dict[str, str] = {
    "hotpotqa": "native_idx",
    "2wikimultihopqa": "row_index",
    "musique": "row_index",
    "narrativeqa": "native_idx",
}

# 样本身份。narrativeqa 没有原生 QA ID，用它在**全量**数据文件里的行号字符串
# （不是子集里的行号）。
SAMPLE_ID_RULES: dict[str, str] = {
    "hotpotqa": "native__id",
    "2wikimultihopqa": "native__id",
    "musique": "native_id",
    "narrativeqa": "full_dataset_row_index",
}


class CapabilityError(RuntimeError):
    """对不支持该指标的数据集请求指标时抛出。"""


class Capability(StrEnum):
    EVIDENCE_RECALL = "evidence_recall"
    ANSWER_F1 = "answer_f1"


class CanonicalSample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset: str
    sample_id: str
    dataset_sample_id: str
    question: str
    answers: tuple[str, ...]
    gold_doc_ids: tuple[str, ...]
    metadata: dict[str, Any]

    @field_validator("answers")
    @classmethod
    def _answers_nonempty(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("answers must hold at least one reference answer")
        return value

    @field_validator("sample_id", "dataset_sample_id", "question", "dataset")
    @classmethod
    def _no_blanks(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class CorpusDoc(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    doc_id: str
    title: str
    text: str

    def to_markdown(self) -> str:
        """渲染成 Akasha 导入用的 Markdown。

        Akasha 的导入服务会取首个 Markdown heading 当 page title 并从正文移除，
        所以 heading 负责 title、**文件名**负责 doc_id。两者独立，
        这样即使 musique 有 647 个重复 title 也不影响身份追踪。
        """
        return f"# {self.title}\n\n{self.text}\n"


def make_sample_id(dataset: str, dataset_sample_id: str) -> str:
    return f"{dataset}:{dataset_sample_id}"
