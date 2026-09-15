"""规范化样本与语料模型，各阶段共用。

两条下游都依赖的规则：

* ``extra="forbid"`` 加 ``frozen=True``，上游改字段名时在构造处报错。
* 数据集声明自己拥有什么数据，不声明支持什么指标。缺依赖时抛
  :class:`DependencyError`，不把假的 0.0 混进汇总。

新增指标不必碰这里的枚举：指标在 ``metrics/registry.py`` 声明 ``requires``。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

# doc_id / sample_id 的口径收在这里一处，预处理与评测两侧都从这里读。
# corpus 行身份：值是原生身份字段名，或哨兵 "row_idx" 表示用原始全量文件行号
# （抽子集后不重新编号）。musique 的 title 有歧义时用 (title, text) 消歧。
CORPUS_ID_RULES: dict[str, str] = {
    "hotpotqa": "idx",
    "2wikimultihopqa": "row_idx",
    "musique": "row_idx",
    "narrativeqa": "idx",
    "itfaq": "id",
}

# 样本身份。hotpotqa / 2wiki 的原生字段是 "_id"，musique 与 itfaq 是 "id"；
# narrativeqa 没有原生 QA ID，用它在全量文件里的行号字符串。
SAMPLE_ID_RULES: dict[str, str] = {
    "hotpotqa": "native_id",
    "2wikimultihopqa": "native_id",
    "musique": "native_id",
    "narrativeqa": "row_idx",
    "itfaq": "native_id",
}


class DependencyError(RuntimeError):
    """数据集缺少某个指标所需的数据依赖时抛出，而不是返回 0.0。"""


class SubsetStrategy(StrEnum):
    """编译抽子集走哪条路。适配器声明，``stages/compile`` 按声明分派。"""

    # 先均匀抽 QA，再取它们的 gold 加负样本。
    QA_THEN_GOLD = "uniform_qa_then_gold_corpus"
    # 同上，但 QA 按跳数分层 —— musique 不分层几乎全是 2hop。
    STRATIFIED_HOP = "stratified_by_hop"
    # 整篇取文档，再取属于这些文档的问题。要求样本带 document_id。
    WHOLE_DOCS = "whole_documents"
    # 语料全量导入，只抽 QA。给「没有 gold 也没有问题→文档映射」的数据集用：
    # 抽语料就无法保证被抽到的问题还答得上。
    FULL_CORPUS = "full_corpus"


class DataDependency(StrEnum):
    """一个数据集拥有什么标注。指标声明需要哪些，闸门做集合比对。

    judge 类指标的 ``requires`` 是空集，所以对四组都成立。
    """

    # 检索、引用归因、多跳指标都依赖它。
    # 2wiki 的 evidences 是关系三元组而不是 gold 文档，不算。
    GOLD_DOCS = "gold_docs"
    # EM / F1 依赖它。
    REFERENCE_ANSWERS = "reference_answers"


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

        导入服务取首个 heading 当 page title，所以 heading 负责 title、
        文件名负责 doc_id，重复 title 因此不影响身份追踪。

        正文首行已经是这个 heading 时不再加一遍（itfaq 的语料自带）。
        比的是首行精确相等：musique 有正文以 ``# `` 开头的表格片段，那些不算。
        """
        first = self.text.lstrip().splitlines()[0] if self.text.strip() else ""
        if first == f"# {self.title}":
            return f"{self.text.strip()}\n"
        return f"# {self.title}\n\n{self.text}\n"


def make_sample_id(dataset: str, dataset_sample_id: str) -> str:
    return f"{dataset}:{dataset_sample_id}"
