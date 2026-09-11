"""规范化样本与语料模型，各阶段共用。

有两条规则是后续所有代码都依赖的：

* ``extra="forbid"`` 加 ``frozen=True`` —— 上游改字段名时在构造处直接报错，
  而不是静默归一成空值；校验通过后任何下游代码都改不动它。
* **数据集声明自己「拥有」什么数据，不声明「支持」什么指标**（PLAN.md §12.4
  决策 10）。narrativeqa 没有 gold 文档标注，所以它不声明
  :attr:`DataDependency.GOLD_DOCS`；对它请求依赖 gold 的指标会抛
  :class:`DependencyError`，而不是把一个假的 0.0 混进汇总。

这个方向此前是反的：两个枚举成员的名字取的是指标名（``EVIDENCE_RECALL`` /
``ANSWER_F1``），而**值**表达的已经是数据依赖。反转之后好处是具体的 ——
judge 类指标（faithfulness、answer relevancy）不需要任何标注，对四组都成立，
用指标名根本没法声明它们；而 narrativeqa 现在整组检索指标省略，
judge 恰好能填上这个洞。

新增指标不再需要碰这个枚举：指标在 ``metrics/registry.py`` 里声明自己的
``requires``，闸门做集合比对。
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


class DependencyError(RuntimeError):
    """数据集缺少某个指标所需的数据依赖时抛出。

    刻意不返回 0.0：假分数会静默污染汇总，而一个异常会在计算之前就停下来。
    """


class DataDependency(StrEnum):
    """一个数据集**拥有**什么标注。指标声明需要哪些，闸门做集合比对。

    只有两个成员不是因为想不出更多，而是因为这份数据里确实只有这两种标注。
    judge 类指标的 ``requires`` 是空集 —— 它们什么都不需要，所以对四组都成立。
    """

    # 有 gold 文档标注：检索、引用归因、多跳指标都依赖它。
    # 2wiki 的 evidences 是关系三元组而不是 gold 文档，不算（§4）。
    GOLD_DOCS = "gold_docs"
    # 有参考答案：EM / F1 依赖它。标的是「可以打分」，与报几个指标无关。
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

        Akasha 的导入服务会取首个 Markdown heading 当 page title 并从正文移除，
        所以 heading 负责 title、**文件名**负责 doc_id。两者独立，
        这样即使 musique 有 647 个重复 title 也不影响身份追踪。
        """
        return f"# {self.title}\n\n{self.text}\n"


def make_sample_id(dataset: str, dataset_sample_id: str) -> str:
    return f"{dataset}:{dataset_sample_id}"
