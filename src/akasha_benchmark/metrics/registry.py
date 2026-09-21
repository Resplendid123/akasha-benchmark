"""指标声明所需依赖，数据集声明已有标注；缺少依赖时省略指标。"""

from __future__ import annotations

from dataclasses import dataclass

from ..datasets.models import DataDependency, DependencyError

# 检索/引用/多跳三族依赖 gold 文档，qa 依赖参考答案，judge 无依赖。
FAMILY_RETRIEVAL = "retrieval"
FAMILY_QA = "qa"
FAMILY_ATTRIBUTION = "attribution"
FAMILY_MULTIHOP = "multihop"
FAMILY_JUDGE = "judge"

KIND_DETERMINISTIC = "deterministic"
KIND_JUDGE = "judge"


@dataclass(frozen=True)
class MetricDefinition:
    """一个指标的身份与它的数据依赖。registry 里存不带 k 的模板名。"""

    name: str
    family: str
    requires: frozenset[DataDependency]
    kind: str
    higher_is_better: bool
    description: str
    # 需要按 k 展开的指标（recall@2、recall@5 ...）。
    per_k: bool = False


def _definition(
    name: str,
    family: str,
    requires: frozenset[DataDependency],
    description: str,
    *,
    kind: str = KIND_DETERMINISTIC,
    higher_is_better: bool = True,
    per_k: bool = False,
) -> MetricDefinition:
    return MetricDefinition(
        name=name,
        family=family,
        requires=requires,
        kind=kind,
        higher_is_better=higher_is_better,
        description=description,
        per_k=per_k,
    )


_GOLD = frozenset({DataDependency.GOLD_DOCS})
_ANSWERS = frozenset({DataDependency.REFERENCE_ANSWERS})
_NONE: frozenset[DataDependency] = frozenset()

METRIC_DEFINITIONS: tuple[MetricDefinition, ...] = (
    # --- 检索：一律用 retrievedSources，不用 citations（后者已被裁剪过）---
    _definition("recall", FAMILY_RETRIEVAL, _GOLD, "前 k 个里命中的 gold 占比", per_k=True),
    _definition("ndcg", FAMILY_RETRIEVAL, _GOLD, "二元相关性下的 nDCG", per_k=True),
    _definition("hit", FAMILY_RETRIEVAL, _GOLD, "前 k 个里至少命中一个 gold", per_k=True),
    _definition(
        "full_coverage",
        FAMILY_RETRIEVAL,
        _GOLD,
        "前 k 个里凑齐全部 gold 才算 1，比 recall 均值更贴近多跳的需求",
        per_k=True,
    ),
    _definition("mrr", FAMILY_RETRIEVAL, _GOLD, "首个 gold 的倒数排名"),
    # --- 答案质量 ---
    _definition(
        "em",
        FAMILY_QA,
        _ANSWERS,
        "Exact Match：归一化后整串相等。解释性长答案得分偏低，需结合 F1 与证据解读",
    ),
    _definition(
        "f1",
        FAMILY_QA,
        _ANSWERS,
        "token F1。会被解释性 token 稀释，绝对值只可同配置比较",
    ),
    # --- 引用归因 ---
    _definition("citation_precision", FAMILY_ATTRIBUTION, _GOLD, "被引文档里 gold 的占比"),
    _definition("citation_recall", FAMILY_ATTRIBUTION, _GOLD, "gold 里被引用的占比"),
    _definition(
        "truncation_loss", FAMILY_ATTRIBUTION, _GOLD, "召回但未被引用的篇数", higher_is_better=False
    ),
    _definition(
        "truncated_gold",
        FAMILY_ATTRIBUTION,
        _GOLD,
        "被截断掉的 gold 篇数",
        higher_is_better=False,
    ),
    # --- 多跳 ---
    _definition(
        "graph_exclusive_gold_share",
        FAMILY_MULTIHOP,
        _GOLD,
        "只靠图扩展才能到达的 gold 占比，即图边的净增量价值",
    ),
    _definition(
        "graph_neighbor_precision",
        FAMILY_MULTIHOP,
        _GOLD,
        "按文档去重后，图扩展命中的文档中 gold 所占比例",
    ),
    # --- 诊断计数：读其他指标时的分母与背景，进 registry 以便 UI 拿到方向声明 ---
    _definition(
        "graph_neighbor_gold_snippets", FAMILY_MULTIHOP, _GOLD, "图扩展 snippet 里命中 gold 的条数"
    ),
    # --- judge：除 answer_correctness 外 requires 都是空集，所以四组都成立 ---
    _definition(
        "faithfulness",
        FAMILY_JUDGE,
        _NONE,
        "答案里的每条陈述能否由检索到的上下文支撑。不需要标注，narrativeqa 也可用",
        kind=KIND_JUDGE,
    ),
    _definition(
        "answer_relevancy",
        FAMILY_JUDGE,
        _NONE,
        "答案有多少句在回答这个问题，判「答没答到点上」而非「答得对不对」",
        kind=KIND_JUDGE,
    ),
    _definition(
        "context_relevancy",
        FAMILY_JUDGE,
        _NONE,
        "召回的上下文里有多少是这个问题用得上的，即检索的信噪比",
        kind=KIND_JUDGE,
    ),
    _definition(
        "answer_correctness",
        FAMILY_JUDGE,
        _ANSWERS,
        "答案与参考答案在事实上是否一致，不受措辞与长度影响。拒答无定义，不记 0",
        kind=KIND_JUDGE,
    ),
)

METRIC_REGISTRY: dict[str, MetricDefinition] = {d.name: d for d in METRIC_DEFINITIONS}


def get_metric(name: str) -> MetricDefinition:
    """按名字取指标定义。带 ``@k`` 的名字会先剥掉 k。"""
    base = name.split("@", 1)[0]
    try:
        return METRIC_REGISTRY[base]
    except KeyError:
        raise KeyError(
            f"unknown metric {name!r}. Known: {', '.join(sorted(METRIC_REGISTRY))}"
        ) from None


def available(provides: frozenset[DataDependency]) -> list[MetricDefinition]:
    """给定数据集拥有的标注，返回能算的全部指标。"""
    return [d for d in METRIC_DEFINITIONS if d.requires <= provides]


def omitted(provides: frozenset[DataDependency]) -> list[MetricDefinition]:
    """算不了的那些。报告连同原因一起写出来，不伪造 0 分。"""
    return [d for d in METRIC_DEFINITIONS if not d.requires <= provides]


def require(dataset: str, provides: frozenset[DataDependency], metric: str) -> MetricDefinition:
    """闸门：数据集缺依赖就抛 :class:`DependencyError`，不返回 0.0。"""
    definition = get_metric(metric)
    missing = definition.requires - provides
    if missing:
        raise DependencyError(
            f"{dataset} cannot compute {metric!r}: missing data dependency "
            f"{sorted(m.value for m in missing)}"
        )
    return definition
