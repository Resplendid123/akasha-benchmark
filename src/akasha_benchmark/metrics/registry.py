from __future__ import annotations

from dataclasses import dataclass

from ..datasets.models import DataDependency, DependencyError


FAMILY_RETRIEVAL = "retrieval"
FAMILY_QA = "qa"
FAMILY_ATTRIBUTION = "attribution"
FAMILY_MULTIHOP = "multihop"
FAMILY_JUDGE = "judge"

KIND_DETERMINISTIC = "deterministic"
KIND_JUDGE = "judge"


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    family: str
    requires: frozenset[DataDependency]
    kind: str
    higher_is_better: bool
    description: str

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

    _definition("precision", FAMILY_RETRIEVAL, _GOLD, "前 k 个实际返回文档中命中的 gold 占比", per_k=True),
    _definition("recall", FAMILY_RETRIEVAL, _GOLD, "前 k 个里命中的 gold 占比", per_k=True),
    _definition(
        "retrieval_f1",
        FAMILY_RETRIEVAL,
        _GOLD,
        "set F1：对系统实际返回的全部文档算一次，多返回和漏召都扣分，不看顺序也不随 k 变化",
    ),
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
    _definition("citation_precision", FAMILY_ATTRIBUTION, _GOLD, "被引文档里 gold 的占比"),
    _definition("citation_recall", FAMILY_ATTRIBUTION, _GOLD, "gold 里被引用的占比"),
    _definition(
        "uncited_count", FAMILY_ATTRIBUTION, _GOLD, "已检索但未被引用的文档数", higher_is_better=False
    ),
    _definition(
        "uncited_gold_count",
        FAMILY_ATTRIBUTION,
        _GOLD,
        "已检索但未被引用的 gold 文档数",
        higher_is_better=False,
    ),
    _definition(
        "graph_exclusive_gold_share",
        FAMILY_MULTIHOP,
        _GOLD,
        "有 knowledge page 只靠图扩展才到达的 gold 占比，即图边的净增量价值",
    ),
    _definition(
        "graph_neighbor_precision",
        FAMILY_MULTIHOP,
        _GOLD,
        "按文档去重后，图扩展命中的文档中 gold 所占比例",
    ),

    _definition(
        "graph_neighbor_gold_snippets", FAMILY_MULTIHOP, _GOLD, "图扩展 snippet 里命中 gold 的条数"
    ),
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

    base = name.split("@", 1)[0]
    try:
        return METRIC_REGISTRY[base]
    except KeyError:
        raise KeyError(
            f"unknown metric {name!r}. Known: {', '.join(sorted(METRIC_REGISTRY))}"
        ) from None


def available(provides: frozenset[DataDependency]) -> list[MetricDefinition]:

    return [d for d in METRIC_DEFINITIONS if d.requires <= provides]


def omitted(provides: frozenset[DataDependency]) -> list[MetricDefinition]:

    return [d for d in METRIC_DEFINITIONS if not d.requires <= provides]


def require(dataset: str, provides: frozenset[DataDependency], metric: str) -> MetricDefinition:

    definition = get_metric(metric)
    missing = definition.requires - provides
    if missing:
        raise DependencyError(
            f"{dataset} cannot compute {metric!r}: missing data dependency "
            f"{sorted(m.value for m in missing)}"
        )
    return definition
