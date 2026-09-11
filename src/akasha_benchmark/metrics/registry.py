"""指标 registry：**指标声明依赖，数据集声明拥有，闸门做集合比对**。

这是 PLAN.md §12.4 决策 10 的落点。反转之前，判据是数据集「支持不支持某个指标」;
反转之后，判据是数据集「有没有这个指标要的那种标注」。

差别在新增指标时才看得出来。faithfulness 不需要任何标注，对四组都成立 ——
旧口径下没法声明它（声明「narrativeqa 支持 faithfulness」是句废话，因为
支持与否根本不取决于数据集），新口径下它的 ``requires`` 就是空集，
闸门自然放行。

**一个具体收获**：narrativeqa 现在整组检索指标省略（无 gold 文档），
而 faithfulness / context precision 不需要 gold 就能算。它恰恰最需要 ——
46% 的参考答案措辞在原文里根本不存在，F1 绝对值在这组上信息量最低。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..datasets.models import DataDependency, DependencyError

# 指标族。检索/引用/多跳三族都依赖 gold 文档，qa 依赖参考答案，judge 无依赖。
FAMILY_RETRIEVAL = "retrieval"
FAMILY_QA = "qa"
FAMILY_ATTRIBUTION = "attribution"
FAMILY_MULTIHOP = "multihop"
FAMILY_JUDGE = "judge"

KIND_DETERMINISTIC = "deterministic"
KIND_JUDGE = "judge"


@dataclass(frozen=True)
class MetricDefinition:
    """一个指标的身份与它的数据依赖。

    ``name`` 里的 ``@k`` 由 :func:`expand` 按实际的 k 展开，registry 里存的是
    不带 k 的模板名（``recall`` 而不是 ``recall@10``）。
    """

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
        "前 k 个里凑齐全部 gold 才算 1。多跳少一跳就答不对，所以它比 recall 均值"
        "更贴近多跳的实际需求：recall 在 2 篇 gold 上只有 0/0.5/1 三个取值",
        per_k=True,
    ),
    _definition("mrr", FAMILY_RETRIEVAL, _GOLD, "首个 gold 的倒数排名"),
    # --- 答案质量 ---
    _definition(
        "em",
        FAMILY_QA,
        _ANSWERS,
        "Exact Match。**在这套架构上预期恒为 0**：Akasha 返回解释性散文，参考答案"
        "是短跨度，整串相等不可能成立。当答案**形态**的探针读，不当质量指标读",
    ),
    _definition(
        "f1",
        FAMILY_QA,
        _ANSWERS,
        "token F1。被解释性 token 稀释（分母是 20-40 个散文 token，参考答案只有"
        " 1-5 个），绝对值不可与公开数字比，只可同配置比较",
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
        "被截断掉的 gold 篇数。标了 citation_dropped 的样本这一项应 > 0，"
        "不一致说明判断或指标有一个错了（§12.5 的交叉验证）",
        higher_is_better=False,
    ),
    _definition(
        "evidence_verifiable_rate", FAMILY_ATTRIBUTION, _GOLD, "有证据窗口可核对的引用占比"
    ),
    # --- 多跳 ---
    _definition("graph_neighbor_share", FAMILY_MULTIHOP, _GOLD, "图扩展产出的 snippet 占比"),
    _definition("graph_neighbor_precision", FAMILY_MULTIHOP, _GOLD, "图扩展 snippet 的 gold 命中率"),
    _definition(
        "graph_exclusive_gold_share",
        FAMILY_MULTIHOP,
        _GOLD,
        "只靠图扩展才能到达的 gold 占比，即图边的净增量价值",
    ),
    # --- 诊断计数：不是「越高越好」的分数，是读其他指标时的分母与背景 ---
    #
    # 它们照样进 registry，因为进了 registry 才有描述与方向声明可以给 UI 用;
    # 不进的话前端只能拿到一个裸数字，而 truncation_loss 这种「越低越好」
    # 的项会被默认当成越高越好来排序。
    _definition(
        "retrieved_count", FAMILY_RETRIEVAL, _GOLD, "召回条数，检索精确率的分母", higher_is_better=True
    ),
    _definition(
        "citation_count", FAMILY_ATTRIBUTION, _GOLD, "被引条数，引用精确率的分母"
    ),
    _definition(
        "evidence_entries", FAMILY_ATTRIBUTION, _GOLD, "带证据窗口的引用条数"
    ),
    _definition("snippet_count", FAMILY_MULTIHOP, _GOLD, "snippet 条数"),
    _definition(
        "graph_neighbor_snippets", FAMILY_MULTIHOP, _GOLD, "图扩展产出的 snippet 条数"
    ),
    _definition(
        "graph_neighbor_gold_snippets", FAMILY_MULTIHOP, _GOLD, "图扩展 snippet 里命中 gold 的条数"
    ),
    _definition(
        "graph_exclusive_gold_count", FAMILY_MULTIHOP, _GOLD, "只靠图扩展才拿到的 gold 篇数"
    ),
    # --- judge：requires 是空集，所以四组都成立 ---
    _definition(
        "faithfulness",
        FAMILY_JUDGE,
        _NONE,
        "答案里的每条陈述能否由检索到的上下文支撑。不需要任何标注，所以它是"
        " narrativeqa 唯一可用的质量指标",
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


def expand(name: str, ks: tuple[int, ...]) -> list[str]:
    """把模板名展开成实际的指标名。``recall`` -> ``recall@2``、``recall@5`` ..."""
    definition = get_metric(name)
    return [f"{definition.name}@{k}" for k in ks] if definition.per_k else [definition.name]


def available(provides: frozenset[DataDependency]) -> list[MetricDefinition]:
    """给定数据集拥有的标注，返回能算的全部指标。"""
    return [d for d in METRIC_DEFINITIONS if d.requires <= provides]


def omitted(provides: frozenset[DataDependency]) -> list[MetricDefinition]:
    """算不了的那些。报告要把它们连同原因一起写出来，而不是伪造 0 分。"""
    return [d for d in METRIC_DEFINITIONS if not d.requires <= provides]


def require(dataset: str, provides: frozenset[DataDependency], metric: str) -> MetricDefinition:
    """闸门：数据集缺依赖就抛异常，**不返回 0.0**。

    返回 0.0 的后果是一个看着合理的假分数进了汇总，而且没有任何地方报错 ——
    等发现的时候，那个数字已经被引用过好几次了。
    """
    definition = get_metric(metric)
    missing = definition.requires - provides
    if missing:
        raise DependencyError(
            f"{dataset} cannot compute {metric!r}: missing data dependency "
            f"{sorted(m.value for m in missing)}. Reporting 0.0 instead would silently "
            "pollute every aggregate that includes it."
        )
    return definition


def as_rows() -> list[dict[str, object]]:
    """供 ``metric_definition`` 表同步用。计算依据始终是这里的代码，不是那张表。"""
    return [
        {
            "name": d.name,
            "family": d.family,
            "requires_json": sorted(r.value for r in d.requires),
            "kind": d.kind,
            "higher_is_better": int(d.higher_is_better),
            "description": d.description,
        }
        for d in METRIC_DEFINITIONS
    ]
