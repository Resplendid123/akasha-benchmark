"""judge 指标：需要模型才能算的那些。

judge **是指标**，不是新的一层（PLAN.md §12 决策 9）—— 与 F1/recall 同层，
产物是分数、进指标层、参与汇总。

别和另外两件事合在一起（§12.5）：LLM 归因产出结构化标签、进标注表、不参与汇总;
人工标注同上，只差作者列。三者的产物、去向、汇总语义都不同。

``run`` 是 ``python -m`` 的入口，刻意不在这里导入。
"""

from . import faithfulness
from .client import (
    FAILURE_PARSE,
    FAILURE_RATE_LIMIT,
    FAILURE_REFUSAL,
    FAILURE_TIMEOUT,
    JudgeClient,
    JudgeConfigError,
    JudgeProvider,
    parse_json_object,
)

__all__ = [
    "FAILURE_PARSE",
    "FAILURE_RATE_LIMIT",
    "FAILURE_REFUSAL",
    "FAILURE_TIMEOUT",
    "JudgeClient",
    "JudgeConfigError",
    "JudgeProvider",
    "faithfulness",
    "parse_json_object",
]
