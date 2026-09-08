"""Akasha 的多跳检索、引用归因与答案质量评测。

五轮，每轮是一个带 CLI 的模块。轮次 1、2、5 纯本地；3、4 需要 Akasha 在线。

    python -m akasha_benchmark.normalize      # 1 四组归一化
    python -m akasha_benchmark.subset         # 2 抽可独立评测的子集
    python -m akasha_benchmark.ingest         # 3 入库 Akasha 并编译
    python -m akasha_benchmark.run_queries    # 4 逐条跑 query，存完整响应
    python -m akasha_benchmark.evaluate       # 5 离线算指标
    python -m akasha_benchmark.audit_join     # 5b 分层归因（需要 psycopg）
"""

__all__ = ["ROUNDS", "main"]

ROUNDS: tuple[tuple[str, str, str], ...] = (
    ("1", "normalize", "四组归一化（全量，不抽样）"),
    ("2", "subset", "在一个 run_id 下抽可独立评测的子集"),
    ("3", "ingest", "导入 Akasha、编译、校验入库完整性"),
    ("4", "run_queries", "逐条跑 query，完整响应落盘"),
    ("5", "evaluate", "从落盘响应算指标，纯离线"),
    ("5b", "audit_join", "从 knowledge_query_audit 做分层归因"),
)


def main() -> None:
    """`akasha-benchmark` 命令的入口，只打印各轮用法。"""
    print(__doc__.split("\n")[0])
    print("\nRounds:")
    for number, module, description in ROUNDS:
        print(f"  {number:<3} python -m akasha_benchmark.{module:<12} {description}")
    print("\n每个模块都支持 --help。详见 README.md 和 PLAN.md。")
