"""Akasha 的多跳检索、引用归因与答案质量评测。

每个阶段是一个带 CLI 的模块。归一化、子集、评测纯本地；入库和查询需要 Akasha 在线。

    python -m akasha_benchmark.normalize      # 四组归一化
    python -m akasha_benchmark.subset         # 抽可独立评测的子集
    python -m akasha_benchmark.ingest         # 入库 Akasha 并编译
    python -m akasha_benchmark.run_queries    # 逐条跑 query，存完整响应
    python -m akasha_benchmark.evaluate       # 离线算指标
    python -m akasha_benchmark.audit_join     # 分层归因（需要 psycopg）
"""

__all__ = ["STAGES", "main"]

# 顺序即执行顺序：上游产物是下游的输入。
STAGES: tuple[tuple[str, str], ...] = (
    ("normalize", "四组归一化（全量，不抽样）"),
    ("subset", "在一个 run_id 下抽可独立评测的子集"),
    ("ingest", "导入 Akasha、编译、校验入库完整性"),
    ("run_queries", "逐条跑 query，完整响应落盘"),
    ("evaluate", "从落盘响应算指标，纯离线"),
    ("audit_join", "从 knowledge_query_audit 做分层归因"),
)


def main() -> None:
    """`akasha-benchmark` 命令的入口，只打印各阶段用法。"""
    print(__doc__.split("\n")[0])
    print("\nStages:")
    for module, description in STAGES:
        print(f"  python -m akasha_benchmark.{module:<12} {description}")
    print("\n每个模块都支持 --help。详见 README.md。")
