"""数据层：SQLite 是评测端的事实来源（PLAN.md §12 决策 2、3）。

方向单向：**SQLite 可写权威，Akasha 的 Postgres 只读外来**。

    from akasha_benchmark.store import connect, repo

    with connect() as connection:
        layers = repo.list_index_layers(connection)
"""

from . import identity, repo
from .db import DEFAULT_DB_PATH, Batcher, batched, connect, query_all, query_one, scalar, transaction

# migrate 与 reindex 刻意不在这里导入：两者都是 ``python -m`` 的入口，
# 在包 __init__ 里先导入一遍会让 runpy 报「found in sys.modules after import of
# package」的警告。需要时 ``from akasha_benchmark.store.migrate import migrate``。

__all__ = [
    "DEFAULT_DB_PATH",
    "Batcher",
    "batched",
    "connect",
    "identity",
    "query_all",
    "query_one",
    "repo",
    "scalar",
    "transaction",
]
