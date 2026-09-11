"""评测平台的 FastAPI 后端。

Akasha 侧仍然只通过 HTTP（加一条只读 SQL），不修改 Akasha 主仓库 ——
这条不变（PLAN.md §12）。

``main`` 是 ``python -m`` 与 ``akasha-platform`` 的入口，刻意不在这里导入。
"""

from .settings import Settings, load_settings

__all__ = ["Settings", "load_settings"]
