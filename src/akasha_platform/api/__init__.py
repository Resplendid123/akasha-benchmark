"""HTTP 路由，按左栏那八层分文件。

原先是一个 532 行的单文件。加了配置层、归因层与各层的样本查看之后会过千，
所以拆开 —— 每个文件对应界面上的一层，改一层时看一个文件。

**写入口只有三类**：起任务、改配置、加标注（含归因写回）。阶段计算一律不在
请求里跑，那是 :mod:`..tasks` 的事 —— 15 小时的 ingest 不能挂在一个 HTTP
连接上。

配置全部在 :mod:`.connections` 一处：Akasha 连接、它那边的四项模型配置、
judge 与归因分析端点。原先编译模型配置在编译层，那让「改一个模型要去哪」
取决于它属于哪一层 —— 而用户想的是「我要改配置」。

注册顺序有一处讲究：``annotations`` 里 ``/annotations/agreement`` 必须在
``/annotations/{level}/{target_id}`` 之前声明，否则 ``agreement`` 会被当成
一个 ``level`` 吃掉。同理 ``datasets`` 里 ``/metrics/definitions`` 与
``/metrics/available`` 都是固定路径，不与任何通配冲突。
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from ._common import settings_of
from .annotations import router as annotations_router
from .badcase import router as badcase_router
from .connections import router as connections_router
from .datasets import router as datasets_router
from .layers import router as layers_router
from .lineage import router as lineage_router
from .samples import router as samples_router
from .tasks import router as tasks_router

router = APIRouter()


@router.get("/api/health")
def health(request: Request) -> dict[str, object]:
    return {"ok": True, "settings": settings_of(request).redacted()}


for sub in (
    connections_router,
    datasets_router,
    layers_router,
    samples_router,
    lineage_router,
    badcase_router,
    tasks_router,
    annotations_router,
):
    router.include_router(sub)

__all__ = ["router"]
