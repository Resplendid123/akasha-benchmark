"""跨路由共用的小常量。

单独一个模块是为了避免路由之间互相 import —— 那种依赖很快会绕成环。
"""

from __future__ import annotations

# Akasha 的四项模型配置。取值与 ingest.MODEL_FEATURES 一致，
# 在这里重述是为了让路由层不必 import 阶段代码。
MODEL_FEATURES = ("compiler", "embedding", "answer", "image")

# 归因层与报告层共用的默认页大小。样本明细体积不小，不设上限会让一次请求
# 拖回整层的响应体。
DEFAULT_PAGE = 50
MAX_PAGE = 500
