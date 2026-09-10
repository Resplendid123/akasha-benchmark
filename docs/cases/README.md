# 案例库

逐样本追到根因的排查记录。每份案例回答同一个问题：**这个数字为什么是这样，
以及它能不能调好。**

写案例的门槛是「结论有库里的证据支撑」，不是「看起来像这么回事」。
指标算错了不会报错，只会给出一个看着合理的假结果 —— 所以这里的每条结论
都要能指到具体的表、具体的行。

| 案例 | 一句话 | 结论类型 |
| --- | --- | --- |
| [召回漏一篇 gold：修饰语被编译掉了](recall-miss-compiled-away.md) | 原文里能逐字命中查询的短语，在编译产物里消失了 | 非调参可解 |

## 排查用的脚本

`scripts/` 下的探查脚本都是**只读 SQL**，可重跑：

| 脚本 | 用途 |
| --- | --- |
| [probe_lineage.py](scripts/probe_lineage.py) | 单篇原文 → artifact → chunk → 图边 → 原文块，走完整条血缘 |
| [probe_extraction.py](scripts/probe_extraction.py) | 跨全库统计：抽取密度、图边稀疏度、relation 词表、压缩率 |

跑之前需要 `akasha.config.json` 里配好 `database_url` 与 `workspace_id`，
并装 `psycopg`（`uv add 'psycopg[binary]'`）。

```bash
uv run python docs/cases/scripts/probe_lineage.py
uv run python docs/cases/scripts/probe_extraction.py
```

这两个脚本走的链路就是平台「血缘视图」要实现的东西（PLAN.md §12.9）——
现在靠手写 SQL，实现完之后应该一屏走完。
