# 案例库

逐样本追到根因的排查记录。

每条结论都要能指到具体的表、具体的行。

| 案例 | 一句话 | 结论类型 |
| --- | --- | --- |
| [召回漏一篇 gold：修饰语被编译掉了](recall-miss-compiled-away.md) | 原文里能逐字命中查询的短语，在编译产物里消失了 | 非调参可解 |

## 排查用的脚本

| 脚本 | 用途 |
| --- | --- |
| [probe_lineage.py](scripts/probe_lineage.py) | 单篇原文 → artifact → chunk → 图边 → 原文块，走完整条血缘 |
| [probe_extraction.py](scripts/probe_extraction.py) | 跨全库统计：抽取密度、图边稀疏度、relation 词表、压缩率 |

```bash
uv run python docs/cases/scripts/probe_lineage.py
uv run python docs/cases/scripts/probe_extraction.py
```
