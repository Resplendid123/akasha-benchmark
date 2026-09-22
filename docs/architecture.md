# 架构说明

Akasha-Benchmark 是一个本地双进程应用：React/Vite 提供界面，FastAPI 提供 API 并在线程中运行长任务。SQLite 保存本地事实，Akasha 负责语料编译与查询；Judge、模型归因和 PostgreSQL 血缘读取均为按需使用的外部能力。

[打开交互式项目架构图](diagrams/project-architecture.html)

```text
React / Vite (:5173)
        │ /api
        ▼
FastAPI (:8848) ── TaskRunner ── 阶段任务
        │                         │
        └──────── SQLite ◀────────┤
                                  ├── Akasha HTTP：编译、查询、模型配置
                                  ├── 模型 HTTP：Judge、整轮归因报告
                                  └── PostgreSQL（只读、可选）：编译血缘
```

## 模块边界

| 模块 | 职责 | 位置 |
| --- | --- | --- |
| Web | 页面状态、表单、结果展示和任务操作 | `web/src/` |
| API | 参数校验、配置读写、任务入口和结果查询 | `src/akasha_platform/api/` |
| 任务运行器 | 阶段互斥、线程调度、暂停续跑和重启恢复 | `src/akasha_platform/tasks.py` |
| 运行树 | 批量组合编译、查询、评测、归因及其任务 | `src/akasha_platform/run_tree.py` |
| 阶段 | 下载、归一化、编译、查询、评测和归因 | `src/akasha_benchmark/stages/` |
| 数据集适配器 | 将不同来源统一为样本和语料模型 | `src/akasha_benchmark/datasets/` |
| 指标与 Judge | 确定性指标定义、依赖检查和模型判定 | `src/akasha_benchmark/metrics/`、`src/akasha_benchmark/judge/` |
| 存储 | SQLite schema、连接和按领域拆分的 SQL | `src/akasha_benchmark/store/` |
| 外部集成 | Akasha HTTP 与只读编译血缘查询 | `src/akasha_benchmark/akasha_client.py`、`src/akasha_benchmark/lineage.py` |

Web 中的指标解释和样本证据链分别由 `MetricInterpretations.tsx` 与 `AttributionChain.tsx` 展示；它们是结果诊断组件，不负责任务编排。FastAPI 不托管前端构建产物，开发时由 Vite 代理 `/api`，生产部署需单独托管 `web/dist/`。

## 流水线

```text
原始 JSON
   │ download
   ▼
dataset/ ── normalize ──▶ sample + corpus_doc
                              │
                              ▼
compile_run ──▶ query_run ──▶ eval_run ──▶ attribution_run
```

1. `download` 将数据集文件下载或整理到 `dataset/`。
2. `normalize` 通过适配器生成统一的 `sample` 和 `corpus_doc`，只写 SQLite。
3. `compile` 抽取样本和语料，创建独立 Akasha 空间，导入并编译文档，同时保存抽样参数、空间信息和远端模型快照。
4. `query` 在指定编译空间逐样本调用 Akasha，保存 HTTP 状态、耗时和完整响应。
5. `evaluate` 将 `sourcePageId` 映射回语料文档，计算可用指标和可选 Judge，并保存逐样本结果与汇总。
6. `attribute` 基于指标与血缘证据给出逐样本规则根因；配置模型时，还可为整轮评测生成一份分析报告。

编译子集策略由数据集适配器声明：有 gold 映射的数据集按 QA 选择语料，`musique` 额外按跳数分层，`narrativeqa` 按完整文档取样，`itfaq` 使用全量语料。指标同样声明所需标注，数据集不满足依赖时直接省略该指标。

## 数据模型

核心运行关系为：

```text
compile_run 1 ── * query_run 1 ── * eval_run 1 ── * attribution_run
```

因此任务页展示的是分叉树，而不是固定的一条四阶段链。各运行表拥有自己的样本、响应、指标或归因明细；外键使用 `ON DELETE CASCADE` 清理下游本地数据。`task` 独立保存阶段、冻结参数、进度和绑定的运行产物，`audit_log` 只追加。

配置分为两类：`akasha_connection` 保存 Akasha 与可选 PostgreSQL 连接，`model_provider` 保存 Judge、归因及可应用到 Akasha 的模型端点。数据库以 `store/schema.sql` 为最终结构，不包含迁移层。

## 任务语义与约束

- 一个任务线程持有一条独立 SQLite 连接；同阶段最多运行一个任务，`download` 和 `normalize` 与所有其他活动任务互斥。
- 阶段在已落库的边界检查暂停信号。继续任务复用冻结参数与原产物 ID：查询跳过已有成功响应，Judge 和归因跳过已有结果。
- 编译暂停先取消远端 exact run；继续时根据累积 RunPage 状态，仅通过 `retry-pages` 重试失败、合并失败或手动取消的页面。
- 后端启动时处理上次进程遗留的活动任务；远端编译确认停止后标记暂停，无法确认时标记失败。
- 编译要求 Akasha owner 权限。编译和查询在开始、结束及续跑时校验 workspace 与模型快照；embedding 与原编译不一致时拒绝查询。运行任务只读取远端模型配置，不隐式修改它。
- 编译完整性由 `missingChunk`、`missingEmbedding`、`missingSource` 和 `stalePageCount` 判定。明确存在成功页面但整体失败的编译可降级用于查询，并标记结果不完整。
- PostgreSQL 仅用于读取编译血缘。不可用时，归因跳过依赖血缘的 `compiled_away` 判据，其他规则照常执行。

当前调度器只协调单个 FastAPI 进程；若部署多个后端实例，需要额外的跨进程任务锁和队列。
