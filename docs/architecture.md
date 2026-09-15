# 代码架构

前端通过 FastAPI 发起操作，后端在线程中执行各阶段任务。

| 模块 | 职责 | 位置 |
| --- | --- | --- |
| 前端页面 | 九个视图，各层的表单、任务操作与结果展示 | `web/src/views/` |
| 前端公共代码 | HTTP 请求、响应类型、异步状态与基础控件 | `web/src/api.ts`、`web/src/types.ts`、`web/src/ui.tsx` |
| 后端路由 | 请求处理、配置读写、结果查询 | `src/akasha_platform/api/` |
| 任务运行器 | 线程调度、启动互斥、暂停/继续、重启恢复 | `src/akasha_platform/tasks.py` |
| 任务执行接口 | 实际参数保存、产物绑定、阶段执行与最终状态 | `src/akasha_benchmark/task.py` |
| 阶段 | 六层流水线加链路测试 | `src/akasha_benchmark/stages/` |
| 存储 | SQLite 连接、schema、按数据职责分的读写 | `src/akasha_benchmark/store/` |
| 外部客户端 | Akasha HTTP、模型 HTTP、只读 PostgreSQL | `src/akasha_benchmark/akasha_client.py`、`src/akasha_benchmark/judge/client.py`、`src/akasha_benchmark/lineage.py` |

## 分层与数据流

```
数据集 → 归一化 → 编译 → 查询 → 评测 → 归因
```

1. **数据集层**：原始 JSON 下载到 `dataset/`，校验能否解析成 JSON 数组。
2. **归一化层**：经适配器转成 `sample` / `corpus_doc` 进 SQLite，**不写 Akasha**。写库后验收 gold 是否都在语料内。
3. **编译层**：按数据集抽子集，创建 Akasha 空间，导入语料并编译，检查编译完整性。`compile_run` 记录本次抽样配置与模型快照。抽子集走哪条路由适配器的 `subset_strategy` 声明：有 gold 的先 QA 后 corpus（musique 按跳数分层），narrativeqa 整篇取文档，itfaq 没有「问题→文档」映射所以语料全量导入。
4. **查询层**：在指定编译的空间中逐条查询，保存完整响应体。
5. **评测层**：从响应算指标，需要时执行 Judge。确定性指标分别汇总为 `overall` 与 `knowledge_only`。
6. **归因层**：规则判据从指标与链路推根因，模型可选地补一段因果叙述。

原始文件保存在 `dataset/`；归一化后，各阶段从 SQLite 读取数据与运行记录。评测通过 `page_id` 将响应中的 `sourcePageId` 映射回语料文档，归因用它查询编译产物。

## 任务：暂停、继续、清理

阶段在后端进程里执行，一个任务一条线程。`execute()` 统一处理执行与链路校验，
`task_store.transition()` 在同一事务内更新任务及绑定产物的状态。阶段只负责参数解析、产物绑定和计算。
任务启动与续跑通过 SQLite 写事务完成互斥检查和占位；运行器用于单个后端进程。

- **暂停**是协作式的：阶段在每个可续跑的边界调 `ctx.checkpoint()`，所以暂停总是停在一个已落库的位置。
- **新建**创建独立任务与产物，名称重复时报错。
- **继续**使用原任务保存的实际参数与产物 ID，由各阶段决定重算或跳过：编译跳过已导入的文档，查询跳过已有响应的样本，确定性评测按数据集重算，Judge 按指标跳过已有判定，归因跳过已有结果的样本。
- **清理运行记录**：删除编译、查询、评测或归因主表记录时，通过 `ON DELETE CASCADE` 删除关联产物及下游记录。删除前检查当前记录、全部下游及排队任务的输入引用；检查和删除处于同一写事务。清理任务仅删除任务记录，远端 Akasha 空间保留。
- **审计日志**（`audit_log`）只追加，清理任务记录不删它。

后端启动时将上次进程遗留的排队、运行中任务及其绑定产物同步标记为暂停，由用户继续执行。

存储按 `compile_store`、`query_store`、`eval_store`、`attribution_store` 分文件，直接执行 SQL。
`run_store` 只维护共享状态与上下游依赖；模型端点解析集中在 `judge/providers.py`。

数据库以 `store/schema.sql` 定义最终结构，不提供迁移或旧结构兼容分支。
`compile_sample`、`query_sample` 只存运行与样本的关联，数据集名从 `sample` 读取；
`sample_metric` 通过复合外键归属 `sample_eval`，数据集名从评测样本读取，删除评测样本时自动清理指标。
查询的回答模式从完整 `response_json` 解析，不重复存列。
查询的问题文本、评测明细、运行模型配置保留为执行时快照；汇总表保存各次评测的结果与有效样本数。

## 执行前检查与报告口径

- **账号权限**：编译前要求账号角色为 owner。
- **模型配置**：查询前比较当前配置与编译快照，embedding 不一致时拒绝执行；compiler、answer、image 不一致时记录警告。
- **Workspace**：编译续跑和查询前，将 `users/me` 返回的 workspace 与 `compile_run.workspace_id` 比对，不一致时拒绝执行。连接测试也提供预检。
- **编译完整性**：`missingChunk`、`missingEmbedding`、`missingSource`、`stalePageCount` 四项计数必须全部取得且为 0。
- **回答模式**：确定性指标分别汇总为 `overall` 与 `knowledge_only`。均值差反映样本范围的影响，不能直接解释为拒答比例或根因。
- **指标依赖**：缺少所需标注的指标省略并记录原因，不计为 0 分。
- **Judge**：跳过和失败条目不计入评分均值；单项失败率超过 10% 时，评测任务标记为失败。
- **样本一致性**：评测按样本 ID 匹配后，再检查响应中保存的问题文本是否与编译子集一致。

## 配置

Akasha 连接与模型端点都在 `akasha_connection` / `model_provider` 两张表，由配置页读写。配置页在线读写 Akasha 部署的 compiler、embedding、answer、image 模型配置；编译时把它们的快照固化在 `compile_run` 上，供查询前比对。

`database_url`（只读 PostgreSQL）用于链路查询与归因证据读取。未配置或连接不可用时，自动归因跳过 `compiled_away` 判据，继续执行其他规则。
