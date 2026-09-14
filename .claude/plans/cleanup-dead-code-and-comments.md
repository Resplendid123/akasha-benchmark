# 清理冗余代码与注释

基线：133 个测试全过。每一步改完重跑 `uv run pytest -q` 与 `npm --prefix web run typecheck`。

## 一、删死代码（全仓库零调用，已逐个 grep 确认）

### akasha_benchmark

| 文件 | 删除项 | 依据 |
| --- | --- | --- |
| `metrics/retrieval.py` | `aggregate()`、`require_gold_docs()`、`hit_at_k` 保留（本文件用） | 汇总改由 `stages/evaluate._aggregate` 做；`require_gold_docs` 是 `registry.require` 的薄封装，只有测试调 |
| `metrics/attribution.py` | `aggregate()` | 同上 |
| `metrics/multihop.py` | `aggregate()`、`stratify()`、`KNOWN_REASONS` | `stratify` 无调用方，`KNOWN_REASONS` 只有定义 |
| `metrics/registry.py` | `as_rows()`、`expand()` | `as_rows` 供的 `metric_definition` 表在 schema.sql 里不存在；`expand` 与 `retrieval.evaluate_sample` 里的 `f"recall@{k}"` 重复，只有测试调 |
| `datasets/common.py` | `join_context_sentences()` | 注释自称「供原文基线用」，无基线代码 |
| `datasets/corpus.py` | `dedup_stats()`、`unique_title_count` | manifest 已不存在 |
| `datasets/narrativeqa.py` | `document_id_of()` | 抽子集处直接 `rsplit`，没走它 |
| `io_utils.py` | `sha256_text()` | 只有 `sha256_file` 在用 |
| `lineage.py` | `chunk_kinds()` + `CHUNK_KINDS` 常量、`artifacts()` | `artifacts()` 是 `lineage()` 的子集；两者都无调用方 |
| `stages/download.py` | `local_files()` | `file_status()` 自己遍历适配器 |
| `task.py` | `TaskContext.derive()` | 链路测试改成四条独立任务后不再需要 |
| `akasha_client.py` | `TERMINAL_RUN_STATUSES` | 轮询只用 `ACTIVE_RUN_STATUSES` |
| `judge/*.py` | 4 个 `PROMPT_VERSION` | 四个模块各定义一次，无人读 |

### akasha_platform

- `tasks.py`：删 `self._threads` —— 只写不读（存进去、pop 出来，没有任何地方遍历它）。
- 删两个前端从未调用的路由（用户选择「前后端一起删」）：
  - `api/runs.py`：`compile_detail`（`GET /compiles/{id}`）、`compile_samples`（`GET /compiles/{id}/samples`）、`query_detail`（`GET /queries/{id}`）
  - `api/results.py`：`eval_samples`（`GET /evals/{id}/samples`）、`worst_samples`（`GET /evals/{id}/worst`）
  - 连带：`results.py` 的 `Query`、`registry` 两个 import 变成未使用，一起删。`runs.py` 的 `Query`/`loads` 仍被其他路由用，保留。
  - **不删** `sample_detail`（`GET /evals/{id}/samples/{sample_id}`，Attribution.tsx 在用）、`compile_docs`、`query_response`、全部 DELETE 路由。
  - store 层无连带删除：`compile_stats` / `sample_evals` / `samples_ranked_by` 等都还被 stages 或列表路由调用。

### web

- `src/api.ts`：删 `compile`、`compileSamples`、`queryRun`、`evalSamples`、`worst` 五个包装，及随之空转的 `WorstList`、`EvalSamples` 两个 type import。
- `src/types.ts`：删 `WorstList`、`EvalSamples`、`EvalSampleRow`（`EvalSampleRow` 只被 `EvalSamples` 引用）。其余「只在本文件用」的 type（`QueryStats`、`Readiness` 等）是嵌套字段类型，保留。
- `src/ui.tsx`：`elapsed`、`ROOT_CAUSE_LABELS` 去掉 `export`（本文件在用，外部没人用）。

### 测试

- `tests/test_metrics.py`：删 `require_gold_docs` 那一行断言（保留同一测试里的 `recall_at_k` ValueError 断言）、删 `test_expand_only_applies_k_to_per_k_metrics`。
- `tests/test_platform.py`：`test_missing_records_return_404` 里的 `/api/compiles/9`、`/api/queries/9` 随 GET 路由一起去掉，只留 `/api/evals/9`、`/api/attributions/9`（这两个 GET 保留）。

## 二、顺手修掉的实际问题（ruff 报的，都不改行为）

- `stages/compile_stage.py:362`：`for index, dataset in enumerate(datasets)` 的 `index` 没用 → 改成 `for dataset in datasets`。
- `stages/compile_stage.py:47`：`{k: 0 for k in weights}` → `dict.fromkeys(weights, 0)`。
- `datasets/narrativeqa.py:46`、`judge/answer_relevancy.py:76`、`judge/context_relevancy.py:96`：形参未用但**必须保留**（前者是抽象方法签名，后两者是四个判据共用调用签名）。只把解释这件事的注释压成半句，不动签名。

## 三、注释与 docstring：严格按 CLAUDE.md「只解释作用」

口径：
- 每个模块/函数 docstring 压到一到两句，只说它做什么、返回什么、什么时候返回 None/抛错。
- 删掉所有「为什么这么设计」「不这么做的代价」「实测数据」段落。
- 行内注释只在解释一行代码在做什么时保留，删掉论证性的。
- 不动 SQL 里标注字段含义的注释（那是在说作用）、不动 `# noqa` 与 `# pragma`。

重点文件（改动量由大到小）：
`akasha_client.py`（20 行模块 docstring + `unwrap_envelope` 的 26 行、`_request_with_retry` 的 12 行、`import_page_text` 的 15 行）、`compile_stage.py`、`attribution.py`、`judge/client.py`、`stages/evaluate.py`、`api/config.py`、`platform/tasks.py`、`datasets/models.py`、`metrics/retrieval.py`、`lineage.py`、`store/run_store.py`、`store/db.py`、`metrics/registry.py`（含各指标 description —— 那些是给 UI 看的说明，压掉里面的论证但保留一句作用）、`api/datasets.py`、`api/runs.py`、`api/results.py`、`textdiff.py`、`task.py`、`judge/faithfulness.py`、`judge/answer_relevancy.py`、`judge/context_relevancy.py`、`judge/answer_correctness.py`、`config_store.py`、`stages/query.py`、`stages/chain.py`、`stages/normalize.py`、`stages/__init__.py`、`stages/attribute.py`、`store/task_store.py`、`store/data_store.py`、`settings.py`、`main.py`、`schema.sql`。

同口径处理测试文件的 docstring（`test_metrics.py`、`test_platform.py` 等）与 web 侧注释（`api.ts`、`ui.tsx`、`types.ts`、各 view）。

预期：src 下注释+docstring 从 1163 行降到 350-400 行。

**保留清单**（属于「作用」而非「原因」，删了会丢失接口事实）：
- `schema.sql` 里各字段的含义标注。
- `metrics/registry.py` 里每个指标的 `description` 字符串 —— 那是 API 响应内容，前端要显示，只压缩不删除。
- `akasha_client.py` 里各端点的请求/响应形状（写成一行「字段：形状」而不是段落）。
- `REMEDIES`、`SYSTEM_PROMPT` 等字符串常量的正文（那是数据，不是注释）。

## 四、验证

1. `uv run pytest -q` —— 应仍是 133 减去删掉的 2 个测试 = 131 通过。
2. `npm --prefix web run typecheck` —— `noUnusedLocals` 会抓出漏删的 import。
3. `npm --prefix web run build`。
4. `uvx ruff check --select F,ARG,B,C4,SIM src tests` —— 确认没引入新问题。
5. 删掉 `tmp/deadcode.py`（我这次分析用的临时脚本）。
