# 配置页改造

四件事：导出全部配置、评估/归因端点加并发（真并发执行）、连接表单去掉并发、Akasha 模型配置改为本地多组可应用到远端。

## 1. 导出全部配置（含明文密钥）

后端 `api/config.py` 加 `GET /api/config/export`，返回一整份可回填的 JSON：
- `connection`：`akasha_connection` 全字段（含 password、database_url）。
- `providers`：`model_provider` 全字段（含 api_key、concurrency）。
- `akasha_configs`：新表 `akasha_config_group` 全部组（含每项 apiKey）。

前端 `api.ts` 加 `exportConfig()`；`Settings.tsx` 顶部加「导出全部配置」按钮，取回后用 Blob 下载 `akasha-config-<日期>.json`。仅导出，不做导入。

## 2. 评估/归因端点加并发，且真并发执行

**Schema**：`model_provider` 加列 `concurrency INTEGER NOT NULL DEFAULT 1`。

**config_store.py**：`upsert_provider` 增参 `concurrency`（insert/update 两路都写）；`list_providers` 自然带出。

**providers.py**：`resolve_provider` 把 `concurrency` 带进 `JudgeProvider`。

**judge/client.py**：`JudgeProvider` 加字段 `concurrency: int = 1`。新增 `complete_many(provider, prompts, concurrency)` —— 建 `concurrency` 个 `JudgeClient` 组成客户端池，用 `ThreadPoolExecutor` 按顺序并发跑，返回 `list[JudgeReply]`；`concurrency<=1` 走单客户端串行。仅网络调用进线程，落库不进。

**stages/evaluate.py `_judge_one`**：改为按 `provider.concurrency` 分批：主线程读 rows、拼 prompt / 判 `None`；一批 prompts 交 `complete_many` 并发；主线程按序解析、写 `record_judge_verdict` / `record_sample_eval`、`ctx.checkpoint()`、`ctx.progress()`。保持 `JudgeClient(provider)` + `.complete` 语义（`concurrency=1` 时兼容现有 monkeypatch 测试）。

**stages/attribute.py `_analyze`**：同法分批。主线程按样本读 sample、page_map、lineage、`classify`、`build_prompt`（DB 与 psycopg 只读都留主线程）；一批 prompts 交 `complete_many`；主线程按序解析并 `record_attribution`。`run` 把 `provider.concurrency` 传下去。

`evaluate.run` / `attribute.run` 无需改签名，provider 自带 concurrency。

**API `put_provider`**：读 `payload["concurrency"]`，`int`、`>=1`，默认 1，透传 upsert；响应带回。

**前端**：`types.ts` `Provider` 加 `concurrency`；`Settings.tsx` `Providers` 表加「并发」列与表单项（数字 min=1），`BLANK` 加 `concurrency: 1`，编辑回填。

## 3. 连接表单去掉并发

`Settings.tsx` `NUMBER_FIELDS` 删掉 `concurrency` 那一行。DB 列与 `config.py` 字段保留（query 阶段仍用 `config.concurrency` 兜底；Query 页已自带并发输入）。`types.ts` 的 `Connection.concurrency` 保留（GET 仍回传）。

## 4. Akasha 模型配置：本地多组，可应用到远端

**Schema**：新表
```sql
CREATE TABLE IF NOT EXISTS akasha_config_group (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    label      TEXT NOT NULL UNIQUE,
    configs_json TEXT NOT NULL,   -- {feature: {model, baseUrl, apiKey, parameters}}
    selected   INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);
```

**config_store.py** 加：`list_config_groups` / `get_config_group` / `upsert_config_group`（apiKey 留空保留原值，同 provider）/ `delete_config_group` / `set_selected_group`（清他人置本组）/ `selected_config_group`。

**model_configs.py** 加 `group_to_live(configs)`：把组内 `{feature: {...}}` 整成 `normalize` 可比的 list（用于与远端 live 比对，`_FIELDS` 已不含 apiKey，天然排除密钥）。

**API `api/config.py`** 加：
- `GET/PUT/DELETE /api/akasha-configs`、`POST /api/akasha-configs/{id}/select`：本地组增删改选。列表不回 apiKey，只回 `apiKeySet`（同 provider 约定；明文只经 export）。
- `POST /api/akasha-configs/{id}/apply`：登录远端，遍历四项 `put_model_config`（沿用现有加 `provider` 逻辑），整组推送。
- 扩展 `POST /api/connection/test`：若有选中组，附 `group_drift`（`drift(live, group_to_live(selected))`，排除 apiKey）与逐项差异，供前端提示。

**前端 `Settings.tsx` `ModelConfigs`**：
- 远端 live 四项改为只读展示（保留现有 `ConfigDrift` 与编译快照比对）。
- 新增本地组区：组列表（label、是否选中、可选中/删除）、「新建组」、组编辑器（四项各 model/baseUrl/apiKey，密钥留空保留）。
- 每组一个「应用到 Akasha」按钮，调 apply；REBUILD 项仍弹确认。
- 连接测试回来若 `group_drift` 有差异，提示「本地选中组与远端不一致，可应用」。

`types.ts` 加 `AkashaConfigGroup`；`api.ts` 加对应方法。

## 测试
- `test_platform.py`：export 端点含密钥；provider concurrency 存取；akasha-configs 增删改选与 apply（monkeypatch client）。
- `test_stages.py`：judge/归因并发路径（concurrency>1）产出与串行一致；resume 测试仍绿（concurrency=1 兼容）。
- `make` 跑后端 + 前端类型检查/lint。

## 影响面
schema.sql、config.py、model_configs.py、store/config_store.py、judge/client.py、judge/providers.py、stages/evaluate.py、stages/attribute.py、api/config.py、web(types.ts/api.ts/Settings.tsx)、tests。schema 不做迁移（`CREATE TABLE IF NOT EXISTS` + 新列，旧库需重建，符合现状约定）。
