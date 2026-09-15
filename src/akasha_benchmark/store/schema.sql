-- 数据库最终结构；初始化可重复执行，不负责迁移旧数据库。

-- ------------------------------------------------------------ 配置层
CREATE TABLE IF NOT EXISTS akasha_connection (
    id                       INTEGER PRIMARY KEY CHECK (id = 1),
    base_url                 TEXT NOT NULL DEFAULT 'http://127.0.0.1:3000',
    email                    TEXT NOT NULL DEFAULT 'test@example.com',
    password                 TEXT NOT NULL DEFAULT '12345678',
    database_url             TEXT NOT NULL DEFAULT 'postgresql://akasha:STRONG_DB_PASSWORD@127.0.0.1:5432/akasha',  -- 只读 PG，归因链路用
    timeout_seconds          REAL NOT NULL DEFAULT 180.0,
    request_interval_seconds REAL NOT NULL DEFAULT 0.5,
    poll_interval_seconds    REAL NOT NULL DEFAULT 10.0, -- 编译轮询间隔
    poll_timeout_seconds     REAL NOT NULL DEFAULT 7200.0,
    updated_at               TEXT NOT NULL
);

INSERT OR IGNORE INTO akasha_connection (id, updated_at) VALUES (1, '1970-01-01T00:00:00Z');

CREATE TABLE IF NOT EXISTS model_provider (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    role        TEXT NOT NULL CHECK (role IN ('judge', 'attribution')),
    label       TEXT NOT NULL,
    base_url    TEXT NOT NULL,
    model       TEXT NOT NULL,
    api_key     TEXT NOT NULL DEFAULT '',
    concurrency INTEGER NOT NULL DEFAULT 1,  -- judge / 归因调用的并发数
    updated_at  TEXT NOT NULL,
    UNIQUE (role, label)
);

-- 本地保存的 Akasha 模型配置组，可整组应用到远端。apiKey 明文存在 configs_json 里。
CREATE TABLE IF NOT EXISTS akasha_config_group (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    label        TEXT NOT NULL UNIQUE,
    configs_json TEXT NOT NULL,   -- {feature: {model, baseUrl, apiKey, parameters}}
    selected     INTEGER NOT NULL DEFAULT 0,
    updated_at   TEXT NOT NULL
);

-- --------------------------------------------- 数据集层 / 归一化层
CREATE TABLE IF NOT EXISTS dataset (
    name          TEXT PRIMARY KEY,
    qa_sha256     TEXT NOT NULL,
    qa_rows       INTEGER NOT NULL,
    corpus_sha256 TEXT NOT NULL,
    corpus_rows   INTEGER NOT NULL,
    normalized_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sample (
    sample_id         TEXT PRIMARY KEY,
    dataset           TEXT NOT NULL REFERENCES dataset(name) ON DELETE CASCADE,
    dataset_sample_id TEXT NOT NULL,
    question          TEXT NOT NULL,
    answers_json      TEXT NOT NULL,
    gold_doc_ids_json TEXT NOT NULL,
    metadata_json     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS sample_dataset_idx ON sample(dataset);

CREATE TABLE IF NOT EXISTS corpus_doc (
    dataset TEXT NOT NULL REFERENCES dataset(name) ON DELETE CASCADE,
    doc_id  TEXT NOT NULL,
    title   TEXT NOT NULL,
    text    TEXT NOT NULL,
    PRIMARY KEY (dataset, doc_id)
);

-- ------------------------------------------------------------ 编译层
CREATE TABLE IF NOT EXISTS compile_run (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id             TEXT NOT NULL UNIQUE,
    datasets_json      TEXT NOT NULL,    -- 当初勾了哪几组；与实际抽出的比对能看出哪组失败
    seed               INTEGER NOT NULL,
    qa_limit           INTEGER NOT NULL,  -- 每组各自的 QA 上限，不是总数
    negatives_ratio    REAL NOT NULL,     -- 每篇 gold 配几篇负样本
    space_id           TEXT,             -- 本次编译随机创建的空间；查询打在它上面
    space_name         TEXT,             -- 随机 slug，只做展示与人工核对
    workspace_id       TEXT,             -- 那个空间属于哪个 workspace
    config_group       TEXT,             -- 编译时选中的本地配置组标签
    model_configs_json TEXT,             -- 这一次编译跑在什么模型上
    quality_json       TEXT,             -- missingChunk / missingEmbedding / missingSource / stalePageCount
    pace_json          TEXT,             -- 每篇编译耗时的估算，见 _compile_pace
    status             TEXT NOT NULL,     -- running / paused / succeeded / failed
    created_at         TEXT NOT NULL,
    finished_at        TEXT
);

CREATE TABLE IF NOT EXISTS compile_sample (
    compile_id INTEGER NOT NULL REFERENCES compile_run(id) ON DELETE CASCADE,
    sample_id  TEXT NOT NULL REFERENCES sample(sample_id) ON DELETE CASCADE,
    dataset    TEXT NOT NULL,
    PRIMARY KEY (compile_id, sample_id)
);

CREATE TABLE IF NOT EXISTS compile_doc (
    compile_id INTEGER NOT NULL REFERENCES compile_run(id) ON DELETE CASCADE,
    dataset    TEXT NOT NULL,
    doc_id     TEXT NOT NULL,
    is_gold    INTEGER NOT NULL,
    page_id    TEXT,
    error      TEXT,
    PRIMARY KEY (compile_id, dataset, doc_id)
);

CREATE INDEX IF NOT EXISTS compile_doc_page_idx ON compile_doc(compile_id, page_id);

-- ------------------------------------------------------------ 查询层
CREATE TABLE IF NOT EXISTS query_run (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    name               TEXT NOT NULL UNIQUE,
    compile_id         INTEGER NOT NULL REFERENCES compile_run(id) ON DELETE CASCADE,
    score_threshold    REAL,             -- 空表示用服务端默认值
    concurrency        INTEGER NOT NULL, -- 并发调用数
    model_configs_json TEXT,             -- 查询时的模型快照
    status             TEXT NOT NULL,    -- running / paused / succeeded / failed
    created_at         TEXT NOT NULL,
    finished_at        TEXT
);

CREATE TABLE IF NOT EXISTS query_sample (
    query_id  INTEGER NOT NULL REFERENCES query_run(id) ON DELETE CASCADE,
    sample_id TEXT NOT NULL,
    dataset   TEXT NOT NULL,
    PRIMARY KEY (query_id, sample_id)
);

CREATE TABLE IF NOT EXISTS query_response (
    query_id      INTEGER NOT NULL REFERENCES query_run(id) ON DELETE CASCADE,
    sample_id     TEXT NOT NULL,
    dataset       TEXT NOT NULL,
    question      TEXT NOT NULL,
    http_status   INTEGER NOT NULL,
    latency_ms    INTEGER,
    error         TEXT,
    answer_mode   TEXT,
    response_json TEXT,   -- 完整响应体，避免为了看新字段重跑
    requested_at  TEXT NOT NULL,
    PRIMARY KEY (query_id, sample_id)
);

-- ------------------------------------------------------------ 评测层
CREATE TABLE IF NOT EXISTS eval_run (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    name              TEXT NOT NULL UNIQUE,
    query_id          INTEGER NOT NULL REFERENCES query_run(id) ON DELETE CASCADE,
    ks_json           TEXT NOT NULL,  -- recall@k 之类按哪些 k 展开
    metrics_json      TEXT NOT NULL,  -- 这一轮勾了哪些指标（模板名，不带 @k）
    judge_provider_id INTEGER REFERENCES model_provider(id) ON DELETE SET NULL,
    status            TEXT NOT NULL,  -- running / paused / succeeded / failed
    created_at        TEXT NOT NULL,
    finished_at       TEXT
);

CREATE TABLE IF NOT EXISTS sample_eval (
    eval_id     INTEGER NOT NULL REFERENCES eval_run(id) ON DELETE CASCADE,
    sample_id   TEXT NOT NULL,
    dataset     TEXT NOT NULL,
    answer_mode TEXT,
    http_status INTEGER NOT NULL,
    answer      TEXT,
    detail_json TEXT NOT NULL,   -- 逐样本明细，归因读它
    PRIMARY KEY (eval_id, sample_id)
);

CREATE TABLE IF NOT EXISTS sample_metric (
    eval_id   INTEGER NOT NULL REFERENCES eval_run(id) ON DELETE CASCADE,
    sample_id TEXT NOT NULL,
    dataset   TEXT NOT NULL,
    metric    TEXT NOT NULL,
    value     REAL NOT NULL,
    PRIMARY KEY (eval_id, sample_id, metric)
);

CREATE INDEX IF NOT EXISTS sample_metric_lookup_idx ON sample_metric(eval_id, metric, value);

-- scope 取 overall / knowledge_only，分别汇总全样本与 knowledge 子集。
CREATE TABLE IF NOT EXISTS metric_summary (
    eval_id      INTEGER NOT NULL REFERENCES eval_run(id) ON DELETE CASCADE,
    dataset      TEXT NOT NULL,
    scope        TEXT NOT NULL,
    metric       TEXT NOT NULL,
    value        REAL,
    sample_count INTEGER NOT NULL,
    PRIMARY KEY (eval_id, dataset, scope, metric)
);

CREATE TABLE IF NOT EXISTS dataset_eval (
    eval_id             INTEGER NOT NULL REFERENCES eval_run(id) ON DELETE CASCADE,
    dataset             TEXT NOT NULL,
    responses_evaluated INTEGER NOT NULL,
    http_failures       INTEGER NOT NULL,
    omitted_metrics_json TEXT NOT NULL,  -- 缺依赖算不了的指标，不伪造 0 分
    answer_modes_json   TEXT NOT NULL,
    PRIMARY KEY (eval_id, dataset)
);

-- 一个样本可以有多条 judge 结论，所以指标名进主键。
CREATE TABLE IF NOT EXISTS judge_verdict (
    eval_id      INTEGER NOT NULL REFERENCES eval_run(id) ON DELETE CASCADE,
    sample_id    TEXT NOT NULL,
    metric       TEXT NOT NULL DEFAULT 'faithfulness',
    score        REAL,           -- 跳过或失败时为 NULL，不记 0
    failure_kind TEXT,
    latency_ms   INTEGER,        -- 跳过的条目没有调用，为 NULL
    detail_json  TEXT,
    PRIMARY KEY (eval_id, sample_id, metric)
);

-- ------------------------------------------------------------ 归因层
CREATE TABLE IF NOT EXISTS attribution_run (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    eval_id     INTEGER NOT NULL REFERENCES eval_run(id) ON DELETE CASCADE,
    metric      TEXT NOT NULL,
    sample_limit INTEGER NOT NULL,
    provider_id INTEGER REFERENCES model_provider(id) ON DELETE SET NULL,
    status      TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    finished_at TEXT
);

CREATE TABLE IF NOT EXISTS attribution_result (
    attribution_id INTEGER NOT NULL REFERENCES attribution_run(id) ON DELETE CASCADE,
    sample_id      TEXT NOT NULL,
    dataset        TEXT NOT NULL,
    root_cause     TEXT NOT NULL,
    evidence_json  TEXT NOT NULL,
    narrative      TEXT,
    rule_based     INTEGER NOT NULL,
    latency_ms     INTEGER,        -- 规则归因没有调用，为 NULL
    PRIMARY KEY (attribution_id, sample_id)
);

CREATE INDEX IF NOT EXISTS attribution_cause_idx ON attribution_result(attribution_id, root_cause);

-- ------------------------------------------------------------ 任务层
CREATE TABLE IF NOT EXISTS task (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    stage          TEXT NOT NULL,
    status         TEXT NOT NULL,  -- queued/running/paused/succeeded/failed
    params_json    TEXT NOT NULL,
    target_kind    TEXT,           -- 产物所属层，供层视图反查任务
    target_id      INTEGER,
    progress_done  INTEGER NOT NULL DEFAULT 0,
    progress_total INTEGER,
    progress_note  TEXT,
    error          TEXT,
    created_at     TEXT NOT NULL,
    started_at     TEXT,
    finished_at    TEXT,
    chain_id       INTEGER,        -- 同一条链的各阶段共用链首任务的 id
    chain_json     TEXT            -- 链上剩余阶段；空表示到此为止
);

CREATE INDEX IF NOT EXISTS task_status_idx ON task(status, id);
CREATE INDEX IF NOT EXISTS task_chain_idx ON task(chain_id, id);

-- 只追加，清理任务不删它，所以 task_id 不设外键。
CREATE TABLE IF NOT EXISTS audit_log (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id INTEGER,
    stage   TEXT NOT NULL,
    level   TEXT NOT NULL,
    message TEXT NOT NULL,
    at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS audit_log_task_idx ON audit_log(task_id, id);
