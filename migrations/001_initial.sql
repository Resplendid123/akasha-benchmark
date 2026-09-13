-- Final schema. Migration bookkeeping is managed by store/migrate.py.

CREATE TABLE dataset (
    name                         TEXT PRIMARY KEY,  -- 数据集名
    adapter                      TEXT NOT NULL,     -- 使用的适配器名
    adapter_version              TEXT NOT NULL,     -- 适配器版本
    provides_json                TEXT NOT NULL,     -- 该数据集提供的能力（JSON 数组）
    identity_rules_json          TEXT NOT NULL,     -- 样本/文档身份识别规则（JSON）
    qa_path                      TEXT NOT NULL,     -- QA 原始文件路径
    qa_sha256                    TEXT NOT NULL,     -- QA 原始文件哈希
    qa_rows                      INTEGER NOT NULL,  -- QA 行数
    corpus_path                  TEXT NOT NULL,     -- 语料原始文件路径
    corpus_sha256                TEXT NOT NULL,     -- 语料原始文件哈希
    corpus_rows                  INTEGER NOT NULL,  -- 语料行数
    dedup_stats_json             TEXT NOT NULL,     -- 去重统计（JSON）
    gold_count_distribution_json TEXT NOT NULL,     -- gold 篇数分布（JSON）
    unique_question_texts        INTEGER NOT NULL,  -- 去重后问题文本数
    normalized_at                TEXT NOT NULL      -- 归一化时间
);

CREATE TABLE sample (
    dataset           TEXT NOT NULL REFERENCES dataset(name) ON DELETE CASCADE,  -- 所属数据集
    sample_id         TEXT PRIMARY KEY,         -- 样本唯一 ID（全局）
    dataset_sample_id TEXT NOT NULL,            -- 数据集内的样本 ID
    question          TEXT NOT NULL,            -- 问题文本
    answers_json      TEXT NOT NULL,            -- 标准答案（JSON 数组）
    gold_doc_ids_json TEXT NOT NULL,            -- 金标文档 ID（JSON 数组）
    metadata_json     TEXT NOT NULL             -- 附加元数据（JSON）
);

CREATE INDEX sample_dataset_idx ON sample(dataset);

CREATE TABLE corpus_doc (
    dataset     TEXT NOT NULL REFERENCES dataset(name) ON DELETE CASCADE,  -- 所属数据集
    doc_id      TEXT NOT NULL,   -- 文档 ID
    title       TEXT NOT NULL,   -- 标题
    text        TEXT NOT NULL,   -- 正文
    text_sha256 TEXT NOT NULL,   -- 正文哈希
    PRIMARY KEY (dataset, doc_id)
);

CREATE INDEX corpus_doc_title_idx ON corpus_doc(dataset, title);

CREATE TABLE index_layer (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,  -- 自增主键
    label             TEXT NOT NULL UNIQUE,  -- 索引层标签（唯一）
    subset_hash       TEXT NOT NULL,         -- 子集哈希
    config_hash       TEXT,                  -- 配置哈希
    seed              INTEGER NOT NULL,      -- 随机种子
    qa_limit          INTEGER NOT NULL,      -- QA 数量上限
    negatives_ratio   REAL NOT NULL,         -- 负样本比例
    narrativeqa_docs  INTEGER NOT NULL,      -- NarrativeQA 文档数
    model_configs_json TEXT,                 -- 模型配置（JSON）
    workspace_id      TEXT,                  -- 工作区 ID
    workspace_name    TEXT,                  -- 工作区名
    akasha_user_id    TEXT,                  -- Akasha 用户 ID
    akasha_user_role  TEXT,                  -- Akasha 用户角色
    connection_json   TEXT,                  -- 连接信息（JSON）
    created_at        TEXT NOT NULL,         -- 创建时间
    subset_built_at   TEXT,                  -- 子集构建时间
    ingested_at       TEXT,                  -- 摄取完成时间
    quality_passed    INTEGER,               -- 质量门是否通过
    notes             TEXT                   -- 备注
);

CREATE INDEX index_layer_config_hash_idx ON index_layer(config_hash);

CREATE INDEX index_layer_subset_hash_idx ON index_layer(subset_hash);
CREATE TABLE index_layer_dataset (
    index_layer_id INTEGER NOT NULL REFERENCES index_layer(id) ON DELETE CASCADE,  -- 索引层 ID
    dataset        TEXT NOT NULL REFERENCES dataset(name) ON DELETE CASCADE,       -- 数据集名
    space_id       TEXT,          -- Akasha 空间 ID
    space_slug     TEXT,          -- 空间 slug
    space_reused   INTEGER,       -- 是否复用已有空间
    strategy       TEXT NOT NULL, -- 采样策略
    qa_count       INTEGER NOT NULL,         -- QA 数量
    corpus_count   INTEGER NOT NULL,         -- 语料文档数
    gold_doc_count INTEGER NOT NULL,         -- 金标文档数
    negative_doc_count INTEGER NOT NULL,     -- 负样本文档数
    strata_json    TEXT,                     -- 分层信息（JSON）
    normalized_qa_sha256     TEXT NOT NULL,  -- 归一化 QA 哈希
    normalized_corpus_sha256 TEXT NOT NULL,  -- 归一化语料哈希
    PRIMARY KEY (index_layer_id, dataset)
);

CREATE TABLE subset_sample (
    index_layer_id INTEGER NOT NULL REFERENCES index_layer(id) ON DELETE CASCADE,  -- 索引层 ID
    dataset        TEXT NOT NULL,   -- 数据集名
    sample_id      TEXT NOT NULL REFERENCES sample(sample_id) ON DELETE CASCADE,   -- 样本 ID
    PRIMARY KEY (index_layer_id, sample_id)
);
CREATE INDEX subset_sample_layer_dataset_idx ON subset_sample(index_layer_id, dataset);

CREATE TABLE subset_doc (
    index_layer_id INTEGER NOT NULL REFERENCES index_layer(id) ON DELETE CASCADE,  -- 索引层 ID
    dataset        TEXT NOT NULL,   -- 数据集名
    doc_id         TEXT NOT NULL,   -- 文档 ID
    md_text        TEXT NOT NULL,   -- Markdown 正文
    md_sha256      TEXT NOT NULL,   -- Markdown 哈希
    is_gold        INTEGER NOT NULL,-- 是否为金标文档
    PRIMARY KEY (index_layer_id, dataset, doc_id)
);

CREATE TABLE page_map (
    index_layer_id INTEGER NOT NULL REFERENCES index_layer(id) ON DELETE CASCADE,  -- 索引层 ID
    dataset        TEXT NOT NULL,   -- 数据集名
    doc_id         TEXT NOT NULL,   -- 文档 ID
    page_id        TEXT NOT NULL,   -- Akasha 页面 ID
    space_id       TEXT NOT NULL,   -- 空间 ID
    title          TEXT,            -- 页面标题
    md_sha256      TEXT NOT NULL,   -- Markdown 哈希
    imported_at    TEXT NOT NULL,   -- 导入时间
    PRIMARY KEY (index_layer_id, dataset, doc_id)
);

CREATE INDEX page_map_page_id_idx ON page_map(index_layer_id, page_id);

CREATE TABLE import_failure (
    index_layer_id INTEGER NOT NULL REFERENCES index_layer(id) ON DELETE CASCADE,  -- 索引层 ID
    dataset        TEXT NOT NULL,   -- 数据集名
    doc_id         TEXT NOT NULL,   -- 文档 ID
    http_status    INTEGER,         -- HTTP 状态码
    error          TEXT,            -- 错误信息
    failed_at      TEXT NOT NULL    -- 失败时间
);

CREATE TABLE compile_run (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,  -- 自增主键
    index_layer_id     INTEGER NOT NULL REFERENCES index_layer(id) ON DELETE CASCADE,  -- 索引层 ID
    requested_at       TEXT NOT NULL,        -- 请求时间
    accepted_run_count INTEGER,              -- 接受运行数
    coalesced_run_count INTEGER,             -- 合并运行数
    status_counts_json TEXT,                 -- 状态计数（JSON）
    terminal_json      TEXT,                 -- 终态信息（JSON）
    timed_out          INTEGER NOT NULL DEFAULT 0,  -- 是否超时
    finished_at        TEXT                  -- 完成时间
);

CREATE TABLE quality_gate (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,  -- 自增主键
    index_layer_id INTEGER NOT NULL REFERENCES index_layer(id) ON DELETE CASCADE,  -- 索引层 ID
    checked_at     TEXT NOT NULL,  -- 检查时间
    missing_chunk_page_count     INTEGER,  -- 缺 chunk 的页面数
    missing_embedding_page_count INTEGER,  -- 缺 embedding 的页面数
    missing_source_page_count    INTEGER,  -- 缺 source 的页面数
    stale_page_count             INTEGER,  -- 过期页面数
    passed         INTEGER NOT NULL,        -- 是否通过
    report_json    TEXT                     -- 报告详情（JSON）
);

CREATE TABLE query_layer (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,  -- 自增主键
    index_layer_id     INTEGER NOT NULL REFERENCES index_layer(id) ON DELETE CASCADE,  -- 索引层 ID
    label              TEXT NOT NULL UNIQUE,  -- 查询层标签（唯一）
    config_hash        TEXT NOT NULL,         -- 配置哈希
    score_threshold    REAL,                  -- 分数阈值
    concurrency        INTEGER NOT NULL,      -- 并发数
    request_interval_seconds REAL NOT NULL,   -- 请求间隔（秒）
    model_configs_json TEXT,                  -- 模型配置（JSON）
    model_configs_match_index INTEGER,        -- 模型配置是否与索引层匹配
    allow_config_drift INTEGER NOT NULL DEFAULT 0,  -- 是否允许配置漂移
    started_at         TEXT NOT NULL,         -- 开始时间
    finished_at        TEXT,                  -- 完成时间
    notes              TEXT                   -- 备注
);

CREATE INDEX query_layer_config_hash_idx ON query_layer(config_hash);

CREATE INDEX query_layer_index_layer_idx ON query_layer(index_layer_id);

CREATE TABLE query_response (
    query_layer_id INTEGER NOT NULL REFERENCES query_layer(id) ON DELETE CASCADE,  -- 查询层 ID
    sample_id      TEXT NOT NULL,   -- 样本 ID
    dataset        TEXT NOT NULL,   -- 数据集名
    question       TEXT NOT NULL,   -- 问题文本
    requested_at   TEXT NOT NULL,   -- 请求时间
    latency_ms     INTEGER,         -- 延迟（毫秒）
    http_status    INTEGER NOT NULL,-- HTTP 状态码
    error          TEXT,            -- 错误信息
    response_json  TEXT,            -- 响应体（JSON）
    answer_mode    TEXT,            -- 答案模式
    PRIMARY KEY (query_layer_id, sample_id)
);

CREATE INDEX query_response_answer_mode_idx
    ON query_response(query_layer_id, dataset, answer_mode);

-- Freeze each query run's sample selection before issuing requests.
CREATE TABLE query_selection (
    query_layer_id INTEGER PRIMARY KEY REFERENCES query_layer(id) ON DELETE CASCADE,
    selection_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE judge_provider (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,  -- 自增主键
    label         TEXT NOT NULL UNIQUE,  -- 标签（唯一）
    base_url      TEXT NOT NULL,         -- API 基地址
    model         TEXT NOT NULL,         -- 模型名
    params_json   TEXT,                  -- 参数（JSON）
    api_key_env   TEXT NOT NULL,         -- API key 环境变量名
    created_at    TEXT NOT NULL          -- 创建时间
);

CREATE TABLE eval_layer (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,  -- 自增主键
    query_layer_id    INTEGER NOT NULL REFERENCES query_layer(id) ON DELETE CASCADE,  -- 查询层 ID
    label             TEXT NOT NULL UNIQUE,  -- 评估层标签（唯一）
    config_hash       TEXT NOT NULL,         -- 配置哈希
    ks_json           TEXT NOT NULL,         -- K 值列表（JSON）
    metrics_json      TEXT NOT NULL,         -- 指标列表（JSON）
    judge_provider_id INTEGER REFERENCES judge_provider(id) ON DELETE SET NULL,  -- 评判提供方 ID
    created_at        TEXT NOT NULL,         -- 创建时间
    finished_at       TEXT,                  -- 完成时间
    notes             TEXT                   -- 备注
);

CREATE INDEX eval_layer_config_hash_idx ON eval_layer(config_hash);

CREATE INDEX eval_layer_query_layer_idx ON eval_layer(query_layer_id);

CREATE TABLE sample_eval (
    eval_layer_id  INTEGER NOT NULL REFERENCES eval_layer(id) ON DELETE CASCADE,  -- 评估层 ID
    sample_id      TEXT NOT NULL,   -- 样本 ID
    dataset        TEXT NOT NULL,   -- 数据集名
    answer_mode    TEXT,            -- 答案模式
    ok             INTEGER NOT NULL,-- 是否成功
    http_status    INTEGER NOT NULL,-- HTTP 状态码
    gold_count     INTEGER NOT NULL,-- 金标数
    retrieved_count INTEGER NOT NULL,-- 检索数
    citation_count INTEGER NOT NULL,-- 引用数
    snippet_count  INTEGER,         -- 片段数
    latency_ms     INTEGER,         -- 延迟（毫秒）
    answer         TEXT,            -- 生成的答案
    detail_json    TEXT NOT NULL,   -- 详情（JSON）
    PRIMARY KEY (eval_layer_id, sample_id)
);

CREATE INDEX sample_eval_mode_idx ON sample_eval(eval_layer_id, dataset, answer_mode);

CREATE TABLE sample_metric (
    eval_layer_id INTEGER NOT NULL REFERENCES eval_layer(id) ON DELETE CASCADE,  -- 评估层 ID
    sample_id     TEXT NOT NULL,   -- 样本 ID
    dataset       TEXT NOT NULL,   -- 数据集名
    metric        TEXT NOT NULL,   -- 指标名
    value         REAL NOT NULL,   -- 指标值
    PRIMARY KEY (eval_layer_id, sample_id, metric)
);

CREATE INDEX sample_metric_lookup_idx ON sample_metric(eval_layer_id, metric, value);

CREATE TABLE metric_summary (
    eval_layer_id INTEGER NOT NULL REFERENCES eval_layer(id) ON DELETE CASCADE,  -- 评估层 ID
    dataset       TEXT NOT NULL,   -- 数据集名
    scope         TEXT NOT NULL,   -- 统计范围
    metric        TEXT NOT NULL,   -- 指标名
    value         REAL,            -- 汇总值
    sample_count  INTEGER NOT NULL,-- 样本数
    PRIMARY KEY (eval_layer_id, dataset, scope, metric)
);

CREATE TABLE dataset_eval (
    eval_layer_id      INTEGER NOT NULL REFERENCES eval_layer(id) ON DELETE CASCADE,  -- 评估层 ID
    dataset            TEXT NOT NULL,       -- 数据集名
    samples_in_subset  INTEGER NOT NULL,    -- 子集样本数
    responses_evaluated INTEGER NOT NULL,   -- 已评估响应数
    http_failures      INTEGER NOT NULL,    -- HTTP 失败数
    missing_responses_json TEXT NOT NULL,   -- 缺失响应（JSON）
    unmapped_page_ids_json TEXT NOT NULL,   -- 未映射页面 ID（JSON）
    omitted_metrics_json   TEXT NOT NULL,   -- 被省略指标（JSON）
    omission_reason    TEXT,                -- 省略原因
    answer_mode_distribution_json TEXT NOT NULL,  -- 答案模式分布（JSON）
    stratified_json    TEXT,                -- 分层统计（JSON）
    PRIMARY KEY (eval_layer_id, dataset)
);

CREATE TABLE audit_record (
    query_layer_id  INTEGER NOT NULL REFERENCES query_layer(id) ON DELETE CASCADE,  -- 查询层 ID
    sample_id       TEXT NOT NULL,   -- 样本 ID
    query_hash      TEXT NOT NULL,   -- 查询哈希
    retrieval_mode  TEXT,            -- 检索模式
    candidate_chunk_count INTEGER,   -- 候选 chunk 数
    ranked_candidate_count INTEGER,  -- 排序后候选数
    filtered_chunk_count  INTEGER,   -- 过滤后 chunk 数
    access_policy_fallback_used INTEGER,  -- 是否用了访问策略回退
    metadata_json   TEXT NOT NULL,   -- 元数据（JSON）
    audit_created_at TEXT,           -- 审计创建时间
    copied_at       TEXT NOT NULL,   -- 拷贝时间
    PRIMARY KEY (query_layer_id, sample_id)
);

CREATE TABLE judge_verdict (
    eval_layer_id  INTEGER NOT NULL REFERENCES eval_layer(id) ON DELETE CASCADE,  -- 评估层 ID
    sample_id      TEXT NOT NULL,   -- 样本 ID
    metric         TEXT NOT NULL,   -- 指标名
    score          REAL,            -- 分数
    failure_kind   TEXT,            -- 失败类型
    reasoning_json TEXT,            -- 推理过程（JSON）
    raw_response   TEXT,            -- 原始响应
    provider_hash  TEXT NOT NULL,   -- 提供方哈希
    prompt_version TEXT NOT NULL,   -- prompt 版本
    judged_at      TEXT NOT NULL,   -- 评判时间
    PRIMARY KEY (eval_layer_id, sample_id, metric)
);

CREATE INDEX judge_verdict_metric_idx ON judge_verdict(eval_layer_id, metric, score);

CREATE TABLE annotation (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,  -- 自增主键
    level       TEXT NOT NULL CHECK (level IN ('sample', 'query_layer', 'eval_layer')),  -- 标注层级
    target_id   TEXT NOT NULL,   -- 目标 ID
    author_kind TEXT NOT NULL CHECK (author_kind IN ('human', 'model')),  -- 标注者类型
    author      TEXT NOT NULL,   -- 标注者
    labels_json TEXT NOT NULL,   -- 标签（JSON）
    note        TEXT,            -- 备注
    source      TEXT NOT NULL,   -- 来源
    confidence  REAL,            -- 置信度
    created_at  TEXT NOT NULL    -- 创建时间
);

CREATE INDEX annotation_target_idx ON annotation(level, target_id);

CREATE TABLE task (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,  -- 自增主键
    stage       TEXT NOT NULL,   -- 阶段名
    status      TEXT NOT NULL,   -- 状态
    index_layer_id INTEGER REFERENCES index_layer(id) ON DELETE SET NULL,  -- 索引层 ID
    query_layer_id INTEGER REFERENCES query_layer(id) ON DELETE SET NULL,  -- 查询层 ID
    eval_layer_id  INTEGER REFERENCES eval_layer(id) ON DELETE SET NULL,   -- 评估层 ID
    argv_json   TEXT NOT NULL,   -- 命令行参数（JSON）
    pid         INTEGER,         -- 进程 ID
    exit_code   INTEGER,         -- 退出码
    progress_done  INTEGER NOT NULL DEFAULT 0,  -- 已完成进度
    progress_total INTEGER,      -- 总进度
    progress_note  TEXT,         -- 进度备注
    log_path    TEXT,            -- 日志路径
    created_at  TEXT NOT NULL,   -- 创建时间
    started_at  TEXT,            -- 开始时间
    finished_at TEXT,            -- 完成时间
    error       TEXT             -- 错误信息
);

CREATE INDEX task_status_idx ON task(status, created_at);

CREATE TABLE task_event (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,  -- 自增主键
    task_id    INTEGER NOT NULL REFERENCES task(id) ON DELETE CASCADE,  -- 任务 ID
    at         TEXT NOT NULL,    -- 事件时间
    level      TEXT NOT NULL,    -- 日志级别
    message    TEXT NOT NULL     -- 日志内容
);

CREATE INDEX task_event_task_idx ON task_event(task_id, id);

CREATE TABLE metric_definition (
    name            TEXT PRIMARY KEY,  -- 指标名
    family          TEXT NOT NULL,     -- 指标家族
    requires_json   TEXT NOT NULL,     -- 依赖项（JSON）
    kind            TEXT NOT NULL,     -- 类型
    higher_is_better INTEGER NOT NULL, -- 是否越大越好
    description     TEXT NOT NULL      -- 描述
);

CREATE TABLE app_config (
    key        TEXT PRIMARY KEY,  -- 配置键
    value_json TEXT NOT NULL,     -- 配置值（JSON）
    updated_at TEXT NOT NULL      -- 更新时间
);

CREATE TABLE model_provider (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,  -- 自增主键
    role        TEXT NOT NULL CHECK (role IN ('judge', 'analysis')),  -- 角色
    label       TEXT NOT NULL,   -- 标签
    base_url    TEXT NOT NULL,   -- API 基地址
    model       TEXT NOT NULL,   -- 模型名
    api_key     TEXT NOT NULL DEFAULT '',      -- API key
    api_key_env TEXT NOT NULL DEFAULT '',      -- API key 环境变量名
    params_json TEXT,            -- 参数（JSON）
    created_at  TEXT NOT NULL,   -- 创建时间
    updated_at  TEXT NOT NULL,   -- 更新时间
    UNIQUE (role, label)
);

CREATE TABLE run_config (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,  -- 自增主键
    stage      TEXT NOT NULL,   -- 阶段名
    args_json  TEXT NOT NULL,   -- 参数（JSON）
    created_at TEXT NOT NULL    -- 创建时间
);

CREATE TABLE badcase_analysis (
    eval_layer_id  INTEGER NOT NULL REFERENCES eval_layer(id) ON DELETE CASCADE,  -- 评估层 ID
    sample_id      TEXT NOT NULL,   -- 样本 ID
    root_cause     TEXT NOT NULL,   -- 根因
    labels_json    TEXT NOT NULL,   -- 标签（JSON）
    evidence_json  TEXT NOT NULL,   -- 证据（JSON）
    narrative      TEXT,            -- 叙述性说明
    rule_based     INTEGER NOT NULL,-- 是否基于规则
    provider_hash  TEXT NOT NULL DEFAULT '',  -- 提供方哈希
    prompt_version TEXT NOT NULL DEFAULT '',  -- prompt 版本
    analyzed_at    TEXT NOT NULL,   -- 分析时间
    PRIMARY KEY (eval_layer_id, sample_id)
);

CREATE INDEX badcase_analysis_cause_idx ON badcase_analysis(eval_layer_id, root_cause);

CREATE TABLE "connection" (
    id                INTEGER PRIMARY KEY CHECK (id = 1),  -- 固定为 1（单行）
    base_url          TEXT NOT NULL DEFAULT 'http://localhost:3000',  -- 基础 URL
    api_prefix        TEXT NOT NULL DEFAULT '/api',  -- API 前缀
    email             TEXT NOT NULL DEFAULT '',      -- 登录邮箱
    password          TEXT NOT NULL DEFAULT '',      -- 登录密码
    database_url      TEXT NOT NULL DEFAULT '',      -- 数据库 URL
    timeout_seconds          REAL NOT NULL DEFAULT 180.0,  -- 超时（秒）
    concurrency              INTEGER NOT NULL DEFAULT 1,   -- 并发数
    request_interval_seconds REAL NOT NULL DEFAULT 0.5,    -- 请求间隔（秒）
    poll_interval_seconds    REAL NOT NULL DEFAULT 10.0,   -- 轮询间隔（秒）
    poll_timeout_seconds     REAL NOT NULL DEFAULT 7200.0, -- 轮询超时（秒）
    last_checked_at   TEXT,          -- 上次检查时间
    last_check_ok     INTEGER,       -- 上次检查是否成功
    last_check_role   TEXT,          -- 上次检查角色
    last_model_configs_json TEXT,    -- 上次模型配置（JSON）
    created_at        TEXT NOT NULL, -- 创建时间
    updated_at        TEXT NOT NULL  -- 更新时间
);

INSERT INTO connection (id, created_at, updated_at)
VALUES (1, '1970-01-01T00:00:00Z', '1970-01-01T00:00:00Z');
