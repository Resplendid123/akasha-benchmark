-- Final schema. Migration bookkeeping is managed by store/migrate.py.

CREATE TABLE dataset (
    name                         TEXT PRIMARY KEY,
    adapter                      TEXT NOT NULL,
    adapter_version              TEXT NOT NULL,
    provides_json                TEXT NOT NULL,
    identity_rules_json          TEXT NOT NULL,
    qa_path                      TEXT NOT NULL,
    qa_sha256                    TEXT NOT NULL,
    qa_rows                      INTEGER NOT NULL,
    corpus_path                  TEXT NOT NULL,
    corpus_sha256                TEXT NOT NULL,
    corpus_rows                  INTEGER NOT NULL,
    dedup_stats_json             TEXT NOT NULL,
    gold_count_distribution_json TEXT NOT NULL,
    unique_question_texts        INTEGER NOT NULL,
    normalized_at                TEXT NOT NULL
);

CREATE TABLE sample (
    dataset           TEXT NOT NULL REFERENCES dataset(name) ON DELETE CASCADE,
    sample_id         TEXT PRIMARY KEY,
    dataset_sample_id TEXT NOT NULL,
    question          TEXT NOT NULL,
    answers_json      TEXT NOT NULL,
    gold_doc_ids_json TEXT NOT NULL,
    metadata_json     TEXT NOT NULL
);

CREATE INDEX sample_dataset_idx ON sample(dataset);

CREATE TABLE corpus_doc (
    dataset     TEXT NOT NULL REFERENCES dataset(name) ON DELETE CASCADE,
    doc_id      TEXT NOT NULL,
    title       TEXT NOT NULL,
    text        TEXT NOT NULL,
    text_sha256 TEXT NOT NULL,
    PRIMARY KEY (dataset, doc_id)
);

CREATE INDEX corpus_doc_title_idx ON corpus_doc(dataset, title);

CREATE TABLE index_layer (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    label             TEXT NOT NULL UNIQUE,
    subset_hash       TEXT NOT NULL,
    config_hash       TEXT,
    seed              INTEGER NOT NULL,
    qa_limit          INTEGER NOT NULL,
    negatives_ratio   REAL NOT NULL,
    narrativeqa_docs  INTEGER NOT NULL,
    model_configs_json TEXT,
    workspace_id      TEXT,
    workspace_name    TEXT,
    akasha_user_id    TEXT,
    akasha_user_role  TEXT,
    connection_json   TEXT,
    created_at        TEXT NOT NULL,
    subset_built_at   TEXT,
    ingested_at       TEXT,
    quality_passed    INTEGER,
    notes             TEXT
);

CREATE INDEX index_layer_config_hash_idx ON index_layer(config_hash);

CREATE INDEX index_layer_subset_hash_idx ON index_layer(subset_hash);

CREATE TABLE index_layer_dataset (
    index_layer_id INTEGER NOT NULL REFERENCES index_layer(id) ON DELETE CASCADE,
    dataset        TEXT NOT NULL REFERENCES dataset(name) ON DELETE CASCADE,
    space_id       TEXT,
    space_slug     TEXT,
    space_reused   INTEGER,
    strategy       TEXT NOT NULL,
    qa_count       INTEGER NOT NULL,
    corpus_count   INTEGER NOT NULL,
    gold_doc_count INTEGER NOT NULL,
    negative_doc_count INTEGER NOT NULL,
    strata_json    TEXT,
    normalized_qa_sha256     TEXT NOT NULL,
    normalized_corpus_sha256 TEXT NOT NULL,
    PRIMARY KEY (index_layer_id, dataset)
);

CREATE TABLE subset_sample (
    index_layer_id INTEGER NOT NULL REFERENCES index_layer(id) ON DELETE CASCADE,
    dataset        TEXT NOT NULL,
    sample_id      TEXT NOT NULL REFERENCES sample(sample_id) ON DELETE CASCADE,
    PRIMARY KEY (index_layer_id, sample_id)
);

CREATE INDEX subset_sample_layer_dataset_idx ON subset_sample(index_layer_id, dataset);

CREATE TABLE subset_doc (
    index_layer_id INTEGER NOT NULL REFERENCES index_layer(id) ON DELETE CASCADE,
    dataset        TEXT NOT NULL,
    doc_id         TEXT NOT NULL,
    md_text        TEXT NOT NULL,
    md_sha256      TEXT NOT NULL,
    is_gold        INTEGER NOT NULL,
    PRIMARY KEY (index_layer_id, dataset, doc_id)
);

CREATE TABLE page_map (
    index_layer_id INTEGER NOT NULL REFERENCES index_layer(id) ON DELETE CASCADE,
    dataset        TEXT NOT NULL,
    doc_id         TEXT NOT NULL,
    page_id        TEXT NOT NULL,
    space_id       TEXT NOT NULL,
    title          TEXT,
    md_sha256      TEXT NOT NULL,
    imported_at    TEXT NOT NULL,
    PRIMARY KEY (index_layer_id, dataset, doc_id)
);

CREATE INDEX page_map_page_id_idx ON page_map(index_layer_id, page_id);

CREATE TABLE import_failure (
    index_layer_id INTEGER NOT NULL REFERENCES index_layer(id) ON DELETE CASCADE,
    dataset        TEXT NOT NULL,
    doc_id         TEXT NOT NULL,
    http_status    INTEGER,
    error          TEXT,
    failed_at      TEXT NOT NULL
);

CREATE TABLE compile_run (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    index_layer_id     INTEGER NOT NULL REFERENCES index_layer(id) ON DELETE CASCADE,
    requested_at       TEXT NOT NULL,
    accepted_run_count INTEGER,
    coalesced_run_count INTEGER,
    status_counts_json TEXT,
    terminal_json      TEXT,
    timed_out          INTEGER NOT NULL DEFAULT 0,
    finished_at        TEXT
);

CREATE TABLE quality_gate (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    index_layer_id INTEGER NOT NULL REFERENCES index_layer(id) ON DELETE CASCADE,
    checked_at     TEXT NOT NULL,
    missing_chunk_page_count     INTEGER,
    missing_embedding_page_count INTEGER,
    missing_source_page_count    INTEGER,
    stale_page_count             INTEGER,
    passed         INTEGER NOT NULL,
    report_json    TEXT
);

CREATE TABLE query_layer (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    index_layer_id     INTEGER NOT NULL REFERENCES index_layer(id) ON DELETE CASCADE,
    label              TEXT NOT NULL UNIQUE,
    config_hash        TEXT NOT NULL,
    score_threshold    REAL,
    concurrency        INTEGER NOT NULL,
    request_interval_seconds REAL NOT NULL,
    model_configs_json TEXT,
    model_configs_match_index INTEGER,
    allow_config_drift INTEGER NOT NULL DEFAULT 0,
    started_at         TEXT NOT NULL,
    finished_at        TEXT,
    notes              TEXT
);

CREATE INDEX query_layer_config_hash_idx ON query_layer(config_hash);

CREATE INDEX query_layer_index_layer_idx ON query_layer(index_layer_id);

CREATE TABLE query_response (
    query_layer_id INTEGER NOT NULL REFERENCES query_layer(id) ON DELETE CASCADE,
    sample_id      TEXT NOT NULL,
    dataset        TEXT NOT NULL,
    question       TEXT NOT NULL,
    requested_at   TEXT NOT NULL,
    latency_ms     INTEGER,
    http_status    INTEGER NOT NULL,
    error          TEXT,
    response_json  TEXT,
    answer_mode    TEXT,
    PRIMARY KEY (query_layer_id, sample_id)
);

CREATE INDEX query_response_answer_mode_idx
    ON query_response(query_layer_id, dataset, answer_mode);

CREATE TABLE judge_provider (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    label         TEXT NOT NULL UNIQUE,
    base_url      TEXT NOT NULL,
    model         TEXT NOT NULL,
    params_json   TEXT,
    api_key_env   TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE eval_layer (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    query_layer_id    INTEGER NOT NULL REFERENCES query_layer(id) ON DELETE CASCADE,
    label             TEXT NOT NULL UNIQUE,
    config_hash       TEXT NOT NULL,
    ks_json           TEXT NOT NULL,
    metrics_json      TEXT NOT NULL,
    judge_provider_id INTEGER REFERENCES judge_provider(id) ON DELETE SET NULL,
    created_at        TEXT NOT NULL,
    finished_at       TEXT,
    notes             TEXT
);

CREATE INDEX eval_layer_config_hash_idx ON eval_layer(config_hash);

CREATE INDEX eval_layer_query_layer_idx ON eval_layer(query_layer_id);

CREATE TABLE sample_eval (
    eval_layer_id  INTEGER NOT NULL REFERENCES eval_layer(id) ON DELETE CASCADE,
    sample_id      TEXT NOT NULL,
    dataset        TEXT NOT NULL,
    answer_mode    TEXT,
    ok             INTEGER NOT NULL,
    http_status    INTEGER NOT NULL,
    gold_count     INTEGER NOT NULL,
    retrieved_count INTEGER NOT NULL,
    citation_count INTEGER NOT NULL,
    snippet_count  INTEGER,
    latency_ms     INTEGER,
    answer         TEXT,
    detail_json    TEXT NOT NULL,
    PRIMARY KEY (eval_layer_id, sample_id)
);

CREATE INDEX sample_eval_mode_idx ON sample_eval(eval_layer_id, dataset, answer_mode);

CREATE TABLE sample_metric (
    eval_layer_id INTEGER NOT NULL REFERENCES eval_layer(id) ON DELETE CASCADE,
    sample_id     TEXT NOT NULL,
    dataset       TEXT NOT NULL,
    metric        TEXT NOT NULL,
    value         REAL NOT NULL,
    PRIMARY KEY (eval_layer_id, sample_id, metric)
);

CREATE INDEX sample_metric_lookup_idx ON sample_metric(eval_layer_id, metric, value);

CREATE TABLE metric_summary (
    eval_layer_id INTEGER NOT NULL REFERENCES eval_layer(id) ON DELETE CASCADE,
    dataset       TEXT NOT NULL,
    scope         TEXT NOT NULL,
    metric        TEXT NOT NULL,
    value         REAL,
    sample_count  INTEGER NOT NULL,
    PRIMARY KEY (eval_layer_id, dataset, scope, metric)
);

CREATE TABLE dataset_eval (
    eval_layer_id      INTEGER NOT NULL REFERENCES eval_layer(id) ON DELETE CASCADE,
    dataset            TEXT NOT NULL,
    samples_in_subset  INTEGER NOT NULL,
    responses_evaluated INTEGER NOT NULL,
    http_failures      INTEGER NOT NULL,
    missing_responses_json TEXT NOT NULL,
    unmapped_page_ids_json TEXT NOT NULL,
    omitted_metrics_json   TEXT NOT NULL,
    omission_reason    TEXT,
    answer_mode_distribution_json TEXT NOT NULL,
    stratified_json    TEXT,
    PRIMARY KEY (eval_layer_id, dataset)
);

CREATE TABLE audit_record (
    query_layer_id  INTEGER NOT NULL REFERENCES query_layer(id) ON DELETE CASCADE,
    sample_id       TEXT NOT NULL,
    query_hash      TEXT NOT NULL,
    retrieval_mode  TEXT,
    candidate_chunk_count INTEGER,
    ranked_candidate_count INTEGER,
    filtered_chunk_count  INTEGER,
    access_policy_fallback_used INTEGER,
    metadata_json   TEXT NOT NULL,
    audit_created_at TEXT,
    copied_at       TEXT NOT NULL,
    PRIMARY KEY (query_layer_id, sample_id)
);

CREATE TABLE judge_verdict (
    eval_layer_id  INTEGER NOT NULL REFERENCES eval_layer(id) ON DELETE CASCADE,
    sample_id      TEXT NOT NULL,
    metric         TEXT NOT NULL,
    score          REAL,
    failure_kind   TEXT,
    reasoning_json TEXT,
    raw_response   TEXT,
    provider_hash  TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    judged_at      TEXT NOT NULL,
    PRIMARY KEY (eval_layer_id, sample_id, metric)
);

CREATE INDEX judge_verdict_metric_idx ON judge_verdict(eval_layer_id, metric, score);

CREATE TABLE annotation (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    level       TEXT NOT NULL CHECK (level IN ('sample', 'query_layer', 'eval_layer')),
    target_id   TEXT NOT NULL,
    author_kind TEXT NOT NULL CHECK (author_kind IN ('human', 'model')),
    author      TEXT NOT NULL,
    labels_json TEXT NOT NULL,
    note        TEXT,
    source      TEXT NOT NULL,
    confidence  REAL,
    created_at  TEXT NOT NULL
);

CREATE INDEX annotation_target_idx ON annotation(level, target_id);

CREATE TABLE task (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    stage       TEXT NOT NULL,
    status      TEXT NOT NULL,
    index_layer_id INTEGER REFERENCES index_layer(id) ON DELETE SET NULL,
    query_layer_id INTEGER REFERENCES query_layer(id) ON DELETE SET NULL,
    eval_layer_id  INTEGER REFERENCES eval_layer(id) ON DELETE SET NULL,
    argv_json   TEXT NOT NULL,
    pid         INTEGER,
    exit_code   INTEGER,
    progress_done  INTEGER NOT NULL DEFAULT 0,
    progress_total INTEGER,
    progress_note  TEXT,
    log_path    TEXT,
    created_at  TEXT NOT NULL,
    started_at  TEXT,
    finished_at TEXT,
    error       TEXT
);

CREATE INDEX task_status_idx ON task(status, created_at);

CREATE TABLE task_event (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id    INTEGER NOT NULL REFERENCES task(id) ON DELETE CASCADE,
    at         TEXT NOT NULL,
    level      TEXT NOT NULL,
    message    TEXT NOT NULL
);

CREATE INDEX task_event_task_idx ON task_event(task_id, id);

CREATE TABLE metric_definition (
    name            TEXT PRIMARY KEY,
    family          TEXT NOT NULL,
    requires_json   TEXT NOT NULL,
    kind            TEXT NOT NULL,
    higher_is_better INTEGER NOT NULL,
    description     TEXT NOT NULL
);

CREATE TABLE app_config (
    key        TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE model_provider (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    role        TEXT NOT NULL CHECK (role IN ('judge', 'analysis')),
    label       TEXT NOT NULL,
    base_url    TEXT NOT NULL,
    model       TEXT NOT NULL,
    api_key     TEXT NOT NULL DEFAULT '',
    api_key_env TEXT NOT NULL DEFAULT '',
    params_json TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    UNIQUE (role, label)
);

CREATE TABLE run_config (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    stage      TEXT NOT NULL,
    args_json  TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE badcase_analysis (
    eval_layer_id  INTEGER NOT NULL REFERENCES eval_layer(id) ON DELETE CASCADE,
    sample_id      TEXT NOT NULL,
    root_cause     TEXT NOT NULL,
    labels_json    TEXT NOT NULL,
    evidence_json  TEXT NOT NULL,
    narrative      TEXT,
    rule_based     INTEGER NOT NULL,
    provider_hash  TEXT NOT NULL DEFAULT '',
    prompt_version TEXT NOT NULL DEFAULT '',
    analyzed_at    TEXT NOT NULL,
    PRIMARY KEY (eval_layer_id, sample_id)
);

CREATE INDEX badcase_analysis_cause_idx ON badcase_analysis(eval_layer_id, root_cause);

CREATE TABLE "connection" (
    id                INTEGER PRIMARY KEY CHECK (id = 1),
    base_url          TEXT NOT NULL DEFAULT 'http://localhost:3000',
    api_prefix        TEXT NOT NULL DEFAULT '/api',
    email             TEXT NOT NULL DEFAULT '',
    password          TEXT NOT NULL DEFAULT '',
    database_url      TEXT NOT NULL DEFAULT '',
    timeout_seconds          REAL NOT NULL DEFAULT 180.0,
    concurrency              INTEGER NOT NULL DEFAULT 1,
    request_interval_seconds REAL NOT NULL DEFAULT 0.5,
    poll_interval_seconds    REAL NOT NULL DEFAULT 10.0,
    poll_timeout_seconds     REAL NOT NULL DEFAULT 7200.0,
    last_checked_at   TEXT,
    last_check_ok     INTEGER,
    last_check_role   TEXT,
    last_model_configs_json TEXT,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);

INSERT INTO connection (id, created_at, updated_at)
VALUES (1, '1970-01-01T00:00:00Z', '1970-01-01T00:00:00Z');
