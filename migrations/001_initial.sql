-- Akasha-Benchmark 评测平台的初始 schema。
--
-- 库是事实来源（PLAN.md §12 决策 2）。归一化产物、三层运行记录、指标与响应
-- 全部落这里；文件降级为可选导出。
--
-- 两类表要分清，reindex 只准动第一类：
--   * 产物表 —— 可从上游重算，reindex 会先清空再重建
--   * 不可重建表 —— annotation / judge_verdict / judge_provider。
--     它们是人和模型的判断，没有上游可重算，一个粗心的 DELETE 就没了。
--
-- 时间戳统一用 io_utils.utc_now() 的形态（UTC、秒精度、以 Z 结尾）。
-- JSON 列存 json.dumps(ensure_ascii=False) 的结果，列名一律带 _json 后缀。

-- ``schema_migration`` 不在这里建：它是迁移运行器自己的账本，由 migrate.py 负责。
-- 放进迁移文件的话，任何一份新建的迁移目录都会在写账本时炸掉。

-- -------------------------------------------------------- 归一化产物（全量）

CREATE TABLE dataset (
    name                         TEXT PRIMARY KEY,
    adapter                      TEXT NOT NULL,
    adapter_version              TEXT NOT NULL,
    -- 数据集「拥有」哪些数据依赖，取值是 DataDependency。指标声明 requires,
    -- 闸门做集合比对（决策 10）。
    provides_json                TEXT NOT NULL,
    identity_rules_json          TEXT NOT NULL,
    qa_path                      TEXT NOT NULL,
    qa_sha256                    TEXT NOT NULL,
    qa_rows                      INTEGER NOT NULL,
    corpus_path                  TEXT NOT NULL,
    corpus_sha256                TEXT NOT NULL,
    corpus_rows                  INTEGER NOT NULL,
    -- 只报告不执行去重：musique 的重复 title 是不同段落，去重会丢 gold。
    dedup_stats_json             TEXT NOT NULL,
    -- 去重后的 gold 篇数分布（§0.2），与原始标注条数不同。
    gold_count_distribution_json TEXT NOT NULL,
    unique_question_texts        INTEGER NOT NULL,
    normalized_at                TEXT NOT NULL
);

CREATE TABLE sample (
    dataset           TEXT NOT NULL REFERENCES dataset(name) ON DELETE CASCADE,
    sample_id         TEXT PRIMARY KEY,
    dataset_sample_id TEXT NOT NULL,
    question          TEXT NOT NULL,
    -- 统一成数组，单答案也是长度 1。musique 的 answer_aliases 已并入。
    answers_json      TEXT NOT NULL,
    -- 已去重成集合再经身份表映射（§0.1、§0.2）。
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

-- corpus 不去重（§3.6），所以 title 可以重复，查询时靠这个索引找同名段落。
CREATE INDEX corpus_doc_title_idx ON corpus_doc(dataset, title);

-- ------------------------------------------------------------------ 索引层
--
-- 由 subset 配置 + compiler + embedding 决定，约 15 小时 / 1722 篇（§12.3）。
-- 身份：自增 id 当主键（UI 里可读），另存 config_hash 判「两个层是否同一个」。
-- config_hash 只吃配置本身，不吃时间戳、不吃 api_key。

CREATE TABLE index_layer (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    label             TEXT NOT NULL UNIQUE,
    -- 两个哈希，回答两个不同的问题。分开是因为 subset 阶段离线运行，
    -- 那时还拿不到模型配置 —— 一个索引层在编译之前其身份本就不完整。
    --
    --   subset_hash  抽样配置本身。离线即可算。答「是同一个子集吗」，
    --                所以「同抽样、换 embedding」的对照实验能靠它找到同伴。
    --   config_hash  抽样配置 + compiler + embedding。**入库时才写**，
    --                在那之前是 NULL。答「是同一个索引层吗」。
    subset_hash       TEXT NOT NULL,
    config_hash       TEXT,
    seed              INTEGER NOT NULL,
    qa_limit          INTEGER NOT NULL,
    negatives_ratio   REAL NOT NULL,
    narrativeqa_docs  INTEGER NOT NULL,
    -- 四项模型配置的快照。embedding 换了旧 chunk 的 embedding_profile 就对不上,
    -- 那些 chunk 永远召回不到，所以闸门对它必须拒绝执行而不是警告（§12.3）。
    model_configs_json TEXT,
    workspace_id      TEXT,
    workspace_name    TEXT,
    akasha_user_id    TEXT,
    akasha_user_role  TEXT,
    connection_json   TEXT,
    created_at        TEXT NOT NULL,
    subset_built_at   TEXT,
    ingested_at       TEXT,
    -- 质量闸门四项计数（§6.4）。未跑时为 NULL，与「跑了且为 0」必须能区分开：
    -- 取不到值时假通过会带着半成品索引继续跑出一堆没意义的指标。
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
    -- 上游哈希链：这一层是从哪份归一化产物抽出来的。
    normalized_qa_sha256     TEXT NOT NULL,
    normalized_corpus_sha256 TEXT NOT NULL,
    PRIMARY KEY (index_layer_id, dataset)
);

-- 子集：这一层实际要评测的 QA 与要导入的语料。
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
    -- 导入 Akasha 的 Markdown 正文，以及它的 sha256。ingest 导入前重算并比对,
    -- 不符直接抛错 —— 语料在两次运行之间被改过，这份 page_map 就不能用了（§6.2）。
    md_text        TEXT NOT NULL,
    md_sha256      TEXT NOT NULL,
    is_gold        INTEGER NOT NULL,
    PRIMARY KEY (index_layer_id, dataset, doc_id)
);

-- 导入结果。续跑的键必须是 (index_layer, dataset, doc_id)：doc_id 是各数据集
-- 内部的裸 ID，跨组会撞 —— run001 子集上共 60 个（§6.2）。
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

-- 评测靠 page_id 把响应里的 sourcePageId 反查回语料文档。
CREATE INDEX page_map_page_id_idx ON page_map(index_layer_id, page_id);

CREATE TABLE import_failure (
    index_layer_id INTEGER NOT NULL REFERENCES index_layer(id) ON DELETE CASCADE,
    dataset        TEXT NOT NULL,
    doc_id         TEXT NOT NULL,
    http_status    INTEGER,
    error          TEXT,
    failed_at      TEXT NOT NULL
);

-- 编译与质量诊断。partial 必须单独记：它意味着一部分页编译成功一部分没有，
-- 指标会因此偏低但不会报错（§6.3）。
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
    -- 四项 camelCase 计数，必须全部为 0（§6.4）。分开存列而不是塞 JSON,
    -- 这样「字段缺失」是 NULL、「跑了且为 0」是 0，两者不会混。
    missing_chunk_page_count     INTEGER,
    missing_embedding_page_count INTEGER,
    missing_source_page_count    INTEGER,
    stale_page_count             INTEGER,
    passed         INTEGER NOT NULL,
    report_json    TEXT
);

-- ------------------------------------------------------------------ 查询层
--
-- 挂某个索引层 + answer 模型 + scoreThreshold + 并发度，10–14 秒 × 每条。

CREATE TABLE query_layer (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    index_layer_id     INTEGER NOT NULL REFERENCES index_layer(id) ON DELETE CASCADE,
    label              TEXT NOT NULL UNIQUE,
    config_hash        TEXT NOT NULL,
    score_threshold    REAL,
    concurrency        INTEGER NOT NULL,
    request_interval_seconds REAL NOT NULL,
    model_configs_json TEXT,
    -- 与索引层的快照比对结果。embedding 漂移必须拒绝执行，见 §12.3。
    model_configs_match_index INTEGER,
    allow_config_drift INTEGER NOT NULL DEFAULT 0,
    started_at         TEXT NOT NULL,
    finished_at        TEXT,
    notes              TEXT
);

CREATE INDEX query_layer_config_hash_idx ON query_layer(config_hash);
CREATE INDEX query_layer_index_layer_idx ON query_layer(index_layer_id);

-- 存完整响应体，不是当下用得到的那几个字段：重跑一次要烧 LLM 调用（§7.2）。
--
-- 失败也照样写一行 —— 失败率本身就是一项结果，静默跳过会把后面所有均值算高。
-- requested_at 逐行存，所以审计的时间窗可以按 min/max 从实际行构造，
-- 不再依赖会被覆盖的 manifest（修掉 §10.1 那条「续跑统计与审计时间窗」）。
CREATE TABLE query_response (
    query_layer_id INTEGER NOT NULL REFERENCES query_layer(id) ON DELETE CASCADE,
    sample_id      TEXT NOT NULL,
    dataset        TEXT NOT NULL,
    -- 与 subset 的 question 再比一次，抓「层还在、子集被原地重建过」。
    question       TEXT NOT NULL,
    requested_at   TEXT NOT NULL,
    latency_ms     INTEGER,
    http_status    INTEGER NOT NULL,
    error          TEXT,
    response_json  TEXT,
    -- 从 response_json 里抽出来的常用切分维度，避免每次筛选都解析 JSON。
    -- no_match / general 会无条件返回空 retrievedSources，所以这一列是
    -- 「检索指标」与「生成端拒答」的分界（§8 两份口径）。
    answer_mode    TEXT,
    PRIMARY KEY (query_layer_id, sample_id)
);

CREATE INDEX query_response_answer_mode_idx
    ON query_response(query_layer_id, dataset, answer_mode);

-- ------------------------------------------------------------------ 评测层
--
-- 挂某个查询层 + 指标组 + k（+ judge 配置）。确定性指标秒级，judge 有网络成本。

CREATE TABLE judge_provider (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    label         TEXT NOT NULL UNIQUE,
    base_url      TEXT NOT NULL,
    model         TEXT NOT NULL,
    params_json   TEXT,
    -- api_key 绝不落库（决策 12）：库文件不该因此变成密钥文件。
    -- 运行时从环境变量读，这里只记环境变量的名字。
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
    -- 逐样本的完整明细（reason_counts 等嵌套结构），报告与血缘视图都读它。
    detail_json    TEXT NOT NULL,
    PRIMARY KEY (eval_layer_id, sample_id)
);

CREATE INDEX sample_eval_mode_idx ON sample_eval(eval_layer_id, dataset, answer_mode);

-- 扁平存：一行一个 (样本, 指标, 值)。UI 要按任意指标筛选排序，
-- 嵌套 JSON 做不到这件事。
CREATE TABLE sample_metric (
    eval_layer_id INTEGER NOT NULL REFERENCES eval_layer(id) ON DELETE CASCADE,
    sample_id     TEXT NOT NULL,
    dataset       TEXT NOT NULL,
    metric        TEXT NOT NULL,
    value         REAL NOT NULL,
    PRIMARY KEY (eval_layer_id, sample_id, metric)
);

CREATE INDEX sample_metric_lookup_idx ON sample_metric(eval_layer_id, metric, value);

-- 汇总。scope 区分「全样本」与「仅 knowledge 切片」，以及各分层桶 ——
-- 两份的差值就是生成端拒答的规模（§8）。
CREATE TABLE metric_summary (
    eval_layer_id INTEGER NOT NULL REFERENCES eval_layer(id) ON DELETE CASCADE,
    dataset       TEXT NOT NULL,
    scope         TEXT NOT NULL,
    metric        TEXT NOT NULL,
    value         REAL,
    sample_count  INTEGER NOT NULL,
    PRIMARY KEY (eval_layer_id, dataset, scope, metric)
);

-- 数据集级的评测记录：覆盖率、失败数、省略了哪些指标及原因。
-- narrativeqa 没有 gold，检索指标一律省略并写明原因，不伪造 0 分（§8）。
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

-- retrievalDiagnostics 不在 HTTP 响应里，controller 解构时排除了它，只写进
-- knowledge_query_audit.metadata（§7.2）。那是运行时表，会随容器重建消失,
-- 所以抄进这里存档（决策 4）。
CREATE TABLE audit_record (
    query_layer_id  INTEGER NOT NULL REFERENCES query_layer(id) ON DELETE CASCADE,
    sample_id       TEXT NOT NULL,
    query_hash      TEXT NOT NULL,
    -- high_completeness 与 high_completeness_fallback 是两种召回口径,
    -- 混在一起平均会掩盖问题，所以必须按它切分报告（§8.5）。
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

-- ------------------------------------------- 不可重建：judge、归因与人工标注
--
-- judge 指标 / LLM 归因 / 人工标注是三件不同的事，产物、去向、汇总语义都不同
-- （§12.5）。judge 的产物是分数、进指标层、参与汇总；后两者是结构化标签,
-- 进标注表、不参与汇总，只差 author_kind 一列 —— 所以 judge-human 一致率
-- 是一个 GROUP BY 就能算出来的免费产物。

CREATE TABLE judge_verdict (
    eval_layer_id  INTEGER NOT NULL REFERENCES eval_layer(id) ON DELETE CASCADE,
    sample_id      TEXT NOT NULL,
    metric         TEXT NOT NULL,
    -- 失败时 score 为 NULL 且该条从汇总里排除，不记 0 ——
    -- 记 0 会让限流伪装成质量差（决策 13）。
    score          REAL,
    -- rate_limit / timeout / parse_error / refusal，四类处置完全不同（§12.5）。
    failure_kind   TEXT,
    reasoning_json TEXT,
    raw_response   TEXT,
    -- 只吃 base_url + model，绝不吃 api_key（§12.5）。
    provider_hash  TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    judged_at      TEXT NOT NULL,
    PRIMARY KEY (eval_layer_id, sample_id, metric)
);

CREATE INDEX judge_verdict_metric_idx ON judge_verdict(eval_layer_id, metric, score);

-- 三层标注，只有样本层跨 run 继承 —— 样本层是资产，另两层是笔记（决策 14）。
-- 所以 target_id 刻意不加外键：删掉一个评测层不该带走样本层的标注。
CREATE TABLE annotation (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    level       TEXT NOT NULL CHECK (level IN ('sample', 'query_layer', 'eval_layer')),
    target_id   TEXT NOT NULL,
    -- human / model。两者同表只差这一列，见上面的说明。
    author_kind TEXT NOT NULL CHECK (author_kind IN ('human', 'model')),
    author      TEXT NOT NULL,
    labels_json TEXT NOT NULL,
    note        TEXT,
    -- human / model / model_confirmed_by_human（§12.6）：任何指标结果都能追溯到
    -- 「它依赖的 gold 有多少是人确认过的」。
    source      TEXT NOT NULL,
    confidence  REAL,
    created_at  TEXT NOT NULL
);

CREATE INDEX annotation_target_idx ON annotation(level, target_id);

-- ----------------------------------------------------------- 任务与进度
--
-- 阶段任务走 subprocess，进度写库（决策 7）：15 小时的 ingest 不能与 Web 后端
-- 同生命周期。写事务必须短、逐批提交，否则 Web 端在这 15 小时里读不到进度。

CREATE TABLE task (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    stage       TEXT NOT NULL,
    -- queued / running / succeeded / failed / cancelled
    status      TEXT NOT NULL,
    index_layer_id INTEGER REFERENCES index_layer(id) ON DELETE SET NULL,
    query_layer_id INTEGER REFERENCES query_layer(id) ON DELETE SET NULL,
    eval_layer_id  INTEGER REFERENCES eval_layer(id) ON DELETE SET NULL,
    argv_json   TEXT NOT NULL,
    pid         INTEGER,
    exit_code   INTEGER,
    -- 进度：done/total 加一句人读的话。ingest 的进度条本质是
    -- 「帮我盯着别人干活」——真正在编译的是 Akasha 的 BullMQ worker（§12.8）。
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

-- ------------------------------------------------------------ 指标 registry
--
-- 决策 10：指标声明依赖、数据集声明拥有、闸门做集合比对。新增指标不再碰枚举,
-- 而「拒绝计算而不是返回 0.0」的保护自动继承。
--
-- 这张表由代码里的 METRIC_REGISTRY 在迁移后同步进来，供前端读取展示,
-- 不作为计算依据 —— 计算依据始终是代码里的声明。

CREATE TABLE metric_definition (
    name            TEXT PRIMARY KEY,
    family          TEXT NOT NULL,
    -- 依赖的 DataDependency 集合。faithfulness 是空集，所以四组都成立 ——
    -- narrativeqa 现在整组检索指标省略，judge 能填上这个洞（§12.4）。
    requires_json   TEXT NOT NULL,
    kind            TEXT NOT NULL,
    higher_is_better INTEGER NOT NULL,
    description     TEXT NOT NULL
);
