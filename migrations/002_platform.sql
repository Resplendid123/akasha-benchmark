-- 平台化：配置入库、阶段参数入库、归因产物入库。
--
-- 增量迁移而不是重建 001。理由具体：page_map 里那 1722 行 page_id 是真实 Akasha
-- 实例里的行，重挣一遍约 15 小时编译。所以这份迁移只增表，不动既有数据。
--
-- 三张新表各自替掉一个「项目目录里的文件」或「命令行参数」：
--   app_config       替 akasha.config.json
--   run_config       替跑实验用的那一串命令行参数
--   badcase_analysis 归因产物，与 judge_verdict 同属不可重算

-- ------------------------------------------------------------------ 应用配置
--
-- 替代 akasha.config.json，项目目录不再留配置文件。
--
-- **这张表存明文密钥**（Akasha 密码、database_url），所以 akasha_bench.db
-- 是一个凭据文件，已在 .gitignore 里。环境变量 AKASHA_* 仍逐项覆盖它，
-- 这样 CI 可以只注入密钥而完全不写这张表。
--
-- 一行一个键，值存 JSON：配置项的类型有 str/float/int，分列存会变成一张
-- 每加一项就要迁移一次的表。

CREATE TABLE app_config (
    key        TEXT PRIMARY KEY,
    value_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- -------------------------------------------------------------- 模型 provider
--
-- judge 与归因分析共用一张表，靠 role 区分。两者都是「OpenAI 兼容端点 + 模型名」,
-- 差别只在提示词与产物去向（judge 进指标、归因进标注），配置形态完全一样。
--
-- api_key 明文入库。这与 judge_provider 的口径不同 —— 那张表只存
-- api_key_env。改口径是因为配置要能在 UI 里填改，而环境变量做不到这件事；
-- 代价是上面说的「.db 是凭据文件」。judge_provider 保留，因为
-- eval_layer.judge_provider_id 引用它。

CREATE TABLE model_provider (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    -- judge / analysis
    role        TEXT NOT NULL CHECK (role IN ('judge', 'analysis')),
    label       TEXT NOT NULL,
    base_url    TEXT NOT NULL,
    model       TEXT NOT NULL,
    api_key     TEXT NOT NULL DEFAULT '',
    -- 读密钥的环境变量名。非空则优先于 api_key —— 不想让密钥落库的人靠它。
    api_key_env TEXT NOT NULL DEFAULT '',
    params_json TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    UNIQUE (role, label)
);

-- ------------------------------------------------------------------ 运行配置
--
-- 替代跑实验用的命令行参数。平台起任务时写一行，argv 里只剩
-- `--db <path> --run-config <id>`。
--
-- 这么做的直接好处是 argv 不再随参数个数增长：原先 14 个参数逐项映射,
-- 每加一个旋钮就要同时改 tasks.py 的映射表和阶段的 argparse。
-- 更要紧的是每次运行的完整配置有了一条库里的记录，而不是散在
-- task.argv_json 的字符串数组里。

CREATE TABLE run_config (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    stage      TEXT NOT NULL,
    args_json  TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- ------------------------------------------------------------- badcase 归因
--
-- 与 judge_verdict 同属不可重算：规则部分能重算，模型部分要花钱且模型会变。
--
-- rule_based 区分两种来源。规则归因不需要任何模型配置就能出结果，
-- 所以「自动根据指标以及链路分析归因」在没配分析模型时也成立;
-- 模型归因在它之上补一段解释。两者都记，因为规则给的是分类、模型给的是因果叙述。
--
-- 同时往 annotation 里写一行 author_kind='model' —— 那让 judge-human
-- 一致率仍然是一个 GROUP BY 就能算出来的免费产物（§12.5）。

CREATE TABLE badcase_analysis (
    eval_layer_id  INTEGER NOT NULL REFERENCES eval_layer(id) ON DELETE CASCADE,
    sample_id      TEXT NOT NULL,
    -- compiled_away / retrieval_miss / citation_dropped / generation_fallback /
    -- graph_edge_missing / gold_annotation_suspect / unknown
    root_cause     TEXT NOT NULL,
    labels_json    TEXT NOT NULL,
    -- 支撑这个判断的链路证据：丢了哪些词、命中了哪些 gold、引用被截断多少等。
    evidence_json  TEXT NOT NULL,
    -- 模型给的因果叙述。规则归因时为 NULL。
    narrative      TEXT,
    -- 1 = 纯规则，0 = 模型参与。分开是因为可信度不同。
    rule_based     INTEGER NOT NULL,
    -- 只吃 base_url + model，绝不吃 api_key。规则归因时为空串。
    provider_hash  TEXT NOT NULL DEFAULT '',
    prompt_version TEXT NOT NULL DEFAULT '',
    analyzed_at    TEXT NOT NULL,
    PRIMARY KEY (eval_layer_id, sample_id)
);

CREATE INDEX badcase_analysis_cause_idx ON badcase_analysis(eval_layer_id, root_cause);
