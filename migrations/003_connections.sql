-- 连接动态化：从「一份全局配置」变成「多个连接，层各自绑一个」。
--
-- 做这件事的主要理由是一个静默失效。切换账号后重跑 ingest：
--
--   1. _find_space 走 list_spaces，那是按当前 workspace 过滤的
--   2. 新账号下找不到同 slug 的 space -> ensure_space 建一个新的
--   3. set_space 覆盖库里的 space_id
--   4. page_map 里的 page_id 还指向旧 workspace 的页
--
-- 那一层从此不可用，**且不报错**：查询照常跑，每条都召回不到，
-- 看起来像「这批语料检索效果差」。
--
-- 根因是 app_config 是单例，而 spaces_of() 拿到 space_id 后用「当前连接」去查。
-- 层里记了 workspace_id，但那是快照，没有任何地方校验。
--
-- page_map 刻意**不加** connection 维：它的键是 (index_layer_id, dataset, doc_id),
-- 而连接是层的属性，所以每行已经通过层传递地属于唯一一个连接。缺的是校验,
-- 不是结构。

-- -------------------------------------------------------------------- 连接

CREATE TABLE connection (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    label             TEXT NOT NULL UNIQUE,
    base_url          TEXT NOT NULL DEFAULT 'http://localhost:3000',
    api_prefix        TEXT NOT NULL DEFAULT '/api',
    email             TEXT NOT NULL DEFAULT '',
    -- 明文。配置要能在 UI 里填改，而环境变量做不到 —— 代价是这个库文件
    -- 成为凭据文件（已 gitignore）。
    password          TEXT NOT NULL DEFAULT '',
    -- 这个连接指向哪个空间。它同时是 page_map 的身份边界：
    -- page_id 只在特定 instance + workspace 里有意义。
    workspace_id      TEXT NOT NULL DEFAULT '',
    -- 只读 PG。跟着连接走 —— 它是这个账号背后的那个库，换实例就得换。
    database_url      TEXT NOT NULL DEFAULT '',
    space_slug_prefix TEXT NOT NULL DEFAULT 'bench',
    -- 速率与超时跟着连接走：不同实例容量不同。它们只影响跑多快、不影响跑出什么,
    -- 所以照旧不进任何 config_hash。
    timeout_seconds          REAL NOT NULL DEFAULT 180.0,
    concurrency              INTEGER NOT NULL DEFAULT 1,
    request_interval_seconds REAL NOT NULL DEFAULT 0.5,
    poll_interval_seconds    REAL NOT NULL DEFAULT 10.0,
    poll_timeout_seconds     REAL NOT NULL DEFAULT 7200.0,
    -- 上次「测连接」的结果，连同当时那份 model_configs。存快照是为了让连接列表
    -- 能直接显示「这个连接现在编译用什么模型」，不必每看一眼都打一次 Akasha。
    last_checked_at   TEXT,
    last_check_ok     INTEGER,
    last_check_role   TEXT,
    last_model_configs_json TEXT,
    created_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);

-- 层绑哪个连接。NULL = 还没定（离线抽的子集在入库前不需要知道）。
ALTER TABLE index_layer ADD COLUMN connection_id INTEGER REFERENCES connection(id);

-- ingest 首次成功后写上。之后换连接要先走 discard_ingest 清掉入库产物 ——
-- 直接换会让 page_map 的 1722 行全部指向另一个 workspace 的页。
ALTER TABLE index_layer ADD COLUMN connection_sealed_at TEXT;

-- 查询层也要记：响应里的 sourcePageId 是那个连接的 page_id，
-- 评测按它反查 page_map。
ALTER TABLE query_layer ADD COLUMN connection_id INTEGER REFERENCES connection(id);

-- ------------------------------------------------------------------ 回填
--
-- app_config 里现有的连接配置迁进 connection 的第一行。那些键之后由
-- default_connection_id 一个键取代。
--
-- json_extract 用不了：app_config.value_json 存的是 json.dumps 的结果，
-- 字符串值带引号（"http://x"），而数值不带。所以用 TRIM 剥引号 ——
-- 对数值列的字符串形态 SQLite 会在 INSERT 时按列类型转换。

INSERT INTO connection (
    label, base_url, api_prefix, email, password, workspace_id, database_url,
    space_slug_prefix, timeout_seconds, concurrency, request_interval_seconds,
    poll_interval_seconds, poll_timeout_seconds, created_at, updated_at
)
SELECT
    'default',
    COALESCE((SELECT TRIM(value_json, '"') FROM app_config WHERE key='base_url'),
             'http://localhost:3000'),
    COALESCE((SELECT TRIM(value_json, '"') FROM app_config WHERE key='api_prefix'), '/api'),
    COALESCE((SELECT TRIM(value_json, '"') FROM app_config WHERE key='email'), ''),
    COALESCE((SELECT TRIM(value_json, '"') FROM app_config WHERE key='password'), ''),
    COALESCE((SELECT TRIM(value_json, '"') FROM app_config WHERE key='workspace_id'), ''),
    COALESCE((SELECT TRIM(value_json, '"') FROM app_config WHERE key='database_url'), ''),
    COALESCE((SELECT TRIM(value_json, '"') FROM app_config WHERE key='space_slug_prefix'),
             'bench'),
    COALESCE((SELECT value_json FROM app_config WHERE key='timeout_seconds'), 180.0),
    COALESCE((SELECT value_json FROM app_config WHERE key='concurrency'), 1),
    COALESCE((SELECT value_json FROM app_config WHERE key='request_interval_seconds'), 0.5),
    COALESCE((SELECT value_json FROM app_config WHERE key='poll_interval_seconds'), 10.0),
    COALESCE((SELECT value_json FROM app_config WHERE key='poll_timeout_seconds'), 7200.0),
    -- 时间戳用库里已有的那份，没有就给一个固定值（迁移里取不到 utc_now()）。
    COALESCE((SELECT MIN(updated_at) FROM app_config), '1970-01-01T00:00:00Z'),
    COALESCE((SELECT MAX(updated_at) FROM app_config), '1970-01-01T00:00:00Z');

-- 已入库的层绑到这一行：它们的 page_map 就是在这个连接上建的。
-- 按 ingested_at 判而不是按 workspace_id 判 —— reindex 导进来的历史层可能
-- 没有 workspace_id，但它的 page_map 一样是真的。
UPDATE index_layer SET connection_id = (SELECT id FROM connection WHERE label='default')
WHERE ingested_at IS NOT NULL;

-- 已入库的层顺带封住：它们的连接已经是既成事实。
UPDATE index_layer SET connection_sealed_at = ingested_at
WHERE ingested_at IS NOT NULL;

-- 查询层跟着它的索引层。
UPDATE query_layer SET connection_id = (
    SELECT connection_id FROM index_layer WHERE index_layer.id = query_layer.index_layer_id
);

-- app_config 退化成只存指针。
DELETE FROM app_config WHERE key IN (
    'base_url', 'api_prefix', 'email', 'password', 'workspace_id', 'database_url',
    'space_slug_prefix', 'timeout_seconds', 'concurrency', 'request_interval_seconds',
    'poll_interval_seconds', 'poll_timeout_seconds'
);

INSERT INTO app_config (key, value_json, updated_at)
SELECT 'default_connection_id', CAST(id AS TEXT), '1970-01-01T00:00:00Z'
FROM connection WHERE label = 'default';
