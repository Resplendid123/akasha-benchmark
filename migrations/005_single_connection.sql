-- 连接收回单例：只有一份，只能改，不能新增或删除。
--
-- 多连接是我上一步想多了。收回来之后这三列成了死重 —— 它们永远指向同一行：
--
--   index_layer.connection_id
--   index_layer.connection_sealed_at
--   query_layer.connection_id
--
-- 历史记录不丢：index_layer.connection_json 本来就存了**入库时**那份配置的
-- redacted 快照（base_url / email / workspace_id / 各项超时），
-- 那比一个外键更有用 —— 它记的是当时的值，而不是「现在那一行是什么」。
--
-- workspace 那道闸门**保留**，而且现在更要紧：改这一份配置的 base_url 或 email
-- 就可能落到另一个 workspace，而已入库的层的 page_map 只在原来那个里有意义。
-- 判据仍然是登录后 users/me 解析出的值，与 index_layer.workspace_id 比对。

-- **先删引用列**。顺序不能反：那两列是指向 connection 的外键，
-- 留着的话下面的 DROP TABLE 会被 FOREIGN KEY 约束挡住。
ALTER TABLE index_layer DROP COLUMN connection_id;
ALTER TABLE index_layer DROP COLUMN connection_sealed_at;
ALTER TABLE query_layer DROP COLUMN connection_id;

-- CHECK (id = 1) 把单例写进 schema，而不是只写在代码里。
-- 建新表再拷数据：SQLite 不能给已有表加 CHECK。
CREATE TABLE connection_new (
    id                INTEGER PRIMARY KEY CHECK (id = 1),
    base_url          TEXT NOT NULL DEFAULT 'http://localhost:3000',
    api_prefix        TEXT NOT NULL DEFAULT '/api',
    email             TEXT NOT NULL DEFAULT '',
    password          TEXT NOT NULL DEFAULT '',
    database_url      TEXT NOT NULL DEFAULT '',
    space_slug_prefix TEXT NOT NULL DEFAULT 'bench',
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

-- 取 id 最小的那一行。多连接期间可能建过第二个，那种情况下留第一个 ——
-- 它是回填出来的那份，已入库的层就是在它上面跑的。
INSERT INTO connection_new (
    id, base_url, api_prefix, email, password, database_url, space_slug_prefix,
    timeout_seconds, concurrency, request_interval_seconds, poll_interval_seconds,
    poll_timeout_seconds, last_checked_at, last_check_ok, last_check_role,
    last_model_configs_json, created_at, updated_at
)
SELECT 1, base_url, api_prefix, email, password, database_url, space_slug_prefix,
       timeout_seconds, concurrency, request_interval_seconds, poll_interval_seconds,
       poll_timeout_seconds, last_checked_at, last_check_ok, last_check_role,
       last_model_configs_json, created_at, updated_at
FROM connection ORDER BY id LIMIT 1;

DROP TABLE connection;
ALTER TABLE connection_new RENAME TO connection;

-- 库里一行都没有时（全新的库）给一行默认值，这样「读配置」永远不必处理空表。
INSERT INTO connection (id, created_at, updated_at)
SELECT 1, '1970-01-01T00:00:00Z', '1970-01-01T00:00:00Z'
WHERE NOT EXISTS (SELECT 1 FROM connection);

-- default_connection_id 也不再有意义。
DELETE FROM app_config WHERE key = 'default_connection_id';
