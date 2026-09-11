-- 删掉 space_slug_prefix。
--
-- 它是个没人会动的旋钮：Space 名字的前缀，用来把评测建的 Space 与用户自己的分开。
-- 默认 `bench` 就够了，而留着它当配置项的代价是真实的 —— 库里那份被改成了
-- `111`，下一次入库会建出 `111hotpotqarun00X`，而没人会想要那个。
--
-- 改成模块级常量（ingest.SPACE_SLUG_PREFIX）。已入库的层不受影响：
-- ensure_space 拿库里记的 space_id 校验，不拿 slug 重算。

ALTER TABLE connection DROP COLUMN space_slug_prefix;
