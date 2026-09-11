"""各张表的读写函数。SQL 集中在这里，阶段代码不写裸 SQL。

命名约定：``upsert_*`` 幂等写单行，``replace_*`` 先删后写一整组（重跑该阶段
的语义），``record_*`` 追加一行，其余是读。

三条口径在这里定死，因为它们是「错了不报错」的那一类：

* **续跑的键**是 ``(index_layer_id, dataset, doc_id)`` / ``(query_layer_id,
  sample_id)``。doc_id 跨数据集会撞（run001 子集上 60 个），少一个维度就会把
  另一组的行误判成已完成。
* **哈希链**逐层往下带。下游读到的上游哈希与实际不符就抛错，不静默继续。
* **时间窗**从 ``query_response.requested_at`` 的 min/max 现算，不存快照 ——
  存快照会在续跑时被本次统计覆盖，审计于是漏掉早期请求。
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Sequence
from typing import Any

from ..io_utils import sha256_text, utc_now


def dumps(value: Any) -> str:
    """JSON 列的统一编码。``ensure_ascii=False`` 让中文在库里直接可读。"""
    return json.dumps(value, ensure_ascii=False, sort_keys=False)


def loads(value: str | None, default: Any = None) -> Any:
    if value is None:
        return default
    return json.loads(value)


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


# ------------------------------------------------------------------ 归一化产物


def upsert_dataset(
    connection: sqlite3.Connection,
    *,
    name: str,
    adapter: str,
    adapter_version: str,
    provides: Sequence[str],
    identity_rules: dict[str, str],
    qa_path: str,
    qa_sha256: str,
    qa_rows: int,
    corpus_path: str,
    corpus_sha256: str,
    corpus_rows: int,
    dedup_stats: dict[str, Any],
    gold_count_distribution: dict[str, int],
    unique_question_texts: int,
) -> None:
    connection.execute(
        """
        INSERT INTO dataset (
            name, adapter, adapter_version, provides_json, identity_rules_json,
            qa_path, qa_sha256, qa_rows, corpus_path, corpus_sha256, corpus_rows,
            dedup_stats_json, gold_count_distribution_json, unique_question_texts,
            normalized_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(name) DO UPDATE SET
            adapter=excluded.adapter,
            adapter_version=excluded.adapter_version,
            provides_json=excluded.provides_json,
            identity_rules_json=excluded.identity_rules_json,
            qa_path=excluded.qa_path,
            qa_sha256=excluded.qa_sha256,
            qa_rows=excluded.qa_rows,
            corpus_path=excluded.corpus_path,
            corpus_sha256=excluded.corpus_sha256,
            corpus_rows=excluded.corpus_rows,
            dedup_stats_json=excluded.dedup_stats_json,
            gold_count_distribution_json=excluded.gold_count_distribution_json,
            unique_question_texts=excluded.unique_question_texts,
            normalized_at=excluded.normalized_at
        """,
        (
            name,
            adapter,
            adapter_version,
            dumps(list(provides)),
            dumps(identity_rules),
            qa_path,
            qa_sha256,
            qa_rows,
            corpus_path,
            corpus_sha256,
            corpus_rows,
            dumps(dedup_stats),
            dumps(gold_count_distribution),
            unique_question_texts,
            utc_now(),
        ),
    )


def get_dataset(connection: sqlite3.Connection, name: str) -> dict[str, Any] | None:
    return row_to_dict(
        connection.execute("SELECT * FROM dataset WHERE name = ?", (name,)).fetchone()
    )


def list_datasets(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    return [dict(r) for r in connection.execute("SELECT * FROM dataset ORDER BY name")]


def replace_samples(
    connection: sqlite3.Connection, dataset: str, samples: Iterable[dict[str, Any]]
) -> int:
    """重写一个数据集的全部样本。**先 upsert，再删不再存在的那些。**

    不能用「先 DELETE 全部再 INSERT」—— ``subset_sample.sample_id`` 对 ``sample``
    是 ``ON DELETE CASCADE``，那一刀会把**每个已有索引层**的 QA 归属清空。
    而它不报错：``subset_doc`` 与 ``page_map`` 都还在，层看起来完好，
    直到跑查询或评测时才报「no subset samples」，那时已经很难联想到是重跑过
    一次归一化。实测踩过：run002 剩 400 篇语料、0 条样本。

    upsert 让 sample_id 不变的行原地更新，cascade 不触发；真正从上游消失的
    sample_id 才删，那时 cascade 是**应该**发生的 —— 层引用了一条不再存在的样本。
    """
    seen: list[str] = []
    for sample in samples:
        connection.execute(
            """
            INSERT INTO sample (dataset, sample_id, dataset_sample_id, question,
                                answers_json, gold_doc_ids_json, metadata_json)
            VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(sample_id) DO UPDATE SET
                dataset=excluded.dataset,
                dataset_sample_id=excluded.dataset_sample_id,
                question=excluded.question,
                answers_json=excluded.answers_json,
                gold_doc_ids_json=excluded.gold_doc_ids_json,
                metadata_json=excluded.metadata_json
            """,
            (
                dataset,
                sample["sample_id"],
                sample["dataset_sample_id"],
                sample["question"],
                dumps(list(sample["answers"])),
                dumps(list(sample["gold_doc_ids"])),
                dumps(sample["metadata"]),
            ),
        )
        seen.append(sample["sample_id"])

    # 删掉上游已不存在的。分批 IN 以避开 SQLite 的变量数上限（默认 999）。
    keep = set(seen)
    existing = [
        row["sample_id"]
        for row in connection.execute(
            "SELECT sample_id FROM sample WHERE dataset = ?", (dataset,)
        )
    ]
    stale = [sid for sid in existing if sid not in keep]
    for start in range(0, len(stale), 500):
        chunk = stale[start : start + 500]
        placeholders = ",".join("?" * len(chunk))
        connection.execute(
            f"DELETE FROM sample WHERE sample_id IN ({placeholders})", chunk
        )
    return len(seen)


def replace_corpus(
    connection: sqlite3.Connection, dataset: str, docs: Iterable[dict[str, Any]]
) -> int:
    """重写一个数据集的全部语料。

    **不去重**：musique 有重复 title 但它们是不同段落，去重会丢 gold（§3.6）。
    主键是 ``(dataset, doc_id)``，所以同名不同段落各占一行。
    """
    connection.execute("DELETE FROM corpus_doc WHERE dataset = ?", (dataset,))
    count = 0
    for doc in docs:
        connection.execute(
            "INSERT INTO corpus_doc (dataset, doc_id, title, text, text_sha256)"
            " VALUES (?,?,?,?,?)",
            (dataset, doc["doc_id"], doc["title"], doc["text"], doc["text_sha256"]),
        )
        count += 1
    return count


def samples_of(connection: sqlite3.Connection, dataset: str) -> list[dict[str, Any]]:
    """一个数据集的全部样本，字段已解回 Python 结构。"""
    return [
        _sample_row(row)
        for row in connection.execute(
            "SELECT * FROM sample WHERE dataset = ? ORDER BY sample_id", (dataset,)
        )
    ]


def _sample_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "dataset": row["dataset"],
        "sample_id": row["sample_id"],
        "dataset_sample_id": row["dataset_sample_id"],
        "question": row["question"],
        "answers": tuple(loads(row["answers_json"], [])),
        "gold_doc_ids": tuple(loads(row["gold_doc_ids_json"], [])),
        "metadata": loads(row["metadata_json"], {}),
    }


def corpus_of(connection: sqlite3.Connection, dataset: str) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in connection.execute(
            "SELECT * FROM corpus_doc WHERE dataset = ? ORDER BY doc_id", (dataset,)
        )
    ]


def corpus_doc(
    connection: sqlite3.Connection, dataset: str, doc_id: str
) -> dict[str, Any] | None:
    return row_to_dict(
        connection.execute(
            "SELECT * FROM corpus_doc WHERE dataset = ? AND doc_id = ?", (dataset, doc_id)
        ).fetchone()
    )


# -------------------------------------------------------------------- 索引层


def create_index_layer(
    connection: sqlite3.Connection,
    *,
    label: str,
    subset_hash: str,
    seed: int,
    qa_limit: int,
    negatives_ratio: float,
    narrativeqa_docs: int,
    config_hash: str | None = None,
) -> int:
    """建一个索引层，返回自增 id。

    ``label`` 唯一，用户可读；哈希判「两个层是否同一个」。两者都要：
    只有 id 则复用只能靠人记，只有哈希则 UI 里没有能念出来的名字。

    ``config_hash`` 默认为 NULL —— 它要吃 compiler 与 embedding，而 subset 阶段
    离线跑，那时拿不到模型配置。入库时由 :func:`seal_index_layer` 补上。
    """
    cursor = connection.execute(
        """
        INSERT INTO index_layer (label, subset_hash, config_hash, seed, qa_limit,
                                 negatives_ratio, narrativeqa_docs, created_at)
        VALUES (?,?,?,?,?,?,?,?)
        """,
        (
            label,
            subset_hash,
            config_hash,
            seed,
            qa_limit,
            negatives_ratio,
            narrativeqa_docs,
            utc_now(),
        ),
    )
    return int(cursor.lastrowid)


def seal_index_layer(connection: sqlite3.Connection, layer_id: int, config_hash: str) -> None:
    """入库时补上完整的 ``config_hash``。

    在此之前这一层的身份是不完整的（没有 embedding 就没有索引），
    所以这不是「改了身份」，是「身份到这一步才凑齐」。
    """
    connection.execute(
        "UPDATE index_layer SET config_hash = ? WHERE id = ?", (config_hash, layer_id)
    )


def index_layers_by_subset_hash(
    connection: sqlite3.Connection, subset_hash: str
) -> list[dict[str, Any]]:
    """同抽样配置的层。「同子集、换 embedding」的对照实验靠它找同伴。"""
    return [
        dict(r)
        for r in connection.execute(
            "SELECT * FROM index_layer WHERE subset_hash = ? ORDER BY id", (subset_hash,)
        )
    ]


def get_index_layer(connection: sqlite3.Connection, layer_id: int) -> dict[str, Any] | None:
    return row_to_dict(
        connection.execute("SELECT * FROM index_layer WHERE id = ?", (layer_id,)).fetchone()
    )


def index_layer_by_label(connection: sqlite3.Connection, label: str) -> dict[str, Any] | None:
    return row_to_dict(
        connection.execute("SELECT * FROM index_layer WHERE label = ?", (label,)).fetchone()
    )


def index_layers_by_hash(connection: sqlite3.Connection, config_hash: str) -> list[dict[str, Any]]:
    """同配置的既有层。UI 用它提示「这套配置已经建过，要不要复用」。"""
    return [
        dict(r)
        for r in connection.execute(
            "SELECT * FROM index_layer WHERE config_hash = ? ORDER BY id", (config_hash,)
        )
    ]


def list_index_layers(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    return [dict(r) for r in connection.execute("SELECT * FROM index_layer ORDER BY id DESC")]


def update_index_layer(connection: sqlite3.Connection, layer_id: int, **fields: Any) -> None:
    """按字段名更新。字段名来自代码常量，不来自请求体。"""
    if not fields:
        return
    allowed = {
        "model_configs_json",
        "workspace_id",
        "workspace_name",
        "akasha_user_id",
        "akasha_user_role",
        "connection_json",
        "subset_built_at",
        "ingested_at",
        "quality_passed",
        "notes",
    }
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError(f"cannot update index_layer fields: {sorted(unknown)}")
    assignments = ", ".join(f"{name} = ?" for name in fields)
    connection.execute(
        f"UPDATE index_layer SET {assignments} WHERE id = ?",
        (*fields.values(), layer_id),
    )


def upsert_index_layer_dataset(
    connection: sqlite3.Connection,
    layer_id: int,
    dataset: str,
    *,
    strategy: str,
    qa_count: int,
    corpus_count: int,
    gold_doc_count: int,
    negative_doc_count: int,
    strata: dict[str, int] | None,
    normalized_qa_sha256: str,
    normalized_corpus_sha256: str,
) -> None:
    connection.execute(
        """
        INSERT INTO index_layer_dataset (
            index_layer_id, dataset, strategy, qa_count, corpus_count, gold_doc_count,
            negative_doc_count, strata_json, normalized_qa_sha256, normalized_corpus_sha256
        ) VALUES (?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(index_layer_id, dataset) DO UPDATE SET
            strategy=excluded.strategy,
            qa_count=excluded.qa_count,
            corpus_count=excluded.corpus_count,
            gold_doc_count=excluded.gold_doc_count,
            negative_doc_count=excluded.negative_doc_count,
            strata_json=excluded.strata_json,
            normalized_qa_sha256=excluded.normalized_qa_sha256,
            normalized_corpus_sha256=excluded.normalized_corpus_sha256
        """,
        (
            layer_id,
            dataset,
            strategy,
            qa_count,
            corpus_count,
            gold_doc_count,
            negative_doc_count,
            dumps(strata or {}),
            normalized_qa_sha256,
            normalized_corpus_sha256,
        ),
    )


def set_space(
    connection: sqlite3.Connection,
    layer_id: int,
    dataset: str,
    *,
    space_id: str,
    space_slug: str,
    space_reused: bool,
) -> None:
    """记下这一层这个数据集用的 Space。每个数据集独立 Space，避免跨数据集
    实体合并污染结果（§6.1）。"""
    connection.execute(
        "UPDATE index_layer_dataset SET space_id = ?, space_slug = ?, space_reused = ?"
        " WHERE index_layer_id = ? AND dataset = ?",
        (space_id, space_slug, int(space_reused), layer_id, dataset),
    )


def index_layer_datasets(connection: sqlite3.Connection, layer_id: int) -> list[dict[str, Any]]:
    return [
        dict(r)
        for r in connection.execute(
            "SELECT * FROM index_layer_dataset WHERE index_layer_id = ? ORDER BY dataset",
            (layer_id,),
        )
    ]


def spaces_of(connection: sqlite3.Connection, layer_id: int) -> dict[str, str]:
    """dataset -> space_id。查询阶段要靠它把 query 打到对应的 Space。"""
    return {
        row["dataset"]: row["space_id"]
        for row in connection.execute(
            "SELECT dataset, space_id FROM index_layer_dataset"
            " WHERE index_layer_id = ? AND space_id IS NOT NULL",
            (layer_id,),
        )
    }


def replace_subset(
    connection: sqlite3.Connection,
    layer_id: int,
    dataset: str,
    sample_ids: Sequence[str],
    docs: Sequence[dict[str, Any]],
) -> None:
    """重写这一层这个数据集的子集。

    重抽样意味着旧的 md 全部作废，所以先删 —— 留着的话 ingest 会把上一次抽样的
    残留一起导进库，语料规模就悄悄变大了（原实现里这是靠 glob 删 md 处理的）。
    """
    connection.execute(
        "DELETE FROM subset_sample WHERE index_layer_id = ? AND dataset = ?",
        (layer_id, dataset),
    )
    connection.execute(
        "DELETE FROM subset_doc WHERE index_layer_id = ? AND dataset = ?", (layer_id, dataset)
    )
    connection.executemany(
        "INSERT INTO subset_sample (index_layer_id, dataset, sample_id) VALUES (?,?,?)",
        [(layer_id, dataset, sid) for sid in sample_ids],
    )
    connection.executemany(
        "INSERT INTO subset_doc (index_layer_id, dataset, doc_id, md_text, md_sha256, is_gold)"
        " VALUES (?,?,?,?,?,?)",
        [
            (layer_id, dataset, d["doc_id"], d["md_text"], d["md_sha256"], int(d["is_gold"]))
            for d in docs
        ],
    )


def subset_samples(
    connection: sqlite3.Connection, layer_id: int, dataset: str
) -> list[dict[str, Any]]:
    """子集里的样本，带上归一化层的全部字段。"""
    return [
        _sample_row(row)
        for row in connection.execute(
            """
            SELECT s.* FROM subset_sample ss
            JOIN sample s ON s.sample_id = ss.sample_id
            WHERE ss.index_layer_id = ? AND ss.dataset = ?
            ORDER BY s.sample_id
            """,
            (layer_id, dataset),
        )
    ]


def subset_docs(
    connection: sqlite3.Connection, layer_id: int, dataset: str, *, with_text: bool = True
) -> list[dict[str, Any]]:
    columns = "doc_id, md_sha256, is_gold" + (", md_text" if with_text else "")
    return [
        dict(row)
        for row in connection.execute(
            f"SELECT {columns} FROM subset_doc WHERE index_layer_id = ? AND dataset = ?"
            " ORDER BY doc_id",
            (layer_id, dataset),
        )
    ]


def recompute_subset_hash(connection: sqlite3.Connection, layer_id: int) -> str:
    """按**实际的文档集**重算 ``subset_hash`` 并写回。

    这个哈希刻意是内容寻址的，不是配置寻址的。配置寻址的版本会说谎，
    而且有两条独立的路径能让它说谎：

    1. 抽样的随机源里有配置之外的东西（曾经含 ``label``，实测同 seed 不同 label
       的两层 hotpotqa 400 篇只重叠 30 篇）。
    2. ``reindex`` 导入的是历史产物 —— 它没做抽样，凭配置算出来的哈希与实际
       导进来的那批文档没有因果关系。

    两种情况下 UI 都会照着哈希把两个不同的子集当成「同一个」并列出来做对照,
    而那种对照的结论是错的。改成对 ``(dataset, doc_id, md_sha256)`` 排序后取哈希,
    它就只能表达一件事：**这两层装的是不是同一批文档**。而这正是
    「同子集、换 embedding」那个对照实验需要判定的东西（§12.3）。

    抽样配置本身没有丢，它在 ``seed`` / ``qa_limit`` / ``negatives_ratio`` 三列里。
    """
    rows = connection.execute(
        "SELECT dataset, doc_id, md_sha256 FROM subset_doc WHERE index_layer_id = ?"
        " ORDER BY dataset, doc_id",
        (layer_id,),
    ).fetchall()
    payload = [[row["dataset"], row["doc_id"], row["md_sha256"]] for row in rows]
    digest = sha256_text(dumps(payload))[:16]
    connection.execute(
        "UPDATE index_layer SET subset_hash = ? WHERE id = ?", (digest, layer_id)
    )
    return digest


def stale_upstream(connection: sqlite3.Connection, layer_id: int) -> list[dict[str, Any]]:
    """上游哈希链断了的数据集：归一化产物在这一层建好之后又变过。

    这是 §12.2 第 1 条（manifest 的 sha256 链）在库里的落点，而且比原来更早、
    更便宜：原实现是在导入时逐篇重算 1722 个 md 的 sha256，而这里两个字符串
    比对就能判出「normalize 换了数据快照，这一层的子集已经过期」。

    逐篇的完整性另有 :func:`corrupt_subset_docs` 负责 —— 两者查的不是一件事：
    这个查「上游变了没」，那个查「这一行自己坏了没」。
    """
    return [
        dict(row)
        for row in connection.execute(
            """
            SELECT ild.dataset,
                   ild.normalized_qa_sha256     AS layer_qa_sha256,
                   d.qa_sha256                  AS current_qa_sha256,
                   ild.normalized_corpus_sha256 AS layer_corpus_sha256,
                   d.corpus_sha256              AS current_corpus_sha256
            FROM index_layer_dataset ild
            JOIN dataset d ON d.name = ild.dataset
            WHERE ild.index_layer_id = ?
              AND (ild.normalized_qa_sha256 <> d.qa_sha256
                   OR ild.normalized_corpus_sha256 <> d.corpus_sha256)
            ORDER BY ild.dataset
            """,
            (layer_id,),
        )
    ]


def subset_doc_hashes(
    connection: sqlite3.Connection, layer_id: int, dataset: str
) -> dict[str, str]:
    return {
        row["doc_id"]: row["md_sha256"]
        for row in connection.execute(
            "SELECT doc_id, md_sha256 FROM subset_doc WHERE index_layer_id = ? AND dataset = ?",
            (layer_id, dataset),
        )
    }


def pending_imports(
    connection: sqlite3.Connection, layer_id: int, dataset: str
) -> list[dict[str, Any]]:
    """还没导入的子集文档，带 md 正文。**这就是续跑判据。**

    ``NOT EXISTS`` 的关联键是 ``(index_layer_id, dataset, doc_id)`` 三项。
    少了 dataset 这一维就会出事：doc_id 是各数据集内部的裸 ID，跨组会撞 ——
    run001 子集上 hotpotqa×2wiki 撞 15 个、hotpotqa×musique 17 个、
    2wiki×musique 28 个，共 60 个。少一维时后导入的组会盖掉前一组，
    续跑于是把前一组这些 doc 误判成已导入而跳过。
    """
    return [
        dict(row)
        for row in connection.execute(
            """
            SELECT sd.doc_id, sd.md_text, sd.md_sha256
            FROM subset_doc sd
            WHERE sd.index_layer_id = ? AND sd.dataset = ?
              AND NOT EXISTS (
                SELECT 1 FROM page_map pm
                WHERE pm.index_layer_id = sd.index_layer_id
                  AND pm.dataset = sd.dataset
                  AND pm.doc_id = sd.doc_id
              )
            ORDER BY sd.doc_id
            """,
            (layer_id, dataset),
        )
    ]


def record_page(
    connection: sqlite3.Connection,
    layer_id: int,
    dataset: str,
    *,
    doc_id: str,
    page_id: str,
    space_id: str,
    title: str | None,
    md_sha256: str,
) -> None:
    """记一条导入结果。

    用 ``INSERT`` 而非 ``INSERT OR REPLACE``：同一 ``(层, 数据集, doc)`` 被导入
    两次意味着 Akasha 里多了一个没人引用的重复 page，那是要报错的情况，
    不是要静默覆盖的情况。
    """
    connection.execute(
        """
        INSERT INTO page_map (index_layer_id, dataset, doc_id, page_id, space_id,
                              title, md_sha256, imported_at)
        VALUES (?,?,?,?,?,?,?,?)
        """,
        (layer_id, dataset, doc_id, page_id, space_id, title, md_sha256, utc_now()),
    )


def record_import_failure(
    connection: sqlite3.Connection,
    layer_id: int,
    dataset: str,
    *,
    doc_id: str,
    http_status: int | None,
    error: str | None,
) -> None:
    connection.execute(
        "INSERT INTO import_failure (index_layer_id, dataset, doc_id, http_status, error,"
        " failed_at) VALUES (?,?,?,?,?,?)",
        (layer_id, dataset, doc_id, http_status, error, utc_now()),
    )


def page_to_doc(connection: sqlite3.Connection, layer_id: int, dataset: str) -> dict[str, str]:
    """page_id -> doc_id 反查表。评测靠它把响应里的 sourcePageId 还原成语料文档。"""
    return {
        row["page_id"]: row["doc_id"]
        for row in connection.execute(
            "SELECT page_id, doc_id FROM page_map WHERE index_layer_id = ? AND dataset = ?",
            (layer_id, dataset),
        )
    }


def page_map_counts(connection: sqlite3.Connection, layer_id: int) -> dict[str, int]:
    return {
        row["dataset"]: row["n"]
        for row in connection.execute(
            "SELECT dataset, COUNT(*) AS n FROM page_map WHERE index_layer_id = ?"
            " GROUP BY dataset",
            (layer_id,),
        )
    }


def import_failures(connection: sqlite3.Connection, layer_id: int) -> list[dict[str, Any]]:
    return [
        dict(r)
        for r in connection.execute(
            "SELECT * FROM import_failure WHERE index_layer_id = ? ORDER BY failed_at",
            (layer_id,),
        )
    ]


def record_compile_run(
    connection: sqlite3.Connection,
    layer_id: int,
    *,
    accepted_run_count: int | None,
    coalesced_run_count: int | None,
    status_counts: dict[str, int] | None,
    terminal: dict[str, int] | None,
    timed_out: bool,
    requested_at: str,
    finished_at: str | None,
) -> int:
    cursor = connection.execute(
        """
        INSERT INTO compile_run (index_layer_id, requested_at, accepted_run_count,
                                 coalesced_run_count, status_counts_json, terminal_json,
                                 timed_out, finished_at)
        VALUES (?,?,?,?,?,?,?,?)
        """,
        (
            layer_id,
            requested_at,
            accepted_run_count,
            coalesced_run_count,
            dumps(status_counts or {}),
            dumps(terminal or {}),
            int(timed_out),
            finished_at,
        ),
    )
    return int(cursor.lastrowid)


def record_quality_gate(
    connection: sqlite3.Connection, layer_id: int, *, gates: dict[str, Any], report: Any
) -> tuple[bool, int]:
    """记一次质量闸门，返回 ``(是否通过, 行 id)``。

    四项计数分开存列而不是塞进 JSON：这样「字段缺失」是 NULL、「跑了且为 0」是 0,
    两者在 SQL 里就能区分。原实现里字段取不到时每一项都是 None，
    而 ``all(value == 0)`` 对空值集合返回 True —— 闸门假通过，
    然后拿一个半成品库跑出一堆没意义的指标（§6.4）。
    """
    values = [
        gates.get("missingChunkPageCount"),
        gates.get("missingEmbeddingPageCount"),
        gates.get("missingSourcePageCount"),
        gates.get("stalePageCount"),
    ]
    # 必须四项都拿到值且都是 0。缺字段（None）一律判失败。
    passed = all(value == 0 for value in values) and all(value is not None for value in values)
    cursor = connection.execute(
        """
        INSERT INTO quality_gate (index_layer_id, checked_at, missing_chunk_page_count,
            missing_embedding_page_count, missing_source_page_count, stale_page_count,
            passed, report_json)
        VALUES (?,?,?,?,?,?,?,?)
        """,
        (layer_id, utc_now(), *values, int(passed), dumps(report)),
    )
    return passed, int(cursor.lastrowid)


def index_layer_readiness(connection: sqlite3.Connection, layer_id: int) -> dict[str, Any]:
    """这一层能不能开始跑查询。返回逐项判据与一个总的 ``ready``。

    §10.1 记的缺口是：查询阶段只要求入库 manifest 存在，没检查 ``quality_passed``、
    导入完整性和 ``runs.timed_out``（Makefile 拦了一道，Python 侧没有）。
    三项各自对应一种「不报错但指标偏低」的失效：

    * 质量闸门未过 —— 索引是半成品，recall 低但不是检索的问题
    * 导入不完整 —— 有 gold 根本不在库里，那些样本的 recall 天然为 0
    * 编译超时 —— 一部分页还在 BullMQ 里排队，chunk 还没生成

    把判据放在库里而不是 Makefile 里，是因为平台的执行控制不走 make。
    """
    layer = get_index_layer(connection, layer_id)
    if layer is None:
        return {"ready": False, "reasons": [f"index layer #{layer_id} does not exist"]}

    gate = latest_quality_gate(connection, layer_id)
    compile_run = latest_compile_run(connection, layer_id)
    # **按集合比，不按条数比。** 条数相等而集合不同是可能的 —— 在已入库的层上
    # 重抽子集就会造出这种状态（``replace_subset`` 不动 page_map），而那时
    # 每条检索指标都是 0，看起来像检索烂到极点。见 subset_page_map_divergence。
    shortfall = [
        dict(row)
        for row in connection.execute(
            """
            SELECT ild.dataset,
                   ild.corpus_count AS expected,
                   (SELECT COUNT(*) FROM page_map pm
                    WHERE pm.index_layer_id = ild.index_layer_id
                      AND pm.dataset = ild.dataset) AS imported,
                   (SELECT COUNT(*) FROM subset_doc sd
                    WHERE sd.index_layer_id = ild.index_layer_id
                      AND sd.dataset = ild.dataset
                      AND NOT EXISTS (
                          SELECT 1 FROM page_map pm
                          WHERE pm.index_layer_id = sd.index_layer_id
                            AND pm.dataset = sd.dataset
                            AND pm.doc_id = sd.doc_id)) AS unmapped,
                   (SELECT COUNT(*) FROM page_map pm
                    WHERE pm.index_layer_id = ild.index_layer_id
                      AND pm.dataset = ild.dataset
                      AND NOT EXISTS (
                          SELECT 1 FROM subset_doc sd
                          WHERE sd.index_layer_id = pm.index_layer_id
                            AND sd.dataset = pm.dataset
                            AND sd.doc_id = pm.doc_id)) AS orphaned
            FROM index_layer_dataset ild
            WHERE ild.index_layer_id = ?
            ORDER BY ild.dataset
            """,
            (layer_id,),
        )
        if row["unmapped"] or row["orphaned"]
    ]

    reasons: list[str] = []
    if gate is None:
        reasons.append("the quality gate has never run (did ingest use --skip-compile?)")
    elif not gate["passed"]:
        counts = {
            "missingChunkPageCount": gate["missing_chunk_page_count"],
            "missingEmbeddingPageCount": gate["missing_embedding_page_count"],
            "missingSourcePageCount": gate["missing_source_page_count"],
            "stalePageCount": gate["stale_page_count"],
        }
        reasons.append(f"the quality gate failed: {counts}")
    if compile_run is None:
        reasons.append("no compile run was recorded")
    elif compile_run["timed_out"]:
        reasons.append("the last compile run timed out with runs still active")
    for row in shortfall:
        if row["orphaned"]:
            # 这一条是「重抽过子集」的指纹：page_map 里有子集里已经没有的文档。
            reasons.append(
                f"{row['dataset']}: the subset was re-sampled after ingest — "
                f"{row['unmapped']} document(s) in the subset were never imported and "
                f"{row['orphaned']} imported page(s) are no longer in it. Queries would "
                "run against the old corpus while metrics score against the new one, "
                "so every retrieval number would be wrong without any error. "
                "Discard this layer's ingest and re-import, or re-sample with the "
                "original parameters."
            )
        else:
            reasons.append(
                f"{row['dataset']}: imported {row['imported']} of {row['expected']} "
                f"documents ({row['unmapped']} still missing)"
            )
    if stale_upstream(connection, layer_id):
        reasons.append("normalized data changed after this layer was built")

    return {
        "ready": not reasons,
        "reasons": reasons,
        "quality_gate": gate,
        "compile_run": compile_run,
        "import_shortfall": shortfall,
        # workspace 比对不在这里：它要拿服务端 users/me 解析出的值来判，而这个函数
        # 是纯库函数、登不了 Akasha。判据在 workspace_mismatch，由 ingest / query
        # 在登录之后调。
        "workspace_recorded": layer["workspace_id"],
    }


def workspace_mismatch(
    connection: sqlite3.Connection, layer_id: int, current_workspace_id: str | None
) -> str | None:
    """这一层入库时那个 workspace 是否就是现在登录到的那个。不符则返回一句说明。

    **这一道对应一个静默失效。** 层已经有 space_id 时，那些 space 与 page_map 里的
    page_id 都属于某一个 workspace。换到另一个 workspace 去跑：

    * ``ensure_space`` 找不到同 slug 的 space（``list_spaces`` 按 workspace 过滤），
      于是建一个新的并覆盖库里的 space_id
    * ``page_map`` 里的 page_id 还指向旧 workspace 的页

    之后查询照常跑，每条都召回不到 —— 看起来像「这批语料检索效果差」，
    而不像一个配置错误。所以这里必须拒绝执行，不是警告。

    ``current_workspace_id`` 来自登录后的 ``users/me``。**刻意不从配置读** ——
    workspace 由服务端决定（自建部署走 ``workspaceRepo.findFirst()``），
    让用户填一个他选不了的值只会带来填错时的误报。
    """
    layer = get_index_layer(connection, layer_id)
    if layer is None:
        return None
    # 层还没入库过（没有 space），跑在哪个 workspace 上都行。
    has_space = connection.execute(
        "SELECT 1 FROM index_layer_dataset WHERE index_layer_id = ? AND space_id IS NOT NULL "
        "LIMIT 1",
        (layer_id,),
    ).fetchone()
    if not has_space:
        return None

    recorded = (layer["workspace_id"] or "").strip()
    current = (current_workspace_id or "").strip()

    # 层是 reindex 导进来的历史数据时可能没记 workspace_id，那时无从比较 ——
    # 但 ensure_space 的 space 身份校验仍然拦得住，那一道更强。
    if not recorded or not current:
        return None
    if recorded == current:
        return None

    return (
        f"this layer was ingested into workspace {recorded}, but the current connection "
        f"just logged into workspace {current}. Its page_map rows and spaces only exist "
        "in the original workspace — running here would create duplicate spaces and "
        "leave every page_id dangling, which looks like poor retrieval rather than a "
        "misconfiguration. Point the connection back at the original deployment, "
        "or discard this layer's ingest and start over."
    )


def latest_quality_gate(connection: sqlite3.Connection, layer_id: int) -> dict[str, Any] | None:
    return row_to_dict(
        connection.execute(
            "SELECT * FROM quality_gate WHERE index_layer_id = ? ORDER BY id DESC LIMIT 1",
            (layer_id,),
        ).fetchone()
    )


def latest_compile_run(connection: sqlite3.Connection, layer_id: int) -> dict[str, Any] | None:
    return row_to_dict(
        connection.execute(
            "SELECT * FROM compile_run WHERE index_layer_id = ? ORDER BY id DESC LIMIT 1",
            (layer_id,),
        ).fetchone()
    )


# -------------------------------------------------------------------- 查询层


def create_query_layer(
    connection: sqlite3.Connection,
    *,
    index_layer_id: int,
    label: str,
    config_hash: str,
    score_threshold: float | None,
    concurrency: int,
    request_interval_seconds: float,
    model_configs: Any,
    model_configs_match_index: bool | None,
    allow_config_drift: bool,
) -> int:
    cursor = connection.execute(
        """
        INSERT INTO query_layer (index_layer_id, label, config_hash, score_threshold,
            concurrency, request_interval_seconds, model_configs_json,
            model_configs_match_index, allow_config_drift, started_at)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        """,
        (
            index_layer_id,
            label,
            config_hash,
            score_threshold,
            concurrency,
            request_interval_seconds,
            dumps(model_configs) if model_configs is not None else None,
            None if model_configs_match_index is None else int(model_configs_match_index),
            int(allow_config_drift),
            utc_now(),
        ),
    )
    return int(cursor.lastrowid)


def get_query_layer(connection: sqlite3.Connection, layer_id: int) -> dict[str, Any] | None:
    return row_to_dict(
        connection.execute("SELECT * FROM query_layer WHERE id = ?", (layer_id,)).fetchone()
    )


def query_layer_by_label(connection: sqlite3.Connection, label: str) -> dict[str, Any] | None:
    return row_to_dict(
        connection.execute("SELECT * FROM query_layer WHERE label = ?", (label,)).fetchone()
    )


def list_query_layers(
    connection: sqlite3.Connection, index_layer_id: int | None = None
) -> list[dict[str, Any]]:
    if index_layer_id is None:
        rows = connection.execute("SELECT * FROM query_layer ORDER BY id DESC")
    else:
        rows = connection.execute(
            "SELECT * FROM query_layer WHERE index_layer_id = ? ORDER BY id DESC",
            (index_layer_id,),
        )
    return [dict(r) for r in rows]


def finish_query_layer(connection: sqlite3.Connection, layer_id: int) -> None:
    connection.execute(
        "UPDATE query_layer SET finished_at = ? WHERE id = ?", (utc_now(), layer_id)
    )


def completed_sample_ids(
    connection: sqlite3.Connection, query_layer_id: int, dataset: str | None = None
) -> set[str]:
    """已有响应的 sample_id，**包括失败行**。这就是查询阶段的续跑判据。

    失败行也算已完成：重跑一次要烧 LLM 调用，而失败率本身是结果的一部分。
    要重试失败样本得显式删掉那些行（见 :func:`delete_failed_responses`）。
    """
    if dataset is None:
        rows = connection.execute(
            "SELECT sample_id FROM query_response WHERE query_layer_id = ?", (query_layer_id,)
        )
    else:
        rows = connection.execute(
            "SELECT sample_id FROM query_response WHERE query_layer_id = ? AND dataset = ?",
            (query_layer_id, dataset),
        )
    return {row["sample_id"] for row in rows}


def delete_failed_responses(
    connection: sqlite3.Connection, query_layer_id: int, dataset: str | None = None
) -> int:
    """删掉非 2xx 的响应行，让下一次运行重试它们。

    这是 §10.1「失败行恢复策略」的落点：原实现里失败行会被续跑无条件跳过,
    而简单追加又会造成重复 sample_id。现在主键是
    ``(query_layer_id, sample_id)``，重复插入直接违反约束 ——
    所以重试的唯一正道是先显式删除，删了多少行是可见的。
    """
    sql = (
        "DELETE FROM query_response WHERE query_layer_id = ?"
        " AND (http_status < 200 OR http_status >= 300)"
    )
    params: tuple[Any, ...] = (query_layer_id,)
    if dataset is not None:
        sql += " AND dataset = ?"
        params = (query_layer_id, dataset)
    return connection.execute(sql, params).rowcount


def record_response(
    connection: sqlite3.Connection,
    query_layer_id: int,
    *,
    sample_id: str,
    dataset: str,
    question: str,
    requested_at: str,
    latency_ms: int | None,
    http_status: int,
    error: str | None,
    response: Any,
) -> None:
    """落一条响应。**存完整响应体**，不是当下用得到的那几个字段（§7.2）。

    ``answer_mode`` 顺手抽成列：它是「检索指标」与「生成端拒答」的分界,
    每次筛选都去解析 JSON 太贵。抽取失败不影响落盘 —— 原始 JSON 仍是权威。
    """
    answer_mode = response.get("answerMode") if isinstance(response, dict) else None
    connection.execute(
        """
        INSERT INTO query_response (query_layer_id, sample_id, dataset, question,
            requested_at, latency_ms, http_status, error, response_json, answer_mode)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        """,
        (
            query_layer_id,
            sample_id,
            dataset,
            question,
            requested_at,
            latency_ms,
            http_status,
            error,
            dumps(response) if response is not None else None,
            answer_mode,
        ),
    )


def responses_of(
    connection: sqlite3.Connection, query_layer_id: int, dataset: str
) -> list[dict[str, Any]]:
    """一个数据集的全部响应，``response`` 已解回 dict。"""
    return [
        {
            "sample_id": row["sample_id"],
            "dataset": row["dataset"],
            "question": row["question"],
            "requested_at": row["requested_at"],
            "latency_ms": row["latency_ms"],
            "http_status": row["http_status"],
            "error": row["error"],
            "answer_mode": row["answer_mode"],
            "response": loads(row["response_json"]),
        }
        for row in connection.execute(
            "SELECT * FROM query_response WHERE query_layer_id = ? AND dataset = ?"
            " ORDER BY sample_id",
            (query_layer_id, dataset),
        )
    ]


def response_of(
    connection: sqlite3.Connection, query_layer_id: int, sample_id: str
) -> dict[str, Any] | None:
    row = connection.execute(
        "SELECT * FROM query_response WHERE query_layer_id = ? AND sample_id = ?",
        (query_layer_id, sample_id),
    ).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["response"] = loads(row["response_json"])
    return result


def response_datasets(connection: sqlite3.Connection, query_layer_id: int) -> list[str]:
    return [
        row["dataset"]
        for row in connection.execute(
            "SELECT DISTINCT dataset FROM query_response WHERE query_layer_id = ?"
            " ORDER BY dataset",
            (query_layer_id,),
        )
    ]


def request_window(connection: sqlite3.Connection, query_layer_id: int) -> tuple[str, str] | None:
    """这一层实际请求的时间窗，从行里现算。

    **不存快照**是有意的：原实现把窗口存进 manifest，而续跑时 manifest 会被本次
    请求的统计覆盖，于是审计只看到最后一段，早期请求全部漏掉（§10.1）。
    从 ``requested_at`` 的 min/max 现算，天然覆盖累积的全部会话。
    """
    row = connection.execute(
        "SELECT MIN(requested_at) AS lo, MAX(requested_at) AS hi FROM query_response"
        " WHERE query_layer_id = ?",
        (query_layer_id,),
    ).fetchone()
    if row is None or row["lo"] is None:
        return None
    return row["lo"], row["hi"]


def response_stats(connection: sqlite3.Connection, query_layer_id: int) -> dict[str, Any]:
    """按数据集给出条数、失败数与延迟。UI 的查询层概览读它。"""
    return {
        row["dataset"]: dict(row)
        for row in connection.execute(
            """
            SELECT dataset,
                   COUNT(*) AS responses,
                   SUM(CASE WHEN http_status BETWEEN 200 AND 299 THEN 0 ELSE 1 END) AS failures,
                   AVG(CASE WHEN http_status BETWEEN 200 AND 299 THEN latency_ms END) AS latency_mean,
                   MAX(latency_ms) AS latency_max
            FROM query_response WHERE query_layer_id = ?
            GROUP BY dataset ORDER BY dataset
            """,
            (query_layer_id,),
        )
    }


# -------------------------------------------------------------------- 评测层


def create_eval_layer(
    connection: sqlite3.Connection,
    *,
    query_layer_id: int,
    label: str,
    config_hash: str,
    ks: Sequence[int],
    metrics: Sequence[str],
    judge_provider_id: int | None = None,
) -> int:
    cursor = connection.execute(
        """
        INSERT INTO eval_layer (query_layer_id, label, config_hash, ks_json, metrics_json,
                                judge_provider_id, created_at)
        VALUES (?,?,?,?,?,?,?)
        """,
        (
            query_layer_id,
            label,
            config_hash,
            dumps(list(ks)),
            dumps(list(metrics)),
            judge_provider_id,
            utc_now(),
        ),
    )
    return int(cursor.lastrowid)


def get_eval_layer(connection: sqlite3.Connection, layer_id: int) -> dict[str, Any] | None:
    return row_to_dict(
        connection.execute("SELECT * FROM eval_layer WHERE id = ?", (layer_id,)).fetchone()
    )


def eval_layer_by_label(connection: sqlite3.Connection, label: str) -> dict[str, Any] | None:
    return row_to_dict(
        connection.execute("SELECT * FROM eval_layer WHERE label = ?", (label,)).fetchone()
    )


def list_eval_layers(
    connection: sqlite3.Connection, query_layer_id: int | None = None
) -> list[dict[str, Any]]:
    if query_layer_id is None:
        rows = connection.execute("SELECT * FROM eval_layer ORDER BY id DESC")
    else:
        rows = connection.execute(
            "SELECT * FROM eval_layer WHERE query_layer_id = ? ORDER BY id DESC",
            (query_layer_id,),
        )
    return [dict(r) for r in rows]


def finish_eval_layer(connection: sqlite3.Connection, layer_id: int) -> None:
    connection.execute(
        "UPDATE eval_layer SET finished_at = ? WHERE id = ?", (utc_now(), layer_id)
    )


def clear_eval_results(connection: sqlite3.Connection, eval_layer_id: int) -> None:
    """清掉一个评测层的确定性结果，供重跑。

    **刻意不动 judge_verdict 与 annotation**：judge 判决要花钱重算，标注根本
    无法重算。重跑确定性指标不该顺手把它们清掉（§12.7）。
    """
    for table in ("sample_metric", "sample_eval", "metric_summary", "dataset_eval"):
        connection.execute(f"DELETE FROM {table} WHERE eval_layer_id = ?", (eval_layer_id,))


def record_sample_eval(
    connection: sqlite3.Connection,
    eval_layer_id: int,
    *,
    sample_id: str,
    dataset: str,
    answer_mode: str | None,
    ok: bool,
    http_status: int,
    gold_count: int,
    retrieved_count: int,
    citation_count: int,
    snippet_count: int | None,
    latency_ms: int | None,
    answer: str | None,
    detail: dict[str, Any],
) -> None:
    connection.execute(
        """
        INSERT INTO sample_eval (eval_layer_id, sample_id, dataset, answer_mode, ok,
            http_status, gold_count, retrieved_count, citation_count, snippet_count,
            latency_ms, answer, detail_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            eval_layer_id,
            sample_id,
            dataset,
            answer_mode,
            int(ok),
            http_status,
            gold_count,
            retrieved_count,
            citation_count,
            snippet_count,
            latency_ms,
            answer,
            dumps(detail),
        ),
    )


def record_sample_metrics(
    connection: sqlite3.Connection,
    eval_layer_id: int,
    sample_id: str,
    dataset: str,
    metrics: dict[str, float],
) -> None:
    """扁平写一批 (样本, 指标, 值)。UI 要按任意指标筛选排序，嵌套 JSON 做不到。"""
    connection.executemany(
        "INSERT INTO sample_metric (eval_layer_id, sample_id, dataset, metric, value)"
        " VALUES (?,?,?,?,?)",
        [
            (eval_layer_id, sample_id, dataset, name, float(value))
            for name, value in metrics.items()
        ],
    )


def record_metric_summary(
    connection: sqlite3.Connection,
    eval_layer_id: int,
    dataset: str,
    scope: str,
    metrics: dict[str, float | None],
    sample_count: int,
) -> None:
    """写一组汇总。``scope`` 是 ``overall`` / ``knowledge_only`` / ``stratum:<名>``。

    两份口径必须都存：``no_match`` 与 ``general`` 无条件返回空 retrievedSources,
    所以 ``overall`` 把「生成端拒答」也算进了检索指标，两份的差值就是这个效应
    的规模（§8）。
    """
    connection.executemany(
        "INSERT INTO metric_summary (eval_layer_id, dataset, scope, metric, value,"
        " sample_count) VALUES (?,?,?,?,?,?)",
        [
            (eval_layer_id, dataset, scope, name, None if value is None else float(value), sample_count)
            for name, value in metrics.items()
        ],
    )


def record_dataset_eval(
    connection: sqlite3.Connection,
    eval_layer_id: int,
    dataset: str,
    *,
    samples_in_subset: int,
    responses_evaluated: int,
    http_failures: int,
    missing_responses: Sequence[str],
    unmapped_page_ids: Sequence[str],
    omitted_metrics: Sequence[str],
    omission_reason: str | None,
    answer_mode_distribution: dict[str, float],
    stratified: dict[str, Any] | None,
) -> None:
    """数据集级的评测记录。

    ``omitted_metrics`` 与 ``omission_reason`` 必须成对写：narrativeqa 没有 gold,
    检索指标一律省略并写明原因，**不伪造 0 分**（§8）。少一列的话报告里就只剩
    一个空白，读者会自己填上「大概是 0」这个错误结论。
    """
    connection.execute(
        """
        INSERT INTO dataset_eval (eval_layer_id, dataset, samples_in_subset,
            responses_evaluated, http_failures, missing_responses_json,
            unmapped_page_ids_json, omitted_metrics_json, omission_reason,
            answer_mode_distribution_json, stratified_json)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            eval_layer_id,
            dataset,
            samples_in_subset,
            responses_evaluated,
            http_failures,
            dumps(list(missing_responses)),
            dumps(list(unmapped_page_ids)),
            dumps(list(omitted_metrics)),
            omission_reason,
            dumps(answer_mode_distribution),
            dumps(stratified) if stratified is not None else None,
        ),
    )


def dataset_evals(connection: sqlite3.Connection, eval_layer_id: int) -> list[dict[str, Any]]:
    return [
        dict(r)
        for r in connection.execute(
            "SELECT * FROM dataset_eval WHERE eval_layer_id = ? ORDER BY dataset",
            (eval_layer_id,),
        )
    ]


def metric_summaries(
    connection: sqlite3.Connection, eval_layer_id: int, dataset: str | None = None
) -> list[dict[str, Any]]:
    if dataset is None:
        rows = connection.execute(
            "SELECT * FROM metric_summary WHERE eval_layer_id = ?"
            " ORDER BY dataset, scope, metric",
            (eval_layer_id,),
        )
    else:
        rows = connection.execute(
            "SELECT * FROM metric_summary WHERE eval_layer_id = ? AND dataset = ?"
            " ORDER BY scope, metric",
            (eval_layer_id, dataset),
        )
    return [dict(r) for r in rows]


def sample_evals(
    connection: sqlite3.Connection,
    eval_layer_id: int,
    *,
    dataset: str | None = None,
    answer_mode: str | None = None,
    limit: int | None = None,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """逐样本评测结果，可按数据集与 answerMode 过滤。

    按 answerMode 过滤是默认视图的基础，不是可选筛选器：run001 上四条
    ``recall@5 < 1.0`` 里三条是 ``answerMode: general``（生成端回落，
    retrievedSources 被无条件清空），只有一条是真的漏 gold。混在一起看会把
    1 条检索问题读成 4 条（§12.10）。
    """
    sql = "SELECT * FROM sample_eval WHERE eval_layer_id = ?"
    params: list[Any] = [eval_layer_id]
    if dataset is not None:
        sql += " AND dataset = ?"
        params.append(dataset)
    if answer_mode is not None:
        sql += " AND answer_mode IS ?"
        params.append(answer_mode)
    sql += " ORDER BY sample_id"
    if limit is not None:
        sql += " LIMIT ? OFFSET ?"
        params += [limit, offset]
    return [dict(r) for r in connection.execute(sql, params)]


def sample_metrics_of(
    connection: sqlite3.Connection, eval_layer_id: int, sample_id: str
) -> dict[str, float]:
    return {
        row["metric"]: row["value"]
        for row in connection.execute(
            "SELECT metric, value FROM sample_metric WHERE eval_layer_id = ? AND sample_id = ?",
            (eval_layer_id, sample_id),
        )
    }


def samples_ranked_by(
    connection: sqlite3.Connection,
    eval_layer_id: int,
    metric: str,
    *,
    dataset: str | None = None,
    ascending: bool = True,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """按某个指标排序取样本，带上 answerMode。失败案例入口用它。

    带 answer_mode 是必须的：不带的话「最差的 N 条」会被生成端拒答刷满,
    而那些行的检索得分按定义就是 0，不是检索失败。
    """
    direction = "ASC" if ascending else "DESC"
    sql = f"""
        SELECT sm.sample_id, sm.dataset, sm.value, se.answer_mode, se.ok, se.answer
        FROM sample_metric sm
        JOIN sample_eval se
          ON se.eval_layer_id = sm.eval_layer_id AND se.sample_id = sm.sample_id
        WHERE sm.eval_layer_id = ? AND sm.metric = ?
        {"AND sm.dataset = ?" if dataset else ""}
        ORDER BY sm.value {direction}, sm.sample_id
        LIMIT ?
    """
    params: list[Any] = [eval_layer_id, metric]
    if dataset:
        params.append(dataset)
    params.append(limit)
    return [dict(r) for r in connection.execute(sql, params)]


# ---------------------------------------------------------------- 审计表存档


def record_audit(
    connection: sqlite3.Connection,
    query_layer_id: int,
    *,
    sample_id: str,
    query_hash: str,
    retrieval_mode: str | None,
    metadata: dict[str, Any],
    audit_created_at: str | None,
) -> None:
    """抄一条审计记录进库。

    ``knowledge_query_audit`` 是 Akasha 的运行时表，会随容器重建消失（决策 4）,
    而 ``retrievalDiagnostics`` 只存在那里 —— controller 把它从 HTTP 响应里
    解构排除了。不抄进来，容器一重建这批归因就永久没了。
    """
    connection.execute(
        """
        INSERT INTO audit_record (query_layer_id, sample_id, query_hash, retrieval_mode,
            candidate_chunk_count, ranked_candidate_count, filtered_chunk_count,
            access_policy_fallback_used, metadata_json, audit_created_at, copied_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(query_layer_id, sample_id) DO UPDATE SET
            query_hash=excluded.query_hash,
            retrieval_mode=excluded.retrieval_mode,
            candidate_chunk_count=excluded.candidate_chunk_count,
            ranked_candidate_count=excluded.ranked_candidate_count,
            filtered_chunk_count=excluded.filtered_chunk_count,
            access_policy_fallback_used=excluded.access_policy_fallback_used,
            metadata_json=excluded.metadata_json,
            audit_created_at=excluded.audit_created_at,
            copied_at=excluded.copied_at
        """,
        (
            query_layer_id,
            sample_id,
            query_hash,
            retrieval_mode,
            metadata.get("candidateChunkCount"),
            metadata.get("rankedCandidateCount"),
            metadata.get("filteredChunkCount"),
            int(bool(metadata.get("accessPolicyFallbackUsed"))),
            dumps(metadata),
            audit_created_at,
            utc_now(),
        ),
    )


def audit_records(connection: sqlite3.Connection, query_layer_id: int) -> list[dict[str, Any]]:
    return [
        dict(r)
        for r in connection.execute(
            "SELECT * FROM audit_record WHERE query_layer_id = ? ORDER BY sample_id",
            (query_layer_id,),
        )
    ]


# ------------------------------------------------- judge（不可重建，谨慎对待）


def upsert_judge_provider(
    connection: sqlite3.Connection,
    *,
    label: str,
    base_url: str,
    model: str,
    params: dict[str, Any] | None = None,
) -> int:
    """登记一个 judge provider。**只存非密字段。**

    这张表是**运行档案**而不是配置：``eval_layer.judge_provider_id`` 引用它，
    记的是「这一层用过哪个端点」。配置在 ``model_provider`` 里。

    ``api_key_env`` 列是遗留的（NOT NULL，所以填空串）—— 环境变量那条路已经删掉,
    密钥只存 ``model_provider.api_key``。为删一个列单开一次迁移不值得。
    """
    connection.execute(
        """
        INSERT INTO judge_provider (label, base_url, model, params_json, api_key_env, created_at)
        VALUES (?,?,?,?,'',?)
        ON CONFLICT(label) DO UPDATE SET
            base_url=excluded.base_url, model=excluded.model,
            params_json=excluded.params_json
        """,
        (label, base_url, model, dumps(params or {}), utc_now()),
    )
    row = connection.execute("SELECT id FROM judge_provider WHERE label = ?", (label,)).fetchone()
    return int(row["id"])


def get_judge_provider(connection: sqlite3.Connection, provider_id: int) -> dict[str, Any] | None:
    return row_to_dict(
        connection.execute(
            "SELECT * FROM judge_provider WHERE id = ?", (provider_id,)
        ).fetchone()
    )


def list_judge_providers(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    return [dict(r) for r in connection.execute("SELECT * FROM judge_provider ORDER BY id")]


def record_judge_verdict(
    connection: sqlite3.Connection,
    eval_layer_id: int,
    *,
    sample_id: str,
    metric: str,
    score: float | None,
    failure_kind: str | None,
    reasoning: Any,
    raw_response: str | None,
    provider_hash: str,
    prompt_version: str,
) -> None:
    """记一条 judge 判决。

    失败时 ``score`` 必须是 NULL 而不是 0.0，并且该条从汇总里**排除**（决策 13）:
    记 0 会让限流伪装成质量差 —— 一次 429 风暴看起来会像模型突然变笨。
    """
    connection.execute(
        """
        INSERT INTO judge_verdict (eval_layer_id, sample_id, metric, score, failure_kind,
            reasoning_json, raw_response, provider_hash, prompt_version, judged_at)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(eval_layer_id, sample_id, metric) DO UPDATE SET
            score=excluded.score, failure_kind=excluded.failure_kind,
            reasoning_json=excluded.reasoning_json, raw_response=excluded.raw_response,
            provider_hash=excluded.provider_hash, prompt_version=excluded.prompt_version,
            judged_at=excluded.judged_at
        """,
        (
            eval_layer_id,
            sample_id,
            metric,
            score,
            failure_kind,
            dumps(reasoning) if reasoning is not None else None,
            raw_response,
            provider_hash,
            prompt_version,
            utc_now(),
        ),
    )


def judged_sample_ids(
    connection: sqlite3.Connection, eval_layer_id: int, metric: str
) -> set[str]:
    """已判过且**成功**的 sample_id。失败的那些留给下次重试。"""
    return {
        row["sample_id"]
        for row in connection.execute(
            "SELECT sample_id FROM judge_verdict WHERE eval_layer_id = ? AND metric = ?"
            " AND failure_kind IS NULL",
            (eval_layer_id, metric),
        )
    }


def judge_summary(
    connection: sqlite3.Connection, eval_layer_id: int, metric: str
) -> dict[str, Any]:
    """judge 的汇总：均值只算成功的，失败按四类分开计数。

    ``mean`` 的分母是 ``scored``，不是全部样本 —— 这就是「该条排除」的落点。
    ``failure_rate`` 单独报出来，它是判断这批分数能不能信的前提。
    """
    row = connection.execute(
        """
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN failure_kind IS NULL THEN 1 ELSE 0 END) AS scored,
               AVG(CASE WHEN failure_kind IS NULL THEN score END) AS mean
        FROM judge_verdict WHERE eval_layer_id = ? AND metric = ?
        """,
        (eval_layer_id, metric),
    ).fetchone()
    failures = {
        r["failure_kind"]: r["n"]
        for r in connection.execute(
            "SELECT failure_kind, COUNT(*) AS n FROM judge_verdict"
            " WHERE eval_layer_id = ? AND metric = ? AND failure_kind IS NOT NULL"
            " GROUP BY failure_kind ORDER BY failure_kind",
            (eval_layer_id, metric),
        )
    }
    total = row["total"] or 0
    scored = row["scored"] or 0
    return {
        "metric": metric,
        "total": total,
        "scored": scored,
        "excluded": total - scored,
        "mean": row["mean"],
        "failures_by_kind": failures,
        "failure_rate": (total - scored) / total if total else 0.0,
    }


def judge_verdicts(
    connection: sqlite3.Connection, eval_layer_id: int, metric: str | None = None
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM judge_verdict WHERE eval_layer_id = ?"
    params: list[Any] = [eval_layer_id]
    if metric is not None:
        sql += " AND metric = ?"
        params.append(metric)
    return [dict(r) for r in connection.execute(sql + " ORDER BY sample_id", params)]


# ------------------------------------------ 标注（不可重建，且样本层跨 run 继承）


def add_annotation(
    connection: sqlite3.Connection,
    *,
    level: str,
    target_id: str,
    author_kind: str,
    author: str,
    labels: Sequence[str],
    note: str | None,
    source: str,
    confidence: float | None,
) -> int:
    """加一条标注。

    ``level='sample'`` 的 ``target_id`` 是 sample_id，它**跨 run 继承** ——
    样本层是资产，查询层与评测层的标注只是笔记（决策 14）。所以 target_id
    刻意不加外键：删掉一个评测层不该带走样本层的标注。

    ``author_kind`` 只有 human / model 两种，同表只差这一列 —— 于是
    judge-human 一致率是一个 GROUP BY 就能算出来的免费产物（§12.5）。
    """
    cursor = connection.execute(
        """
        INSERT INTO annotation (level, target_id, author_kind, author, labels_json, note,
                                source, confidence, created_at)
        VALUES (?,?,?,?,?,?,?,?,?)
        """,
        (
            level,
            target_id,
            author_kind,
            author,
            dumps(list(labels)),
            note,
            source,
            confidence,
            utc_now(),
        ),
    )
    return int(cursor.lastrowid)


def annotations_for(
    connection: sqlite3.Connection, level: str, target_id: str
) -> list[dict[str, Any]]:
    return [
        dict(r)
        for r in connection.execute(
            "SELECT * FROM annotation WHERE level = ? AND target_id = ? ORDER BY id",
            (level, target_id),
        )
    ]


def delete_annotation(connection: sqlite3.Connection, annotation_id: int) -> int:
    return connection.execute("DELETE FROM annotation WHERE id = ?", (annotation_id,)).rowcount


def label_agreement(connection: sqlite3.Connection, level: str = "sample") -> list[dict[str, Any]]:
    """同一目标上 human 与 model 标注的一致情况。

    这是把两者放进同一张表换来的免费产物，而它是判断「这个 LLM 归因能不能信」
    的唯一办法（§12.5）。只统计两边都标过的目标 —— 单边标注无从比较。
    """
    return [
        dict(r)
        for r in connection.execute(
            """
            SELECT h.target_id,
                   h.labels_json AS human_labels,
                   m.labels_json AS model_labels,
                   CASE WHEN h.labels_json = m.labels_json THEN 1 ELSE 0 END AS exact_match
            FROM annotation h
            JOIN annotation m
              ON m.level = h.level AND m.target_id = h.target_id AND m.author_kind = 'model'
            WHERE h.level = ? AND h.author_kind = 'human'
            ORDER BY h.target_id
            """,
            (level,),
        )
    ]


# ------------------------------------------------------------------ 任务与进度


def create_task(
    connection: sqlite3.Connection,
    *,
    stage: str,
    argv: Sequence[str],
    index_layer_id: int | None = None,
    query_layer_id: int | None = None,
    eval_layer_id: int | None = None,
    progress_total: int | None = None,
) -> int:
    cursor = connection.execute(
        """
        INSERT INTO task (stage, status, index_layer_id, query_layer_id, eval_layer_id,
                          argv_json, progress_total, created_at)
        VALUES (?, 'queued', ?,?,?,?,?,?)
        """,
        (
            stage,
            index_layer_id,
            query_layer_id,
            eval_layer_id,
            dumps(list(argv)),
            progress_total,
            utc_now(),
        ),
    )
    return int(cursor.lastrowid)


def start_task(
    connection: sqlite3.Connection, task_id: int, *, pid: int, log_path: str | None
) -> None:
    connection.execute(
        "UPDATE task SET status = 'running', pid = ?, log_path = ?, started_at = ?"
        " WHERE id = ?",
        (pid, log_path, utc_now(), task_id),
    )


def update_task_progress(
    connection: sqlite3.Connection,
    task_id: int,
    *,
    done: int | None = None,
    total: int | None = None,
    note: str | None = None,
) -> None:
    """更新进度。**必须逐批提交**，否则 Web 端在 ingest 的 15 小时里看不到动静。"""
    fields: dict[str, Any] = {}
    if done is not None:
        fields["progress_done"] = done
    if total is not None:
        fields["progress_total"] = total
    if note is not None:
        fields["progress_note"] = note
    if not fields:
        return
    assignments = ", ".join(f"{name} = ?" for name in fields)
    connection.execute(
        f"UPDATE task SET {assignments} WHERE id = ?", (*fields.values(), task_id)
    )


def finish_task(
    connection: sqlite3.Connection,
    task_id: int,
    *,
    status: str,
    exit_code: int | None,
    error: str | None = None,
) -> None:
    connection.execute(
        "UPDATE task SET status = ?, exit_code = ?, error = ?, finished_at = ? WHERE id = ?",
        (status, exit_code, error, utc_now(), task_id),
    )


def get_task(connection: sqlite3.Connection, task_id: int) -> dict[str, Any] | None:
    return row_to_dict(
        connection.execute("SELECT * FROM task WHERE id = ?", (task_id,)).fetchone()
    )


def list_tasks(
    connection: sqlite3.Connection, *, status: str | None = None, limit: int = 50
) -> list[dict[str, Any]]:
    if status is None:
        rows = connection.execute("SELECT * FROM task ORDER BY id DESC LIMIT ?", (limit,))
    else:
        rows = connection.execute(
            "SELECT * FROM task WHERE status = ? ORDER BY id DESC LIMIT ?", (status, limit)
        )
    return [dict(r) for r in rows]


def running_tasks(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    return [
        dict(r)
        for r in connection.execute(
            "SELECT * FROM task WHERE status IN ('queued','running') ORDER BY id"
        )
    ]


def add_task_event(
    connection: sqlite3.Connection, task_id: int, level: str, message: str
) -> None:
    connection.execute(
        "INSERT INTO task_event (task_id, at, level, message) VALUES (?,?,?,?)",
        (task_id, utc_now(), level, message),
    )


def task_events(
    connection: sqlite3.Connection, task_id: int, *, after_id: int = 0, limit: int = 500
) -> list[dict[str, Any]]:
    """任务日志，按 id 递增取。``after_id`` 让前端增量拉取而不是每次全量。"""
    return [
        dict(r)
        for r in connection.execute(
            "SELECT * FROM task_event WHERE task_id = ? AND id > ? ORDER BY id LIMIT ?",
            (task_id, after_id, limit),
        )
    ]


# -------------------------------------------------------------- 指标 registry


def sync_metric_definitions(
    connection: sqlite3.Connection, rows: Iterable[dict[str, Any]]
) -> None:
    """把代码里的指标声明同步进表，供前端读取。

    **计算依据始终是代码里的声明，不是这张表。** 表只是给 UI 一个可查询的副本 ——
    反过来的话，改一行 SQL 就能让闸门放行一个算不了的指标。
    """
    for row in rows:
        connection.execute(
            """
            INSERT INTO metric_definition (name, family, requires_json, kind,
                                           higher_is_better, description)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(name) DO UPDATE SET
                family=excluded.family, requires_json=excluded.requires_json,
                kind=excluded.kind, higher_is_better=excluded.higher_is_better,
                description=excluded.description
            """,
            (
                row["name"],
                row["family"],
                dumps(row["requires_json"]),
                row["kind"],
                row["higher_is_better"],
                row["description"],
            ),
        )


def list_metric_definitions(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    return [
        dict(r)
        for r in connection.execute("SELECT * FROM metric_definition ORDER BY family, name")
    ]


# -------------------------------------------------------------------- 连接
#
# **只有一份**（``CHECK (id = 1)`` 写在 schema 里），只能改，不能新增或删除。
# 这里存明文密钥，所以库文件是凭据文件。
#
# 历史记录不靠外键：``index_layer.connection_json`` 存了**入库时**那份配置的
# redacted 快照。那比一个指向当前配置的外键有用 —— 它记的是当时的值。


CONNECTION_ID = 1


def get_connection_row(connection: sqlite3.Connection) -> sqlite3.Row | None:
    """那一份连接配置。迁移保证它总是存在，所以正常路径下不会是 None。"""
    return connection.execute(
        "SELECT * FROM connection WHERE id = ?", (CONNECTION_ID,)
    ).fetchone()


def update_connection(connection: sqlite3.Connection, **fields: Any) -> None:
    """改连接配置。**只写传进来的字段** —— 没传的保持原值。

    整体替换会让「只改 base_url」的请求把密码清空，而那个失效要等到下一次
    ingest 登录失败才会发现。
    """
    if not fields:
        return
    assignments = ", ".join(f"{name} = ?" for name in fields)
    connection.execute(
        f"UPDATE connection SET {assignments}, updated_at = ? WHERE id = ?",
        [*fields.values(), utc_now(), CONNECTION_ID],
    )


def record_connection_check(
    connection: sqlite3.Connection,
    *,
    ok: bool,
    role: str | None,
    model_configs: Any = None,
) -> None:
    """记一次「测连接」的结果，连同当时那份 model_configs。

    存快照是为了让配置页能直接显示「现在编译用什么模型」，
    否则每看一眼都要打一次 Akasha。
    """
    connection.execute(
        """
        UPDATE connection SET last_checked_at = ?, last_check_ok = ?, last_check_role = ?,
                              last_model_configs_json = ?
        WHERE id = ?
        """,
        (
            utc_now(),
            int(ok),
            role,
            dumps(model_configs) if model_configs is not None else None,
            CONNECTION_ID,
        ),
    )


def ingested_layers(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    """已入库的层。改配置前要看它们 —— 改到另一个 workspace 会让它们跑不了。"""
    return [
        dict(row)
        for row in connection.execute(
            """
            SELECT id, label, workspace_id, ingested_at
            FROM index_layer WHERE ingested_at IS NOT NULL ORDER BY id
            """
        )
    ]


def discard_ingest(connection: sqlite3.Connection, layer_id: int) -> dict[str, int]:
    """清掉一层的入库产物，让它退回「已抽子集、未入库」。

    改配置改到了另一个 workspace、而这一层的 page_map 只在原来那个里有意义时,
    走这条路。**远端的 space 不删** —— 我们不删别人的数据，那些 space 留在那边,
    配置改回去还能复用。

    子集（``subset_sample`` / ``subset_doc``）不动：它是离线抽出来的，
    与连接无关。
    """
    counts: dict[str, int] = {}
    for table in ("page_map", "import_failure", "quality_gate", "compile_run"):
        cursor = connection.execute(
            f"DELETE FROM {table} WHERE index_layer_id = ?", (layer_id,)
        )
        counts[table] = cursor.rowcount
    connection.execute(
        "UPDATE index_layer_dataset SET space_id = NULL, space_slug = NULL, "
        "space_reused = NULL WHERE index_layer_id = ?",
        (layer_id,),
    )
    connection.execute(
        "UPDATE index_layer SET ingested_at = NULL, quality_passed = NULL, "
        "config_hash = NULL, workspace_id = NULL, workspace_name = NULL, "
        "akasha_user_id = NULL, akasha_user_role = NULL, connection_json = NULL, "
        "model_configs_json = NULL WHERE id = ?",
        (layer_id,),
    )
    return counts


# ------------------------------------------------------------------ 应用配置
#
# 现在只剩一个键：default_connection_id。连接本身在 connection 表里。


def get_app_config(connection: sqlite3.Connection) -> dict[str, Any]:
    """读全部配置项。空表返回空字典，由 config 层套默认值。"""
    return {
        row["key"]: loads(row["value_json"])
        for row in connection.execute("SELECT key, value_json FROM app_config")
    }


def set_app_config(connection: sqlite3.Connection, values: dict[str, Any]) -> int:
    """逐项写入。**只写传进来的键**，没传的保持原值。

    整体替换会让「只改一个 base_url」的请求把密码清空 —— 而那个失效要等到
    下一次 ingest 登录失败才会发现。
    """
    now = utc_now()
    for key, value in values.items():
        connection.execute(
            """
            INSERT INTO app_config (key, value_json, updated_at) VALUES (?,?,?)
            ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json,
                                           updated_at=excluded.updated_at
            """,
            (key, dumps(value), now),
        )
    return len(values)


def delete_app_config(connection: sqlite3.Connection, key: str) -> int:
    cursor = connection.execute("DELETE FROM app_config WHERE key = ?", (key,))
    return cursor.rowcount


# -------------------------------------------------------------- 模型 provider


def upsert_model_provider(
    connection: sqlite3.Connection,
    *,
    role: str,
    label: str,
    base_url: str,
    model: str,
    api_key: str = "",
    params: dict[str, Any] | None = None,
) -> int:
    """存一个 judge / analysis 端点。同 (role, label) 覆盖。

    密钥只有这一条来路。曾经还支持「存环境变量名、运行时从那里读」，删了 ——
    同一份密钥有两个来源时，「填了但没生效」查不出来。
    """
    now = utc_now()
    connection.execute(
        """
        INSERT INTO model_provider (role, label, base_url, model, api_key,
                                    params_json, created_at, updated_at)
        VALUES (?,?,?,?,?,?,?,?)
        ON CONFLICT(role, label) DO UPDATE SET
            base_url=excluded.base_url, model=excluded.model,
            api_key=excluded.api_key,
            params_json=excluded.params_json, updated_at=excluded.updated_at
        """,
        (role, label, base_url, model, api_key, dumps(params or {}), now, now),
    )
    row = connection.execute(
        "SELECT id FROM model_provider WHERE role = ? AND label = ?", (role, label)
    ).fetchone()
    return int(row["id"])


def list_model_providers(
    connection: sqlite3.Connection, role: str | None = None
) -> list[dict[str, Any]]:
    if role is None:
        rows = connection.execute("SELECT * FROM model_provider ORDER BY role, label")
    else:
        rows = connection.execute(
            "SELECT * FROM model_provider WHERE role = ? ORDER BY label", (role,)
        )
    return [dict(r) for r in rows]


def get_model_provider(
    connection: sqlite3.Connection, role: str, label: str | None = None
) -> dict[str, Any] | None:
    """按角色取 provider。``label`` 为空时取该角色最近更新的那个。"""
    if label:
        row = connection.execute(
            "SELECT * FROM model_provider WHERE role = ? AND label = ?", (role, label)
        ).fetchone()
    else:
        row = connection.execute(
            "SELECT * FROM model_provider WHERE role = ? ORDER BY updated_at DESC LIMIT 1",
            (role,),
        ).fetchone()
    return row_to_dict(row)


def delete_model_provider(connection: sqlite3.Connection, provider_id: int) -> int:
    cursor = connection.execute("DELETE FROM model_provider WHERE id = ?", (provider_id,))
    return cursor.rowcount


# ------------------------------------------------------------------ 运行配置


def create_run_config(
    connection: sqlite3.Connection, stage: str, args: dict[str, Any]
) -> int:
    """存一次运行的参数，返回 id。阶段进程靠 ``--run-config <id>`` 读回它。"""
    cursor = connection.execute(
        "INSERT INTO run_config (stage, args_json, created_at) VALUES (?,?,?)",
        (stage, dumps(args), utc_now()),
    )
    return int(cursor.lastrowid or 0)


def get_run_config(connection: sqlite3.Connection, run_config_id: int) -> dict[str, Any] | None:
    row = connection.execute(
        "SELECT * FROM run_config WHERE id = ?", (run_config_id,)
    ).fetchone()
    record = row_to_dict(row)
    if record is not None:
        record["args"] = loads(record["args_json"], {})
    return record


# ------------------------------------------------------------- badcase 归因


def record_badcase_analysis(
    connection: sqlite3.Connection,
    eval_layer_id: int,
    *,
    sample_id: str,
    root_cause: str,
    labels: list[str],
    evidence: dict[str, Any],
    narrative: str | None,
    rule_based: bool,
    provider_hash: str = "",
    prompt_version: str = "",
) -> None:
    """写一条归因。同 (层, 样本) 覆盖 —— 重跑归因是「换一个更好的判断」。"""
    connection.execute(
        """
        INSERT INTO badcase_analysis (eval_layer_id, sample_id, root_cause, labels_json,
                                      evidence_json, narrative, rule_based,
                                      provider_hash, prompt_version, analyzed_at)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(eval_layer_id, sample_id) DO UPDATE SET
            root_cause=excluded.root_cause, labels_json=excluded.labels_json,
            evidence_json=excluded.evidence_json, narrative=excluded.narrative,
            rule_based=excluded.rule_based, provider_hash=excluded.provider_hash,
            prompt_version=excluded.prompt_version, analyzed_at=excluded.analyzed_at
        """,
        (
            eval_layer_id,
            sample_id,
            root_cause,
            dumps(labels),
            dumps(evidence),
            narrative,
            int(rule_based),
            provider_hash,
            prompt_version,
            utc_now(),
        ),
    )


def badcase_analyses(
    connection: sqlite3.Connection, eval_layer_id: int, sample_id: str | None = None
) -> list[dict[str, Any]]:
    sql = "SELECT * FROM badcase_analysis WHERE eval_layer_id = ?"
    params: list[Any] = [eval_layer_id]
    if sample_id is not None:
        sql += " AND sample_id = ?"
        params.append(sample_id)
    return [
        {
            **{k: v for k, v in dict(row).items() if not k.endswith("_json")},
            "labels": loads(row["labels_json"], []),
            "evidence": loads(row["evidence_json"], {}),
        }
        for row in connection.execute(sql + " ORDER BY sample_id", params)
    ]


def badcase_cause_counts(connection: sqlite3.Connection, eval_layer_id: int) -> dict[str, int]:
    """各根因的样本数。归因层的总览读它。"""
    return {
        row["root_cause"]: row["n"]
        for row in connection.execute(
            """
            SELECT root_cause, COUNT(*) AS n FROM badcase_analysis
            WHERE eval_layer_id = ? GROUP BY root_cause ORDER BY n DESC
            """,
            (eval_layer_id,),
        )
    }


def delete_task(connection: sqlite3.Connection, task_id: int) -> int:
    """删一条任务记录（连同事件，走 CASCADE）。在跑的任务不许删。

    清理的语义是「这条记录不用看了」，不是「停掉它」—— 后者是 cancel。
    删一个在跑的任务会留下一个没人认领的子进程。
    """
    row = connection.execute("SELECT status FROM task WHERE id = ?", (task_id,)).fetchone()
    status = row["status"] if row else None
    if status in {"queued", "running"}:
        raise ValueError(
            f"task #{task_id} is {status}; cancel it before cleaning up. "
            "Deleting a running task would leave an orphaned subprocess."
        )
    cursor = connection.execute("DELETE FROM task WHERE id = ?", (task_id,))
    return cursor.rowcount


def delete_finished_tasks(connection: sqlite3.Connection) -> int:
    """清掉所有已终态的任务。在跑的不动。"""
    cursor = connection.execute(
        "DELETE FROM task WHERE status NOT IN ('queued', 'running')"
    )
    return cursor.rowcount
