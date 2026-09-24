"""归一化产物存取：dataset / sample / corpus_doc。"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from typing import Any

from .db import dumps, loads, utc_now


def upsert_dataset(
    connection: sqlite3.Connection,
    *,
    name: str,
    qa_sha256: str,
    qa_rows: int,
    corpus_sha256: str,
    corpus_rows: int,
) -> None:
    connection.execute(
        """
        INSERT INTO dataset
            (name, qa_sha256, qa_rows, corpus_sha256, corpus_rows, normalized_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(name) DO UPDATE SET
            qa_sha256 = excluded.qa_sha256,
            qa_rows = excluded.qa_rows,
            corpus_sha256 = excluded.corpus_sha256,
            corpus_rows = excluded.corpus_rows,
            normalized_at = excluded.normalized_at
        """,
        (name, qa_sha256, qa_rows, corpus_sha256, corpus_rows, utc_now()),
    )


def get_dataset(connection: sqlite3.Connection, name: str) -> dict[str, Any] | None:
    row = connection.execute("SELECT * FROM dataset WHERE name = ?", (name,)).fetchone()
    return dict(row) if row else None


def list_datasets(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    return [dict(r) for r in connection.execute("SELECT * FROM dataset ORDER BY name")]


def delete_dataset(connection: sqlite3.Connection, name: str) -> int:
    """删归一化产物。已被编译层引用时拒绝，否则会连带删掉编译与其下游。"""
    used = connection.execute(
        "SELECT COUNT(*) AS n FROM compile_sample cs "
        "JOIN sample s ON s.sample_id = cs.sample_id WHERE s.dataset = ?",
        (name,)
    ).fetchone()["n"]
    if used:
        raise ValueError(
            f"{name} 已被编译层引用（{used} 条样本）。先清理相关编译记录再删。"
        )
    return connection.execute("DELETE FROM dataset WHERE name = ?", (name,)).rowcount


def replace_samples(
    connection: sqlite3.Connection, dataset: str, samples: Iterable[dict[str, Any]]
) -> int:
    connection.execute("DELETE FROM sample WHERE dataset = ?", (dataset,))
    count = 0
    for sample in samples:
        connection.execute(
            """
            INSERT INTO sample
                (sample_id, dataset, dataset_sample_id, question,
                 answers_json, gold_doc_ids_json, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                sample["sample_id"],
                dataset,
                sample["dataset_sample_id"],
                sample["question"],
                dumps(list(sample["answers"])),
                dumps(list(sample["gold_doc_ids"])),
                dumps(sample["metadata"]),
            ),
        )
        count += 1
    return count


def replace_corpus(
    connection: sqlite3.Connection, dataset: str, docs: Iterable[dict[str, Any]]
) -> int:
    connection.execute("DELETE FROM corpus_doc WHERE dataset = ?", (dataset,))
    count = 0
    for doc in docs:
        connection.execute(
            "INSERT INTO corpus_doc (dataset, doc_id, title, text) VALUES (?, ?, ?, ?)",
            (dataset, doc["doc_id"], doc["title"], doc["text"]),
        )
        count += 1
    return count


def sample_from_row(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "sample_id": row["sample_id"],
        "dataset": row["dataset"],
        "dataset_sample_id": row["dataset_sample_id"],
        "question": row["question"],
        "answers": loads(row["answers_json"], []),
        "gold_doc_ids": loads(row["gold_doc_ids_json"], []),
        "metadata": loads(row["metadata_json"], {}),
    }


def samples_of(connection: sqlite3.Connection, dataset: str) -> list[dict[str, Any]]:
    return [
        sample_from_row(row)
        for row in connection.execute(
            "SELECT * FROM sample WHERE dataset = ? ORDER BY sample_id", (dataset,)
        )
    ]


def sample_page(
    connection: sqlite3.Connection,
    dataset: str,
    *,
    search: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> tuple[int, list[dict[str, Any]]]:
    """过滤归一化样本；搜索覆盖问题、答案以及 gold 文档标题和正文。"""
    if search:
        # 先匹配语料，避免为每条样本重复扫描 corpus。
        needle = search.lower()
        matched_doc_rows = connection.execute(
            """
            SELECT doc_id
            FROM corpus_doc
            WHERE dataset = ?
              AND (INSTR(LOWER(title), ?) > 0 OR INSTR(LOWER(text), ?) > 0)
            """,
            (dataset, needle, needle),
        ).fetchall()
        matched_doc_ids = {str(row["doc_id"]) for row in matched_doc_rows}
        candidates = connection.execute(
            "SELECT * FROM sample WHERE dataset = ? ORDER BY sample_id", (dataset,)
        ).fetchall()
        matched = []
        for row in candidates:
            gold_doc_ids = loads(row["gold_doc_ids_json"], [])
            if (
                needle in str(row["sample_id"]).lower()
                or needle in str(row["question"]).lower()
                or needle in str(row["answers_json"]).lower()
                or any(str(doc_id) in matched_doc_ids for doc_id in gold_doc_ids)
            ):
                matched.append(row)

        page = matched[offset : offset + limit]
        titles_by_id: dict[str, str] = {}
        page_doc_ids = {
            str(doc_id)
            for row in page
            for doc_id in loads(row["gold_doc_ids_json"], [])
        }
        if page_doc_ids:
            placeholders = ", ".join("?" for _ in page_doc_ids)
            title_rows = connection.execute(
                f"SELECT doc_id, title FROM corpus_doc WHERE dataset = ? "
                f"AND doc_id IN ({placeholders})",
                (dataset, *page_doc_ids),
            ).fetchall()
            titles_by_id = {str(row["doc_id"]): str(row["title"]) for row in title_rows}
        return len(matched), [
            {
                **sample_from_row(row),
                "gold_titles": [
                    titles_by_id[str(doc_id)]
                    for doc_id in loads(row["gold_doc_ids_json"], [])
                    if str(doc_id) in titles_by_id
                ],
            }
            for row in page
        ]

    where = ["s.dataset = ?"]
    params: list[Any] = [dataset]
    scope = " AND ".join(where)
    total = int(
        connection.execute(f"SELECT COUNT(*) FROM sample s WHERE {scope}", params).fetchone()[0]
    )
    rows = connection.execute(
        f"""
        SELECT s.*, COALESCE((
            SELECT json_group_array(cd.title)
            FROM json_each(s.gold_doc_ids_json) gold
            JOIN corpus_doc cd ON cd.dataset = s.dataset AND cd.doc_id = gold.value
        ), '[]') AS gold_titles_json
        FROM sample s WHERE {scope} ORDER BY s.sample_id LIMIT ? OFFSET ?
        """,
        (*params, limit, offset),
    )
    return total, [
        {**sample_from_row(row), "gold_titles": loads(row["gold_titles_json"], [])}
        for row in rows
    ]


def get_sample(connection: sqlite3.Connection, sample_id: str) -> dict[str, Any] | None:
    row = connection.execute(
        "SELECT * FROM sample WHERE sample_id = ?", (sample_id,)
    ).fetchone()
    return sample_from_row(row) if row else None


def corpus_of(connection: sqlite3.Connection, dataset: str) -> list[dict[str, Any]]:
    return [
        dict(row)
        for row in connection.execute(
            "SELECT * FROM corpus_doc WHERE dataset = ? ORDER BY doc_id", (dataset,)
        )
    ]


def corpus_page(
    connection: sqlite3.Connection,
    dataset: str,
    *,
    search: str | None = None,
    limit: int = 10,
    offset: int = 0,
) -> tuple[int, list[dict[str, Any]]]:
    """在 SQLite 内过滤并分页语料，只读取当前页的正文。"""
    where = ["dataset = ?"]
    params: list[Any] = [dataset]
    if search:
        where.append("(INSTR(LOWER(doc_id), ?) > 0 OR INSTR(LOWER(title), ?) > 0)")
        needle = search.lower()
        params.extend((needle, needle))
    scope = " AND ".join(where)
    total = int(
        connection.execute(f"SELECT COUNT(*) FROM corpus_doc WHERE {scope}", params).fetchone()[0]
    )
    rows = connection.execute(
        f"SELECT * FROM corpus_doc WHERE {scope} ORDER BY doc_id LIMIT ? OFFSET ?",
        (*params, limit, offset),
    )
    return total, [dict(row) for row in rows]


def corpus_doc(
    connection: sqlite3.Connection, dataset: str, doc_id: str
) -> dict[str, Any] | None:
    row = connection.execute(
        "SELECT * FROM corpus_doc WHERE dataset = ? AND doc_id = ?", (dataset, doc_id)
    ).fetchone()
    return dict(row) if row else None
