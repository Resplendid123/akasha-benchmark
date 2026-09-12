"""入库：把索引层的子集语料灌进 Akasha、完成编译、校验入库完整性。

正文从库里取（``subset_doc.md_text``），产出的 ``doc_id -> page_id`` 映射也写回库。
后续每个阶段都要靠它把响应里的 ``sourcePageId`` 反查回语料文档。

流程：
    1. 登录，并要求用户是 OWNER（非 OWNER 会在第三道授权闸门静默丢弃 chunk，
       症状看起来像召回质量差，而不是一个错误）
    2. 校验上游哈希链：归一化产物在这一层建好之后不能变过
    3. 每个数据集建独立 Space，避免跨数据集实体合并污染结果
    4. 补齐索引层的 config_hash（吃 compiler + embedding，这里才拿得到）
    5. 串行导入，逐条写库并提交，中断后可续跑而不是重来
    6. compile-spaces 绕过 1 小时静默期，然后轮询到全部终态
    7. 质量闸门：任何 missing / stale 计数非 0 或取不到值就终止

    uv run python -m akasha_benchmark.ingest --label run002 --dataset hotpotqa
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

from . import run_args
from .akasha_client import ACTIVE_RUN_STATUSES, TERMINAL_RUN_STATUSES, AkashaClient, AkashaError
from .config import load_config
from .datasets import DATASET_NAMES, get_adapter
from .io_utils import sha256_text, utc_now
from .store import connect, identity, repo

# 需要固定并记录快照的四项模型配置。
MODEL_FEATURES = ("compiler", "embedding", "answer", "image")

# 评测建的 Space 的名字前缀。**不是配置项** —— 它的唯一作用是把这个平台生成的
# Space 与用户自己的分开，而 `bench` 就够了。在线冒烟用另一个前缀（smoke），
# 跑完自己删，不碰这些。
SPACE_SLUG_PREFIX = "bench"


def _slug(prefix: str, dataset: str, label: str) -> str:
    """Space 的 slug 必须是纯字母数字、长度 2-100，所以其他字符全部剔掉。"""
    raw = f"{prefix}{dataset}{label}"
    slug = "".join(ch for ch in raw if ch.isalnum())
    if len(slug) < 2:
        raise ValueError(f"cannot build an alphanumeric slug from {raw!r}")
    return slug[:100]


def _find_space(client: AkashaClient, slug: str) -> dict[str, Any] | None:
    """按 slug 翻页查找已存在的 Space，找不到返回 None。"""
    page = 1
    while True:
        result = client.list_spaces(page=page, limit=100)
        items = result.get("items", result if isinstance(result, list) else [])
        for space in items:
            if space.get("slug") == slug:
                return space
        meta = result.get("meta") if isinstance(result, dict) else None
        if not meta or not meta.get("hasNextPage"):
            return None
        page += 1


def ensure_space(
    client: AkashaClient,
    dataset: str,
    label: str,
    prefix: str,
    expected_space_id: str | None = None,
) -> dict[str, Any]:
    """取或建该数据集本层专用的 Space，``reused`` 标明是复用还是新建。

    ``expected_space_id`` 是库里已记的那个。给了它就必须对上 —— **slug 相同
    不等于同一个 space**：``list_spaces`` 按当前 workspace 过滤，换了账号之后
    同一个 slug 会解析到另一个 workspace 里的另一个 space（或者找不到，
    于是这里建一个新的）。两种情况下 ``page_map`` 里的 page_id 都失效了，
    而查询不会报错，只会每条都召回不到。
    """
    slug = _slug(prefix, dataset, label)
    existing = _find_space(client, slug)

    if expected_space_id:
        if existing is None:
            raise RuntimeError(
                f"{dataset}: this layer's space {expected_space_id} is not visible under "
                f"the current connection (no space with slug {slug!r}). Its page_map rows "
                "point at pages in another workspace. Rebind the original connection, "
                "or discard this layer's ingest."
            )
        if existing.get("id") != expected_space_id:
            raise RuntimeError(
                f"{dataset}: slug {slug!r} resolves to space {existing.get('id')} under the "
                f"current connection, but this layer's page_map was built against "
                f"{expected_space_id}. Same slug, different space — importing here would "
                "duplicate the corpus and leave every recorded page_id dangling."
            )
        return {**existing, "reused": True}

    if existing:
        return {**existing, "reused": True}
    created = client.create_space(
        name=f"bench {dataset} {label}"[:100],
        slug=slug,
        description=f"Akasha-Benchmark {label} / {dataset}. Generated, safe to delete.",
    )
    return {**created, "reused": False}


def verify_upstream(connection: sqlite3.Connection, layer_id: int) -> None:
    """校验上游 sha256 链。断了就抛错，不静默继续。

    断链的含义是「normalize 换过数据快照，而这一层的子集是照旧快照抽的」。
    带着这种状态导入，page_map 记的身份与库里的语料对不上，之后每个指标都
    失去可追溯性 —— 而这种失效不会报错，只会给出一份看着正常的报告。
    """
    stale = repo.stale_upstream(connection, layer_id)
    if stale:
        details = ", ".join(
            f"{row['dataset']}(qa {row['layer_qa_sha256'][:12]}->"
            f"{row['current_qa_sha256'][:12]})"
            for row in stale
        )
        raise RuntimeError(
            f"normalized data changed after this index layer was built: {details}. "
            "Re-run the subset stage for this layer, or build a new layer."
        )


def import_corpus(
    client: AkashaClient,
    connection: sqlite3.Connection,
    layer_id: int,
    dataset: str,
    space_id: str,
) -> dict[str, Any]:
    """串行导入一个数据集的子集语料，并发 1。已导入的跳过（续跑）。"""
    todo = repo.pending_imports(connection, layer_id, dataset)
    expected = len(repo.subset_docs(connection, layer_id, dataset, with_text=False))
    already = expected - len(todo)
    failures = 0
    imported = 0

    for position, doc in enumerate(todo, 1):
        markdown = doc["md_text"]
        actual = sha256_text(markdown)
        # 逐行完整性：正文与它自己的 sha256 必须一致。上游是否变过由
        # verify_upstream 负责，两者查的不是一件事。
        if actual != doc["md_sha256"]:
            raise RuntimeError(
                f"{dataset}/{doc['doc_id']}: md_text does not match its recorded sha256 "
                f"({doc['md_sha256'][:12]} != {actual[:12]}). The database row is corrupt; "
                "re-run the subset stage for this layer."
            )

        try:
            page = client.import_page_text(f"{doc['doc_id']}.md", markdown, space_id)
        except AkashaError as exc:
            repo.record_import_failure(
                connection,
                layer_id,
                dataset,
                doc_id=doc["doc_id"],
                http_status=exc.status,
                error=exc.body[:300],
            )
            connection.commit()
            failures += 1
            print(f"  FAIL {dataset}/{doc['doc_id']}: HTTP {exc.status}", file=sys.stderr)
            continue

        page_id = (page or {}).get("id")
        if not page_id:
            repo.record_import_failure(
                connection,
                layer_id,
                dataset,
                doc_id=doc["doc_id"],
                http_status=200,
                error=f"no page id in response: {page!r}"[:300],
            )
            connection.commit()
            failures += 1
            continue

        repo.record_page(
            connection,
            layer_id,
            dataset,
            doc_id=doc["doc_id"],
            page_id=page_id,
            space_id=space_id,
            title=page.get("title"),
            md_sha256=actual,
        )
        # 逐条提交：Web 端要在这 15 小时里看到进度，憋到最后一次性提交
        # 等于整个入库期间库里是空的。
        connection.commit()
        imported += 1
        if position % 25 == 0 or position == len(todo):
            print(f"  {dataset}: imported {position}/{len(todo)}")

    return {
        "dataset": dataset,
        "expected": expected,
        "skipped_already_present": already,
        "imported": imported,
        "failures": failures,
    }


def wait_for_runs(client: AkashaClient, space_ids: list[str], config: Any) -> dict[str, Any]:
    """轮询到全部编译 Run 进入终态。超时也返回，由调用方决定怎么处理。

    区分 succeeded / partial / failed，**partial 也要记录** —— 它意味着
    一部分页编译成功一部分没有，指标会因此偏低但不会报错。

    编译并发不在我们手里：真正在编译的是 Akasha 的 BullMQ worker，
    观察到的约 40 秒/篇是那边的吞吐，客户端怎么调都改不了。
    """
    deadline = time.monotonic() + config.poll_timeout_seconds
    last: dict[str, Any] = {}
    while True:
        last = client.run_diagnostics_summary(space_ids)
        status_counts: dict[str, int] = last.get("statusCounts") or {}
        active = sum(count for name, count in status_counts.items() if name in ACTIVE_RUN_STATUSES)
        terminal = {n: c for n, c in status_counts.items() if n in TERMINAL_RUN_STATUSES}
        if active == 0:
            return {"status_counts": status_counts, "terminal": terminal, "timed_out": False}
        if time.monotonic() > deadline:
            print(
                f"  TIMEOUT after {config.poll_timeout_seconds}s with {active} run(s) active",
                file=sys.stderr,
            )
            return {"status_counts": status_counts, "terminal": terminal, "timed_out": True}
        print(f"  waiting: active={active} terminal={terminal}")
        time.sleep(config.poll_interval_seconds)


def check_quality(
    client: AkashaClient, connection: sqlite3.Connection, layer_id: int, space_ids: list[str]
) -> tuple[bool, dict[str, Any]]:
    """入库完整性闸门，结果写进 ``quality_gate``。

    字段名是 camelCase（``knowledge-quality.service.ts``）。取不到值时四项都是
    None，而 ``all(value == 0)`` 对空值集合返回 True —— 那正是信封没剥那次
    假通过的成因。判据收在 :func:`repo.record_quality_gate` 里，
    它要求四项都拿到值**且**都为 0。
    """
    report = client.quality_diagnostics(space_ids)
    summary = report.get("summary") or {}
    gates = {
        "missingChunkPageCount": summary.get("missingChunkPageCount"),
        "missingEmbeddingPageCount": summary.get("missingEmbeddingPageCount"),
        "missingSourcePageCount": summary.get("missingSourcePageCount"),
        "stalePageCount": summary.get("stalePageCount"),
    }
    passed, _ = repo.record_quality_gate(connection, layer_id, gates=gates, report=report)
    repo.update_index_layer(connection, layer_id, quality_passed=int(passed))
    connection.commit()
    return passed, {"gates": gates, "report": report}


def run(
    label: str,
    datasets: list[str],
    db_path: Path | None,
    skip_compile: bool = False,
) -> int:
    """执行入库。返回进程退出码：0 表示可以开始跑查询。"""
    connection = connect(db_path)
    try:
        layer = repo.index_layer_by_label(connection, label)
        if layer is None:
            print(
                f"ERROR no index layer labelled {label!r}; run "
                f"`python -m akasha_benchmark.subset --label {label}` first",
                file=sys.stderr,
            )
            return 1
        layer_id = int(layer["id"])

        config = load_config(connection)
        config.require_credentials()
        print(f"akasha: {config.base_url} as {config.email}")

        verify_upstream(connection, layer_id)

        with AkashaClient(config) as client:
            client.login()
            me = client.current_user()
            user = me.get("user") or {}
            role = user.get("role")
            # 权限不足必须在导入前拦住：否则 chunk 被静默丢掉，
            # 你会把它误判成召回质量问题，再去调一堆调不动的参数。
            if role != "owner":
                raise RuntimeError(
                    f"evaluation user role is {role!r}, not 'owner'. A non-owner loses chunks "
                    "silently at the authorization gate, which looks like poor recall. "
                    "Promote the user before ingesting."
                )
            workspace = me.get("workspace") or {}

            # workspace 不匹配拒绝执行。判据是**服务端刚解析出的** workspace,
            # 不是任何配置项 —— 见 repo.workspace_mismatch 的说明。
            # 必须在发出任何写入之前：一旦 ensure_space 跑过，库里的 space_id
            # 就已经被覆盖了。
            mismatch = repo.workspace_mismatch(connection, layer_id, workspace.get("id"))
            if mismatch:
                print(f"ERROR {mismatch}", file=sys.stderr)
                return 1

            model_configs = client.get_model_configs()

            # 索引层的身份到这一步才凑齐：config_hash 要吃 compiler 与 embedding，
            # 而 subset 阶段离线跑，拿不到它们。
            repo.seal_index_layer(
                connection,
                layer_id,
                identity.index_layer_hash(
                    subset_hash=layer["subset_hash"],
                    model_configs=model_configs,
                ),
            )
            repo.update_index_layer(
                connection,
                layer_id,
                model_configs_json=repo.dumps(model_configs),
                workspace_id=workspace.get("id"),
                workspace_name=workspace.get("name"),
                akasha_user_id=user.get("id"),
                akasha_user_role=role,
                connection_json=repo.dumps(config.redacted()),
            )
            connection.commit()

            import_results = []
            for dataset in datasets:
                get_adapter(dataset)  # 数据集名不认识就尽早失败，别等导到一半
                if not repo.subset_docs(connection, layer_id, dataset, with_text=False):
                    print(f"skip {dataset}: no subset in layer #{layer_id}", file=sys.stderr)
                    continue
                # 库里已记的 space_id 必须对上 —— slug 相同不等于同一个 space。
                recorded = next(
                    (
                        row["space_id"]
                        for row in repo.index_layer_datasets(connection, layer_id)
                        if row["dataset"] == dataset
                    ),
                    None,
                )
                space = ensure_space(client, dataset, label, SPACE_SLUG_PREFIX, recorded)
                repo.set_space(
                    connection,
                    layer_id,
                    dataset,
                    space_id=space["id"],
                    space_slug=space.get("slug") or "",
                    space_reused=bool(space["reused"]),
                )
                connection.commit()
                print(
                    f"{dataset}: space {space['id']} "
                    f"({'reused' if space['reused'] else 'created'})"
                )
                import_results.append(
                    import_corpus(client, connection, layer_id, dataset, space["id"])
                )

            space_ids = list(repo.spaces_of(connection, layer_id).values())
            passed = False
            timed_out = False

            if skip_compile:
                print("skipping compile (--skip-compile)")
            else:
                requested_at = utc_now()
                compile_result = client.compile_spaces(space_ids)
                print(
                    f"compile requested: accepted={compile_result.get('acceptedRunCount')} "
                    f"coalesced={compile_result.get('coalescedRunCount')}"
                )
                wait_result = wait_for_runs(client, space_ids, config)
                timed_out = bool(wait_result["timed_out"])
                repo.record_compile_run(
                    connection,
                    layer_id,
                    accepted_run_count=compile_result.get("acceptedRunCount"),
                    coalesced_run_count=compile_result.get("coalescedRunCount"),
                    status_counts=wait_result["status_counts"],
                    terminal=wait_result["terminal"],
                    timed_out=timed_out,
                    requested_at=requested_at,
                    finished_at=utc_now(),
                )
                connection.commit()
                print(f"runs terminal: {wait_result['terminal']}")
                passed, quality = check_quality(client, connection, layer_id, space_ids)
                print(f"quality gates: {quality['gates']} -> {'PASS' if passed else 'FAIL'}")

        repo.update_index_layer(connection, layer_id, ingested_at=utc_now())
        connection.commit()

        for result in import_results:
            mapped = result["skipped_already_present"] + result["imported"]
            marker = "ok  " if result["expected"] == mapped and not result["failures"] else "WARN"
            print(
                f"{marker} {result['dataset']:<16} expected={result['expected']} "
                f"mapped={mapped} failures={result['failures']}"
            )

        # 验收标准之一：page_map 条数 == 子集 corpus 条数。
        incomplete = [
            r["dataset"]
            for r in import_results
            if r["expected"] != r["skipped_already_present"] + r["imported"]
        ]
        if incomplete:
            print(f"\nincomplete page_map for: {incomplete}", file=sys.stderr)
            return 1
        # 超时独立触发失败退出：编译没跑完就去查询，指标偏低但不报错。
        if timed_out:
            print(
                "\ncompile runs timed out — do not proceed to the query stage. "
                "Some pages are still compiling; metrics would be silently low.",
                file=sys.stderr,
            )
            return 1
        if not skip_compile and not passed:
            print(
                "\nquality gate FAILED — do not proceed to the query stage. "
                "A half-built index produces meaningless metrics. "
                "Use POST /api/llm-wiki/admin/retry-pages for failed pages.",
                file=sys.stderr,
            )
            return 1
        if skip_compile:
            # --skip-compile 是调试入口，不等于质量验收通过。
            print(
                "\n--skip-compile: the index is incomplete and the quality gate did not run. "
                "This is NOT an accepted ingest; the query stage will refuse to start.",
                file=sys.stderr,
            )
            return 1
        return 0
    finally:
        connection.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default=None, help="索引层标签")
    parser.add_argument("--dataset", action="append", choices=list(DATASET_NAMES))
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="只导入不编译。索引处于不完整状态，且**以非零码退出** —— 它不是验收通过",
    )
    run_args.add_argument(parser)

    try:
        args = run_args.apply(parser.parse_args(argv), stage="ingest")
        run_args.require(args.label, "label", "ingest")
        return run(
            args.label,
            args.dataset or list(DATASET_NAMES),
            args.db,
            args.skip_compile,
        )
    except (AkashaError, RuntimeError, ValueError, FileNotFoundError, sqlite3.Error) as exc:
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
