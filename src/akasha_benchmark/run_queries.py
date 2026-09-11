"""查询：建一个**查询层**，逐条跑 query 并把完整响应写库。

**这一步不算任何指标。** 它只负责产出证据，解释证据是评测的事。

存的是**完整响应体**，不是当下用得到的那几个字段。重跑一次要烧 LLM 调用，
所以以后想到要看某个新字段时，不该被迫重跑。

失败也照样写一行 —— 失败率本身就是一项结果，静默跳过失败会把后面
所有的平均值都算高。

开跑之前有一道**不可跳过的前置闸门**（§10.1）：索引层的质量闸门必须通过、
导入必须完整、编译不能超时。少了它，一个半成品索引会产出一份「recall 低、
拒答率高」的报告，而那看起来像配置问题，不像索引问题。

    uv run python -m akasha_benchmark.run_queries --label run002
"""

from __future__ import annotations

import argparse
import queue
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx

from . import run_args
from .akasha_client import AkashaClient, AkashaError
from .config import load_config
from .datasets import DATASET_NAMES
from .io_utils import utc_now
from .store import connect, identity, repo


def _query_one(
    client: AkashaClient,
    dataset: str,
    sample: dict[str, Any],
    space_id: str,
    score_threshold: float | None,
) -> tuple[dict[str, Any], int | None]:
    """跑一条 query，返回 ``(待落库的行, 成功时的 latency_ms)``。

    失败也返回行 —— 失败率本身就是结果。第二个返回值为 ``None`` 表示这条
    不计入延迟统计，否则超时会把分位数带偏。
    """
    requested_at = utc_now()
    try:
        response = client.query(sample["question"], [space_id], score_threshold=score_threshold)
        status, body, latency_ms = response.status, response.body, response.latency_ms
        error = None
    except (AkashaError, httpx.RequestError, OSError) as exc:
        # 连接层面的失败（超时、断连），同样写一行，记下错误。
        #
        # httpx.RequestError **不是** OSError 的子类，它走的是
        # TransportError -> RequestError -> HTTPError -> Exception。
        # 少了它，跑到一半网络抖一下整个阶段就带 traceback 崩掉，
        # 那一行也不会落库 —— 而 query() 用 raise_for_status=False，
        # 非 2xx 根本不抛，所以传输层异常是这里唯一能逃出来的东西。
        status, body, latency_ms = 0, None, 0
        error = f"{type(exc).__name__}: {exc}"

    row = {
        "sample_id": sample["sample_id"],
        "dataset": dataset,
        "question": sample["question"],
        "requested_at": requested_at,
        "latency_ms": latency_ms,
        "http_status": status,
        "error": error,
        "response": body,
    }
    ok = bool(status and 200 <= status < 300)
    return row, (latency_ms if ok else None)


def run_dataset(
    client: AkashaClient,
    connection: sqlite3.Connection,
    query_layer_id: int,
    index_layer_id: int,
    dataset: str,
    space_id: str,
    score_threshold: float | None,
    limit: int | None,
    concurrency: int = 1,
) -> dict[str, Any]:
    """跑完一个数据集的全部 query，返回该数据集的统计。"""
    samples = repo.subset_samples(connection, index_layer_id, dataset)
    if limit is not None:
        samples = samples[:limit]

    done = repo.completed_sample_ids(connection, query_layer_id, dataset)
    todo = [s for s in samples if s["sample_id"] not in done]

    failures = 0
    written = 0

    def emit(position: int, row: dict[str, Any], latency: int | None) -> None:
        nonlocal failures, written
        if latency is None:
            failures += 1
        repo.record_response(
            connection,
            query_layer_id,
            sample_id=row["sample_id"],
            dataset=row["dataset"],
            question=row["question"],
            requested_at=row["requested_at"],
            latency_ms=row["latency_ms"],
            http_status=row["http_status"],
            error=row["error"],
            response=row["response"],
        )
        # 逐条提交，Web 端才能看到进度。
        connection.commit()
        written += 1
        if position % 10 == 0 or position == len(todo):
            print(f"  {dataset}: {position}/{len(todo)} (failures={failures})")

    if concurrency <= 1:
        for position, sample in enumerate(todo, 1):
            row, latency = _query_one(client, dataset, sample, space_id, score_threshold)
            emit(position, row, latency)
    else:
        # 每个 worker 一个独立客户端：AkashaClient 的限流器用共享的
        # _last_request_at，多线程共用一个实例会互相踩，且 request_interval
        # 会退化成「一起睡、一起发」。各自持有则每个连接独立按间隔发送。
        #
        # 落库仍然只在主线程做（worker 只发请求、只返回行），所以不需要写锁,
        # sqlite 连接也不跨线程使用。
        extra = [AkashaClient(client.config) for _ in range(concurrency - 1)]
        try:
            for spare in extra:
                spare.login()
            pool: queue.Queue[AkashaClient] = queue.Queue()
            for worker_client in (client, *extra):
                pool.put(worker_client)

            def task(sample: dict[str, Any]) -> tuple[dict[str, Any], int | None]:
                borrowed = pool.get()
                try:
                    return _query_one(borrowed, dataset, sample, space_id, score_threshold)
                finally:
                    pool.put(borrowed)

            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(task, sample) for sample in todo]
                for position, future in enumerate(as_completed(futures), 1):
                    row, latency = future.result()
                    emit(position, row, latency)
        finally:
            for spare in extra:
                spare.close()

    stats = repo.response_stats(connection, query_layer_id).get(dataset) or {}
    return {
        "dataset": dataset,
        "samples": len(samples),
        "skipped_already_done": len(done & {s["sample_id"] for s in samples}),
        "requested": len(todo),
        "written": written,
        "failures": failures,
        "latency_mean": stats.get("latency_mean"),
        "latency_max": stats.get("latency_max"),
    }


def ensure_query_layer(
    connection: sqlite3.Connection,
    *,
    index_layer_id: int,
    label: str,
    score_threshold: float | None,
    concurrency: int,
    request_interval_seconds: float,
    model_configs: Any,
    model_configs_match_index: bool,
    allow_config_drift: bool,
) -> tuple[int, bool]:
    """取或建查询层，返回 ``(id, 是否新建)``。同 label 复用，这样重跑是续跑。"""
    existing = repo.query_layer_by_label(connection, label)
    if existing:
        return int(existing["id"]), False
    index_layer = repo.get_index_layer(connection, index_layer_id) or {}
    layer_id = repo.create_query_layer(
        connection,
        index_layer_id=index_layer_id,
        label=label,
        config_hash=identity.query_layer_hash(
            index_config_hash=index_layer.get("config_hash") or "",
            score_threshold=score_threshold,
            model_configs=model_configs,
        ),
        score_threshold=score_threshold,
        concurrency=concurrency,
        request_interval_seconds=request_interval_seconds,
        model_configs=model_configs,
        model_configs_match_index=model_configs_match_index,
        allow_config_drift=allow_config_drift,
    )
    connection.commit()
    return layer_id, True


def run(
    label: str,
    datasets: list[str],
    db_path: Path | None,
    score_threshold: float | None,
    limit: int | None,
    allow_config_drift: bool,
    query_label: str | None = None,
    retry_failed: bool = False,
) -> int:
    """执行查询。返回进程退出码。"""
    connection = connect(db_path)
    try:
        index_layer = repo.index_layer_by_label(connection, label)
        if index_layer is None:
            print(f"ERROR no index layer labelled {label!r}", file=sys.stderr)
            return 1
        index_layer_id = int(index_layer["id"])

        config = load_config(connection)
        config.require_credentials()
        print(f"akasha: {config.base_url} as {config.email}")

        # 不可跳过的前置闸门（§10.1）。这一道以前只在 Makefile 里，
        # 而平台的执行控制不走 make。
        readiness = repo.index_layer_readiness(connection, index_layer_id)
        if not readiness["ready"]:
            print(
                f"ERROR index layer {label!r} is not ready for querying:", file=sys.stderr
            )
            for reason in readiness["reasons"]:
                print(f"  - {reason}", file=sys.stderr)
            print(
                "A half-built index yields a report that looks like poor configuration "
                "rather than a broken index. Fix the layer first.",
                file=sys.stderr,
            )
            return 1

        with AkashaClient(config) as client:
            client.login()

            # workspace 不匹配拒绝执行。这一层的 space_id 只在它入库时那个
            # workspace 里解析得到 —— 换个地方跑，每条 query 都会打到一个空
            # space，而那不报错，只会给出一份「recall 全 0」的报告。
            me = client.current_user()
            mismatch = repo.workspace_mismatch(
                connection, index_layer_id, ((me or {}).get("workspace") or {}).get("id")
            )
            if mismatch:
                print(f"ERROR {mismatch}", file=sys.stderr)
                return 1

            # 入库和查询之间换了 embedding 或 answer 模型，两次运行就不可比了。
            # embedding 变了更糟：旧 chunk 的 embedding_profile 对不上，
            # 那些 chunk 永远召回不到，而评测会照常算出一份看着合理的坏报告。
            current_configs = client.get_model_configs()
            indexed_configs = repo.loads(index_layer["model_configs_json"])
            embedding_ok = identity.embedding_matches(current_configs, indexed_configs)
            compiler_ok = identity.compiler_matches(current_configs, indexed_configs)
            drift = current_configs != indexed_configs

            if not embedding_ok:
                # 这一项**拒绝执行**，--allow-config-drift 也不放行（§12.3）。
                print(
                    "ERROR the embedding model changed since this index layer was built. "
                    "Existing chunks carry the old embedding_profile and can never be "
                    "retrieved again, so every retrieval metric would be silently wrong. "
                    "This is not overridable; build a new index layer instead.",
                    file=sys.stderr,
                )
                return 1
            if not compiler_ok:
                print(
                    "WARNING the compiler model changed since this index layer was built. "
                    "Already-compiled artifacts stay self-consistent, but they are not "
                    "comparable with anything compiled by the new model.",
                    file=sys.stderr,
                )
            if drift and not allow_config_drift:
                print(
                    "ERROR model configs changed since ingest. Metrics would mix two "
                    "configurations. Pass --allow-config-drift to override.",
                    file=sys.stderr,
                )
                return 1

            query_layer_id, created = ensure_query_layer(
                connection,
                index_layer_id=index_layer_id,
                label=query_label or f"{label}-query",
                score_threshold=score_threshold,
                concurrency=config.concurrency,
                request_interval_seconds=config.request_interval_seconds,
                model_configs=current_configs,
                model_configs_match_index=not drift,
                allow_config_drift=allow_config_drift,
            )
            print(
                f"query layer #{query_layer_id} "
                f"({'created' if created else 'reused'}), concurrency={config.concurrency}"
            )

            if retry_failed:
                removed = repo.delete_failed_responses(connection, query_layer_id)
                connection.commit()
                print(f"cleared {removed} failed response(s) for retry")

            spaces = repo.spaces_of(connection, index_layer_id)
            results = []
            for dataset in datasets:
                if dataset not in spaces:
                    print(f"skip {dataset}: no space in this index layer", file=sys.stderr)
                    continue
                results.append(
                    run_dataset(
                        client,
                        connection,
                        query_layer_id,
                        index_layer_id,
                        dataset,
                        spaces[dataset],
                        score_threshold,
                        limit,
                        config.concurrency,
                    )
                )

        repo.finish_query_layer(connection, query_layer_id)
        connection.commit()

        for result in results:
            mean = result["latency_mean"] or 0
            print(
                f"ok   {result['dataset']:<16} samples={result['samples']:<4} "
                f"requested={result['requested']:<4} failures={result['failures']:<3} "
                f"mean={mean:.0f}ms max={result['latency_max'] or 0}ms"
            )
        window = repo.request_window(connection, query_layer_id)
        if window:
            print(f"\nrequest window (all sessions): {window[0]} .. {window[1]}")
        return 0
    finally:
        connection.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default=None, help="索引层标签")
    parser.add_argument("--query-label", default=None, help="查询层标签，默认 {label}-query")
    parser.add_argument("--dataset", action="append", choices=list(DATASET_NAMES))
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="不传则用服务端默认值；只在做敏感性分析扫参时才设",
    )
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条，用于打通流程")
    parser.add_argument("--allow-config-drift", action="store_true")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="先删掉这一层的失败行再跑，让它们重试（删了多少条会打印出来）",
    )
    run_args.add_argument(parser)

    try:
        args = run_args.apply(parser.parse_args(argv), stage="query")
        run_args.require(args.label, "label", "query")
        return run(
            args.label,
            args.dataset or list(DATASET_NAMES),
            args.db,
            args.score_threshold,
            args.limit,
            args.allow_config_drift,
            args.query_label,
            args.retry_failed,
        )
    except (AkashaError, RuntimeError, ValueError, FileNotFoundError, sqlite3.Error) as exc:
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
