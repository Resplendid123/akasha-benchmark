"""查询：跑 query 并把完整响应落盘。并发由 ``concurrency`` 配置决定，默认 1（串行）。

**这一步不算任何指标。** 它只负责产出证据，解释证据是评测的事。

存的是**完整响应体**，不是当下用得到的那几个字段。重跑一次要烧 LLM 调用，
所以以后想到要看某个新字段时，不该被迫重跑。

失败也照样写一行 —— 失败率本身就是一项结果，静默跳过失败会把后面
所有的平均值都算高。

    uv run python -m akasha_benchmark.run_queries --run-id run001
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx

from .akasha_client import AkashaClient, AkashaError
from .config import DEFAULT_CONFIG_PATH, load_config
from .datasets import DATASET_NAMES, CanonicalSample, subset_dir
from .datasets.resolver import DEFAULT_DATA_DIR
from .ingest import ingest_dir
from .io_utils import atomic_write_json, load_json, read_jsonl, utc_now


def responses_dir(run_id: str, data_dir: Path | None = None) -> Path:
    return (data_dir or DEFAULT_DATA_DIR) / "responses" / run_id


def _percentile(values: list[int], fraction: float) -> int:
    """最近秩法取分位数，样本量小的时候够用。"""
    if not values:
        return 0
    ordered = sorted(values)
    index = min(int(round(fraction * (len(ordered) - 1))), len(ordered) - 1)
    return ordered[index]


def _completed_sample_ids(path: Path) -> set[str]:
    """断点续跑：已经有行的 sample_id 不再重新请求。"""
    if not path.is_file():
        return set()
    return {row["sample_id"] for row in read_jsonl(path) if row.get("sample_id")}


def _query_one(
    client: AkashaClient,
    dataset: str,
    sample: CanonicalSample,
    space_id: str,
    score_threshold: float | None,
) -> tuple[dict[str, Any], int | None]:
    """跑一条 query，返回 ``(待落盘的行, 成功时的 latency_ms)``。

    失败也返回行 —— 失败率本身就是结果，见模块 docstring。第二个返回值为
    ``None`` 表示这条不计入延迟统计，否则超时会把 p95 带偏。
    """
    requested_at = utc_now()
    try:
        response = client.query(sample.question, [space_id], score_threshold=score_threshold)
        status, body, latency_ms = response.status, response.body, response.latency_ms
        error = None
    except (AkashaError, httpx.RequestError, OSError) as exc:
        # 连接层面的失败（超时、断连），同样写一行，记下错误。
        #
        # httpx.RequestError **不是** OSError 的子类，它走的是
        # TransportError -> RequestError -> HTTPError -> Exception。
        # 少了它，跑到一半网络抖一下整个阶段就带 traceback 崩掉，
        # 那一行也不会落盘 —— 而 query() 用 raise_for_status=False，
        # 非 2xx 根本不抛，所以传输层异常是这里唯一能逃出来的东西。
        status, body, latency_ms = 0, None, 0
        error = f"{type(exc).__name__}: {exc}"

    row = {
        "sample_id": sample.sample_id,
        "dataset": dataset,
        "question": sample.question,
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
    dataset: str,
    run_id: str,
    space_id: str,
    data_dir: Path | None,
    score_threshold: float | None,
    limit: int | None,
    concurrency: int = 1,
) -> dict[str, Any]:
    """跑完一个数据集的全部 query，返回该数据集的统计。"""
    samples = [
        CanonicalSample.model_validate(row)
        for row in read_jsonl(subset_dir(run_id, dataset, data_dir) / "samples.jsonl")
    ]
    if limit is not None:
        samples = samples[:limit]

    out_path = responses_dir(run_id, data_dir) / f"{dataset}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _completed_sample_ids(out_path)
    todo = [s for s in samples if s.sample_id not in done]

    latencies: list[int] = []
    failures = 0
    # 追加模式 + 每条 flush，中断后已完成的部分不丢。
    #
    # 并发时所有落盘都在主线程做（worker 只发请求、只返回行），于是不需要写锁，
    # 续跑语义也和串行时完全一样。行序变成完成顺序而非样本顺序 —— 评测按
    # sample_id 关联并显式拒绝重复，不依赖行序。
    with out_path.open("a", encoding="utf-8", newline="\n") as sink:

        def emit(position: int, row: dict[str, Any], latency: int | None) -> None:
            nonlocal failures
            if latency is None:
                failures += 1
            else:
                latencies.append(latency)
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            if position % 10 == 0 or position == len(todo):
                print(f"  {dataset}: {position}/{len(todo)} (failures={failures})")

        if concurrency <= 1:
            for position, sample in enumerate(todo, 1):
                row, latency = _query_one(client, dataset, sample, space_id, score_threshold)
                emit(position, row, latency)
        else:
            # 每个 worker 一个独立客户端：AkashaClient 的限流器用共享的
            # _last_request_at，多线程共用一个实例会互相踩，且 request_interval
            # 会退化成「一起睡、一起发」。各自持有则每个连接独立按间隔发送，
            # 聚合速率约为 concurrency / request_interval。
            extra = [AkashaClient(client.config) for _ in range(concurrency - 1)]
            try:
                for spare in extra:
                    spare.login()
                pool: queue.Queue[AkashaClient] = queue.Queue()
                for worker_client in (client, *extra):
                    pool.put(worker_client)

                def task(sample: CanonicalSample) -> tuple[dict[str, Any], int | None]:
                    borrowed = pool.get()
                    try:
                        return _query_one(
                            borrowed, dataset, sample, space_id, score_threshold
                        )
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

    return {
        "dataset": dataset,
        "samples": len(samples),
        "skipped_already_done": len(done & {s.sample_id for s in samples}),
        "requested": len(todo),
        "failures": failures,
        "latency_ms": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "max": max(latencies, default=0),
        },
    }


def run(
    run_id: str,
    datasets: list[str],
    config_path: Path | None,
    data_dir: Path | None,
    score_threshold: float | None,
    limit: int | None,
    allow_config_drift: bool,
) -> int:
    """执行查询。返回进程退出码。"""
    config = load_config(config_path)
    config.require_credentials()
    # 给审计表查询划定时间窗；没有它就得扫这个 workspace 的全部审计行，
    # 而且会把上一次运行的记录混进来。
    started_at = utc_now()

    ingest_manifest_path = ingest_dir(run_id, data_dir) / "manifest.json"
    if not ingest_manifest_path.is_file():
        print(
            f"ERROR missing {ingest_manifest_path}; run the ingest stage first",
            file=sys.stderr,
        )
        return 1
    ingest_manifest = load_json(ingest_manifest_path)
    spaces = ingest_manifest["spaces"]

    with AkashaClient(config) as client:
        client.login()

        # PLAN.md 7.3：入库和查询之间换了 embedding 或 answer 模型，
        # 两次运行就不可比了（换 embedding 还会让旧 chunk 永远召回不到），
        # 所以比对快照，不一致就停。
        current_configs = client.get_model_configs()
        drift = current_configs != ingest_manifest.get("model_configs")
        if drift:
            message = (
                "model configs changed since ingest. Metrics would mix two "
                "configurations; a changed embedding model also orphans existing chunks."
            )
            if not allow_config_drift:
                print(f"ERROR {message} Pass --allow-config-drift to override.", file=sys.stderr)
                return 1
            print(f"WARNING {message}", file=sys.stderr)

        results = []
        for dataset in datasets:
            if dataset not in spaces:
                print(f"skip {dataset}: not in the ingest manifest", file=sys.stderr)
                continue
            results.append(
                run_dataset(
                    client,
                    dataset,
                    run_id,
                    spaces[dataset]["id"],
                    data_dir,
                    score_threshold,
                    limit,
                    config.concurrency,
                )
            )

    manifest = {
        "stage": "query",
        "run_id": run_id,
        "started_at": started_at,
        "generated_at": utc_now(),
        "connection": config.redacted(),
        "score_threshold": score_threshold,
        "concurrency": config.concurrency,
        "request_interval_seconds": config.request_interval_seconds,
        "model_configs": current_configs,
        "model_configs_match_ingest": not drift,
        "spaces": {name: spaces[name]["id"] for name in spaces},
        "datasets": results,
        "total_requested": sum(r["requested"] for r in results),
        "total_failures": sum(r["failures"] for r in results),
    }
    atomic_write_json(responses_dir(run_id, data_dir) / "manifest.json", manifest)

    for result in results:
        print(
            f"ok   {result['dataset']:<16} samples={result['samples']:<4} "
            f"requested={result['requested']:<4} failures={result['failures']:<3} "
            f"p50={result['latency_ms']['p50']}ms p95={result['latency_ms']['p95']}ms"
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset", action="append", choices=list(DATASET_NAMES))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="不传则用服务端默认值 0.45；只在做敏感性分析扫参时才设",
    )
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条，用于打通流程")
    parser.add_argument("--allow-config-drift", action="store_true")
    args = parser.parse_args(argv)

    try:
        return run(
            args.run_id,
            args.dataset or list(DATASET_NAMES),
            args.config,
            args.data_dir,
            args.score_threshold,
            args.limit,
            args.allow_config_drift,
        )
    except (AkashaError, RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
