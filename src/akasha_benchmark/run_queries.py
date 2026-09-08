"""轮次 4：逐条串行跑 query，把完整响应落盘。

**这一轮不算任何指标。** 它只负责产出证据，解释证据是轮次 5 的事。

存的是**完整响应体**，不是当下用得到的那几个字段。重跑一次要烧 LLM 调用，
所以以后想到要看某个新字段时，不该被迫重跑。

失败也照样写一行 —— 失败率本身就是一项结果，静默跳过失败会把后面
所有的平均值都算高。

    uv run python -m akasha_benchmark.run_queries --run-id run001
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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


def run_dataset(
    client: AkashaClient,
    dataset: str,
    run_id: str,
    space_id: str,
    data_dir: Path | None,
    score_threshold: float | None,
    limit: int | None,
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
    with out_path.open("a", encoding="utf-8", newline="\n") as sink:
        for position, sample in enumerate(todo, 1):
            requested_at = utc_now()
            try:
                response = client.query(
                    sample.question,
                    [space_id],
                    score_threshold=score_threshold,
                )
                status, body, latency_ms = response.status, response.body, response.latency_ms
                error = None
            except (AkashaError, OSError) as exc:
                # 连接层面的失败（超时、断连），同样写一行，记下错误。
                status, body, latency_ms = 0, None, 0
                error = f"{type(exc).__name__}: {exc}"

            # 只有成功的请求计入延迟统计，否则超时会把 p95 带偏。
            if status and 200 <= status < 300:
                latencies.append(latency_ms)
            else:
                failures += 1

            sink.write(
                json.dumps(
                    {
                        "sample_id": sample.sample_id,
                        "dataset": dataset,
                        "question": sample.question,
                        "requested_at": requested_at,
                        "latency_ms": latency_ms,
                        "http_status": status,
                        "error": error,
                        "response": body,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            sink.flush()

            if position % 10 == 0 or position == len(todo):
                print(f"  {dataset}: {position}/{len(todo)} (failures={failures})")

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
    """执行轮次 4。返回进程退出码。"""
    config = load_config(config_path)
    config.require_credentials()
    # 给轮次 5 的审计表查询划定时间窗；没有它就得扫这个 workspace 的全部审计行，
    # 而且会把上一次运行的记录混进来。
    started_at = utc_now()

    ingest_manifest_path = ingest_dir(run_id, data_dir) / "manifest.json"
    if not ingest_manifest_path.is_file():
        print(
            f"ERROR missing {ingest_manifest_path}; run round 3 (ingest) first",
            file=sys.stderr,
        )
        return 1
    ingest_manifest = load_json(ingest_manifest_path)
    spaces = ingest_manifest["spaces"]

    with AkashaClient(config) as client:
        client.login()

        # PLAN.md 7.3：入库和查询之间换了 embedding 或 answer 模型，
        # 两轮就不可比了（换 embedding 还会让旧 chunk 永远召回不到），
        # 所以比对快照，不一致就停。
        current_configs = client.get_model_configs()
        drift = current_configs != ingest_manifest.get("model_configs")
        if drift:
            message = (
                "model configs changed since round 3. Metrics would mix two "
                "configurations; a changed embedding model also orphans existing chunks."
            )
            if not allow_config_drift:
                print(f"ERROR {message} Pass --allow-config-drift to override.", file=sys.stderr)
                return 1
            print(f"WARNING {message}", file=sys.stderr)

        results = []
        for dataset in datasets:
            if dataset not in spaces:
                print(f"skip {dataset}: not in round 3 manifest", file=sys.stderr)
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
                )
            )

    manifest = {
        "round": 4,
        "run_id": run_id,
        "started_at": started_at,
        "generated_at": utc_now(),
        "connection": config.redacted(),
        "score_threshold": score_threshold,
        "concurrency": config.concurrency,
        "request_interval_seconds": config.request_interval_seconds,
        "model_configs": current_configs,
        "model_configs_match_round3": not drift,
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
