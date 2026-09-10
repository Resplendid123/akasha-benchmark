"""查某个 run 的知识编译进度，输出一行摘要 + 增量对比。

用法：
    python scripts/watch_compile.py --run-id run001              # 走 HTTP 诊断接口
    uv run --with 'psycopg[binary]' python scripts/watch_compile.py \
        --run-id run001 --via-db                                 # 直连 Postgres

``--via-db`` 存在的原因：登录接口会在认证通过后的会话创建阶段返回 502，
HTTP 通道因此整条不可用，而编译本身照常在 worker 上推进。直连库既能拿到
真实进度，也避免了「失败登录 + 5 次重试」反过来消耗 auth 限流配额。

space_id 从 data/ingest/<run-id>/page_map.jsonl 里读，所以不用手填 UUID。
每次执行会把 (时间, succeeded) 存进同目录的 .watch_state.json，
下次执行时打印两次之间的增量 —— 用来分辨「在跑」和「卡住」。
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from akasha_benchmark.akasha_client import ACTIVE_RUN_STATUSES, AkashaClient
from akasha_benchmark.config import load_config

LOCAL_OFFSET = timedelta(hours=8)  # 输出用本地时间，方便和墙上时钟对照


def _spaces(page_map_path: Path) -> dict[str, str]:
    """从 page_map 读出 ``{space_id: dataset}``，顺序即首次出现的顺序。"""
    seen: dict[str, str] = {}
    with page_map_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                row = json.loads(line)
                seen.setdefault(row["space_id"], row["dataset"])
    return seen


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _fmt_local(moment: datetime) -> str:
    return (moment + LOCAL_OFFSET).strftime("%H:%M")


def _pick_run(items: list[dict[str, Any]]) -> dict[str, Any] | None:
    """优先取还在跑的 run，否则取最近创建的那个。"""
    active = [it for it in items if it.get("status") in ACTIVE_RUN_STATUSES]
    pool = active or items
    if not pool:
        return None
    return max(pool, key=lambda it: it.get("createdAt") or "")


def _fetch_via_db(database_url: str, space_ids: list[str]) -> dict[str, dict[str, Any]]:
    """直连库，按 space 取最新的 run（优先活跃的），整形成诊断接口那种结构。

    返回 ``{space_id: run}``。多 space 时逐个取，而不是只挑一个全局活跃 run ——
    编译一次只跑一个 space，只看活跃的那个会让其余 space 的进度从视野里消失。
    """
    import psycopg  # 可选依赖，只在 --via-db 时需要

    runs: dict[str, dict[str, Any]] = {}
    with psycopg.connect(database_url, connect_timeout=10) as conn, conn.cursor() as cur:
        for space_id in space_ids:
            cur.execute(
                """
                select id, status, phase, mode, knowledge_generation,
                       started_at, last_yield_at, last_yield_reason, error_code
                from knowledge_space_compile_runs
                where space_id = %s
                order by
                    (status in ('queued','compiling','aggregate_pending','aggregating')) desc,
                    created_at desc
                limit 1
                """,
                (space_id,),
            )
            row = cur.fetchone()
            if row is None:
                continue

            cur.execute(
                """
                select status, count(*)
                from knowledge_space_compile_run_pages
                where run_id = %s
                group by status
                """,
                (row[0],),
            )
            counts = dict(cur.fetchall())
            runs[space_id] = {
                "runId": str(row[0]),
                "status": row[1],
                "phase": row[2],
                "mode": row[3],
                "knowledgeGeneration": row[4],
                "startedAt": row[5].isoformat().replace("+00:00", "Z") if row[5] else None,
                "lastYieldAt": row[6].isoformat().replace("+00:00", "Z") if row[6] else None,
                "lastYieldReason": row[7],
                "errorCode": row[8],
                "progress": {
                    "text": {
                        "expected": sum(counts.values()),
                        "succeeded": counts.get("succeeded", 0),
                        "failed": counts.get("failed", 0),
                        "skipped": counts.get("skipped", 0),
                    }
                },
            }
    return runs


def _report_multi(
    runs: dict[str, dict[str, Any]],
    datasets: dict[str, str],
    expected: dict[str, int],
    state_path: Path,
) -> None:
    """多 space 的汇总：逐 space 一行，再给一行总计和增量。

    ``expected`` 取自 page_map 的每数据集行数 —— run 还没初始化时它的
    expected_page_count 是 0，用 page_map 才能在编译开始前就显示正确的分母。
    """
    now = datetime.now(UTC)
    total_done = total_failed = total_expect = 0
    active_any = False

    for space_id, dataset in datasets.items():
        want = expected.get(dataset, 0)
        total_expect += want
        run = runs.get(space_id)
        if run is None:
            print(f"{dataset:<18} 尚未建 run          0/{want}")
            continue

        text = run["progress"]["text"]
        done, failed = text["succeeded"], text["failed"]
        # run 刚建时页表还没铺开，expected 会是 0，这时用 page_map 的数当分母。
        want = max(want, text["expected"])
        total_done += done
        total_failed += failed
        if run["status"] in ACTIVE_RUN_STATUSES:
            active_any = True
        pct = done / want * 100 if want else 0.0
        note = f"  failed={failed}" if failed else ""
        print(
            f"{dataset:<18} {run['status']:<11} {done}/{want} ({pct:.1f}%)"
            f"{note}  run={run['runId'][:8]}"
        )
        if run.get("errorCode"):
            print(f"  warn: {dataset} errorCode={run['errorCode']}")

    pct = total_done / total_expect * 100 if total_expect else 0.0
    print(f"{'总计':<17} {total_done}/{total_expect} ({pct:.1f}%)  failed={total_failed}")

    prev = None
    if state_path.exists():
        prev = json.loads(state_path.read_text(encoding="utf-8"))
    if prev and "total" in prev:
        delta = total_done - prev["total"]
        gap_min = (now - _parse_ts(prev["checked_at"])).total_seconds() / 60
        verdict = "STALLED" if delta == 0 and active_any else "推进中"
        rate = delta / gap_min if gap_min > 0 else 0.0
        remaining = total_expect - total_done - total_failed
        eta = ""
        if rate > 0 and remaining > 0:
            eta = f"  ETA ~{_fmt_local(now + timedelta(minutes=remaining / rate))}"
        print(
            f"距上次检查 {gap_min:.0f} 分钟：+{delta} 篇（{rate:.2f} 篇/分钟）"
            f"  [{verdict}]{eta}"
        )
    if not active_any:
        print("没有活跃 run —— 编译已全部进入终态")

    state_path.write_text(
        json.dumps(
            {"total": total_done, "checked_at": now.isoformat().replace("+00:00", "Z")}
        ),
        encoding="utf-8",
    )


def _report(run: dict[str, Any], summary: dict[str, Any], state_path: Path) -> None:
    text = run["progress"]["text"]
    done, expected = text["succeeded"], text["expected"]
    failed, skipped = text["failed"], text["skipped"]
    now = datetime.now(UTC)

    print(
        f"run {run['runId'][:8]}  status={run['status']}  phase={run['phase']}  "
        f"gen={run['knowledgeGeneration']}  mode={run['mode']}"
    )
    pct = done / expected * 100 if expected else 0.0
    print(f"text {done}/{expected} ({pct:.1f}%)  failed={failed}  skipped={skipped}")

    # 速率与 ETA 用整段墙上时间算：yield 让位的空档也是真实开销，算进去才不乐观。
    started = _parse_ts(run["startedAt"]) if run.get("startedAt") else None
    if started and done:
        elapsed_min = (now - started).total_seconds() / 60
        rate = done / elapsed_min
        remaining = expected - done - failed - skipped
        if rate > 0 and remaining > 0:
            eta = now + timedelta(minutes=remaining / rate)
            print(
                f"已跑 {elapsed_min:.0f} 分钟，{rate:.2f} 篇/分钟，"
                f"剩 {remaining} 篇，ETA ~{_fmt_local(eta)}"
            )

    prev = None
    if state_path.exists():
        prev = json.loads(state_path.read_text(encoding="utf-8"))
    if prev and prev.get("run_id") == run["runId"]:
        delta = done - prev["succeeded"]
        gap_min = (now - _parse_ts(prev["checked_at"])) .total_seconds() / 60
        verdict = "STALLED" if delta == 0 else "推进中"
        print(f"距上次检查 {gap_min:.0f} 分钟：+{delta} 篇  [{verdict}]")

    stalled = summary.get("workerEvents", {}).get("stalled", 0)
    if stalled:
        print(f"warn: 近一小时 worker stalled={stalled}")
    if run.get("errorCode"):
        print(f"warn: errorCode={run['errorCode']}")
    if run.get("lastYieldReason"):
        print(f"lastYield={run['lastYieldReason']} @ {_parse_ts(run['lastYieldAt']):%H:%M}Z")

    state_path.write_text(
        json.dumps(
            {
                "run_id": run["runId"],
                "succeeded": done,
                "checked_at": now.isoformat().replace("+00:00", "Z"),
            }
        ),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--via-db",
        action="store_true",
        help="绕过 HTTP，直连 Postgres 读进度（需要 psycopg 和 database_url）",
    )
    args = parser.parse_args(argv)

    out_dir = args.data_dir / "ingest" / args.run_id
    page_map_path = out_dir / "page_map.jsonl"
    if not page_map_path.exists():
        print(f"ERROR 缺 {page_map_path}", file=sys.stderr)
        return 2

    datasets = _spaces(page_map_path)
    space_ids = list(datasets)
    expected = Counter(
        json.loads(line)["dataset"]
        for line in page_map_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    config = load_config()
    state_path = out_dir / ".watch_state.json"

    if args.via_db:
        if not config.database_url:
            print("ERROR --via-db 需要配置 database_url", file=sys.stderr)
            return 2
        runs_by_space = _fetch_via_db(config.database_url, space_ids)
        if not runs_by_space:
            print("没有找到任何 compile run")
            return 1
        _report_multi(runs_by_space, datasets, dict(expected), state_path)
        return 0

    with AkashaClient(config) as client:
        client.login()
        runs = client.run_diagnostics(space_ids)
        summary = client.run_diagnostics_summary(space_ids)
    run = _pick_run(runs.get("items", []))
    if run is None:
        print("没有找到任何 compile run")
        return 1

    _report(run, summary, state_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

