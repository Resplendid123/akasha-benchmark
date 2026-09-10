"""入库：把子集语料灌进 Akasha、完成编译、校验入库完整性。

产出 ``doc_id -> page_id`` 映射，后续每个阶段都要靠它把响应里的
``sourcePageId`` 反查回语料文档。

流程：
    1. 登录，并要求用户是 OWNER（非 OWNER 会在第三道授权闸门静默丢弃 chunk，
       症状看起来像召回质量差，而不是一个错误）
    2. 每个数据集建独立 Space，避免跨数据集实体合并污染结果
    3. 把模型配置快照写进 manifest —— embedding 换模型后，旧 chunk 的
       ``embedding_profile`` 就对不上了，那些 chunk 永远召回不到
    4. 串行导入，边导边追加写 page_map.jsonl，中断后可续跑而不是重来
    5. compile-spaces 绕过 1 小时静默期，然后轮询到全部终态
    6. 质量闸门：任何 missing / stale 计数非 0 就终止

    uv run python -m akasha_benchmark.ingest --run-id run001 --dataset hotpotqa
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from .akasha_client import ACTIVE_RUN_STATUSES, TERMINAL_RUN_STATUSES, AkashaClient, AkashaError
from .config import DEFAULT_CONFIG_PATH, load_config
from .datasets import DATASET_NAMES, get_adapter, subset_dir
from .datasets.resolver import DEFAULT_DATA_DIR
from .io_utils import atomic_write_json, load_json, read_jsonl, sha256_text, utc_now

# 需要固定并记录快照的四项模型配置。
MODEL_FEATURES = ("compiler", "embedding", "answer", "image")


def ingest_dir(run_id: str, data_dir: Path | None = None) -> Path:
    return (data_dir or DEFAULT_DATA_DIR) / "ingest" / run_id


def _slug(prefix: str, dataset: str, run_id: str) -> str:
    """Space 的 slug 必须是纯字母数字、长度 2-100，所以其他字符全部剔掉。"""
    raw = f"{prefix}{dataset}{run_id}"
    slug = "".join(ch for ch in raw if ch.isalnum())
    if len(slug) < 2:
        raise ValueError(f"cannot build an alphanumeric slug from {raw!r}")
    return slug[:100]


def _load_page_map(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    """读已有的 page_map，用于断点续跑。文件不存在视为从零开始。

    键必须是 ``(dataset, doc_id)``。四组的 doc_id 是各自数据集里的裸 ID，
    跨组会撞：在锁定的 run001 子集上，hotpotqa×2wiki 撞 15 个、
    hotpotqa×musique 17 个、2wiki×musique 28 个。只按 doc_id 建字典的话，
    后导入的那组会覆盖前一组的行，于是续跑时前一组的这些 doc 被误判成
    「已导入」而跳过，manifest 的 page_map_rows 也会少算。
    """
    if not path.is_file():
        return {}
    return {
        (row["dataset"], row["doc_id"]): row for row in read_jsonl(path) if row.get("dataset")
    }


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


def ensure_space(client: AkashaClient, dataset: str, run_id: str, prefix: str) -> dict[str, Any]:
    """取或建该数据集本次运行专用的 Space，``reused`` 标明是复用还是新建。"""
    slug = _slug(prefix, dataset, run_id)
    existing = _find_space(client, slug)
    if existing:
        return {**existing, "reused": True}
    created = client.create_space(
        name=f"bench {dataset} {run_id}"[:100],
        slug=slug,
        description=f"Akasha-Benchmark {run_id} / {dataset}. Generated, safe to delete.",
    )
    return {**created, "reused": False}


def import_corpus(
    client: AkashaClient,
    dataset: str,
    run_id: str,
    space_id: str,
    data_dir: Path | None,
    page_map_path: Path,
) -> dict[str, Any]:
    """串行导入一个数据集的子集语料，并发 1。已导入的 doc_id 跳过。"""
    src = subset_dir(run_id, dataset, data_dir)
    manifest = load_json(src / "manifest.json")
    expected_hashes: dict[str, str] = manifest["corpus_md_sha256"]

    done = _load_page_map(page_map_path)
    already = {doc_id for (ds, doc_id) in done if ds == dataset}

    corpus_dir = src / "corpus"
    todo = sorted(set(expected_hashes) - already)
    failures: list[dict[str, Any]] = []
    imported = 0

    # 追加模式打开，每条 flush：中途崩了也不会丢已导入的记录。
    page_map_path.parent.mkdir(parents=True, exist_ok=True)
    with page_map_path.open("a", encoding="utf-8", newline="\n") as sink:
        for position, doc_id in enumerate(todo, 1):
            md_path = corpus_dir / f"{doc_id}.md"
            markdown = md_path.read_text(encoding="utf-8")
            actual = sha256_text(markdown)
            # md 与抽样时记录的 sha256 不符，说明子集产物被改过，
            # 此时导进去的内容和 manifest 记的对不上，指标无从追溯。
            if actual != expected_hashes[doc_id]:
                raise RuntimeError(
                    f"{dataset}/{doc_id}.md changed since the subset was built "
                    f"(manifest {expected_hashes[doc_id][:12]} != disk {actual[:12]}). "
                    "Re-run subset sampling or restore the file."
                )

            try:
                page = client.import_page(md_path, space_id)
            except AkashaError as exc:
                failures.append({"doc_id": doc_id, "status": exc.status, "error": exc.body[:300]})
                print(f"  FAIL {dataset}/{doc_id}: HTTP {exc.status}", file=sys.stderr)
                continue

            page_id = (page or {}).get("id")
            if not page_id:
                failures.append({"doc_id": doc_id, "status": 200, "error": f"no page id: {page!r}"})
                continue

            sink.write(
                json.dumps(
                    {
                        "dataset": dataset,
                        "doc_id": doc_id,
                        "page_id": page_id,
                        "space_id": space_id,
                        "title": page.get("title"),
                        "md_sha256": actual,
                        "imported_at": utc_now(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            sink.flush()
            imported += 1
            if position % 25 == 0 or position == len(todo):
                print(f"  {dataset}: imported {position}/{len(todo)}")

    return {
        "dataset": dataset,
        "expected": len(expected_hashes),
        "skipped_already_present": len(already),
        "imported": imported,
        "failures": failures,
    }


def wait_for_runs(client: AkashaClient, space_ids: list[str], config: Any) -> dict[str, Any]:
    """轮询到全部编译 Run 进入终态。超时也返回，由调用方决定怎么处理。

    区分 succeeded / partial / failed，**partial 也要记录** —— 它意味着
    一部分页编译成功一部分没有，指标会因此偏低但不会报错。
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


def check_quality(client: AkashaClient, space_ids: list[str]) -> tuple[bool, dict[str, Any]]:
    """PLAN.md 6.4 的入库完整性闸门。

    注意字段名是 camelCase（``knowledge-quality.service.ts``），
    PLAN.md 里写的 snake_case 取不到值 —— 那样每一项都是 None，
    闸门会假通过，然后拿一个半成品库跑出一堆没意义的指标。
    """
    report = client.quality_diagnostics(space_ids)
    summary = report.get("summary") or {}
    gates = {
        "missingChunkPageCount": summary.get("missingChunkPageCount"),
        "missingEmbeddingPageCount": summary.get("missingEmbeddingPageCount"),
        "missingSourcePageCount": summary.get("missingSourcePageCount"),
        "stalePageCount": summary.get("stalePageCount"),
    }
    passed = all(value == 0 for value in gates.values())
    return passed, {"gates": gates, "report": report}


def run(
    run_id: str,
    datasets: list[str],
    config_path: Path | None,
    data_dir: Path | None,
    skip_compile: bool = False,
) -> int:
    """执行入库。返回进程退出码：0 表示可以开始跑查询。"""
    config = load_config(config_path)
    config.require_credentials()
    out_dir = ingest_dir(run_id, data_dir)
    page_map_path = out_dir / "page_map.jsonl"

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

        # 导入前拉一次模型配置快照，跑查询时会拿它比对，不一致就终止。
        model_configs = client.get_model_configs()

        spaces: dict[str, dict[str, Any]] = {}
        import_results = []
        for dataset in datasets:
            get_adapter(dataset)  # 数据集名不认识就尽早失败，别等导到一半
            space = ensure_space(client, dataset, run_id, config.space_slug_prefix)
            spaces[dataset] = space
            print(
                f"{dataset}: space {space['id']} ({'reused' if space['reused'] else 'created'})"
            )
            import_results.append(
                import_corpus(client, dataset, run_id, space["id"], data_dir, page_map_path)
            )

        space_ids = [s["id"] for s in spaces.values()]
        compile_result: dict[str, Any] | None = None
        wait_result: dict[str, Any] | None = None
        quality: dict[str, Any] | None = None
        passed = False

        if skip_compile:
            print("skipping compile (--skip-compile)")
        else:
            compile_result = client.compile_spaces(space_ids)
            print(
                f"compile requested: accepted={compile_result.get('acceptedRunCount')} "
                f"coalesced={compile_result.get('coalescedRunCount')}"
            )
            wait_result = wait_for_runs(client, space_ids, config)
            print(f"runs terminal: {wait_result['terminal']}")
            passed, quality = check_quality(client, space_ids)
            print(f"quality gates: {quality['gates']} -> {'PASS' if passed else 'FAIL'}")

        total_failures = sum(len(r["failures"]) for r in import_results)
        page_map = _load_page_map(page_map_path)
        manifest = {
            "stage": "ingest",
            "run_id": run_id,
            "generated_at": utc_now(),
            "connection": config.redacted(),
            "workspace": {"id": workspace.get("id"), "name": workspace.get("name")},
            "user": {"id": user.get("id"), "role": role},
            "spaces": {
                name: {"id": s["id"], "slug": s.get("slug"), "reused": s["reused"]}
                for name, s in spaces.items()
            },
            "model_configs": model_configs,
            "model_features_tracked": list(MODEL_FEATURES),
            "imports": import_results,
            "import_failure_count": total_failures,
            "page_map_rows": len(page_map),
            "compile": compile_result,
            "runs": wait_result,
            "quality": quality,
            "quality_passed": passed,
        }
        atomic_write_json(out_dir / "manifest.json", manifest)

    for result in import_results:
        expected, mapped = result["expected"], result["skipped_already_present"] + result["imported"]
        marker = "ok  " if expected == mapped and not result["failures"] else "WARN"
        print(
            f"{marker} {result['dataset']:<16} expected={expected} mapped={mapped} "
            f"failures={len(result['failures'])}"
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
    if not skip_compile and not passed:
        print(
            "\nquality gate FAILED — do not proceed to the query stage. "
            "A half-built index produces meaningless metrics. "
            "Use POST /api/llm-wiki/admin/retry-pages for failed pages.",
            file=sys.stderr,
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset", action="append", choices=list(DATASET_NAMES))
    parser.add_argument(
        "--config", type=Path, default=None, help=f"默认读 {DEFAULT_CONFIG_PATH.name}"
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="只导入不编译。索引处于不完整状态，仅供分阶段执行时用",
    )
    args = parser.parse_args(argv)

    try:
        return run(
            args.run_id,
            args.dataset or list(DATASET_NAMES),
            args.config,
            args.data_dir,
            args.skip_compile,
        )
    except (AkashaError, RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
