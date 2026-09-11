"""跑 judge 指标。judge **是指标**，不是新的一层（决策 9）。

所以它落在 ``judge_verdict``、进指标层、参与汇总 —— 与 F1/recall 同层，
只是需要一个模型。LLM 归因与人工标注是另外两件事，它们进 ``annotation``、
不参与汇总（§12.5），别合。

两条硬规则：

* **失败该条排除，不记 0**（决策 13）。记 0 会让限流伪装成质量差 ——
  一次 429 风暴看起来会像模型突然变笨。
* **另叠失败率闸门**。排除得太多时那个均值就不可信了，所以超阈值直接判整个
  judge 运行失败，而不是给一个「基于 40% 样本」的分数。

provider 配置（端点、模型、密钥）从库里的 ``model_provider(role='judge')`` 读,
由设置页写入。

    uv run python -m akasha_benchmark.judge.run --eval-label run001-query-eval
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path
from typing import Any

from .. import run_args
from ..datasets import DATASET_NAMES
from ..store import connect, identity, repo
from . import faithfulness
from .client import JudgeClient, JudgeConfigError, JudgeProvider, parse_json_object

# 失败率超过这个值就判整个运行失败。0.1 是个判断：排除一成以内还能说
# 「这批分数代表整体」，超过就不能了。
DEFAULT_MAX_FAILURE_RATE = 0.1

METRIC = "faithfulness"


def resolve_provider(
    connection: sqlite3.Connection, label: str | None, role: str = "judge"
) -> JudgeProvider:
    """从库里取一个 provider 配置，凑不齐就抛 :class:`JudgeConfigError`。"""
    record = repo.get_model_provider(connection, role, label)
    if record is None:
        hint = f" labelled {label!r}" if label else ""
        raise JudgeConfigError(
            f"no {role} model provider{hint} configured. Set one in the platform's "
            "settings view, or via the API."
        )

    api_key = (record.get("api_key") or "").strip()
    if not api_key:
        raise JudgeConfigError(
            f"{role} provider {record['label']!r} has no api key. Fill it in the "
            "settings view."
        )

    params = repo.loads(record.get("params_json"), {}) or {}
    return JudgeProvider(
        base_url=record["base_url"],
        model=record["model"],
        api_key=api_key,
        temperature=float(params.get("temperature", 0.0)),
        max_tokens=int(params.get("max_tokens", 1024)),
    )


def judge_sample(
    client: JudgeClient, question: str, answer: str, response: dict[str, Any]
) -> dict[str, Any]:
    """判一条样本。返回可直接写 ``judge_verdict`` 的字段。"""
    prompt = faithfulness.build_prompt(question, answer, response)
    if prompt is None:
        # 上下文为空（no_match / general 会清空 retrievedSources）。
        # 此时 faithfulness 无定义 —— 跳过，不记 0。
        return {
            "score": None,
            "failure_kind": None,
            "reasoning": {"skipped": "no retrieved context"},
            "raw_response": None,
            "skipped": True,
        }

    reply = client.complete(*prompt)
    if reply.failure_kind:
        return {
            "score": None,
            "failure_kind": reply.failure_kind,
            "reasoning": None,
            "raw_response": reply.raw,
            "skipped": False,
        }

    try:
        payload = parse_json_object(reply.content or "")
        score, reasoning = faithfulness.parse_verdict(payload)
    except ValueError as exc:
        # 模型没按 schema 输出。记 parse_error，**不猜分数** ——
        # 静默当成 unsupported 会把「prompt 不听话」伪装成「答案不忠实」。
        return {
            "score": None,
            "failure_kind": "parse_error",
            "reasoning": {"error": str(exc)[:300]},
            "raw_response": reply.raw,
            "skipped": False,
        }

    return {
        "score": score,
        "failure_kind": None,
        "reasoning": reasoning,
        "raw_response": reply.raw,
        # 答案里没有任何事实陈述（拒答），分数无定义但不算失败。
        "skipped": score is None,
    }


def run(
    eval_label: str,
    datasets: list[str],
    db_path: Path | None,
    provider_label: str | None,
    limit: int | None,
    max_failure_rate: float,
) -> int:
    """对一个评测层跑 faithfulness。返回退出码。

    provider 配置（端点、模型、密钥）全部从库里的 ``model_provider(role='judge')``
    读，命令行只需要指一个 label。
    """
    connection = connect(db_path)
    try:
        eval_layer = repo.eval_layer_by_label(connection, eval_label)
        if eval_layer is None:
            print(f"ERROR no eval layer labelled {eval_label!r}", file=sys.stderr)
            return 1
        eval_layer_id = int(eval_layer["id"])
        query_layer_id = int(eval_layer["query_layer_id"])

        try:
            provider = resolve_provider(connection, provider_label)
        except JudgeConfigError as exc:
            print(f"ERROR {exc}", file=sys.stderr)
            return 1

        # judge_provider 那张表继续维护：eval_layer.judge_provider_id 引用它，
        # 而它记的是「这一层用过哪个端点」，属于运行档案而不是配置。
        provider_id = repo.upsert_judge_provider(
            connection,
            label=provider_label or "default",
            base_url=provider.base_url,
            model=provider.model,
            params={"temperature": provider.temperature, "max_tokens": provider.max_tokens},
        )
        connection.execute(
            "UPDATE eval_layer SET judge_provider_id = ? WHERE id = ?",
            (provider_id, eval_layer_id),
        )
        connection.commit()

        # provider 的身份哈希只吃 base_url + model，绝不吃 api_key（§12.5）。
        provider_hash = identity.judge_hash(
            base_url=provider.base_url, model=provider.model, params=None
        )
        print(f"judge provider {provider_label} ({provider.model}) hash={provider_hash}")

        already = repo.judged_sample_ids(connection, eval_layer_id, METRIC)
        rows = [
            row
            for row in repo.sample_evals(connection, eval_layer_id)
            if row["dataset"] in datasets and row["sample_id"] not in already
        ]
        if limit is not None:
            rows = rows[:limit]
        if already:
            print(f"resuming: {len(already)} already judged, {len(rows)} to go")

        counts = {"scored": 0, "skipped": 0, "failed": 0}
        with JudgeClient(provider) as client:
            for position, row in enumerate(rows, 1):
                response = repo.response_of(connection, query_layer_id, row["sample_id"])
                body = (response or {}).get("response") or {}
                result = judge_sample(
                    client, response["question"] if response else "", row["answer"] or "", body
                )
                repo.record_judge_verdict(
                    connection,
                    eval_layer_id,
                    sample_id=row["sample_id"],
                    metric=METRIC,
                    score=result["score"],
                    failure_kind=result["failure_kind"],
                    reasoning=result["reasoning"],
                    raw_response=result["raw_response"],
                    provider_hash=provider_hash,
                    prompt_version=faithfulness.PROMPT_VERSION,
                )
                connection.commit()

                if result["failure_kind"]:
                    counts["failed"] += 1
                elif result["skipped"]:
                    counts["skipped"] += 1
                else:
                    counts["scored"] += 1
                if position % 10 == 0 or position == len(rows):
                    print(f"  {position}/{len(rows)} {counts}")

        summary = repo.judge_summary(connection, eval_layer_id, METRIC)
        # judge 分数也进 metric_summary，这样它和 F1/recall 在同一张表里 ——
        # 那是决策 9「judge 是指标」在数据层的落点。
        for dataset in datasets:
            per_dataset = [
                v
                for v in repo.judge_verdicts(connection, eval_layer_id, METRIC)
                if v["score"] is not None
            ]
            if per_dataset:
                repo.record_metric_summary(
                    connection,
                    eval_layer_id,
                    dataset,
                    "judge",
                    {METRIC: sum(v["score"] for v in per_dataset) / len(per_dataset)},
                    len(per_dataset),
                )
        connection.commit()

        mean = summary["mean"]
        print(
            f"\n{METRIC}: mean={mean if mean is None else round(mean, 4)} "
            f"scored={summary['scored']} excluded={summary['excluded']} "
            f"failure_rate={summary['failure_rate']:.3f}"
        )
        if summary["failures_by_kind"]:
            print(f"failures by kind: {summary['failures_by_kind']}")

        # 失败率闸门：排除得太多时那个均值就不代表整体了。
        if summary["failure_rate"] > max_failure_rate:
            print(
                f"\nERROR judge failure rate {summary['failure_rate']:.3f} exceeds "
                f"{max_failure_rate:.3f}. The mean is computed over the successful subset "
                "only, so it no longer represents the dataset. Fix the failures "
                "(see failures by kind) and re-run; verdicts already recorded are kept.",
                file=sys.stderr,
            )
            return 1
        return 0
    finally:
        connection.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-label", default=None)
    parser.add_argument("--dataset", action="append", choices=list(DATASET_NAMES))
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument(
        "--provider",
        default=None,
        dest="provider_label",
        help="model_provider(role='judge') 里的标签。不传则取最近更新的那个",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-failure-rate", type=float, default=DEFAULT_MAX_FAILURE_RATE)
    run_args.add_argument(parser)

    try:
        args = run_args.apply(parser.parse_args(argv), stage="judge")
        run_args.require(args.eval_label, "eval_label", "judge")
        return run(
            args.eval_label,
            args.dataset or list(DATASET_NAMES),
            args.db,
            args.provider_label,
            args.limit,
            args.max_failure_rate,
        )
    except (RuntimeError, ValueError, sqlite3.Error) as exc:
        print(f"ERROR {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
