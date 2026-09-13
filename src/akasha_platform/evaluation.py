"""评测任务：计算所选指标，并在需要时执行 Judge。"""

import argparse
from pathlib import Path

from akasha_benchmark import evaluate, run_args, progress
from akasha_benchmark.judge import run as judge
from akasha_benchmark.metrics import registry
from akasha_benchmark.store import connect, repo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path)
    parser.add_argument('--query-label')
    parser.add_argument('--eval-label')
    parser.add_argument('--dataset', action='append')
    parser.add_argument('--metric', dest='metrics', action='append')
    parser.add_argument('--k', type=int, action='append')
    parser.add_argument('--provider-label')
    parser.add_argument('--export', action='store_true')
    run_args.add_argument(parser)
    args = run_args.apply(parser.parse_args(argv), stage='evaluate')
    run_args.require(args.query_label, 'query_label', 'evaluate')
    run_args.require(args.eval_label, 'eval_label', 'evaluate')
    connection = connect(args.db, read_only=True)
    try:
        query = repo.query_layer_by_label(connection, args.query_label)
        if query is None or not query['finished_at']:
            raise ValueError('请选择已结束的查询记录')
        available = repo.response_datasets(connection, query['id'])
        names = args.dataset or available
        if not names or not set(names).issubset(available):
            raise ValueError('所选数据集没有查询响应')
        if repo.eval_layer_by_label(connection, args.eval_label):
            raise ValueError('评测记录名称已存在，请使用新名称')
    finally:
        connection.close()
    metrics = args.metrics or []
    if not metrics:
        raise ValueError('请选择评测指标')
    judge_selected = any(d.kind == 'judge' and d.name in metrics for d in registry.METRIC_DEFINITIONS)
    if judge_selected:
        run_args.require(args.provider_label, 'provider_label', 'evaluate')
    shared = ['--query-label', args.query_label, '--eval-label', args.eval_label]
    if args.db:
        shared += ['--db', str(args.db)]
    for name in names:
        shared += ['--dataset', name]
    for metric in metrics:
        shared += ['--metric', metric]
    for k in args.k or [2, 5, 10]:
        if k < 1:
            raise ValueError('k 必须大于 0')
        shared += ['--k', str(k)]
    if args.export:
        shared += ['--export']
    with progress.scope(0, 1, 2 if judge_selected else 1, "指标计算"):
        result = evaluate.main(shared)
    if result or not judge_selected:
        return result
    connection = connect(args.db)
    try:
        evaluation = repo.eval_layer_by_label(connection, args.eval_label)
        repo.mark_eval_pending(connection, evaluation['id'])
        connection.commit()
    finally:
        connection.close()
    judge_args = ['--eval-label', args.eval_label, '--provider', args.provider_label]
    if args.db:
        judge_args += ['--db', str(args.db)]
    for name in names:
        judge_args += ['--dataset', name]
    with progress.scope(1, 2, 2, "Judge"):
        result = judge.main(judge_args)
    if result == 0:
        connection = connect(args.db)
        try:
            repo.finish_eval_layer(connection, evaluation['id'])
            connection.commit()
        finally:
            connection.close()
    return result


if __name__ == '__main__':
    raise SystemExit(main())
