"""归一化任务：检查原始文件、归一化、验收数据库产物。"""

import argparse
import runpy
from pathlib import Path

from akasha_benchmark import normalize, run_args, progress
from akasha_benchmark.datasets import DATASET_NAMES, resolve
from akasha_benchmark.io_utils import load_json

from .download import SCRIPTS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path)
    parser.add_argument("--dataset", action="append", choices=list(DATASET_NAMES))
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--export", action="store_true")
    run_args.add_argument(parser)
    args = run_args.apply(parser.parse_args(argv), stage="normalize")
    names = args.dataset or list(DATASET_NAMES)
    shared: list[str] = []
    if args.db:
        shared += ["--db", str(args.db)]
    if args.dataset_dir:
        shared += ["--dataset-dir", str(args.dataset_dir)]
    with progress.scope(0, 1, 3, "原始文件校验"):
        for index, name in enumerate(names):
            progress.report(index, len(names), f"检查 {name}")
            shared += ["--dataset", name]
            resolved = resolve(name, args.dataset_dir)
            for path in (resolved.qa_path, resolved.corpus_path):
                rows = load_json(path)
                if not isinstance(rows, list) or not rows:
                    raise ValueError(f"{path.name}: expected a non-empty JSON array")
            print(f"{name}: 原始文件校验通过", flush=True)
            progress.report(index + 1, len(names), f"{name} 校验通过")

    with progress.scope(1, 2, 3, "归一化"):
        code = normalize.main(shared + (["--export"] if args.export else []))
    if code:
        return code
    print("校验归一化产物…", flush=True)
    validation = runpy.run_path(str(SCRIPTS / "validate_datasets.py"))
    with progress.scope(2, 3, 3, "归一化验收"):
        return validation["main"](shared)


if __name__ == "__main__":
    raise SystemExit(main())
