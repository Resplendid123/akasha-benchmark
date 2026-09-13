"""平台下载任务：复用下载器，下载完成后自动校验原始文件。"""

import argparse
import runpy
from pathlib import Path

from akasha_benchmark import run_args

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path)
    run_args.add_argument(parser)
    parser.parse_args(argv)
    download = runpy.run_path(str(SCRIPTS / "download_datasets.py"))
    return download["main"]([])


if __name__ == "__main__":
    raise SystemExit(main())
