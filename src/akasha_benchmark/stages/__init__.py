"""六个流水线阶段。

``STAGES`` 是任务运行器唯一认的阶段来源：请求体里的阶段名必须在这里，
参数按 ``params`` 声明过滤，未声明的键不传给阶段代码。

链路测试不在这里，它是 :mod:`.chain` 摊平出来的四条普通任务。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..task import Stage
from . import attribute, chain, compile, download, evaluate, normalize, query


@dataclass(frozen=True)
class StageSpec:
    label: str
    run: Stage
    # 参数名 -> 类型。未声明的键丢掉。
    params: dict[str, type] = field(default_factory=dict)


STAGES: dict[str, StageSpec] = {
    "download": StageSpec(
        label="下载数据集",
        run=download.run,
        params={"datasets": list},
    ),
    "normalize": StageSpec(
        label="归一化",
        run=normalize.run,
        params={"datasets": list},
    ),
    "compile": StageSpec(
        label="编译",
        run=compile.run,
        params={
            "run_id": str,
            "datasets": list,
            "seed": int,
            "qa_limit": int,
            "sample_ids": list,
            "negatives_ratio": float,
            "full_corpus": bool,
            "import_concurrency": int,
        },
    ),
    "query": StageSpec(
        label="查询",
        run=query.run,
        params={
            "compile_id": int,
            "name": str,
            "datasets": list,
            "sample_limit": int,
            "concurrency": int,
            "retry_failed": bool,
        },
    ),
    "evaluate": StageSpec(
        label="评测",
        run=evaluate.run,
        params={
            "query_id": int,
            "name": str,
            "datasets": list,
            "ks": list,
            "metrics": list,
            "judge_provider_id": int,
            "concurrency": int,
        },
    ),
    "attribute": StageSpec(
        label="归因",
        run=attribute.run,
        params={
            "eval_id": int,
            "name": str,
            "use_model": bool,
            "provider_id": int,
        },
    ),
}


def clean_params(stage: str, args: dict[str, Any]) -> dict[str, Any]:
    """按声明过滤并转换类型。未声明的键直接丢掉。"""
    spec = STAGES.get(stage)
    if spec is None:
        raise ValueError(f"未知阶段 {stage!r}；可用：{sorted(STAGES)}")

    cleaned: dict[str, Any] = {}
    for key, kind in spec.params.items():
        if key not in args:
            continue
        value = args[key]
        if value is None or value == "":
            continue
        try:
            if kind is list:
                if not isinstance(value, list):
                    raise ValueError("expected a JSON array")
                cleaned[key] = [int(v) for v in value] if key == "ks" else [str(v) for v in value]
            elif kind is bool:
                if not isinstance(value, bool):
                    raise ValueError("expected a JSON boolean")
                cleaned[key] = value
            else:
                cleaned[key] = kind(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{stage}.{key}: 需要 {kind.__name__}，收到 {value!r}") from exc
    return cleaned


__all__ = ["STAGES", "StageSpec", "chain", "clean_params"]
