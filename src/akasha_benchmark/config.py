"""需要连 Akasha 的那几轮的配置。

一切都不硬编码。取值来自 JSON 配置文件（默认仓库根的 ``akasha.config.json``,
已 gitignore），并可被同名环境变量逐项覆盖，**环境变量优先**。

密钥只存在 :class:`AkashaConfig` 里，绝不写进 manifest ——
轮次 3、4 记录的是 :meth:`AkashaConfig.redacted` 的结果。

    AKASHA_BASE_URL=http://localhost:3000 \
    AKASHA_EMAIL=eval@example.com \
    AKASHA_PASSWORD=... \
    uv run python -m akasha_benchmark.ingest --run-id run001
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from .io_utils import load_json

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "akasha.config.json"

ENV_PREFIX = "AKASHA_"


@dataclass(frozen=True)
class AkashaConfig:
    base_url: str = "http://localhost:3000"
    email: str = ""
    password: str = ""
    # 自建部署的 Akasha 用 workspaceRepo.findFirst() 定位 workspace，
    # 所以这一项只有轮次 5 join 审计表时才必须填。
    workspace_id: str = ""
    api_prefix: str = "/api"
    timeout_seconds: float = 180.0
    # PLAN.md 6.2 / 7.1：严格串行。调大会让延迟数字失去意义，
    # 也容易在跑到一半时撞上 LLM 配额限流。
    concurrency: int = 1
    request_interval_seconds: float = 0.5
    poll_interval_seconds: float = 10.0
    poll_timeout_seconds: float = 7200.0
    # 仅轮次 5 需要，且是可选的：不填则跳过审计表归因，其余指标照常算。
    database_url: str = ""
    space_slug_prefix: str = "bench"
    # 配置文件里出现的未知键，隔离存放，不静默丢弃。
    extra: dict[str, Any] = field(default_factory=dict)

    def api(self, path: str) -> str:
        """拼出完整 API URL，两侧的斜杠都容错。"""
        return f"{self.base_url.rstrip('/')}{self.api_prefix}/{path.lstrip('/')}"

    def require_credentials(self) -> None:
        missing = [n for n in ("base_url", "email", "password") if not getattr(self, n)]
        if missing:
            raise ValueError(
                f"missing Akasha credentials: {missing}. Set {', '.join(ENV_PREFIX + m.upper() for m in missing)} "
                f"or create {DEFAULT_CONFIG_PATH.name}."
            )

    def redacted(self) -> dict[str, Any]:
        """可以安全写进 manifest 的视图：只有连接形态，没有密钥。"""
        return {
            "base_url": self.base_url,
            "api_prefix": self.api_prefix,
            "email": self.email,
            "password": "***" if self.password else "",
            "workspace_id": self.workspace_id,
            "database_url": "***" if self.database_url else "",
            "timeout_seconds": self.timeout_seconds,
            "concurrency": self.concurrency,
            "request_interval_seconds": self.request_interval_seconds,
        }


# 环境变量是字符串，这两组需要按类型转换。
_FLOATS = {
    "timeout_seconds",
    "request_interval_seconds",
    "poll_interval_seconds",
    "poll_timeout_seconds",
}
_INTS = {"concurrency"}


def load_config(path: Path | None = None) -> AkashaConfig:
    config_path = path or DEFAULT_CONFIG_PATH
    values: dict[str, Any] = {}

    if config_path.is_file():
        raw = load_json(config_path)
        if not isinstance(raw, dict):
            raise ValueError(f"{config_path}: expected a JSON object")
        known = {f for f in AkashaConfig.__dataclass_fields__ if f != "extra"}
        values = {k: v for k, v in raw.items() if k in known}
        unknown = {k: v for k, v in raw.items() if k not in known}
        if unknown:
            values["extra"] = unknown
    elif path is not None:
        # 显式指定了配置文件却不存在，属于用户输入错误，不该静默用默认值。
        raise FileNotFoundError(f"config file not found: {config_path}")

    config = AkashaConfig(**values)

    # 环境变量覆盖文件，这样 CI 可以只注入密钥而不落地配置文件。
    overrides: dict[str, Any] = {}
    for name in AkashaConfig.__dataclass_fields__:
        if name == "extra":
            continue
        raw_value = os.environ.get(f"{ENV_PREFIX}{name.upper()}")
        if raw_value is None:
            continue
        if name in _FLOATS:
            overrides[name] = float(raw_value)
        elif name in _INTS:
            overrides[name] = int(raw_value)
        else:
            overrides[name] = raw_value

    return replace(config, **overrides) if overrides else config
