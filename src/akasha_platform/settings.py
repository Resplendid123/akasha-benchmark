"""平台启动设置。绑非回环地址时必须配置访问令牌。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from akasha_benchmark.store import DEFAULT_DB_PATH

ENV_PREFIX = "AKASHA_PLATFORM_"
LOOPBACK = {"127.0.0.1", "::1", "localhost"}


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8848
    db_path: Path = DEFAULT_DB_PATH
    # 非空则要求所有请求带 X-Auth-Token。绑非回环地址时必须非空。
    auth_token: str = ""
    # 开发时 Vite dev server 的地址，用于 CORS。
    dev_origins: tuple[str, ...] = ("http://127.0.0.1:5173", "http://localhost:5173")

    def is_loopback(self) -> bool:
        return self.host in LOOPBACK

    def validate_binding(self) -> None:
        """绑非回环地址且没有令牌时拒绝启动。

        这个服务能启动长任务、持有 Akasha 管理员凭据、并读一个只读数据库。
        把它暴露到 0.0.0.0 而不设认证等于把这三样一起交出去。
        """
        if not self.is_loopback() and not self.auth_token:
            raise RuntimeError(
                f"拒绝绑定 {self.host}：该服务持有 Akasha 管理员凭据并能启动长任务。"
                f"请设置 {ENV_PREFIX}AUTH_TOKEN，或绑定 127.0.0.1。"
            )

    def redacted(self) -> dict[str, object]:
        return {
            "host": self.host,
            "port": self.port,
            "db_path": str(self.db_path),
            "auth_required": bool(self.auth_token),
            "loopback_only": self.is_loopback(),
        }


def load_settings() -> Settings:
    """读取 AKASHA_PLATFORM_* 环境变量。"""

    def env(name: str, default: str = "") -> str:
        return os.environ.get(f"{ENV_PREFIX}{name}", default)

    return Settings(
        host=env("HOST", "127.0.0.1"),
        port=int(env("PORT", "8848")),
        db_path=Path(env("DB", str(DEFAULT_DB_PATH))),
        # 绑非回环但没给令牌时不悄悄生成一个 —— 那样用户不知道令牌是什么,
        # 服务却「看起来」启动成功了。让 validate_binding 报错。
        auth_token=env("AUTH_TOKEN"),
    )
