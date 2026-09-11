"""平台设置。

**安全红线**（§12.10）：这个服务持有 Akasha 管理员凭据、只读数据库连接、
以及启动长任务的能力。默认只绑 ``127.0.0.1``；绑非回环地址时若没设访问令牌,
:func:`Settings.validate_binding` 会**拒绝启动** —— 不是「以后再说」的项。
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path

from akasha_benchmark.store.db import DEFAULT_DB_PATH

REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_DIST = REPO_ROOT / "web" / "dist"

ENV_PREFIX = "AKASHA_PLATFORM_"
LOOPBACK = {"127.0.0.1", "::1", "localhost"}


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8848
    db_path: Path = DEFAULT_DB_PATH
    # 前端构建产物。存在就挂 StaticFiles，单进程单端口；不存在就只提供 API,
    # 开发时用 Vite dev server 代理过来。
    web_dist: Path = WEB_DIST
    # 非空则要求所有请求带 X-Auth-Token。绑非回环地址时必须非空。
    auth_token: str = ""
    # 开发时 Vite dev server 的地址，用于 CORS。
    dev_origins: tuple[str, ...] = ("http://127.0.0.1:5173", "http://localhost:5173")
    # 任务日志目录。
    log_dir: Path = field(default_factory=lambda: REPO_ROOT / "logs" / "platform")

    def is_loopback(self) -> bool:
        return self.host in LOOPBACK

    def validate_binding(self) -> None:
        """绑非回环地址且没有令牌时拒绝启动。

        这个服务能启动 15 小时的任务、能读只读数据库、并持有 Akasha 管理员
        凭据。把它暴露到 ``0.0.0.0`` 而不设认证，等于把这三样一起交出去。
        """
        if not self.is_loopback() and not self.auth_token:
            raise RuntimeError(
                f"refusing to bind {self.host}: this service holds Akasha admin credentials, "
                "a read-only database connection, and the ability to start long-running "
                f"tasks. Set {ENV_PREFIX}AUTH_TOKEN before binding a non-loopback address, "
                "or bind 127.0.0.1."
            )

    def redacted(self) -> dict[str, object]:
        """白名单式，只输出连接形态。"""
        return {
            "host": self.host,
            "port": self.port,
            "db_path": str(self.db_path),
            "web_dist_present": self.web_dist.is_dir(),
            "auth_required": bool(self.auth_token),
            "loopback_only": self.is_loopback(),
        }


def load_settings() -> Settings:
    """从环境变量读设置。``AKASHA_PLATFORM_*`` 前缀。

    **这是唯一一组仍走环境变量的配置**，因为它是 bootstrap：库的位置本身就在
    里面（``DB``），所以它不可能从库里读。它也从不与库里的值重复，
    所以不属于「配置有两个来源」那个问题。

    host / port / db 另有命令行参数（见 :func:`..main.main`）。
    """

    def env(name: str, default: str = "") -> str:
        return os.environ.get(f"{ENV_PREFIX}{name}", default)

    host = env("HOST", "127.0.0.1")
    token = env("AUTH_TOKEN")
    # 绑非回环但没给令牌时，不悄悄生成一个 —— 那样用户不知道令牌是什么,
    # 服务却"看起来"启动成功了。让 validate_binding 报错。
    return Settings(
        host=host,
        port=int(env("PORT", "8848")),
        db_path=Path(env("DB", str(DEFAULT_DB_PATH))),
        web_dist=Path(env("WEB_DIST", str(WEB_DIST))),
        auth_token=token,
    )


def generate_token() -> str:
    """给用户生成一个令牌用的辅助函数（``--print-token``）。"""
    return secrets.token_urlsafe(32)
