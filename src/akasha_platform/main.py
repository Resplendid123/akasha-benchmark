"""FastAPI 应用与 ``akasha-platform`` 入口。
    uv run akasha-platform
    uv run akasha-platform --port 9000
"""

from __future__ import annotations

import argparse
import os
import secrets
import sys
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .api import router
from .settings import Settings, generate_token, load_settings


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or load_settings()
    app = FastAPI(
        title="Akasha-Benchmark 评测平台",
        summary="跑评测、看数据处理过程、追样本与归因",
        version="0.1.0",
    )
    app.state.settings = resolved

    # 开发时前端在 Vite dev server 上，需要 CORS；同源部署时也安全。
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved.dev_origins),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def require_token(request: Request, call_next):  # type: ignore[no-untyped-def]
        """设了令牌就逐请求校验。用 ``compare_digest`` 避免时序泄露。"""
        token = request.app.state.settings.auth_token
        if token:
            provided = request.headers.get("X-Auth-Token", "")
            if not secrets.compare_digest(provided, token):
                return JSONResponse({"detail": "invalid or missing X-Auth-Token"}, status_code=401)
        return await call_next(request)

    app.include_router(router)
    return app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--reload", action="store_true", help="开发用，改代码自动重启")
    parser.add_argument(
        "--print-token", action="store_true", help="生成一个访问令牌并打印，然后退出"
    )
    args = parser.parse_args(argv)

    if args.print_token:
        print(generate_token())
        return 0

    settings = load_settings()
    overrides: dict[str, object] = {}
    if args.host:
        overrides["host"] = args.host
    if args.port:
        overrides["port"] = args.port
    if args.db:
        overrides["db_path"] = args.db
    if overrides:
        from dataclasses import replace

        settings = replace(settings, **overrides)

    try:
        settings.validate_binding()
    except RuntimeError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 1

    if not settings.db_path.is_file():
        print(
            f"ERROR no database at {settings.db_path}. Run "
            "`uv run python -m akasha_benchmark.store.migrate` first.",
            file=sys.stderr,
        )
        return 1

    import uvicorn

    print(f"API on http://{settings.host}:{settings.port}  db={settings.db_path}")
    print("前端 dev: npm --prefix web run dev  → http://127.0.0.1:5173")
    if not settings.auth_token:
        print("no auth token set — bound to loopback only")
    # Uvicorn 热更新子进程会重新调用工厂，通过环境继承已解析的启动参数。
    os.environ.update({
        "AKASHA_PLATFORM_HOST": settings.host,
        "AKASHA_PLATFORM_PORT": str(settings.port),
        "AKASHA_PLATFORM_DB": str(settings.db_path.resolve()),
    })
    uvicorn.run(
        "akasha_platform.main:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
