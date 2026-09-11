"""FastAPI 应用与 ``akasha-platform`` 入口。

开发时两个进程：Vite dev server + 这个后端。生产 ``npm run build`` 出静态文件,
由这里挂 ``StaticFiles``，单进程单端口（§12.7）。

**安全**：默认只绑 ``127.0.0.1``。绑非回环地址且没设访问令牌时**拒绝启动** ——
这个服务持有 Akasha 管理员凭据、只读数据库连接、以及启动长任务的能力。

    uv run akasha-platform
    uv run akasha-platform --port 9000
"""

from __future__ import annotations

import argparse
import secrets
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

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

    # 开发时前端在 Vite dev server 上，需要 CORS；生产同源，用不到。
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

    dist = resolved.web_dist
    if dist.is_dir():
        # assets 走 StaticFiles，其余路径回落到 index.html（前端是 SPA）。
        assets = dist / "assets"
        if assets.is_dir():
            app.mount("/assets", StaticFiles(directory=assets), name="assets")

        @app.get("/{path:path}")
        async def spa(path: str) -> Any:  # type: ignore[no-untyped-def]
            candidate = dist / path
            if path and candidate.is_file():
                return FileResponse(candidate)
            index = dist / "index.html"
            if not index.is_file():
                raise HTTPException(404, "web/dist/index.html missing; run `npm run build`")
            return FileResponse(index)
    else:

        @app.get("/")
        async def no_frontend() -> dict[str, Any]:  # type: ignore[no-untyped-def]
            return {
                "detail": (
                    "frontend not built. Run `npm --prefix web install && "
                    "npm --prefix web run build`, or use the Vite dev server on :5173."
                ),
                "api": "/api/health",
            }

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
    overrides: dict[str, Any] = {}
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

    print(f"serving on http://{settings.host}:{settings.port}  db={settings.db_path}")
    if not settings.auth_token:
        print("no auth token set — bound to loopback only")
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
