"""FastAPI 应用。

所有操作都从前端发起、由这里执行；本项目不提供命令行入口。
起服务：``uv run uvicorn akasha_platform.main:app``
"""

from __future__ import annotations

import secrets

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from akasha_benchmark.store import init_db

from .api import router
from .settings import Settings, load_settings
from .tasks import TaskRunner


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or load_settings()
    resolved.validate_binding()

    app = FastAPI(title="Akasha-Benchmark 评测平台", version="0.2.0")
    app.state.settings = resolved

    # 启动时按 schema.sql 建表。
    init_db(resolved.db_path)
    app.state.runner = TaskRunner(resolved)
    # 上次进程留下的「运行中」任务标成暂停，让用户显式继续。
    recovered = app.state.runner.recover()
    app.state.startup = {
        "recovered_tasks": recovered,
    }

    # 开发时前端在 Vite dev server 上，需要 CORS。
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(resolved.dev_origins),
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def require_token(request: Request, call_next):  # type: ignore[no-untyped-def]
        """设了令牌就逐请求校验。``compare_digest`` 避免时序泄露。"""
        token = request.app.state.settings.auth_token
        # CORS 预检不校验业务令牌。
        if token and request.method != "OPTIONS":
            provided = request.headers.get("X-Auth-Token", "")
            if not secrets.compare_digest(provided, token):
                return JSONResponse({"detail": "invalid or missing X-Auth-Token"}, 401)
        return await call_next(request)

    app.include_router(router)
    return app


app = create_app()
