"""Akasha 的 HTTP 客户端，只覆盖轮次 3、4 用到的端点。

认证：``POST /api/auth/login`` 会 set 一个 httpOnly 的 ``authToken`` cookie
（``auth.controller.ts:222``），响应体是空的。JWT 策略同时接受该 cookie
和 bearer header（``jwt.strategy.ts:28``），所以 httpx 的 cookie jar 就够了。
自建部署用 ``workspaceRepo.findFirst()`` 定位 workspace
（``domain.middleware.ts:18``），不需要伪造 Host 头。

以下请求/响应形状都是照服务端源码核对过的，不是猜的：

* ``POST /api/pages/import`` —— multipart，字段 ``file`` + ``spaceId``，
  返回创建的 page 对象（含 ``id``）（``import.controller.ts:93-121``）
* ``POST /api/llm-wiki/query`` —— ``{query, spaceIds[], type?, scoreThreshold?,
  chatContext?}``（``query-knowledge.dto.ts``）；响应里**没有**
  ``retrievalDiagnostics`` 和 ``retrievalScope``，controller 解构时排除了它们
  （``llm-wiki.controller.ts:174``）
* ``POST /api/llm-wiki/admin/diagnostics/quality`` ——
  ``{summary, spaces[], topIssues[]}``，计数字段是 camelCase
  （``knowledge-quality.service.ts:20-46``）
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .config import AkashaConfig

# knowledge_space_compile_run 的状态取值（admin-diagnostics.dto.ts）。
# 轮询编译时靠这两个集合判断「还在跑」还是「已终态」。
TERMINAL_RUN_STATUSES = frozenset(
    {"succeeded", "partial", "failed", "superseded", "cancelled"}
)
ACTIVE_RUN_STATUSES = frozenset(
    {"queued", "compiling", "aggregate_pending", "aggregating"}
)


class AkashaError(RuntimeError):
    """非 2xx 响应。带足够上下文，不用重跑一次就能定位问题。"""

    def __init__(self, method: str, url: str, status: int, body: str) -> None:
        super().__init__(f"{method} {url} -> HTTP {status}: {body[:500]}")
        self.method = method
        self.url = url
        self.status = status
        self.body = body


@dataclass
class Response:
    status: int
    body: Any
    latency_ms: int


class AkashaClient:
    def __init__(self, config: AkashaConfig) -> None:
        self.config = config
        self._client = httpx.Client(
            timeout=httpx.Timeout(config.timeout_seconds),
            follow_redirects=False,
        )
        self._last_request_at = 0.0

    def __enter__(self) -> AkashaClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    # --- 底层管道 ---

    def _throttle(self) -> None:
        """请求之间留固定间隔，避免瞬间打满 LLM 配额。"""
        gap = self.config.request_interval_seconds
        if gap <= 0:
            return
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < gap:
            time.sleep(gap - elapsed)

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Any | None = None,
        files: Any | None = None,
        data: Any | None = None,
        raise_for_status: bool = True,
    ) -> Response:
        url = self.config.api(path)
        self._throttle()
        started = time.perf_counter()
        try:
            response = self._client.request(method, url, json=json_body, files=files, data=data)
        finally:
            self._last_request_at = time.monotonic()
        latency_ms = int((time.perf_counter() - started) * 1000)

        try:
            body: Any = response.json() if response.content else None
        except ValueError:
            body = response.text

        if raise_for_status and not response.is_success:
            raise AkashaError(method, url, response.status_code, response.text)
        return Response(status=response.status_code, body=body, latency_ms=latency_ms)

    def post(self, path: str, json_body: Any | None = None, **kwargs: Any) -> Any:
        return self.request("POST", path, json_body=json_body, **kwargs).body

    def get(self, path: str, **kwargs: Any) -> Any:
        return self.request("GET", path, **kwargs).body

    # --- 认证 ---

    def login(self) -> None:
        """登录并拿到 authToken cookie。"""
        self.config.require_credentials()
        # 成功时只 set cookie，响应体是空的。
        self.request(
            "POST",
            "auth/login",
            json_body={"email": self.config.email, "password": self.config.password},
        )
        if "authToken" not in self._client.cookies:
            raise AkashaError(
                "POST", self.config.api("auth/login"), 200,
                "login returned success but set no authToken cookie; "
                "MFA may be enabled for this account",
            )

    def current_user(self) -> dict[str, Any]:
        """``/api/users/me`` 同时带回解析出的 workspace，以及 user.role。"""
        return self.post("users/me")

    # --- Space 管理 ---

    def list_spaces(self, page: int = 1, limit: int = 100) -> dict[str, Any]:
        return self.post("spaces", {"page": page, "limit": limit})

    def create_space(self, name: str, slug: str, description: str = "") -> dict[str, Any]:
        # slug 必须是纯字母数字（CreateSpaceDto 的 @IsAlphanumeric），长度 2-100。
        return self.post(
            "spaces/create", {"name": name, "slug": slug, "description": description}
        )

    # --- 导入 ---

    def import_page(self, markdown_path: Path, space_id: str) -> dict[str, Any]:
        """导入一个 .md 文件，返回创建的 page（含 ``id``）。

        导入服务会取首个 Markdown heading 当 page title 并从正文移除，
        所以文件名只承担 doc_id 的职责，两者互不干扰。
        """
        with markdown_path.open("rb") as handle:
            return self.post(
                "pages/import",
                files={"file": (markdown_path.name, handle, "text/markdown")},
                data={"spaceId": space_id},
            )

    # --- 知识编译 ---

    def compile_spaces(self, space_ids: list[str]) -> dict[str, Any]:
        """立即建 Run，绕过 1 小时静默期。"""
        return self.post("llm-wiki/admin/compile-spaces", {"spaceIds": space_ids})

    def run_diagnostics_summary(self, space_ids: list[str]) -> dict[str, Any]:
        return self.post("llm-wiki/admin/diagnostics/summary", {"spaceIds": space_ids})

    def run_diagnostics(self, space_ids: list[str], **kwargs: Any) -> dict[str, Any]:
        return self.post(
            "llm-wiki/admin/diagnostics/runs", {"spaceIds": space_ids, **kwargs}
        )

    def quality_diagnostics(self, space_ids: list[str]) -> dict[str, Any]:
        return self.post("llm-wiki/admin/diagnostics/quality", {"spaceIds": space_ids})

    def retry_pages(self, space_ids: list[str], **kwargs: Any) -> dict[str, Any]:
        return self.post("llm-wiki/admin/retry-pages", {"spaceIds": space_ids, **kwargs})

    # --- 模型配置 ---

    def get_model_configs(self) -> Any:
        """拉取 compiler / embedding / answer / image 四项配置，用于快照比对。"""
        return self.get("llm-wiki/admin/model-configs")

    def put_model_config(self, feature: str, payload: dict[str, Any]) -> Any:
        return self.request(
            "PUT", f"llm-wiki/admin/model-configs/{feature}", json_body=payload
        ).body

    # --- 查询 ---

    def query(
        self,
        query: str,
        space_ids: list[str],
        *,
        query_type: str = "user",
        score_threshold: float | None = None,
        chat_context: list[str] | None = None,
    ) -> Response:
        """跑一条知识查询。返回原始 Response，好让失败也能照样落盘。

        注意 ``retrievalDiagnostics`` 不在响应体里，轮次 5 得从
        ``knowledge_query_audit.metadata`` 取。
        """
        payload: dict[str, Any] = {"query": query, "spaceIds": space_ids, "type": query_type}
        if score_threshold is not None:
            payload["scoreThreshold"] = score_threshold
        if chat_context:
            payload["chatContext"] = chat_context
        return self.request("POST", "llm-wiki/query", json_body=payload, raise_for_status=False)
