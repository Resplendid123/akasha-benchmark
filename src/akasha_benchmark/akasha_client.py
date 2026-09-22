"""Akasha 的 HTTP 客户端，只覆盖入库与查询用到的端点。

认证：``POST /api/auth/login`` set 一个 httpOnly 的 ``authToken`` cookie，
响应体为空。

各端点的请求/响应形状：

* ``POST /api/pages/import`` —— multipart，字段 ``file`` + ``spaceId``，
  返回创建的 page 对象（含 ``id``）
* ``POST /api/llm-wiki/query`` —— ``{query, spaceIds[], type?,
  chatContext?}``
* ``POST /api/llm-wiki/admin/diagnostics/quality`` ——
  ``{summary, spaces[], topIssues[]}``，计数字段是 camelCase
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import random
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .config import AkashaConfig

# knowledge_space_compile_run 里表示「还在跑」的状态取值。
ACTIVE_RUN_STATUSES = frozenset(
    {"queued", "compiling", "aggregate_pending", "aggregating"}
)
TERMINAL_RUN_STATUSES = frozenset(
    {"succeeded", "partial", "failed", "superseded", "cancelled"}
)


def validate_cancel_result(run_id: str, result: dict[str, Any]) -> dict[str, str]:
    """确认 exact-run cancel 已把指定 Run 带到远端终态。"""
    disposition = str(result.get("disposition") or "")
    returned_run_id = str(result.get("runId") or "")
    status = str(result.get("status") or "")
    if (
        returned_run_id != str(run_id)
        or disposition not in {"cancelled", "already_terminal"}
        or status not in TERMINAL_RUN_STATUSES
        or (disposition == "cancelled" and status != "cancelled")
    ):
        raise ValueError(
            f"取消 Run {run_id} 返回无效结果：runId={returned_run_id!r}, "
            f"disposition={disposition!r}, status={status!r}"
        )
    return {"run_id": returned_run_id, "disposition": disposition, "status": status}

# 只重试瞬时故障：5xx 是服务端重启或代理抖动，429 是限流。4xx 不重试。
RETRYABLE_STATUSES = frozenset({429, 502, 503, 504})
RETRYABLE_EXCEPTIONS = (httpx.TransportError,)
MAX_RETRIES = 2
RETRY_BASE_DELAY = 2.0
RETRY_MAX_DELAY = 30.0
AUTH_TOKEN_COOKIE = "authToken"
AUTH_EXPIRY_SKEW_SECONDS = 30.0
_AUTH_CACHE_PATH = Path(tempfile.gettempdir()) / "akasha-benchmark" / "auth-tokens.json"
_AUTH_LOCK = threading.Lock()
_AUTH_TOKENS: dict[str, str] = {}


def _auth_cache_key(config: AkashaConfig) -> str:
    """凭据只参与摘要，不把账号或密码写入 JWT 缓存。"""
    identity = "\0".join(
        (config.base_url.rstrip("/").lower(), config.email.lower(), config.password)
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _jwt_expiry(token: str) -> float | None:
    """不校验签名，只读取 JWT 的 exp；签名仍由 Akasha 服务端校验。"""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        decoded = json.loads(base64.urlsafe_b64decode(payload).decode("utf-8"))
        expiry = decoded.get("exp")
        return float(expiry) if isinstance(expiry, (int, float)) else None
    except (IndexError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _usable_auth_token(token: str | None) -> bool:
    if not token:
        return False
    expiry = _jwt_expiry(token)
    return expiry is not None and expiry > time.time() + AUTH_EXPIRY_SKEW_SECONDS


def _read_auth_cache() -> dict[str, str]:
    try:
        body = json.loads(_AUTH_CACHE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(body, dict):
        return {}
    return {
        key: value
        for key, value in body.items()
        if isinstance(key, str) and isinstance(value, str)
    }


def _write_auth_cache(tokens: dict[str, str]) -> None:
    """原子替换本地缓存；缓存失败不能让一次成功登录变成任务失败。"""
    temporary: Path | None = None
    try:
        _AUTH_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = _AUTH_CACHE_PATH.with_name(
            f".{_AUTH_CACHE_PATH.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        temporary.write_text(json.dumps(tokens), encoding="utf-8")
        if os.name != "nt":
            temporary.chmod(0o600)
        temporary.replace(_AUTH_CACHE_PATH)
    except OSError:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _rewind_files(files: Any) -> None:
    """把 multipart 里的文件句柄拨回开头，供重试重新读取。

    files 可以是 dict 或 (name, value) 列表，value 可以是裸句柄或
    ``(filename, handle, content_type)`` 元组，所以逐层剥。不可 seek 的跳过。
    """
    values = files.values() if isinstance(files, dict) else [v for _, v in files]
    for value in values:
        handle = value[1] if isinstance(value, (tuple, list)) and len(value) > 1 else value
        seek = getattr(handle, "seek", None)
        if callable(seek):
            try:
                seek(0)
            except (OSError, ValueError):
                pass


def unwrap_envelope(body: Any) -> Any:
    """剥掉全局响应信封 ``{data, success, status}``。这里用到的端点全都套着它。

    判据收紧到信封自身的形状，而不是只看有没有 ``data`` —— 正常载荷里也可能
    有 ``data`` 的字段。login 的信封没有 ``data`` 键，剥出来是 ``None``。
    """
    if not isinstance(body, dict):
        return body
    if not isinstance(body.get("success"), bool) or not isinstance(body.get("status"), int):
        return body
    if not set(body) <= {"data", "success", "status"}:
        return body
    return body.get("data")


class AkashaError(RuntimeError):
    """非 2xx 响应，带上方法、URL、状态码与响应体。"""

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
        retry: bool = True,
    ) -> Response:
        """``retry=False`` 用于结果有歧义就会造成重复的写入端点。"""
        url = self.config.api(path)
        response, latency_ms = self._request_with_retry(
            method, url, json_body, files, data, retry
        )
        if response.status_code == 401 and path.strip("/") != "auth/login":
            rejected = self._client.cookies.get(AUTH_TOKEN_COOKIE)
            if rejected:
                self._refresh_auth(rejected)
                if files:
                    _rewind_files(files)
                response, latency_ms = self._request_with_retry(
                    method, url, json_body, files, data, retry
                )

        try:
            body: Any = response.json() if response.content else None
        except ValueError:
            body = response.text
        body = unwrap_envelope(body)

        if raise_for_status and not response.is_success:
            raise AkashaError(method, url, response.status_code, response.text)
        return Response(status=response.status_code, body=body, latency_ms=latency_ms)

    def _request_with_retry(
        self,
        method: str,
        url: str,
        json_body: Any | None,
        files: Any | None,
        data: Any | None,
        retry: bool = True,
    ) -> tuple[httpx.Response, int]:
        """发一次请求，瞬时故障按退避重试，返回 ``(response, latency_ms)``。

        只重试 :data:`RETRYABLE_STATUSES` 与连接层异常，4xx 一律不重试。
        ``retry=False`` 关掉重试，留给结果有歧义的写入端点。
        """
        max_retries = MAX_RETRIES if retry else 0
        attempt = 0
        while True:
            # 上一次尝试已把文件句柄读到末尾，重试前拨回开头，否则重发空 body。
            if files and attempt:
                _rewind_files(files)
            self._throttle()
            started = time.perf_counter()
            try:
                response = self._client.request(
                    method, url, json=json_body, files=files, data=data
                )
            except RETRYABLE_EXCEPTIONS as exc:
                self._last_request_at = time.monotonic()
                if attempt >= max_retries:
                    raise
                delay = self._retry_delay(attempt)
                print(
                    f"  retry {attempt + 1}/{max_retries} after {type(exc).__name__} "
                    f"on {method} {url} in {delay:.1f}s",
                    file=sys.stderr,
                )
                time.sleep(delay)
                attempt += 1
                continue
            finally:
                self._last_request_at = time.monotonic()
            latency_ms = int((time.perf_counter() - started) * 1000)

            if response.status_code in RETRYABLE_STATUSES and attempt < max_retries:
                delay = self._retry_delay(attempt)
                print(
                    f"  retry {attempt + 1}/{max_retries} after HTTP {response.status_code} "
                    f"on {method} {url} in {delay:.1f}s",
                    file=sys.stderr,
                )
                time.sleep(delay)
                attempt += 1
                continue
            return response, latency_ms

    @staticmethod
    def _retry_delay(attempt: int) -> float:
        """指数退避，带抖动并设上限，避免多个请求齐步重试。"""
        base = min(RETRY_BASE_DELAY * (2**attempt), RETRY_MAX_DELAY)
        return base * (0.75 + random.random() * 0.5)

    def post(self, path: str, json_body: Any | None = None, **kwargs: Any) -> Any:
        return self.request("POST", path, json_body=json_body, **kwargs).body

    def get(self, path: str, **kwargs: Any) -> Any:
        return self.request("GET", path, **kwargs).body

    # --- 认证 ---

    def login(self) -> None:
        """复用进程内或本地 JWT；同一进程的客户端只会有一个实际登录。"""
        self.config.require_credentials()
        key = _auth_cache_key(self.config)
        with _AUTH_LOCK:
            token = _AUTH_TOKENS.get(key)
            if not _usable_auth_token(token):
                token = _read_auth_cache().get(key)
            if _usable_auth_token(token):
                _AUTH_TOKENS[key] = token
                self._client.cookies.set(AUTH_TOKEN_COOKIE, token)
                return
            self._perform_login(key)

    def _perform_login(self, key: str) -> None:
        self._client.cookies.delete(AUTH_TOKEN_COOKIE)
        self.request(
            "POST",
            "auth/login",
            json_body={"email": self.config.email, "password": self.config.password},
        )
        token = self._client.cookies.get(AUTH_TOKEN_COOKIE)
        if not token:
            raise AkashaError(
                "POST", self.config.api("auth/login"), 200,
                "login returned success but set no authToken cookie; "
                "MFA may be enabled for this account",
            )
        _AUTH_TOKENS[key] = token
        cached = _read_auth_cache()
        cached[key] = token
        _write_auth_cache(cached)

    def _refresh_auth(self, rejected: str) -> None:
        """401 后刷新一次；若别的客户端已刷新，则直接接管它的新 JWT。"""
        key = _auth_cache_key(self.config)
        with _AUTH_LOCK:
            current = _AUTH_TOKENS.get(key)
            if current != rejected and _usable_auth_token(current):
                self._client.cookies.set(AUTH_TOKEN_COOKIE, current)
                return
            _AUTH_TOKENS.pop(key, None)
            self._perform_login(key)

    def current_user(self) -> dict[str, Any]:
        """``/api/users/me`` 同时带回解析出的 workspace，以及 user.role。"""
        return self.post("users/me")

    # --- Space 管理 ---

    def create_space(self, name: str, slug: str, description: str = "") -> dict[str, Any]:
        # slug 必须是纯字母数字，长度 2-100。不重试：重试可能建出第二个 Space。
        return self.post(
            "spaces/create",
            {"name": name, "slug": slug, "description": description},
            retry=False,
        )

    # --- 导入 ---

    def import_page_text(
        self, filename: str, markdown: str, space_id: str
    ) -> dict[str, Any]:
        """导入一段 Markdown 正文，返回创建的 page（含 ``id``）。

        ``filename`` 只承担 doc_id 的职责：导入服务取首个 Markdown heading 当
        page title 并从正文移除，所以 title 重复也不影响身份追踪。

        不重试：5xx 的结果有歧义，重试可能建出第二个 page。缺篇能被发现
        （``page_id`` 为空），重复不能。
        """
        return self.post(
            "pages/import",
            files={"file": (filename, markdown.encode("utf-8"), "text/markdown")},
            data={"spaceId": space_id},
            retry=False,
        )

    # --- 知识编译 ---

    def compile_spaces(self, space_ids: list[str]) -> dict[str, Any]:
        """立即建 Run，绕过 1 小时静默期。"""
        return self.post("llm-wiki/admin/compile-spaces", {"spaceIds": space_ids})

    def run_pages(
        self, run_id: str, *, page: int = 1, limit: int = 100
    ) -> dict[str, Any]:
        """一条编译 Run 的逐页结果。Akasha 单页最多返回 100 条。"""
        return self.get(
            f"llm-wiki/admin/diagnostics/runs/{run_id}/pages?page={page}&limit={limit}"
        )

    def retryable_run_page_ids(self, run_ids: list[str]) -> list[str]:
        """按 Run 顺序取每页最新状态，只返回仍需重试的源页面。"""
        latest: dict[str, dict[str, Any]] = {}
        for run_id in dict.fromkeys(run_ids):
            page = 1
            while True:
                result = self.run_pages(run_id, page=page, limit=100)
                items = result.get("items") or []
                for item in items:
                    page_id = item.get("sourcePageId")
                    if page_id:
                        latest[str(page_id)] = item
                total = int(result.get("total") or 0)
                limit = int(result.get("limit") or 100)
                if page * limit >= total or not items:
                    break
                page += 1
        return [
            page_id
            for page_id, item in latest.items()
            if item.get("status") == "failed"
            or item.get("mergeStatus") == "failed"
            or (
                item.get("status") == "skipped"
                and item.get("errorCode") == "manual_cancelled"
            )
        ]

    def retry_pages(self, page_ids: list[str]) -> dict[str, Any]:
        """提交至多 100 个页面；更多页面由阶段串行分批。"""
        unique = list(dict.fromkeys(page_ids))
        if len(unique) > 100:
            raise ValueError("retry_pages 每次最多 100 篇；调用方必须串行分批")
        return self.post("llm-wiki/admin/retry-pages", {"pageIds": unique})

    def cancel_compile_run(self, run_id: str, reason: str) -> dict[str, Any]:
        """取消一个精确的编译 Run；服务端同时清理对应 BullMQ job。"""
        return self.post(
            f"llm-wiki/admin/compilation-runs/{run_id}/cancel",
            {"reason": reason},
        )

    def quality_diagnostics(self, space_ids: list[str]) -> dict[str, Any]:
        return self.post("llm-wiki/admin/diagnostics/quality", {"spaceIds": space_ids})

    def run_diagnostics(self, space_ids: list[str], *, limit: int = 50) -> dict[str, Any]:
        """编译 Run 明细。``runDurationMs`` 与 ``progress`` 用于估算每篇的耗时。"""
        return self.post(
            "llm-wiki/admin/diagnostics/runs", {"spaceIds": space_ids, "limit": limit}
        )

    def page_log(self, space_ids: list[str], *, limit: int = 100) -> dict[str, Any]:
        """逐页编译日志，``errorCode`` / ``errorSummary`` 在这里。"""
        return self.post(
            "llm-wiki/admin/diagnostics/page-log",
            {"spaceIds": space_ids, "limit": limit},
        )

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
        chat_context: list[str] | None = None,
    ) -> Response:
        """跑一条知识查询。返回原始 Response，失败也落盘。"""
        payload: dict[str, Any] = {"query": query, "spaceIds": space_ids, "type": query_type}
        if chat_context:
            payload["chatContext"] = chat_context
        return self.request("POST", "llm-wiki/query", json_body=payload, raise_for_status=False)
