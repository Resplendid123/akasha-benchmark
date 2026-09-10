"""Akasha 的 HTTP 客户端，只覆盖入库与查询用到的端点。

认证：``POST /api/auth/login`` 会 set 一个 httpOnly 的 ``authToken`` cookie
（``auth.controller.ts:222``），响应体是空的。JWT 策略同时接受该 cookie
和 bearer header（``jwt.strategy.ts:28``），所以 httpx 的 cookie jar 就够了。
自建部署用 ``workspaceRepo.findFirst()`` 定位 workspace
（``domain.middleware.ts:18``），不需要伪造 Host 头。

以下请求/响应形状都是照服务端源码核对过的：

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

import random
import sys
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

# 只重试瞬时故障。502/503/504 是 dev server 重启或代理抖动，429 是限流，
# 都与请求内容无关，重发就能过。4xx 不在其列 —— 那是请求本身的问题。
RETRYABLE_STATUSES = frozenset({429, 502, 503, 504})
RETRYABLE_EXCEPTIONS = (httpx.TransportError,)
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2.0
RETRY_MAX_DELAY = 30.0


def _rewind_files(files: Any) -> None:
    """把 multipart 里的文件句柄拨回开头，供重试重新读取。

    ``httpx`` 的 files 可以是 dict 或 (name, value) 列表，value 又可以是
    裸句柄或 ``(filename, handle, content_type)`` 元组，所以这里逐层剥。
    不可 seek 的对象（比如生成器）跳过 —— 那种情况下重试本就不安全，
    交给上层的 4xx/5xx 判断去处理。
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
    """剥掉全局响应信封 ``{data, success, status}``。

    ``main.ts:160`` 给所有路由挂了 ``TransformHttpResponseInterceptor``，它把每个
    handler 的返回值包成 ``{data, success: true, status}``
    （``http-response.interceptor.ts:33-38``）。只有标了 ``@SkipTransform()`` 的
    handler 例外，而那三个（mcp / health / robots.txt）本评测都不用 —— 也就是说
    **这里用到的每一个端点都套着信封**。

    不剥的后果不是报错，而是静默读空：``users/me`` 取不到 role 会让 OWNER 闸门
    永远拒绝执行，导入取不到 ``id`` 会让每篇都记成失败，而质量诊断取不到
    ``summary`` 会让四项闸门全部读成 ``None`` —— ``all(value == 0)`` 于是假通过，
    带着一个半成品索引继续往下跑。

    login 是特例：handler 没有返回值，所以信封里没有 ``data`` 键，只有
    ``{success, status}``，剥出来是 ``None``。

    判据要收紧到信封自身的形状，不能只看有没有 ``data`` —— 某个端点的正常载荷里
    完全可以有一个叫 ``data`` 的字段，那种不能动。
    """
    if not isinstance(body, dict):
        return body
    if not isinstance(body.get("success"), bool) or not isinstance(body.get("status"), int):
        return body
    if not set(body) <= {"data", "success", "status"}:
        return body
    return body.get("data")


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
        retry: bool = True,
    ) -> Response:
        """``retry=False`` 用于结果有歧义就会造成重复的写入端点，见 :meth:`import_page`。"""
        url = self.config.api(path)
        response, latency_ms = self._request_with_retry(
            method, url, json_body, files, data, retry
        )

        try:
            body: Any = response.json() if response.content else None
        except ValueError:
            body = response.text
        # 全局拦截器给所有端点套了信封，在这里统一剥掉，各阶段就能按文档里
        # 记的形状直接读字段。见 unwrap_envelope 的说明。
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

        重试只针对 :data:`RETRYABLE_STATUSES` 和连接层异常 —— 这些是 dev server
        重启（``nest start --watch``）或反向代理抖动的表现，与请求内容无关。
        4xx 一律不重试：那是请求本身的问题，重试只会放大。

        不重试的代价在长任务上很实际：编译 400 页要轮询上千次，途中任何一次
        502 都会让整个 ingest 进程退出，而服务端的编译还在 BullMQ 里继续跑，
        于是产物写不出来、进度也无人接管。

        ``retry=False`` 关掉重试，留给结果有歧义的写入端点用 —— 见 :meth:`import_page`。
        """
        max_retries = MAX_RETRIES if retry else 0
        attempt = 0
        while True:
            # multipart 的文件句柄在上一次尝试里已被读到末尾，重试前必须回到开头，
            # 否则重发的是空 body，服务端会收下一个空文件。
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
                # 重试用尽后原样抛出，不包成 AkashaError —— 调用方（run_queries）
                # 按 httpx.RequestError 捕获传输层失败并落盘，换了异常类型
                # 那条通路就断了，一次网络抖动会让整个阶段带 traceback 崩掉。
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
        """指数退避，带抖动，并设上限。抖动避免多个请求在同一刻齐步重试。"""
        base = min(RETRY_BASE_DELAY * (2**attempt), RETRY_MAX_DELAY)
        return base * (0.75 + random.random() * 0.5)

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
        # 与 import_page 同理不重试：建 Space 是写入，重试可能建出第二个。
        # slug 唯一约束大概率会拦住，但报错形态会变成难懂的冲突而不是原本的 502。
        return self.post(
            "spaces/create",
            {"name": name, "slug": slug, "description": description},
            retry=False,
        )

    def delete_space(self, space_id: str) -> Any:
        """删 Space。**只给在线冒烟测试收尾用**，评测流程本身不删任何东西。

        ``SpaceIdDto`` 只要求非空字符串（``space-id.dto.ts:3-8``，那里的
        ``@IsUUID`` 是注释掉的）；调用者需要 Manage Settings 权限，OWNER 满足。
        """
        return self.post("spaces/delete", {"spaceId": space_id})

    # --- 导入 ---

    def import_page(self, markdown_path: Path, space_id: str) -> dict[str, Any]:
        """导入一个 .md 文件，返回创建的 page（含 ``id``）。

        导入服务会取首个 Markdown heading 当 page title 并从正文移除，
        所以文件名只承担 doc_id 的职责，两者互不干扰。

        **不重试**（``retry=False``）。5xx 的结果是有歧义的：服务端可能已经建好 page，
        只是代理在响应前挂了。重试于是建出第二个 page —— 语料里多一篇没人引用的重复，
        它不在 ``page_map`` 里，续跑也发现不了，只会悄悄抬高语料规模并污染检索指标。

        不重试的代价很小：导入失败会被记进 ``failures`` 并继续跑下一篇，
        而入库阶段本身可续跑，重跑一次就会把缺的补上（缺篇能被发现，重复不能）。
        """
        with markdown_path.open("rb") as handle:
            return self.post(
                "pages/import",
                files={"file": (markdown_path.name, handle, "text/markdown")},
                data={"spaceId": space_id},
                retry=False,
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

    def retry_pages(self, page_ids: list[str]) -> dict[str, Any]:
        """按源页 id 重试编译，一次最多 100 篇。

        服务端 DTO（``admin-retry-pages.dto.ts``）只认 ``pageIds``，且要求页面
        在编译 run 里出现过 —— 失败页也算，所以这是补 ``partial`` 缺口的正道：
        它建的是 ``page_retry`` run，范围恰好是这些页，不会像 ``follow_up``
        那样把整个 space 拖去重编译。
        """
        return self.post("llm-wiki/admin/retry-pages", {"pageIds": page_ids})

    def cancel_run(self, run_id: str, reason: str | None = None) -> dict[str, Any]:
        """取消一个未终态的编译 run。``reason`` 会进审计日志，上限 400 字。"""
        body = {"reason": reason} if reason else {}
        return self.post(f"llm-wiki/admin/compilation-runs/{run_id}/cancel", body)

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

        注意 ``retrievalDiagnostics`` 不在响应体里，评测得从
        ``knowledge_query_audit.metadata`` 取。
        """
        payload: dict[str, Any] = {"query": query, "spaceIds": space_ids, "type": query_type}
        if score_threshold is not None:
            payload["scoreThreshold"] = score_threshold
        if chat_context:
            payload["chatContext"] = chat_context
        return self.request("POST", "llm-wiki/query", json_body=payload, raise_for_status=False)
