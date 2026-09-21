"""评估与归因共用的 OpenAI 兼容 HTTP 客户端。

凭据从 model_provider 读取，网络重试复用 Akasha 客户端的常量。
"""

from __future__ import annotations

import json
import queue
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import httpx

from ..akasha_client import (
    MAX_RETRIES,
    RETRY_BASE_DELAY,
    RETRY_MAX_DELAY,
    RETRYABLE_STATUSES,
)

# judge 失败分四类，处置各不相同：
#   rate_limit   —— 退避重试或降并发
#   timeout      —— 同上，也可能是 prompt 太长
#   parse_error  —— 模型没按 schema 输出，要改 prompt
#   refusal      —— 模型拒答
FAILURE_RATE_LIMIT = "rate_limit"
FAILURE_TIMEOUT = "timeout"
FAILURE_PARSE = "parse_error"
FAILURE_REFUSAL = "refusal"

# 不做成配置项：judge 分数要在两次运行之间可比，温度必须是 0。
TEMPERATURE = 0.0
MAX_TOKENS = 1024


class JudgeConfigError(RuntimeError):
    """凭据或端点没配好。"""


@dataclass(frozen=True)
class JudgeProvider:
    """一个 judge / 归因分析端点。

    密钥只有一条来路：库里的 ``model_provider.api_key``，在配置页填。
    Akasha 的 ``/model-configs`` 只回传 ``apiKeySet`` 布尔量，所以即便用同一个
    端点同一个模型，平台也得自己配一份。
    """

    base_url: str
    model: str
    api_key: str = ""
    timeout_seconds: float = 120.0
    provider_id: int | None = None

    def resolve_key(self) -> str:
        key = (self.api_key or "").strip()
        if not key:
            raise JudgeConfigError(
                "no api key configured for this provider. Akasha's /model-configs only "
                "reports apiKeySet as a boolean and never returns the key itself, "
                "so this credential has to be filled in the settings view."
            )
        return key

    def redacted(self) -> dict[str, Any]:
        """可以安全落库或记日志的视图。

        白名单式，所以新增字段的默认行为是不输出。``api_key`` 只报是否有值。
        """
        return {
            "base_url": self.base_url,
            "model": self.model,
            "api_key_set": bool((self.api_key or "").strip()),
        }


@dataclass
class JudgeReply:
    """一次调用的结果。``failure_kind`` 非空表示失败，此时 ``content`` 无意义。"""

    content: str | None
    failure_kind: str | None
    raw: str | None
    status: int | None
    # 含重试的墙钟耗时。
    latency_ms: int | None = None


def _delay(attempt: int) -> float:
    base = min(RETRY_BASE_DELAY * (2**attempt), RETRY_MAX_DELAY)
    return base * (0.75 + random.random() * 0.5)


class JudgeClient:
    """极小的 chat completions 客户端。只做 judge 要的那一件事。"""

    def __init__(self, provider: JudgeProvider, client: httpx.Client | None = None) -> None:
        self.provider = provider
        self._client = client or httpx.Client(
            timeout=httpx.Timeout(provider.timeout_seconds), follow_redirects=False
        )

    def __enter__(self) -> JudgeClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def complete(self, system: str, user: str) -> JudgeReply:
        """发一次请求，返回文本或失败类别。不抛异常，失败也是数据。

        计时统一在这一层做，免得 ``_complete`` 的每个返回点各填一次。
        """
        started = time.monotonic()
        reply = self._complete(system, user)
        reply.latency_ms = int((time.monotonic() - started) * 1000)
        return reply

    def _complete(self, system: str, user: str) -> JudgeReply:
        url = f"{self.provider.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.provider.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            # 端点不支持时会忽略这一项，所以解析侧仍要容错。
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.provider.resolve_key()}"}

        attempt = 0
        while True:
            try:
                response = self._client.post(url, json=payload, headers=headers)
            except httpx.TimeoutException as exc:
                if attempt >= MAX_RETRIES:
                    return JudgeReply(None, FAILURE_TIMEOUT, f"{type(exc).__name__}: {exc}", None)
                time.sleep(_delay(attempt))
                attempt += 1
                continue
            except httpx.RequestError as exc:
                # 传输层失败按超时归类，处置一样。
                if attempt >= MAX_RETRIES:
                    return JudgeReply(None, FAILURE_TIMEOUT, f"{type(exc).__name__}: {exc}", None)
                time.sleep(_delay(attempt))
                attempt += 1
                continue

            if response.status_code in RETRYABLE_STATUSES and attempt < MAX_RETRIES:
                print(
                    f"  judge retry {attempt + 1}/{MAX_RETRIES} after HTTP {response.status_code}",
                    file=sys.stderr,
                )
                time.sleep(_delay(attempt))
                attempt += 1
                continue

            if response.status_code == 429:
                # 重试用尽仍在限流，单独记 rate_limit 而不混进质量分。
                return JudgeReply(None, FAILURE_RATE_LIMIT, response.text[:500], 429)
            if not response.is_success:
                return JudgeReply(None, FAILURE_PARSE, response.text[:500], response.status_code)

            try:
                body = response.json()
                choice = (body.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                content = message.get("content")
                # 拒答的两种表现：finish_reason 是 content_filter，或有 refusal 字段。
                if choice.get("finish_reason") == "content_filter" or message.get("refusal"):
                    return JudgeReply(None, FAILURE_REFUSAL, response.text[:500], 200)
                if not content:
                    return JudgeReply(None, FAILURE_PARSE, response.text[:500], 200)
            except (ValueError, KeyError, IndexError, TypeError):
                return JudgeReply(None, FAILURE_PARSE, response.text[:500], response.status_code)

            return JudgeReply(content, None, response.text[:2000], 200)


def complete_many(
    provider: JudgeProvider, prompts: list[tuple[str, str]], concurrency: int
) -> list[JudgeReply]:
    """并发跑一批 ``(system, user)``，按输入顺序返回结果。

    每个 worker 一个独立 ``JudgeClient``：httpx.Client 不宜跨线程共用。
    只有网络调用进线程池，调用方在主线程解析与落库。
    """
    if concurrency <= 1 or len(prompts) <= 1:
        with JudgeClient(provider) as client:
            return [client.complete(system, user) for system, user in prompts]

    workers = min(concurrency, len(prompts))
    clients = [JudgeClient(provider) for _ in range(workers)]
    try:
        pool: queue.Queue[JudgeClient] = queue.Queue()
        for client in clients:
            pool.put(client)

        def task(prompt: tuple[str, str]) -> JudgeReply:
            borrowed = pool.get()
            try:
                return borrowed.complete(*prompt)
            finally:
                pool.put(borrowed)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(task, prompts))
    finally:
        for client in clients:
            client.close()


def parse_json_object(content: str) -> dict[str, Any]:
    """从模型输出里取 JSON 对象。

    先直解，失败再退到「取第一个 ``{`` 到最后一个 ``}``」（模型常加围栏或
    前后缀）。取不到就抛 ValueError，由调用方记成 ``parse_error``。
    """
    text = content.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"no JSON object in judge output: {content[:200]!r}") from None
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError(f"judge returned {type(parsed).__name__}, expected an object")
    return parsed
