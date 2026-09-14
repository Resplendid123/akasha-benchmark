"""faithfulness：答案里的每条陈述能否由检索到的上下文支撑。

依赖是空集，所以对四组都成立 —— narrativeqa 无 gold 标注、检索族指标全省略，
这一项能填上那个洞。

口径：把答案拆成原子陈述，逐条判上下文是否支撑，得分 = 被支撑数 / 总条数。
拆句与判定在同一次调用里做完。

上下文取 ``snippets`` 与 ``retrievedSources`` 而不取 ``citations``：后者已被裁剪，
拿它当上下文会把「引用漏了但检索到了」误判成不忠实。
"""

from __future__ import annotations

from typing import Any

SYSTEM = """You are a strict evaluator of grounding in retrieval-augmented answers.

You will receive a QUESTION, the CONTEXT that was retrieved, and an ANSWER.

Do this:
1. Split the ANSWER into atomic factual claims. A claim is one verifiable
   assertion. Ignore hedges, restatements of the question, and pure connectives.
2. For each claim decide whether the CONTEXT supports it:
   - "supported": the context states or directly entails the claim.
   - "unsupported": the context does not contain it. This includes claims that
     are true in the world but absent from the context.
   - "contradicted": the context states the opposite.
3. Judge only against the CONTEXT. Your own knowledge does not count as support.

Return JSON only, no prose, in exactly this shape:
{"claims": [{"claim": "<text>", "verdict": "supported|unsupported|contradicted",
             "evidence": "<short quote from context, or empty>"}]}

If the ANSWER contains no factual claims (a refusal, or "I could not find this"),
return {"claims": []}."""

USER_TEMPLATE = """QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}"""

# 单条上下文的截断长度：编译产物是扩写过的，全塞进去会撑爆上下文窗口。
MAX_SNIPPET_CHARS = 1200
MAX_SNIPPETS = 20


def build_context(response: dict[str, Any]) -> str:
    """从响应体拼出判定用的上下文。

    优先用 ``snippets``（带正文），退到 ``retrievedSources``（只有标题）。
    都空时返回空串，此时这个指标无定义，调用方应跳过而不是记 0。
    """
    parts: list[str] = []
    for snippet in (response.get("snippets") or [])[:MAX_SNIPPETS]:
        title = snippet.get("title") or ""
        text = (snippet.get("text") or "")[:MAX_SNIPPET_CHARS]
        if text:
            parts.append(f"[{title}]\n{text}")
    if not parts:
        for source in (response.get("retrievedSources") or [])[:MAX_SNIPPETS]:
            title = source.get("title")
            if title:
                parts.append(f"[{title}]")
    return "\n\n".join(parts)


def score_claims(claims: list[dict[str, Any]]) -> float | None:
    """被支撑的条数占比。没有任何陈述时返回 ``None``（拒答上这个指标无定义）。"""
    if not claims:
        return None
    supported = sum(1 for c in claims if c.get("verdict") == "supported")
    return supported / len(claims)


def parse_verdict(payload: dict[str, Any]) -> tuple[float | None, dict[str, Any]]:
    """把模型输出解析成 ``(分数, 结构化理由)``。

    ``verdict`` 取值不在预期集合里就抛 ValueError，不静默当成 unsupported。
    """
    claims = payload.get("claims")
    if not isinstance(claims, list):
        raise ValueError(f"expected a list under 'claims', got {type(claims).__name__}")

    allowed = {"supported", "unsupported", "contradicted"}
    cleaned: list[dict[str, Any]] = []
    for claim in claims:
        if not isinstance(claim, dict):
            raise ValueError(f"claim entries must be objects, got {type(claim).__name__}")
        verdict = claim.get("verdict")
        if verdict not in allowed:
            raise ValueError(f"unknown verdict {verdict!r}; expected one of {sorted(allowed)}")
        cleaned.append(
            {
                "claim": str(claim.get("claim") or "")[:500],
                "verdict": verdict,
                "evidence": str(claim.get("evidence") or "")[:500],
            }
        )

    score = score_claims(cleaned)
    return score, {
        "claims": cleaned,
        "claim_count": len(cleaned),
        "supported": sum(1 for c in cleaned if c["verdict"] == "supported"),
        "unsupported": sum(1 for c in cleaned if c["verdict"] == "unsupported"),
        "contradicted": sum(1 for c in cleaned if c["verdict"] == "contradicted"),
    }


def build_prompt(question: str, answer: str, response: dict[str, Any]) -> tuple[str, str] | None:
    """拼出 ``(system, user)``。上下文为空时返回 ``None``，该条跳过。"""
    context = build_context(response)
    if not context:
        return None
    return SYSTEM, USER_TEMPLATE.format(question=question, context=context, answer=answer)
