"""faithfulness：答案里的每条陈述能否由检索到的上下文支撑。

**依赖是空集** —— 不需要 gold 文档，也不需要参考答案。所以它对四组都成立，
而这正是它的价值所在：narrativeqa 现在整族检索指标省略（无 gold 标注），
faithfulness 能填上那个洞，且它恰恰最需要 —— 46% 的参考答案措辞在原文里
根本不存在，F1 的绝对值在这组上信息量最低。

口径：把答案拆成原子陈述，逐条判「上下文是否支撑」，得分 = 被支撑的条数 / 总条数。
拆句与判定在同一次调用里做完 —— 分两次调用要花两倍的钱，而这个指标本来就
只用于同配置比较。

**上下文取 ``retrievedSources`` 与 ``snippets``，不取 ``citations``**：后者已被
「被引 ∩ 有证据」裁剪过，拿它当上下文会把「引用漏了但检索到了」误判成不忠实。
"""

from __future__ import annotations

from typing import Any

PROMPT_VERSION = "faithfulness-v1"

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

# 单条上下文的截断长度。编译产物是扩写过的（中位 2.19 倍），全塞进去会撑爆
# 上下文窗口而且大部分是无关内容。
MAX_SNIPPET_CHARS = 1200
MAX_SNIPPETS = 20


def build_context(response: dict[str, Any]) -> str:
    """从响应体拼出判定用的上下文。

    优先用 ``snippets``（带正文），退到 ``retrievedSources``（只有标题）。
    两者都空时返回空串 —— 此时 faithfulness 无定义，调用方要跳过而不是记 0。
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
    """被支撑的条数占比。

    没有任何陈述时返回 ``None`` 而不是 0.0 或 1.0：一个拒答（``no_match``）
    既不忠实也不不忠实，这个指标在它上面无定义。记 0 会把「没找到资料」
    算成「胡说」，记 1 会把它算成「完美」，两者都是错的。
    """
    if not claims:
        return None
    supported = sum(1 for c in claims if c.get("verdict") == "supported")
    return supported / len(claims)


def parse_verdict(payload: dict[str, Any]) -> tuple[float | None, dict[str, Any]]:
    """把模型输出解析成 ``(分数, 结构化理由)``。

    ``verdict`` 取值不在预期集合里直接抛 ValueError —— 由调用方记成
    ``parse_error``。静默当成 unsupported 会把「prompt 不听话」伪装成「答案不忠实」。
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
    """拼出 ``(system, user)``。上下文为空时返回 ``None``，表示这条应当跳过。"""
    context = build_context(response)
    if not context:
        return None
    return SYSTEM, USER_TEMPLATE.format(question=question, context=context, answer=answer)
