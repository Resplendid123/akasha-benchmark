"""比较原文与编译产物的词汇，返回遗漏词、新增词和查询词覆盖情况。"""

from __future__ import annotations

import re
from typing import Any

# 停用词：diff 里逐词比对时，这些词的增删不说明任何问题。
STOPWORDS = frozenset(
    """a an the of and or but in on at to for with by from as is are was were be been
    being have has had do does did will would could should may might must can this that
    these those it its his her their our your my he she they we you i not no nor so than
    then there here when where which who whom whose what how why all any both each few
    more most other some such only own same too very s t just don now""".split()
)

_WORD = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*")


def content_words(text: str) -> list[str]:
    """取实词，小写。专有名词与修饰语都在里面 —— 那正是要看的东西。"""
    return [w.lower() for w in _WORD.findall(text or "") if w.lower() not in STOPWORDS]


def diff_vocabulary(source_text: str, compiled_text: str) -> dict[str, Any]:
    """原文与编译产物的词汇差。

    ``dropped`` 是**关键输出**：原文里有、编译产物里没有的实词。查询命中它们
    的时候，词法召回就断了 —— 而这不是调参能救的，词已经不在索引文本里。
    """
    source_words = content_words(source_text)
    compiled_words = content_words(compiled_text)
    source_set, compiled_set = set(source_words), set(compiled_words)

    source_chars = len(source_text or "")
    compiled_chars = len(compiled_text or "")
    return {
        "source_chars": source_chars,
        "compiled_chars": compiled_chars,
        # > 1 是扩写。实测中位 2.19 倍，仅 2.7% 净压缩 —— 所以丢词是改写策略，
        # 不是空间不足。这个比值让读者自己看到这一点。
        "expansion_ratio": (compiled_chars / source_chars) if source_chars else None,
        "source_word_count": len(source_words),
        "compiled_word_count": len(compiled_words),
        # 原文有、编译没有：词法召回在这些词上必然断。
        "dropped": sorted(source_set - compiled_set),
        # 编译新增的：可能是同义改写，也可能是模型自己加的内容。
        "added": sorted(compiled_set - source_set),
        "kept": len(source_set & compiled_set),
        "retention": (len(source_set & compiled_set) / len(source_set)) if source_set else None,
    }


def question_terms_lost(question: str, diff: dict[str, Any]) -> list[str]:
    """问题里的实词有哪些落在 ``dropped`` 里。

    非空就意味着这条样本的词法召回**在这个问题上**已经断了。这是从
    「编译丢了一些词」到「这条样本因此漏召回」之间那一步，
    也是失败案例入口最该先显示的东西。
    """
    dropped = set(diff.get("dropped") or [])
    return sorted({w for w in content_words(question) if w in dropped})


def build(
    lineage: dict[str, Any], question: str = "", *, top_terms: int = 40
) -> dict[str, Any]:
    """把一条血缘结果整成并排 diff。

    原文侧把该 page 的全部 source chunk 拼起来，编译侧把全部 artifact 的 chunk
    拼起来 —— 逐 chunk 对齐是做不到的（编译器会跨 chunk 重组），而这个视图要
    回答的是「哪些词没进索引文本」，那是集合层面的问题。
    """
    source_text = "\n\n".join(c["text"] or "" for c in lineage.get("source_chunks") or [])
    compiled_text = "\n\n".join(c["text"] or "" for c in lineage.get("chunks") or [])
    diff = diff_vocabulary(source_text, compiled_text)
    lost = question_terms_lost(question, diff) if question else []

    return {
        "source_page_id": lineage.get("source_page_id"),
        "source": {
            "text": source_text,
            "chunk_count": len(lineage.get("source_chunks") or []),
            "note": "knowledge_source_chunks — 原文，**不参与召回**",
        },
        "compiled": {
            "text": compiled_text,
            "chunk_count": len(lineage.get("chunks") or []),
            "artifact_count": len(lineage.get("artifacts") or []),
            "note": "knowledge_chunks — 编译产物，**这才是被检索的文本**",
        },
        "diff": {
            **diff,
            # 列表可能很长，UI 只需要前若干个；总数另外给。
            "dropped": diff["dropped"][:top_terms],
            "dropped_total": len(diff["dropped"]),
            "added": diff["added"][:top_terms],
            "added_total": len(diff["added"]),
        },
        # 这条是把「编译丢词」与「这条样本漏召回」连起来的证据。
        "question_terms_lost": lost,
        "verdict": (
            "编译产物里缺少问题中的实词，词法召回在这条样本上必然断"
            if lost
            else "问题中的实词都还在编译产物里"
        ),
    }
