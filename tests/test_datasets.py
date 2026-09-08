"""适配器与模型的几条保证：严格校验、身份显式声明、绝不猜。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from akasha_benchmark.datasets import (
    CORPUS_ID_RULES,
    Capability,
    CanonicalSample,
    CorpusDoc,
    get_adapter,
)
from akasha_benchmark.datasets.corpus import CorpusIndex, assign_doc_id
from akasha_benchmark.datasets.musique import hop_count, hop_prefix


def make_corpus(dataset: str, rows: list[tuple[str, str]]) -> CorpusIndex:
    """用 (title, text) 列表快速造一个 corpus，doc_id 就是行号。"""
    return CorpusIndex(
        dataset, [CorpusDoc(doc_id=str(i), title=t, text=x) for i, (t, x) in enumerate(rows)]
    )


# --- 规范化模型 ---


def test_canonical_sample_forbids_extra_fields():
    """上游多出字段时必须报错，不能静默忽略。"""
    with pytest.raises(ValidationError):
        CanonicalSample(
            dataset="hotpotqa",
            sample_id="hotpotqa:1",
            dataset_sample_id="1",
            question="q",
            answers=("a",),
            gold_doc_ids=(),
            metadata={},
            surprise_field="boom",
        )


def test_canonical_sample_is_immutable():
    """校验通过后不可变，下游改不动它。"""
    sample = CanonicalSample(
        dataset="hotpotqa",
        sample_id="hotpotqa:1",
        dataset_sample_id="1",
        question="q",
        answers=("a",),
        gold_doc_ids=(),
        metadata={},
    )
    with pytest.raises(ValidationError):
        sample.question = "changed"


def test_canonical_sample_rejects_empty_answers():
    """没有参考答案的样本没法打分，构造阶段就拒掉。"""
    with pytest.raises(ValidationError):
        CanonicalSample(
            dataset="hotpotqa",
            sample_id="hotpotqa:1",
            dataset_sample_id="1",
            question="q",
            answers=(),
            gold_doc_ids=(),
            metadata={},
        )


def test_markdown_puts_title_in_heading():
    """title 放 heading，Akasha 导入时会取它当 page title。"""
    doc = CorpusDoc(doc_id="42", title="Some Title", text="Body text.")
    assert doc.to_markdown() == "# Some Title\n\nBody text.\n"


# --- corpus 行身份 ---


def test_corpus_index_rejects_duplicate_title_text_pairs():
    """(title, text) 也重复时已无键可用，必须报错。"""
    with pytest.raises(ValueError, match="duplicate \\(title, text\\)"):
        make_corpus("musique", [("T", "same"), ("T", "same")])


def test_corpus_index_allows_duplicate_titles_with_different_text():
    """title 重复但正文不同是合法的（musique 有 647 例）。"""
    corpus = make_corpus("musique", [("T", "one"), ("T", "two")])
    assert corpus.unique_title_count == 1
    assert len(corpus) == 2
    # 单靠 title 有歧义，此时必须拒绝，不能随便挑一个。
    with pytest.raises(KeyError, match="maps to 2 corpus rows"):
        corpus.id_for_title("T")
    assert corpus.id_for_pair("T", "two") == "1"


def test_corpus_dedup_stats_are_reported_not_applied():
    """去重统计只报告，行数不变。"""
    corpus = make_corpus("musique", [("T", "one"), ("T", "two"), ("U", "three")])
    stats = corpus.dedup_stats()
    assert stats["rows"] == 3
    assert stats["unique_titles"] == 2
    assert stats["rows_in_duplicate_title_groups"] == 2


def test_assign_doc_id_follows_the_declared_rule():
    """按各数据集声明的规则赋 id，与规则不符就报错。"""
    assert assign_doc_id("hotpotqa", {"idx": 7, "title": "t", "text": "x"}, 99) == "7"
    assert assign_doc_id("musique", {"title": "t", "text": "x"}, 99) == "99"
    # hotpotqa 必须有原生 idx。
    with pytest.raises(ValueError, match="no 'idx'"):
        assign_doc_id("hotpotqa", {"title": "t", "text": "x"}, 0)
    # 本来没有 idx 的 corpus 突然有了，说明数据换版，行号身份不再可信。
    with pytest.raises(ValueError, match="unexpectedly has an 'idx'"):
        assign_doc_id("musique", {"idx": 3, "title": "t", "text": "x"}, 0)


def test_corpus_id_rules_cover_every_dataset():
    """四组都得有明确的身份规则，不能漏。"""
    assert set(CORPUS_ID_RULES) == {"hotpotqa", "2wikimultihopqa", "musique", "narrativeqa"}


# --- 注册表 ---


def test_registry_resolves_aliases_case_insensitively():
    """别名大小写不敏感；不认识的名字要报错而不是猜。"""
    assert get_adapter("2wiki").name == "2wikimultihopqa"
    assert get_adapter("HotPot").name == "hotpotqa"
    assert get_adapter("narrativeqa_dev_10_doc").name == "narrativeqa"
    with pytest.raises(KeyError, match="unknown dataset"):
        get_adapter("triviaqa")


def test_only_narrativeqa_lacks_evidence_recall():
    """只有 narrativeqa 不声明 EVIDENCE_RECALL。"""
    for name in ("hotpotqa", "2wikimultihopqa", "musique"):
        assert get_adapter(name).supports(Capability.EVIDENCE_RECALL)
    assert not get_adapter("narrativeqa").supports(Capability.EVIDENCE_RECALL)
    assert get_adapter("narrativeqa").supports(Capability.ANSWER_EM_F1)


# --- hotpotqa / 2wiki 的 gold 解析 ---


def test_supporting_facts_gold_is_deduplicated():
    """gold 必须去重成 title 集合，否则 recall 分母会算大。"""
    corpus = make_corpus("hotpotqa", [("A", "a"), ("B", "b")])
    adapter = get_adapter("hotpotqa")
    row = {
        "_id": "x1",
        "question": "q",
        "answer": "ans",
        # 同一个 title 出现两次（两条支撑句），外加另一篇文档。
        "supporting_facts": [["A", 0], ["A", 1], ["B", 0]],
        "type": "bridge",
        "level": "hard",
    }
    sample = adapter.parse_row(row, 0, corpus)
    assert sample.gold_doc_ids == ("0", "1")
    assert sample.metadata["gold_count"] == 2
    assert sample.metadata["supporting_fact_count"] == 3


def test_adapter_rejects_missing_id():
    """缺原生 ID 直接报错，不退化成行号。"""
    corpus = make_corpus("hotpotqa", [("A", "a")])
    with pytest.raises(ValueError, match="no usable '_id'"):
        get_adapter("hotpotqa").parse_row(
            {"question": "q", "answer": "a", "supporting_facts": [["A", 0]]}, 0, corpus
        )


def test_adapter_rejects_unresolvable_gold():
    """gold 在 corpus 里找不到时报错，不静默丢掉这条 gold。"""
    corpus = make_corpus("hotpotqa", [("A", "a")])
    with pytest.raises(ValueError, match="gold unresolvable"):
        get_adapter("hotpotqa").parse_row(
            {
                "_id": "x",
                "question": "q",
                "answer": "a",
                "supporting_facts": [["Nonexistent", 0]],
            },
            0,
            corpus,
        )


# --- musique ---


def test_musique_resolves_gold_by_title_and_text():
    """同 title 不同段落，只有 (title, text) 能定位到行。"""
    corpus = make_corpus("musique", [("T", "first para"), ("T", "second para")])
    row = {
        "id": "2hop__1_2",
        "question": "q",
        "answer": "ans",
        "answer_aliases": ["ANS", "ans"],
        "answerable": True,
        "paragraphs": [
            {"idx": 0, "title": "T", "paragraph_text": "second para", "is_supporting": True},
            {"idx": 1, "title": "T", "paragraph_text": "first para", "is_supporting": False},
        ],
        "question_decomposition": [{}, {}],
    }
    sample = get_adapter("musique").parse_row(row, 0, corpus)
    assert sample.gold_doc_ids == ("1",)
    # 别名并入 answers，去重后主答案排最前。
    assert sample.answers == ("ans", "ANS")
    assert sample.metadata["hop_count"] == 2
    assert sample.metadata["gold_with_ambiguous_title"] == 1


def test_musique_requires_a_supporting_paragraph():
    """一条 is_supporting 都没有属于坏数据。"""
    corpus = make_corpus("musique", [("T", "p")])
    with pytest.raises(ValueError, match="no is_supporting paragraph"):
        get_adapter("musique").parse_row(
            {
                "id": "2hop__1_2",
                "question": "q",
                "answer": "a",
                "paragraphs": [
                    {"idx": 0, "title": "T", "paragraph_text": "p", "is_supporting": False}
                ],
            },
            0,
            corpus,
        )


def test_hop_prefix_parsing():
    """分隔符是双下划线；前缀不在白名单里要报错。"""
    assert hop_prefix("3hop1__9285_5188_23307") == "3hop1"
    assert hop_count("4hop2") == 4
    with pytest.raises(ValueError, match="unrecognized hop prefix"):
        hop_prefix("9hop__1_2")


# --- narrativeqa ---


def test_narrativeqa_uses_row_index_and_declares_no_gold():
    """没有原生 ID 时用全量行号，且不产出 gold。"""
    corpus = make_corpus("narrativeqa", [("Book", "chunk")])
    row = {
        "question": "q",
        "answer": ["ref one", "ref two"],
        "document": {"id": "abc123", "kind": "movie", "summary": {"title": "Book"}},
    }
    sample = get_adapter("narrativeqa").parse_row(row, 17, corpus)
    assert sample.dataset_sample_id == "17"
    assert sample.sample_id == "narrativeqa:17"
    assert sample.gold_doc_ids == ()
    assert sample.answers == ("ref one", "ref two")
    # document.id 是文档级的，只留给轮次 2 抽样用，不是行身份。
    assert sample.metadata["document_id"] == "abc123"


def test_narrativeqa_rejects_non_list_answer():
    """answer 必须是参考答案列表，单个字符串是错的形状。"""
    corpus = make_corpus("narrativeqa", [("Book", "chunk")])
    with pytest.raises(ValueError, match="non-empty list"):
        get_adapter("narrativeqa").parse_row(
            {"question": "q", "answer": "single string", "document": {"id": "a"}}, 0, corpus
        )
