"""审计表 join 里不依赖数据库的部分：哈希格式与三段归因。"""

from __future__ import annotations

import hashlib

from akasha_benchmark.audit_join import _stage_attribution, query_hash


def test_query_hash_carries_the_server_prefix():
    """必须带 sha256: 前缀，裸的十六进制值一行都匹配不上。"""
    text = "who directed the film?"
    expected = "sha256:" + hashlib.sha256(text.encode()).hexdigest()
    assert query_hash(text) == expected
    assert query_hash(text).startswith("sha256:")


def test_stage_attribution_splits_the_three_losses():
    """候选 50 -> 排序后 20 -> 授权丢 5，三段损失要各自算清。"""
    metadata = {
        "candidateChunkCount": 50,
        "rankedCandidateCount": 20,
        "filteredChunkCount": 5,
        "accessPolicyFallbackUsed": True,
    }
    stages = _stage_attribution(metadata, gold_hit=False)
    assert stages["ranking_loss"] == 30
    assert stages["authorization_loss"] == 5
    assert stages["recall_ceiling_miss"] is False
    assert stages["access_policy_fallback_used"] is True


def test_zero_candidates_is_a_recall_ceiling_miss():
    """候选集为空属于召回上限问题，下游没有任何损失可言。"""
    stages = _stage_attribution({"candidateChunkCount": 0}, gold_hit=False)
    assert stages["recall_ceiling_miss"] is True
    assert stages["ranking_loss"] == 0
    assert stages["authorization_loss"] == 0


def test_ranking_loss_never_goes_negative():
    """防御性检查：排序后数量本不该超过候选数，但真出现时也不能给出负值。"""
    stages = _stage_attribution(
        {"candidateChunkCount": 5, "rankedCandidateCount": 9}, gold_hit=True
    )
    assert stages["ranking_loss"] == 0
