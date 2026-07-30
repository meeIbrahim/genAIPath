from app.retrieval.fusion import RankedHit, reciprocal_rank_fusion


def test_chunk_in_both_lists_outranks_single_list_hit():
    bm25 = [RankedHit("a", 1, 9.0), RankedHit("b", 2, 5.0)]
    semantic = [RankedHit("a", 1, 0.9)]
    fused = reciprocal_rank_fusion(bm25, semantic, k=60, top_k=5)
    assert fused[0].chunk_id == "a"
    assert fused[0].matched_methods == ["bm25", "semantic"]
    assert fused[1].chunk_id == "b"
    assert fused[1].matched_methods == ["bm25"]


def test_single_list_hit_still_scored_and_included():
    bm25 = []
    semantic = [RankedHit("only-semantic", 1, 0.5)]
    fused = reciprocal_rank_fusion(bm25, semantic, k=60, top_k=5)
    assert len(fused) == 1
    assert fused[0].chunk_id == "only-semantic"
    assert fused[0].rrf_score == 1 / 61
    assert fused[0].bm25_rank is None


def test_rrf_score_formula_exact():
    bm25 = [RankedHit("a", 3, 1.0)]
    semantic = [RankedHit("a", 5, 1.0)]
    fused = reciprocal_rank_fusion(bm25, semantic, k=60, top_k=5)
    expected = 1 / (60 + 3) + 1 / (60 + 5)
    assert abs(fused[0].rrf_score - expected) < 1e-9


def test_top_k_trims_result_and_fused_rank_is_1_indexed():
    bm25 = [RankedHit(f"c{i}", i + 1, float(10 - i)) for i in range(10)]
    fused = reciprocal_rank_fusion(bm25, [], k=60, top_k=3)
    assert len(fused) == 3
    assert [f.fused_rank for f in fused] == [1, 2, 3]
    assert fused[0].chunk_id == "c0"
