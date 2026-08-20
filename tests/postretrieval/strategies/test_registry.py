import pytest

from app.extraction.preferences import QueryPreferences
from app.postretrieval.strategies import POST_RETRIEVAL_STRATEGIES
from app.retrieval.models import FusedChunk


def test_registry_has_exactly_the_three_expected_strategies():
    assert set(POST_RETRIEVAL_STRATEGIES.keys()) == {"none", "metadata_filter", "cross_encoder_rerank"}


def test_cross_encoder_rerank_stub_raises_not_implemented():
    chunk = FusedChunk(
        chunk_id="c1", text="text", source_url="https://example.com", page_number=1,
        bm25_rank=1, bm25_score=1.0, semantic_rank=1, semantic_score=1.0,
        fused_rank=1, rrf_score=0.03, matched_methods=["bm25"],
    )
    with pytest.raises(NotImplementedError):
        POST_RETRIEVAL_STRATEGIES["cross_encoder_rerank"]([chunk], QueryPreferences())
