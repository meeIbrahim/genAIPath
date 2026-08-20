from app.extraction.preferences import QueryPreferences
from app.postretrieval.strategies.none import apply
from app.retrieval.models import FusedChunk


def _chunk(chunk_id: str) -> FusedChunk:
    return FusedChunk(
        chunk_id=chunk_id, text="text", source_url="https://example.com", page_number=1,
        bm25_rank=1, bm25_score=1.0, semantic_rank=1, semantic_score=1.0,
        fused_rank=1, rrf_score=0.03, matched_methods=["bm25", "semantic"],
    )


def test_none_passes_everything_through_unfiltered():
    chunks = [_chunk("c1"), _chunk("c2")]
    kept, excluded_count = apply(chunks, QueryPreferences())
    assert kept == chunks
    assert excluded_count == 0
