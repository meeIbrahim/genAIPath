from app.extraction.preferences import QueryPreferences
from app.retrieval.filtering import filter_chunks
from app.retrieval.models import FusedChunk


def _chunk(chunk_id: str, city: str | None = None, price: float | None = None) -> FusedChunk:
    return FusedChunk(
        chunk_id=chunk_id,
        text="text",
        source_url="https://example.com",
        page_number=1,
        city=city,
        price=price,
        bm25_rank=1,
        bm25_score=1.0,
        semantic_rank=1,
        semantic_score=1.0,
        fused_rank=1,
        rrf_score=0.03,
        matched_methods=["bm25", "semantic"],
    )


def test_filter_chunks_excludes_conflicting_city():
    chunks = [_chunk("c1", city="paris"), _chunk("c2", city="lahore")]
    kept, excluded_count = filter_chunks(chunks, QueryPreferences(city="lahore"))
    assert [c.chunk_id for c in kept] == ["c2"]
    assert excluded_count == 1


def test_filter_chunks_excludes_over_budget_price():
    chunks = [_chunk("c1", price=1000.0), _chunk("c2", price=100.0)]
    kept, excluded_count = filter_chunks(chunks, QueryPreferences(budget=500.0))
    assert [c.chunk_id for c in kept] == ["c2"]
    assert excluded_count == 1


def test_filter_chunks_permissive_on_missing_metadata():
    chunks = [_chunk("c1", city=None, price=None), _chunk("c2", city="lahore", price=100.0)]
    kept, excluded_count = filter_chunks(chunks, QueryPreferences(city="paris", budget=50.0))
    assert [c.chunk_id for c in kept] == ["c1"]
    assert excluded_count == 1


def test_filter_chunks_passes_everything_with_no_preferences():
    chunks = [_chunk("c1", city="paris", price=1000.0), _chunk("c2")]
    kept, excluded_count = filter_chunks(chunks, QueryPreferences())
    assert len(kept) == 2
    assert excluded_count == 0


def test_filter_chunks_city_match_is_case_insensitive_via_lowercased_storage():
    # Both extraction paths lowercase city, so equality is effectively case-insensitive
    chunks = [_chunk("c1", city="lahore")]
    kept, excluded_count = filter_chunks(chunks, QueryPreferences(city="lahore"))
    assert len(kept) == 1
    assert excluded_count == 0
