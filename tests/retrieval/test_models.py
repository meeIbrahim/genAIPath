from app.extraction.preferences import QueryPreferences
from app.retrieval.models import Citation, FusedChunk, QueryRequest, QueryResponse


def test_query_request_top_k_optional():
    request = QueryRequest(query="what is RAG?")
    assert request.top_k is None


def test_fused_chunk_defaults_used_in_synthesis_false():
    chunk = FusedChunk(
        chunk_id="c1", text="hello", source_url="https://example.com", page_number=1,
        bm25_rank=1, bm25_score=8.3, semantic_rank=None, semantic_score=None,
        fused_rank=1, rrf_score=0.016, matched_methods=["bm25"],
    )
    assert chunk.used_in_synthesis is False


def test_fused_chunk_city_and_price_default_none():
    chunk = FusedChunk(
        chunk_id="c1", text="hello", source_url="https://example.com", page_number=1,
        bm25_rank=1, bm25_score=8.3, semantic_rank=None, semantic_score=None,
        fused_rank=1, rrf_score=0.016, matched_methods=["bm25"],
    )
    assert chunk.city is None
    assert chunk.price is None


def test_query_response_round_trip():
    response = QueryResponse(
        query="q", answer="answer [1]", citations=[Citation(marker=1, chunk_id="c1")],
        retrieved_chunks=[], preferences=QueryPreferences(), filtered_out_count=0,
    )
    assert response.model_dump()["citations"][0]["chunk_id"] == "c1"
    assert "judge_attempts" not in response.model_dump()
