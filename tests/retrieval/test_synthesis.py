import httpx
import pytest

from app.config import Settings
from app.retrieval.models import FusedChunk
from app.retrieval.synthesis import SynthesisError, synthesize_answer


def _chunk(chunk_id: str, text: str) -> FusedChunk:
    return FusedChunk(
        chunk_id=chunk_id,
        text=text,
        source_url="https://example.com",
        page_number=1,
        bm25_rank=1,
        bm25_score=1.0,
        semantic_rank=1,
        semantic_score=1.0,
        fused_rank=1,
        rrf_score=0.03,
        matched_methods=["bm25", "semantic"],
    )


def _groq_response(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


async def test_synthesize_answer_extracts_citations_within_budget():
    settings = Settings(groq_api_key="test-key", synthesis_context_budget=2)
    chunks = [_chunk("c1", "Paris is the capital of France."), _chunk("c2", "It has the Eiffel Tower.")]

    def handler(request):
        assert request.headers["authorization"] == "Bearer test-key"
        return _groq_response("Paris is the capital of France [1], home to the Eiffel Tower [2].")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        answer, citations, used_chunk_ids = await synthesize_answer("capital of France?", chunks, client, settings)

    assert "[1]" in answer and "[2]" in answer
    assert {c.chunk_id for c in citations} == {"c1", "c2"}
    assert used_chunk_ids == {"c1", "c2"}


async def test_synthesize_answer_respects_context_budget():
    settings = Settings(groq_api_key="test-key", synthesis_context_budget=1)
    chunks = [_chunk("c1", "first"), _chunk("c2", "second")]

    captured = {}

    def handler(request):
        captured["body"] = request.read()
        return _groq_response("first fact [1].")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        _answer, citations, used_chunk_ids = await synthesize_answer("q", chunks, client, settings)

    assert b"second" not in captured["body"]
    assert used_chunk_ids == {"c1"}
    assert {c.chunk_id for c in citations} == {"c1"}


async def test_synthesize_answer_raises_without_api_key():
    settings = Settings(groq_api_key="")
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200))) as client:
        with pytest.raises(SynthesisError, match="GROQ_API_KEY"):
            await synthesize_answer("q", [_chunk("c1", "x")], client, settings)


async def test_synthesize_answer_raises_on_non_200():
    settings = Settings(groq_api_key="test-key")

    def handler(request):
        return httpx.Response(500, text="boom")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(SynthesisError, match="500"):
            await synthesize_answer("q", [_chunk("c1", "x")], client, settings)


async def test_synthesize_answer_raises_on_invalid_json():
    """Malformed 200 response with invalid JSON should raise SynthesisError."""
    settings = Settings(groq_api_key="test-key")

    def handler(request):
        return httpx.Response(200, text="not json")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(SynthesisError, match="synthesis response missing answer content"):
            await synthesize_answer("q", [_chunk("c1", "x")], client, settings)


async def test_synthesize_answer_raises_on_unexpected_json_shape():
    """Malformed 200 response with unexpected JSON structure should raise SynthesisError."""
    settings = Settings(groq_api_key="test-key")

    def handler(request):
        return httpx.Response(200, json=["unexpected", "shape"])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(SynthesisError, match="synthesis response missing answer content"):
            await synthesize_answer("q", [_chunk("c1", "x")], client, settings)
