import httpx
import pytest

from app.config import Settings
from app.retrieval.judge import JudgeError, judge_context
from app.retrieval.models import FusedChunk


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


async def test_judge_context_returns_good_verdict():
    settings = Settings(groq_api_key="test-key")
    chunks = [_chunk("c1", "Paris is the capital of France.")]

    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: _groq_response("context_good"))) as client:
        verdict = await judge_context("capital of France?", chunks, client, settings)

    assert verdict.verdict == "context_good"
    assert verdict.raw_response == "context_good"


async def test_judge_context_returns_insufficient_verdict():
    settings = Settings(groq_api_key="test-key")
    chunks = [_chunk("c1", "unrelated text")]

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: _groq_response("context_insufficient"))
    ) as client:
        verdict = await judge_context("capital of France?", chunks, client, settings)

    assert verdict.verdict == "context_insufficient"
    assert verdict.raw_response == "context_insufficient"


async def test_judge_context_uses_judge_model_not_synthesis_model():
    settings = Settings(groq_api_key="test-key")
    captured = {}

    def handler(request):
        import json

        captured["model"] = json.loads(request.read())["model"]
        return _groq_response("context_good")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await judge_context("q", [_chunk("c1", "x")], client, settings)

    assert captured["model"] == settings.judge_model


async def test_judge_context_empty_chunks_short_circuits_without_request():
    settings = Settings(groq_api_key="test-key")

    def handler(request):
        raise AssertionError("should not make a request for empty chunks")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        verdict = await judge_context("q", [], client, settings)

    assert verdict.verdict == "context_insufficient"
    assert verdict.raw_response == "(no chunks retrieved)"


async def test_judge_context_raw_response_preserves_verbose_content():
    # raw_response must carry the FULL text, not just the matched substring —
    # this is what the side panel will render verbatim.
    settings = Settings(groq_api_key="test-key")
    verbose = "This context is definitely context_good because it directly answers the question."

    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: _groq_response(verbose))) as client:
        verdict = await judge_context("q", [_chunk("c1", "x")], client, settings)

    assert verdict.verdict == "context_good"
    assert verdict.raw_response == verbose


async def test_judge_context_raises_without_api_key():
    settings = Settings(groq_api_key="")
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200))) as client:
        with pytest.raises(JudgeError, match="GROQ_API_KEY"):
            await judge_context("q", [_chunk("c1", "x")], client, settings)


async def test_judge_context_raises_on_non_200():
    settings = Settings(groq_api_key="test-key")

    def handler(request):
        return httpx.Response(500, text="boom")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(JudgeError, match="500"):
            await judge_context("q", [_chunk("c1", "x")], client, settings)


async def test_judge_context_raises_on_unrecognized_verdict():
    settings = Settings(groq_api_key="test-key")

    def handler(request):
        return _groq_response("maybe? unclear")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(JudgeError, match="recognizable verdict"):
            await judge_context("q", [_chunk("c1", "x")], client, settings)
