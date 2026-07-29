# tests/indexing/test_embeddings.py
import httpx
import pytest

from app.config import Settings
from app.indexing.embeddings import EmbeddingError, embed_texts

SETTINGS = Settings()


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_embed_texts_returns_vectors():
    def handler(request):
        body = request.read()
        assert b"qwen3-embedding" in body
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]})

    async with _client(handler) as client:
        vectors = await embed_texts(client, ["a", "b"], SETTINGS)
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


async def test_embed_texts_empty_input_short_circuits():
    async def unreachable(request):
        raise AssertionError("should not be called for empty input")

    async with _client(unreachable) as client:
        vectors = await embed_texts(client, [], SETTINGS)
    assert vectors == []


async def test_embed_texts_raises_on_non_200():
    def handler(request):
        return httpx.Response(500, text="boom")

    async with _client(handler) as client:
        with pytest.raises(EmbeddingError, match="500"):
            await embed_texts(client, ["a"], SETTINGS)


async def test_embed_texts_raises_on_mismatched_vector_count():
    def handler(request):
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2]]})

    async with _client(handler) as client:
        with pytest.raises(EmbeddingError, match="mismatched"):
            await embed_texts(client, ["a", "b"], SETTINGS)
