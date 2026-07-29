from __future__ import annotations

import httpx

from app.config import Settings


class EmbeddingError(Exception):
    pass


async def embed_texts(client: httpx.AsyncClient, texts: list[str], settings: Settings) -> list[list[float]]:
    if not texts:
        return []

    try:
        response = await client.post(
            f"{settings.ollama_base_url}/api/embed",
            json={"model": settings.embedding_model, "input": texts},
            timeout=settings.fetch_timeout_seconds,
        )
    except httpx.RequestError as exc:
        raise EmbeddingError(f"embedding request failed: {exc}") from exc

    if response.status_code != 200:
        raise EmbeddingError(f"embedding request returned status {response.status_code}")

    embeddings = response.json().get("embeddings")
    if not embeddings or len(embeddings) != len(texts):
        raise EmbeddingError("embedding response missing or mismatched vectors")
    return embeddings
