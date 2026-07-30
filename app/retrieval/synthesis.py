from __future__ import annotations

import re

import httpx

from app.config import Settings, settings as default_settings
from app.retrieval.models import Citation, FusedChunk

_CITATION_RE = re.compile(r"\[(\d+)\]")

SYSTEM_PROMPT = (
    "Answer only using the numbered context chunks provided below. "
    "Every claim you make must be immediately followed by a citation marker "
    "like [1] or [2], referencing the chunk's position in the context list. "
    "If the context does not contain the answer, say so plainly."
)


class SynthesisError(Exception):
    pass


def _build_context_block(chunks: list[FusedChunk]) -> str:
    return "\n\n".join(f"[{index + 1}] {chunk.text}" for index, chunk in enumerate(chunks))


def _extract_citations(answer: str, chunks: list[FusedChunk]) -> list[Citation]:
    markers = sorted({int(marker) for marker in _CITATION_RE.findall(answer)})
    return [
        Citation(marker=marker, chunk_id=chunks[marker - 1].chunk_id)
        for marker in markers
        if 1 <= marker <= len(chunks)
    ]


async def synthesize_answer(
    query: str,
    fused_chunks: list[FusedChunk],
    http_client: httpx.AsyncClient,
    settings: Settings = default_settings,
) -> tuple[str, list[Citation], set[str]]:
    if not settings.groq_api_key:
        raise SynthesisError("GROQ_API_KEY is not configured")

    context_chunks = fused_chunks[: settings.synthesis_context_budget]
    context_block = _build_context_block(context_chunks)

    try:
        response = await http_client.post(
            f"{settings.groq_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {settings.groq_api_key}"},
            json={
                "model": settings.groq_model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {query}"},
                ],
            },
            timeout=settings.fetch_timeout_seconds,
        )
    except httpx.RequestError as exc:
        raise SynthesisError(f"synthesis request failed: {exc}") from exc

    if response.status_code != 200:
        raise SynthesisError(f"synthesis request returned status {response.status_code}")

    try:
        answer = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise SynthesisError("synthesis response missing answer content") from exc

    citations = _extract_citations(answer, context_chunks)
    used_chunk_ids = {chunk.chunk_id for chunk in context_chunks}
    return answer, citations, used_chunk_ids
