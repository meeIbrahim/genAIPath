from __future__ import annotations

from typing import Literal

import httpx
from pydantic import BaseModel

from app.config import Settings, settings as default_settings
from app.retrieval.models import FusedChunk

JUDGE_SYSTEM_PROMPT = (
    "You are a strict grader. Given a question and numbered context excerpts, decide "
    "whether the excerpts contain enough information to answer the question accurately "
    "and completely. Respond with exactly one word: \"context_good\" if the excerpts are "
    "sufficient, or \"context_insufficient\" if they are not."
)


class JudgeVerdict(BaseModel):
    verdict: Literal["context_good", "context_insufficient"]
    raw_response: str


class JudgeError(Exception):
    pass


def _build_context_block(chunks: list[FusedChunk]) -> str:
    return "\n\n".join(f"[{index + 1}] {chunk.text}" for index, chunk in enumerate(chunks))


async def judge_context(
    query: str,
    chunks: list[FusedChunk],
    http_client: httpx.AsyncClient,
    settings: Settings = default_settings,
) -> JudgeVerdict:
    if not chunks:
        return JudgeVerdict(verdict="context_insufficient", raw_response="(no chunks retrieved)")

    if not settings.groq_api_key:
        raise JudgeError("GROQ_API_KEY is not configured")

    context_block = _build_context_block(chunks)

    try:
        response = await http_client.post(
            f"{settings.groq_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {settings.groq_api_key}"},
            json={
                "model": settings.judge_model,
                "messages": [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Question: {query}\n\nContext:\n{context_block}"},
                ],
            },
            timeout=settings.fetch_timeout_seconds,
        )
    except httpx.RequestError as exc:
        raise JudgeError(f"judge request failed: {exc}") from exc

    if response.status_code != 200:
        raise JudgeError(f"judge request returned status {response.status_code}")

    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise JudgeError("judge response missing verdict content") from exc

    normalized = content.strip().lower()
    if "insufficient" in normalized:
        return JudgeVerdict(verdict="context_insufficient", raw_response=content)
    if "good" in normalized:
        return JudgeVerdict(verdict="context_good", raw_response=content)
    raise JudgeError(f"judge response did not contain a recognizable verdict: {content!r}")
