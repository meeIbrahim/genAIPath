from __future__ import annotations

from app.extraction.preferences import QueryPreferences
from app.retrieval.models import FusedChunk


def filter_chunks(
    fused_chunks: list[FusedChunk],
    preferences: QueryPreferences,
) -> tuple[list[FusedChunk], int]:
    kept = [chunk for chunk in fused_chunks if not _conflicts(chunk, preferences)]
    return kept, len(fused_chunks) - len(kept)


def _conflicts(chunk: FusedChunk, preferences: QueryPreferences) -> bool:
    if chunk.city is not None and preferences.city is not None and chunk.city != preferences.city:
        return True
    if chunk.price is not None and preferences.budget is not None and chunk.price > preferences.budget:
        return True
    return False
