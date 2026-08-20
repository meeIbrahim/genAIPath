from __future__ import annotations

from app.config import Settings
from app.indexing.chunker import TextChunk, chunk_text


def chunk(text: str, settings: Settings) -> list[TextChunk]:
    return chunk_text(text, settings.chunk_size_tokens, settings.chunk_overlap_tokens)
