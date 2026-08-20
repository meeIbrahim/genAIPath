from __future__ import annotations

from app.config import Settings
from app.indexing.chunker import TextChunk


def chunk(text: str, settings: Settings) -> list[TextChunk]:
    raise NotImplementedError("hierarchical chunking strategy not yet implemented — see piece B")
