from __future__ import annotations

from pydantic import BaseModel


class ChunkMetadata(BaseModel):
    chunk_id: str
    doc_id: str
    source_url: str
    page_number: int
    chunk_index: int
    char_start: int
    char_end: int
    overlap_with_prev: int
    indexed_at: str
    text: str
    city: str | None = None
    price: float | None = None


class IndexResult(BaseModel):
    doc_id: str
    status: str
    chunk_count: int
