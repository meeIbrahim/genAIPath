from __future__ import annotations

from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    top_k: int | None = None


class Citation(BaseModel):
    marker: int
    chunk_id: str


class FusedChunk(BaseModel):
    chunk_id: str
    text: str
    source_url: str
    page_number: int
    bm25_rank: int | None
    bm25_score: float | None
    semantic_rank: int | None
    semantic_score: float | None
    fused_rank: int
    rrf_score: float
    matched_methods: list[str]
    used_in_synthesis: bool = False


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[FusedChunk]
