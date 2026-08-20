from __future__ import annotations

from pydantic import BaseModel

from app.pipeline.config import PipelineConfig


class DocFailure(BaseModel):
    path: str
    error: str


class IndexedSummary(BaseModel):
    new_docs: int
    total_docs: int
    failures: list[DocFailure]


class EvalResult(BaseModel):
    status: str


class PipelineLoadResult(BaseModel):
    indexed: IndexedSummary
    eval: EvalResult


class PipelineStatus(BaseModel):
    active: PipelineConfig | None
    doc_counts: dict[str, int]
