from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class Stage(str, Enum):
    QUEUED = "queued"
    FETCHING = "fetching"
    PAGINATING = "paginating"
    CLEANING = "cleaning"
    INDEXING = "indexing"
    DONE = "done"
    ERROR = "error"


class IngestRequest(BaseModel):
    urls: list[str]


class IngestResponse(BaseModel):
    job_id: str


class UrlStatus(BaseModel):
    url: str
    stage: Stage = Stage.QUEUED
    pages_fetched: int = 0
    pages_total: int | None = None
    error: str | None = None


class JobStatusResponse(BaseModel):
    job_id: str
    urls: list[UrlStatus]


class PageMapEntry(BaseModel):
    page: int
    char_start: int
    char_end: int


class IngestionPayload(BaseModel):
    source_url: str
    cleaned_text: str
    pages_fetched: int
    fetched_at: str
    page_map: list[PageMapEntry]
