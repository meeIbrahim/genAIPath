from __future__ import annotations

import uuid

from app.ingestion.models import JobStatusResponse, UrlStatus


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, UrlStatus]] = {}

    def create_job(self, urls: list[str]) -> str:
        job_id = str(uuid.uuid4())
        self._jobs[job_id] = {url: UrlStatus(url=url) for url in urls}
        return job_id

    def get_status(self, job_id: str) -> JobStatusResponse:
        urls = list(self._jobs[job_id].values())
        return JobStatusResponse(job_id=job_id, urls=urls)

    def update(self, job_id: str, url: str, **fields) -> None:
        current = self._jobs[job_id][url]
        self._jobs[job_id][url] = current.model_copy(update=fields)

    def exists(self, job_id: str) -> bool:
        return job_id in self._jobs
