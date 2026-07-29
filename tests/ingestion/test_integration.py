# tests/ingestion/test_integration.py
import asyncio
from unittest.mock import patch

import httpx
from fastapi import FastAPI

from app.ingestion.job_store import JobStore
from app.ingestion.models import Stage
from app.ingestion.router import build_ingestion_router

GOOD_PAGE_1 = """
<html><head><link rel="next" href="/good?page=2"></head>
<body><article><p>Good article page one, plenty of real content here to pass the threshold.
This paragraph continues with several additional sentences describing background
and supporting details, giving the extraction pipeline more than enough real prose
to comfortably clear the two hundred character minimum length requirement on its own.</p></article></body></html>
"""
GOOD_PAGE_2 = """
<html><body><article><p>Good article page two, also plenty of real content to pass.
This second page likewise continues with several more sentences of realistic
supporting narrative, ensuring the extracted text for this page alone comfortably
clears the two hundred character minimum length requirement as well.</p></article></body></html>
"""


def _handler(request: httpx.Request) -> httpx.Response:
    url = str(request.url)
    if url == "https://example.com/good":
        return httpx.Response(200, text=GOOD_PAGE_1)
    if url == "https://example.com/good?page=2":
        return httpx.Response(200, text=GOOD_PAGE_2)
    if url == "https://example.com/broken":
        return httpx.Response(500, text="server error")
    return httpx.Response(404, text="not found")


async def _wait_for_terminal_status(client: httpx.AsyncClient, job_id: str, timeout: float = 2.0) -> dict:
    elapsed = 0.0
    interval = 0.01
    while elapsed < timeout:
        response = await client.get(f"/ingest/{job_id}/status")
        body = response.json()
        if all(u["stage"] in ("done", "error") for u in body["urls"]):
            return body
        await asyncio.sleep(interval)
        elapsed += interval
    raise AssertionError(f"job {job_id} did not reach a terminal state within {timeout}s")


async def test_two_urls_isolated_success_and_failure():
    store = JobStore()
    app = FastAPI()
    app.include_router(
        build_ingestion_router(
            store,
            client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(_handler)),
        )
    )

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/ingest",
            json={"urls": ["https://example.com/good", "https://example.com/broken"]},
        )
        job_id = response.json()["job_id"]
        body = await _wait_for_terminal_status(client, job_id)

    by_url = {u["url"]: u for u in body["urls"]}
    good = by_url["https://example.com/good"]
    broken = by_url["https://example.com/broken"]

    assert good["stage"] == Stage.DONE.value
    assert good["pages_fetched"] == 2
    assert good["error"] is None

    assert broken["stage"] == Stage.ERROR.value
    assert broken["error"] is not None
    assert "500" in broken["error"]


async def test_two_urls_one_raises_unexpected_exception_sibling_still_completes():
    # Regression test for the failure-isolation defect: an unexpected
    # exception (not FetchError/ExtractionError) from one URL's coroutine
    # must not propagate through asyncio.gather and close the shared httpx
    # client while a sibling URL is still mid-flight.
    store = JobStore()
    app = FastAPI()
    app.include_router(
        build_ingestion_router(
            store,
            client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(_handler)),
        )
    )

    real_extract_main_text = __import__(
        "app.ingestion.worker", fromlist=["extract_main_text"]
    ).extract_main_text

    def flaky_extract_main_text(html, settings):
        if "page two" in html:
            raise RuntimeError("boom")
        return real_extract_main_text(html, settings)

    with patch("app.ingestion.worker.extract_main_text", side_effect=flaky_extract_main_text):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/ingest",
                json={"urls": ["https://example.com/good", "https://example.com/broken"]},
            )
            job_id = response.json()["job_id"]
            body = await _wait_for_terminal_status(client, job_id)

    by_url = {u["url"]: u for u in body["urls"]}
    good = by_url["https://example.com/good"]
    broken = by_url["https://example.com/broken"]

    # "good" hits the patched extractor on its second page and raises the
    # unexpected RuntimeError -- it must land in ERROR, not propagate.
    assert good["stage"] == Stage.ERROR.value
    assert good["error"] == "boom"

    # The sibling URL must still reach a terminal stage normally, proving it
    # was not affected by the other URL's unexpected exception.
    assert broken["stage"] == Stage.ERROR.value
    assert "500" in broken["error"]
