from unittest.mock import patch

import httpx
import pytest

from app.config import Settings
from app.ingestion.job_store import JobStore
from app.ingestion.models import IngestionPayload, Stage
from app.ingestion.worker import ingest_url

SETTINGS = Settings(min_extract_length=10, max_pages=5)

PAGE_1 = """
<html><head><link rel="next" href="/post?page=2"></head>
<body><article><p>First page content, long enough to pass extraction.</p></article></body></html>
"""
PAGE_2 = """
<html><body><article><p>Second page content, also long enough.</p></article></body></html>
"""


def _handler(pages: dict[str, httpx.Response]):
    def handler(request):
        return pages[str(request.url)]
    return handler


async def test_ingest_url_success_multi_page_sets_done_and_calls_sink():
    pages = {
        "https://example.com/post": httpx.Response(200, text=PAGE_1),
        "https://example.com/post?page=2": httpx.Response(200, text=PAGE_2),
    }
    store = JobStore()
    job_id = store.create_job(["https://example.com/post"])
    captured: list[IngestionPayload] = []

    async def sink(payload: IngestionPayload) -> None:
        captured.append(payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(_handler(pages))) as client:
        await ingest_url(job_id, "https://example.com/post", store, client, sink=sink, settings=SETTINGS)

    status = store.get_status(job_id).urls[0]
    assert status.stage == Stage.DONE
    assert status.pages_fetched == 2
    assert len(captured) == 1
    assert captured[0].pages_fetched == 2
    assert len(captured[0].page_map) == 2
    assert "First page content" in captured[0].cleaned_text
    assert "Second page content" in captured[0].cleaned_text


async def test_ingest_url_page_map_char_offsets_are_correct():
    # Pin the char-offset arithmetic with real numbers, not just a length
    # check: an off-by-N error here would silently corrupt downstream
    # retrieval-chunk offsets. Segments are single paragraphs in this
    # fixture, so the "\n\n" join separator is the only thing between them.
    pages = {
        "https://example.com/post": httpx.Response(200, text=PAGE_1),
        "https://example.com/post?page=2": httpx.Response(200, text=PAGE_2),
    }
    store = JobStore()
    job_id = store.create_job(["https://example.com/post"])
    captured: list[IngestionPayload] = []

    async def sink(payload: IngestionPayload) -> None:
        captured.append(payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(_handler(pages))) as client:
        await ingest_url(job_id, "https://example.com/post", store, client, sink=sink, settings=SETTINGS)

    segment_1, segment_2 = captured[0].cleaned_text.split("\n\n")
    page_map = captured[0].page_map
    assert page_map[0].char_start == 0
    assert page_map[0].char_end == len(segment_1)
    assert page_map[1].char_start == len(segment_1) + 2
    assert page_map[1].char_end == page_map[1].char_start + len(segment_2)
    assert page_map[1].char_end == len(captured[0].cleaned_text)


async def test_ingest_url_fetch_failure_sets_error_and_does_not_call_sink():
    def handler(request):
        return httpx.Response(500, text="boom")

    store = JobStore()
    job_id = store.create_job(["https://example.com/broken"])
    called = False

    async def sink(payload: IngestionPayload) -> None:
        nonlocal called
        called = True

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await ingest_url(job_id, "https://example.com/broken", store, client, sink=sink, settings=SETTINGS)

    status = store.get_status(job_id).urls[0]
    assert status.stage == Stage.ERROR
    assert "500" in status.error
    assert called is False


async def test_ingest_url_extraction_failure_sets_error():
    def handler(request):
        return httpx.Response(200, text="<html><body><nav>x</nav></body></html>")

    store = JobStore()
    job_id = store.create_job(["https://example.com/empty"])

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await ingest_url(job_id, "https://example.com/empty", store, client, settings=SETTINGS)

    status = store.get_status(job_id).urls[0]
    assert status.stage == Stage.ERROR
    assert status.error == "no extractable content"


async def test_ingest_url_unexpected_exception_sets_error_and_does_not_propagate():
    # A non-FetchError/non-ExtractionError exception (e.g. a bug in the
    # extraction library) must still be isolated to this URL: ingest_url
    # should never let it escape, since a sibling URL in the same job may
    # still be awaiting on the same shared httpx client (see Global
    # Constraints: failure isolation).
    def handler(request):
        return httpx.Response(200, text=PAGE_1)

    store = JobStore()
    job_id = store.create_job(["https://example.com/post"])

    with patch("app.ingestion.worker.extract_main_text", side_effect=RuntimeError("boom")):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await ingest_url(job_id, "https://example.com/post", store, client, settings=SETTINGS)

    status = store.get_status(job_id).urls[0]
    assert status.stage == Stage.ERROR
    assert status.error == "boom"


TEMPLATE_PAGE_1 = """
<html><body>
<article><p>Numbered page one content, long enough for extraction.</p></article>
<a href="/articles/1?page=2">2</a>
<a href="/articles/1?page=3">3</a>
</body></html>
"""
TEMPLATE_PAGE_2 = """
<html><body><article><p>Numbered page two content, also long enough.</p></article></body></html>
"""
TEMPLATE_PAGE_3 = """
<html><body><article><p>Numbered page three content, also long enough.</p></article></body></html>
"""

TEMPLATE_SETTINGS = Settings(min_extract_length=10, max_pages=3)


async def test_ingest_url_template_mode_pagination_increments_via_render_template():
    base_url = "https://example.com/articles/1"
    page2_url = "https://example.com/articles/1?page=2"
    page3_url = "https://example.com/articles/1?page=3"
    pages = {
        base_url: httpx.Response(200, text=TEMPLATE_PAGE_1),
        page2_url: httpx.Response(200, text=TEMPLATE_PAGE_2),
        page3_url: httpx.Response(200, text=TEMPLATE_PAGE_3),
    }
    requested_urls: list[str] = []

    def handler(request):
        requested_urls.append(str(request.url))
        return pages[str(request.url)]

    store = JobStore()
    job_id = store.create_job([base_url])
    captured: list[IngestionPayload] = []

    async def sink(payload: IngestionPayload) -> None:
        captured.append(payload)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await ingest_url(job_id, base_url, store, client, sink=sink, settings=TEMPLATE_SETTINGS)

    status = store.get_status(job_id).urls[0]
    assert status.stage == Stage.DONE
    assert status.pages_fetched == 3
    # Confirms the increment-via-render_template sequence fetched exactly the
    # expected URLs in order (page 1, then page=2, then page=3) with no
    # re-detection of pagination on later pages.
    assert requested_urls == [base_url, page2_url, page3_url]
    assert len(captured) == 1
    assert captured[0].pages_fetched == 3
    assert "Numbered page one" in captured[0].cleaned_text
    assert "Numbered page two" in captured[0].cleaned_text
    assert "Numbered page three" in captured[0].cleaned_text


CAP_PAGE_1 = """
<html><head><link rel="next" href="/capped?page=2"></head>
<body><article><p>Capped first page content, long enough for extraction.</p></article></body></html>
"""
CAP_PAGE_2 = """
<html><head><link rel="next" href="/capped?page=3"></head>
<body><article><p>Capped second page content, also long enough.</p></article></body></html>
"""

CAPPED_SETTINGS = Settings(min_extract_length=10, max_pages=2)


async def test_ingest_url_max_pages_caps_chain_pagination():
    # PAGE_2 links onward to a page=3 that is never registered in this mock
    # and would raise KeyError in the handler if ever fetched -- proving the
    # max_pages cap actually halts the chain instead of merely reporting a
    # smaller number after the fact.
    base_url = "https://example.com/capped"
    page2_url = "https://example.com/capped?page=2"
    pages = {
        base_url: httpx.Response(200, text=CAP_PAGE_1),
        page2_url: httpx.Response(200, text=CAP_PAGE_2),
    }
    store = JobStore()
    job_id = store.create_job([base_url])

    async with httpx.AsyncClient(transport=httpx.MockTransport(_handler(pages))) as client:
        await ingest_url(job_id, base_url, store, client, settings=CAPPED_SETTINGS)

    status = store.get_status(job_id).urls[0]
    assert status.stage == Stage.DONE
    assert status.pages_fetched == 2
