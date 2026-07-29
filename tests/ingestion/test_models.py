from app.ingestion.models import (
    IngestionPayload,
    IngestRequest,
    JobStatusResponse,
    PageMapEntry,
    Stage,
    UrlStatus,
)


def test_stage_values_match_contract():
    assert Stage.QUEUED.value == "queued"
    assert Stage.ERROR.value == "error"
    assert [s.value for s in Stage] == [
        "queued", "fetching", "paginating", "cleaning", "indexing", "done", "error",
    ]


def test_url_status_defaults():
    status = UrlStatus(url="https://example.com")
    assert status.stage == Stage.QUEUED
    assert status.pages_fetched == 0
    assert status.pages_total is None
    assert status.error is None


def test_job_status_response_serializes():
    response = JobStatusResponse(
        job_id="abc",
        urls=[UrlStatus(url="https://example.com", stage=Stage.DONE, pages_fetched=2, pages_total=2)],
    )
    dumped = response.model_dump()
    assert dumped["urls"][0]["stage"] == "done"


def test_ingestion_payload_with_page_map():
    payload = IngestionPayload(
        source_url="https://example.com",
        cleaned_text="hello world",
        pages_fetched=1,
        fetched_at="2026-07-29T12:00:00+00:00",
        page_map=[PageMapEntry(page=1, char_start=0, char_end=11)],
    )
    assert payload.page_map[0].page == 1


def test_ingest_request_parses_url_list():
    request = IngestRequest(urls=["https://a.com", "https://b.com"])
    assert len(request.urls) == 2
