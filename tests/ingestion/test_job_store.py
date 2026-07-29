import pytest

from app.ingestion.job_store import JobStore
from app.ingestion.models import Stage


def test_create_job_returns_id_with_queued_urls():
    store = JobStore()
    job_id = store.create_job(["https://a.com", "https://b.com"])
    status = store.get_status(job_id)
    assert status.job_id == job_id
    assert [u.url for u in status.urls] == ["https://a.com", "https://b.com"]
    assert all(u.stage == Stage.QUEUED for u in status.urls)


def test_update_changes_single_url_status():
    store = JobStore()
    job_id = store.create_job(["https://a.com", "https://b.com"])
    store.update(job_id, "https://a.com", stage=Stage.FETCHING, pages_fetched=1)
    status = store.get_status(job_id)
    a_status = next(u for u in status.urls if u.url == "https://a.com")
    b_status = next(u for u in status.urls if u.url == "https://b.com")
    assert a_status.stage == Stage.FETCHING
    assert a_status.pages_fetched == 1
    assert b_status.stage == Stage.QUEUED


def test_exists_false_for_unknown_job():
    store = JobStore()
    assert store.exists("does-not-exist") is False
