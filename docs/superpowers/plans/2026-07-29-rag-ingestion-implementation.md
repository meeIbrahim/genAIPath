# RAG Ingestion Subsystem Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the ingestion subsystem — `POST /ingest`, `GET /ingest/{job_id}/status`, and the per-URL async worker (fetch → paginate → clean → merge → emit) — per `doc/development/ingestion.md`.

**Architecture:** FastAPI app with one router (`app/ingestion/router.py`) backed by an in-memory `JobStore`. Each submitted URL runs as an independent `asyncio.Task` (`app/ingestion/worker.py`) that fetches with `httpx`, detects pagination with BeautifulSoup, extracts main content with `trafilatura` (fallback `readability-lxml`), merges pages, and hands a normalized payload to an injectable sink (a no-op stub in this plan — Plan 2 wires in the real indexer).

**Tech Stack:** Python 3.11, FastAPI, httpx (async), BeautifulSoup4 + lxml, trafilatura, readability-lxml, pydantic, pytest + pytest-asyncio, uv.

## Global Constraints

- Python: bump `requires-python` to `>=3.11` (was `>=3.9`, unused elsewhere).
- Fetch: `httpx` async client, timeout `10s`, realistic `User-Agent`, standard HTTP redirects only (`follow_redirects=True`), no link-following.
- Pagination detection priority (stop at first match): `<link rel="next">` → `<a rel="next">`/text matching `/^(next|›|»|more)$/i` with same domain+path-prefix → numbered cluster (`?page=N`, `/page/N/`) → single page.
- Hard cap: max pages = `20` (config, not hardcoded).
- Extraction: `trafilatura` primary, `readability-lxml` fallback. Near-empty result (below `min_extract_length` config, default `200` chars) → error, exact string `"no extractable content"`.
- Pagination parsing library: BeautifulSoup (BS4), per user decision — not `readability`/regex-only.
- Stage enum (stable contract): `queued | fetching | paginating | cleaning | indexing | done | error`.
- Failure isolation: one URL's exception must never affect sibling URLs in the same job.
- Task orchestration: `asyncio.Task` per URL (no Celery/Redis).
- Progress transport: polling only (`GET /ingest/{job_id}/status`), no SSE/WebSocket.
- No recursive crawling, no JS rendering / headless browser.

---

## File Structure

```
app/
  __init__.py
  config.py                  # Settings (timeout, user-agent, max_pages, min_extract_length)
  main.py                    # FastAPI app factory + module-level `app`
  ingestion/
    __init__.py
    models.py                # Stage, IngestRequest/Response, UrlStatus, JobStatusResponse, PageMapEntry, IngestionPayload
    fetcher.py                # fetch_page() + FetchError
    pagination.py             # detect_pagination(), detect_next_url(), detect_numbered_template(), render_template()
    extractor.py               # extract_main_text() + ExtractionError
    job_store.py                # JobStore class
    worker.py                    # ingest_url() orchestrator + IngestSink type
    router.py                     # build_ingestion_router()
tests/
  ingestion/
    test_config.py
    test_models.py
    test_pagination.py
    test_fetcher.py
    test_extractor.py
    test_job_store.py
    test_worker.py
    test_router.py
    test_integration.py
```

Each module has one responsibility; `worker.py` is the only module that imports the other four ingestion modules together (it's the orchestrator).

---

### Task 1: Project setup & Settings config

**Files:**
- Modify: `pyproject.toml`
- Modify: `.python-version`
- Create: `app/__init__.py`
- Create: `app/config.py`
- Test: `tests/ingestion/test_config.py`

**Interfaces:**
- Produces: `Settings` frozen dataclass with fields `fetch_timeout_seconds: float`, `user_agent: str`, `max_pages: int`, `min_extract_length: int`; module-level instance `settings: Settings`.

- [ ] **Step 1: Bump Python version and add core dependencies**

```bash
echo "3.11" > .python-version
```

Edit `pyproject.toml`: change `requires-python = ">=3.9"` to `requires-python = ">=3.11"`.

```bash
uv add fastapi "uvicorn[standard]" httpx
uv add --dev pytest pytest-asyncio
```

- [ ] **Step 2: Add pytest-asyncio config**

Append to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

- [ ] **Step 3: Write the failing test**

```python
# tests/ingestion/test_config.py
from app.config import Settings, settings


def test_defaults():
    assert settings.fetch_timeout_seconds == 10.0
    assert settings.max_pages == 20
    assert settings.min_extract_length == 200
    assert "GenAI" in settings.user_agent


def test_settings_is_frozen():
    with __import__("pytest").raises(Exception):
        settings.max_pages = 5
```

- [ ] **Step 4: Run test to verify it fails**

Run: `uv run pytest tests/ingestion/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app'`

- [ ] **Step 5: Create package init and implementation**

```python
# app/__init__.py
```

```python
# app/config.py
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    fetch_timeout_seconds: float = 10.0
    user_agent: str = "Mozilla/5.0 (compatible; GenAI-RAG-Ingest/1.0)"
    max_pages: int = 20
    min_extract_length: int = 200


settings = Settings()
```

- [ ] **Step 6: Run test to verify it passes**

Run: `uv run pytest tests/ingestion/test_config.py -v`
Expected: PASS (2 tests)

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock .python-version app/__init__.py app/config.py tests/ingestion/test_config.py
git commit -m "chore: bump to Python 3.11, add FastAPI/httpx deps, Settings config"
```

---

### Task 2: Data models

**Files:**
- Create: `app/ingestion/__init__.py`
- Create: `app/ingestion/models.py`
- Test: `tests/ingestion/test_models.py`

**Interfaces:**
- Consumes: nothing (no dependency on other ingestion modules).
- Produces:
  - `Stage(str, Enum)`: `QUEUED, FETCHING, PAGINATING, CLEANING, INDEXING, DONE, ERROR` (values are the lowercase strings from the contract).
  - `IngestRequest(BaseModel)`: `urls: list[str]`
  - `IngestResponse(BaseModel)`: `job_id: str`
  - `UrlStatus(BaseModel)`: `url: str`, `stage: Stage = Stage.QUEUED`, `pages_fetched: int = 0`, `pages_total: int | None = None`, `error: str | None = None`
  - `JobStatusResponse(BaseModel)`: `job_id: str`, `urls: list[UrlStatus]`
  - `PageMapEntry(BaseModel)`: `page: int`, `char_start: int`, `char_end: int`
  - `IngestionPayload(BaseModel)`: `source_url: str`, `cleaned_text: str`, `pages_fetched: int`, `fetched_at: str`, `page_map: list[PageMapEntry]`

- [ ] **Step 1: Write the failing test**

```python
# tests/ingestion/test_models.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ingestion/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.ingestion'`

- [ ] **Step 3: Implement models**

```python
# app/ingestion/__init__.py
```

```python
# app/ingestion/models.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ingestion/test_models.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add app/ingestion/__init__.py app/ingestion/models.py tests/ingestion/test_models.py
git commit -m "feat: add ingestion data models"
```

---

### Task 3: Pagination detection

**Files:**
- Create: `app/ingestion/pagination.py`
- Test: `tests/ingestion/test_pagination.py`

**Interfaces:**
- Consumes: nothing (pure functions over HTML strings).
- Produces:
  - `PaginationPlan` frozen dataclass: `mode: str` (`"single" | "chain" | "template"`), `next_url: str | None`, `template: str | None`, `start_page_number: int`
  - `detect_pagination(html: str, current_url: str) -> PaginationPlan`
  - `detect_next_url(html: str, current_url: str) -> str | None` (checks `rel=next` link, then anchor rel/text match with same domain+path-prefix)
  - `detect_numbered_template(html: str, current_url: str) -> str | None`
  - `render_template(template: str, page_number: int) -> str`

**Design note for the implementer:** two distinct pagination behaviors, per the spec's "detected pagination chain" language:
- `chain` mode (rel=next / anchor-next): re-run `detect_next_url` on *each newly fetched page* to get the next link. This is still bounded/deterministic (same single rule, not general link scanning) — it is the mechanism the spec calls "chain."
- `template` mode (numbered cluster): detected **once** on the first page. Subsequent pages are produced by incrementing the integer in the template — the worker must NOT re-parse later pages for more numbered links.

- [ ] **Step 1: Write the failing test**

```python
# tests/ingestion/test_pagination.py
from app.ingestion.pagination import detect_pagination, detect_next_url, render_template

RENDER_NEXT_LINK = """
<html><head><link rel="next" href="/blog/post-1?page=2"></head><body>x</body></html>
"""

ANCHOR_NEXT_TEXT = """
<html><body><a href="/blog/post-1/2">Next</a></body></html>
"""

ANCHOR_NEXT_DIFFERENT_DOMAIN = """
<html><body><a href="https://other.com/2">Next</a></body></html>
"""

NUMBERED_CLUSTER = """
<html><body>
<a href="/blog?page=2">2</a>
<a href="/blog?page=3">3</a>
<a href="/blog?page=4">4</a>
</body></html>
"""

NO_PAGINATION = "<html><body><p>just an article</p></body></html>"


def test_link_rel_next_wins_top_priority():
    plan = detect_pagination(RENDER_NEXT_LINK, "https://example.com/blog/post-1")
    assert plan.mode == "chain"
    assert plan.next_url == "https://example.com/blog/post-1?page=2"


def test_anchor_next_text_same_path_prefix():
    url = detect_next_url(ANCHOR_NEXT_TEXT, "https://example.com/blog/post-1")
    assert url == "https://example.com/blog/post-1/2"


def test_anchor_next_different_domain_rejected():
    url = detect_next_url(ANCHOR_NEXT_DIFFERENT_DOMAIN, "https://example.com/blog/post-1")
    assert url is None


def test_numbered_cluster_detected_as_template():
    plan = detect_pagination(NUMBERED_CLUSTER, "https://example.com/blog")
    assert plan.mode == "template"
    assert plan.start_page_number == 2
    assert render_template(plan.template, 2) == "https://example.com/blog?page=2"
    assert render_template(plan.template, 3) == "https://example.com/blog?page=3"


def test_no_pagination_signal_is_single():
    plan = detect_pagination(NO_PAGINATION, "https://example.com/blog")
    assert plan.mode == "single"
    assert plan.next_url is None
    assert plan.template is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ingestion/test_pagination.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.ingestion.pagination'`

- [ ] **Step 3: Implement pagination detection**

```python
# app/ingestion/pagination.py
from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

_NEXT_TEXT_RE = re.compile(r"^(next|›|»|more)$", re.IGNORECASE)
_NUMBERED_RE = re.compile(r"(\?page=|&page=|/page/)(\d+)")


@dataclass(frozen=True)
class PaginationPlan:
    mode: str  # "single" | "chain" | "template"
    next_url: str | None = None
    template: str | None = None
    start_page_number: int = 1


def _same_path_prefix(current_url: str, candidate_url: str) -> bool:
    current = urlparse(current_url)
    candidate = urlparse(candidate_url)
    if current.netloc != candidate.netloc:
        return False
    return current.path.rsplit("/", 1)[0] == candidate.path.rsplit("/", 1)[0]


def detect_next_url(html: str, current_url: str) -> str | None:
    soup = BeautifulSoup(html, "lxml")

    link_next = soup.find("link", rel="next")
    if link_next and link_next.get("href"):
        return urljoin(current_url, link_next["href"])

    for anchor in soup.find_all("a", href=True):
        rel = anchor.get("rel") or []
        text = anchor.get_text(strip=True)
        is_next_rel = "next" in rel
        is_next_text = bool(text) and bool(_NEXT_TEXT_RE.match(text))
        if not (is_next_rel or is_next_text):
            continue
        candidate = urljoin(current_url, anchor["href"])
        if _same_path_prefix(current_url, candidate):
            return candidate
    return None


def detect_numbered_template(html: str, current_url: str) -> str | None:
    soup = BeautifulSoup(html, "lxml")
    current_domain = urlparse(current_url).netloc
    candidates: dict[str, set[int]] = {}

    for anchor in soup.find_all("a", href=True):
        href = urljoin(current_url, anchor["href"])
        if urlparse(href).netloc != current_domain:
            continue
        match = _NUMBERED_RE.search(href)
        if not match:
            continue
        number = int(match.group(2))
        template = href[: match.start(2)] + "{n}" + href[match.end(2):]
        candidates.setdefault(template, set()).add(number)

    clusters = {template: nums for template, nums in candidates.items() if len(nums) >= 2}
    if not clusters:
        return None
    return max(clusters, key=lambda template: len(clusters[template]))


def render_template(template: str, page_number: int) -> str:
    return template.replace("{n}", str(page_number))


def detect_pagination(html: str, current_url: str) -> PaginationPlan:
    next_url = detect_next_url(html, current_url)
    if next_url:
        return PaginationPlan(mode="chain", next_url=next_url)

    template = detect_numbered_template(html, current_url)
    if template:
        current_match = _NUMBERED_RE.search(current_url)
        start_page_number = int(current_match.group(2)) + 1 if current_match else 2
        return PaginationPlan(mode="template", template=template, start_page_number=start_page_number)

    return PaginationPlan(mode="single")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ingestion/test_pagination.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add app/ingestion/pagination.py tests/ingestion/test_pagination.py
git commit -m "feat: add pagination detection (rel=next, anchor-next, numbered cluster)"
```

---

### Task 4: Fetcher

**Files:**
- Create: `app/ingestion/fetcher.py`
- Test: `tests/ingestion/test_fetcher.py`

**Interfaces:**
- Consumes: `Settings` from `app.config` (fields `fetch_timeout_seconds`, `user_agent`).
- Produces: `FetchError(Exception)` with `.reason: str`; `async def fetch_page(client: httpx.AsyncClient, url: str, settings: Settings) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/ingestion/test_fetcher.py
import httpx
import pytest

from app.config import Settings
from app.ingestion.fetcher import FetchError, fetch_page

SETTINGS = Settings()


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_fetch_page_returns_body_on_200():
    def handler(request):
        assert request.headers["user-agent"] == SETTINGS.user_agent
        return httpx.Response(200, text="<html>ok</html>")

    async with _client(handler) as client:
        body = await fetch_page(client, "https://example.com", SETTINGS)
    assert body == "<html>ok</html>"


async def test_fetch_page_raises_on_non_200():
    def handler(request):
        return httpx.Response(404, text="not found")

    async with _client(handler) as client:
        with pytest.raises(FetchError, match="404"):
            await fetch_page(client, "https://example.com", SETTINGS)


async def test_fetch_page_raises_on_timeout():
    def handler(request):
        raise httpx.TimeoutException("timed out")

    async with _client(handler) as client:
        with pytest.raises(FetchError, match="timeout"):
            await fetch_page(client, "https://example.com", SETTINGS)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ingestion/test_fetcher.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.ingestion.fetcher'`

- [ ] **Step 3: Implement fetcher**

```python
# app/ingestion/fetcher.py
from __future__ import annotations

import httpx

from app.config import Settings


class FetchError(Exception):
    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


async def fetch_page(client: httpx.AsyncClient, url: str, settings: Settings) -> str:
    try:
        response = await client.get(
            url,
            timeout=settings.fetch_timeout_seconds,
            headers={"User-Agent": settings.user_agent},
            follow_redirects=True,
        )
    except httpx.TimeoutException as exc:
        raise FetchError(f"timeout fetching {url}") from exc
    except httpx.RequestError as exc:
        raise FetchError(f"request failed for {url}: {exc}") from exc

    if response.status_code != 200:
        raise FetchError(f"non-200 status {response.status_code} for {url}")
    return response.text
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ingestion/test_fetcher.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add app/ingestion/fetcher.py tests/ingestion/test_fetcher.py
git commit -m "feat: add async HTTP fetcher with timeout/status error handling"
```

---

### Task 5: Extractor

**Files:**
- Create: `app/ingestion/extractor.py`
- Test: `tests/ingestion/test_extractor.py`

**Interfaces:**
- Consumes: `Settings` from `app.config` (field `min_extract_length`).
- Produces: `ExtractionError(Exception)`; `def extract_main_text(html: str, settings: Settings) -> str`.

- [ ] **Step 1: Add readability-lxml dependency**

```bash
uv add readability-lxml
```

- [ ] **Step 2: Write the failing test**

```python
# tests/ingestion/test_extractor.py
import pytest

from app.config import Settings
from app.ingestion.extractor import ExtractionError, extract_main_text

SETTINGS = Settings(min_extract_length=20)

ARTICLE_HTML = """
<html><body>
<nav>Home | About</nav>
<article>
<p>   This   is a real   article  with enough content to pass the
threshold   easily.   </p>

<p>Second paragraph here.</p>
</article>
<footer>copyright 2026</footer>
</body></html>
"""

EMPTY_HTML = "<html><body><nav>Home</nav><footer>copyright</footer></body></html>"


def test_extract_main_text_strips_boilerplate_and_collapses_whitespace():
    text = extract_main_text(ARTICLE_HTML, SETTINGS)
    assert "Home" not in text
    assert "copyright" not in text
    assert "This is a real article with enough content to pass the threshold easily." in text
    assert "Second paragraph here." in text


def test_extract_main_text_raises_on_near_empty_content():
    with pytest.raises(ExtractionError, match="no extractable content"):
        extract_main_text(EMPTY_HTML, SETTINGS)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/ingestion/test_extractor.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.ingestion.extractor'`

- [ ] **Step 4: Implement extractor**

```python
# app/ingestion/extractor.py
from __future__ import annotations

import re

import trafilatura
from bs4 import BeautifulSoup
from readability import Document

from app.config import Settings


class ExtractionError(Exception):
    pass


def _normalize(text: str) -> str:
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in text.split("\n\n")]
    paragraphs = [p for p in paragraphs if p]
    return "\n\n".join(paragraphs)


def _extract_with_trafilatura(html: str) -> str | None:
    return trafilatura.extract(html, include_comments=False, include_tables=False)


def _extract_with_readability(html: str) -> str | None:
    try:
        summary_html = Document(html).summary()
    except Exception:
        return None
    return BeautifulSoup(summary_html, "lxml").get_text("\n\n")


def extract_main_text(html: str, settings: Settings) -> str:
    text = _extract_with_trafilatura(html)
    if not text:
        text = _extract_with_readability(html)

    normalized = _normalize(text or "")
    if len(normalized) < settings.min_extract_length:
        raise ExtractionError("no extractable content")
    return normalized
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/ingestion/test_extractor.py -v`
Expected: PASS (2 tests)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock app/ingestion/extractor.py tests/ingestion/test_extractor.py
git commit -m "feat: add main-content extraction (trafilatura + readability-lxml fallback)"
```

---

### Task 6: Job store

**Files:**
- Create: `app/ingestion/job_store.py`
- Test: `tests/ingestion/test_job_store.py`

**Interfaces:**
- Consumes: `UrlStatus`, `JobStatusResponse`, `Stage` from `app.ingestion.models`.
- Produces: `class JobStore` with `create_job(urls: list[str]) -> str`, `get_status(job_id: str) -> JobStatusResponse`, `update(job_id: str, url: str, **fields) -> None`, `exists(job_id: str) -> bool`.

- [ ] **Step 1: Write the failing test**

```python
# tests/ingestion/test_job_store.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ingestion/test_job_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.ingestion.job_store'`

- [ ] **Step 3: Implement job store**

```python
# app/ingestion/job_store.py
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ingestion/test_job_store.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add app/ingestion/job_store.py tests/ingestion/test_job_store.py
git commit -m "feat: add in-memory job store"
```

---

### Task 7: Worker orchestrator

**Files:**
- Create: `app/ingestion/worker.py`
- Test: `tests/ingestion/test_worker.py`

**Interfaces:**
- Consumes:
  - `fetch_page(client, url, settings) -> str`, `FetchError` from `app.ingestion.fetcher`
  - `extract_main_text(html, settings) -> str`, `ExtractionError` from `app.ingestion.extractor`
  - `detect_pagination(html, current_url) -> PaginationPlan`, `detect_next_url(html, current_url) -> str | None`, `render_template(template, page_number) -> str` from `app.ingestion.pagination`
  - `JobStore` from `app.ingestion.job_store`
  - `IngestionPayload`, `PageMapEntry`, `Stage` from `app.ingestion.models`
  - `Settings`, `settings` from `app.config`
- Produces: `IngestSink = Callable[[IngestionPayload], Awaitable[None]]`; `async def ingest_url(job_id: str, url: str, store: JobStore, client: httpx.AsyncClient, sink: IngestSink = _noop_sink, settings: Settings = settings) -> None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/ingestion/test_worker.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ingestion/test_worker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.ingestion.worker'`

- [ ] **Step 3: Implement worker**

```python
# app/ingestion/worker.py
from __future__ import annotations

from datetime import datetime, timezone
from typing import Awaitable, Callable

import httpx

from app.config import Settings, settings as default_settings
from app.ingestion.extractor import ExtractionError, extract_main_text
from app.ingestion.fetcher import FetchError, fetch_page
from app.ingestion.job_store import JobStore
from app.ingestion.models import IngestionPayload, PageMapEntry, Stage
from app.ingestion.pagination import detect_next_url, detect_pagination, render_template

IngestSink = Callable[[IngestionPayload], Awaitable[None]]


async def _noop_sink(payload: IngestionPayload) -> None:
    return None


async def ingest_url(
    job_id: str,
    url: str,
    store: JobStore,
    client: httpx.AsyncClient,
    sink: IngestSink = _noop_sink,
    settings: Settings = default_settings,
) -> None:
    try:
        store.update(job_id, url, stage=Stage.FETCHING)
        first_html = await fetch_page(client, url, settings)

        plan = detect_pagination(first_html, url)
        html_pages = [first_html]
        store.update(job_id, url, pages_fetched=1, pages_total=1 if plan.mode == "single" else None)

        if plan.mode == "chain":
            current_url = url
            next_url = plan.next_url
            while next_url and next_url != current_url and len(html_pages) < settings.max_pages:
                store.update(job_id, url, stage=Stage.PAGINATING)
                next_html = await fetch_page(client, next_url, settings)
                html_pages.append(next_html)
                store.update(job_id, url, pages_fetched=len(html_pages))
                current_url = next_url
                next_url = detect_next_url(next_html, current_url)

        elif plan.mode == "template":
            page_number = plan.start_page_number
            seen_urls = {url}
            while len(html_pages) < settings.max_pages:
                candidate_url = render_template(plan.template, page_number)
                if candidate_url in seen_urls:
                    break
                store.update(job_id, url, stage=Stage.PAGINATING)
                try:
                    next_html = await fetch_page(client, candidate_url, settings)
                except FetchError:
                    break
                html_pages.append(next_html)
                seen_urls.add(candidate_url)
                store.update(job_id, url, pages_fetched=len(html_pages))
                page_number += 1

        store.update(job_id, url, stage=Stage.CLEANING, pages_total=len(html_pages))

        page_map: list[PageMapEntry] = []
        cleaned_segments: list[str] = []
        cursor = 0
        for page_index, page_html in enumerate(html_pages, start=1):
            segment = extract_main_text(page_html, settings)
            start = cursor
            end = start + len(segment)
            page_map.append(PageMapEntry(page=page_index, char_start=start, char_end=end))
            cleaned_segments.append(segment)
            cursor = end + 2  # accounts for the "\n\n" join separator below

        payload = IngestionPayload(
            source_url=url,
            cleaned_text="\n\n".join(cleaned_segments),
            pages_fetched=len(html_pages),
            fetched_at=datetime.now(timezone.utc).isoformat(),
            page_map=page_map,
        )

        store.update(job_id, url, stage=Stage.INDEXING)
        await sink(payload)
        store.update(job_id, url, stage=Stage.DONE)

    except FetchError as exc:
        store.update(job_id, url, stage=Stage.ERROR, error=exc.reason)
    except ExtractionError as exc:
        store.update(job_id, url, stage=Stage.ERROR, error=str(exc))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ingestion/test_worker.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add app/ingestion/worker.py tests/ingestion/test_worker.py
git commit -m "feat: add ingestion worker orchestrator with failure isolation"
```

---

### Task 8: Router

**Files:**
- Create: `app/ingestion/router.py`
- Test: `tests/ingestion/test_router.py`

**Interfaces:**
- Consumes: `JobStore` from `app.ingestion.job_store`; `ingest_url` from `app.ingestion.worker`; `IngestRequest`, `IngestResponse`, `JobStatusResponse` from `app.ingestion.models`.
- Produces: `def build_ingestion_router(store: JobStore, client_factory: Callable[[], httpx.AsyncClient] = httpx.AsyncClient) -> APIRouter` exposing `POST /ingest` and `GET /ingest/{job_id}/status`.

- [ ] **Step 1: Write the failing test**

```python
# tests/ingestion/test_router.py
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.ingestion.job_store import JobStore
from app.ingestion.router import build_ingestion_router


def _app_with_store() -> tuple[FastAPI, JobStore]:
    store = JobStore()
    app = FastAPI()
    app.include_router(build_ingestion_router(store))
    return app, store


def test_post_ingest_returns_job_id_and_creates_queued_status():
    app, store = _app_with_store()
    with patch("app.ingestion.router.ingest_url", new=AsyncMock()):
        with TestClient(app) as client:
            response = client.post("/ingest", json={"urls": ["https://example.com"]})
    assert response.status_code == 200
    job_id = response.json()["job_id"]
    assert store.exists(job_id)


def test_get_status_returns_404_for_unknown_job():
    app, _store = _app_with_store()
    with TestClient(app) as client:
        response = client.get("/ingest/does-not-exist/status")
    assert response.status_code == 404


def test_get_status_returns_contract_shape():
    app, store = _app_with_store()
    job_id = store.create_job(["https://example.com"])
    with TestClient(app) as client:
        response = client.get(f"/ingest/{job_id}/status")
    body = response.json()
    assert body["job_id"] == job_id
    assert body["urls"][0]["url"] == "https://example.com"
    assert body["urls"][0]["stage"] == "queued"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ingestion/test_router.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.ingestion.router'`

- [ ] **Step 3: Implement router**

```python
# app/ingestion/router.py
from __future__ import annotations

import asyncio
from typing import Callable

import httpx
from fastapi import APIRouter, HTTPException

from app.ingestion.job_store import JobStore
from app.ingestion.models import IngestRequest, IngestResponse, JobStatusResponse
from app.ingestion.worker import ingest_url


def build_ingestion_router(
    store: JobStore,
    client_factory: Callable[[], httpx.AsyncClient] = httpx.AsyncClient,
) -> APIRouter:
    router = APIRouter()

    @router.post("/ingest", response_model=IngestResponse)
    async def create_ingest_job(request: IngestRequest) -> IngestResponse:
        job_id = store.create_job(request.urls)
        client = client_factory()

        async def run_all() -> None:
            try:
                await asyncio.gather(*(ingest_url(job_id, url, store, client) for url in request.urls))
            finally:
                await client.aclose()

        asyncio.create_task(run_all())
        return IngestResponse(job_id=job_id)

    @router.get("/ingest/{job_id}/status", response_model=JobStatusResponse)
    async def get_ingest_status(job_id: str) -> JobStatusResponse:
        if not store.exists(job_id):
            raise HTTPException(status_code=404, detail="job not found")
        return store.get_status(job_id)

    return router
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ingestion/test_router.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add app/ingestion/router.py tests/ingestion/test_router.py
git commit -m "feat: add /ingest and /ingest/{job_id}/status endpoints"
```

---

### Task 9: FastAPI app wiring

**Files:**
- Create: `app/main.py`
- Test: `tests/ingestion/test_main.py`

**Interfaces:**
- Consumes: `JobStore` from `app.ingestion.job_store`; `build_ingestion_router` from `app.ingestion.router`.
- Produces: `def create_app() -> FastAPI`; module-level `app: FastAPI` for `uvicorn app.main:app`.

- [ ] **Step 1: Write the failing test**

```python
# tests/ingestion/test_main.py
from fastapi.testclient import TestClient

from app.main import create_app


def test_ingest_routes_are_registered():
    app = create_app()
    paths = {route.path for route in app.routes}
    assert "/ingest" in paths
    assert "/ingest/{job_id}/status" in paths


def test_full_ingest_and_status_round_trip_with_fresh_app():
    app = create_app()
    with TestClient(app) as client:
        response = client.post("/ingest", json={"urls": ["https://example.com"]})
        job_id = response.json()["job_id"]
        status = client.get(f"/ingest/{job_id}/status")
    assert status.status_code == 200
    assert status.json()["job_id"] == job_id
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ingestion/test_main.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.main'`

- [ ] **Step 3: Implement app factory**

```python
# app/main.py
from __future__ import annotations

from fastapi import FastAPI

from app.ingestion.job_store import JobStore
from app.ingestion.router import build_ingestion_router


def create_app() -> FastAPI:
    app = FastAPI(title="RAG Ingestion Service")
    store = JobStore()
    app.include_router(build_ingestion_router(store))
    return app


app = create_app()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/ingestion/test_main.py -v`
Expected: PASS (2 tests). Note: the second test's real worker will run against the live network in the background (fire-and-forget `asyncio.create_task`) since no mock transport is wired here — that's expected; the test only asserts the route contract, not the fetch outcome.

- [ ] **Step 5: Commit**

```bash
git add app/main.py tests/ingestion/test_main.py
git commit -m "feat: wire FastAPI app factory for ingestion service"
```

---

### Task 10: Full pipeline integration test

**Files:**
- Test: `tests/ingestion/test_integration.py`

**Interfaces:**
- Consumes: `create_app` is NOT reused here because it hardcodes a real `httpx.AsyncClient` — this task builds the app directly from `build_ingestion_router` with an injected `client_factory` so no real network call happens.

- [ ] **Step 1: Write the integration test**

```python
# tests/ingestion/test_integration.py
import asyncio

import httpx
from fastapi import FastAPI

from app.ingestion.job_store import JobStore
from app.ingestion.models import Stage
from app.ingestion.router import build_ingestion_router

GOOD_PAGE_1 = """
<html><head><link rel="next" href="/good?page=2"></head>
<body><article><p>Good article page one, plenty of real content here to pass the threshold.</p></article></body></html>
"""
GOOD_PAGE_2 = """
<html><body><article><p>Good article page two, also plenty of real content to pass.</p></article></body></html>
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/ingestion/test_integration.py -v`
Expected: at this point all prior tasks are implemented, so this should already largely work — run it first to confirm; if it fails, the failure will point at a specific assertion (e.g. wrong stage), not an import error. Fix the pointed-at module before proceeding.

- [ ] **Step 3: Run test to verify it passes**

Run: `uv run pytest tests/ingestion/test_integration.py -v`
Expected: PASS (1 test)

- [ ] **Step 4: Run the full ingestion test suite**

Run: `uv run pytest tests/ingestion/ -v`
Expected: all tests across Tasks 1–10 PASS

- [ ] **Step 5: Commit**

```bash
git add tests/ingestion/test_integration.py
git commit -m "test: add end-to-end ingestion pipeline integration test"
```

---

## Self-Review Notes

- **Spec coverage:** UI (Component 1) is out of scope for this plan (Plan 4 — Frontend). API layer (Component 2), worker (Component 3, Steps A–E), pagination priority order, failure isolation, and progress stage enum are all covered (Tasks 3, 4, 5, 7, 8). Emit-to-indexer (Step E) is covered via the injectable `IngestSink` in Task 7 — Plan 2 will pass a real sink instead of the no-op default.
- **Type consistency checked:** `Stage`, `UrlStatus`, `JobStatusResponse`, `IngestionPayload`, `PageMapEntry` are defined once in Task 2 and referenced identically (same field names/types) in Tasks 6–10. `IngestSink` defined in Task 7, consumed unchanged by Task 8's `build_ingestion_router` (default sink, not overridden at the router level — sink wiring for the real indexer happens in Plan 2, not here).
- **No placeholders:** every step has runnable code; no "TBD" left.
