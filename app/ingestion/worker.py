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
    except Exception as exc:
        # Any other unexpected failure must stay isolated to this URL so it
        # can never propagate out of ingest_url and affect sibling URLs
        # processed concurrently in the same job (see Global Constraints:
        # failure isolation).
        store.update(job_id, url, stage=Stage.ERROR, error=str(exc))
