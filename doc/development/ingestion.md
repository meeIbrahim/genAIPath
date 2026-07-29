# RAG ingestion subsystem — architecture spec

## Scope
Ingestion is the stage before indexing: user submits URLs → system fetches, paginates (bounded), cleans, and emits normalized text + metadata to the existing chunk/index pipeline. Out of scope: chunking, embedding, BM25, retrieval (see prior architecture).

## Non-goals (explicit constraints)
- No recursive crawling. Never follow nav links, sidebar links, or arbitrary `<a>` tags on the page.
- The only permitted "follow" behavior is a detected pagination chain for the same document/listing.
- No JS rendering / headless browser by default. Static HTML fetch only. (Flag as a future extension if the target sites are JS-rendered SPAs — out of scope here.)

## High-level flow
```
UI (URL list) → POST /ingest → Job created (job_id, per-URL sub-status)
                                      │
                            Ingestion worker (async, per URL)
                                      │
              fetch → detect pagination → fetch next pages (bounded) → clean/extract → merge
                                      │
                       emit progress events per stage ──────► UI polls/subscribes
                                      │
                          normalized {text, metadata} → indexer (chunk + BM25 + embed)
```

## Components

### 1. UI — URL list + progress
- Simple input to add N URLs to a pending list (client-side array, no dedup logic needed beyond basic exact-match).
- On submit: `POST /ingest {urls: [...]}` → returns `job_id`.
- Progress surface: poll `GET /ingest/{job_id}/status` every 1–2s.
- Render one progress row per URL showing current `stage` + `pages_fetched/pages_total` when known.

### 2. API layer
`POST /ingest`
```json
{ "urls": ["https://example.com/blog/post-1", "..."] }
→ { "job_id": "uuid" }
```
`GET /ingest/{job_id}/status`
```json
{
  "job_id": "uuid",
  "urls": [
    {
      "url": "https://example.com/blog/post-1",
      "stage": "cleaning",          // queued | fetching | paginating | cleaning | indexing | done | error
      "pages_fetched": 2,
      "pages_total": null,          // unknown until pagination probe completes; null = single-page or undetermined
      "error": null
    }
  ]
}
```
Stage enum is the contract the progress bar renders against — keep it stable.

### 3. Ingestion worker (per URL, async task)
Runs independently per URL so one failure doesn't block others. Task orchestration: `asyncio.Task` per URL — no persistence/retries across restarts, acceptable at showcase scale.

**Step A — Fetch**
- HTTP GET with timeout (e.g. 10s), realistic User-Agent, follow redirects (standard HTTP redirects only, not link-following).
- On non-200 / timeout → stage = `error`, capture reason, stop this URL, continue others.

**Step B — Pagination detection** (bounded, deterministic — no heuristístic link discovery)
Parse with BeautifulSoup (BS4). Check in priority order, stop at first match:
1. `<link rel="next" href="...">` in `<head>` — most reliable signal.
2. `<a rel="next">` or `<a>` with visible text matching `/^(next|›|»|more)$/i` **and** href matching the same path pattern as the current URL (same domain + same path prefix, differing only in a page/offset parameter). This same-pattern check is what prevents this from becoming general link-following.
3. Numbered pagination cluster (e.g. `?page=2`, `/page/2/`) — detect by finding multiple sibling links sharing a common URL template with an incrementing integer; take the template, do not follow individual numbered links found elsewhere on the page.
4. No match → single page, `pages_total = 1`.

Cap: hard max page count (config, default 20) to bound worst case regardless of what the site reports.

**Step C — Fetch subsequent pages**
- Follow only the URL(s) produced by the pattern in Step B (increment the template), not by re-scanning each new page for more links. Stop when: max pages hit, a fetch fails, or the "next" pattern stops resolving (404 / repeats a prior URL / no more increments).

**Step D — Clean / extract**
Mandatory main-content extraction, not a generic tag-strip:
- Use a readability-style extractor (e.g. `trafilatura` or `readability-lxml`) to isolate main article/body content from a full HTML page.
- Explicitly discard: `<script>`, `<style>`, `<nav>`, `<header>`, `<footer>`, ad/cookie-banner containers, comment sections (best-effort via extractor's boilerplate removal).
- Normalize: collapse whitespace, decode HTML entities, strip empty lines, preserve paragraph breaks (needed for later chunking).
- If extraction yields near-empty text (below a min-length threshold) → mark `error: "no extractable content"` rather than silently indexing junk.

**Step E — Merge & emit**
- Concatenate cleaned text across paginated pages in page order, tagging each segment's origin page number for metadata.
- Emit final payload to the indexer with metadata:
```json
{
  "source_url": "https://example.com/blog/post-1",
  "pages_fetched": 3,
  "fetched_at": "2026-07-29T12:00:00Z",
  "page_map": [ {"page": 1, "char_start": 0, "char_end": 4210}, ... ]
}
```
This `page_map` lets downstream chunk metadata (from the earlier indexing spec) resolve `page_number` per chunk even after pages are concatenated.

## Progress bar semantics (for handoff clarity)
| Stage | Meaning | UI display |
|---|---|---|
| `queued` | accepted, not started | "Waiting…" |
| `fetching` | HTTP GET in flight | "Fetching page 1" |
| `paginating` | fetching page 2..N | "Fetching page {n} of {pages_total or '?'}" |
| `cleaning` | extraction/normalization running | "Cleaning content" |
| `indexing` | handed to chunk/BM25/embed pipeline | "Indexing" |
| `done` | fully indexed | checkmark |
| `error` | terminal failure, reason attached | error text, no retry auto-triggered |

## Failure isolation
Each URL is an independent unit of work end-to-end (fetch → clean → index). A failure at any step sets that URL's status to `error` with a human-readable reason and does not affect sibling URLs in the same job.

## Libraries
- Fetch: `httpx` (async-native)
- Pagination detection/parsing: `beautifulsoup4` (BS4)
- Extraction: `trafilatura` (handles boilerplate removal + is good at pagination-adjacent article sites), `readability-lxml` as fallback
- Task orchestration: `asyncio.Task` per URL
- Progress transport: polling endpoint
