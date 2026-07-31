# Design: Preference Extraction, City/Price Filtering, and Context Quality Judge

**Date:** 2026-07-31
**Status:** Approved (pending spec review)

## Purpose

Add three capabilities to the existing RAG query pipeline (`POST /query`):

1. **Preference extraction** — pull `city`, `budget`, and `interests` out of the user's query text using non-LLM NER/regex/keyword techniques.
2. **Metadata filtering** — filter retrieved chunks by `city` and `price`, using new per-chunk metadata tagged at index time.
3. **Context quality judge** — a small, fast LLM call that classifies retrieved context as sufficient or insufficient before synthesis runs; retries retrieval once (widened depth) on insufficiency, then falls back to a fixed "can't answer" response rather than risking a hallucinated synthesis.

## Current State (baseline, verified against code)

- Pipeline: `Ingestion → Indexing → Retrieval → Synthesis → UI`. `POST /query {query, top_k}` → `retrieve()` (BM25 + semantic, RRF-fused) → `synthesize_answer()` (Groq, `openai/gpt-oss-120b`, raw `httpx`, **not** LangChain despite it being an installed-but-unused dependency) → `QueryResponse{query, answer, citations, retrieved_chunks}`.
- `ChunkMetadata` (`app/indexing/models.py`): `chunk_id, doc_id, source_url, page_number, chunk_index, char_start, char_end, overlap_with_prev, indexed_at, text`. **No city/price/location field exists anywhere in the codebase** (confirmed via full-tree grep).
- `FusedChunk` (`app/retrieval/models.py`): scoring/provenance only, no metadata beyond `source_url`/`page_number`.
- No preference-extraction, filtering, or judge/quality-check logic exists anywhere in the query path today. This is greenfield.
- Test convention: `tests/<package>/test_<module>.py` mirroring `app/`, no `conftest.py`, `httpx.MockTransport` for external calls, `asyncio_mode=auto`, `tmp_path`-isolated Qdrant per test.
- Router pattern: `build_*_router(...deps, settings=default_settings)` factory functions, dependency injection via constructor params, matched by all three existing routers (ingestion/indexing/retrieval).

## Architecture

```
Indexing (existing, extended):
  chunk_text() → ChunkMetadata (NEW: + city, price via app/extraction/) → embed → BM25 + Qdrant + chunk_store

Query flow (existing, extended):
  POST /query {query, top_k}
    1. NEW: extract_preferences(query) → QueryPreferences{city, budget, interests}
    2. retrieve() [UNCHANGED] → FusedChunk list
    3. NEW: filter_chunks(fused_chunks, preferences) → (kept_chunks, filtered_out_count)
    4. NEW: judge_context(query, kept_chunks) → JudgeVerdict
       - context_good → synthesize_answer(kept_chunks) [UNCHANGED] → normal QueryResponse
       - context_insufficient →
         a. retry once: retrieve() with retrieval_top_k/display_top_k doubled → filter_chunks() → judge_context() again
         b. still insufficient → skip synthesis; return fallback QueryResponse (200 OK, fixed answer text, citations=[], retrieved_chunks = retry-attempt's kept_chunks, all used_in_synthesis=false)
    5. QueryResponse [EXTENDED]: + preferences, + filtered_out_count

Frontend (Query screen, extended):
  - preference badges row (rendered only for detected fields)
  - "N chunks excluded by filter" note near the chunk panel (rendered only when filtered_out_count > 0)
```

## Components

### 1. `app/extraction/` (new package) — shared, no LLM calls

- **`location.py`**: `extract_city(text: str) -> str | None`. Uses spaCy (`en_core_web_sm`), returns the first `GPE` entity found, lowercased. No alias/canonicalization (see Non-Goals).
- **`price.py`**: `extract_price(text: str) -> float | None`. Regex over currency-shaped tokens (`\$?\d[\d,]*\.?\d*\s*[kK]?`) plus qualifier words ("under", "budget of", "around", "less than") to bias toward a ceiling interpretation. Normalizes `"20k"` / `"20,000"` → `20000.0`. First match wins; no multi-price-per-text handling.
- **`interests.py`**: `extract_interests(text: str) -> list[str]`. Static keyword→category dict (e.g. `{"hiking": "outdoors", "trek": "outdoors", "food": "food", "restaurant": "food", "cuisine": "food"}`). Returns the deduplicated list of matched categories, in first-match order.
- **`preferences.py`**: `QueryPreferences(BaseModel)` — `city: str | None`, `budget: float | None`, `interests: list[str]`. `extract_preferences(query: str) -> QueryPreferences` composes the three extractors above. Pure function, no I/O, no LLM call — this fulfills the "non-LLM preference extraction" requirement.

**New dependency:** `spacy` + `en_core_web_sm` model artifact (one-time download, ~50MB). Fully offline at request time.

### 2. Chunk metadata tagging (`app/indexing/`)

- `ChunkMetadata` gains: `city: str | None = None`, `price: float | None = None`.
- `indexer.py`: after chunking, before embedding, call `extract_city(chunk.text)` / `extract_price(chunk.text)` per chunk and populate the new fields. Same extraction functions as query-side (DRY — one location/price extractor, two call sites).
- No Qdrant schema migration needed — payload is already a generic dict of `ChunkMetadata.model_dump()`.
- **Existing indexed chunks are not backfilled.** Given BM25/`chunk_store` are in-memory (already lost on process restart today, independent of this change), re-ingestion is the existing recovery path and covers backfill too. No new migration tooling in scope.

### 3. Filtering (`app/retrieval/filtering.py`, new)

- `filter_chunks(fused_chunks: list[FusedChunk], preferences: QueryPreferences) -> tuple[list[FusedChunk], int]`.
- A chunk is excluded **only** if it has a non-null `city`/`price` that **conflicts** with a stated preference:
  - `city` conflict: `chunk.city is not None and preferences.city is not None and chunk.city != preferences.city` (case-insensitive, exact string match — no alias resolution).
  - `price` conflict: `chunk.price is not None and preferences.budget is not None and chunk.price > preferences.budget`.
- Chunks with `city=None` or `price=None` always pass through (permissive-on-missing, per design discussion — avoids over-filtering a corpus where most chunks won't have extractable metadata).
- Returns `(kept_chunks, len(fused_chunks) - len(kept_chunks))`.
- Called in `router.py`, after `retrieve()`, before the judge step.

### 4. Context Quality Judge (`app/retrieval/judge.py`, new)

- Mirrors `synthesis.py`'s raw-httpx-to-Groq pattern exactly — no LangChain (matches the existing convention; LangChain remains an unused dependency, unchanged by this PRD).
- New config: `settings.judge_model` (default: `"llama-3.1-8b-instant"` — smaller/faster than the synthesis model, since classification is a simpler task). Reuses existing `groq_base_url`/`groq_api_key`.
- `JudgeVerdict(BaseModel)`: `verdict: Literal["context_good", "context_insufficient"]`.
- `judge_context(query: str, chunks: list[FusedChunk], http_client, settings) -> JudgeVerdict`. One Groq chat-completion call. Prompt: given the query and the numbered chunk excerpts, classify whether they can support a grounded answer; respond with exactly `context_good` or `context_insufficient`.
- Error handling: mirrors `SynthesisError` conventions — a new `JudgeError` for network/parse/non-200 failures. **Fail closed**: any `JudgeError` is treated as `context_insufficient` (never silently proceed to synthesis on a broken judge call).
- Empty `chunks` list (e.g. everything got filtered out) short-circuits to `context_insufficient` without a Groq call.

### 5. Retry orchestration (`app/retrieval/router.py`, modified)

- New config: `judge_retry_top_k_multiplier` (default `2`).
- On first `context_insufficient`: re-run `retrieve()` with a per-request `dataclasses.replace(settings, retrieval_top_k=..., display_top_k=...)` (each multiplied by `judge_retry_top_k_multiplier`), passed as `retrieve()`'s existing `settings` param. The module-level `settings` singleton is never mutated — the replaced copy lives only for the retry call.
- Re-run `filter_chunks()` and `judge_context()` on the retry's results.
- Still `context_insufficient` after retry: skip `synthesize_answer()` entirely. Build `QueryResponse` with:
  - `answer = "I don't have enough reliable information in the indexed content to answer this question confidently."`
  - `citations = []`
  - `retrieved_chunks` = the retry attempt's filtered chunks, each with `used_in_synthesis = false`
  - `preferences` and `filtered_out_count` still populated from the retry attempt
- Exactly one retry, no configurable retry count in v1.

### 6. API contract changes (`app/retrieval/models.py`)

`QueryResponse` gains:
- `preferences: QueryPreferences` (always present; fields individually nullable/empty when nothing detected)
- `filtered_out_count: int` (count of chunks excluded by the filtering step on whichever attempt — initial or retry — produced the final result)

### 7. Frontend (`app/static/js/query.js`, `app/static/query.html`)

- New preference-badges row rendered above the answer block, populated from `result.preferences`. Only renders a badge for each non-null/non-empty field (e.g. skip the budget badge if `budget` is null). Follows the existing `.badge` CSS class pattern already used for chunk scoring badges.
- New small note near the chunk panel: `"{filtered_out_count} chunk(s) excluded by filter"`, rendered only when `filtered_out_count > 0`.
- No new screens, no layout overhaul — additive to the existing Query screen.

## Testing Plan

Follows existing repo convention (mirrored test dirs, no conftest, `httpx.MockTransport`, `asyncio_mode=auto`):

- `tests/extraction/test_location.py`, `test_price.py`, `test_interests.py`, `test_preferences.py` — pure unit tests, table-driven (input text → expected extracted value), including negative cases (no city/price/interest present → `None`/`[]`).
- `tests/indexing/test_indexer.py` (extended) — assert `ChunkMetadata.city`/`.price` populated correctly for chunks whose text contains detectable values, and `None` when not.
- `tests/retrieval/test_filtering.py` (new) — table-driven: matching city/price passes, conflicting city/price excluded, missing metadata always passes, empty preferences passes everything.
- `tests/retrieval/test_judge.py` (new) — `httpx.MockTransport`-based, mirrors `test_synthesis.py` structure: verdict parsing, `JudgeError` on network/non-200/malformed response, empty-chunks short-circuit.
- `tests/retrieval/test_router.py` (extended) — integration-level: full retry flow (first judge insufficient → retry → still insufficient → fallback response), and the happy path (judge good on first attempt → synthesis runs, response unchanged in shape from today plus the two new fields).
- `tests/frontend/test_query_page.py` (extended if needed) — page-serve assertions only, per existing convention (no JS test harness in this repo — same documented exception as the frontend plan).

## Non-Goals (v1)

- No city alias/canonicization ("NYC" and "New York" are treated as different values).
- No use of `interests` for retrieval boosting, re-ranking, or query rewriting — captured and returned only.
- No backfill/migration tooling for chunks indexed before this change — re-ingestion is the existing (pre-existing-limitation) recovery path.
- No configurable retry count — exactly one retry, hardcoded multiplier.
- No judge-verdict caching — every query re-judges, even exact repeats.
- No new visual design system for the frontend badges — reuses the existing `.badge`/row patterns.

## Open Risks / Follow-ups

- Judge-model false negatives (`context_insufficient` on genuinely good context) double retrieval latency and can still end in a needless fallback; if this proves noisy in practice, a follow-up could tune the judge prompt or add a confidence threshold.
- Exact-string city matching means query "lahore" won't match a chunk tagged "Lahore, Pakistan" if spaCy's GPE span differs in scope — worth watching in practice; alias/normalization is explicitly deferred, not accidentally missed.
- Regex-based price extraction on chunk text will misfire on non-price numbers (page counts, years, phone numbers) in generic web content — acceptable given permissive-on-missing filtering, but a source of noisy `price` tags worth monitoring post-launch.
