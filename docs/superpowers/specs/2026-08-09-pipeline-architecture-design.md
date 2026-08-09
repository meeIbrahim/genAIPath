# Design: Pipeline Architecture (Strategy Registry + Local-Archive Indexing)

**Date:** 2026-08-09
**Status:** Approved (pending spec review)

## Purpose

Transform the app from one hardcoded RAG pipeline into a showcase where each stage — Indexing, Retrieval, Post-retrieval — is swappable between predefined strategies, selected via a `POST /pipeline/load` call. This is piece A of a four-piece initiative:

- **A (this spec):** pipeline architecture — strategy registries, config identity, per-strategy storage isolation, Load/Query wiring.
- **B (future spec):** concrete strategy implementations (chunking methods, retrieval methods, post-retrieval methods).
- **C (future spec):** gold-standard eval harness (fixed test-case set, hit rate / MRR / citation regex / LLM-judge-or-cosine metrics) that runs automatically on every Load.
- **D (future spec):** single combined UI screen (strategy pickers + Load + live eval sidebar + query box).

This spec only builds the scaffolding B/C/D plug into — not the strategies' algorithms, not the eval metrics, not the UI.

## Current State (baseline, verified against code)

- Corpus enters via URL crawling: `app/ingestion/{fetcher,pagination,job_store,worker,router}.py`, driven by an Ingest screen where users type URLs. `POST /ingest` kicks off a background worker that fetches, paginates, extracts, and calls an `index_sink` per completed doc.
- Indexing is single-strategy, hardcoded: `app/indexing/indexer.py::index_document()` always does fixed-window token chunking (`chunk_text()`, `chunk_size_tokens=400`/`chunk_overlap_tokens=75` from `Settings`) → embed via Ollama → dual-write to one shared `InMemoryBM25Index` instance and one shared `QdrantVectorIndex` instance (collection name fixed: `settings.qdrant_collection = "rag_chunks"`, local path storage at `.data/qdrant`). One collection, one BM25 index, no concept of alternate strategies.
- Retrieval is single-strategy, hardcoded: `app/retrieval/retriever.py::retrieve()` always fires BM25 + semantic search in parallel and fuses with RRF (`app/retrieval/fusion.py`).
- Post-retrieval is a fixed sequence, not a choice: `app/retrieval/router.py` always runs metadata filter (`app/retrieval/filtering.py::filter_chunks()`, city/price from `QueryPreferences`) then the context-quality Judge (`app/retrieval/judge.py::judge_context()`, Groq call, retry-once-then-fallback) before synthesis.
- `ChunkMetadata` (`app/indexing/models.py`) has no field distinguishing which chunking strategy produced it, and `chunk_id` is a fresh UUID every index run — there's no stable identity for "the same source content indexed two different ways."
- `app/main.py::create_app()` wires exactly one `bm25_index`, one `vector_index`, one `chunk_store` at startup and passes them into both the indexing and retrieval routers — no per-strategy indirection anywhere.
- `archive/` currently holds PDFs used only for manual/ad-hoc testing — no code scans it; it is not part of the ingestion flow today.
- Test convention: `tests/<package>/test_<module>.py` mirroring `app/`, `httpx.MockTransport` for external calls, `asyncio_mode=auto`, `tmp_path`-isolated Qdrant per test (per existing `tests/indexing/test_wiring.py` pattern).

## Architecture

```
Load:  UI → POST /pipeline/load {indexing_strategy, retrieval_strategy, post_retrieval_strategy}
         → scan_archive() → diff vs that indexing strategy's collection's known doc_id_hashes
         → for each new doc: extract text → INDEXING_STRATEGIES[indexing](text, settings)
                            → embed → dual-write to that strategy's Qdrant collection + BM25 index
         → set ACTIVE_PIPELINE = the requested config
         → run_gold_eval(ACTIVE_PIPELINE)  [stub in this spec, real in piece C]
         → return {indexed: {new_docs, total_docs, failures}, eval: {...}}

Query: UI → POST /query {query}
         → read ACTIVE_PIPELINE (400 if none set)
         → RETRIEVAL_STRATEGIES[active.retrieval](query, active indexing strategy's collection + BM25)
         → POST_RETRIEVAL_STRATEGIES[active.post_retrieval](fused_chunks)
         → synthesize_answer()  [unchanged]
         → QueryResponse
```

### Pipeline config identity

```python
class PipelineConfig(BaseModel):
    indexing_strategy: Literal["fixed_window", "semantic", "hierarchical", "hierarchical_summary"]
    retrieval_strategy: Literal["bm25_only", "semantic_only", "hybrid_rrf"]
    post_retrieval_strategy: Literal["none", "metadata_filter", "cross_encoder_rerank"]
```

- **Indexing strategy has storage identity**: each value maps to its own Qdrant collection (e.g. `rag_chunks__fixed_window`) and its own `InMemoryBM25Index` instance. Switching indexing strategy means switching which collection/BM25 pair retrieval reads from — never a rebuild-in-place.
- **Retrieval and post-retrieval strategies are pure query-time logic** — no storage identity, free to swap instantly against whatever indexing collection is active.
- **Doc identity across strategies**: `doc_id_hash = sha256(file_bytes)`, computed once per archive file. Since `chunk_id` is regenerated per index run and chunk boundaries differ per strategy, `doc_id_hash` (not `chunk_id`) is what a collection's "already indexed" set is keyed on — this is what makes the lazy diff in `/pipeline/load` possible.
- **Sync model is lazy**: a collection only gets new archive PDFs indexed into it when it is the target of a Load. Other cached collections go stale until they're Loaded again. (Confirmed default — no eager background sync.)

## Changes

### 1. New: `app/archive/scanner.py`
```python
class ArchiveDoc(BaseModel):
    doc_id_hash: str
    path: str
    filename: str

def scan_archive(archive_dir: Path = Path("archive")) -> list[ArchiveDoc]: ...
```
Lists PDFs in `archive/`, computes `doc_id_hash` per file. Missing/empty directory returns `[]`, not an error.

### 2. New: `app/indexing/strategies/`
One module per strategy, common signature:
```python
def chunk(text: str, settings: Settings) -> list[Chunk]: ...
```
- `fixed_window.py` — today's `chunk_text()` logic, moved here unchanged.
- `semantic.py`, `hierarchical.py`, `hierarchical_summary.py` — stubs in this spec (raise `NotImplementedError` or return a trivial single-chunk split), real implementations are piece B's job. This spec only needs the registry shape and one working strategy (`fixed_window`) to prove the architecture end-to-end.

`strategies/__init__.py` exposes:
```python
INDEXING_STRATEGIES: dict[str, Callable[[str, Settings], list[Chunk]]]
```

### 3. New: `app/retrieval/strategies/`
Same pattern: `bm25_only.py`, `semantic_only.py`, `hybrid_rrf.py` (today's `retriever.py` logic, moved here, only fully-implemented one at this stage), registry `RETRIEVAL_STRATEGIES`.

### 4. New: `app/postretrieval/strategies/`
`none.py` (identity pass-through, fully implemented), `metadata_filter.py` (today's `filtering.py` logic, moved here, fully implemented), `cross_encoder_rerank.py` (stub, piece B). Registry `POST_RETRIEVAL_STRATEGIES`.

### 5. New: `app/pipeline/` package
- `config.py` — `PipelineConfig` model; `ACTIVE_PIPELINE: PipelineConfig | None` module-level holder; `IndexingCollectionRegistry` — maps each indexing strategy id to its `(QdrantVectorIndex, InMemoryBM25Index, ChunkStore)` triple, constructed once at app startup for all 4 indexing strategies (collections created lazily inside `QdrantVectorIndex._ensure_collection`, so an unused strategy's collection just stays empty, not un-created).
- `loader.py` — `async def load_pipeline(config: PipelineConfig) -> PipelineLoadResult`, implements the Load sequence above, guarded by a module-level `asyncio.Lock` so concurrent Load calls serialize instead of racing `ACTIVE_PIPELINE` or double-writing a collection. Per-doc indexing failures are caught individually — a bad PDF is recorded in `failures: list[{path, error}]` and skipped, not fatal to the whole batch. `run_gold_eval()` is a stub returning `{"status": "not_implemented"}` in this spec (piece C replaces it).
- `router.py` — `POST /pipeline/load`, `GET /pipeline/status` (returns `ACTIVE_PIPELINE` plus each indexing collection's current doc count, for the UI to render without a Load).

### 6. `app/main.py`
- Remove ingestion router wiring for URL-crawl endpoints; construct `IndexingCollectionRegistry` (4 collection/BM25/chunk_store triples) instead of one shared triple; wire `build_pipeline_router(registry, ...)`.
- `/query`'s router constructor now takes the `IndexingCollectionRegistry` + `ACTIVE_PIPELINE` accessor instead of a single fixed `bm25_index`/`vector_index` pair.

### 7. Deletions
- `app/ingestion/fetcher.py`, `pagination.py`, `job_store.py`, `worker.py`, and the URL-crawl parts of `app/ingestion/router.py` (PDF-text-extraction in `extractor.py` is kept — reused by the archive scanner's per-doc text extraction).
- `app/retrieval/judge.py`; `JudgeAttempt`/`judge_attempts` fields off `app/retrieval/models.py::QueryResponse`; judge-call sites and retry-on-insufficient logic in `app/retrieval/router.py`; the judge side panel in `app/static/js/query.js` / `query.html`.
- Corresponding tests: `tests/ingestion/test_fetcher.py`, `test_pagination.py`, `test_job_store.py`, `test_worker.py`'s fetch cases, `tests/retrieval/test_judge.py`.

### 8. `app/indexing/models.py`
`ChunkMetadata` gains `doc_id_hash: str` (replaces reliance on ephemeral `doc_id`/`chunk_id` for cross-strategy dedup checks — `doc_id` stays as today's per-index-run UUID, `doc_id_hash` is the new stable content-identity field used by the loader's diff logic).

## Non-Goals

- No real chunking/retrieval/post-retrieval algorithm work beyond `fixed_window` / `hybrid_rrf` / `metadata_filter` (and `none`) — the other 5 strategy slots are stubs, filled in by piece B.
- No gold test-case set or metric formulas — `run_gold_eval()` is a stub, filled in by piece C.
- No UI changes beyond what's needed to stop referencing deleted judge/URL-ingestion code — the combined pipeline screen is piece D.
- No eager cross-collection sync when new PDFs are dropped in `archive/` — confirmed lazy-only for this iteration.
- No multi-user/session isolation for `ACTIVE_PIPELINE` — single shared mutable state, matching the app's existing single-shared-corpus, no-auth posture.
- No embedding-model or chunk-size variation within `fixed_window` — that's a possible future strategy axis, not part of this spec's 4-strategy indexing set.

## Testing Plan

- `tests/pipeline/test_loader.py`: new-doc diff only indexes unseen `doc_id_hash`es; loading strategy B after strategy A never touches strategy A's collection/BM25 (isolation); one corrupt/unreadable PDF in the batch is recorded in `failures` and doesn't stop the rest from indexing; concurrent `load_pipeline()` calls serialize via the lock (second call's effects are visible only after the first completes, not interleaved); empty/missing `archive/` returns `indexed.new_docs == 0` without error.
- `tests/pipeline/test_router.py`: `/pipeline/load` response shape; `/pipeline/status` reflects `ACTIVE_PIPELINE` plus per-collection doc counts before any Load (all zero) and after; `/query` returns `400` with no active pipeline set.
- `tests/indexing/strategies/test_registry.py`, `tests/retrieval/strategies/test_registry.py`, `tests/postretrieval/strategies/test_registry.py`: each registry contains exactly the expected strategy ids, each entry is callable with the documented signature; `fixed_window`/`hybrid_rrf`/`metadata_filter`/`none` get real behavioral tests (ported from today's `test_chunker.py`/`test_retriever.py`/`test_filtering.py`), stub strategies just assert they raise/return the documented placeholder.
- `tests/archive/test_scanner.py`: `doc_id_hash` is stable across repeated scans of the same file, changes if file content changes, empty dir returns `[]`.
- Delete: `tests/ingestion/test_fetcher.py`, `test_pagination.py`, `test_job_store.py`, `test_worker.py`'s fetch-specific cases, `tests/retrieval/test_judge.py`.
