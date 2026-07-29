# RAG Retrieval & Synthesis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `POST /query` — parallel BM25 + semantic search, Reciprocal Rank Fusion, and LLM answer synthesis with `[n]` citations — per `doc/development/arch.md` sections 3 and 4.

**Architecture:** `app/retrieval/fusion.py` is a pure RRF implementation over two ranked-hit lists. `app/retrieval/retriever.py` dispatches BM25 search (`InMemoryBM25Index.search`, Plan 2) and semantic search (query embedding via `embed_texts` + `QdrantVectorIndex.search`, Plan 2) concurrently with `asyncio.gather`, fuses them, and attaches full chunk metadata from `ChunkStore` (Plan 2). `app/retrieval/synthesis.py` calls Groq's OpenAI-compatible chat completions endpoint over `httpx` with a citation-enforcing system prompt, and parses `[n]` markers back into `chunk_id` citations. `app/retrieval/router.py` wires both into `POST /query`.

**Tech Stack:** Python 3.11, FastAPI, httpx (query embedding + Groq calls), the Plan 2 indexing components (`InMemoryBM25Index`, `QdrantVectorIndex`, `ChunkStore`, `embed_texts`), pydantic, pytest + pytest-asyncio. No new third-party dependency — Groq is called as a plain HTTP API, no SDK.

## Global Constraints

- `POST /query` request: `{query: str, top_k: int | None}` — `top_k` overrides the configured display count for this request only.
- Retrieval dispatch is parallel (`asyncio.gather`, not sequential), each method retrieves `retrieval_top_k = 20` (config) candidates before fusion trims to the display count.
- RRF formula, exact: `rrf_score(chunk) = Σ 1 / (k + rank_in_list)` over lists containing the chunk, `k = 60` (config `rrf_k`). A chunk in only one list still gets a score from that list — never dropped for lacking a second hit.
- `matched_methods` is assembled server-side as `["bm25"]`, `["semantic"]`, or `["bm25", "semantic"]` — never recomputed client-side.
- Display count: config `display_top_k = 5`, overridable per-request via `QueryRequest.top_k`.
- Synthesis context budget: config `synthesis_context_budget = 6` (spec: "top 5–8 chunks") — the first N fused chunks (already in `fused_rank` order) go to the LLM; the rest are returned in `retrieved_chunks` for the UI but never sent to the model.
- Every chunk in `retrieved_chunks` carries `used_in_synthesis: bool` — `true` for chunks inside the context budget, `false` for chunks retrieved but not sent to the LLM. This is what lets the UI (Plan 4) distinguish "used in answer" from "retrieved but not used."
- Synthesis model: Groq API, model `openai/gpt-oss-120b`, OpenAI-compatible endpoint `{groq_base_url}/chat/completions` (default `https://api.groq.com/openai/v1`), auth header `Authorization: Bearer {groq_api_key}`.
- **Security:** `groq_api_key` is read from the `GROQ_API_KEY` environment variable at `Settings` construction time — never hardcoded, never committed. Tests always inject a fake key via `Settings(groq_api_key="test-key")`, never touching the real environment variable.
- System prompt requires the model to answer only from provided chunks and mark every claim with `[n]` referencing the chunk's 1-based position in the provided context list. No structural enforcement beyond the prompt — an unmarked sentence is a prompt-quality issue, not something this backend rejects (explicit spec non-goal for v1).
- Query embedding uses the *same* `Settings.embedding_model`/`ollama_base_url` already defined in Plan 2 — do not introduce a second embedding config; a mismatched model between indexing and querying silently degrades relevance (explicit spec warning).
- Response contract: `{query, answer, citations: [{marker, chunk_id}], retrieved_chunks: [FusedChunk...]}`.

---

## File Structure

```
app/
  config.py                       # MODIFY: add retrieval/synthesis settings
  main.py                          # MODIFY: wire /query router in
  retrieval/
    __init__.py
    models.py                      # QueryRequest, FusedChunk, Citation, QueryResponse
    fusion.py                       # RankedHit, FusedHit, reciprocal_rank_fusion()
    retriever.py                     # retrieve() — parallel dispatch + fuse + attach metadata
    synthesis.py                      # synthesize_answer() — Groq call + citation extraction
    router.py                          # build_retrieval_router() -> POST /query
tests/
  retrieval/
    test_fusion.py
    test_retriever.py
    test_synthesis.py
    test_router.py
```

---

### Task 1: Config extension & retrieval models

**Files:**
- Modify: `app/config.py`
- Create: `app/retrieval/__init__.py`
- Create: `app/retrieval/models.py`
- Modify: `tests/ingestion/test_config.py`
- Test: `tests/retrieval/test_models.py`

**Interfaces:**
- Produces: new `Settings` fields — `retrieval_top_k: int = 20`, `display_top_k: int = 5`, `rrf_k: int = 60`, `synthesis_context_budget: int = 6`, `groq_model: str = "openai/gpt-oss-120b"`, `groq_base_url: str = "https://api.groq.com/openai/v1"`, `groq_api_key: str` (defaults from `os.environ.get("GROQ_API_KEY", "")`).
- Produces: `QueryRequest(BaseModel)`: `query: str`, `top_k: int | None = None`. `Citation(BaseModel)`: `marker: int`, `chunk_id: str`. `FusedChunk(BaseModel)`: `chunk_id, text, source_url, page_number: int, bm25_rank: int | None, bm25_score: float | None, semantic_rank: int | None, semantic_score: float | None, fused_rank: int, rrf_score: float, matched_methods: list[str], used_in_synthesis: bool = False`. `QueryResponse(BaseModel)`: `query: str, answer: str, citations: list[Citation], retrieved_chunks: list[FusedChunk]`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/ingestion/test_config.py
def test_retrieval_and_synthesis_defaults(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    import importlib
    import app.config as config_module
    importlib.reload(config_module)

    assert config_module.settings.retrieval_top_k == 20
    assert config_module.settings.display_top_k == 5
    assert config_module.settings.rrf_k == 60
    assert config_module.settings.synthesis_context_budget == 6
    assert config_module.settings.groq_model == "openai/gpt-oss-120b"
    assert config_module.settings.groq_api_key == ""
```

```python
# tests/retrieval/test_models.py
from app.retrieval.models import Citation, FusedChunk, QueryRequest, QueryResponse


def test_query_request_top_k_optional():
    request = QueryRequest(query="what is RAG?")
    assert request.top_k is None


def test_fused_chunk_defaults_used_in_synthesis_false():
    chunk = FusedChunk(
        chunk_id="c1",
        text="hello",
        source_url="https://example.com",
        page_number=1,
        bm25_rank=1,
        bm25_score=8.3,
        semantic_rank=None,
        semantic_score=None,
        fused_rank=1,
        rrf_score=0.016,
        matched_methods=["bm25"],
    )
    assert chunk.used_in_synthesis is False


def test_query_response_round_trip():
    response = QueryResponse(
        query="q",
        answer="answer [1]",
        citations=[Citation(marker=1, chunk_id="c1")],
        retrieved_chunks=[],
    )
    assert response.model_dump()["citations"][0]["chunk_id"] == "c1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/ingestion/test_config.py tests/retrieval/test_models.py -v`
Expected: FAIL — new config test with `AttributeError`, models test with `ModuleNotFoundError: No module named 'app.retrieval'`

- [ ] **Step 3: Extend Settings and add models**

```python
# app/config.py (full file after edit)
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    fetch_timeout_seconds: float = 10.0
    user_agent: str = "Mozilla/5.0 (compatible; GenAI-RAG-Ingest/1.0)"
    max_pages: int = 20
    min_extract_length: int = 200
    chunk_size_tokens: int = 400
    chunk_overlap_tokens: int = 75
    embedding_model: str = "qwen3-embedding:0.6b"
    ollama_base_url: str = "http://localhost:11434"
    qdrant_url: str | None = None
    qdrant_path: str = ".data/qdrant"
    qdrant_collection: str = "rag_chunks"
    vector_size: int = 1024
    retrieval_top_k: int = 20
    display_top_k: int = 5
    rrf_k: int = 60
    synthesis_context_budget: int = 6
    groq_model: str = "openai/gpt-oss-120b"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_api_key: str = field(default_factory=lambda: os.environ.get("GROQ_API_KEY", ""))


settings = Settings()
```

```python
# app/retrieval/__init__.py
```

```python
# app/retrieval/models.py
from __future__ import annotations

from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str
    top_k: int | None = None


class Citation(BaseModel):
    marker: int
    chunk_id: str


class FusedChunk(BaseModel):
    chunk_id: str
    text: str
    source_url: str
    page_number: int
    bm25_rank: int | None
    bm25_score: float | None
    semantic_rank: int | None
    semantic_score: float | None
    fused_rank: int
    rrf_score: float
    matched_methods: list[str]
    used_in_synthesis: bool = False


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[FusedChunk]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/ingestion/test_config.py tests/retrieval/test_models.py -v`
Expected: PASS (new tests included)

- [ ] **Step 5: Commit**

```bash
git add app/config.py app/retrieval/__init__.py app/retrieval/models.py tests/ingestion/test_config.py tests/retrieval/test_models.py
git commit -m "feat: add retrieval/synthesis settings and query models"
```

---

### Task 2: Reciprocal Rank Fusion

**Files:**
- Create: `app/retrieval/fusion.py`
- Test: `tests/retrieval/test_fusion.py`

**Interfaces:**
- Consumes: nothing (pure function).
- Produces: `RankedHit` frozen dataclass (`chunk_id: str, rank: int, score: float`), `FusedHit` frozen dataclass (`chunk_id, fused_rank: int, rrf_score: float, bm25_rank: int | None, bm25_score: float | None, semantic_rank: int | None, semantic_score: float | None, matched_methods: list[str]`), `reciprocal_rank_fusion(bm25_hits: list[RankedHit], semantic_hits: list[RankedHit], k: int, top_k: int) -> list[FusedHit]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/retrieval/test_fusion.py
from app.retrieval.fusion import RankedHit, reciprocal_rank_fusion


def test_chunk_in_both_lists_outranks_single_list_hit():
    bm25 = [RankedHit("a", 1, 9.0), RankedHit("b", 2, 5.0)]
    semantic = [RankedHit("a", 1, 0.9)]
    fused = reciprocal_rank_fusion(bm25, semantic, k=60, top_k=5)
    assert fused[0].chunk_id == "a"
    assert fused[0].matched_methods == ["bm25", "semantic"]
    assert fused[1].chunk_id == "b"
    assert fused[1].matched_methods == ["bm25"]


def test_single_list_hit_still_scored_and_included():
    bm25 = []
    semantic = [RankedHit("only-semantic", 1, 0.5)]
    fused = reciprocal_rank_fusion(bm25, semantic, k=60, top_k=5)
    assert len(fused) == 1
    assert fused[0].chunk_id == "only-semantic"
    assert fused[0].rrf_score == 1 / 61
    assert fused[0].bm25_rank is None


def test_rrf_score_formula_exact():
    bm25 = [RankedHit("a", 3, 1.0)]
    semantic = [RankedHit("a", 5, 1.0)]
    fused = reciprocal_rank_fusion(bm25, semantic, k=60, top_k=5)
    expected = 1 / (60 + 3) + 1 / (60 + 5)
    assert abs(fused[0].rrf_score - expected) < 1e-9


def test_top_k_trims_result_and_fused_rank_is_1_indexed():
    bm25 = [RankedHit(f"c{i}", i + 1, float(10 - i)) for i in range(10)]
    fused = reciprocal_rank_fusion(bm25, [], k=60, top_k=3)
    assert len(fused) == 3
    assert [f.fused_rank for f in fused] == [1, 2, 3]
    assert fused[0].chunk_id == "c0"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/retrieval/test_fusion.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.retrieval.fusion'`

- [ ] **Step 3: Implement fusion**

```python
# app/retrieval/fusion.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RankedHit:
    chunk_id: str
    rank: int
    score: float


@dataclass(frozen=True)
class FusedHit:
    chunk_id: str
    fused_rank: int
    rrf_score: float
    bm25_rank: int | None
    bm25_score: float | None
    semantic_rank: int | None
    semantic_score: float | None
    matched_methods: list[str]


def reciprocal_rank_fusion(
    bm25_hits: list[RankedHit],
    semantic_hits: list[RankedHit],
    k: int,
    top_k: int,
) -> list[FusedHit]:
    bm25_by_id = {hit.chunk_id: hit for hit in bm25_hits}
    semantic_by_id = {hit.chunk_id: hit for hit in semantic_hits}
    ordered_ids = list(dict.fromkeys([h.chunk_id for h in bm25_hits] + [h.chunk_id for h in semantic_hits]))

    scored = []
    for chunk_id in ordered_ids:
        bm25_hit = bm25_by_id.get(chunk_id)
        semantic_hit = semantic_by_id.get(chunk_id)
        rrf_score = 0.0
        matched_methods = []
        if bm25_hit:
            rrf_score += 1.0 / (k + bm25_hit.rank)
            matched_methods.append("bm25")
        if semantic_hit:
            rrf_score += 1.0 / (k + semantic_hit.rank)
            matched_methods.append("semantic")
        scored.append((chunk_id, rrf_score, bm25_hit, semantic_hit, matched_methods))

    scored.sort(key=lambda item: item[1], reverse=True)

    return [
        FusedHit(
            chunk_id=chunk_id,
            fused_rank=index + 1,
            rrf_score=rrf_score,
            bm25_rank=bm25_hit.rank if bm25_hit else None,
            bm25_score=bm25_hit.score if bm25_hit else None,
            semantic_rank=semantic_hit.rank if semantic_hit else None,
            semantic_score=semantic_hit.score if semantic_hit else None,
            matched_methods=matched_methods,
        )
        for index, (chunk_id, rrf_score, bm25_hit, semantic_hit, matched_methods) in enumerate(scored[:top_k])
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/retrieval/test_fusion.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add app/retrieval/fusion.py tests/retrieval/test_fusion.py
git commit -m "feat: add Reciprocal Rank Fusion"
```

---

### Task 3: Retriever (parallel dispatch + metadata attach)

**Files:**
- Create: `app/retrieval/retriever.py`
- Test: `tests/retrieval/test_retriever.py`

**Interfaces:**
- Consumes: `InMemoryBM25Index.search(query, top_k) -> list[tuple[str, float]]`, `QdrantVectorIndex.search(vector, top_k) -> list[tuple[str, float]]`, `ChunkStore.get_many(chunk_ids) -> list[ChunkMetadata]` (Plan 2); `embed_texts(client, texts, settings) -> list[list[float]]` (Plan 2); `RankedHit`, `reciprocal_rank_fusion` (Task 2); `FusedChunk` (Task 1).
- Produces: `async def retrieve(query: str, bm25_index, vector_index, chunk_store, http_client: httpx.AsyncClient, settings: Settings = settings, top_k: int | None = None) -> list[FusedChunk]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/retrieval/test_retriever.py
import httpx

from app.config import Settings
from app.indexing.bm25_index import InMemoryBM25Index
from app.indexing.chunk_store import ChunkStore
from app.indexing.models import ChunkMetadata
from app.indexing.vector_index import QdrantVectorIndex
from app.retrieval.retriever import retrieve


def _seed(tmp_path):
    settings = Settings(qdrant_path=str(tmp_path / "qdrant"), qdrant_collection="t", vector_size=2)
    bm25 = InMemoryBM25Index()
    vectors = QdrantVectorIndex(settings)
    store = ChunkStore()

    chunk = ChunkMetadata(
        chunk_id="c1",
        doc_id="d1",
        source_url="https://example.com",
        page_number=1,
        chunk_index=0,
        char_start=0,
        char_end=20,
        overlap_with_prev=0,
        indexed_at="2026-07-29T12:00:00+00:00",
        text="the quick brown fox",
    )
    bm25.add_documents(["c1"], ["the quick brown fox"])
    vectors.upsert(["c1"], [[1.0, 0.0]], [chunk.model_dump()])
    store.add([chunk])
    return settings, bm25, vectors, store


async def test_retrieve_attaches_metadata_and_both_scores(tmp_path):
    settings, bm25, vectors, store = _seed(tmp_path)

    def handler(request):
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        results = await retrieve("fox", bm25, vectors, store, client, settings)

    assert len(results) == 1
    chunk = results[0]
    assert chunk.chunk_id == "c1"
    assert chunk.text == "the quick brown fox"
    assert chunk.page_number == 1
    assert chunk.matched_methods == ["bm25", "semantic"]
    assert chunk.bm25_rank == 1
    assert chunk.semantic_rank == 1
    assert chunk.used_in_synthesis is False


async def test_retrieve_respects_top_k_override(tmp_path):
    settings, bm25, vectors, store = _seed(tmp_path)

    def handler(request):
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        results = await retrieve("fox", bm25, vectors, store, client, settings, top_k=0)

    assert results == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/retrieval/test_retriever.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.retrieval.retriever'`

- [ ] **Step 3: Implement retriever**

```python
# app/retrieval/retriever.py
from __future__ import annotations

import asyncio

import httpx

from app.config import Settings, settings as default_settings
from app.indexing.bm25_index import InMemoryBM25Index
from app.indexing.chunk_store import ChunkStore
from app.indexing.embeddings import embed_texts
from app.indexing.vector_index import QdrantVectorIndex
from app.retrieval.fusion import RankedHit, reciprocal_rank_fusion
from app.retrieval.models import FusedChunk


async def retrieve(
    query: str,
    bm25_index: InMemoryBM25Index,
    vector_index: QdrantVectorIndex,
    chunk_store: ChunkStore,
    http_client: httpx.AsyncClient,
    settings: Settings = default_settings,
    top_k: int | None = None,
) -> list[FusedChunk]:
    display_top_k = settings.display_top_k if top_k is None else top_k

    async def bm25_search() -> list[RankedHit]:
        results = bm25_index.search(query, settings.retrieval_top_k)
        return [RankedHit(chunk_id, rank + 1, score) for rank, (chunk_id, score) in enumerate(results)]

    async def semantic_search() -> list[RankedHit]:
        vectors = await embed_texts(http_client, [query], settings)
        results = vector_index.search(vectors[0], settings.retrieval_top_k)
        return [RankedHit(chunk_id, rank + 1, score) for rank, (chunk_id, score) in enumerate(results)]

    bm25_hits, semantic_hits = await asyncio.gather(bm25_search(), semantic_search())
    fused_hits = reciprocal_rank_fusion(bm25_hits, semantic_hits, settings.rrf_k, display_top_k)

    metadata_by_id = {
        chunk.chunk_id: chunk for chunk in chunk_store.get_many([hit.chunk_id for hit in fused_hits])
    }

    return [
        FusedChunk(
            chunk_id=hit.chunk_id,
            text=metadata_by_id[hit.chunk_id].text,
            source_url=metadata_by_id[hit.chunk_id].source_url,
            page_number=metadata_by_id[hit.chunk_id].page_number,
            bm25_rank=hit.bm25_rank,
            bm25_score=hit.bm25_score,
            semantic_rank=hit.semantic_rank,
            semantic_score=hit.semantic_score,
            fused_rank=hit.fused_rank,
            rrf_score=hit.rrf_score,
            matched_methods=hit.matched_methods,
        )
        for hit in fused_hits
        if hit.chunk_id in metadata_by_id
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/retrieval/test_retriever.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add app/retrieval/retriever.py tests/retrieval/test_retriever.py
git commit -m "feat: add retriever with parallel BM25/semantic dispatch and RRF fusion"
```

---

### Task 4: Synthesis (Groq)

**Files:**
- Create: `app/retrieval/synthesis.py`
- Test: `tests/retrieval/test_synthesis.py`

**Interfaces:**
- Consumes: `FusedChunk`, `Citation` from `app.retrieval.models`; `Settings` (`groq_model`, `groq_base_url`, `groq_api_key`, `synthesis_context_budget`, `fetch_timeout_seconds`).
- Produces: `SynthesisError(Exception)`; `async def synthesize_answer(query: str, fused_chunks: list[FusedChunk], http_client: httpx.AsyncClient, settings: Settings = settings) -> tuple[str, list[Citation], set[str]]` — returns `(answer_text, citations, used_chunk_ids)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/retrieval/test_synthesis.py
import httpx
import pytest

from app.config import Settings
from app.retrieval.models import FusedChunk
from app.retrieval.synthesis import SynthesisError, synthesize_answer


def _chunk(chunk_id: str, text: str) -> FusedChunk:
    return FusedChunk(
        chunk_id=chunk_id,
        text=text,
        source_url="https://example.com",
        page_number=1,
        bm25_rank=1,
        bm25_score=1.0,
        semantic_rank=1,
        semantic_score=1.0,
        fused_rank=1,
        rrf_score=0.03,
        matched_methods=["bm25", "semantic"],
    )


def _groq_response(content: str) -> httpx.Response:
    return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


async def test_synthesize_answer_extracts_citations_within_budget():
    settings = Settings(groq_api_key="test-key", synthesis_context_budget=2)
    chunks = [_chunk("c1", "Paris is the capital of France."), _chunk("c2", "It has the Eiffel Tower.")]

    def handler(request):
        assert request.headers["authorization"] == "Bearer test-key"
        return _groq_response("Paris is the capital of France [1], home to the Eiffel Tower [2].")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        answer, citations, used_chunk_ids = await synthesize_answer("capital of France?", chunks, client, settings)

    assert "[1]" in answer and "[2]" in answer
    assert {c.chunk_id for c in citations} == {"c1", "c2"}
    assert used_chunk_ids == {"c1", "c2"}


async def test_synthesize_answer_respects_context_budget():
    settings = Settings(groq_api_key="test-key", synthesis_context_budget=1)
    chunks = [_chunk("c1", "first"), _chunk("c2", "second")]

    captured = {}

    def handler(request):
        captured["body"] = request.read()
        return _groq_response("first fact [1].")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        _answer, citations, used_chunk_ids = await synthesize_answer("q", chunks, client, settings)

    assert b"second" not in captured["body"]
    assert used_chunk_ids == {"c1"}
    assert {c.chunk_id for c in citations} == {"c1"}


async def test_synthesize_answer_raises_without_api_key():
    settings = Settings(groq_api_key="")
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200))) as client:
        with pytest.raises(SynthesisError, match="GROQ_API_KEY"):
            await synthesize_answer("q", [_chunk("c1", "x")], client, settings)


async def test_synthesize_answer_raises_on_non_200():
    settings = Settings(groq_api_key="test-key")

    def handler(request):
        return httpx.Response(500, text="boom")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(SynthesisError, match="500"):
            await synthesize_answer("q", [_chunk("c1", "x")], client, settings)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/retrieval/test_synthesis.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.retrieval.synthesis'`

- [ ] **Step 3: Implement synthesis**

```python
# app/retrieval/synthesis.py
from __future__ import annotations

import re

import httpx

from app.config import Settings, settings as default_settings
from app.retrieval.models import Citation, FusedChunk

_CITATION_RE = re.compile(r"\[(\d+)\]")

SYSTEM_PROMPT = (
    "Answer only using the numbered context chunks provided below. "
    "Every claim you make must be immediately followed by a citation marker "
    "like [1] or [2], referencing the chunk's position in the context list. "
    "If the context does not contain the answer, say so plainly."
)


class SynthesisError(Exception):
    pass


def _build_context_block(chunks: list[FusedChunk]) -> str:
    return "\n\n".join(f"[{index + 1}] {chunk.text}" for index, chunk in enumerate(chunks))


def _extract_citations(answer: str, chunks: list[FusedChunk]) -> list[Citation]:
    markers = sorted({int(marker) for marker in _CITATION_RE.findall(answer)})
    return [
        Citation(marker=marker, chunk_id=chunks[marker - 1].chunk_id)
        for marker in markers
        if 1 <= marker <= len(chunks)
    ]


async def synthesize_answer(
    query: str,
    fused_chunks: list[FusedChunk],
    http_client: httpx.AsyncClient,
    settings: Settings = default_settings,
) -> tuple[str, list[Citation], set[str]]:
    if not settings.groq_api_key:
        raise SynthesisError("GROQ_API_KEY is not configured")

    context_chunks = fused_chunks[: settings.synthesis_context_budget]
    context_block = _build_context_block(context_chunks)

    try:
        response = await http_client.post(
            f"{settings.groq_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {settings.groq_api_key}"},
            json={
                "model": settings.groq_model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {query}"},
                ],
            },
            timeout=settings.fetch_timeout_seconds,
        )
    except httpx.RequestError as exc:
        raise SynthesisError(f"synthesis request failed: {exc}") from exc

    if response.status_code != 200:
        raise SynthesisError(f"synthesis request returned status {response.status_code}")

    try:
        answer = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise SynthesisError("synthesis response missing answer content") from exc

    citations = _extract_citations(answer, context_chunks)
    used_chunk_ids = {chunk.chunk_id for chunk in context_chunks}
    return answer, citations, used_chunk_ids
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/retrieval/test_synthesis.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add app/retrieval/synthesis.py tests/retrieval/test_synthesis.py
git commit -m "feat: add Groq-backed synthesis with citation extraction and context budget"
```

---

### Task 5: Router + wiring

**Files:**
- Create: `app/retrieval/router.py`
- Modify: `app/main.py`
- Test: `tests/retrieval/test_router.py`

**Interfaces:**
- Consumes: `retrieve()` (Task 3), `synthesize_answer()` (Task 4), `QueryRequest`/`QueryResponse` (Task 1), and the Plan 2 indexing components already constructed in `create_app()`.
- Produces: `def build_retrieval_router(bm25_index, vector_index, chunk_store, embedding_client: httpx.AsyncClient, synthesis_client: httpx.AsyncClient, settings=settings) -> APIRouter` exposing `POST /query`.

- [ ] **Step 1: Write the failing test**

```python
# tests/retrieval/test_router.py
import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import Settings
from app.indexing.bm25_index import InMemoryBM25Index
from app.indexing.chunk_store import ChunkStore
from app.indexing.models import ChunkMetadata
from app.indexing.vector_index import QdrantVectorIndex
from app.retrieval.router import build_retrieval_router


def test_post_query_returns_answer_with_citations_and_chunks(tmp_path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="t",
        vector_size=2,
        groq_api_key="test-key",
        synthesis_context_budget=5,
    )
    bm25 = InMemoryBM25Index()
    vectors = QdrantVectorIndex(settings)
    store = ChunkStore()

    chunk = ChunkMetadata(
        chunk_id="c1",
        doc_id="d1",
        source_url="https://example.com",
        page_number=1,
        chunk_index=0,
        char_start=0,
        char_end=20,
        overlap_with_prev=0,
        indexed_at="2026-07-29T12:00:00+00:00",
        text="the quick brown fox",
    )
    bm25.add_documents(["c1"], ["the quick brown fox"])
    vectors.upsert(["c1"], [[1.0, 0.0]], [chunk.model_dump()])
    store.add([chunk])

    def embed_handler(request):
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})

    def groq_handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "The fox is quick [1]."}}]})

    embedding_client = httpx.AsyncClient(transport=httpx.MockTransport(embed_handler))
    synthesis_client = httpx.AsyncClient(transport=httpx.MockTransport(groq_handler))

    app = FastAPI()
    app.include_router(build_retrieval_router(bm25, vectors, store, embedding_client, synthesis_client, settings))

    with TestClient(app) as client:
        response = client.post("/query", json={"query": "tell me about the fox"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "The fox is quick [1]."
    assert body["citations"] == [{"marker": 1, "chunk_id": "c1"}]
    assert body["retrieved_chunks"][0]["chunk_id"] == "c1"
    assert body["retrieved_chunks"][0]["used_in_synthesis"] is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/retrieval/test_router.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.retrieval.router'`

- [ ] **Step 3: Implement router and wire main.py**

```python
# app/retrieval/router.py
from __future__ import annotations

import httpx
from fastapi import APIRouter

from app.config import Settings, settings as default_settings
from app.indexing.bm25_index import InMemoryBM25Index
from app.indexing.chunk_store import ChunkStore
from app.indexing.vector_index import QdrantVectorIndex
from app.retrieval.models import QueryRequest, QueryResponse
from app.retrieval.retriever import retrieve
from app.retrieval.synthesis import synthesize_answer


def build_retrieval_router(
    bm25_index: InMemoryBM25Index,
    vector_index: QdrantVectorIndex,
    chunk_store: ChunkStore,
    embedding_client: httpx.AsyncClient,
    synthesis_client: httpx.AsyncClient,
    settings: Settings = default_settings,
) -> APIRouter:
    router = APIRouter()

    @router.post("/query", response_model=QueryResponse)
    async def query(request: QueryRequest) -> QueryResponse:
        fused_chunks = await retrieve(
            request.query, bm25_index, vector_index, chunk_store, embedding_client, settings, request.top_k
        )
        answer, citations, used_chunk_ids = await synthesize_answer(
            request.query, fused_chunks, synthesis_client, settings
        )
        for chunk in fused_chunks:
            chunk.used_in_synthesis = chunk.chunk_id in used_chunk_ids

        return QueryResponse(query=request.query, answer=answer, citations=citations, retrieved_chunks=fused_chunks)

    return router
```

```python
# app/main.py — full file after edit
from __future__ import annotations

import httpx
from fastapi import FastAPI

from app.config import settings
from app.indexing.bm25_index import InMemoryBM25Index
from app.indexing.chunk_store import ChunkStore
from app.indexing.indexer import index_document
from app.indexing.router import build_indexing_router
from app.indexing.vector_index import QdrantVectorIndex
from app.ingestion.job_store import JobStore
from app.ingestion.router import build_ingestion_router
from app.retrieval.router import build_retrieval_router


def create_app() -> FastAPI:
    app = FastAPI(title="RAG Ingestion Service")

    job_store = JobStore()
    bm25_index = InMemoryBM25Index()
    vector_index = QdrantVectorIndex(settings)
    chunk_store = ChunkStore()
    embedding_client = httpx.AsyncClient()
    synthesis_client = httpx.AsyncClient()

    async def index_sink(payload) -> None:
        await index_document(payload, bm25_index, vector_index, chunk_store, embedding_client, settings)

    app.include_router(build_ingestion_router(job_store, sink=index_sink))
    app.include_router(build_indexing_router(bm25_index, vector_index, chunk_store, embedding_client))
    app.include_router(
        build_retrieval_router(bm25_index, vector_index, chunk_store, embedding_client, synthesis_client)
    )

    return app


app = create_app()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/retrieval/test_router.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Run the full test suite**

Run: `uv run pytest -v`
Expected: all tests across all three plans PASS

- [ ] **Step 6: Commit**

```bash
git add app/retrieval/router.py app/main.py tests/retrieval/test_router.py
git commit -m "feat: wire POST /query (retrieval + fusion + synthesis)"
```

---

## Self-Review Notes

- **Spec coverage:** Query contract, parallel dispatch, BM25/semantic search, RRF fusion (exact formula, single-list hits kept), result assembly with both sub-scores and `matched_methods` — Tasks 1–3. Synthesis prompt/citation contract, context budget, `used_in_synthesis` marking — Task 4. Full `POST /query` response contract — Task 5.
- **Type consistency checked:** `FusedChunk`/`Citation`/`QueryRequest`/`QueryResponse` defined once in Task 1, consumed unchanged by Tasks 3–5. `RankedHit`/`FusedHit` defined in Task 2, consumed unchanged by Task 3.
- **Cross-plan dependency:** Tasks 3–5 depend on Plan 2's `InMemoryBM25Index`, `QdrantVectorIndex`, `ChunkStore`, and `embed_texts` — Plan 2 must be implemented first (matches the intended 1→2→3→4 execution order).
- **Security:** no real Groq credential appears anywhere in this plan; every test supplies a fake key via `Settings(groq_api_key=...)`.
- **No placeholders:** every step has runnable code; no "TBD" left.
