# RAG Indexing Subsystem Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the indexing subsystem — sentence-aware token chunking, chunk metadata assembly, and dual write to an in-memory BM25 index + Qdrant vector index — per `doc/development/arch.md` section 2. Wire it in as the ingestion worker's sink so a completed ingest automatically indexes.

**Architecture:** `app/indexing/chunker.py` splits `cleaned_text` into token-windowed, sentence-boundary-respecting chunks. `app/indexing/indexer.py` orchestrates: chunk → resolve page numbers via `page_map` → embed (Ollama, via httpx) → dual-write to `InMemoryBM25Index` and `QdrantVectorIndex`, rolling back the BM25 write if the vector write fails → record full metadata in `ChunkStore` (the document store). `app/indexing/router.py` exposes the internal `POST /index/chunk` contract; `app/main.py` wires `index_document` in as the real `IngestSink` for the ingestion worker (replacing Plan 1's no-op default).

**Tech Stack:** Python 3.11 (continues Plan 1's bump), `tiktoken` (token counting), `rank-bm25`, `qdrant-client` (already a dependency), `httpx` (already a dependency, reused for the Ollama embeddings call), pydantic, pytest + pytest-asyncio.

## Global Constraints

- Chunking: token-based (not char-based), `chunk_size = 400` tokens, `overlap = 75` tokens — both config (`Settings`), not hardcoded. Respect sentence boundaries where feasible to avoid mid-sentence overlap cuts; a single sentence longer than `chunk_size` is its own chunk (unavoidable overflow).
- Token counting uses `tiktoken`'s `cl100k_base` encoding as an approximation of the embedding model's real tokenizer — the spec only requires "token-based," not an exact match to Qwen3's tokenizer; `chunk_size`/`overlap` are already tunable config.
- Chunk metadata is a fixed schema (this is the retrieval/UI contract, do not rename fields): `chunk_id, doc_id, source_url, page_number, chunk_index, char_start, char_end, overlap_with_prev, indexed_at, text`.
- `page_number` is resolved per chunk from the ingestion `page_map` (list of `{page, char_start, char_end}`) by locating which page's char range contains the chunk's `char_start`.
- Dual write: the same `chunk_id` goes to both the BM25 index and the vector index — never one without the other. If the vector upsert fails after the BM25 write succeeded, the BM25 write is rolled back and the error re-raised.
- BM25 store: in-memory `rank_bm25` (`BM25Okapi`), rebuilt on each batch add/remove — acceptable at showcase scale per the resolved decision.
- Vector store: Qdrant (`qdrant-client`), local on-disk mode by default (`QdrantClient(path=...)`) unless `qdrant_url` is configured for a remote server. Distance: cosine. Full chunk metadata is stored as the point payload (needed later for score display, not just retrieval).
- Embedding: Ollama, model `qwen3-embedding:0.6b` (or higher), called over HTTP (`POST {ollama_base_url}/api/embed`) via `httpx` — no new embedding-client dependency.
- Non-goals (do not build): no re-chunking strategy switching at query time, no dedup/near-dup detection across sources.
- API: `POST /index/chunk` is internal (not user-facing), request body is the ingestion `IngestionPayload` contract, response is `{doc_id, status: "indexed", chunk_count}`.

---

## File Structure

```
app/
  config.py                    # MODIFY: add chunking/embedding/qdrant settings
  ingestion/
    worker.py                  # MODIFY: rename _noop_sink -> noop_sink (public)
    router.py                  # MODIFY: build_ingestion_router gains a `sink` param
  main.py                       # MODIFY: wire index_document in as the real ingestion sink
  indexing/
    __init__.py
    models.py                  # ChunkMetadata, IndexResult
    chunker.py                  # sentence_spans(), count_tokens(), chunk_text() -> list[TextChunk]
    embeddings.py                # embed_texts() over Ollama's /api/embed
    bm25_index.py                 # InMemoryBM25Index
    vector_index.py                # QdrantVectorIndex
    chunk_store.py                   # ChunkStore (the document store)
    indexer.py                        # index_document() orchestrator, dual-write + rollback
    router.py                          # build_indexing_router() -> POST /index/chunk
tests/
  indexing/
    test_chunker.py
    test_embeddings.py
    test_bm25_index.py
    test_vector_index.py
    test_chunk_store.py
    test_indexer.py
    test_router.py
    test_wiring.py
```

---

### Task 1: Config extension & indexing models

**Files:**
- Modify: `app/config.py`
- Create: `app/indexing/__init__.py`
- Create: `app/indexing/models.py`
- Modify: `tests/ingestion/test_config.py`
- Test: `tests/indexing/test_models.py`

**Interfaces:**
- Produces: new `Settings` fields — `chunk_size_tokens: int = 400`, `chunk_overlap_tokens: int = 75`, `embedding_model: str = "qwen3-embedding:0.6b"`, `ollama_base_url: str = "http://localhost:11434"`, `qdrant_url: str | None = None`, `qdrant_path: str = ".data/qdrant"`, `qdrant_collection: str = "rag_chunks"`, `vector_size: int = 1024`.
- Produces: `ChunkMetadata(BaseModel)` with the exact fields listed in Global Constraints; `IndexResult(BaseModel)`: `doc_id: str`, `status: str`, `chunk_count: int`.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/ingestion/test_config.py
def test_indexing_defaults():
    from app.config import settings
    assert settings.chunk_size_tokens == 400
    assert settings.chunk_overlap_tokens == 75
    assert settings.embedding_model == "qwen3-embedding:0.6b"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.qdrant_url is None
    assert settings.qdrant_collection == "rag_chunks"
    assert settings.vector_size == 1024
```

```python
# tests/indexing/test_models.py
from app.indexing.models import ChunkMetadata, IndexResult


def test_chunk_metadata_round_trip():
    chunk = ChunkMetadata(
        chunk_id="c1",
        doc_id="d1",
        source_url="https://example.com",
        page_number=1,
        chunk_index=0,
        char_start=0,
        char_end=100,
        overlap_with_prev=0,
        indexed_at="2026-07-29T12:00:00+00:00",
        text="hello world",
    )
    dumped = chunk.model_dump()
    assert dumped["chunk_id"] == "c1"
    assert dumped["overlap_with_prev"] == 0


def test_index_result():
    result = IndexResult(doc_id="d1", status="indexed", chunk_count=3)
    assert result.model_dump() == {"doc_id": "d1", "status": "indexed", "chunk_count": 3}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/ingestion/test_config.py tests/indexing/test_models.py -v`
Expected: FAIL — `test_indexing_defaults` with `AttributeError`, `test_models.py` with `ModuleNotFoundError: No module named 'app.indexing'`

- [ ] **Step 3: Extend Settings and add models**

```python
# app/config.py (full file after edit)
from dataclasses import dataclass


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


settings = Settings()
```

```python
# app/indexing/__init__.py
```

```python
# app/indexing/models.py
from __future__ import annotations

from pydantic import BaseModel


class ChunkMetadata(BaseModel):
    chunk_id: str
    doc_id: str
    source_url: str
    page_number: int
    chunk_index: int
    char_start: int
    char_end: int
    overlap_with_prev: int
    indexed_at: str
    text: str


class IndexResult(BaseModel):
    doc_id: str
    status: str
    chunk_count: int
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/ingestion/test_config.py tests/indexing/test_models.py -v`
Expected: PASS (4 + 2 tests)

- [ ] **Step 5: Commit**

```bash
git add app/config.py app/indexing/__init__.py app/indexing/models.py tests/ingestion/test_config.py tests/indexing/test_models.py
git commit -m "feat: add indexing settings and ChunkMetadata/IndexResult models"
```

---

### Task 2: Chunker

**Files:**
- Create: `app/indexing/chunker.py`
- Test: `tests/indexing/test_chunker.py`

**Interfaces:**
- Consumes: nothing (pure functions over strings; token counting is injectable for testability).
- Produces:
  - `TextChunk` frozen dataclass: `text: str`, `char_start: int`, `char_end: int`, `overlap_with_prev: int`
  - `sentence_spans(text: str) -> list[tuple[int, int]]`
  - `count_tokens(text: str) -> int` (tiktoken `cl100k_base`, the default counter)
  - `chunk_text(text: str, chunk_size: int, overlap: int, token_counter: Callable[[str], int] = count_tokens) -> list[TextChunk]`

- [ ] **Step 1: Add tiktoken dependency**

```bash
uv add tiktoken
```

- [ ] **Step 2: Write the failing test**

```python
# tests/indexing/test_chunker.py
from app.indexing.chunker import chunk_text, sentence_spans

WORD_COUNTER = lambda text: len(text.split())  # noqa: E731 — predictable counts for these tests


def test_sentence_spans_splits_on_terminal_punctuation():
    text = "First sentence. Second sentence! Third one?"
    spans = sentence_spans(text)
    assert [text[s:e] for s, e in spans] == [
        "First sentence.",
        "Second sentence!",
        "Third one?",
    ]


def test_chunk_text_packs_sentences_within_token_budget():
    text = "one two three. four five six. seven eight nine. ten eleven twelve."
    chunks = chunk_text(text, chunk_size=6, overlap=0, token_counter=WORD_COUNTER)
    assert [c.text for c in chunks] == [
        "one two three. four five six.",
        "seven eight nine. ten eleven twelve.",
    ]
    assert chunks[0].char_start == 0
    assert chunks[1].char_start == text.index("seven")


def test_chunk_text_applies_overlap_between_windows():
    text = "one two. three four. five six. seven eight."
    chunks = chunk_text(text, chunk_size=4, overlap=2, token_counter=WORD_COUNTER)
    assert len(chunks) >= 2
    assert chunks[0].overlap_with_prev == 0
    assert chunks[1].overlap_with_prev > 0
    # the overlapping words from the end of chunk 0 reappear at the start of chunk 1
    assert chunks[0].text.split(".")[-2].strip() in chunks[1].text


def test_chunk_text_oversized_single_sentence_stands_alone():
    text = "one two three four five six seven eight nine ten."
    chunks = chunk_text(text, chunk_size=3, overlap=0, token_counter=WORD_COUNTER)
    assert len(chunks) == 1
    assert chunks[0].text == text


def test_chunk_text_empty_input_returns_no_chunks():
    assert chunk_text("", chunk_size=400, overlap=75) == []


def test_count_tokens_default_uses_tiktoken():
    from app.indexing.chunker import count_tokens
    assert count_tokens("hello world") > 0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/indexing/test_chunker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.indexing.chunker'`

- [ ] **Step 4: Implement chunker**

```python
# app/indexing/chunker.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import tiktoken

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
_ENCODING = tiktoken.get_encoding("cl100k_base")


@dataclass(frozen=True)
class TextChunk:
    text: str
    char_start: int
    char_end: int
    overlap_with_prev: int


def sentence_spans(text: str) -> list[tuple[int, int]]:
    if not text:
        return []
    spans = []
    start = 0
    for match in _SENTENCE_BOUNDARY_RE.finditer(text):
        spans.append((start, match.start()))
        start = match.end()
    spans.append((start, len(text)))
    return [span for span in spans if span[1] > span[0]]


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
    token_counter: Callable[[str], int] = count_tokens,
) -> list[TextChunk]:
    sentences = [(s, e, token_counter(text[s:e])) for s, e in sentence_spans(text)]
    n = len(sentences)
    chunks: list[TextChunk] = []
    i = 0
    prev_overlap = 0

    while i < n:
        window_end = i
        window_tokens = 0
        while window_end < n:
            tok = sentences[window_end][2]
            if window_end > i and window_tokens + tok > chunk_size:
                break
            window_tokens += tok
            window_end += 1

        chunk_start = sentences[i][0]
        chunk_end = sentences[window_end - 1][1]
        chunks.append(TextChunk(text[chunk_start:chunk_end], chunk_start, chunk_end, prev_overlap))

        if window_end >= n:
            break

        overlap_tokens = 0
        k = window_end - 1
        while k >= i and overlap_tokens < overlap:
            overlap_tokens += sentences[k][2]
            k -= 1
        next_i = k + 1
        if next_i <= i:
            next_i = window_end
            overlap_tokens = 0
        prev_overlap = overlap_tokens
        i = next_i

    return chunks
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/indexing/test_chunker.py -v`
Expected: PASS (6 tests)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock app/indexing/chunker.py tests/indexing/test_chunker.py
git commit -m "feat: add sentence-aware token-windowed chunker with overlap"
```

---

### Task 3: Embeddings client

**Files:**
- Create: `app/indexing/embeddings.py`
- Test: `tests/indexing/test_embeddings.py`

**Interfaces:**
- Consumes: `Settings` (`embedding_model`, `ollama_base_url`, `fetch_timeout_seconds`).
- Produces: `EmbeddingError(Exception)`; `async def embed_texts(client: httpx.AsyncClient, texts: list[str], settings: Settings) -> list[list[float]]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/indexing/test_embeddings.py
import httpx
import pytest

from app.config import Settings
from app.indexing.embeddings import EmbeddingError, embed_texts

SETTINGS = Settings()


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


async def test_embed_texts_returns_vectors():
    def handler(request):
        body = request.read()
        assert b"qwen3-embedding" in body
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]})

    async with _client(handler) as client:
        vectors = await embed_texts(client, ["a", "b"], SETTINGS)
    assert vectors == [[0.1, 0.2], [0.3, 0.4]]


async def test_embed_texts_empty_input_short_circuits():
    async def unreachable(request):
        raise AssertionError("should not be called for empty input")

    async with _client(unreachable) as client:
        vectors = await embed_texts(client, [], SETTINGS)
    assert vectors == []


async def test_embed_texts_raises_on_non_200():
    def handler(request):
        return httpx.Response(500, text="boom")

    async with _client(handler) as client:
        with pytest.raises(EmbeddingError, match="500"):
            await embed_texts(client, ["a"], SETTINGS)


async def test_embed_texts_raises_on_mismatched_vector_count():
    def handler(request):
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2]]})

    async with _client(handler) as client:
        with pytest.raises(EmbeddingError, match="mismatched"):
            await embed_texts(client, ["a", "b"], SETTINGS)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/indexing/test_embeddings.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.indexing.embeddings'`

- [ ] **Step 3: Implement embeddings client**

```python
# app/indexing/embeddings.py
from __future__ import annotations

import httpx

from app.config import Settings


class EmbeddingError(Exception):
    pass


async def embed_texts(client: httpx.AsyncClient, texts: list[str], settings: Settings) -> list[list[float]]:
    if not texts:
        return []

    try:
        response = await client.post(
            f"{settings.ollama_base_url}/api/embed",
            json={"model": settings.embedding_model, "input": texts},
            timeout=settings.fetch_timeout_seconds,
        )
    except httpx.RequestError as exc:
        raise EmbeddingError(f"embedding request failed: {exc}") from exc

    if response.status_code != 200:
        raise EmbeddingError(f"embedding request returned status {response.status_code}")

    embeddings = response.json().get("embeddings")
    if not embeddings or len(embeddings) != len(texts):
        raise EmbeddingError("embedding response missing or mismatched vectors")
    return embeddings
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/indexing/test_embeddings.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add app/indexing/embeddings.py tests/indexing/test_embeddings.py
git commit -m "feat: add Ollama embeddings client"
```

---

### Task 4: BM25 index

**Files:**
- Create: `app/indexing/bm25_index.py`
- Test: `tests/indexing/test_bm25_index.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `tokenize(text: str) -> list[str]`; `class InMemoryBM25Index` with `add_documents(chunk_ids: list[str], texts: list[str]) -> None`, `remove_documents(chunk_ids: set[str]) -> None`, `search(query: str, top_k: int) -> list[tuple[str, float]]`.

- [ ] **Step 1: Add rank-bm25 dependency**

```bash
uv add rank-bm25
```

- [ ] **Step 2: Write the failing test**

```python
# tests/indexing/test_bm25_index.py
from app.indexing.bm25_index import InMemoryBM25Index


def test_search_returns_best_matching_chunk_first():
    index = InMemoryBM25Index()
    index.add_documents(
        ["c1", "c2", "c3"],
        [
            "the quick brown fox jumps over the lazy dog",
            "completely unrelated text about kubernetes deployments",
            "another fox story, a second fox appears here",
        ],
    )
    results = index.search("fox", top_k=2)
    result_ids = [chunk_id for chunk_id, _score in results]
    assert result_ids[0] == "c3"  # two mentions of "fox" outranks one
    assert "c2" not in result_ids


def test_search_on_empty_index_returns_empty():
    index = InMemoryBM25Index()
    assert index.search("anything", top_k=5) == []


def test_remove_documents_excludes_them_from_future_searches():
    index = InMemoryBM25Index()
    index.add_documents(["c1", "c2"], ["fox fox fox", "fox"])
    index.remove_documents({"c1"})
    results = index.search("fox", top_k=5)
    assert [chunk_id for chunk_id, _score in results] == ["c2"]
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/indexing/test_bm25_index.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.indexing.bm25_index'`

- [ ] **Step 4: Implement BM25 index**

```python
# app/indexing/bm25_index.py
from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class InMemoryBM25Index:
    def __init__(self) -> None:
        self._chunk_ids: list[str] = []
        self._tokenized_docs: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    def add_documents(self, chunk_ids: list[str], texts: list[str]) -> None:
        for chunk_id, text in zip(chunk_ids, texts):
            self._chunk_ids.append(chunk_id)
            self._tokenized_docs.append(tokenize(text))
        self._rebuild()

    def remove_documents(self, chunk_ids: set[str]) -> None:
        keep = [i for i, cid in enumerate(self._chunk_ids) if cid not in chunk_ids]
        self._chunk_ids = [self._chunk_ids[i] for i in keep]
        self._tokenized_docs = [self._tokenized_docs[i] for i in keep]
        self._rebuild()

    def _rebuild(self) -> None:
        self._bm25 = BM25Okapi(self._tokenized_docs) if self._tokenized_docs else None

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self._chunk_ids, scores), key=lambda pair: pair[1], reverse=True)
        return [(chunk_id, float(score)) for chunk_id, score in ranked[:top_k] if score > 0]
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/indexing/test_bm25_index.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock app/indexing/bm25_index.py tests/indexing/test_bm25_index.py
git commit -m "feat: add in-memory BM25 index"
```

---

### Task 5: Vector index (Qdrant)

**Files:**
- Create: `app/indexing/vector_index.py`
- Test: `tests/indexing/test_vector_index.py`

**Interfaces:**
- Consumes: `Settings` (`qdrant_url`, `qdrant_path`, `qdrant_collection`, `vector_size`).
- Produces: `class QdrantVectorIndex` with `__init__(settings: Settings)`, `upsert(chunk_ids: list[str], vectors: list[list[float]], payloads: list[dict]) -> None`, `delete(chunk_ids: list[str]) -> None`, `search(query_vector: list[float], top_k: int) -> list[tuple[str, float]]`.

**Note for the implementer:** point IDs must be valid Qdrant IDs (unsigned int or UUID string). This plan always generates `chunk_id` as `str(uuid.uuid4())` (Task 7), so `chunk_id` is used directly as the Qdrant point ID — no separate ID mapping needed.

- [ ] **Step 1: Write the failing test**

```python
# tests/indexing/test_vector_index.py
import uuid

from app.config import Settings
from app.indexing.vector_index import QdrantVectorIndex


def _settings(tmp_path) -> Settings:
    return Settings(qdrant_path=str(tmp_path / "qdrant"), qdrant_collection="test_chunks", vector_size=4)


def test_upsert_and_search_returns_closest_match(tmp_path):
    index = QdrantVectorIndex(_settings(tmp_path))
    id_a, id_b = str(uuid.uuid4()), str(uuid.uuid4())
    index.upsert(
        [id_a, id_b],
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        [{"chunk_id": id_a, "text": "a"}, {"chunk_id": id_b, "text": "b"}],
    )
    results = index.search([1.0, 0.0, 0.0, 0.0], top_k=1)
    assert results[0][0] == id_a


def test_delete_removes_point_from_future_searches(tmp_path):
    index = QdrantVectorIndex(_settings(tmp_path))
    id_a, id_b = str(uuid.uuid4()), str(uuid.uuid4())
    index.upsert(
        [id_a, id_b],
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        [{"chunk_id": id_a}, {"chunk_id": id_b}],
    )
    index.delete([id_a])
    results = index.search([1.0, 0.0, 0.0, 0.0], top_k=5)
    result_ids = [chunk_id for chunk_id, _score in results]
    assert id_a not in result_ids
    assert id_b in result_ids
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/indexing/test_vector_index.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.indexing.vector_index'`

- [ ] **Step 3: Implement vector index**

```python
# app/indexing/vector_index.py
from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import Settings


class QdrantVectorIndex:
    def __init__(self, settings: Settings) -> None:
        if settings.qdrant_url:
            self._client = QdrantClient(url=settings.qdrant_url)
        else:
            self._client = QdrantClient(path=settings.qdrant_path)
        self._collection = settings.qdrant_collection
        self._ensure_collection(settings.vector_size)

    def _ensure_collection(self, vector_size: int) -> None:
        existing = [c.name for c in self._client.get_collections().collections]
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
            )

    def upsert(self, chunk_ids: list[str], vectors: list[list[float]], payloads: list[dict]) -> None:
        points = [
            qmodels.PointStruct(id=chunk_id, vector=vector, payload=payload)
            for chunk_id, vector, payload in zip(chunk_ids, vectors, payloads)
        ]
        self._client.upsert(collection_name=self._collection, points=points)

    def delete(self, chunk_ids: list[str]) -> None:
        self._client.delete(
            collection_name=self._collection,
            points_selector=qmodels.PointIdsList(points=chunk_ids),
        )

    def search(self, query_vector: list[float], top_k: int) -> list[tuple[str, float]]:
        results = self._client.query_points(
            collection_name=self._collection, query=query_vector, limit=top_k
        ).points
        return [(str(point.id), point.score) for point in results]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/indexing/test_vector_index.py -v`
Expected: PASS (2 tests). Uses Qdrant's local on-disk mode — no running Qdrant server needed for this test.

- [ ] **Step 5: Commit**

```bash
git add app/indexing/vector_index.py tests/indexing/test_vector_index.py
git commit -m "feat: add Qdrant vector index wrapper"
```

---

### Task 6: Chunk store (document store)

**Files:**
- Create: `app/indexing/chunk_store.py`
- Test: `tests/indexing/test_chunk_store.py`

**Interfaces:**
- Consumes: `ChunkMetadata` from `app.indexing.models`.
- Produces: `class ChunkStore` with `add(chunks: list[ChunkMetadata]) -> None`, `remove(chunk_ids: list[str]) -> None`, `get(chunk_id: str) -> ChunkMetadata | None`, `get_many(chunk_ids: list[str]) -> list[ChunkMetadata]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/indexing/test_chunk_store.py
from app.indexing.chunk_store import ChunkStore
from app.indexing.models import ChunkMetadata


def _chunk(chunk_id: str) -> ChunkMetadata:
    return ChunkMetadata(
        chunk_id=chunk_id,
        doc_id="d1",
        source_url="https://example.com",
        page_number=1,
        chunk_index=0,
        char_start=0,
        char_end=10,
        overlap_with_prev=0,
        indexed_at="2026-07-29T12:00:00+00:00",
        text="hello",
    )


def test_add_and_get():
    store = ChunkStore()
    store.add([_chunk("c1"), _chunk("c2")])
    assert store.get("c1").chunk_id == "c1"
    assert store.get("does-not-exist") is None


def test_get_many_preserves_requested_order_and_skips_missing():
    store = ChunkStore()
    store.add([_chunk("c1"), _chunk("c2")])
    result = store.get_many(["c2", "missing", "c1"])
    assert [c.chunk_id for c in result] == ["c2", "c1"]


def test_remove_deletes_chunks():
    store = ChunkStore()
    store.add([_chunk("c1"), _chunk("c2")])
    store.remove(["c1"])
    assert store.get("c1") is None
    assert store.get("c2") is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/indexing/test_chunk_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.indexing.chunk_store'`

- [ ] **Step 3: Implement chunk store**

```python
# app/indexing/chunk_store.py
from __future__ import annotations

from app.indexing.models import ChunkMetadata


class ChunkStore:
    def __init__(self) -> None:
        self._chunks: dict[str, ChunkMetadata] = {}

    def add(self, chunks: list[ChunkMetadata]) -> None:
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk

    def remove(self, chunk_ids: list[str]) -> None:
        for chunk_id in chunk_ids:
            self._chunks.pop(chunk_id, None)

    def get(self, chunk_id: str) -> ChunkMetadata | None:
        return self._chunks.get(chunk_id)

    def get_many(self, chunk_ids: list[str]) -> list[ChunkMetadata]:
        return [self._chunks[chunk_id] for chunk_id in chunk_ids if chunk_id in self._chunks]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/indexing/test_chunk_store.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add app/indexing/chunk_store.py tests/indexing/test_chunk_store.py
git commit -m "feat: add in-memory chunk store (document store)"
```

---

### Task 7: Indexer orchestrator

**Files:**
- Create: `app/indexing/indexer.py`
- Test: `tests/indexing/test_indexer.py`

**Interfaces:**
- Consumes:
  - `chunk_text(text, chunk_size, overlap) -> list[TextChunk]` from `app.indexing.chunker`
  - `embed_texts(client, texts, settings) -> list[list[float]]` from `app.indexing.embeddings`
  - `InMemoryBM25Index` from `app.indexing.bm25_index`
  - `QdrantVectorIndex` from `app.indexing.vector_index`
  - `ChunkStore` from `app.indexing.chunk_store`
  - `ChunkMetadata`, `IndexResult` from `app.indexing.models`
  - `IngestionPayload`, `PageMapEntry` from `app.ingestion.models` (Plan 1)
  - `Settings`, `settings` from `app.config`
- Produces: `async def index_document(payload: IngestionPayload, bm25_index: InMemoryBM25Index, vector_index: QdrantVectorIndex, chunk_store: ChunkStore, http_client: httpx.AsyncClient, settings: Settings = settings) -> IndexResult`

- [ ] **Step 1: Write the failing test**

```python
# tests/indexing/test_indexer.py
import httpx
import pytest

from app.config import Settings
from app.ingestion.models import IngestionPayload, PageMapEntry
from app.indexing.bm25_index import InMemoryBM25Index
from app.indexing.chunk_store import ChunkStore
from app.indexing.indexer import index_document
from app.indexing.vector_index import QdrantVectorIndex


def _settings(tmp_path) -> Settings:
    return Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="test_chunks",
        vector_size=3,
        chunk_size_tokens=6,
        chunk_overlap_tokens=0,
    )


def _payload() -> IngestionPayload:
    text = "one two three. four five six."
    return IngestionPayload(
        source_url="https://example.com/post",
        cleaned_text=text,
        pages_fetched=1,
        fetched_at="2026-07-29T12:00:00+00:00",
        page_map=[PageMapEntry(page=1, char_start=0, char_end=len(text))],
    )


async def test_index_document_writes_to_both_indexes_and_chunk_store(tmp_path):
    settings = _settings(tmp_path)
    bm25 = InMemoryBM25Index()
    vectors = QdrantVectorIndex(settings)
    store = ChunkStore()

    def handler(request):
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await index_document(_payload(), bm25, vectors, store, client, settings)

    assert result.status == "indexed"
    assert result.chunk_count == 2
    assert len(bm25.search("one two three", top_k=5)) >= 1
    assert len(vectors.search([0.1, 0.2, 0.3], top_k=5)) == 2
    stored = store.get_many([c.chunk_id for c in store._chunks.values()])
    assert len(stored) == 2
    assert all(c.doc_id == result.doc_id for c in stored)
    assert stored[0].page_number == 1


async def test_index_document_rolls_back_bm25_on_vector_failure(tmp_path):
    settings = _settings(tmp_path)
    bm25 = InMemoryBM25Index()
    vectors = QdrantVectorIndex(settings)
    store = ChunkStore()

    def handler(request):
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2], [0.4, 0.5]]})  # wrong vector_size (3 expected)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(Exception):
            await index_document(_payload(), bm25, vectors, store, client, settings)

    assert bm25.search("one two three", top_k=5) == []
    assert len(store._chunks) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/indexing/test_indexer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.indexing.indexer'`

- [ ] **Step 3: Implement indexer**

```python
# app/indexing/indexer.py
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import httpx

from app.config import Settings, settings as default_settings
from app.indexing.bm25_index import InMemoryBM25Index
from app.indexing.chunk_store import ChunkStore
from app.indexing.chunker import chunk_text
from app.indexing.embeddings import embed_texts
from app.indexing.models import ChunkMetadata, IndexResult
from app.indexing.vector_index import QdrantVectorIndex
from app.ingestion.models import IngestionPayload, PageMapEntry


def _resolve_page_number(page_map: list[PageMapEntry], char_start: int) -> int:
    for entry in page_map:
        if entry.char_start <= char_start < entry.char_end:
            return entry.page
    return page_map[-1].page if page_map else 1


async def index_document(
    payload: IngestionPayload,
    bm25_index: InMemoryBM25Index,
    vector_index: QdrantVectorIndex,
    chunk_store: ChunkStore,
    http_client: httpx.AsyncClient,
    settings: Settings = default_settings,
) -> IndexResult:
    text_chunks = chunk_text(payload.cleaned_text, settings.chunk_size_tokens, settings.chunk_overlap_tokens)
    doc_id = str(uuid.uuid4())
    indexed_at = datetime.now(timezone.utc).isoformat()

    chunk_metadatas = [
        ChunkMetadata(
            chunk_id=str(uuid.uuid4()),
            doc_id=doc_id,
            source_url=payload.source_url,
            page_number=_resolve_page_number(payload.page_map, text_chunk.char_start),
            chunk_index=index,
            char_start=text_chunk.char_start,
            char_end=text_chunk.char_end,
            overlap_with_prev=text_chunk.overlap_with_prev,
            indexed_at=indexed_at,
            text=text_chunk.text,
        )
        for index, text_chunk in enumerate(text_chunks)
    ]

    if not chunk_metadatas:
        return IndexResult(doc_id=doc_id, status="indexed", chunk_count=0)

    chunk_ids = [chunk.chunk_id for chunk in chunk_metadatas]
    texts = [chunk.text for chunk in chunk_metadatas]
    vectors = await embed_texts(http_client, texts, settings)

    bm25_index.add_documents(chunk_ids, texts)
    try:
        vector_index.upsert(chunk_ids, vectors, [chunk.model_dump() for chunk in chunk_metadatas])
    except Exception:
        bm25_index.remove_documents(set(chunk_ids))
        raise

    chunk_store.add(chunk_metadatas)

    return IndexResult(doc_id=doc_id, status="indexed", chunk_count=len(chunk_metadatas))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/indexing/test_indexer.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add app/indexing/indexer.py tests/indexing/test_indexer.py
git commit -m "feat: add indexer orchestrator with dual-write rollback"
```

---

### Task 8: Router + wiring into the ingestion worker

**Files:**
- Modify: `app/ingestion/worker.py` (rename `_noop_sink` → `noop_sink`, public)
- Modify: `app/ingestion/router.py` (`build_ingestion_router` gains a `sink` param, passed through to `ingest_url`)
- Create: `app/indexing/router.py`
- Modify: `app/main.py` (build indexing components, mount `/index/chunk`, wire `index_document` as the ingestion sink)
- Test: `tests/indexing/test_router.py`
- Test: `tests/indexing/test_wiring.py`

**Interfaces:**
- Consumes: everything from Tasks 1–7, plus `build_ingestion_router` from `app.ingestion.router` (Plan 1).
- Produces: `def build_indexing_router(bm25_index, vector_index, chunk_store, http_client, settings=settings) -> APIRouter` exposing `POST /index/chunk`; `create_app()` in `app/main.py` now indexes automatically on ingest completion.

- [ ] **Step 1: Rename the ingestion sink default to public and thread it through the router**

In `app/ingestion/worker.py`, rename `_noop_sink` to `noop_sink` everywhere it appears (the function definition and the `ingest_url` default parameter value).

In `app/ingestion/router.py`, change the import and the endpoint body:

```python
# app/ingestion/router.py — full file after edit
from __future__ import annotations

import asyncio
from typing import Callable

import httpx
from fastapi import APIRouter, HTTPException

from app.ingestion.job_store import JobStore
from app.ingestion.models import IngestRequest, IngestResponse, JobStatusResponse
from app.ingestion.worker import IngestSink, ingest_url, noop_sink


def build_ingestion_router(
    store: JobStore,
    client_factory: Callable[[], httpx.AsyncClient] = httpx.AsyncClient,
    sink: IngestSink = noop_sink,
) -> APIRouter:
    router = APIRouter()

    @router.post("/ingest", response_model=IngestResponse)
    async def create_ingest_job(request: IngestRequest) -> IngestResponse:
        job_id = store.create_job(request.urls)
        client = client_factory()

        async def run_all() -> None:
            try:
                await asyncio.gather(
                    *(ingest_url(job_id, url, store, client, sink=sink) for url in request.urls)
                )
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

- [ ] **Step 2: Run the existing ingestion suite to confirm the rename didn't break anything**

Run: `uv run pytest tests/ingestion/ -v`
Expected: PASS, same counts as before (this step is a refactor check, not new coverage)

- [ ] **Step 3: Write the failing tests for the indexing router and the wiring**

```python
# tests/indexing/test_router.py
import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import Settings
from app.indexing.bm25_index import InMemoryBM25Index
from app.indexing.chunk_store import ChunkStore
from app.indexing.router import build_indexing_router
from app.indexing.vector_index import QdrantVectorIndex
from app.ingestion.models import PageMapEntry


def test_post_index_chunk_returns_result_and_populates_stores(tmp_path):
    settings = Settings(qdrant_path=str(tmp_path / "qdrant"), qdrant_collection="t", vector_size=2)
    bm25 = InMemoryBM25Index()
    vectors = QdrantVectorIndex(settings)
    store = ChunkStore()

    def handler(request):
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    app = FastAPI()
    app.include_router(build_indexing_router(bm25, vectors, store, http_client, settings))

    body = {
        "source_url": "https://example.com",
        "cleaned_text": "just one short sentence here.",
        "pages_fetched": 1,
        "fetched_at": "2026-07-29T12:00:00+00:00",
        "page_map": [{"page": 1, "char_start": 0, "char_end": 30}],
    }
    with TestClient(app) as client:
        response = client.post("/index/chunk", json=body)

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "indexed"
    assert result["chunk_count"] == 1
```

```python
# tests/indexing/test_wiring.py
import asyncio

import httpx
from fastapi import FastAPI

from app.config import Settings
from app.indexing.bm25_index import InMemoryBM25Index
from app.indexing.chunk_store import ChunkStore
from app.indexing.indexer import index_document
from app.indexing.vector_index import QdrantVectorIndex
from app.ingestion.job_store import JobStore
from app.ingestion.router import build_ingestion_router

GOOD_PAGE = "<html><body><article><p>Enough real content here to clear the extraction threshold easily.</p></article></body></html>"


def _fetch_handler(request: httpx.Request) -> httpx.Response:
    if str(request.url) == "https://example.com/article":
        return httpx.Response(200, text=GOOD_PAGE)
    return httpx.Response(404, text="not found")


async def test_ingest_then_index_populates_chunk_store(tmp_path):
    settings = Settings(qdrant_path=str(tmp_path / "qdrant"), qdrant_collection="wiring", vector_size=2)
    bm25 = InMemoryBM25Index()
    vectors = QdrantVectorIndex(settings)
    store = ChunkStore()

    def embed_handler(request):
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})

    embed_client = httpx.AsyncClient(transport=httpx.MockTransport(embed_handler))

    async def sink(payload):
        await index_document(payload, bm25, vectors, store, embed_client, settings)

    job_store = JobStore()
    app = FastAPI()
    app.include_router(
        build_ingestion_router(
            job_store,
            client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(_fetch_handler)),
            sink=sink,
        )
    )

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/ingest", json={"urls": ["https://example.com/article"]})
        job_id = response.json()["job_id"]

        elapsed = 0.0
        while elapsed < 2.0:
            status = (await client.get(f"/ingest/{job_id}/status")).json()
            if status["urls"][0]["stage"] in ("done", "error"):
                break
            await asyncio.sleep(0.01)
            elapsed += 0.01

    assert status["urls"][0]["stage"] == "done"
    assert len(store._chunks) >= 1
    await embed_client.aclose()
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `uv run pytest tests/indexing/test_router.py tests/indexing/test_wiring.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.indexing.router'`

- [ ] **Step 5: Implement the indexing router and wire main.py**

```python
# app/indexing/router.py
from __future__ import annotations

import httpx
from fastapi import APIRouter

from app.config import Settings, settings as default_settings
from app.indexing.bm25_index import InMemoryBM25Index
from app.indexing.chunk_store import ChunkStore
from app.indexing.indexer import index_document
from app.indexing.models import IndexResult
from app.indexing.vector_index import QdrantVectorIndex
from app.ingestion.models import IngestionPayload


def build_indexing_router(
    bm25_index: InMemoryBM25Index,
    vector_index: QdrantVectorIndex,
    chunk_store: ChunkStore,
    http_client: httpx.AsyncClient,
    settings: Settings = default_settings,
) -> APIRouter:
    router = APIRouter()

    @router.post("/index/chunk", response_model=IndexResult)
    async def index_chunk(payload: IngestionPayload) -> IndexResult:
        return await index_document(payload, bm25_index, vector_index, chunk_store, http_client, settings)

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


def create_app() -> FastAPI:
    app = FastAPI(title="RAG Ingestion Service")

    job_store = JobStore()
    bm25_index = InMemoryBM25Index()
    vector_index = QdrantVectorIndex(settings)
    chunk_store = ChunkStore()
    embedding_client = httpx.AsyncClient()

    async def index_sink(payload) -> None:
        await index_document(payload, bm25_index, vector_index, chunk_store, embedding_client, settings)

    app.include_router(build_ingestion_router(job_store, sink=index_sink))
    app.include_router(build_indexing_router(bm25_index, vector_index, chunk_store, embedding_client))

    return app


app = create_app()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/indexing/test_router.py tests/indexing/test_wiring.py -v`
Expected: PASS (1 + 1 tests)

- [ ] **Step 7: Run the full test suite**

Run: `uv run pytest -v`
Expected: all tests across both plans PASS

- [ ] **Step 8: Commit**

```bash
git add app/ingestion/worker.py app/ingestion/router.py app/indexing/router.py app/main.py tests/indexing/test_router.py tests/indexing/test_wiring.py
git commit -m "feat: wire indexer as the ingestion worker's sink, expose POST /index/chunk"
```

---

## Self-Review Notes

- **Spec coverage:** Chunking (token-based, sentence-boundary-respecting, configurable size/overlap) — Task 2. Metadata assembly with the exact fixed schema — Task 1 model + Task 7 assembly. Dual write with rollback — Task 7. BM25/vector store choices — Tasks 4, 5. `POST /index/chunk` contract — Task 8. Cross-plan wiring so ingestion actually triggers indexing (closing the loop Plan 1 left as a no-op sink) — Task 8.
- **Type consistency checked:** `ChunkMetadata`/`IndexResult` defined once in Task 1, used unchanged through Tasks 6–8. `TextChunk` defined in Task 2, consumed unchanged in Task 7. `index_document`'s signature (defined Task 7) is called identically by Task 8's router and `main.py` wiring.
- **Cross-plan touch:** Task 8 modifies two Plan 1 files (`app/ingestion/worker.py`, `app/ingestion/router.py`) to make the sink injectable — this was Plan 1's intentional extension point (`IngestSink`), not a violation of Plan 1's scope.
- **No placeholders:** every step has runnable code; no "TBD" left.
