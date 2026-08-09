# Pipeline Architecture (Strategy Registry + Local-Archive Indexing) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the app's single hardcoded RAG pipeline with a `PipelineConfig`-driven architecture where Indexing, Retrieval, and Post-retrieval each pick from a registry of predefined strategies, indexing reads PDFs from a local `archive/` folder into per-strategy Qdrant collections instead of crawling URLs, and `/query` always runs whichever pipeline was last `POST /pipeline/load`ed.

**Architecture:** Three new strategy registries (`app/indexing/strategies/`, `app/retrieval/strategies/`, `app/postretrieval/strategies/`) expose a uniform callable per strategy id. A new `app/pipeline/` package owns `PipelineConfig`, a single shared `ACTIVE_PIPELINE` module-level value, an `IndexingCollectionRegistry` (one Qdrant collection + BM25 index per indexing strategy, sharing one on-disk `QdrantClient`), and the `load_pipeline()` orchestration that scans `archive/`, indexes only unseen PDFs (by content hash) into the target strategy's collection, and activates the config. `/query` reads `ACTIVE_PIPELINE` and dispatches to the active strategies. URL-crawl ingestion and the context-quality Judge are deleted outright.

**Tech Stack:** FastAPI, Pydantic, `qdrant-client` (local path storage), `rank_bm25`, `httpx`, `pytest`/`pytest-asyncio`, new dependency: `pypdf`.

## Global Constraints

- Python ≥3.11, existing test convention: `tests/<package>/test_<module>.py` mirroring `app/`, `httpx.MockTransport` for external HTTP calls, `asyncio_mode = auto` (no `@pytest.mark.asyncio` needed), one isolated `tmp_path`-scoped `qdrant_path` + unique `qdrant_collection` per test that touches Qdrant.
- `Settings` (`app/config.py`) is a frozen stdlib `@dataclass` — use `dataclasses.replace(settings, ...)` to derive variants, never mutate.
- `ACTIVE_PIPELINE` is process-global mutable state (single shared corpus, no auth, matches existing app posture) — any test that calls `set_active()` MUST reset it to `None` afterward (autouse fixture), or later tests in the same process see a stale active pipeline.
- Local-mode `QdrantClient(path=...)` takes an exclusive lock on its storage directory — never construct two independent `QdrantClient` instances against the same `qdrant_path` at once; share one client across collections (see Task 1).
- No backwards-compatibility shims: when a module/endpoint is replaced (retriever.py, filtering.py, judge.py, all of `app/ingestion/`), delete it and its tests in the same task that replaces it — do not leave dead code importable "just in case."
- Run `uv run pytest` (or the project's configured pytest invocation) after every task; the full suite must be green before moving to the next task.

---

### Task 1: `QdrantVectorIndex` — shared client, `scroll_all()`, `close()`

**Files:**
- Modify: `app/indexing/vector_index.py`
- Test: `tests/indexing/test_vector_index.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `QdrantVectorIndex(settings, client: QdrantClient | None = None)` — when `client` is given, it's reused instead of opening a new one. `scroll_all() -> list[dict]` (all payloads in the collection). `close() -> None`. Task 8's `IndexingCollectionRegistry` depends on all three.

- [ ] **Step 1: Write the failing tests**

Append to `tests/indexing/test_vector_index.py`:

```python
def test_scroll_all_returns_all_payloads(tmp_path):
    index = QdrantVectorIndex(_settings(tmp_path))
    id_a, id_b = str(uuid.uuid4()), str(uuid.uuid4())
    index.upsert(
        [id_a, id_b],
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        [{"chunk_id": id_a, "text": "a"}, {"chunk_id": id_b, "text": "b"}],
    )
    payloads = index.scroll_all()
    assert {p["chunk_id"] for p in payloads} == {id_a, id_b}


def test_scroll_all_empty_collection_returns_empty_list(tmp_path):
    index = QdrantVectorIndex(_settings(tmp_path))
    assert index.scroll_all() == []


def test_shares_provided_client_instead_of_creating_its_own(tmp_path):
    from qdrant_client import QdrantClient

    settings = _settings(tmp_path)
    shared_client = QdrantClient(path=settings.qdrant_path)
    index_a = QdrantVectorIndex(settings, client=shared_client)
    index_b = QdrantVectorIndex(
        Settings(qdrant_path=settings.qdrant_path, qdrant_collection="other", vector_size=4),
        client=shared_client,
    )
    id_a = str(uuid.uuid4())
    index_a.upsert([id_a], [[1.0, 0.0, 0.0, 0.0]], [{"chunk_id": id_a}])
    assert index_b.scroll_all() == []  # different collection, same client, no cross-contamination
    shared_client.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/indexing/test_vector_index.py -v`
Expected: the three new tests FAIL (`AttributeError: 'QdrantVectorIndex' object has no attribute 'scroll_all'`, and the `client=` kwarg is rejected).

- [ ] **Step 3: Implement**

Replace `app/indexing/vector_index.py` with:

```python
from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import Settings


class QdrantVectorIndex:
    def __init__(self, settings: Settings, client: QdrantClient | None = None) -> None:
        if client is not None:
            self._client = client
        elif settings.qdrant_url:
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

    def scroll_all(self) -> list[dict]:
        payloads: list[dict] = []
        offset = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self._collection, limit=256, with_payload=True, with_vectors=False, offset=offset
            )
            payloads.extend(point.payload for point in points)
            if offset is None:
                break
        return payloads

    def close(self) -> None:
        self._client.close()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/indexing/test_vector_index.py -v`
Expected: all PASS (existing two tests plus the three new ones).

- [ ] **Step 5: Commit**

```bash
git add app/indexing/vector_index.py tests/indexing/test_vector_index.py
git commit -m "feat: allow QdrantVectorIndex to share a client, add scroll_all/close"
```

---

### Task 2: `ChunkMetadata.doc_id_hash` + `ChunkStore.doc_id_hashes()`

**Files:**
- Modify: `app/indexing/models.py`, `app/indexing/chunk_store.py`
- Test: `tests/indexing/test_models.py`, `tests/indexing/test_chunk_store.py`

**Interfaces:**
- Produces: `ChunkMetadata.doc_id_hash: str = ""` (stable content-hash identity, independent of `chunk_id`/`doc_id` which are regenerated every index run). `ChunkStore.doc_id_hashes() -> set[str]`. Task 9's loader diffs against this set to decide which archive docs are new.

- [ ] **Step 1: Write the failing tests**

Append to `tests/indexing/test_models.py`:

```python
def test_chunk_metadata_defaults_doc_id_hash_to_empty_string():
    chunk = ChunkMetadata(
        chunk_id="c1", doc_id="d1", source_url="https://example.com", page_number=1,
        chunk_index=0, char_start=0, char_end=100, overlap_with_prev=0,
        indexed_at="2026-07-29T12:00:00+00:00", text="hello world",
    )
    assert chunk.doc_id_hash == ""


def test_chunk_metadata_round_trip_carries_doc_id_hash():
    chunk = ChunkMetadata(
        chunk_id="c1", doc_id="d1", doc_id_hash="abc123", source_url="https://example.com", page_number=1,
        chunk_index=0, char_start=0, char_end=100, overlap_with_prev=0,
        indexed_at="2026-07-29T12:00:00+00:00", text="hello world",
    )
    assert chunk.model_dump()["doc_id_hash"] == "abc123"
```

Append to `tests/indexing/test_chunk_store.py` (read the existing file first to match its `_chunk` helper style before appending — if it defines a local factory helper, reuse it and just add `doc_id_hash=...` where needed):

```python
def test_doc_id_hashes_returns_distinct_hashes_across_stored_chunks():
    store = ChunkStore()
    store.add([
        ChunkMetadata(
            chunk_id="c1", doc_id="d1", doc_id_hash="h1", source_url="a.pdf", page_number=1,
            chunk_index=0, char_start=0, char_end=10, overlap_with_prev=0,
            indexed_at="2026-08-09T00:00:00+00:00", text="one",
        ),
        ChunkMetadata(
            chunk_id="c2", doc_id="d1", doc_id_hash="h1", source_url="a.pdf", page_number=1,
            chunk_index=1, char_start=10, char_end=20, overlap_with_prev=0,
            indexed_at="2026-08-09T00:00:00+00:00", text="two",
        ),
        ChunkMetadata(
            chunk_id="c3", doc_id="d2", doc_id_hash="h2", source_url="b.pdf", page_number=1,
            chunk_index=0, char_start=0, char_end=10, overlap_with_prev=0,
            indexed_at="2026-08-09T00:00:00+00:00", text="three",
        ),
    ])
    assert store.doc_id_hashes() == {"h1", "h2"}


def test_doc_id_hashes_empty_store_returns_empty_set():
    assert ChunkStore().doc_id_hashes() == set()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/indexing/test_models.py tests/indexing/test_chunk_store.py -v`
Expected: FAIL (`doc_id_hash` field doesn't exist; `doc_id_hashes` method doesn't exist).

- [ ] **Step 3: Implement**

In `app/indexing/models.py`, add the field to `ChunkMetadata` (after `doc_id`):

```python
class ChunkMetadata(BaseModel):
    chunk_id: str
    doc_id: str
    doc_id_hash: str = ""
    source_url: str
    page_number: int
    chunk_index: int
    char_start: int
    char_end: int
    overlap_with_prev: int
    indexed_at: str
    text: str
    city: str | None = None
    price: float | None = None
```

In `app/indexing/chunk_store.py`, add:

```python
    def doc_id_hashes(self) -> set[str]:
        return {chunk.doc_id_hash for chunk in self._chunks.values()}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/indexing/test_models.py tests/indexing/test_chunk_store.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the full suite to confirm no regressions**

Run: `uv run pytest -q`
Expected: all PASS (the default `doc_id_hash = ""` keeps every existing `ChunkMetadata(...)` call site, which doesn't pass the field, working unchanged).

- [ ] **Step 6: Commit**

```bash
git add app/indexing/models.py app/indexing/chunk_store.py tests/indexing/test_models.py tests/indexing/test_chunk_store.py
git commit -m "feat: add doc_id_hash identity field and ChunkStore.doc_id_hashes()"
```

---

### Task 3: Archive scanning — `scan_archive()` + `extract_pdf_text()`

**Files:**
- Create: `app/archive/__init__.py` (empty), `app/archive/scanner.py`, `app/archive/pdf_extractor.py`
- Create: `tests/archive/__init__.py` (empty), `tests/archive/test_scanner.py`, `tests/archive/test_pdf_extractor.py`
- Modify: `pyproject.toml` (new dependency)

**Interfaces:**
- Produces: `ArchiveDoc(doc_id_hash: str, path: str, filename: str)`; `scan_archive(archive_dir: Path = Path("archive")) -> list[ArchiveDoc]`; `extract_pdf_text(path: Path) -> str`; `PdfExtractionError`. Task 9's loader consumes all four.

- [ ] **Step 1: Add the `pypdf` dependency**

Run: `uv add pypdf`
Expected: `pyproject.toml`'s `dependencies` list gains a `pypdf>=...` entry and `uv.lock` updates.

- [ ] **Step 2: Write the failing scanner test**

Create `tests/archive/__init__.py` (empty file) and `tests/archive/test_scanner.py`:

```python
from pathlib import Path

from app.archive.scanner import scan_archive


def test_scan_archive_returns_empty_list_for_missing_directory(tmp_path):
    assert scan_archive(tmp_path / "does-not-exist") == []


def test_scan_archive_lists_pdfs_with_stable_hash(tmp_path):
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    (archive_dir / "a.pdf").write_bytes(b"pdf-content-a")
    (archive_dir / "b.pdf").write_bytes(b"pdf-content-b")
    (archive_dir / "notes.txt").write_bytes(b"ignore me")

    first_scan = scan_archive(archive_dir)
    second_scan = scan_archive(archive_dir)

    assert {doc.filename for doc in first_scan} == {"a.pdf", "b.pdf"}
    assert {doc.doc_id_hash for doc in first_scan} == {doc.doc_id_hash for doc in second_scan}
    assert len({doc.doc_id_hash for doc in first_scan}) == 2


def test_scan_archive_hash_changes_when_file_content_changes(tmp_path):
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    path = archive_dir / "a.pdf"
    path.write_bytes(b"version one")
    first_hash = scan_archive(archive_dir)[0].doc_id_hash
    path.write_bytes(b"version two")
    second_hash = scan_archive(archive_dir)[0].doc_id_hash
    assert first_hash != second_hash
```

- [ ] **Step 3: Run to verify it fails**

Run: `uv run pytest tests/archive/test_scanner.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'app.archive'`).

- [ ] **Step 4: Implement the scanner**

Create `app/archive/__init__.py` (empty).

Create `app/archive/scanner.py`:

```python
from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel


class ArchiveDoc(BaseModel):
    doc_id_hash: str
    path: str
    filename: str


def scan_archive(archive_dir: Path = Path("archive")) -> list[ArchiveDoc]:
    if not archive_dir.is_dir():
        return []
    docs = []
    for path in sorted(archive_dir.glob("*.pdf")):
        doc_id_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        docs.append(ArchiveDoc(doc_id_hash=doc_id_hash, path=str(path), filename=path.name))
    return docs
```

- [ ] **Step 5: Run to verify it passes**

Run: `uv run pytest tests/archive/test_scanner.py -v`
Expected: all PASS.

- [ ] **Step 6: Write the failing PDF-extraction test**

Create `tests/archive/test_pdf_extractor.py`:

```python
from pathlib import Path

import pytest

from app.archive.pdf_extractor import PdfExtractionError, extract_pdf_text


def test_extract_pdf_text_returns_nonempty_text_for_real_pdf():
    text = extract_pdf_text(Path("archive/LungPaper.pdf"))
    assert len(text.strip()) > 0


def test_extract_pdf_text_raises_on_corrupt_file(tmp_path):
    corrupt = tmp_path / "not_a_pdf.pdf"
    corrupt.write_bytes(b"this is not a valid pdf file at all")
    with pytest.raises(PdfExtractionError):
        extract_pdf_text(corrupt)
```

- [ ] **Step 7: Run to verify it fails**

Run: `uv run pytest tests/archive/test_pdf_extractor.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'app.archive.pdf_extractor'`).

- [ ] **Step 8: Implement PDF extraction**

Create `app/archive/pdf_extractor.py`:

```python
from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


class PdfExtractionError(Exception):
    pass


def extract_pdf_text(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:
        raise PdfExtractionError(f"failed to extract text from {path}: {exc}") from exc

    text = "\n\n".join(page for page in pages if page.strip())
    if not text.strip():
        raise PdfExtractionError(f"no extractable text in {path}")
    return text
```

- [ ] **Step 9: Run to verify it passes**

Run: `uv run pytest tests/archive/test_pdf_extractor.py -v`
Expected: all PASS.

- [ ] **Step 10: Commit**

```bash
git add app/archive tests/archive pyproject.toml uv.lock
git commit -m "feat: scan local archive/ folder and extract PDF text"
```

---

### Task 4: Indexing strategy registry

**Files:**
- Create: `app/indexing/strategies/__init__.py`, `app/indexing/strategies/fixed_window.py`, `app/indexing/strategies/semantic.py`, `app/indexing/strategies/hierarchical.py`, `app/indexing/strategies/hierarchical_summary.py`
- Create: `tests/indexing/strategies/__init__.py` (empty), `tests/indexing/strategies/test_registry.py`

**Interfaces:**
- Consumes: `app.indexing.chunker.TextChunk`, `chunk_text()` (unchanged, `app/indexing/chunker.py`).
- Produces: `INDEXING_STRATEGIES: dict[str, Callable[[str, Settings], list[TextChunk]]]` with keys `"fixed_window"`, `"semantic"`, `"hierarchical"`, `"hierarchical_summary"`. Task 9's loader looks up `INDEXING_STRATEGIES[config.indexing_strategy]`.

- [ ] **Step 1: Write the failing registry test**

Create `tests/indexing/strategies/__init__.py` (empty) and `tests/indexing/strategies/test_registry.py`:

```python
import pytest

from app.config import Settings
from app.indexing.chunker import chunk_text
from app.indexing.strategies import INDEXING_STRATEGIES


def test_registry_has_exactly_the_four_expected_strategies():
    assert set(INDEXING_STRATEGIES.keys()) == {
        "fixed_window", "semantic", "hierarchical", "hierarchical_summary",
    }


def test_fixed_window_matches_chunk_text_output():
    settings = Settings(chunk_size_tokens=6, chunk_overlap_tokens=0)
    text = "one two three. four five six."
    assert INDEXING_STRATEGIES["fixed_window"](text, settings) == chunk_text(text, 6, 0)


@pytest.mark.parametrize("strategy_id", ["semantic", "hierarchical", "hierarchical_summary"])
def test_stub_strategies_raise_not_implemented(strategy_id):
    with pytest.raises(NotImplementedError):
        INDEXING_STRATEGIES[strategy_id]("some text", Settings())
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/indexing/strategies/test_registry.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'app.indexing.strategies'`).

- [ ] **Step 3: Implement**

Create `app/indexing/strategies/fixed_window.py`:

```python
from __future__ import annotations

from app.config import Settings
from app.indexing.chunker import TextChunk, chunk_text


def chunk(text: str, settings: Settings) -> list[TextChunk]:
    return chunk_text(text, settings.chunk_size_tokens, settings.chunk_overlap_tokens)
```

Create `app/indexing/strategies/semantic.py`:

```python
from __future__ import annotations

from app.config import Settings
from app.indexing.chunker import TextChunk


def chunk(text: str, settings: Settings) -> list[TextChunk]:
    raise NotImplementedError("semantic chunking strategy not yet implemented — see piece B")
```

Create `app/indexing/strategies/hierarchical.py`:

```python
from __future__ import annotations

from app.config import Settings
from app.indexing.chunker import TextChunk


def chunk(text: str, settings: Settings) -> list[TextChunk]:
    raise NotImplementedError("hierarchical chunking strategy not yet implemented — see piece B")
```

Create `app/indexing/strategies/hierarchical_summary.py`:

```python
from __future__ import annotations

from app.config import Settings
from app.indexing.chunker import TextChunk


def chunk(text: str, settings: Settings) -> list[TextChunk]:
    raise NotImplementedError("hierarchical+summary chunking strategy not yet implemented — see piece B")
```

Create `app/indexing/strategies/__init__.py`:

```python
from __future__ import annotations

from typing import Callable

from app.config import Settings
from app.indexing.chunker import TextChunk
from app.indexing.strategies import fixed_window, hierarchical, hierarchical_summary, semantic

IndexingStrategyFn = Callable[[str, Settings], list[TextChunk]]

INDEXING_STRATEGIES: dict[str, IndexingStrategyFn] = {
    "fixed_window": fixed_window.chunk,
    "semantic": semantic.chunk,
    "hierarchical": hierarchical.chunk,
    "hierarchical_summary": hierarchical_summary.chunk,
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/indexing/strategies/test_registry.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add app/indexing/strategies tests/indexing/strategies
git commit -m "feat: add indexing strategy registry (fixed_window implemented, 3 stubs)"
```

---

### Task 5: Retrieval strategy registry

**Files:**
- Create: `app/retrieval/strategies/__init__.py`, `app/retrieval/strategies/hybrid_rrf.py`, `app/retrieval/strategies/bm25_only.py`, `app/retrieval/strategies/semantic_only.py`
- Create: `tests/retrieval/strategies/__init__.py` (empty), `tests/retrieval/strategies/test_hybrid_rrf.py`, `tests/retrieval/strategies/test_bm25_only.py`, `tests/retrieval/strategies/test_semantic_only.py`, `tests/retrieval/strategies/test_registry.py`
- Delete: `app/retrieval/retriever.py`, `tests/retrieval/test_retriever.py`

**Interfaces:**
- Consumes: `app.retrieval.fusion.{RankedHit, reciprocal_rank_fusion}` (unchanged), `app.indexing.embeddings.embed_texts` (unchanged).
- Produces: `RETRIEVAL_STRATEGIES: dict[str, RetrievalStrategyFn]` with keys `"bm25_only"`, `"semantic_only"`, `"hybrid_rrf"`, each `async def search(query, bm25_index, vector_index, chunk_store, http_client, settings=default_settings, top_k=None) -> list[FusedChunk]` — identical call signature to today's (now-deleted) `retriever.retrieve()`. Task 9's loader doesn't use this; Task 11's `/query` router does.

- [ ] **Step 1: Write the failing tests**

Create `tests/retrieval/strategies/__init__.py` (empty).

Create `tests/retrieval/strategies/test_hybrid_rrf.py` (ports `tests/retrieval/test_retriever.py`, same fixtures, new import/call site):

```python
import httpx

from app.config import Settings
from app.indexing.bm25_index import InMemoryBM25Index
from app.indexing.chunk_store import ChunkStore
from app.indexing.models import ChunkMetadata
from app.indexing.vector_index import QdrantVectorIndex
from app.retrieval.strategies.hybrid_rrf import search


def _seed(tmp_path):
    settings = Settings(qdrant_path=str(tmp_path / "qdrant"), qdrant_collection="t", vector_size=2)
    bm25 = InMemoryBM25Index()
    vectors = QdrantVectorIndex(settings)
    store = ChunkStore()

    chunk_id = "11111111-1111-1111-1111-111111111111"
    chunk = ChunkMetadata(
        chunk_id=chunk_id, doc_id="d1", source_url="https://example.com", page_number=1,
        chunk_index=0, char_start=0, char_end=20, overlap_with_prev=0,
        indexed_at="2026-07-29T12:00:00+00:00", text="the quick brown fox",
        city="paris", price=500.0,
    )
    bm25.add_documents([chunk_id], ["the quick brown fox"])
    vectors.upsert([chunk_id], [[1.0, 0.0]], [chunk.model_dump()])
    store.add([chunk])
    return settings, bm25, vectors, store, chunk_id


async def test_search_attaches_metadata_and_both_scores(tmp_path):
    settings, bm25, vectors, store, chunk_id = _seed(tmp_path)

    def handler(request):
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        results = await search("fox", bm25, vectors, store, client, settings)

    assert len(results) == 1
    chunk = results[0]
    assert chunk.chunk_id == chunk_id
    assert chunk.text == "the quick brown fox"
    assert chunk.matched_methods == ["bm25", "semantic"]
    assert chunk.bm25_rank == 1
    assert chunk.semantic_rank == 1
    assert chunk.used_in_synthesis is False


async def test_search_carries_city_and_price_into_fused_chunk(tmp_path):
    settings, bm25, vectors, store, chunk_id = _seed(tmp_path)

    def handler(request):
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        results = await search("fox", bm25, vectors, store, client, settings)

    assert results[0].city == "paris"
    assert results[0].price == 500.0


async def test_search_respects_top_k_override(tmp_path):
    settings, bm25, vectors, store, _chunk_id = _seed(tmp_path)

    def handler(request):
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        results = await search("fox", bm25, vectors, store, client, settings, top_k=0)

    assert results == []
```

Create `tests/retrieval/strategies/test_bm25_only.py`:

```python
import httpx

from app.config import Settings
from app.indexing.bm25_index import InMemoryBM25Index
from app.indexing.chunk_store import ChunkStore
from app.indexing.models import ChunkMetadata
from app.indexing.vector_index import QdrantVectorIndex
from app.retrieval.strategies.bm25_only import search


async def test_bm25_only_never_calls_embeddings_and_has_no_semantic_score(tmp_path):
    settings = Settings(qdrant_path=str(tmp_path / "qdrant"), qdrant_collection="t", vector_size=2)
    bm25 = InMemoryBM25Index()
    vectors = QdrantVectorIndex(settings)
    store = ChunkStore()

    chunk_id = "22222222-2222-2222-2222-222222222222"
    chunk = ChunkMetadata(
        chunk_id=chunk_id, doc_id="d1", source_url="https://example.com", page_number=1,
        chunk_index=0, char_start=0, char_end=20, overlap_with_prev=0,
        indexed_at="2026-07-29T12:00:00+00:00", text="the quick brown fox",
    )
    bm25.add_documents([chunk_id], ["the quick brown fox"])
    vectors.upsert([chunk_id], [[1.0, 0.0]], [chunk.model_dump()])
    store.add([chunk])

    def handler(request):
        raise AssertionError("bm25_only must not call the embedding endpoint")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        results = await search("fox", bm25, vectors, store, client, settings)

    assert len(results) == 1
    assert results[0].matched_methods == ["bm25"]
    assert results[0].semantic_rank is None
    assert results[0].semantic_score is None
```

Create `tests/retrieval/strategies/test_semantic_only.py`:

```python
import httpx

from app.config import Settings
from app.indexing.bm25_index import InMemoryBM25Index
from app.indexing.chunk_store import ChunkStore
from app.indexing.models import ChunkMetadata
from app.indexing.vector_index import QdrantVectorIndex
from app.retrieval.strategies.semantic_only import search


async def test_semantic_only_ignores_bm25_index_and_has_no_bm25_score(tmp_path):
    settings = Settings(qdrant_path=str(tmp_path / "qdrant"), qdrant_collection="t", vector_size=2)
    bm25 = InMemoryBM25Index()
    vectors = QdrantVectorIndex(settings)
    store = ChunkStore()

    chunk_id = "33333333-3333-3333-3333-333333333333"
    chunk = ChunkMetadata(
        chunk_id=chunk_id, doc_id="d1", source_url="https://example.com", page_number=1,
        chunk_index=0, char_start=0, char_end=20, overlap_with_prev=0,
        indexed_at="2026-07-29T12:00:00+00:00", text="the quick brown fox",
    )
    # Deliberately do NOT add this chunk to bm25, to prove bm25 is never consulted.
    vectors.upsert([chunk_id], [[1.0, 0.0]], [chunk.model_dump()])
    store.add([chunk])

    def handler(request):
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        results = await search("fox", bm25, vectors, store, client, settings)

    assert len(results) == 1
    assert results[0].matched_methods == ["semantic"]
    assert results[0].bm25_rank is None
    assert results[0].bm25_score is None
```

Create `tests/retrieval/strategies/test_registry.py`:

```python
from app.retrieval.strategies import RETRIEVAL_STRATEGIES


def test_registry_has_exactly_the_three_expected_strategies():
    assert set(RETRIEVAL_STRATEGIES.keys()) == {"bm25_only", "semantic_only", "hybrid_rrf"}
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/retrieval/strategies -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'app.retrieval.strategies'`).

- [ ] **Step 3: Implement**

Create `app/retrieval/strategies/hybrid_rrf.py` (today's `app/retrieval/retriever.py`, `retrieve` renamed `search`, plus an extracted `_assemble` helper shared with the two single-method strategies):

```python
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


def assemble(
    bm25_hits: list[RankedHit],
    semantic_hits: list[RankedHit],
    chunk_store: ChunkStore,
    settings: Settings,
    display_top_k: int,
) -> list[FusedChunk]:
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
            city=metadata_by_id[hit.chunk_id].city,
            price=metadata_by_id[hit.chunk_id].price,
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


async def search(
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
    return assemble(bm25_hits, semantic_hits, chunk_store, settings, display_top_k)
```

Create `app/retrieval/strategies/bm25_only.py`:

```python
from __future__ import annotations

import httpx

from app.config import Settings, settings as default_settings
from app.indexing.bm25_index import InMemoryBM25Index
from app.indexing.chunk_store import ChunkStore
from app.indexing.vector_index import QdrantVectorIndex
from app.retrieval.fusion import RankedHit
from app.retrieval.models import FusedChunk
from app.retrieval.strategies.hybrid_rrf import assemble


async def search(
    query: str,
    bm25_index: InMemoryBM25Index,
    vector_index: QdrantVectorIndex,
    chunk_store: ChunkStore,
    http_client: httpx.AsyncClient,
    settings: Settings = default_settings,
    top_k: int | None = None,
) -> list[FusedChunk]:
    display_top_k = settings.display_top_k if top_k is None else top_k
    results = bm25_index.search(query, settings.retrieval_top_k)
    bm25_hits = [RankedHit(chunk_id, rank + 1, score) for rank, (chunk_id, score) in enumerate(results)]
    return assemble(bm25_hits, [], chunk_store, settings, display_top_k)
```

Create `app/retrieval/strategies/semantic_only.py`:

```python
from __future__ import annotations

import httpx

from app.config import Settings, settings as default_settings
from app.indexing.bm25_index import InMemoryBM25Index
from app.indexing.chunk_store import ChunkStore
from app.indexing.embeddings import embed_texts
from app.indexing.vector_index import QdrantVectorIndex
from app.retrieval.fusion import RankedHit
from app.retrieval.models import FusedChunk
from app.retrieval.strategies.hybrid_rrf import assemble


async def search(
    query: str,
    bm25_index: InMemoryBM25Index,
    vector_index: QdrantVectorIndex,
    chunk_store: ChunkStore,
    http_client: httpx.AsyncClient,
    settings: Settings = default_settings,
    top_k: int | None = None,
) -> list[FusedChunk]:
    display_top_k = settings.display_top_k if top_k is None else top_k
    vectors = await embed_texts(http_client, [query], settings)
    results = vector_index.search(vectors[0], settings.retrieval_top_k)
    semantic_hits = [RankedHit(chunk_id, rank + 1, score) for rank, (chunk_id, score) in enumerate(results)]
    return assemble([], semantic_hits, chunk_store, settings, display_top_k)
```

Create `app/retrieval/strategies/__init__.py`:

```python
from __future__ import annotations

from typing import Awaitable, Callable

import httpx

from app.config import Settings
from app.indexing.bm25_index import InMemoryBM25Index
from app.indexing.chunk_store import ChunkStore
from app.indexing.vector_index import QdrantVectorIndex
from app.retrieval.models import FusedChunk
from app.retrieval.strategies import bm25_only, hybrid_rrf, semantic_only

RetrievalStrategyFn = Callable[
    [str, InMemoryBM25Index, QdrantVectorIndex, ChunkStore, httpx.AsyncClient, Settings, int | None],
    Awaitable[list[FusedChunk]],
]

RETRIEVAL_STRATEGIES: dict[str, RetrievalStrategyFn] = {
    "bm25_only": bm25_only.search,
    "semantic_only": semantic_only.search,
    "hybrid_rrf": hybrid_rrf.search,
}
```

Delete `app/retrieval/retriever.py` and `tests/retrieval/test_retriever.py` (fully superseded by `strategies/hybrid_rrf.py` and `tests/retrieval/strategies/test_hybrid_rrf.py`).

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/retrieval/strategies -v`
Expected: all PASS.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: **known, expected failures** in every test that imports `app.main` or `app.retrieval.router` — `tests/retrieval/test_router.py`, `tests/frontend/test_ingest_page.py`, `tests/frontend/test_query_page.py`, `tests/frontend/test_static_assets.py`, `tests/ingestion/test_main.py` — because `app/retrieval/router.py` still does `from app.retrieval.retriever import retrieve`, and that module is now gone. This is expected: `router.py` isn't rewired until Task 11, which is the task that also fixes `app/main.py` and re-greens all of these. Confirm the failure in each is exactly an `ImportError`/collection error mentioning `app.retrieval.retriever` — if anything else fails, stop and investigate.

- [ ] **Step 6: Commit**

```bash
git add app/retrieval/strategies tests/retrieval/strategies
git rm app/retrieval/retriever.py tests/retrieval/test_retriever.py
git commit -m "feat: add retrieval strategy registry, move hybrid_rrf out of retriever.py"
```

---

### Task 6: Post-retrieval strategy registry

**Files:**
- Create: `app/postretrieval/__init__.py` (empty), `app/postretrieval/strategies/__init__.py`, `app/postretrieval/strategies/none.py`, `app/postretrieval/strategies/metadata_filter.py`, `app/postretrieval/strategies/cross_encoder_rerank.py`
- Create: `tests/postretrieval/__init__.py` (empty), `tests/postretrieval/strategies/__init__.py` (empty), `tests/postretrieval/strategies/test_none.py`, `tests/postretrieval/strategies/test_metadata_filter.py`, `tests/postretrieval/strategies/test_registry.py`
- Delete: `app/retrieval/filtering.py`, `tests/retrieval/test_filtering.py`

**Interfaces:**
- Consumes: `app.retrieval.models.FusedChunk`, `app.extraction.preferences.QueryPreferences` (both unchanged).
- Produces: `POST_RETRIEVAL_STRATEGIES: dict[str, PostRetrievalStrategyFn]` with keys `"none"`, `"metadata_filter"`, `"cross_encoder_rerank"`, each `def apply(fused_chunks: list[FusedChunk], preferences: QueryPreferences) -> tuple[list[FusedChunk], int]`. Task 11's `/query` router consumes this.

- [ ] **Step 1: Write the failing tests**

Create `tests/postretrieval/__init__.py` and `tests/postretrieval/strategies/__init__.py` (both empty).

Create `tests/postretrieval/strategies/test_none.py`:

```python
from app.extraction.preferences import QueryPreferences
from app.postretrieval.strategies.none import apply
from app.retrieval.models import FusedChunk


def _chunk(chunk_id: str) -> FusedChunk:
    return FusedChunk(
        chunk_id=chunk_id, text="text", source_url="https://example.com", page_number=1,
        bm25_rank=1, bm25_score=1.0, semantic_rank=1, semantic_score=1.0,
        fused_rank=1, rrf_score=0.03, matched_methods=["bm25", "semantic"],
    )


def test_none_passes_everything_through_unfiltered():
    chunks = [_chunk("c1"), _chunk("c2")]
    kept, excluded_count = apply(chunks, QueryPreferences())
    assert kept == chunks
    assert excluded_count == 0
```

Create `tests/postretrieval/strategies/test_metadata_filter.py` (ports `tests/retrieval/test_filtering.py`, new import, `filter_chunks` renamed `apply`):

```python
from app.extraction.preferences import QueryPreferences
from app.postretrieval.strategies.metadata_filter import apply
from app.retrieval.models import FusedChunk


def _chunk(chunk_id: str, city: str | None = None, price: float | None = None) -> FusedChunk:
    return FusedChunk(
        chunk_id=chunk_id, text="text", source_url="https://example.com", page_number=1,
        city=city, price=price, bm25_rank=1, bm25_score=1.0, semantic_rank=1, semantic_score=1.0,
        fused_rank=1, rrf_score=0.03, matched_methods=["bm25", "semantic"],
    )


def test_apply_excludes_conflicting_city():
    chunks = [_chunk("c1", city="paris"), _chunk("c2", city="lahore")]
    kept, excluded_count = apply(chunks, QueryPreferences(city="lahore"))
    assert [c.chunk_id for c in kept] == ["c2"]
    assert excluded_count == 1


def test_apply_excludes_over_budget_price():
    chunks = [_chunk("c1", price=1000.0), _chunk("c2", price=100.0)]
    kept, excluded_count = apply(chunks, QueryPreferences(budget=500.0))
    assert [c.chunk_id for c in kept] == ["c2"]
    assert excluded_count == 1


def test_apply_permissive_on_missing_metadata():
    chunks = [_chunk("c1", city=None, price=None), _chunk("c2", city="lahore", price=100.0)]
    kept, excluded_count = apply(chunks, QueryPreferences(city="paris", budget=50.0))
    assert [c.chunk_id for c in kept] == ["c1"]
    assert excluded_count == 1


def test_apply_passes_everything_with_no_preferences():
    chunks = [_chunk("c1", city="paris", price=1000.0), _chunk("c2")]
    kept, excluded_count = apply(chunks, QueryPreferences())
    assert len(kept) == 2
    assert excluded_count == 0


def test_apply_city_match_is_case_insensitive_via_lowercased_storage():
    chunks = [_chunk("c1", city="lahore")]
    kept, excluded_count = apply(chunks, QueryPreferences(city="lahore"))
    assert len(kept) == 1
    assert excluded_count == 0
```

Create `tests/postretrieval/strategies/test_registry.py`:

```python
import pytest

from app.extraction.preferences import QueryPreferences
from app.postretrieval.strategies import POST_RETRIEVAL_STRATEGIES
from app.retrieval.models import FusedChunk


def test_registry_has_exactly_the_three_expected_strategies():
    assert set(POST_RETRIEVAL_STRATEGIES.keys()) == {"none", "metadata_filter", "cross_encoder_rerank"}


def test_cross_encoder_rerank_stub_raises_not_implemented():
    chunk = FusedChunk(
        chunk_id="c1", text="text", source_url="https://example.com", page_number=1,
        bm25_rank=1, bm25_score=1.0, semantic_rank=1, semantic_score=1.0,
        fused_rank=1, rrf_score=0.03, matched_methods=["bm25"],
    )
    with pytest.raises(NotImplementedError):
        POST_RETRIEVAL_STRATEGIES["cross_encoder_rerank"]([chunk], QueryPreferences())
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/postretrieval -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'app.postretrieval'`).

- [ ] **Step 3: Implement**

Create `app/postretrieval/__init__.py` (empty).

Create `app/postretrieval/strategies/none.py`:

```python
from __future__ import annotations

from app.extraction.preferences import QueryPreferences
from app.retrieval.models import FusedChunk


def apply(fused_chunks: list[FusedChunk], preferences: QueryPreferences) -> tuple[list[FusedChunk], int]:
    return fused_chunks, 0
```

Create `app/postretrieval/strategies/metadata_filter.py` (today's `app/retrieval/filtering.py`, `filter_chunks` renamed `apply`):

```python
from __future__ import annotations

from app.extraction.preferences import QueryPreferences
from app.retrieval.models import FusedChunk


def apply(fused_chunks: list[FusedChunk], preferences: QueryPreferences) -> tuple[list[FusedChunk], int]:
    kept = [chunk for chunk in fused_chunks if not _conflicts(chunk, preferences)]
    return kept, len(fused_chunks) - len(kept)


def _conflicts(chunk: FusedChunk, preferences: QueryPreferences) -> bool:
    if chunk.city is not None and preferences.city is not None and chunk.city != preferences.city:
        return True
    if chunk.price is not None and preferences.budget is not None and chunk.price > preferences.budget:
        return True
    return False
```

Create `app/postretrieval/strategies/cross_encoder_rerank.py`:

```python
from __future__ import annotations

from app.extraction.preferences import QueryPreferences
from app.retrieval.models import FusedChunk


def apply(fused_chunks: list[FusedChunk], preferences: QueryPreferences) -> tuple[list[FusedChunk], int]:
    raise NotImplementedError("cross-encoder rerank strategy not yet implemented — see piece B")
```

Create `app/postretrieval/strategies/__init__.py`:

```python
from __future__ import annotations

from typing import Callable

from app.extraction.preferences import QueryPreferences
from app.postretrieval.strategies import cross_encoder_rerank, metadata_filter, none
from app.retrieval.models import FusedChunk

PostRetrievalStrategyFn = Callable[[list[FusedChunk], QueryPreferences], tuple[list[FusedChunk], int]]

POST_RETRIEVAL_STRATEGIES: dict[str, PostRetrievalStrategyFn] = {
    "none": none.apply,
    "metadata_filter": metadata_filter.apply,
    "cross_encoder_rerank": cross_encoder_rerank.apply,
}
```

Delete `app/retrieval/filtering.py` and `tests/retrieval/test_filtering.py`.

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/postretrieval -v`
Expected: all PASS.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: the same known set of failures as after Task 5 — every test that imports `app.main` or `app.retrieval.router`: `tests/retrieval/test_router.py`, `tests/frontend/test_ingest_page.py`, `tests/frontend/test_query_page.py`, `tests/frontend/test_static_assets.py`, `tests/ingestion/test_main.py` — now failing on `app.retrieval.filtering` in addition to `app.retrieval.retriever` (both still imported by the not-yet-rewired `app/retrieval/router.py`). That's expected and gets fixed in Task 11. If anything outside that known set fails, stop and investigate before continuing.

- [ ] **Step 6: Commit**

```bash
git add app/postretrieval tests/postretrieval
git rm app/retrieval/filtering.py tests/retrieval/test_filtering.py
git commit -m "feat: add post-retrieval strategy registry, move metadata_filter out of filtering.py"
```

---

### Task 7: Refactor `app/indexing/indexer.py` to `index_chunks()`

**Files:**
- Modify: `app/indexing/indexer.py`
- Modify: `tests/indexing/test_indexer.py`
- Delete: `app/indexing/router.py`, `tests/indexing/test_router.py`, `tests/indexing/test_wiring.py`

**Interfaces:**
- Consumes: `app.indexing.chunker.TextChunk` (unchanged).
- Produces: `async def index_chunks(text_chunks: list[TextChunk], source_name: str, doc_id_hash: str, bm25_index, vector_index, chunk_store, http_client, settings=default_settings) -> IndexResult` — decoupled from `IngestionPayload` (which is being deleted in Task 11 along with the rest of `app/ingestion/`). Task 9's loader calls this per new archive doc.

**Note on scope:** `app/indexing/router.py` exposed `POST /index/chunk`, documented as "internal, called by ingestion worker per completed document." Once the ingestion worker (Task 11) is deleted and the pipeline loader calls `index_chunks()` in-process, nothing calls this endpoint — it is dead code, deleted here rather than kept around unused. `tests/indexing/test_wiring.py` exercises the URL-ingest-to-index round trip via that now-gone endpoint and is also deleted.

- [ ] **Step 1: Write the failing tests**

Replace `tests/indexing/test_indexer.py`:

```python
import httpx
import pytest

from app.config import Settings
from app.indexing.bm25_index import InMemoryBM25Index
from app.indexing.chunk_store import ChunkStore
from app.indexing.chunker import chunk_text
from app.indexing.indexer import index_chunks
from app.indexing.vector_index import QdrantVectorIndex


def _settings(tmp_path) -> Settings:
    return Settings(
        qdrant_path=str(tmp_path / "qdrant"), qdrant_collection="test_chunks", vector_size=3,
        chunk_size_tokens=6, chunk_overlap_tokens=0,
    )


async def test_index_chunks_writes_to_both_indexes_and_chunk_store(tmp_path):
    settings = _settings(tmp_path)
    bm25 = InMemoryBM25Index()
    vectors = QdrantVectorIndex(settings)
    store = ChunkStore()
    text_chunks = chunk_text("one two three. four five six.", 6, 0)

    def handler(request):
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await index_chunks(text_chunks, "paper.pdf", "hash1", bm25, vectors, store, client, settings)

    assert result.status == "indexed"
    assert result.chunk_count == 2
    stored = store.get_many([c.chunk_id for c in store._chunks.values()])
    assert len(stored) == 2
    assert all(c.doc_id == result.doc_id for c in stored)
    assert all(c.doc_id_hash == "hash1" for c in stored)
    assert all(c.source_url == "paper.pdf" for c in stored)
    # See tests/indexing/test_indexer.py's original note: BM25Okapi's idf formula
    # zeroes out relevance scores for disjoint-vocabulary 2-document corpora, so
    # assert on BM25 index membership instead of a relevance-score search hit.
    assert {c.chunk_id for c in stored}.issubset(set(bm25._chunk_ids))
    assert len(vectors.search([0.1, 0.2, 0.3], top_k=5)) == 2


async def test_index_chunks_tags_city_and_price(tmp_path):
    settings = _settings(tmp_path)
    bm25 = InMemoryBM25Index()
    vectors = QdrantVectorIndex(settings)
    store = ChunkStore()
    text_chunks = chunk_text("A budget hotel in Paris costs around $500 per night.", 400, 75)

    def handler(request):
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3]]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await index_chunks(text_chunks, "paper.pdf", "hash2", bm25, vectors, store, client, settings)

    stored = store.get_many([c.chunk_id for c in store._chunks.values()])
    assert result.chunk_count == 1
    assert stored[0].city == "paris"
    assert stored[0].price == 500.0


async def test_index_chunks_leaves_city_and_price_none_when_absent(tmp_path):
    settings = _settings(tmp_path)
    bm25 = InMemoryBM25Index()
    vectors = QdrantVectorIndex(settings)
    store = ChunkStore()
    text_chunks = chunk_text("one two three. four five six.", 6, 0)

    def handler(request):
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await index_chunks(text_chunks, "paper.pdf", "hash3", bm25, vectors, store, client, settings)

    stored = store.get_many([c.chunk_id for c in store._chunks.values()])
    assert all(c.city is None for c in stored)
    assert all(c.price is None for c in stored)


async def test_index_chunks_rolls_back_bm25_on_vector_failure(tmp_path):
    settings = _settings(tmp_path)
    bm25 = InMemoryBM25Index()
    vectors = QdrantVectorIndex(settings)
    store = ChunkStore()
    text_chunks = chunk_text("one two three. four five six.", 6, 0)

    def handler(request):
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2], [0.4, 0.5]]})  # wrong vector_size (3 expected)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(Exception):
            await index_chunks(text_chunks, "paper.pdf", "hash4", bm25, vectors, store, client, settings)

    assert bm25.search("one two three", top_k=5) == []
    assert len(store._chunks) == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/indexing/test_indexer.py -v`
Expected: FAIL (`index_chunks` doesn't exist yet; `index_document` has a different signature).

- [ ] **Step 3: Implement**

Replace `app/indexing/indexer.py`:

```python
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import httpx

from app.config import Settings, settings as default_settings
from app.extraction.location import extract_city
from app.extraction.price import extract_price
from app.indexing.bm25_index import InMemoryBM25Index
from app.indexing.chunk_store import ChunkStore
from app.indexing.chunker import TextChunk
from app.indexing.embeddings import embed_texts
from app.indexing.models import ChunkMetadata, IndexResult
from app.indexing.vector_index import QdrantVectorIndex


async def index_chunks(
    text_chunks: list[TextChunk],
    source_name: str,
    doc_id_hash: str,
    bm25_index: InMemoryBM25Index,
    vector_index: QdrantVectorIndex,
    chunk_store: ChunkStore,
    http_client: httpx.AsyncClient,
    settings: Settings = default_settings,
) -> IndexResult:
    doc_id = str(uuid.uuid4())
    indexed_at = datetime.now(timezone.utc).isoformat()

    chunk_metadatas = [
        ChunkMetadata(
            chunk_id=str(uuid.uuid4()),
            doc_id=doc_id,
            doc_id_hash=doc_id_hash,
            source_url=source_name,
            page_number=1,
            chunk_index=index,
            char_start=text_chunk.char_start,
            char_end=text_chunk.char_end,
            overlap_with_prev=text_chunk.overlap_with_prev,
            indexed_at=indexed_at,
            text=text_chunk.text,
            city=extract_city(text_chunk.text),
            price=extract_price(text_chunk.text),
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

Delete `app/indexing/router.py`, `tests/indexing/test_router.py`, `tests/indexing/test_wiring.py`.

**Note:** `page_number` is hardcoded to `1` for all archive-sourced chunks — PDF page boundaries aren't tracked through the chunking strategies in this piece (the old HTML `page_map` concept doesn't carry over to raw PDF text extraction). Preserving true per-page numbers is deferred to piece B, when the hierarchical chunking strategies are implemented and page-aware chunking becomes relevant anyway.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/indexing/test_indexer.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest -q`
Expected: the same known set of failures as after Task 6 — `tests/retrieval/test_router.py`, `tests/frontend/test_ingest_page.py`, `tests/frontend/test_query_page.py`, `tests/frontend/test_static_assets.py`, `tests/ingestion/test_main.py` (all still failing on `app/retrieval/router.py`'s imports of the deleted `retriever`/`filtering`/`judge` modules — unrelated to this task's changes). No other regressions; everything in `app/ingestion` and its tests still passes as-is since that package isn't touched until Task 11.

- [ ] **Step 6: Commit**

```bash
git add app/indexing/indexer.py tests/indexing/test_indexer.py
git rm app/indexing/router.py tests/indexing/test_router.py tests/indexing/test_wiring.py
git commit -m "refactor: decouple index_chunks() from IngestionPayload, drop dead /index/chunk endpoint"
```

---

### Task 8: `app/pipeline/config.py` + `app/pipeline/registry.py` + `app/pipeline/models.py`

**Files:**
- Create: `app/pipeline/__init__.py` (empty), `app/pipeline/config.py`, `app/pipeline/registry.py`, `app/pipeline/models.py`
- Create: `tests/pipeline/__init__.py` (empty), `tests/pipeline/test_config.py`, `tests/pipeline/test_registry.py`

**Interfaces:**
- Produces:
  - `PipelineConfig(indexing_strategy, retrieval_strategy, post_retrieval_strategy)`, `collection_name_for(base, strategy_id) -> str`, `get_active() -> PipelineConfig | None`, `set_active(config: PipelineConfig | None) -> None`.
  - `IndexingCollectionRegistry(settings)` with `.get(strategy_id) -> IndexingCollection` (`.vector_index`, `.bm25_index`, `.chunk_store`), `.doc_count(strategy_id) -> int`, `.close_all() -> None`. `INDEXING_STRATEGY_IDS: tuple[str, ...]`.
  - `DocFailure(path, error)`, `IndexedSummary(new_docs, total_docs, failures)`, `EvalResult(status)`, `PipelineLoadResult(indexed, eval)`, `PipelineStatus(active, doc_counts)`.
- Task 9 (`loader.py`) and Task 10 (`router.py`) both depend on all three files.

- [ ] **Step 1: Write the failing config test**

Create `tests/pipeline/__init__.py` (empty) and `tests/pipeline/test_config.py`:

```python
import pytest
from pydantic import ValidationError

from app.pipeline.config import PipelineConfig, collection_name_for, get_active, set_active


@pytest.fixture(autouse=True)
def _reset_active_pipeline():
    yield
    set_active(None)


def test_get_active_defaults_to_none():
    assert get_active() is None


def test_set_active_then_get_active_round_trips():
    config = PipelineConfig(indexing_strategy="fixed_window", retrieval_strategy="hybrid_rrf", post_retrieval_strategy="none")
    set_active(config)
    assert get_active() == config


def test_pipeline_config_rejects_unknown_strategy_id():
    with pytest.raises(ValidationError):
        PipelineConfig(indexing_strategy="not_a_real_strategy", retrieval_strategy="hybrid_rrf", post_retrieval_strategy="none")


def test_collection_name_for_namespaces_by_strategy():
    assert collection_name_for("rag_chunks", "fixed_window") == "rag_chunks__fixed_window"
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/pipeline/test_config.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'app.pipeline'`).

- [ ] **Step 3: Implement `config.py`**

Create `app/pipeline/__init__.py` (empty).

Create `app/pipeline/config.py`:

```python
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

IndexingStrategyId = Literal["fixed_window", "semantic", "hierarchical", "hierarchical_summary"]
RetrievalStrategyId = Literal["bm25_only", "semantic_only", "hybrid_rrf"]
PostRetrievalStrategyId = Literal["none", "metadata_filter", "cross_encoder_rerank"]


class PipelineConfig(BaseModel):
    indexing_strategy: IndexingStrategyId
    retrieval_strategy: RetrievalStrategyId
    post_retrieval_strategy: PostRetrievalStrategyId


def collection_name_for(base_collection: str, indexing_strategy: IndexingStrategyId) -> str:
    return f"{base_collection}__{indexing_strategy}"


_active: PipelineConfig | None = None


def get_active() -> PipelineConfig | None:
    return _active


def set_active(config: PipelineConfig | None) -> None:
    global _active
    _active = config
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/pipeline/test_config.py -v`
Expected: all PASS.

- [ ] **Step 5: Write the failing registry test**

Create `tests/pipeline/test_registry.py`:

```python
from app.config import Settings
from app.indexing.models import ChunkMetadata
from app.pipeline.registry import INDEXING_STRATEGY_IDS, IndexingCollectionRegistry


def _settings(tmp_path) -> Settings:
    return Settings(qdrant_path=str(tmp_path / "qdrant"), qdrant_collection="preg", vector_size=2)


def test_registry_creates_one_empty_collection_per_strategy(tmp_path):
    registry = IndexingCollectionRegistry(_settings(tmp_path))
    for strategy_id in INDEXING_STRATEGY_IDS:
        assert registry.doc_count(strategy_id) == 0
    registry.close_all()


def test_registry_isolates_strategies_from_each_other(tmp_path):
    registry = IndexingCollectionRegistry(_settings(tmp_path))
    fixed = registry.get("fixed_window")
    chunk = ChunkMetadata(
        chunk_id="c1", doc_id="d1", doc_id_hash="h1", source_url="a.pdf", page_number=1,
        chunk_index=0, char_start=0, char_end=10, overlap_with_prev=0,
        indexed_at="2026-08-09T00:00:00+00:00", text="the quick brown fox",
    )
    fixed.vector_index.upsert(["c1"], [[1.0, 0.0]], [chunk.model_dump()])
    fixed.chunk_store.add([chunk])
    fixed.bm25_index.add_documents(["c1"], ["the quick brown fox"])

    assert registry.doc_count("fixed_window") == 1
    assert registry.doc_count("semantic") == 0
    assert registry.get("semantic").vector_index.scroll_all() == []
    registry.close_all()


def test_registry_rehydrates_chunk_store_and_bm25_from_persisted_qdrant_data(tmp_path):
    settings = _settings(tmp_path)
    first_registry = IndexingCollectionRegistry(settings)
    collection = first_registry.get("fixed_window")
    chunk = ChunkMetadata(
        chunk_id="c1", doc_id="d1", doc_id_hash="h1", source_url="paper.pdf", page_number=1,
        chunk_index=0, char_start=0, char_end=10, overlap_with_prev=0,
        indexed_at="2026-08-09T00:00:00+00:00", text="the quick brown fox",
    )
    collection.vector_index.upsert(["c1"], [[1.0, 0.0]], [chunk.model_dump()])
    first_registry.close_all()  # release the local-mode Qdrant lock before reopening

    second_registry = IndexingCollectionRegistry(settings)
    rehydrated = second_registry.get("fixed_window")

    assert rehydrated.chunk_store.get("c1").text == "the quick brown fox"
    assert rehydrated.bm25_index.search("quick brown fox", top_k=5) != []
    assert second_registry.doc_count("fixed_window") == 1
    second_registry.close_all()
```

- [ ] **Step 6: Run to verify it fails**

Run: `uv run pytest tests/pipeline/test_registry.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'app.pipeline.registry'`).

- [ ] **Step 7: Implement `registry.py` and `models.py`**

Create `app/pipeline/registry.py`:

```python
from __future__ import annotations

import dataclasses

from qdrant_client import QdrantClient

from app.config import Settings
from app.indexing.bm25_index import InMemoryBM25Index
from app.indexing.chunk_store import ChunkStore
from app.indexing.models import ChunkMetadata
from app.indexing.vector_index import QdrantVectorIndex
from app.pipeline.config import IndexingStrategyId, collection_name_for

INDEXING_STRATEGY_IDS: tuple[IndexingStrategyId, ...] = (
    "fixed_window",
    "semantic",
    "hierarchical",
    "hierarchical_summary",
)


@dataclasses.dataclass
class IndexingCollection:
    vector_index: QdrantVectorIndex
    bm25_index: InMemoryBM25Index
    chunk_store: ChunkStore


class IndexingCollectionRegistry:
    def __init__(self, settings: Settings) -> None:
        self._client = QdrantClient(url=settings.qdrant_url) if settings.qdrant_url else QdrantClient(path=settings.qdrant_path)
        self._collections: dict[IndexingStrategyId, IndexingCollection] = {}
        for strategy_id in INDEXING_STRATEGY_IDS:
            strategy_settings = dataclasses.replace(
                settings, qdrant_collection=collection_name_for(settings.qdrant_collection, strategy_id)
            )
            vector_index = QdrantVectorIndex(strategy_settings, client=self._client)
            bm25_index = InMemoryBM25Index()
            chunk_store = ChunkStore()
            _rehydrate(vector_index, bm25_index, chunk_store)
            self._collections[strategy_id] = IndexingCollection(vector_index, bm25_index, chunk_store)

    def get(self, strategy_id: IndexingStrategyId) -> IndexingCollection:
        return self._collections[strategy_id]

    def doc_count(self, strategy_id: IndexingStrategyId) -> int:
        return len(self._collections[strategy_id].chunk_store.doc_id_hashes())

    def close_all(self) -> None:
        self._client.close()


def _rehydrate(vector_index: QdrantVectorIndex, bm25_index: InMemoryBM25Index, chunk_store: ChunkStore) -> None:
    payloads = vector_index.scroll_all()
    if not payloads:
        return
    chunk_metadatas = [ChunkMetadata(**payload) for payload in payloads]
    chunk_store.add(chunk_metadatas)
    bm25_index.add_documents([c.chunk_id for c in chunk_metadatas], [c.text for c in chunk_metadatas])
```

Create `app/pipeline/models.py`:

```python
from __future__ import annotations

from pydantic import BaseModel

from app.pipeline.config import PipelineConfig


class DocFailure(BaseModel):
    path: str
    error: str


class IndexedSummary(BaseModel):
    new_docs: int
    total_docs: int
    failures: list[DocFailure]


class EvalResult(BaseModel):
    status: str


class PipelineLoadResult(BaseModel):
    indexed: IndexedSummary
    eval: EvalResult


class PipelineStatus(BaseModel):
    active: PipelineConfig | None
    doc_counts: dict[str, int]
```

- [ ] **Step 8: Run to verify it passes**

Run: `uv run pytest tests/pipeline -v`
Expected: all PASS.

- [ ] **Step 9: Commit**

```bash
git add app/pipeline tests/pipeline
git commit -m "feat: add PipelineConfig, IndexingCollectionRegistry, and pipeline result models"
```

---

### Task 9: `app/pipeline/loader.py` — `load_pipeline()`

**Files:**
- Create: `app/pipeline/loader.py`
- Create: `tests/pipeline/test_loader.py`

**Interfaces:**
- Consumes: `scan_archive`, `extract_pdf_text`/`PdfExtractionError` (Task 3), `INDEXING_STRATEGIES` (Task 4), `index_chunks` (Task 7), `IndexingCollectionRegistry`, `PipelineConfig`, `set_active` (Task 8), `DocFailure`/`IndexedSummary`/`EvalResult`/`PipelineLoadResult` (Task 8).
- Produces: `async def load_pipeline(config, registry, http_client, settings=default_settings, archive_dir=None) -> PipelineLoadResult`; module-level `DEFAULT_ARCHIVE_DIR = Path("archive")` (read inside the function body, not baked into the default parameter, so tests can monkeypatch it). Task 10's router calls this directly.

- [ ] **Step 1: Write the failing tests**

Create `tests/pipeline/test_loader.py`:

```python
import asyncio
import json
from pathlib import Path

import httpx
import pytest

import app.pipeline.loader as loader_module
from app.config import Settings
from app.pipeline.config import PipelineConfig, get_active, set_active
from app.pipeline.loader import load_pipeline
from app.pipeline.registry import IndexingCollectionRegistry


@pytest.fixture(autouse=True)
def _reset_active_pipeline():
    yield
    set_active(None)


def _settings(tmp_path) -> Settings:
    return Settings(
        qdrant_path=str(tmp_path / "qdrant"), qdrant_collection="ploader", vector_size=2,
        chunk_size_tokens=6, chunk_overlap_tokens=0,
    )


def _embed_client() -> httpx.AsyncClient:
    def handler(request):
        body = json.loads(request.read())
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2] for _ in body["input"]]})
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _make_archive(tmp_path, names: list[str]) -> Path:
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    for name in names:
        (archive_dir / name).write_bytes(name.encode())
    return archive_dir


def _config(**overrides) -> PipelineConfig:
    defaults = dict(indexing_strategy="fixed_window", retrieval_strategy="hybrid_rrf", post_retrieval_strategy="none")
    defaults.update(overrides)
    return PipelineConfig(**defaults)


async def test_load_pipeline_indexes_new_docs_and_sets_active(tmp_path, monkeypatch):
    monkeypatch.setattr(loader_module, "extract_pdf_text", lambda path: "one two three. four five six.")
    settings = _settings(tmp_path)
    registry = IndexingCollectionRegistry(settings)
    archive_dir = _make_archive(tmp_path, ["a.pdf", "b.pdf"])
    config = _config()

    client = _embed_client()
    result = await load_pipeline(config, registry, client, settings, archive_dir=archive_dir)
    await client.aclose()
    registry.close_all()

    assert result.indexed.new_docs == 2
    assert result.indexed.total_docs == 2
    assert result.indexed.failures == []
    assert get_active() == config


async def test_load_pipeline_skips_already_indexed_docs(tmp_path, monkeypatch):
    monkeypatch.setattr(loader_module, "extract_pdf_text", lambda path: "one two three. four five six.")
    settings = _settings(tmp_path)
    registry = IndexingCollectionRegistry(settings)
    archive_dir = _make_archive(tmp_path, ["a.pdf"])
    config = _config()

    client = _embed_client()
    first = await load_pipeline(config, registry, client, settings, archive_dir=archive_dir)
    second = await load_pipeline(config, registry, client, settings, archive_dir=archive_dir)
    await client.aclose()
    registry.close_all()

    assert first.indexed.new_docs == 1
    assert second.indexed.new_docs == 0
    assert second.indexed.total_docs == 1


async def test_load_pipeline_isolates_strategies_in_separate_collections(tmp_path, monkeypatch):
    monkeypatch.setattr(loader_module, "extract_pdf_text", lambda path: "one two three. four five six.")
    settings = _settings(tmp_path)
    registry = IndexingCollectionRegistry(settings)
    archive_dir = _make_archive(tmp_path, ["a.pdf"])

    client = _embed_client()
    await load_pipeline(_config(indexing_strategy="fixed_window"), registry, client, settings, archive_dir=archive_dir)
    await client.aclose()
    registry.close_all()

    settings2 = _settings(tmp_path)
    registry2 = IndexingCollectionRegistry(settings2)
    assert registry2.doc_count("fixed_window") == 1
    assert registry2.doc_count("semantic") == 0
    registry2.close_all()


async def test_load_pipeline_records_per_doc_failure_without_stopping_batch(tmp_path, monkeypatch):
    def flaky_extract(path):
        if path.name == "bad.pdf":
            raise RuntimeError("corrupt pdf")
        return "one two three. four five six."

    monkeypatch.setattr(loader_module, "extract_pdf_text", flaky_extract)
    settings = _settings(tmp_path)
    registry = IndexingCollectionRegistry(settings)
    archive_dir = _make_archive(tmp_path, ["bad.pdf", "good.pdf"])
    config = _config()

    client = _embed_client()
    result = await load_pipeline(config, registry, client, settings, archive_dir=archive_dir)
    await client.aclose()
    registry.close_all()

    assert result.indexed.new_docs == 1
    assert len(result.indexed.failures) == 1
    assert result.indexed.failures[0].path.endswith("bad.pdf")
    assert result.indexed.failures[0].error == "corrupt pdf"


async def test_load_pipeline_empty_archive_returns_zero_without_error(tmp_path):
    settings = _settings(tmp_path)
    registry = IndexingCollectionRegistry(settings)
    config = _config()

    client = _embed_client()
    result = await load_pipeline(config, registry, client, settings, archive_dir=tmp_path / "missing")
    await client.aclose()
    registry.close_all()

    assert result.indexed.new_docs == 0
    assert result.indexed.total_docs == 0


async def test_load_pipeline_serializes_concurrent_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(loader_module, "extract_pdf_text", lambda path: "one two three. four five six.")
    settings = _settings(tmp_path)
    registry = IndexingCollectionRegistry(settings)
    archive_dir = _make_archive(tmp_path, ["a.pdf"])
    config = _config()

    concurrent = {"active": 0, "max": 0}

    async def slow_embed(client, texts, settings):
        concurrent["active"] += 1
        concurrent["max"] = max(concurrent["max"], concurrent["active"])
        await asyncio.sleep(0.05)
        concurrent["active"] -= 1
        return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr(loader_module, "embed_texts", slow_embed)

    client = _embed_client()
    await asyncio.gather(
        load_pipeline(config, registry, client, settings, archive_dir=archive_dir),
        load_pipeline(config, registry, client, settings, archive_dir=archive_dir),
    )
    await client.aclose()
    registry.close_all()

    assert concurrent["max"] == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/pipeline/test_loader.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'app.pipeline.loader'`).

- [ ] **Step 3: Implement**

Create `app/pipeline/loader.py`:

```python
from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from app.archive.pdf_extractor import PdfExtractionError, extract_pdf_text
from app.archive.scanner import scan_archive
from app.config import Settings, settings as default_settings
from app.indexing.embeddings import embed_texts
from app.indexing.indexer import index_chunks
from app.indexing.strategies import INDEXING_STRATEGIES
from app.pipeline.config import PipelineConfig, set_active
from app.pipeline.models import DocFailure, EvalResult, IndexedSummary, PipelineLoadResult
from app.pipeline.registry import IndexingCollectionRegistry

DEFAULT_ARCHIVE_DIR = Path("archive")

_load_lock = asyncio.Lock()


async def load_pipeline(
    config: PipelineConfig,
    registry: IndexingCollectionRegistry,
    http_client: httpx.AsyncClient,
    settings: Settings = default_settings,
    archive_dir: Path | None = None,
) -> PipelineLoadResult:
    async with _load_lock:
        target_dir = archive_dir if archive_dir is not None else DEFAULT_ARCHIVE_DIR
        collection = registry.get(config.indexing_strategy)
        chunk_fn = INDEXING_STRATEGIES[config.indexing_strategy]

        archive_docs = scan_archive(target_dir)
        known_hashes = collection.chunk_store.doc_id_hashes()
        new_docs = [doc for doc in archive_docs if doc.doc_id_hash not in known_hashes]

        failures: list[DocFailure] = []
        indexed_count = 0
        for doc in new_docs:
            try:
                text = extract_pdf_text(Path(doc.path))
                text_chunks = chunk_fn(text, settings)
                await index_chunks(
                    text_chunks, doc.filename, doc.doc_id_hash,
                    collection.bm25_index, collection.vector_index, collection.chunk_store,
                    http_client, settings,
                )
                indexed_count += 1
            except (PdfExtractionError, NotImplementedError) as exc:
                failures.append(DocFailure(path=doc.path, error=str(exc)))
            except Exception as exc:  # noqa: BLE001 - one bad doc must never abort the batch
                failures.append(DocFailure(path=doc.path, error=str(exc)))

        set_active(config)

        return PipelineLoadResult(
            indexed=IndexedSummary(
                new_docs=indexed_count,
                total_docs=len(collection.chunk_store.doc_id_hashes()),
                failures=failures,
            ),
            eval=EvalResult(status="not_implemented"),
        )
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/pipeline/test_loader.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add app/pipeline/loader.py tests/pipeline/test_loader.py
git commit -m "feat: add load_pipeline() orchestration (scan, diff, index, activate)"
```

---

### Task 10: `app/pipeline/router.py` — `/pipeline/load`, `/pipeline/status`

**Files:**
- Create: `app/pipeline/router.py`
- Create: `tests/pipeline/test_router.py`

**Interfaces:**
- Consumes: `load_pipeline`, `DEFAULT_ARCHIVE_DIR` (Task 9), `get_active`, `PipelineConfig` (Task 8), `INDEXING_STRATEGY_IDS`, `IndexingCollectionRegistry` (Task 8), `PipelineStatus` (Task 8).
- Produces: `build_pipeline_router(registry, http_client, settings=default_settings) -> APIRouter` exposing `POST /pipeline/load` and `GET /pipeline/status`. Task 11's `main.py` wires this in.

- [ ] **Step 1: Write the failing tests**

Create `tests/pipeline/test_router.py`:

```python
import json

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

import app.pipeline.loader as loader_module
from app.config import Settings
from app.pipeline.config import get_active, set_active
from app.pipeline.registry import IndexingCollectionRegistry
from app.pipeline.router import build_pipeline_router


def _settings(tmp_path) -> Settings:
    return Settings(qdrant_path=str(tmp_path / "qdrant"), qdrant_collection="proute", vector_size=2)


def _embed_client() -> httpx.AsyncClient:
    def handler(request):
        body = json.loads(request.read())
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2] for _ in body["input"]]})
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_pipeline_status_before_any_load(tmp_path):
    set_active(None)
    settings = _settings(tmp_path)
    registry = IndexingCollectionRegistry(settings)
    app = FastAPI()
    app.include_router(build_pipeline_router(registry, _embed_client(), settings))

    with TestClient(app) as client:
        response = client.get("/pipeline/status")

    assert response.status_code == 200
    body = response.json()
    assert body["active"] is None
    assert body["doc_counts"] == {"fixed_window": 0, "semantic": 0, "hierarchical": 0, "hierarchical_summary": 0}
    registry.close_all()


def test_pipeline_load_then_status_reflects_active_config(tmp_path, monkeypatch):
    set_active(None)
    monkeypatch.setattr(loader_module, "extract_pdf_text", lambda path: "one two three. four five six.")
    settings = _settings(tmp_path)
    registry = IndexingCollectionRegistry(settings)

    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    (archive_dir / "a.pdf").write_bytes(b"a")
    monkeypatch.setattr(loader_module, "DEFAULT_ARCHIVE_DIR", archive_dir)

    app = FastAPI()
    app.include_router(build_pipeline_router(registry, _embed_client(), settings))

    with TestClient(app) as client:
        load_response = client.post("/pipeline/load", json={
            "indexing_strategy": "fixed_window", "retrieval_strategy": "hybrid_rrf", "post_retrieval_strategy": "none",
        })
        status_response = client.get("/pipeline/status")

    assert load_response.status_code == 200
    assert load_response.json()["indexed"]["new_docs"] == 1
    assert status_response.json()["active"] == {
        "indexing_strategy": "fixed_window", "retrieval_strategy": "hybrid_rrf", "post_retrieval_strategy": "none",
    }
    assert status_response.json()["doc_counts"]["fixed_window"] == 1
    registry.close_all()
    set_active(None)


def test_pipeline_load_rejects_unknown_strategy_id(tmp_path):
    set_active(None)
    settings = _settings(tmp_path)
    registry = IndexingCollectionRegistry(settings)
    app = FastAPI()
    app.include_router(build_pipeline_router(registry, _embed_client(), settings))

    with TestClient(app) as client:
        response = client.post("/pipeline/load", json={
            "indexing_strategy": "not_real", "retrieval_strategy": "hybrid_rrf", "post_retrieval_strategy": "none",
        })

    assert response.status_code == 422
    registry.close_all()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/pipeline/test_router.py -v`
Expected: FAIL (`ModuleNotFoundError: No module named 'app.pipeline.router'`).

- [ ] **Step 3: Implement**

Create `app/pipeline/router.py`:

```python
from __future__ import annotations

import httpx
from fastapi import APIRouter

from app.config import Settings, settings as default_settings
from app.pipeline.config import PipelineConfig, get_active
from app.pipeline.loader import load_pipeline
from app.pipeline.models import PipelineLoadResult, PipelineStatus
from app.pipeline.registry import INDEXING_STRATEGY_IDS, IndexingCollectionRegistry


def build_pipeline_router(
    registry: IndexingCollectionRegistry,
    http_client: httpx.AsyncClient,
    settings: Settings = default_settings,
) -> APIRouter:
    router = APIRouter()

    @router.post("/pipeline/load", response_model=PipelineLoadResult)
    async def load(config: PipelineConfig) -> PipelineLoadResult:
        return await load_pipeline(config, registry, http_client, settings)

    @router.get("/pipeline/status", response_model=PipelineStatus)
    async def status() -> PipelineStatus:
        return PipelineStatus(
            active=get_active(),
            doc_counts={strategy_id: registry.doc_count(strategy_id) for strategy_id in INDEXING_STRATEGY_IDS},
        )

    return router
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/pipeline/test_router.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add app/pipeline/router.py tests/pipeline/test_router.py
git commit -m "feat: add POST /pipeline/load and GET /pipeline/status endpoints"
```

---

### Task 11: Rewire `/query`, delete Judge + URL-crawl ingestion, wire `app/main.py`

**Files:**
- Modify: `app/retrieval/router.py`, `app/retrieval/models.py`, `app/main.py`, `app/config.py`
- Modify: `tests/retrieval/test_router.py`, `tests/retrieval/test_models.py`, `tests/ingestion/test_config.py` (moved to `tests/test_config.py`)
- Delete: `app/retrieval/judge.py`, `tests/retrieval/test_judge.py`
- Delete: `app/ingestion/` (entire directory: `__init__.py`, `fetcher.py`, `pagination.py`, `job_store.py`, `worker.py`, `router.py`, `models.py`)
- Delete: `tests/ingestion/` (entire directory: `test_extractor.py`, `test_fetcher.py`, `test_integration.py`, `test_job_store.py`, `test_main.py`, `test_pagination.py`, `test_router.py`, `test_worker.py`, `test_models.py`, plus `test_config.py` which moves out first)

**Interfaces:**
- Produces: `build_retrieval_router(registry, embedding_client, synthesis_client, settings=default_settings) -> APIRouter` (replaces the old fixed-`bm25_index`/`vector_index`/`chunk_store` signature). `/query` returns `400` when no pipeline is active. `QueryResponse` drops `judge_attempts`.

- [ ] **Step 1: Write the failing retrieval-router and model tests**

Replace `tests/retrieval/test_models.py`:

```python
from app.extraction.preferences import QueryPreferences
from app.retrieval.models import Citation, FusedChunk, QueryRequest, QueryResponse


def test_query_request_top_k_optional():
    request = QueryRequest(query="what is RAG?")
    assert request.top_k is None


def test_fused_chunk_defaults_used_in_synthesis_false():
    chunk = FusedChunk(
        chunk_id="c1", text="hello", source_url="https://example.com", page_number=1,
        bm25_rank=1, bm25_score=8.3, semantic_rank=None, semantic_score=None,
        fused_rank=1, rrf_score=0.016, matched_methods=["bm25"],
    )
    assert chunk.used_in_synthesis is False


def test_fused_chunk_city_and_price_default_none():
    chunk = FusedChunk(
        chunk_id="c1", text="hello", source_url="https://example.com", page_number=1,
        bm25_rank=1, bm25_score=8.3, semantic_rank=None, semantic_score=None,
        fused_rank=1, rrf_score=0.016, matched_methods=["bm25"],
    )
    assert chunk.city is None
    assert chunk.price is None


def test_query_response_round_trip():
    response = QueryResponse(
        query="q", answer="answer [1]", citations=[Citation(marker=1, chunk_id="c1")],
        retrieved_chunks=[], preferences=QueryPreferences(), filtered_out_count=0,
    )
    assert response.model_dump()["citations"][0]["chunk_id"] == "c1"
    assert "judge_attempts" not in response.model_dump()
```

Replace `tests/retrieval/test_router.py`:

```python
import json

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.config import Settings
from app.indexing.models import ChunkMetadata
from app.pipeline.config import PipelineConfig, set_active
from app.pipeline.registry import IndexingCollectionRegistry
from app.retrieval.router import build_retrieval_router


def _settings(tmp_path, **overrides) -> Settings:
    defaults = dict(qdrant_path=str(tmp_path / "qdrant"), qdrant_collection="t", vector_size=2, groq_api_key="test-key")
    defaults.update(overrides)
    return Settings(**defaults)


def _embed_handler(request):
    return httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})


def test_query_without_active_pipeline_returns_400(tmp_path):
    set_active(None)
    settings = _settings(tmp_path)
    registry = IndexingCollectionRegistry(settings)
    embedding_client = httpx.AsyncClient(transport=httpx.MockTransport(_embed_handler))
    synthesis_client = httpx.AsyncClient(transport=httpx.MockTransport(_embed_handler))
    app = FastAPI()
    app.include_router(build_retrieval_router(registry, embedding_client, synthesis_client, settings))

    with TestClient(app) as client:
        response = client.post("/query", json={"query": "anything"})

    assert response.status_code == 400
    registry.close_all()


def test_query_uses_active_pipeline_returns_answer_with_citations_and_chunks(tmp_path):
    set_active(None)
    settings = _settings(tmp_path)
    registry = IndexingCollectionRegistry(settings)
    collection = registry.get("fixed_window")

    chunk_id = "11111111-1111-1111-1111-111111111111"
    chunk = ChunkMetadata(
        chunk_id=chunk_id, doc_id="d1", doc_id_hash="h1", source_url="paper.pdf", page_number=1,
        chunk_index=0, char_start=0, char_end=20, overlap_with_prev=0,
        indexed_at="2026-07-29T12:00:00+00:00", text="the quick brown fox",
    )
    collection.bm25_index.add_documents([chunk_id], ["the quick brown fox"])
    collection.vector_index.upsert([chunk_id], [[1.0, 0.0]], [chunk.model_dump()])
    collection.chunk_store.add([chunk])

    set_active(PipelineConfig(indexing_strategy="fixed_window", retrieval_strategy="hybrid_rrf", post_retrieval_strategy="none"))

    def groq_handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "The fox is quick [1]."}}]})

    embedding_client = httpx.AsyncClient(transport=httpx.MockTransport(_embed_handler))
    synthesis_client = httpx.AsyncClient(transport=httpx.MockTransport(groq_handler))
    app = FastAPI()
    app.include_router(build_retrieval_router(registry, embedding_client, synthesis_client, settings))

    with TestClient(app) as client:
        response = client.post("/query", json={"query": "tell me about the fox"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "The fox is quick [1]."
    assert body["citations"] == [{"marker": 1, "chunk_id": chunk_id}]
    assert body["retrieved_chunks"][0]["chunk_id"] == chunk_id
    assert body["retrieved_chunks"][0]["used_in_synthesis"] is True
    assert "judge_attempts" not in body
    registry.close_all()
    set_active(None)


def test_query_includes_preferences_and_filtered_out_count_with_metadata_filter_strategy(tmp_path):
    set_active(None)
    settings = _settings(tmp_path, qdrant_collection="t2")
    registry = IndexingCollectionRegistry(settings)
    collection = registry.get("fixed_window")

    kept_id = "22222222-2222-2222-2222-222222222222"
    excluded_id = "33333333-3333-3333-3333-333333333333"
    kept_chunk = ChunkMetadata(
        chunk_id=kept_id, doc_id="d1", doc_id_hash="h1", source_url="paper.pdf", page_number=1,
        chunk_index=0, char_start=0, char_end=10, overlap_with_prev=0,
        indexed_at="2026-07-29T12:00:00+00:00", text="a hotel in lahore", city="lahore", price=None,
    )
    excluded_chunk = ChunkMetadata(
        chunk_id=excluded_id, doc_id="d1", doc_id_hash="h1", source_url="paper.pdf", page_number=1,
        chunk_index=1, char_start=10, char_end=20, overlap_with_prev=0,
        indexed_at="2026-07-29T12:00:00+00:00", text="a hotel in paris", city="paris", price=None,
    )
    collection.bm25_index.add_documents([kept_id, excluded_id], ["a hotel in lahore", "a hotel in paris"])
    collection.vector_index.upsert(
        [kept_id, excluded_id], [[1.0, 0.0], [0.9, 0.1]],
        [kept_chunk.model_dump(), excluded_chunk.model_dump()],
    )
    collection.chunk_store.add([kept_chunk, excluded_chunk])

    set_active(PipelineConfig(indexing_strategy="fixed_window", retrieval_strategy="hybrid_rrf", post_retrieval_strategy="metadata_filter"))

    def groq_handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "A hotel in Lahore [1]."}}]})

    embedding_client = httpx.AsyncClient(transport=httpx.MockTransport(_embed_handler))
    synthesis_client = httpx.AsyncClient(transport=httpx.MockTransport(groq_handler))
    app = FastAPI()
    app.include_router(build_retrieval_router(registry, embedding_client, synthesis_client, settings))

    with TestClient(app) as client:
        response = client.post("/query", json={"query": "hotel in Lahore"})

    assert response.status_code == 200
    body = response.json()
    assert body["preferences"]["city"] == "lahore"
    assert body["filtered_out_count"] == 1
    assert [c["chunk_id"] for c in body["retrieved_chunks"]] == [kept_id]
    registry.close_all()
    set_active(None)


def test_query_default_settings_marks_overflow_chunks_not_used_in_synthesis(tmp_path):
    set_active(None)
    settings = _settings(tmp_path, qdrant_collection="t3")
    assert settings.display_top_k == 8
    assert settings.synthesis_context_budget == 6

    registry = IndexingCollectionRegistry(settings)
    collection = registry.get("fixed_window")

    chunk_ids = [f"4444444{i}-4444-4444-4444-444444444444" for i in range(8)]
    chunks = [
        ChunkMetadata(
            chunk_id=chunk_id, doc_id="d1", doc_id_hash="h1", source_url="paper.pdf", page_number=1,
            chunk_index=index, char_start=0, char_end=20, overlap_with_prev=0,
            indexed_at="2026-07-29T12:00:00+00:00", text=f"chunk number {index} about the quick brown fox",
        )
        for index, chunk_id in enumerate(chunk_ids)
    ]
    vector_list = [[1.0, index * 0.1] for index in range(8)]
    collection.vector_index.upsert(chunk_ids, vector_list, [chunk.model_dump() for chunk in chunks])
    collection.chunk_store.add(chunks)

    set_active(PipelineConfig(indexing_strategy="fixed_window", retrieval_strategy="semantic_only", post_retrieval_strategy="none"))

    def groq_handler(request):
        return httpx.Response(200, json={"choices": [{"message": {"content": "A generic answer with no citations."}}]})

    embedding_client = httpx.AsyncClient(transport=httpx.MockTransport(_embed_handler))
    synthesis_client = httpx.AsyncClient(transport=httpx.MockTransport(groq_handler))
    app = FastAPI()
    app.include_router(build_retrieval_router(registry, embedding_client, synthesis_client, settings))

    with TestClient(app) as client:
        response = client.post("/query", json={"query": "tell me about the fox"})

    assert response.status_code == 200
    body = response.json()
    retrieved = body["retrieved_chunks"]
    assert len(retrieved) == 8
    assert [chunk["chunk_id"] for chunk in retrieved] == chunk_ids

    used_flags = [chunk["used_in_synthesis"] for chunk in retrieved]
    assert used_flags[:6] == [True] * 6
    assert used_flags[6:] == [False] * 2
    registry.close_all()
    set_active(None)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/retrieval/test_router.py tests/retrieval/test_models.py -v`
Expected: FAIL (old `build_retrieval_router` signature, `judge_attempts` still required, `app.retrieval.judge`/`app.retrieval.filtering` imports broken from Tasks 5/6).

- [ ] **Step 3: Implement — `app/retrieval/models.py`**

Replace `app/retrieval/models.py`:

```python
from __future__ import annotations

from pydantic import BaseModel

from app.extraction.preferences import QueryPreferences


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
    city: str | None = None
    price: float | None = None
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
    preferences: QueryPreferences
    filtered_out_count: int
```

- [ ] **Step 4: Implement — `app/retrieval/router.py`**

Replace `app/retrieval/router.py`:

```python
from __future__ import annotations

import httpx
from fastapi import APIRouter, HTTPException

from app.config import Settings, settings as default_settings
from app.extraction.preferences import extract_preferences
from app.pipeline.config import get_active
from app.pipeline.registry import IndexingCollectionRegistry
from app.postretrieval.strategies import POST_RETRIEVAL_STRATEGIES
from app.retrieval.models import QueryRequest, QueryResponse
from app.retrieval.strategies import RETRIEVAL_STRATEGIES
from app.retrieval.synthesis import synthesize_answer


def build_retrieval_router(
    registry: IndexingCollectionRegistry,
    embedding_client: httpx.AsyncClient,
    synthesis_client: httpx.AsyncClient,
    settings: Settings = default_settings,
) -> APIRouter:
    router = APIRouter()

    @router.post("/query", response_model=QueryResponse)
    async def query(request: QueryRequest) -> QueryResponse:
        active = get_active()
        if active is None:
            raise HTTPException(status_code=400, detail="no pipeline loaded — call POST /pipeline/load first")

        collection = registry.get(active.indexing_strategy)
        retrieval_fn = RETRIEVAL_STRATEGIES[active.retrieval_strategy]
        post_retrieval_fn = POST_RETRIEVAL_STRATEGIES[active.post_retrieval_strategy]

        preferences = extract_preferences(request.query)

        fused_chunks = await retrieval_fn(
            request.query, collection.bm25_index, collection.vector_index, collection.chunk_store,
            embedding_client, settings, request.top_k,
        )
        kept_chunks, filtered_out_count = post_retrieval_fn(fused_chunks, preferences)

        answer, citations, used_chunk_ids = await synthesize_answer(request.query, kept_chunks, synthesis_client, settings)
        for chunk in kept_chunks:
            chunk.used_in_synthesis = chunk.chunk_id in used_chunk_ids

        return QueryResponse(
            query=request.query, answer=answer, citations=citations, retrieved_chunks=kept_chunks,
            preferences=preferences, filtered_out_count=filtered_out_count,
        )

    return router
```

Delete `app/retrieval/judge.py` and `tests/retrieval/test_judge.py`.

- [ ] **Step 5: Run retrieval tests to verify they pass**

Run: `uv run pytest tests/retrieval -v`
Expected: all PASS.

- [ ] **Step 6: Remove obsolete Settings fields**

In `app/config.py`, remove `user_agent`, `max_pages`, `min_extract_length` (only ever used by the now-deleted `app/ingestion/fetcher.py`, `worker.py`, `extractor.py`). Result:

```python
import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Settings:
    fetch_timeout_seconds: float = 10.0
    chunk_size_tokens: int = 400
    chunk_overlap_tokens: int = 75
    embedding_model: str = "qwen3-embedding:0.6b"
    ollama_base_url: str = "http://localhost:11434"
    qdrant_url: str | None = None
    qdrant_path: str = ".data/qdrant"
    qdrant_collection: str = "rag_chunks"
    vector_size: int = 1024
    retrieval_top_k: int = 20
    display_top_k: int = 8
    rrf_k: int = 60
    synthesis_context_budget: int = 6
    groq_model: str = "openai/gpt-oss-120b"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_api_key: str = field(default_factory=lambda: os.environ.get("GROQ_API_KEY", ""))


settings = Settings()
```

(`judge_model` and `judge_retry_top_k_multiplier` are also removed here — nothing references them once `app/retrieval/judge.py` is gone.)

Create `tests/test_config.py` with the content below — this supersedes `tests/ingestion/test_config.py` (dropping its assertions on the three removed fields; `test_settings_is_frozen` targets `chunk_size_tokens` instead of the removed `max_pages` since a frozen dataclass raises `FrozenInstanceError` on any attribute assignment regardless of which field, so any surviving field proves it). Step 8 deletes the old `tests/ingestion/test_config.py` along with the rest of the ingestion test directory.

```python
from app.config import Settings, settings


def test_defaults():
    assert settings.fetch_timeout_seconds == 10.0


def test_settings_is_frozen():
    with __import__("pytest").raises(Exception):
        settings.chunk_size_tokens = 5


def test_indexing_defaults():
    assert settings.chunk_size_tokens == 400
    assert settings.chunk_overlap_tokens == 75
    assert settings.embedding_model == "qwen3-embedding:0.6b"
    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.qdrant_url is None
    assert settings.qdrant_collection == "rag_chunks"
    assert settings.vector_size == 1024


def test_retrieval_and_synthesis_defaults(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    import importlib
    import app.config as config_module
    importlib.reload(config_module)

    assert config_module.settings.retrieval_top_k == 20
    assert config_module.settings.display_top_k == 8
    assert config_module.settings.rrf_k == 60
    assert config_module.settings.synthesis_context_budget == 6
    assert config_module.settings.groq_model == "openai/gpt-oss-120b"
    assert config_module.settings.groq_api_key == ""
```

- [ ] **Step 7: Run to verify config tests pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: all PASS.

- [ ] **Step 8: Delete the ingestion package and its tests**

```bash
git rm -r app/ingestion tests/ingestion
```

(`tests/ingestion/test_config.py` was already moved out in Step 6, so this only removes the URL-crawl-specific files.)

- [ ] **Step 9: Rewire `app/main.py`**

Replace `app/main.py`:

```python
from __future__ import annotations

import httpx
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import Settings, settings
from app.pipeline.registry import IndexingCollectionRegistry
from app.pipeline.router import build_pipeline_router
from app.retrieval.router import build_retrieval_router


def create_app(app_settings: Settings = settings) -> FastAPI:
    app = FastAPI(title="RAG Pipeline Showcase")

    registry = IndexingCollectionRegistry(app_settings)
    embedding_client = httpx.AsyncClient()
    synthesis_client = httpx.AsyncClient()

    app.include_router(build_pipeline_router(registry, embedding_client, app_settings))
    app.include_router(build_retrieval_router(registry, embedding_client, synthesis_client, app_settings))

    app.mount("/static", StaticFiles(directory="app/static"), name="static")

    @app.get("/")
    async def serve_ingest_page() -> FileResponse:
        return FileResponse("app/static/ingest.html")

    @app.get("/query-ui")
    async def serve_query_page() -> FileResponse:
        return FileResponse("app/static/query.html")

    return app


app = create_app()
```

- [ ] **Step 10: Run the full suite**

Run: `uv run pytest -q`
Expected: all PASS, including `tests/frontend/*` — those tests assert against `app/static/*.html`/`*.js` content, which this task hasn't touched yet, so their existing assertions (`"Ingest URLs"`, `id="judge-panel"`, `function postIngest`) still match. Task 12 changes the static files and their assertions together. If anything fails here, investigate before continuing — the known red window from Tasks 5-7 should be fully resolved by this task.

- [ ] **Step 11: Commit**

```bash
git add app/retrieval/router.py app/retrieval/models.py app/main.py app/config.py tests/retrieval/test_router.py tests/retrieval/test_models.py tests/test_config.py
git rm app/retrieval/judge.py tests/retrieval/test_judge.py
git commit -m "feat: rewire /query through ACTIVE_PIPELINE, remove Judge and URL-crawl ingestion"
```

---

### Task 12: Frontend cleanup + final verification

**Files:**
- Modify: `app/static/ingest.html`, `app/static/js/ingest.js`, `app/static/query.html`, `app/static/js/query.js`, `app/static/js/api.js`
- Modify: `tests/frontend/test_ingest_page.py`, `tests/frontend/test_query_page.py`, `tests/frontend/test_static_assets.py`

**Interfaces:**
- No new Python interfaces — this task only makes the two existing HTML pages stop referencing deleted backend endpoints/fields. The full strategy-picker + Load + eval-sidebar UI is piece D's job (see the spec's Non-Goals); this task's only obligation is "the app doesn't reference dead code."

- [ ] **Step 1: Write the failing frontend tests**

Replace `tests/frontend/test_ingest_page.py`:

```python
import uuid

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def _isolated_settings(tmp_path) -> Settings:
    return Settings(qdrant_path=str(tmp_path / "qdrant"), qdrant_collection=f"test-{uuid.uuid4()}", vector_size=2)


def test_ingest_page_served_at_root(tmp_path):
    app = create_app(_isolated_settings(tmp_path))
    with TestClient(app) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "Archive" in response.text
    assert 'id="doc-counts"' in response.text
    assert '/static/js/ingest.js' in response.text
```

Replace `tests/frontend/test_query_page.py`:

```python
import uuid

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def _isolated_settings(tmp_path) -> Settings:
    return Settings(qdrant_path=str(tmp_path / "qdrant"), qdrant_collection=f"test-{uuid.uuid4()}", vector_size=2)


def test_query_page_served(tmp_path):
    app = create_app(_isolated_settings(tmp_path))
    with TestClient(app) as client:
        response = client.get("/query-ui")
    assert response.status_code == 200
    assert "Ask a question" in response.text
    assert '/static/js/query.js' in response.text
    assert 'id="preferences"' in response.text
    assert 'id="filtered-note"' in response.text
    assert 'id="judge-panel"' not in response.text
```

In `tests/frontend/test_static_assets.py`, replace the `test_shared_api_js_is_served` assertions:

```python
def test_shared_api_js_is_served(tmp_path):
    app = create_app(_isolated_settings(tmp_path))
    with TestClient(app) as client:
        response = client.get("/static/js/api.js")
    assert response.status_code == 200
    assert "function getPipelineStatus" in response.text
    assert "function postQuery" in response.text
    assert "function postIngest" not in response.text
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/frontend -v`
Expected: FAIL (pages still contain the old content).

- [ ] **Step 3: Implement — `app/static/ingest.html` and `app/static/js/ingest.js`**

Replace `app/static/ingest.html`:

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>RAG Ingest</title>
  <link rel="stylesheet" href="/static/css/app.css" />
</head>
<body>
  <nav><a href="/">Ingest</a><a href="/query-ui">Query</a></nav>
  <h1>Archive</h1>
  <p>PDFs are read from the local <code>archive/</code> folder and indexed per strategy via <code>POST /pipeline/load</code>. This screen shows current indexed status per strategy.</p>
  <div id="doc-counts"></div>

  <script src="/static/js/api.js"></script>
  <script src="/static/js/ingest.js"></script>
</body>
</html>
```

Replace `app/static/js/ingest.js`:

```javascript
async function renderPipelineStatus() {
  const status = await getPipelineStatus();
  const container = document.getElementById("doc-counts");
  container.innerHTML = "";
  for (const [strategy, count] of Object.entries(status.doc_counts)) {
    const row = document.createElement("div");
    row.className = "row";

    const label = document.createElement("span");
    label.textContent = strategy;

    const badge = document.createElement("span");
    badge.className = "badge";
    badge.textContent = `${count} doc(s) indexed`;

    row.appendChild(label);
    row.appendChild(badge);
    container.appendChild(row);
  }
}

renderPipelineStatus();
```

- [ ] **Step 4: Implement — `app/static/js/api.js`**

Replace `app/static/js/api.js`:

```javascript
async function getPipelineStatus() {
  const response = await fetch("/pipeline/status");
  if (!response.ok) {
    throw new Error(`GET /pipeline/status failed: ${response.status}`);
  }
  return response.json();
}

async function postQuery(query, topK) {
  const response = await fetch("/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: topK ?? null }),
  });
  if (!response.ok) {
    throw new Error(`POST /query failed: ${response.status}`);
  }
  return response.json();
}
```

- [ ] **Step 5: Implement — `app/static/query.html` and `app/static/js/query.js`**

In `app/static/query.html`, remove the `<aside id="judge-panel"></aside>` line (keep everything else, including the `page-layout`/`main-column` wrapper divs, unchanged).

In `app/static/js/query.js`, delete the `renderJudgePanel` function entirely and remove the `renderJudgePanel(result.judge_attempts);` call inside the `ask` button handler.

- [ ] **Step 6: Run to verify frontend tests pass**

Run: `uv run pytest tests/frontend -v`
Expected: all PASS.

- [ ] **Step 7: Run the entire test suite**

Run: `uv run pytest -q`
Expected: all PASS, zero skips beyond any pre-existing ones unrelated to this work.

- [ ] **Step 8: Manual smoke check**

Run: `uv run uvicorn app.main:app --reload` and in a browser:
- Visit `http://localhost:8000/` — confirm the Archive page loads and shows 4 rows (one per indexing strategy) all at `0 doc(s) indexed` (assuming no prior `/pipeline/load` call against the real `.data/qdrant` store).
- `curl -X POST localhost:8000/pipeline/load -H 'Content-Type: application/json' -d '{"indexing_strategy":"fixed_window","retrieval_strategy":"hybrid_rrf","post_retrieval_strategy":"none"}'` — confirm it indexes the real PDFs in `archive/` (requires Ollama running locally for embeddings) and returns a 200 with `indexed.new_docs > 0`.
- Reload `/` — confirm the doc count for `fixed_window` now reflects the indexed archive.
- Visit `http://localhost:8000/query-ui`, ask a question — confirm an answer with citations renders and there's no judge panel/column artifact in the layout.

- [ ] **Step 9: Commit**

```bash
git add app/static tests/frontend
git commit -m "feat: update frontend for archive-based ingestion, remove judge panel"
```

---

## Post-Plan Note

This plan implements piece A only (per the approved spec's four-piece breakdown). At the end of Task 12:
- `semantic`, `hierarchical`, `hierarchical_summary` indexing strategies and `cross_encoder_rerank` post-retrieval strategy are registered but raise `NotImplementedError` — piece B fills these in.
- `PipelineLoadResult.eval` always returns `{"status": "not_implemented"}` — piece C replaces the stub with real gold-test-case metrics.
- The UI has no strategy pickers or Load button — piece D adds the combined pipeline screen.

Each of B, C, D should go through its own brainstorming → spec → plan cycle before implementation, per the earlier decomposition decision, since each depends on the previous piece's real (non-stub) interfaces existing first.
