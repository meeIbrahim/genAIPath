# Preference Extraction, City/Price Filtering, and Context Quality Judge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add non-LLM preference extraction (city/budget/interests) from the query, permissive city/price filtering of retrieved chunks, and a Groq-backed context quality judge (with one retry before a fixed fallback answer) to the existing `POST /query` pipeline — plus Query screen surfacing of detected preferences and the filtered-out count.

**Architecture:** Three new modules (`app/extraction/`, `app/retrieval/filtering.py`, `app/retrieval/judge.py`) compose in front of and around the existing `retrieve() → synthesize_answer()` flow inside `app/retrieval/router.py`. Chunk metadata gains `city`/`price` fields tagged at index time using the same extractors used on the query, so filtering has real data to act on. No new Qdrant migration (payload is a generic dict already).

**Tech Stack:** spaCy (`en_core_web_sm`) for city NER, stdlib `re` for price extraction, a static dict for interest keywords — all offline, no LLM call. Judge reuses the existing raw-`httpx`-to-Groq pattern from `synthesis.py` (not LangChain, which remains an unused dependency).

## Global Constraints

- No LangChain for the judge — follow the existing raw-`httpx` Groq chat-completions pattern from `app/retrieval/synthesis.py`.
- Filtering is permissive on missing metadata: a chunk is excluded ONLY if it has a non-null `city`/`price` that conflicts with a stated preference. Chunks with `city=None`/`price=None` always pass through.
- No city alias/canonicalization — exact case-insensitive string match only ("nyc" ≠ "new york").
- No use of `interests` for retrieval boosting/re-ranking/query-rewriting in this plan — captured and returned only.
- No backfill/migration tooling for chunks indexed before this change.
- Exactly one retry on `context_insufficient` — no configurable retry count.
- Judge failure (`JudgeError`) is fail-closed: treated as `context_insufficient`, never silently proceeds to synthesis.
- Fallback response is `200 OK`, same `QueryResponse` shape: `answer` = the fixed string `"I don't have enough reliable information in the indexed content to answer this question confidently."`, `citations=[]`, `retrieved_chunks` = the last attempt's filtered chunks with `used_in_synthesis=False`.
- Test convention: mirror `app/` structure under `tests/`, no `conftest.py`, `httpx.MockTransport` for external calls, `asyncio_mode=auto` (no `@pytest.mark.asyncio` needed), `tmp_path`-isolated Qdrant per test.
- Router/module pattern: settings-as-last-param-with-default (`settings: Settings = default_settings`), matching every existing module.

---

## File Structure

```
app/
  extraction/                       # NEW package
    __init__.py
    location.py                     # extract_city
    price.py                        # extract_price
    interests.py                    # extract_interests
    preferences.py                  # QueryPreferences, extract_preferences
  indexing/
    models.py                       # MODIFY: ChunkMetadata + city, price
    indexer.py                      # MODIFY: tag city/price per chunk
  retrieval/
    models.py                       # MODIFY: FusedChunk + city/price; QueryResponse + preferences, filtered_out_count
    retriever.py                    # MODIFY: carry city/price into FusedChunk
    filtering.py                    # NEW: filter_chunks
    judge.py                        # NEW: JudgeVerdict, JudgeError, judge_context
    router.py                       # MODIFY: orchestrate preferences -> retrieve -> filter -> judge -> retry -> fallback/synthesis
  config.py                         # MODIFY: + judge_model, judge_retry_top_k_multiplier
  static/
    query.html                      # MODIFY: + #preferences, #filtered-note containers
    js/query.js                     # MODIFY: + renderPreferences, renderFilteredNote
    css/app.css                     # MODIFY: + #preferences, #filtered-note rules
pyproject.toml                       # MODIFY: + spacy dependency
tests/
  extraction/                       # NEW
    __init__.py
    test_location.py
    test_price.py
    test_interests.py
    test_preferences.py
  indexing/
    test_indexer.py                 # MODIFY: assert city/price tagged
  retrieval/
    test_retriever.py               # MODIFY: assert city/price carried through
    test_models.py                  # MODIFY: assert new fields
    test_filtering.py               # NEW
    test_judge.py                   # NEW
    test_router.py                  # MODIFY: retry/fallback/preferences integration tests
  frontend/
    test_query_page.py              # MODIFY: assert new containers present
```

---

### Task 1: Preference extraction package

**Files:**
- Create: `app/extraction/__init__.py`
- Create: `app/extraction/location.py`
- Create: `app/extraction/price.py`
- Create: `app/extraction/interests.py`
- Create: `app/extraction/preferences.py`
- Modify: `pyproject.toml` (add `spacy` dependency)
- Test: `tests/extraction/__init__.py`, `tests/extraction/test_location.py`, `tests/extraction/test_price.py`, `tests/extraction/test_interests.py`, `tests/extraction/test_preferences.py`

**Interfaces:**
- Produces: `extract_city(text: str) -> str | None`, `extract_price(text: str) -> float | None`, `extract_interests(text: str) -> list[str]`, `QueryPreferences(BaseModel)` with fields `city: str | None`, `budget: float | None`, `interests: list[str]`, and `extract_preferences(query: str) -> QueryPreferences`. These are pure, synchronous, no I/O — later tasks import them directly.

- [ ] **Step 1: Add spaCy and download the English model**

```bash
uv add spacy
uv run python -m spacy download en_core_web_sm
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/extraction/__init__.py
```

```python
# tests/extraction/test_location.py
from app.extraction.location import extract_city


def test_extract_city_finds_gpe_entity():
    assert extract_city("I want to visit Paris next month") == "paris"


def test_extract_city_finds_first_of_multiple():
    assert extract_city("Flying from London to Tokyo") == "london"


def test_extract_city_returns_none_when_absent():
    assert extract_city("I like hiking and cheap food") is None


def test_extract_city_finds_lahore():
    # Exercised again in Task 5's router integration test — confirmed here first,
    # at the unit level, so a spaCy GPE-tagging surprise fails fast in this task.
    assert extract_city("A hotel in Lahore") == "lahore"
```

```python
# tests/extraction/test_price.py
from app.extraction.price import extract_price


def test_extract_price_dollar_amount():
    assert extract_price("Looking for a hotel under $500 in Lahore") == 500.0


def test_extract_price_k_suffix():
    assert extract_price("My budget is 20k") == 20000.0


def test_extract_price_comma_separated():
    assert extract_price("around 2,500 for the trip") == 2500.0


def test_extract_price_returns_none_when_absent():
    assert extract_price("I want a nice hotel in Lahore") is None
```

```python
# tests/extraction/test_interests.py
from app.extraction.interests import extract_interests


def test_extract_interests_multiple_categories():
    assert extract_interests("I love hiking and trying local food") == ["outdoors", "food"]


def test_extract_interests_dedupes_same_category():
    assert extract_interests("I love hiking and camping") == ["outdoors"]


def test_extract_interests_returns_empty_list_when_absent():
    assert extract_interests("What is the capital of France?") == []
```

```python
# tests/extraction/test_preferences.py
from app.extraction.preferences import QueryPreferences, extract_preferences


def test_extract_preferences_combines_all_three():
    prefs = extract_preferences("Looking for a hotel under $500 in Paris, I love hiking")
    assert prefs == QueryPreferences(city="paris", budget=500.0, interests=["outdoors"])


def test_extract_preferences_all_none_when_nothing_detected():
    # No GPE, no price signal, no interest keyword — unlike "capital of France",
    # which spaCy correctly tags "France" as a GPE (so city would NOT be None there).
    prefs = extract_preferences("What is the square root of 144?")
    assert prefs.city is None
    assert prefs.budget is None
    assert prefs.interests == []
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/extraction/ -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.extraction'`

- [ ] **Step 4: Implement the extraction package**

```python
# app/extraction/__init__.py
```

```python
# app/extraction/location.py
from __future__ import annotations

import spacy

_nlp = spacy.load("en_core_web_sm")


def extract_city(text: str) -> str | None:
    doc = _nlp(text)
    for ent in doc.ents:
        if ent.label_ == "GPE":
            return ent.text.lower()
    return None
```

```python
# app/extraction/price.py
from __future__ import annotations

import re

_PRICE_RE = re.compile(
    r"\$\s?(?P<dollar>\d[\d,]*\.?\d*)(?P<dollar_k>[kK])?"
    r"|(?:under|less than|budget of|budget is|around|for)\s+\$?\s?(?P<qualified>\d[\d,]*\.?\d*)(?P<qualified_k>[kK])?"
    r"|(?P<bare_k>\d[\d,]*\.?\d*)\s?[kK]\b",
    re.IGNORECASE,
)


def extract_price(text: str) -> float | None:
    match = _PRICE_RE.search(text)
    if not match:
        return None

    groups = match.groupdict()
    if groups["dollar"] is not None:
        raw, is_k = groups["dollar"], groups["dollar_k"]
    elif groups["qualified"] is not None:
        raw, is_k = groups["qualified"], groups["qualified_k"]
    else:
        raw, is_k = groups["bare_k"], "k"

    value = float(raw.replace(",", ""))
    return value * 1000 if is_k else value
```

```python
# app/extraction/interests.py
from __future__ import annotations

_INTEREST_KEYWORDS: dict[str, str] = {
    "hiking": "outdoors",
    "trek": "outdoors",
    "trekking": "outdoors",
    "camping": "outdoors",
    "food": "food",
    "restaurant": "food",
    "cuisine": "food",
    "museum": "culture",
    "history": "culture",
    "art": "culture",
    "shopping": "shopping",
    "nightlife": "nightlife",
}


def extract_interests(text: str) -> list[str]:
    lowered = text.lower()
    matched: list[str] = []
    for keyword, category in _INTEREST_KEYWORDS.items():
        if keyword in lowered and category not in matched:
            matched.append(category)
    return matched
```

```python
# app/extraction/preferences.py
from __future__ import annotations

from pydantic import BaseModel, Field

from app.extraction.interests import extract_interests
from app.extraction.location import extract_city
from app.extraction.price import extract_price


class QueryPreferences(BaseModel):
    city: str | None = None
    budget: float | None = None
    interests: list[str] = Field(default_factory=list)


def extract_preferences(query: str) -> QueryPreferences:
    return QueryPreferences(
        city=extract_city(query),
        budget=extract_price(query),
        interests=extract_interests(query),
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/extraction/ -v`
Expected: PASS (11 tests)

- [ ] **Step 6: Commit**

```bash
git add app/extraction/ tests/extraction/ pyproject.toml uv.lock
git commit -m "feat: add non-LLM preference extraction (city, budget, interests)"
```

---

### Task 2: Propagate city/price through the chunk pipeline

**Files:**
- Modify: `app/indexing/models.py` (`ChunkMetadata` + `city`, `price`)
- Modify: `app/indexing/indexer.py` (tag `city`/`price` per chunk at index time)
- Modify: `app/retrieval/models.py` (`FusedChunk` + `city`, `price`)
- Modify: `app/retrieval/retriever.py` (carry `city`/`price` from chunk metadata into `FusedChunk`)
- Test: `tests/indexing/test_indexer.py`, `tests/retrieval/test_retriever.py`, `tests/retrieval/test_models.py`

**Interfaces:**
- Consumes: `extract_city`, `extract_price` from Task 1 (`app.extraction.location`, `app.extraction.price`).
- Produces: `ChunkMetadata.city: str | None`, `ChunkMetadata.price: float | None`; `FusedChunk.city: str | None`, `FusedChunk.price: float | None`. Task 3 (filtering) and Task 6 (frontend, via the API response) depend on these exact field names.

- [ ] **Step 1: Write the failing tests**

```python
# tests/indexing/test_indexer.py — ADD these two tests (keep existing ones)
async def test_index_document_tags_city_and_price(tmp_path):
    settings = _settings(tmp_path)
    bm25 = InMemoryBM25Index()
    vectors = QdrantVectorIndex(settings)
    store = ChunkStore()

    payload = IngestionPayload(
        source_url="https://example.com/post",
        cleaned_text="A budget hotel in Paris costs around $500 per night.",
        pages_fetched=1,
        fetched_at="2026-07-29T12:00:00+00:00",
        page_map=[PageMapEntry(page=1, char_start=0, char_end=54)],
    )

    def handler(request):
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3]]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await index_document(payload, bm25, vectors, store, client, settings)

    stored = store.get_many([c.chunk_id for c in store._chunks.values()])
    assert result.chunk_count == 1
    assert stored[0].city == "paris"
    assert stored[0].price == 500.0


async def test_index_document_leaves_city_and_price_none_when_absent(tmp_path):
    settings = _settings(tmp_path)
    bm25 = InMemoryBM25Index()
    vectors = QdrantVectorIndex(settings)
    store = ChunkStore()

    payload = _payload()  # "one two three. four five six." — no city/price

    def handler(request):
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await index_document(payload, bm25, vectors, store, client, settings)

    stored = store.get_many([c.chunk_id for c in store._chunks.values()])
    assert all(c.city is None for c in stored)
    assert all(c.price is None for c in stored)
```

```python
# tests/retrieval/test_retriever.py — MODIFY _seed() to tag the chunk, ADD one test
def _seed(tmp_path):
    settings = Settings(qdrant_path=str(tmp_path / "qdrant"), qdrant_collection="t", vector_size=2)
    bm25 = InMemoryBM25Index()
    vectors = QdrantVectorIndex(settings)
    store = ChunkStore()

    chunk_id = "11111111-1111-1111-1111-111111111111"
    chunk = ChunkMetadata(
        chunk_id=chunk_id,
        doc_id="d1",
        source_url="https://example.com",
        page_number=1,
        chunk_index=0,
        char_start=0,
        char_end=20,
        overlap_with_prev=0,
        indexed_at="2026-07-29T12:00:00+00:00",
        text="the quick brown fox",
        city="paris",
        price=500.0,
    )
    bm25.add_documents([chunk_id], ["the quick brown fox"])
    vectors.upsert([chunk_id], [[1.0, 0.0]], [chunk.model_dump()])
    store.add([chunk])
    return settings, bm25, vectors, store, chunk_id


async def test_retrieve_carries_city_and_price_into_fused_chunk(tmp_path):
    settings, bm25, vectors, store, chunk_id = _seed(tmp_path)

    def handler(request):
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        results = await retrieve("fox", bm25, vectors, store, client, settings)

    assert results[0].city == "paris"
    assert results[0].price == 500.0
```

```python
# tests/retrieval/test_models.py — ADD this test
def test_fused_chunk_city_and_price_default_none():
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
    assert chunk.city is None
    assert chunk.price is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/indexing/test_indexer.py tests/retrieval/test_retriever.py tests/retrieval/test_models.py -v`
Expected: FAIL — `city`/`price` are unexpected keyword arguments on `ChunkMetadata`/`FusedChunk`, and `stored[0].city`/`results[0].city` raise `AttributeError`.

- [ ] **Step 3: Implement the schema and wiring changes**

```python
# app/indexing/models.py — full file
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
    city: str | None = None
    price: float | None = None


class IndexResult(BaseModel):
    doc_id: str
    status: str
    chunk_count: int
```

```python
# app/indexing/indexer.py — add import, modify ChunkMetadata construction
# Add near the top, alongside the other app.indexing imports:
from app.extraction.location import extract_city
from app.extraction.price import extract_price

# Inside index_document(), modify the ChunkMetadata construction (same list comprehension):
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
            city=extract_city(text_chunk.text),
            price=extract_price(text_chunk.text),
        )
        for index, text_chunk in enumerate(text_chunks)
    ]
```

```python
# app/retrieval/models.py — FusedChunk gains two fields (full class shown)
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
```

```python
# app/retrieval/retriever.py — modify the FusedChunk construction inside retrieve()
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/indexing/ tests/retrieval/test_retriever.py tests/retrieval/test_models.py -v`
Expected: PASS

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `uv run pytest -q`
Expected: all pass (existing `test_router.py` tests still construct `FusedChunk`/`ChunkMetadata` without `city`/`price` — both are optional with `None` defaults, so no other test should break)

- [ ] **Step 6: Commit**

```bash
git add app/indexing/models.py app/indexing/indexer.py app/retrieval/models.py app/retrieval/retriever.py tests/indexing/test_indexer.py tests/retrieval/test_retriever.py tests/retrieval/test_models.py
git commit -m "feat: tag chunks with city/price at index time and carry through to FusedChunk"
```

---

### Task 3: Metadata filtering

**Files:**
- Create: `app/retrieval/filtering.py`
- Test: `tests/retrieval/test_filtering.py`

**Interfaces:**
- Consumes: `QueryPreferences` (Task 1, `app.extraction.preferences`), `FusedChunk` (Task 2, `app.retrieval.models`, now with `city`/`price`).
- Produces: `filter_chunks(fused_chunks: list[FusedChunk], preferences: QueryPreferences) -> tuple[list[FusedChunk], int]` — Task 5 (router) calls this directly.

- [ ] **Step 1: Write the failing test**

```python
# tests/retrieval/test_filtering.py
from app.extraction.preferences import QueryPreferences
from app.retrieval.filtering import filter_chunks
from app.retrieval.models import FusedChunk


def _chunk(chunk_id: str, city: str | None = None, price: float | None = None) -> FusedChunk:
    return FusedChunk(
        chunk_id=chunk_id,
        text="text",
        source_url="https://example.com",
        page_number=1,
        city=city,
        price=price,
        bm25_rank=1,
        bm25_score=1.0,
        semantic_rank=1,
        semantic_score=1.0,
        fused_rank=1,
        rrf_score=0.03,
        matched_methods=["bm25", "semantic"],
    )


def test_filter_chunks_excludes_conflicting_city():
    chunks = [_chunk("c1", city="paris"), _chunk("c2", city="lahore")]
    kept, excluded_count = filter_chunks(chunks, QueryPreferences(city="lahore"))
    assert [c.chunk_id for c in kept] == ["c2"]
    assert excluded_count == 1


def test_filter_chunks_excludes_over_budget_price():
    chunks = [_chunk("c1", price=1000.0), _chunk("c2", price=100.0)]
    kept, excluded_count = filter_chunks(chunks, QueryPreferences(budget=500.0))
    assert [c.chunk_id for c in kept] == ["c2"]
    assert excluded_count == 1


def test_filter_chunks_permissive_on_missing_metadata():
    chunks = [_chunk("c1", city=None, price=None), _chunk("c2", city="lahore", price=100.0)]
    kept, excluded_count = filter_chunks(chunks, QueryPreferences(city="paris", budget=50.0))
    assert [c.chunk_id for c in kept] == ["c1"]
    assert excluded_count == 1


def test_filter_chunks_passes_everything_with_no_preferences():
    chunks = [_chunk("c1", city="paris", price=1000.0), _chunk("c2")]
    kept, excluded_count = filter_chunks(chunks, QueryPreferences())
    assert len(kept) == 2
    assert excluded_count == 0


def test_filter_chunks_city_match_is_case_insensitive_via_lowercased_storage():
    # Both extraction paths lowercase city, so equality is effectively case-insensitive
    chunks = [_chunk("c1", city="lahore")]
    kept, excluded_count = filter_chunks(chunks, QueryPreferences(city="lahore"))
    assert len(kept) == 1
    assert excluded_count == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/retrieval/test_filtering.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.retrieval.filtering'`

- [ ] **Step 3: Implement filtering**

```python
# app/retrieval/filtering.py
from __future__ import annotations

from app.extraction.preferences import QueryPreferences
from app.retrieval.models import FusedChunk


def filter_chunks(
    fused_chunks: list[FusedChunk],
    preferences: QueryPreferences,
) -> tuple[list[FusedChunk], int]:
    kept = [chunk for chunk in fused_chunks if not _conflicts(chunk, preferences)]
    return kept, len(fused_chunks) - len(kept)


def _conflicts(chunk: FusedChunk, preferences: QueryPreferences) -> bool:
    if chunk.city is not None and preferences.city is not None and chunk.city != preferences.city:
        return True
    if chunk.price is not None and preferences.budget is not None and chunk.price > preferences.budget:
        return True
    return False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/retrieval/test_filtering.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add app/retrieval/filtering.py tests/retrieval/test_filtering.py
git commit -m "feat: add permissive city/price chunk filtering"
```

---

### Task 4: Context Quality Judge

**Files:**
- Modify: `app/config.py` (add `judge_model`)
- Create: `app/retrieval/judge.py`
- Test: `tests/retrieval/test_judge.py`

**Interfaces:**
- Consumes: `FusedChunk` (Task 2), `Settings` (existing).
- Produces: `JudgeVerdict(BaseModel)` with `verdict: Literal["context_good", "context_insufficient"]`; `JudgeError(Exception)`; `judge_context(query: str, chunks: list[FusedChunk], http_client: httpx.AsyncClient, settings: Settings = default_settings) -> JudgeVerdict`. Task 5 (router) calls this directly and catches `JudgeError`.

- [ ] **Step 1: Add the judge model config field**

```python
# app/config.py — add one field to the Settings dataclass, after groq_api_key
    judge_model: str = "llama-3.1-8b-instant"
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/retrieval/test_judge.py
import httpx
import pytest

from app.config import Settings
from app.retrieval.judge import JudgeError, judge_context
from app.retrieval.models import FusedChunk


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


async def test_judge_context_returns_good_verdict():
    settings = Settings(groq_api_key="test-key")
    chunks = [_chunk("c1", "Paris is the capital of France.")]

    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: _groq_response("context_good"))) as client:
        verdict = await judge_context("capital of France?", chunks, client, settings)

    assert verdict.verdict == "context_good"


async def test_judge_context_returns_insufficient_verdict():
    settings = Settings(groq_api_key="test-key")
    chunks = [_chunk("c1", "unrelated text")]

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: _groq_response("context_insufficient"))
    ) as client:
        verdict = await judge_context("capital of France?", chunks, client, settings)

    assert verdict.verdict == "context_insufficient"


async def test_judge_context_uses_judge_model_not_synthesis_model():
    settings = Settings(groq_api_key="test-key")
    captured = {}

    def handler(request):
        import json

        captured["model"] = json.loads(request.read())["model"]
        return _groq_response("context_good")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        await judge_context("q", [_chunk("c1", "x")], client, settings)

    assert captured["model"] == settings.judge_model


async def test_judge_context_empty_chunks_short_circuits_without_request():
    settings = Settings(groq_api_key="test-key")

    def handler(request):
        raise AssertionError("should not make a request for empty chunks")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        verdict = await judge_context("q", [], client, settings)

    assert verdict.verdict == "context_insufficient"


async def test_judge_context_raises_without_api_key():
    settings = Settings(groq_api_key="")
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200))) as client:
        with pytest.raises(JudgeError, match="GROQ_API_KEY"):
            await judge_context("q", [_chunk("c1", "x")], client, settings)


async def test_judge_context_raises_on_non_200():
    settings = Settings(groq_api_key="test-key")

    def handler(request):
        return httpx.Response(500, text="boom")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(JudgeError, match="500"):
            await judge_context("q", [_chunk("c1", "x")], client, settings)


async def test_judge_context_raises_on_unrecognized_verdict():
    settings = Settings(groq_api_key="test-key")

    def handler(request):
        return _groq_response("maybe? unclear")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(JudgeError, match="recognizable verdict"):
            await judge_context("q", [_chunk("c1", "x")], client, settings)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/retrieval/test_judge.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.retrieval.judge'`

- [ ] **Step 4: Implement the judge**

```python
# app/retrieval/judge.py
from __future__ import annotations

from typing import Literal

import httpx
from pydantic import BaseModel

from app.config import Settings, settings as default_settings
from app.retrieval.models import FusedChunk

JUDGE_SYSTEM_PROMPT = (
    "You are a strict grader. Given a question and numbered context excerpts, decide "
    "whether the excerpts contain enough information to answer the question accurately "
    "and completely. Respond with exactly one word: \"context_good\" if the excerpts are "
    "sufficient, or \"context_insufficient\" if they are not."
)


class JudgeVerdict(BaseModel):
    verdict: Literal["context_good", "context_insufficient"]


class JudgeError(Exception):
    pass


def _build_context_block(chunks: list[FusedChunk]) -> str:
    return "\n\n".join(f"[{index + 1}] {chunk.text}" for index, chunk in enumerate(chunks))


async def judge_context(
    query: str,
    chunks: list[FusedChunk],
    http_client: httpx.AsyncClient,
    settings: Settings = default_settings,
) -> JudgeVerdict:
    if not chunks:
        return JudgeVerdict(verdict="context_insufficient")

    if not settings.groq_api_key:
        raise JudgeError("GROQ_API_KEY is not configured")

    context_block = _build_context_block(chunks)

    try:
        response = await http_client.post(
            f"{settings.groq_base_url}/chat/completions",
            headers={"Authorization": f"Bearer {settings.groq_api_key}"},
            json={
                "model": settings.judge_model,
                "messages": [
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Question: {query}\n\nContext:\n{context_block}"},
                ],
            },
            timeout=settings.fetch_timeout_seconds,
        )
    except httpx.RequestError as exc:
        raise JudgeError(f"judge request failed: {exc}") from exc

    if response.status_code != 200:
        raise JudgeError(f"judge request returned status {response.status_code}")

    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise JudgeError("judge response missing verdict content") from exc

    normalized = content.strip().lower()
    if "insufficient" in normalized:
        return JudgeVerdict(verdict="context_insufficient")
    if "good" in normalized:
        return JudgeVerdict(verdict="context_good")
    raise JudgeError(f"judge response did not contain a recognizable verdict: {content!r}")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/retrieval/test_judge.py -v`
Expected: PASS (7 tests)

- [ ] **Step 6: Commit**

```bash
git add app/config.py app/retrieval/judge.py tests/retrieval/test_judge.py
git commit -m "feat: add context quality judge (Groq-backed, fail-closed on error)"
```

---

### Task 5: Router orchestration — preferences, filtering, judge, retry, fallback

**Files:**
- Modify: `app/config.py` (add `judge_retry_top_k_multiplier`)
- Modify: `app/retrieval/models.py` (`QueryResponse` + `preferences`, `filtered_out_count`)
- Modify: `app/retrieval/router.py`
- Test: `tests/retrieval/test_router.py`

**Interfaces:**
- Consumes: `extract_preferences` (Task 1), `filter_chunks` (Task 3), `judge_context`/`JudgeVerdict`/`JudgeError` (Task 4).
- Produces: `QueryResponse.preferences: QueryPreferences`, `QueryResponse.filtered_out_count: int`. Task 6 (frontend) consumes these two fields by exact name from the JSON response.

- [ ] **Step 1: Add the retry multiplier config field**

```python
# app/config.py — add one field to the Settings dataclass, after judge_model
    judge_retry_top_k_multiplier: int = 2
```

- [ ] **Step 2: Extend QueryResponse**

```python
# app/retrieval/models.py — add import at top, extend QueryResponse (full class shown)
from app.extraction.preferences import QueryPreferences


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[FusedChunk]
    preferences: QueryPreferences
    filtered_out_count: int
```

- [ ] **Step 3: Write the failing tests**

```python
# tests/retrieval/test_router.py — ADD these tests (keep existing ones; existing tests
# will need `preferences` and `filtered_out_count` added to their assertions, or left
# unchecked since they only assert on specific keys — no existing assertion breaks
# structurally, but Step 4 below requires the groq_handler in EVERY existing test to also
# handle judge-model requests, since /query now calls the judge before synthesis).
# This step also requires `import json` at the top of the file — see Step 4.


def _judge_good_and_synthesis_handler(answer_content: str, judge_model: str):
    def handler(request):
        body = json.loads(request.read())
        if body["model"] == judge_model:
            return httpx.Response(200, json={"choices": [{"message": {"content": "context_good"}}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": answer_content}}]})

    return handler


def test_post_query_includes_preferences_and_filtered_out_count(tmp_path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="t2",
        vector_size=2,
        groq_api_key="test-key",
    )
    bm25 = InMemoryBM25Index()
    vectors = QdrantVectorIndex(settings)
    store = ChunkStore()

    kept_id = "22222222-2222-2222-2222-222222222222"
    excluded_id = "33333333-3333-3333-3333-333333333333"
    kept_chunk = ChunkMetadata(
        chunk_id=kept_id, doc_id="d1", source_url="https://example.com", page_number=1,
        chunk_index=0, char_start=0, char_end=10, overlap_with_prev=0,
        indexed_at="2026-07-29T12:00:00+00:00", text="a hotel in lahore", city="lahore", price=None,
    )
    excluded_chunk = ChunkMetadata(
        chunk_id=excluded_id, doc_id="d1", source_url="https://example.com", page_number=1,
        chunk_index=1, char_start=10, char_end=20, overlap_with_prev=0,
        indexed_at="2026-07-29T12:00:00+00:00", text="a hotel in paris", city="paris", price=None,
    )
    bm25.add_documents([kept_id, excluded_id], ["a hotel in lahore", "a hotel in paris"])
    vectors.upsert(
        [kept_id, excluded_id], [[1.0, 0.0], [0.9, 0.1]],
        [kept_chunk.model_dump(), excluded_chunk.model_dump()],
    )
    store.add([kept_chunk, excluded_chunk])

    def embed_handler(request):
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})

    embedding_client = httpx.AsyncClient(transport=httpx.MockTransport(embed_handler))
    synthesis_client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            _judge_good_and_synthesis_handler("A hotel in Lahore [1].", settings.judge_model)
        )
    )

    app = FastAPI()
    app.include_router(build_retrieval_router(bm25, vectors, store, embedding_client, synthesis_client, settings))

    with TestClient(app) as client:
        response = client.post("/query", json={"query": "hotel in Lahore"})

    assert response.status_code == 200
    body = response.json()
    assert body["preferences"]["city"] == "lahore"
    assert body["filtered_out_count"] == 1
    assert [c["chunk_id"] for c in body["retrieved_chunks"]] == [kept_id]


def test_post_query_retries_once_then_succeeds_when_judge_recovers(tmp_path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="t3",
        vector_size=2,
        groq_api_key="test-key",
        retrieval_top_k=1,
    )
    bm25 = InMemoryBM25Index()
    vectors = QdrantVectorIndex(settings)
    store = ChunkStore()

    chunk_id = "44444444-4444-4444-4444-444444444444"
    chunk = ChunkMetadata(
        chunk_id=chunk_id, doc_id="d1", source_url="https://example.com", page_number=1,
        chunk_index=0, char_start=0, char_end=10, overlap_with_prev=0,
        indexed_at="2026-07-29T12:00:00+00:00", text="the quick brown fox",
    )
    bm25.add_documents([chunk_id], ["the quick brown fox"])
    vectors.upsert([chunk_id], [[1.0, 0.0]], [chunk.model_dump()])
    store.add([chunk])

    def embed_handler(request):
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})

    judge_call_count = {"n": 0}

    def groq_handler(request):
        body = json.loads(request.read())
        if body["model"] == settings.judge_model:
            judge_call_count["n"] += 1
            verdict = "context_insufficient" if judge_call_count["n"] == 1 else "context_good"
            return httpx.Response(200, json={"choices": [{"message": {"content": verdict}}]})
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
    assert judge_call_count["n"] == 2


def test_post_query_returns_fallback_after_two_insufficient_judgments(tmp_path):
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="t4",
        vector_size=2,
        groq_api_key="test-key",
    )
    bm25 = InMemoryBM25Index()
    vectors = QdrantVectorIndex(settings)
    store = ChunkStore()

    chunk_id = "55555555-5555-5555-5555-555555555555"
    chunk = ChunkMetadata(
        chunk_id=chunk_id, doc_id="d1", source_url="https://example.com", page_number=1,
        chunk_index=0, char_start=0, char_end=10, overlap_with_prev=0,
        indexed_at="2026-07-29T12:00:00+00:00", text="the quick brown fox",
    )
    bm25.add_documents([chunk_id], ["the quick brown fox"])
    vectors.upsert([chunk_id], [[1.0, 0.0]], [chunk.model_dump()])
    store.add([chunk])

    def embed_handler(request):
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})

    synthesis_call_count = {"n": 0}

    def groq_handler(request):
        body = json.loads(request.read())
        if body["model"] == settings.judge_model:
            return httpx.Response(200, json={"choices": [{"message": {"content": "context_insufficient"}}]})
        synthesis_call_count["n"] += 1
        return httpx.Response(200, json={"choices": [{"message": {"content": "should not be called"}}]})

    embedding_client = httpx.AsyncClient(transport=httpx.MockTransport(embed_handler))
    synthesis_client = httpx.AsyncClient(transport=httpx.MockTransport(groq_handler))

    app = FastAPI()
    app.include_router(build_retrieval_router(bm25, vectors, store, embedding_client, synthesis_client, settings))

    with TestClient(app) as client:
        response = client.post("/query", json={"query": "tell me about the fox"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == (
        "I don't have enough reliable information in the indexed content to answer this question confidently."
    )
    assert body["citations"] == []
    assert all(not c["used_in_synthesis"] for c in body["retrieved_chunks"])
    assert synthesis_call_count["n"] == 0
```

- [ ] **Step 4: Update the two pre-existing tests' Groq handlers to also answer judge calls**

First, add `import json` to the top of `tests/retrieval/test_router.py`, alongside the existing `import uuid` / `import httpx` block.

The two existing tests in `tests/retrieval/test_router.py` — `test_post_query_returns_answer_with_citations_and_chunks` and `test_post_query_default_settings_marks_overflow_chunks_not_used_in_synthesis` — each define a `groq_handler` that only returns a synthesis-shaped response. Since `/query` now calls the judge first (same `synthesis_client`), update both handlers to branch on `body["model"]`, same pattern as `_judge_good_and_synthesis_handler` above:

```python
    def groq_handler(request):
        body = json.loads(request.read())
        if body["model"] == settings.judge_model:
            return httpx.Response(200, json={"choices": [{"message": {"content": "context_good"}}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": "The fox is quick [1]."}}]})
```

(For the second test, keep its existing synthesis content — only add the `model` branch and the `import json` at the top of the file.)

- [ ] **Step 5: Run tests to verify they fail**

Run: `uv run pytest tests/retrieval/test_router.py -v`
Expected: FAIL — `QueryResponse` missing `preferences`, and the pre-existing tests' Groq handlers don't yet branch on `model`, so the judge call would get a synthesis-shaped response that doesn't parse as a verdict, raising `JudgeError` before the router changes exist at all.

- [ ] **Step 6: Implement router orchestration**

```python
# app/retrieval/router.py — full file
from __future__ import annotations

import dataclasses

import httpx
from fastapi import APIRouter

from app.config import Settings, settings as default_settings
from app.extraction.preferences import extract_preferences
from app.indexing.bm25_index import InMemoryBM25Index
from app.indexing.chunk_store import ChunkStore
from app.indexing.vector_index import QdrantVectorIndex
from app.retrieval.filtering import filter_chunks
from app.retrieval.judge import JudgeError, JudgeVerdict, judge_context
from app.retrieval.models import FusedChunk, QueryRequest, QueryResponse
from app.retrieval.retriever import retrieve
from app.retrieval.synthesis import synthesize_answer

FALLBACK_ANSWER = (
    "I don't have enough reliable information in the indexed content to answer this question confidently."
)


async def _judge_safely(
    query: str, chunks: list[FusedChunk], http_client: httpx.AsyncClient, settings: Settings
) -> JudgeVerdict:
    try:
        return await judge_context(query, chunks, http_client, settings)
    except JudgeError:
        return JudgeVerdict(verdict="context_insufficient")


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
        preferences = extract_preferences(request.query)

        fused_chunks = await retrieve(
            request.query, bm25_index, vector_index, chunk_store, embedding_client, settings, request.top_k
        )
        kept_chunks, filtered_out_count = filter_chunks(fused_chunks, preferences)
        verdict = await _judge_safely(request.query, kept_chunks, synthesis_client, settings)

        if verdict.verdict == "context_insufficient":
            retry_settings = dataclasses.replace(
                settings,
                retrieval_top_k=settings.retrieval_top_k * settings.judge_retry_top_k_multiplier,
                display_top_k=settings.display_top_k * settings.judge_retry_top_k_multiplier,
            )
            fused_chunks = await retrieve(
                request.query, bm25_index, vector_index, chunk_store, embedding_client, retry_settings, request.top_k
            )
            kept_chunks, filtered_out_count = filter_chunks(fused_chunks, preferences)
            verdict = await _judge_safely(request.query, kept_chunks, synthesis_client, retry_settings)

        if verdict.verdict == "context_insufficient":
            for chunk in kept_chunks:
                chunk.used_in_synthesis = False
            return QueryResponse(
                query=request.query,
                answer=FALLBACK_ANSWER,
                citations=[],
                retrieved_chunks=kept_chunks,
                preferences=preferences,
                filtered_out_count=filtered_out_count,
            )

        answer, citations, used_chunk_ids = await synthesize_answer(
            request.query, kept_chunks, synthesis_client, settings
        )
        for chunk in kept_chunks:
            chunk.used_in_synthesis = chunk.chunk_id in used_chunk_ids

        return QueryResponse(
            query=request.query,
            answer=answer,
            citations=citations,
            retrieved_chunks=kept_chunks,
            preferences=preferences,
            filtered_out_count=filtered_out_count,
        )

    return router
```

- [ ] **Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/retrieval/test_router.py -v`
Expected: PASS (7 tests: 4 pre-existing + 3 new)

- [ ] **Step 8: Run the full test suite to check for regressions**

Run: `uv run pytest -q`
Expected: all pass

- [ ] **Step 9: Commit**

```bash
git add app/config.py app/retrieval/models.py app/retrieval/router.py tests/retrieval/test_router.py
git commit -m "feat: orchestrate preference filtering and context quality judge with retry and fallback"
```

---

### Task 6: Query screen — preference badges and filtered-out count

**Files:**
- Modify: `app/static/query.html`
- Modify: `app/static/js/query.js`
- Modify: `app/static/css/app.css`
- Test: `tests/frontend/test_query_page.py`

**Interfaces:**
- Consumes: `QueryResponse.preferences`, `QueryResponse.filtered_out_count` (Task 5), exact JSON field names.
- Produces: `#preferences` and `#filtered-note` DOM containers, populated by `renderPreferences(preferences)` and `renderFilteredNote(count)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/frontend/test_query_page.py — add these assertions to test_query_page_served
def test_query_page_served(tmp_path):
    app = create_app(_isolated_settings(tmp_path))
    with TestClient(app) as client:
        response = client.get("/query-ui")
    assert response.status_code == 200
    assert "Ask a question" in response.text
    assert '/static/js/query.js' in response.text
    assert 'id="preferences"' in response.text
    assert 'id="filtered-note"' in response.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/frontend/test_query_page.py -v`
Expected: FAIL — `'id="preferences"' in response.text` is `False`

- [ ] **Step 3: Add the new containers to query.html**

```html
<!-- app/static/query.html — full file -->
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>RAG Query</title>
  <link rel="stylesheet" href="/static/css/app.css" />
</head>
<body>
  <nav><a href="/">Ingest</a><a href="/query-ui">Query</a></nav>
  <h1>Ask a question</h1>
  <div class="row">
    <input id="query-input" type="text" placeholder="What do you want to know?" />
    <button id="ask">Ask</button>
  </div>
  <div id="preferences"></div>
  <div id="answer"></div>

  <div class="filter-toggle" id="filter-toggle">
    <button data-filter="all" aria-pressed="true">All</button>
    <button data-filter="bm25" aria-pressed="false">BM25-only</button>
    <button data-filter="semantic" aria-pressed="false">Semantic-only</button>
    <button data-filter="both" aria-pressed="false">Both</button>
  </div>
  <div id="filtered-note"></div>
  <div id="chunks"></div>

  <script src="/static/js/api.js"></script>
  <script src="/static/js/query.js"></script>
</body>
</html>
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/frontend/test_query_page.py -v`
Expected: PASS

- [ ] **Step 5: Add rendering logic to query.js**

```javascript
// app/static/js/query.js — add these two functions (anywhere before the "ask" handler)
function renderPreferences(preferences) {
  const container = document.getElementById("preferences");
  container.innerHTML = "";
  const entries = [];
  if (preferences.city) entries.push(`City: ${preferences.city}`);
  if (preferences.budget != null) entries.push(`Budget: <= ${preferences.budget}`);
  if (preferences.interests.length > 0) entries.push(`Interests: ${preferences.interests.join(", ")}`);
  for (const entry of entries) {
    const badge = document.createElement("span");
    badge.className = "badge";
    badge.textContent = entry;
    container.appendChild(badge);
  }
}

function renderFilteredNote(count) {
  const container = document.getElementById("filtered-note");
  container.textContent = count > 0 ? `${count} chunk(s) excluded by filter` : "";
}
```

```javascript
// app/static/js/query.js — modify the "ask" handler to call the two new functions
document.getElementById("ask").onclick = async () => {
  const query = document.getElementById("query-input").value.trim();
  if (!query) return;
  const result = await postQuery(query);
  lastChunks = result.retrieved_chunks;
  citationsByMarker = Object.fromEntries(result.citations.map((c) => [String(c.marker), c.chunk_id]));
  renderPreferences(result.preferences);
  renderFilteredNote(result.filtered_out_count);
  renderAnswer(result.answer);
  renderChunks();
};
```

- [ ] **Step 6: Add CSS spacing for the new rows**

```css
/* app/static/css/app.css — append at the end of the file */
#preferences {
  margin-bottom: 0.5rem;
}

#filtered-note {
  font-size: 0.8rem;
  color: var(--muted);
  margin-bottom: 0.5rem;
}
```

- [ ] **Step 7: Run the full test suite**

Run: `uv run pytest -q`
Expected: all pass

- [ ] **Step 8: Commit**

```bash
git add app/static/query.html app/static/js/query.js app/static/css/app.css tests/frontend/test_query_page.py
git commit -m "feat: surface detected preferences and filtered-out count on the Query screen"
```

---

### Task 7: Full regression and manual end-to-end verification

**Files:** none (verification only; fix inline and commit separately if a real bug surfaces)

This task has no new automated test — see the frontend plan's precedent (`docs/superpowers/plans/2026-07-29-rag-frontend-implementation.md`, Task 4) for why JS-level browser behavior isn't covered by the Python test suite. This is the functional sign-off for the whole plan.

- [ ] **Step 1: Run the full automated suite**

Run: `uv run pytest -q`
Expected: all tests pass (baseline 87 + this plan's new tests)

- [ ] **Step 2: Start dependencies and the app**

Ensure Ollama is running with `qwen3-embedding:0.6b` pulled, export `GROQ_API_KEY`, then:

```bash
uv run uvicorn app.main:app --reload
```

- [ ] **Step 3: Ingest content with detectable city/price signal**

Open `http://localhost:8000/`, ingest a URL whose content plausibly mentions a city and a price (or ingest two different pages about two different cities) so filtering has something real to act on.

- [ ] **Step 4: Exercise preference extraction and filtering on the Query screen**

Open `http://localhost:8000/query-ui`. Ask a question that names a city and a budget (e.g. "What's a good budget hotel under $200 in Lahore?"). Confirm: preference badges appear above the answer (City/Budget/Interests, only for detected fields), and if any chunk had a conflicting city/price, the "N chunks excluded by filter" note appears near the chunk panel.

- [ ] **Step 5: Exercise the judge fallback path**

Ask a question clearly unrelated to anything ingested (e.g., if you ingested travel content, ask a specific unrelated technical question). Confirm the answer renders the fixed fallback string, citations are empty, and no chunk card shows "Used in answer".

- [ ] **Step 6: Record the result**

If everything in Steps 3–5 holds, note "manual verification passed" in the SDD ledger for this plan. If something is broken, fix it directly, re-run the relevant steps, then commit the fix with a normal descriptive message (not a plan-step commit).

---

## Self-Review Notes

- **Spec coverage:** Preference extraction (spec §1) — Task 1. City/price chunk tagging + filtering (spec §2) — Tasks 2–3. Context quality judge + retry + fallback (spec §3) — Tasks 4–5. Frontend surfacing — Task 6. Functional sign-off — Task 7.
- **Type consistency checked:** `QueryPreferences` (Task 1) is the exact type threaded through `filter_chunks` (Task 3), `QueryResponse.preferences` (Task 5), and the JSON field the frontend (Task 6) reads. `FusedChunk.city`/`.price` (Task 2) are the exact field names `filter_chunks` (Task 3) reads. `JudgeVerdict`/`JudgeError` (Task 4) are the exact names `router.py` (Task 5) imports and catches.
- **Deliberate scope boundary:** no JS-level automated tests (matches the frontend plan's documented exception); no city alias resolution; no backfill migration; exactly one retry with a hardcoded multiplier — all called out in Global Constraints rather than repeated per task.
- **No placeholders:** every step shows the real code to write; no "TBD" left.
