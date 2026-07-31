# Judge Response Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface the context-quality judge's raw response(s) on the Query screen in a fixed side panel, so a user can see exactly what the judge said and why — a prerequisite step before redefining the judge's grading behavior.

**Architecture:** `JudgeVerdict` gains a `raw_response: str` field so the judge's actual text survives past parsing. The router accumulates a `list[JudgeAttempt]` as it makes its 1-2 judge calls per request and threads it through `QueryResponse` unchanged on every response path. The Query screen becomes a two-column layout: existing content stays in a main column, a new `<aside>` renders each judge attempt.

**Tech Stack:** No new dependencies. Same Pydantic models, same vanilla JS/CSS, same FastAPI router pattern already in use.

## Global Constraints

- No change to the judge's actual grading logic/prompt in this plan — purely making existing behavior visible, a prerequisite before redefining it.
- No persistence of judge history across queries — panel always reflects only the most recent query, fully replaced (not appended) on each new query.
- No collapse/toggle affordance — panel is always visible, simplest first pass.
- `raw_response` is LLM-generated text and must never be interpolated via `innerHTML` — always `textContent`, matching the existing file's discipline for all non-numeric, non-hardcoded content.
- Judge failure (`JudgeError`) must still be visible in the panel, not silently swallowed — `_judge_safely`'s except-branch fabricates a `raw_response` describing the failure rather than an empty string.
- Test convention: mirror `app/` structure under `tests/`, no `conftest.py`, `httpx.MockTransport` for external calls, `asyncio_mode=auto`, `tmp_path`-isolated Qdrant per test.

---

## File Structure

```
app/
  retrieval/
    judge.py                        # MODIFY: JudgeVerdict + raw_response
    models.py                       # MODIFY: + JudgeAttempt; QueryResponse + judge_attempts
    router.py                       # MODIFY: accumulate judge_attempts, fix _judge_safely
  static/
    query.html                      # MODIFY: two-column layout, + <aside id="judge-panel">
    js/query.js                     # MODIFY: + renderJudgePanel
    css/app.css                     # MODIFY: widen body, + .page-layout/.main-column/#judge-panel rules
tests/
  retrieval/
    test_judge.py                   # MODIFY: assert raw_response on all paths
    test_models.py                  # MODIFY: JudgeAttempt round-trip; existing QueryResponse test needs judge_attempts
    test_router.py                  # MODIFY: assert judge_attempts shape on every existing test + new ones
  frontend/
    test_query_page.py              # MODIFY: assert judge-panel container present
```

---

### Task 1: `JudgeVerdict` gains `raw_response`

**Files:**
- Modify: `app/retrieval/judge.py`
- Modify: `app/retrieval/router.py` (one line in `_judge_safely` — required so the app keeps constructing valid `JudgeVerdict`s once `raw_response` is required)
- Test: `tests/retrieval/test_judge.py`

**Interfaces:**
- Produces: `JudgeVerdict(verdict: Literal["context_good","context_insufficient"], raw_response: str)`. Task 3 reads `verdict.raw_response` directly.

- [ ] **Step 1: Write the failing tests**

```python
# tests/retrieval/test_judge.py — MODIFY existing tests to assert raw_response, ADD one new test

async def test_judge_context_returns_good_verdict():
    settings = Settings(groq_api_key="test-key")
    chunks = [_chunk("c1", "Paris is the capital of France.")]

    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: _groq_response("context_good"))) as client:
        verdict = await judge_context("capital of France?", chunks, client, settings)

    assert verdict.verdict == "context_good"
    assert verdict.raw_response == "context_good"


async def test_judge_context_returns_insufficient_verdict():
    settings = Settings(groq_api_key="test-key")
    chunks = [_chunk("c1", "unrelated text")]

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: _groq_response("context_insufficient"))
    ) as client:
        verdict = await judge_context("capital of France?", chunks, client, settings)

    assert verdict.verdict == "context_insufficient"
    assert verdict.raw_response == "context_insufficient"


async def test_judge_context_empty_chunks_short_circuits_without_request():
    settings = Settings(groq_api_key="test-key")

    def handler(request):
        raise AssertionError("should not make a request for empty chunks")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        verdict = await judge_context("q", [], client, settings)

    assert verdict.verdict == "context_insufficient"
    assert verdict.raw_response == "(no chunks retrieved)"


async def test_judge_context_raw_response_preserves_verbose_content():
    # raw_response must carry the FULL text, not just the matched substring —
    # this is what the side panel will render verbatim.
    settings = Settings(groq_api_key="test-key")
    verbose = "This context is definitely context_good because it directly answers the question."

    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: _groq_response(verbose))) as client:
        verdict = await judge_context("q", [_chunk("c1", "x")], client, settings)

    assert verdict.verdict == "context_good"
    assert verdict.raw_response == verbose
```

Leave the other existing tests in `test_judge.py` (`test_judge_context_uses_judge_model_not_synthesis_model`, `test_judge_context_raises_without_api_key`, `test_judge_context_raises_on_non_200`, `test_judge_context_raises_on_unrecognized_verdict`) unchanged — they test failure/routing paths that don't construct a `JudgeVerdict`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/retrieval/test_judge.py -v`
Expected: FAIL — `AttributeError: 'JudgeVerdict' object has no attribute 'raw_response'` (or a `pydantic.ValidationError` once the field is added but not yet populated on some path)

- [ ] **Step 3: Add the field and populate it on every path**

```python
# app/retrieval/judge.py — MODIFY JudgeVerdict and the two return sites

class JudgeVerdict(BaseModel):
    verdict: Literal["context_good", "context_insufficient"]
    raw_response: str
```

```python
# app/retrieval/judge.py — inside judge_context(), modify the empty-chunks short-circuit
    if not chunks:
        return JudgeVerdict(verdict="context_insufficient", raw_response="(no chunks retrieved)")
```

```python
# app/retrieval/judge.py — modify the verdict-parsing tail of judge_context()
    normalized = content.strip().lower()
    if "insufficient" in normalized:
        return JudgeVerdict(verdict="context_insufficient", raw_response=content)
    if "good" in normalized:
        return JudgeVerdict(verdict="context_good", raw_response=content)
    raise JudgeError(f"judge response did not contain a recognizable verdict: {content!r}")
```

```python
# app/retrieval/router.py — modify _judge_safely's except branch (one line)
async def _judge_safely(
    query: str, chunks: list[FusedChunk], http_client: httpx.AsyncClient, settings: Settings
) -> JudgeVerdict:
    try:
        return await judge_context(query, chunks, http_client, settings)
    except JudgeError as exc:
        return JudgeVerdict(verdict="context_insufficient", raw_response=f"(judge error: {exc})")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/retrieval/test_judge.py -v`
Expected: PASS (8 tests: 7 existing, 3 with new assertions + 1 new)

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `uv run pytest -q`
Expected: all pass (baseline 122 — `_judge_safely`'s one-line fix keeps `router.py` valid; no other code constructs `JudgeVerdict` directly)

- [ ] **Step 6: Commit**

```bash
git add app/retrieval/judge.py app/retrieval/router.py tests/retrieval/test_judge.py
git commit -m "feat: carry judge's raw response text through JudgeVerdict"
```

---

### Task 2: `QueryResponse` gains `judge_attempts`

**Files:**
- Modify: `app/retrieval/models.py`
- Test: `tests/retrieval/test_models.py`

**Interfaces:**
- Consumes: nothing new (pure schema addition).
- Produces: `JudgeAttempt(BaseModel)` with `attempt: int, verdict: str, raw_response: str`; `QueryResponse.judge_attempts: list[JudgeAttempt]`. Task 3 (router) constructs and populates this list on every response.

- [ ] **Step 1: Write the failing tests**

```python
# tests/retrieval/test_models.py — ADD this import and these tests, MODIFY the existing round-trip test

from app.retrieval.models import Citation, FusedChunk, JudgeAttempt, QueryRequest, QueryResponse


def test_judge_attempt_round_trip():
    attempt = JudgeAttempt(attempt=1, verdict="context_good", raw_response="context_good")
    assert attempt.model_dump() == {"attempt": 1, "verdict": "context_good", "raw_response": "context_good"}


def test_query_response_round_trip():
    response = QueryResponse(
        query="q",
        answer="answer [1]",
        citations=[Citation(marker=1, chunk_id="c1")],
        retrieved_chunks=[],
        preferences=QueryPreferences(),
        filtered_out_count=0,
        judge_attempts=[JudgeAttempt(attempt=1, verdict="context_good", raw_response="context_good")],
    )
    assert response.model_dump()["citations"][0]["chunk_id"] == "c1"
    assert response.model_dump()["judge_attempts"][0]["verdict"] == "context_good"
```

(The `QueryPreferences` import is already present at the top of this file from the earlier preference-extraction plan — keep it.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/retrieval/test_models.py -v`
Expected: FAIL — `ImportError: cannot import name 'JudgeAttempt'`

- [ ] **Step 3: Add the model and field**

```python
# app/retrieval/models.py — full file
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


class JudgeAttempt(BaseModel):
    attempt: int
    verdict: str
    raw_response: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[FusedChunk]
    preferences: QueryPreferences
    filtered_out_count: int
    judge_attempts: list[JudgeAttempt]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/retrieval/test_models.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `uv run pytest -q`
Expected: FAIL is acceptable here ONLY in `tests/retrieval/test_router.py` (it constructs `QueryResponse` indirectly via the live router, which doesn't pass `judge_attempts` yet — that's exactly Task 3). Confirm the failures are isolated to `test_router.py` and nothing else broke.

- [ ] **Step 6: Commit**

```bash
git add app/retrieval/models.py tests/retrieval/test_models.py
git commit -m "feat: add JudgeAttempt model and QueryResponse.judge_attempts field"
```

---

### Task 3: Router accumulates and returns judge attempts

**Files:**
- Modify: `app/retrieval/router.py`
- Test: `tests/retrieval/test_router.py`

**Interfaces:**
- Consumes: `JudgeAttempt` (Task 2), `JudgeVerdict.raw_response` (Task 1).
- Produces: `QueryResponse.judge_attempts` populated with 1 entry (no retry) or 2 entries (retry occurred), in order, on every response path (happy and fallback). Task 4 (frontend) reads this field by exact name.

- [ ] **Step 1: Write the failing tests**

```python
# tests/retrieval/test_router.py — MODIFY these four existing tests' assertions (add judge_attempts checks)
# and ADD one new test at the end. Keep every existing line of setup/handler code unchanged —
# only the assertions below `assert response.status_code == 200` change/grow.

# In test_post_query_returns_answer_with_citations_and_chunks, ADD after the existing asserts:
    assert body["judge_attempts"] == [
        {"attempt": 1, "verdict": "context_good", "raw_response": "context_good"}
    ]

# In test_post_query_default_settings_marks_overflow_chunks_not_used_in_synthesis, ADD:
    assert body["judge_attempts"] == [
        {"attempt": 1, "verdict": "context_good", "raw_response": "context_good"}
    ]

# In test_post_query_includes_preferences_and_filtered_out_count, ADD:
    assert body["judge_attempts"] == [
        {"attempt": 1, "verdict": "context_good", "raw_response": "context_good"}
    ]

# In test_post_query_retries_once_then_succeeds_when_judge_recovers, ADD:
    assert body["judge_attempts"] == [
        {"attempt": 1, "verdict": "context_insufficient", "raw_response": "context_insufficient"},
        {"attempt": 2, "verdict": "context_good", "raw_response": "context_good"},
    ]

# In test_post_query_returns_fallback_after_two_insufficient_judgments, ADD:
    assert body["judge_attempts"] == [
        {"attempt": 1, "verdict": "context_insufficient", "raw_response": "context_insufficient"},
        {"attempt": 2, "verdict": "context_insufficient", "raw_response": "context_insufficient"},
    ]

# In test_post_query_falls_back_when_judge_call_raises_judge_error, ADD:
    assert len(body["judge_attempts"]) == 2
    assert all(a["verdict"] == "context_insufficient" for a in body["judge_attempts"])
    assert all("(judge error:" in a["raw_response"] for a in body["judge_attempts"])
    assert [a["attempt"] for a in body["judge_attempts"]] == [1, 2]
```

```python
# tests/retrieval/test_router.py — ADD this new test at the end of the file
def test_post_query_judge_attempts_has_one_entry_when_no_retry_needed(tmp_path):
    # Explicit single-attempt-shape test, distinct from the retry tests above:
    # confirms judge_attempts has exactly ONE entry (not a stray second one) when
    # the first judge call already returns context_good.
    settings = Settings(
        qdrant_path=str(tmp_path / "qdrant"),
        qdrant_collection="t6",
        vector_size=2,
        groq_api_key="test-key",
    )
    bm25 = InMemoryBM25Index()
    vectors = QdrantVectorIndex(settings)
    store = ChunkStore()

    chunk_id = "77777777-7777-7777-7777-777777777777"
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

    embedding_client = httpx.AsyncClient(transport=httpx.MockTransport(embed_handler))
    synthesis_client = httpx.AsyncClient(
        transport=httpx.MockTransport(
            _judge_good_and_synthesis_handler("The fox is quick [1].", settings.judge_model)
        )
    )

    app = FastAPI()
    app.include_router(build_retrieval_router(bm25, vectors, store, embedding_client, synthesis_client, settings))

    with TestClient(app) as client:
        response = client.post("/query", json={"query": "tell me about the fox"})

    assert response.status_code == 200
    body = response.json()
    assert len(body["judge_attempts"]) == 1
    assert body["judge_attempts"][0]["attempt"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/retrieval/test_router.py -v`
Expected: FAIL — `KeyError: 'judge_attempts'` (field doesn't exist in the response yet)

- [ ] **Step 3: Implement router accumulation**

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
from app.retrieval.models import FusedChunk, JudgeAttempt, QueryRequest, QueryResponse
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
    except JudgeError as exc:
        return JudgeVerdict(verdict="context_insufficient", raw_response=f"(judge error: {exc})")


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
        judge_attempts = [JudgeAttempt(attempt=1, verdict=verdict.verdict, raw_response=verdict.raw_response)]

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
            judge_attempts.append(
                JudgeAttempt(attempt=2, verdict=verdict.verdict, raw_response=verdict.raw_response)
            )

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
                judge_attempts=judge_attempts,
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
            judge_attempts=judge_attempts,
        )

    return router
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/retrieval/test_router.py -v`
Expected: PASS (7 tests: 6 existing with new assertions + 1 new)

- [ ] **Step 5: Run the full test suite to check for regressions**

Run: `uv run pytest -q`
Expected: all pass (baseline 122 + this plan's Tasks 1-2 additions + this task's new test)

- [ ] **Step 6: Commit**

```bash
git add app/retrieval/router.py tests/retrieval/test_router.py
git commit -m "feat: accumulate and return judge attempts on every /query response"
```

---

### Task 4: Query screen — judge response side panel

**Files:**
- Modify: `app/static/query.html`
- Modify: `app/static/js/query.js`
- Modify: `app/static/css/app.css`
- Test: `tests/frontend/test_query_page.py`

**Interfaces:**
- Consumes: `QueryResponse.judge_attempts` (Task 3), exact JSON shape `[{attempt, verdict, raw_response}, ...]`.
- Produces: `#judge-panel` DOM container, populated by `renderJudgePanel(judgeAttempts)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/frontend/test_query_page.py — add this assertion to test_query_page_served
def test_query_page_served(tmp_path):
    app = create_app(_isolated_settings(tmp_path))
    with TestClient(app) as client:
        response = client.get("/query-ui")
    assert response.status_code == 200
    assert "Ask a question" in response.text
    assert '/static/js/query.js' in response.text
    assert 'id="preferences"' in response.text
    assert 'id="filtered-note"' in response.text
    assert 'id="judge-panel"' in response.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/frontend/test_query_page.py -v`
Expected: FAIL — `'id="judge-panel"' in response.text` is `False`

- [ ] **Step 3: Restructure query.html into a two-column layout**

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
  <div class="page-layout">
    <div class="main-column">
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
    </div>
    <aside id="judge-panel"></aside>
  </div>

  <script src="/static/js/api.js"></script>
  <script src="/static/js/query.js"></script>
</body>
</html>
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/frontend/test_query_page.py -v`
Expected: PASS

- [ ] **Step 5: Add the rendering function and wire it into the ask handler**

```javascript
// app/static/js/query.js — add this function (anywhere before the "ask" handler)
function renderJudgePanel(judgeAttempts) {
  const container = document.getElementById("judge-panel");
  container.innerHTML = "";
  for (const attempt of judgeAttempts) {
    const block = document.createElement("div");
    block.className = "judge-attempt";

    const heading = document.createElement("h3");
    heading.textContent = `Attempt ${attempt.attempt}: ${attempt.verdict}`;

    const raw = document.createElement("p");
    raw.textContent = attempt.raw_response;

    block.appendChild(heading);
    block.appendChild(raw);
    container.appendChild(block);
  }
}
```

```javascript
// app/static/js/query.js — modify the "ask" handler to call the new function
document.getElementById("ask").onclick = async () => {
  const query = document.getElementById("query-input").value.trim();
  if (!query) return;
  const result = await postQuery(query);
  lastChunks = result.retrieved_chunks;
  citationsByMarker = Object.fromEntries(result.citations.map((c) => [String(c.marker), c.chunk_id]));
  renderPreferences(result.preferences);
  renderFilteredNote(result.filtered_out_count);
  renderJudgePanel(result.judge_attempts);
  renderAnswer(result.answer);
  renderChunks();
};
```

- [ ] **Step 6: Add layout CSS**

```css
/* app/static/css/app.css — MODIFY the existing body rule's max-width */
body {
  font-family: system-ui, sans-serif;
  max-width: 1100px;
  margin: 2rem auto;
  padding: 0 1rem;
}
```

```css
/* app/static/css/app.css — append at the end of the file */
.page-layout {
  display: flex;
  gap: 1.5rem;
  align-items: flex-start;
}

.main-column {
  flex: 1;
  min-width: 0;
}

#judge-panel {
  flex: 0 0 280px;
  border: 1px solid var(--border);
  border-radius: 0.5rem;
  padding: 0.75rem;
  max-height: 80vh;
  overflow-y: auto;
}

.judge-attempt {
  margin-bottom: 0.75rem;
}

.judge-attempt h3 {
  font-size: 0.85rem;
  margin: 0 0 0.25rem 0;
}

.judge-attempt p {
  font-size: 0.8rem;
  color: var(--muted);
  white-space: pre-wrap;
  margin: 0;
}
```

- [ ] **Step 7: Run the full test suite**

Run: `uv run pytest -q`
Expected: all pass

- [ ] **Step 8: Commit**

```bash
git add app/static/query.html app/static/js/query.js app/static/css/app.css tests/frontend/test_query_page.py
git commit -m "feat: add judge response side panel to the Query screen"
```

---

### Task 5: Manual end-to-end verification

**Files:** none (verification only; fix inline and commit separately if a real bug surfaces)

This task has no new automated test — same documented exception as prior frontend work in this repo (no JS test harness). This is the functional sign-off.

- [ ] **Step 1: Run the full automated suite**

Run: `uv run pytest -q`
Expected: all tests pass (baseline 122 + this plan's new tests)

- [ ] **Step 2: Start dependencies and the app**

Ensure Ollama is running with `qwen3-embedding:0.6b` pulled, export `GROQ_API_KEY`, then:

```bash
uv run uvicorn app.main:app --reload
```

- [ ] **Step 3: Exercise the judge panel — single attempt**

Open `http://localhost:8000/query-ui`. Ask a question clearly answerable from already-ingested content. Confirm: the side panel appears to the right of the main column, shows exactly one "Attempt 1: context_good" block, and the raw response text below it matches whatever the judge model actually returned (not just "context_good" if it replied verbosely).

- [ ] **Step 4: Exercise the judge panel — retry path**

Ask a question likely to trigger a retry (something borderline, or reuse a query known from earlier testing to have triggered the fallback path). Confirm: the panel shows two blocks, "Attempt 1: context_insufficient" and "Attempt 2: ..." (either verdict), each with distinct raw response text.

- [ ] **Step 5: Record the result**

If everything in Steps 3–4 holds, note "manual verification passed" in the SDD ledger for this plan. If something is broken, fix it directly, re-run the relevant steps, then commit the fix with a normal descriptive message (not a plan-step commit).

---

## Self-Review Notes

- **Spec coverage:** `raw_response` field — Task 1. `JudgeAttempt`/`QueryResponse.judge_attempts` schema — Task 2. Router accumulation across both attempts, every response path — Task 3. Two-column layout + panel rendering — Task 4. Functional sign-off — Task 5.
- **Type consistency checked:** `JudgeVerdict.raw_response` (Task 1) is the exact field `router.py` (Task 3) reads into `JudgeAttempt.raw_response` (Task 2), which is the exact JSON field `query.js` (Task 4) reads as `attempt.raw_response`.
- **Deliberate scope boundary:** no judge prompt/logic changes; no history persistence; no collapse/toggle; no JS-level automated tests (matches existing repo convention) — all called out once in Global Constraints.
- **No placeholders:** every step shows the real code to write; no "TBD" left.
