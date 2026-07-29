# RAG Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the two UI screens — Ingest (URL list + progress polling) and Query (answer with citations + chunk-card transparency panel) — per `doc/development/arch.md` section 5, against the already-implemented `POST /ingest`, `GET /ingest/{job_id}/status`, and `POST /query` endpoints.

**Architecture:** Plain HTML/CSS/vanilla JS, no build step, no framework — served directly by FastAPI via `StaticFiles` (chosen over React/htmx: the spec says the UI stays deliberately minimal, "the product IS the transparency, not the polish," and this repo has no JS toolchain to justify introducing). Two page routes (`GET /`, `GET /query-ui`) each serve a static HTML shell that loads a shared `api.js` (fetch helpers) plus a page-specific script.

**Tech Stack:** FastAPI `StaticFiles` (already a FastAPI dependency, no new package), vanilla JS (`fetch`, DOM APIs), plain CSS. No new Python dependency.

## Global Constraints

- No build step, no JS framework, no new dependency — this was an explicit stack decision, not an oversight.
- **Testing approach differs from Plans 1–3 by necessity:** the FastAPI routes that *serve* the static assets are tested with `TestClient` (Tasks 1–3 keep the TDD red/green cycle). The JS/HTML/CSS *behavior* has no test harness in this repo (no `package.json`, no Jest/Vitest) and introducing one for a two-page showcase UI would be disproportionate — that behavior is verified manually in a real browser in Task 4, with an explicit checklist. This is a deliberate, documented exception, not a gap.
- Screen 1 (Ingest): URL list input with add/remove rows, submit button, one progress row per URL showing `stage` + `pages_fetched`/`pages_total` (per the ingestion status contract), polled every 1.5s until every URL reaches `done` or `error`.
- Screen 2 (Query): single query input; answer block with inline `[n]` citation markers that, on click, scroll to and highlight the matching chunk card; retrieved-chunks panel with one card per chunk showing text, source + page, and three badges (BM25 rank/score, semantic rank/score, fused rank/score); a filter toggle over `matched_methods` (all / BM25-only / semantic-only / both); a visual marker distinguishing chunks actually used in synthesis (`used_in_synthesis: true`) from chunks retrieved but not used.
- API contracts consumed exactly as already implemented — no client-side reshaping beyond what's needed to render (e.g. `matched_methods` filtering is a client-side *display* filter over data the server already computed; it must not recompute scores or ranks).

---

## File Structure

```
app/
  main.py                     # MODIFY: mount StaticFiles, add GET / and GET /query-ui routes
  static/
    css/
      app.css                 # shared minimal styling
    js/
      api.js                   # shared fetch helpers: postIngest, getIngestStatus, postQuery
      ingest.js                 # Screen 1 logic
      query.js                   # Screen 2 logic
    ingest.html                 # Screen 1 shell
    query.html                   # Screen 2 shell
tests/
  frontend/
    test_static_assets.py
    test_ingest_page.py
    test_query_page.py
```

---

### Task 1: Static asset wiring + shared CSS/JS

**Files:**
- Modify: `app/main.py` (mount `StaticFiles`)
- Create: `app/static/css/app.css`
- Create: `app/static/js/api.js`
- Test: `tests/frontend/test_static_assets.py`

**Interfaces:**
- Produces: `/static/css/app.css`, `/static/js/api.js` served as static files. `api.js` exposes global functions `postIngest(urls) -> Promise<{job_id}>`, `getIngestStatus(jobId) -> Promise<JobStatusResponse>`, `postQuery(query, topK) -> Promise<QueryResponse>` — matching the contracts from Plans 1 and 3 exactly.

- [ ] **Step 1: Write the failing test**

```python
# tests/frontend/test_static_assets.py
from fastapi.testclient import TestClient

from app.main import create_app


def test_shared_css_is_served():
    with TestClient(create_app()) as client:
        response = client.get("/static/css/app.css")
    assert response.status_code == 200
    assert "chunk-card" in response.text


def test_shared_api_js_is_served():
    with TestClient(create_app()) as client:
        response = client.get("/static/js/api.js")
    assert response.status_code == 200
    assert "function postIngest" in response.text
    assert "function getIngestStatus" in response.text
    assert "function postQuery" in response.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/frontend/test_static_assets.py -v`
Expected: FAIL with 404 (no `/static` mount yet)

- [ ] **Step 3: Add the shared CSS, shared JS, and mount StaticFiles**

```css
/* app/static/css/app.css */
:root {
  color-scheme: light dark;
  --border: #888;
  --muted: #666;
  --accent: #2b6cb0;
}

body {
  font-family: system-ui, sans-serif;
  max-width: 800px;
  margin: 2rem auto;
  padding: 0 1rem;
}

nav a {
  margin-right: 1rem;
}

.row {
  display: flex;
  gap: 0.5rem;
  align-items: center;
  margin-bottom: 0.5rem;
}

.row input[type="text"],
.row input[type="url"] {
  flex: 1;
  padding: 0.4rem;
}

.stage-badge {
  padding: 0.15rem 0.5rem;
  border: 1px solid var(--border);
  border-radius: 0.75rem;
  font-size: 0.8rem;
  text-transform: uppercase;
}

.stage-badge.error {
  border-color: #c0392b;
  color: #c0392b;
}

.stage-badge.done {
  border-color: #27ae60;
  color: #27ae60;
}

.chunk-card {
  border: 1px solid var(--border);
  border-radius: 0.5rem;
  padding: 0.75rem;
  margin-bottom: 0.75rem;
}

.chunk-card.used {
  border-left: 4px solid var(--accent);
}

.chunk-card.not-used {
  opacity: 0.7;
}

.badge {
  display: inline-block;
  border: 1px solid var(--border);
  border-radius: 0.5rem;
  padding: 0.1rem 0.4rem;
  font-size: 0.75rem;
  margin-right: 0.4rem;
}

.filter-toggle button {
  margin-right: 0.4rem;
}

.filter-toggle button[aria-pressed="true"] {
  background: var(--accent);
  color: white;
}

.citation-marker {
  cursor: pointer;
  color: var(--accent);
  font-weight: bold;
}

.highlight {
  outline: 2px solid var(--accent);
}
```

```js
// app/static/js/api.js
async function postIngest(urls) {
  const response = await fetch("/ingest", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ urls }),
  });
  if (!response.ok) {
    throw new Error(`POST /ingest failed: ${response.status}`);
  }
  return response.json();
}

async function getIngestStatus(jobId) {
  const response = await fetch(`/ingest/${jobId}/status`);
  if (!response.ok) {
    throw new Error(`GET /ingest/${jobId}/status failed: ${response.status}`);
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

```python
# app/main.py — add near the top and inside create_app()
from fastapi.staticfiles import StaticFiles
# ... existing imports stay

def create_app() -> FastAPI:
    app = FastAPI(title="RAG Ingestion Service")
    # ... existing job_store / bm25_index / vector_index / chunk_store / clients / routers stay

    app.mount("/static", StaticFiles(directory="app/static"), name="static")

    return app
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/frontend/test_static_assets.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add app/main.py app/static/css/app.css app/static/js/api.js tests/frontend/test_static_assets.py
git commit -m "feat: mount static assets, add shared CSS and API fetch helpers"
```

---

### Task 2: Ingest screen

**Files:**
- Create: `app/static/ingest.html`
- Create: `app/static/js/ingest.js`
- Modify: `app/main.py` (`GET /` serves `ingest.html`)
- Test: `tests/frontend/test_ingest_page.py`

**Interfaces:**
- Consumes: `postIngest`, `getIngestStatus` from `api.js` (Task 1); the `IngestResponse`/`JobStatusResponse` JSON shapes from Plan 1 (`{job_id}` and `{job_id, urls: [{url, stage, pages_fetched, pages_total, error}]}`).
- Produces: `GET /` returns `ingest.html`.

- [ ] **Step 1: Write the failing test**

```python
# tests/frontend/test_ingest_page.py
from fastapi.testclient import TestClient

from app.main import create_app


def test_ingest_page_served_at_root():
    with TestClient(create_app()) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "Ingest URLs" in response.text
    assert '/static/js/ingest.js' in response.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/frontend/test_ingest_page.py -v`
Expected: FAIL with 404 (no `/` route yet)

- [ ] **Step 3: Implement the Ingest screen**

```html
<!-- app/static/ingest.html -->
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>RAG Ingest</title>
  <link rel="stylesheet" href="/static/css/app.css" />
</head>
<body>
  <nav><a href="/">Ingest</a><a href="/query-ui">Query</a></nav>
  <h1>Ingest URLs</h1>
  <div id="url-rows"></div>
  <button id="add-row">Add URL</button>
  <button id="submit">Start Ingest</button>
  <div id="progress"></div>

  <script src="/static/js/api.js"></script>
  <script src="/static/js/ingest.js"></script>
</body>
</html>
```

```js
// app/static/js/ingest.js
const urlRows = document.getElementById("url-rows");
const progress = document.getElementById("progress");
let pollTimer = null;

function addUrlRow(value = "") {
  const row = document.createElement("div");
  row.className = "row";

  const input = document.createElement("input");
  input.type = "url";
  input.placeholder = "https://example.com/article";
  input.value = value;

  const remove = document.createElement("button");
  remove.textContent = "Remove";
  remove.onclick = () => row.remove();

  row.appendChild(input);
  row.appendChild(remove);
  urlRows.appendChild(row);
}

function collectUrls() {
  return Array.from(urlRows.querySelectorAll("input"))
    .map((input) => input.value.trim())
    .filter((value) => value.length > 0);
}

function renderStatus(status) {
  progress.innerHTML = "";
  for (const urlStatus of status.urls) {
    const row = document.createElement("div");
    row.className = "row";

    const label = document.createElement("span");
    label.textContent = urlStatus.url;

    const badge = document.createElement("span");
    badge.className = `stage-badge ${urlStatus.stage}`;
    const pages = urlStatus.pages_total
      ? `${urlStatus.pages_fetched}/${urlStatus.pages_total}`
      : `${urlStatus.pages_fetched}`;
    badge.textContent = urlStatus.error ? `${urlStatus.stage}: ${urlStatus.error}` : `${urlStatus.stage} (${pages})`;

    row.appendChild(label);
    row.appendChild(badge);
    progress.appendChild(row);
  }
}

async function pollStatus(jobId) {
  const status = await getIngestStatus(jobId);
  renderStatus(status);
  const allTerminal = status.urls.every((u) => u.stage === "done" || u.stage === "error");
  if (allTerminal && pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
}

document.getElementById("add-row").onclick = () => addUrlRow();
document.getElementById("submit").onclick = async () => {
  const urls = collectUrls();
  if (urls.length === 0) return;
  const { job_id } = await postIngest(urls);
  if (pollTimer) clearInterval(pollTimer);
  await pollStatus(job_id);
  pollTimer = setInterval(() => pollStatus(job_id), 1500);
};

addUrlRow();
```

```python
# app/main.py — add import and route
from fastapi.responses import FileResponse
# ... existing imports stay

# inside create_app(), after the StaticFiles mount:
    @app.get("/")
    async def serve_ingest_page() -> FileResponse:
        return FileResponse("app/static/ingest.html")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/frontend/test_ingest_page.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add app/static/ingest.html app/static/js/ingest.js app/main.py tests/frontend/test_ingest_page.py
git commit -m "feat: add Ingest screen (URL list, submit, progress polling)"
```

---

### Task 3: Query screen

**Files:**
- Create: `app/static/query.html`
- Create: `app/static/js/query.js`
- Modify: `app/main.py` (`GET /query-ui` serves `query.html`)
- Test: `tests/frontend/test_query_page.py`

**Interfaces:**
- Consumes: `postQuery` from `api.js` (Task 1); the `QueryResponse` JSON shape from Plan 3 (`{query, answer, citations: [{marker, chunk_id}], retrieved_chunks: [FusedChunk...]}`, where each `FusedChunk` has `chunk_id, text, source_url, page_number, bm25_rank, bm25_score, semantic_rank, semantic_score, fused_rank, rrf_score, matched_methods, used_in_synthesis`).
- Produces: `GET /query-ui` returns `query.html`.

- [ ] **Step 1: Write the failing test**

```python
# tests/frontend/test_query_page.py
from fastapi.testclient import TestClient

from app.main import create_app


def test_query_page_served():
    with TestClient(create_app()) as client:
        response = client.get("/query-ui")
    assert response.status_code == 200
    assert "Ask a question" in response.text
    assert '/static/js/query.js' in response.text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/frontend/test_query_page.py -v`
Expected: FAIL with 404 (no `/query-ui` route yet)

- [ ] **Step 3: Implement the Query screen**

```html
<!-- app/static/query.html -->
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
  <div id="answer"></div>

  <div class="filter-toggle" id="filter-toggle">
    <button data-filter="all" aria-pressed="true">All</button>
    <button data-filter="bm25" aria-pressed="false">BM25-only</button>
    <button data-filter="semantic" aria-pressed="false">Semantic-only</button>
    <button data-filter="both" aria-pressed="false">Both</button>
  </div>
  <div id="chunks"></div>

  <script src="/static/js/api.js"></script>
  <script src="/static/js/query.js"></script>
</body>
</html>
```

```js
// app/static/js/query.js
let lastChunks = [];
let currentFilter = "all";
let citationsByMarker = {};

function matchesFilter(chunk, filter) {
  if (filter === "all") return true;
  if (filter === "both") return chunk.matched_methods.length === 2;
  return chunk.matched_methods.length === 1 && chunk.matched_methods[0] === filter;
}

function scrollToChunkByMarker(marker) {
  const chunkId = citationsByMarker[marker];
  if (!chunkId) return;
  const card = document.querySelector(`[data-chunk-id="${chunkId}"]`);
  if (!card) return;
  card.scrollIntoView({ behavior: "smooth", block: "center" });
  card.classList.add("highlight");
  setTimeout(() => card.classList.remove("highlight"), 1500);
}

function renderAnswer(answer) {
  const container = document.getElementById("answer");
  container.innerHTML = "";
  const parts = answer.split(/(\[\d+\])/g);
  for (const part of parts) {
    const match = part.match(/^\[(\d+)\]$/);
    if (match) {
      const marker = document.createElement("span");
      marker.className = "citation-marker";
      marker.textContent = part;
      marker.onclick = () => scrollToChunkByMarker(match[1]);
      container.appendChild(marker);
    } else {
      container.appendChild(document.createTextNode(part));
    }
  }
}

function renderChunks() {
  const container = document.getElementById("chunks");
  container.innerHTML = "";
  for (const chunk of lastChunks) {
    if (!matchesFilter(chunk, currentFilter)) continue;

    const card = document.createElement("div");
    card.className = `chunk-card ${chunk.used_in_synthesis ? "used" : "not-used"}`;
    card.dataset.chunkId = chunk.chunk_id;

    const source = document.createElement("div");
    source.textContent = `${chunk.source_url} — page ${chunk.page_number}`;

    const badges = document.createElement("div");
    const badgeParts = [`<span class="badge">Fused #${chunk.fused_rank} (${chunk.rrf_score.toFixed(3)})</span>`];
    if (chunk.bm25_rank != null) {
      badgeParts.unshift(`<span class="badge">BM25 #${chunk.bm25_rank} (${chunk.bm25_score.toFixed(2)})</span>`);
    }
    if (chunk.semantic_rank != null) {
      badgeParts.splice(1, 0, `<span class="badge">Semantic #${chunk.semantic_rank} (${chunk.semantic_score.toFixed(2)})</span>`);
    }
    badges.innerHTML = badgeParts.join("");

    const text = document.createElement("p");
    text.textContent = chunk.text;

    const usedNote = document.createElement("div");
    usedNote.textContent = chunk.used_in_synthesis ? "Used in answer" : "Retrieved but not used";

    card.appendChild(source);
    card.appendChild(badges);
    card.appendChild(text);
    card.appendChild(usedNote);
    container.appendChild(card);
  }
}

document.getElementById("filter-toggle").addEventListener("click", (event) => {
  const button = event.target.closest("button[data-filter]");
  if (!button) return;
  currentFilter = button.dataset.filter;
  for (const btn of document.querySelectorAll("#filter-toggle button")) {
    btn.setAttribute("aria-pressed", String(btn === button));
  }
  renderChunks();
});

document.getElementById("ask").onclick = async () => {
  const query = document.getElementById("query-input").value.trim();
  if (!query) return;
  const result = await postQuery(query);
  lastChunks = result.retrieved_chunks;
  citationsByMarker = Object.fromEntries(result.citations.map((c) => [String(c.marker), c.chunk_id]));
  renderAnswer(result.answer);
  renderChunks();
};
```

```python
# app/main.py — add route (alongside the "/" route from Task 2)
    @app.get("/query-ui")
    async def serve_query_page() -> FileResponse:
        return FileResponse("app/static/query.html")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/frontend/test_query_page.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Run the full backend test suite**

Run: `uv run pytest -v`
Expected: all tests across all four plans PASS

- [ ] **Step 6: Commit**

```bash
git add app/static/query.html app/static/js/query.js app/main.py tests/frontend/test_query_page.py
git commit -m "feat: add Query screen (answer, citations, chunk cards, matched-methods filter)"
```

---

### Task 4: Manual end-to-end browser verification

**Files:** none (verification only; fix inline and commit if a real bug surfaces)

This task has no automated test — see Global Constraints for why. It is the actual functional sign-off for this plan.

- [ ] **Step 1: Start dependencies**

Ensure Ollama is running with the embedding model pulled (`ollama pull qwen3-embedding:0.6b`), export `GROQ_API_KEY` in the shell, and confirm Qdrant's local on-disk mode needs no separate server (already true per Plan 2).

- [ ] **Step 2: Start the app**

Run: `uv run uvicorn app.main:app --reload`

- [ ] **Step 3: Exercise the Ingest screen**

Open `http://localhost:8000/`. Add two URL rows, remove one, add another back, then submit with two real URLs. Confirm: a progress row appears per URL, the stage badge advances through the stage enum (`queued → fetching → paginating?/cleaning → indexing → done`), page counts update, and polling stops once both rows are terminal (`done`/`error`).

- [ ] **Step 4: Exercise the Query screen**

Open `http://localhost:8000/query-ui`. Ask a question about the content just ingested. Confirm: the answer renders with clickable `[n]` markers; clicking a marker scrolls to and highlights the matching chunk card; each chunk card shows source + page and three badges (BM25, semantic, fused — the two search-specific badges only appear when that method actually matched); the filter toggle correctly narrows to BM25-only / semantic-only / both; chunks used in the answer are visually distinct from chunks retrieved but not used.

- [ ] **Step 5: Record the result**

If everything in Steps 3–4 holds, note "manual verification passed" in the SDD ledger for this plan. If something is broken, fix it directly, re-run Steps 3–4, then commit the fix with a normal descriptive message (not a plan-step commit).

---

## Self-Review Notes

- **Spec coverage:** Screen 1 (URL rows, submit, per-URL progress) — Task 2. Screen 2 (query input, cited answer, chunk cards with three badges, matched-methods filter, used-vs-not-used marker) — Task 3. Static serving — Task 1. Functional sign-off — Task 4.
- **Type consistency checked:** `ingest.js` and `query.js` consume the exact JSON field names from Plan 1's `JobStatusResponse`/`UrlStatus` and Plan 3's `QueryResponse`/`FusedChunk` — no renamed fields on the client side.
- **Deliberate scope boundary:** no automated JS tests — documented once in Global Constraints rather than repeated per task, and Task 4 is the explicit compensating control.
- **No placeholders:** every file's full, real, working content is given inline; no "TBD" left.
