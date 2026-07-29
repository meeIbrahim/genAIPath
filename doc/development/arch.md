# RAG indexing, retrieval & base system — architecture spec

## Scope
Covers everything downstream of ingestion (see `rag-ingestion-spec.md`): chunking + dual indexing, hybrid retrieval + fusion, answer synthesis, and the base UI/BE scaffolding that ties ingestion → indexing → retrieval together. This is the handoff contract — component responsibilities, data contracts, and API shapes, not implementation code.

---

## 1. System overview

```
[Ingestion]  →  [Indexing]  →  [Retrieval]  →  [Synthesis]  →  [UI]
 (separate spec)   chunk +        BM25 +          LLM +          query box,
                    dual index    semantic +       citations      chunk cards,
                                  RRF fusion                       progress
```

Two persistent stores back the whole system:
- **Document store** — raw chunk text + metadata (source of truth for citations)
- **Index pair** — BM25 index + vector index, both keyed by the same `chunk_id`

---

## 2. Indexing module

### Input contract (from ingestion)
```json
{
  "source_url": "https://example.com/blog/post-1",
  "cleaned_text": "...",
  "page_map": [ {"page": 1, "char_start": 0, "char_end": 4210} ],
  "fetched_at": "2026-07-29T12:00:00Z"
}
```

### Responsibilities
1. **Chunking** — token-based split (not char-based), fixed window with overlap.
   - `chunk_size = 400 tokens`, `overlap = 75 tokens` (config, not hardcoded).
   - Respect sentence boundaries where feasible to avoid mid-sentence overlap cuts.
   - Resolve each chunk's `page_number` via `page_map` (char offset lookup).
2. **Metadata assembly** — every chunk gets a fixed schema, this is the contract retrieval and UI both depend on:
```json
{
  "chunk_id": "uuid",
  "doc_id": "uuid",
  "source_url": "https://example.com/blog/post-1",
  "page_number": 1,
  "chunk_index": 3,
  "char_start": 1200,
  "char_end": 1650,
  "overlap_with_prev": 75,
  "indexed_at": "2026-07-29T12:00:05Z",
  "text": "..."
}
```
3. **Dual write** — same `chunk_id` written to both indexes; a chunk must never exist in one and not the other (write both or roll back both — treat as one transaction at the app level even if the underlying stores don't support real transactions).
   - **BM25 index**: tokenized text → term-frequency structure. In-memory (`rank_bm25`) is fine for showcase scale; swap for OpenSearch/Elasticsearch if corpus grows past what fits in memory.
   - **Vector index**: embed `text` (via Ollama, `qwen3-embedding:0.6b` or higher) → dense vector → upsert into Qdrant with full metadata attached as payload (needed later for score display, not just retrieval).

### Non-goals
- No re-chunking strategy switching at query time — chunk size is an indexing-time decision, changing it requires re-indexing.
- No dedup/near-dup detection across sources in v1 — flag as future work, don't build it into this handoff.

### API surface
`POST /index/chunk` — internal, called by ingestion worker per completed document, not user-facing.
```json
{ "doc_id": "uuid", "status": "indexed", "chunk_count": 14 }
```

---

## 3. Retrieval module

### Query contract
`POST /query`
```json
{ "query": "text", "top_k": 5 }
```

### Responsibilities
1. **Parallel dispatch** — fire BM25 search and semantic search concurrently (not sequentially), each independently configurable for `top_k` (retrieval `top_k` can exceed final display `top_k`, e.g. retrieve 20 per method, fuse, then trim to 5).
2. **BM25 search** — tokenize query, score against index, return `[{chunk_id, bm25_score, bm25_rank}]`.
3. **Semantic search** — embed query via Ollama `qwen3-embedding:0.6b`+ (same model as indexing — this must match, mismatched embedding models silently degrades relevance), cosine similarity against Qdrant, return `[{chunk_id, semantic_score, semantic_rank}]`.
4. **Fusion (RRF)** — merge both ranked lists:
   ```
   rrf_score(chunk) = Σ 1 / (k + rank_in_list)   over lists containing chunk, k = 60 (config)
   ```
   - A chunk appearing in only one list still gets a score (from that one list) — don't drop single-method hits.
   - Sort by `rrf_score` descending, take final `top_k`.
5. **Result assembly** — for each fused chunk, attach full metadata + both raw sub-scores (this is what the UI needs to show method-level transparency):
```json
{
  "chunk_id": "uuid",
  "text": "...",
  "source_url": "...",
  "page_number": 1,
  "bm25_rank": 2, "bm25_score": 8.31,
  "semantic_rank": 1, "semantic_score": 0.87,
  "fused_rank": 1, "rrf_score": 0.033,
  "matched_methods": ["bm25", "semantic"]
}
```
   `matched_methods` drives the "only BM25 / only semantic / both" UI filter directly — assemble it here, don't recompute client-side.

### Response contract
`POST /query` →
```json
{
  "query": "text",
  "answer": "synthesized answer with [1][2] markers",
  "citations": [
    { "marker": 1, "chunk_id": "uuid" },
    { "marker": 2, "chunk_id": "uuid" }
  ],
  "retrieved_chunks": [ /* array of the fused chunk objects above, same order as citations where applicable */ ]
}
```

---

## 4. Synthesis module
- Model: Groq API, `openai/gpt-oss-120b`.
- Prompt template: system instructs the LLM to answer **only from provided chunks**, and to mark claims with `[n]` referencing the chunk's position in the provided context list — this is what makes `citations` in the response contract mappable back to `retrieved_chunks`.
- Pass chunks in `fused_rank` order, capped to a context budget (config, e.g. top 5–8 chunks) — not all retrieved chunks need to go to the LLM even if more are shown in the UI's "retrieved chunks" panel (UI can show more chunks than were actually used for synthesis; if so, mark which ones were used vs. just retrieved).
- No citation → no claim: if the LLM produces an unmarked sentence, that's a prompt-quality issue to iterate on, not something the backend needs to enforce structurally in v1.

---

## 5. Base UI/BE scaffolding

### Backend
- Framework: FastAPI (async-native, matches the async ingestion worker model).
- Endpoints owned by this spec: `POST /index/chunk` (internal), `POST /query` (user-facing). Ingestion endpoints (`POST /ingest`, `GET /ingest/{job_id}/status`) are defined in the ingestion spec but live in the same service.
- State: document store + BM25 index can be in-process for a showcase; vector store as a separate lightweight service/library call. No auth/multi-tenancy in v1 — single shared corpus.

### Frontend
Two screens, kept deliberately minimal since the product **is** the transparency, not the polish:

**Screen 1 — Ingest**
- URL list input (add/remove rows) + submit.
- Per-URL progress row (stage + page count), per ingestion spec's status contract.

**Screen 2 — Query**
- Single query input.
- Answer block with inline `[n]` citation markers (click → scroll/highlight the matching chunk card below).
- Retrieved chunks panel: one card per chunk showing text, source + page, and three badges (BM25 rank/score, semantic rank/score, fused rank/score); filter toggle for `matched_methods` (all / BM25-only / semantic-only / both).
- Chunks that were actually passed to the LLM (vs. just retrieved) get a visual marker distinguishing "used in answer" from "retrieved but not used."

### Data flow recap (ties both specs together)
```
Ingest UI → /ingest → ingestion worker → cleaned text → /index/chunk → dual index write
Query UI  → /query  → parallel BM25 + semantic search → RRF fusion → LLM synthesis → answer + chunk cards
```

## Resolved decisions
- **Embedding model**: Ollama, `qwen3-embedding:0.6b` or higher — local serve, no external calls.
- **Vector store**: Qdrant — separate service, persistence-ready.
- **BM25 store**: in-memory `rank_bm25` — fine at showcase scale; revisit only if corpus outgrows memory.
- **Synthesis LLM**: Groq API, `openai/gpt-oss-120b`.
- **RRF defaults**: `k=60`, retrieve `top_k=20` per method, fuse, trim to display `top_k=5`. Tune after seeing real query behavior.
