import httpx

from app.config import Settings
from app.indexing.bm25_index import InMemoryBM25Index
from app.indexing.chunk_store import ChunkStore
from app.indexing.models import ChunkMetadata
from app.indexing.vector_index import QdrantVectorIndex
from app.retrieval.strategies.semantic_only import search


async def test_semantic_only_ignores_bm25_index_and_has_no_bm25_score(tmp_path, monkeypatch):
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
    vectors.upsert([chunk_id], [[1.0, 0.0]], [chunk.model_dump()])
    store.add([chunk])

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("semantic_only must not call bm25_index.search")
    monkeypatch.setattr(bm25, "search", _fail_if_called)

    def handler(request):
        return httpx.Response(200, json={"embeddings": [[1.0, 0.0]]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        results = await search("fox", bm25, vectors, store, client, settings)

    assert len(results) == 1
    assert results[0].matched_methods == ["semantic"]
    assert results[0].bm25_rank is None
    assert results[0].bm25_score is None
