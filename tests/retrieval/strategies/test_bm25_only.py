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
