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
