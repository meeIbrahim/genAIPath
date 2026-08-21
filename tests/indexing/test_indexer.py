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


async def test_index_chunks_returns_zero_and_writes_nothing_for_empty_input(tmp_path):
    settings = _settings(tmp_path)
    bm25 = InMemoryBM25Index()
    vectors = QdrantVectorIndex(settings)
    store = ChunkStore()

    def handler(request):  # pragma: no cover - must never be reached
        raise AssertionError("embeddings should not be requested for an empty chunk list")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await index_chunks([], "source.pdf", "somehash", bm25, vectors, store, client, settings)

    assert result.chunk_count == 0
    assert result.status == "indexed"
    assert bm25._chunk_ids == []
    assert store._chunks == {}
    assert store.doc_id_hashes() == set()


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
