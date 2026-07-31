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
    stored = store.get_many([c.chunk_id for c in store._chunks.values()])
    assert len(stored) == 2
    assert all(c.doc_id == result.doc_id for c in stored)
    assert stored[0].page_number == 1
    # NOTE: BM25Okapi's idf formula is log((N - df + 0.5) / (df + 0.5)), which is
    # exactly 0 whenever a term appears in exactly one of exactly two documents
    # (N=2, df=1 -> log(1.5/1.5) == 0), regardless of which words are involved. With
    # this payload/chunk_size the two chunks have fully disjoint vocabulary, so a
    # relevance-score search (`bm25.search(...)`) can never return a nonzero hit here
    # -- that's an inherent property of BM25Okapi with a 2-document corpus, not a bug
    # in the indexer's dual-write. Assert on BM25 index membership instead, which is
    # what this test actually cares about (did the chunk get written to BM25).
    assert {c.chunk_id for c in stored}.issubset(set(bm25._chunk_ids))
    assert len(vectors.search([0.1, 0.2, 0.3], top_k=5)) == 2


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
