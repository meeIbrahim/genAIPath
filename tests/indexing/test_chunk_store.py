from app.indexing.chunk_store import ChunkStore
from app.indexing.models import ChunkMetadata


def _chunk(chunk_id: str) -> ChunkMetadata:
    return ChunkMetadata(
        chunk_id=chunk_id,
        doc_id="d1",
        source_url="https://example.com",
        page_number=1,
        chunk_index=0,
        char_start=0,
        char_end=10,
        overlap_with_prev=0,
        indexed_at="2026-07-29T12:00:00+00:00",
        text="hello",
    )


def test_add_and_get():
    store = ChunkStore()
    store.add([_chunk("c1"), _chunk("c2")])
    assert store.get("c1").chunk_id == "c1"
    assert store.get("does-not-exist") is None


def test_get_many_preserves_requested_order_and_skips_missing():
    store = ChunkStore()
    store.add([_chunk("c1"), _chunk("c2")])
    result = store.get_many(["c2", "missing", "c1"])
    assert [c.chunk_id for c in result] == ["c2", "c1"]


def test_remove_deletes_chunks():
    store = ChunkStore()
    store.add([_chunk("c1"), _chunk("c2")])
    store.remove(["c1"])
    assert store.get("c1") is None
    assert store.get("c2") is not None


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
