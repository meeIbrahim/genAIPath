from app.indexing.models import ChunkMetadata, IndexResult


def test_chunk_metadata_round_trip():
    chunk = ChunkMetadata(
        chunk_id="c1",
        doc_id="d1",
        source_url="https://example.com",
        page_number=1,
        chunk_index=0,
        char_start=0,
        char_end=100,
        overlap_with_prev=0,
        indexed_at="2026-07-29T12:00:00+00:00",
        text="hello world",
    )
    dumped = chunk.model_dump()
    assert dumped["chunk_id"] == "c1"
    assert dumped["overlap_with_prev"] == 0


def test_index_result():
    result = IndexResult(doc_id="d1", status="indexed", chunk_count=3)
    assert result.model_dump() == {"doc_id": "d1", "status": "indexed", "chunk_count": 3}


def test_chunk_metadata_defaults_doc_id_hash_to_empty_string():
    chunk = ChunkMetadata(
        chunk_id="c1", doc_id="d1", source_url="https://example.com", page_number=1,
        chunk_index=0, char_start=0, char_end=100, overlap_with_prev=0,
        indexed_at="2026-07-29T12:00:00+00:00", text="hello world",
    )
    assert chunk.doc_id_hash == ""


def test_chunk_metadata_round_trip_carries_doc_id_hash():
    chunk = ChunkMetadata(
        chunk_id="c1", doc_id="d1", doc_id_hash="abc123", source_url="https://example.com", page_number=1,
        chunk_index=0, char_start=0, char_end=100, overlap_with_prev=0,
        indexed_at="2026-07-29T12:00:00+00:00", text="hello world",
    )
    assert chunk.model_dump()["doc_id_hash"] == "abc123"
