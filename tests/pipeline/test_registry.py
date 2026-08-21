import uuid

from app.config import Settings
from app.indexing.models import ChunkMetadata
from app.pipeline.registry import INDEXING_STRATEGY_IDS, IndexingCollectionRegistry


def _settings(tmp_path) -> Settings:
    return Settings(qdrant_path=str(tmp_path / "qdrant"), qdrant_collection="preg", vector_size=2)


def test_registry_creates_one_empty_collection_per_strategy(tmp_path):
    registry = IndexingCollectionRegistry(_settings(tmp_path))
    for strategy_id in INDEXING_STRATEGY_IDS:
        assert registry.doc_count(strategy_id) == 0
    registry.close_all()


def test_registry_isolates_strategies_from_each_other(tmp_path):
    registry = IndexingCollectionRegistry(_settings(tmp_path))
    fixed = registry.get("fixed_window")
    chunk_id = str(uuid.uuid4())
    chunk = ChunkMetadata(
        chunk_id=chunk_id, doc_id="d1", doc_id_hash="h1", source_url="a.pdf", page_number=1,
        chunk_index=0, char_start=0, char_end=10, overlap_with_prev=0,
        indexed_at="2026-08-09T00:00:00+00:00", text="the quick brown fox",
    )
    fixed.vector_index.upsert([chunk_id], [[1.0, 0.0]], [chunk.model_dump()])
    fixed.chunk_store.add([chunk])
    fixed.bm25_index.add_documents([chunk_id], ["the quick brown fox"])

    assert registry.doc_count("fixed_window") == 1
    assert registry.doc_count("semantic") == 0
    assert registry.get("semantic").vector_index.scroll_all() == []
    registry.close_all()


def test_doc_count_counts_docs_not_chunks(tmp_path):
    registry = IndexingCollectionRegistry(_settings(tmp_path))
    fixed = registry.get("fixed_window")

    chunks = [
        ChunkMetadata(
            chunk_id=str(uuid.uuid4()), doc_id="d1", doc_id_hash="same-doc", source_url="a.pdf", page_number=1,
            chunk_index=index, char_start=index * 10, char_end=(index + 1) * 10, overlap_with_prev=0,
            indexed_at="2026-08-09T00:00:00+00:00", text=f"chunk {index} of the quick brown fox",
        )
        for index in range(2)
    ]
    fixed.vector_index.upsert(
        [chunk.chunk_id for chunk in chunks], [[1.0, 0.0], [0.0, 1.0]],
        [chunk.model_dump() for chunk in chunks],
    )
    fixed.chunk_store.add(chunks)

    assert len(fixed.chunk_store._chunks) == 2
    assert registry.doc_count("fixed_window") == 1  # two chunks, one doc
    registry.close_all()


def test_registry_rehydrates_chunk_store_and_bm25_from_persisted_qdrant_data(tmp_path):
    settings = _settings(tmp_path)
    first_registry = IndexingCollectionRegistry(settings)
    collection = first_registry.get("fixed_window")
    chunk_id = str(uuid.uuid4())
    chunk = ChunkMetadata(
        chunk_id=chunk_id, doc_id="d1", doc_id_hash="h1", source_url="paper.pdf", page_number=1,
        chunk_index=0, char_start=0, char_end=10, overlap_with_prev=0,
        indexed_at="2026-08-09T00:00:00+00:00", text="the quick brown fox",
    )
    collection.vector_index.upsert([chunk_id], [[1.0, 0.0]], [chunk.model_dump()])
    first_registry.close_all()  # release the local-mode Qdrant lock before reopening

    second_registry = IndexingCollectionRegistry(settings)
    rehydrated = second_registry.get("fixed_window")

    assert rehydrated.chunk_store.get(chunk_id).text == "the quick brown fox"
    assert rehydrated.bm25_index.search("quick brown fox", top_k=5) != []
    assert second_registry.doc_count("fixed_window") == 1
    second_registry.close_all()
