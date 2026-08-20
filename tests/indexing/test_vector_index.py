import uuid

from app.config import Settings
from app.indexing.vector_index import QdrantVectorIndex


def _settings(tmp_path) -> Settings:
    return Settings(qdrant_path=str(tmp_path / "qdrant"), qdrant_collection="test_chunks", vector_size=4)


def test_upsert_and_search_returns_closest_match(tmp_path):
    index = QdrantVectorIndex(_settings(tmp_path))
    id_a, id_b = str(uuid.uuid4()), str(uuid.uuid4())
    index.upsert(
        [id_a, id_b],
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        [{"chunk_id": id_a, "text": "a"}, {"chunk_id": id_b, "text": "b"}],
    )
    results = index.search([1.0, 0.0, 0.0, 0.0], top_k=1)
    assert results[0][0] == id_a


def test_delete_removes_point_from_future_searches(tmp_path):
    index = QdrantVectorIndex(_settings(tmp_path))
    id_a, id_b = str(uuid.uuid4()), str(uuid.uuid4())
    index.upsert(
        [id_a, id_b],
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        [{"chunk_id": id_a}, {"chunk_id": id_b}],
    )
    index.delete([id_a])
    results = index.search([1.0, 0.0, 0.0, 0.0], top_k=5)
    result_ids = [chunk_id for chunk_id, _score in results]
    assert id_a not in result_ids
    assert id_b in result_ids


def test_scroll_all_returns_all_payloads(tmp_path):
    index = QdrantVectorIndex(_settings(tmp_path))
    id_a, id_b = str(uuid.uuid4()), str(uuid.uuid4())
    index.upsert(
        [id_a, id_b],
        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
        [{"chunk_id": id_a, "text": "a"}, {"chunk_id": id_b, "text": "b"}],
    )
    payloads = index.scroll_all()
    assert {p["chunk_id"] for p in payloads} == {id_a, id_b}


def test_scroll_all_empty_collection_returns_empty_list(tmp_path):
    index = QdrantVectorIndex(_settings(tmp_path))
    assert index.scroll_all() == []


def test_shares_provided_client_instead_of_creating_its_own(tmp_path):
    from qdrant_client import QdrantClient

    settings = _settings(tmp_path)
    shared_client = QdrantClient(path=settings.qdrant_path)
    index_a = QdrantVectorIndex(settings, client=shared_client)
    index_b = QdrantVectorIndex(
        Settings(qdrant_path=settings.qdrant_path, qdrant_collection="other", vector_size=4),
        client=shared_client,
    )
    id_a = str(uuid.uuid4())
    index_a.upsert([id_a], [[1.0, 0.0, 0.0, 0.0]], [{"chunk_id": id_a}])
    assert index_b.scroll_all() == []  # different collection, same client, no cross-contamination
    shared_client.close()
