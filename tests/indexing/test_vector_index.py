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
