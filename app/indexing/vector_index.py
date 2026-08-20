from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import Settings


class QdrantVectorIndex:
    def __init__(self, settings: Settings, client: QdrantClient | None = None) -> None:
        if client is not None:
            self._client = client
        elif settings.qdrant_url:
            self._client = QdrantClient(url=settings.qdrant_url)
        else:
            self._client = QdrantClient(path=settings.qdrant_path)
        self._collection = settings.qdrant_collection
        self._ensure_collection(settings.vector_size)

    def _ensure_collection(self, vector_size: int) -> None:
        existing = [c.name for c in self._client.get_collections().collections]
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
            )

    def upsert(self, chunk_ids: list[str], vectors: list[list[float]], payloads: list[dict]) -> None:
        points = [
            qmodels.PointStruct(id=chunk_id, vector=vector, payload=payload)
            for chunk_id, vector, payload in zip(chunk_ids, vectors, payloads)
        ]
        self._client.upsert(collection_name=self._collection, points=points)

    def delete(self, chunk_ids: list[str]) -> None:
        self._client.delete(
            collection_name=self._collection,
            points_selector=qmodels.PointIdsList(points=chunk_ids),
        )

    def search(self, query_vector: list[float], top_k: int) -> list[tuple[str, float]]:
        results = self._client.query_points(
            collection_name=self._collection, query=query_vector, limit=top_k
        ).points
        return [(str(point.id), point.score) for point in results]

    def scroll_all(self) -> list[dict]:
        payloads: list[dict] = []
        offset = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self._collection, limit=256, with_payload=True, with_vectors=False, offset=offset
            )
            payloads.extend(point.payload for point in points)
            if offset is None:
                break
        return payloads

    def close(self) -> None:
        self._client.close()
