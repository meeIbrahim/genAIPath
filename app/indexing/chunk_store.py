from __future__ import annotations

from app.indexing.models import ChunkMetadata


class ChunkStore:
    def __init__(self) -> None:
        self._chunks: dict[str, ChunkMetadata] = {}

    def add(self, chunks: list[ChunkMetadata]) -> None:
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk

    def remove(self, chunk_ids: list[str]) -> None:
        for chunk_id in chunk_ids:
            self._chunks.pop(chunk_id, None)

    def get(self, chunk_id: str) -> ChunkMetadata | None:
        return self._chunks.get(chunk_id)

    def get_many(self, chunk_ids: list[str]) -> list[ChunkMetadata]:
        return [self._chunks[chunk_id] for chunk_id in chunk_ids if chunk_id in self._chunks]
