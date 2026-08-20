from __future__ import annotations

import dataclasses

from qdrant_client import QdrantClient

from app.config import Settings
from app.indexing.bm25_index import InMemoryBM25Index
from app.indexing.chunk_store import ChunkStore
from app.indexing.models import ChunkMetadata
from app.indexing.vector_index import QdrantVectorIndex
from app.pipeline.config import IndexingStrategyId, collection_name_for

INDEXING_STRATEGY_IDS: tuple[IndexingStrategyId, ...] = (
    "fixed_window",
    "semantic",
    "hierarchical",
    "hierarchical_summary",
)


@dataclasses.dataclass
class IndexingCollection:
    vector_index: QdrantVectorIndex
    bm25_index: InMemoryBM25Index
    chunk_store: ChunkStore


class IndexingCollectionRegistry:
    def __init__(self, settings: Settings) -> None:
        self._client = QdrantClient(url=settings.qdrant_url) if settings.qdrant_url else QdrantClient(path=settings.qdrant_path)
        self._collections: dict[IndexingStrategyId, IndexingCollection] = {}
        for strategy_id in INDEXING_STRATEGY_IDS:
            strategy_settings = dataclasses.replace(
                settings, qdrant_collection=collection_name_for(settings.qdrant_collection, strategy_id)
            )
            vector_index = QdrantVectorIndex(strategy_settings, client=self._client)
            bm25_index = InMemoryBM25Index()
            chunk_store = ChunkStore()
            _rehydrate(vector_index, bm25_index, chunk_store)
            self._collections[strategy_id] = IndexingCollection(vector_index, bm25_index, chunk_store)

    def get(self, strategy_id: IndexingStrategyId) -> IndexingCollection:
        return self._collections[strategy_id]

    def doc_count(self, strategy_id: IndexingStrategyId) -> int:
        return len(self._collections[strategy_id].chunk_store.doc_id_hashes())

    def close_all(self) -> None:
        self._client.close()


def _rehydrate(vector_index: QdrantVectorIndex, bm25_index: InMemoryBM25Index, chunk_store: ChunkStore) -> None:
    payloads = vector_index.scroll_all()
    if not payloads:
        return
    chunk_metadatas = [ChunkMetadata(**payload) for payload in payloads]
    chunk_store.add(chunk_metadatas)
    bm25_index.add_documents([c.chunk_id for c in chunk_metadatas], [c.text for c in chunk_metadatas])
