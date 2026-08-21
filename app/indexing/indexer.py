from __future__ import annotations

import uuid
from datetime import datetime, timezone

import httpx

from app.config import Settings, settings as default_settings
from app.extraction.location import extract_city
from app.extraction.price import extract_price
from app.indexing.bm25_index import InMemoryBM25Index
from app.indexing.chunk_store import ChunkStore
from app.indexing.chunker import TextChunk
from app.indexing.embeddings import embed_texts
from app.indexing.models import ChunkMetadata, IndexResult
from app.indexing.vector_index import QdrantVectorIndex


async def index_chunks(
    text_chunks: list[TextChunk],
    source_name: str,
    doc_id_hash: str,
    bm25_index: InMemoryBM25Index,
    vector_index: QdrantVectorIndex,
    chunk_store: ChunkStore,
    http_client: httpx.AsyncClient,
    settings: Settings = default_settings,
) -> IndexResult:
    doc_id = str(uuid.uuid4())
    indexed_at = datetime.now(timezone.utc).isoformat()

    chunk_metadatas = [
        ChunkMetadata(
            chunk_id=str(uuid.uuid4()),
            doc_id=doc_id,
            doc_id_hash=doc_id_hash,
            source_url=source_name,
            # TODO(piece B): PDF page boundaries aren't tracked through chunking strategies yet;
            # every chunk reports page 1. Deliberate simplification, not a bug.
            page_number=1,
            chunk_index=index,
            char_start=text_chunk.char_start,
            char_end=text_chunk.char_end,
            overlap_with_prev=text_chunk.overlap_with_prev,
            indexed_at=indexed_at,
            text=text_chunk.text,
            city=extract_city(text_chunk.text),
            price=extract_price(text_chunk.text),
        )
        for index, text_chunk in enumerate(text_chunks)
    ]

    if not chunk_metadatas:
        return IndexResult(doc_id=doc_id, status="indexed", chunk_count=0)

    chunk_ids = [chunk.chunk_id for chunk in chunk_metadatas]
    texts = [chunk.text for chunk in chunk_metadatas]
    vectors = await embed_texts(http_client, texts, settings)

    bm25_index.add_documents(chunk_ids, texts)
    try:
        vector_index.upsert(chunk_ids, vectors, [chunk.model_dump() for chunk in chunk_metadatas])
    except Exception:
        bm25_index.remove_documents(set(chunk_ids))
        raise

    chunk_store.add(chunk_metadatas)

    return IndexResult(doc_id=doc_id, status="indexed", chunk_count=len(chunk_metadatas))
