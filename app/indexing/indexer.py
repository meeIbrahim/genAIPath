# app/indexing/indexer.py
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import httpx

from app.config import Settings, settings as default_settings
from app.extraction.location import extract_city
from app.extraction.price import extract_price
from app.indexing.bm25_index import InMemoryBM25Index
from app.indexing.chunk_store import ChunkStore
from app.indexing.chunker import chunk_text
from app.indexing.embeddings import embed_texts
from app.indexing.models import ChunkMetadata, IndexResult
from app.indexing.vector_index import QdrantVectorIndex
from app.ingestion.models import IngestionPayload, PageMapEntry


def _resolve_page_number(page_map: list[PageMapEntry], char_start: int) -> int:
    for entry in page_map:
        if entry.char_start <= char_start < entry.char_end:
            return entry.page
    return page_map[-1].page if page_map else 1


async def index_document(
    payload: IngestionPayload,
    bm25_index: InMemoryBM25Index,
    vector_index: QdrantVectorIndex,
    chunk_store: ChunkStore,
    http_client: httpx.AsyncClient,
    settings: Settings = default_settings,
) -> IndexResult:
    text_chunks = chunk_text(payload.cleaned_text, settings.chunk_size_tokens, settings.chunk_overlap_tokens)
    doc_id = str(uuid.uuid4())
    indexed_at = datetime.now(timezone.utc).isoformat()

    chunk_metadatas = [
        ChunkMetadata(
            chunk_id=str(uuid.uuid4()),
            doc_id=doc_id,
            source_url=payload.source_url,
            page_number=_resolve_page_number(payload.page_map, text_chunk.char_start),
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
