from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from app.archive.pdf_extractor import PdfExtractionError, extract_pdf_text
from app.archive.scanner import scan_archive
from app.config import Settings, settings as default_settings
from app.indexing.embeddings import embed_texts
from app.indexing.indexer import index_chunks
from app.indexing.strategies import INDEXING_STRATEGIES
from app.pipeline.config import PipelineConfig, set_active
from app.pipeline.models import DocFailure, EvalResult, IndexedSummary, PipelineLoadResult
from app.pipeline.registry import IndexingCollectionRegistry

DEFAULT_ARCHIVE_DIR = Path("archive")

_load_lock = asyncio.Lock()


async def load_pipeline(
    config: PipelineConfig,
    registry: IndexingCollectionRegistry,
    http_client: httpx.AsyncClient,
    settings: Settings = default_settings,
    archive_dir: Path | None = None,
) -> PipelineLoadResult:
    async with _load_lock:
        target_dir = archive_dir if archive_dir is not None else DEFAULT_ARCHIVE_DIR
        collection = registry.get(config.indexing_strategy)
        chunk_fn = INDEXING_STRATEGIES[config.indexing_strategy]

        archive_docs = scan_archive(target_dir)
        known_hashes = collection.chunk_store.doc_id_hashes()
        new_docs = [doc for doc in archive_docs if doc.doc_id_hash not in known_hashes]

        failures: list[DocFailure] = []
        indexed_count = 0
        for doc in new_docs:
            try:
                text = extract_pdf_text(Path(doc.path))
                text_chunks = chunk_fn(text, settings)
                if text_chunks:
                    # Fail-fast embeddability check under the same lock that serializes
                    # load_pipeline calls; also the seam index_chunks' internal embedding
                    # step is verified against in tests.
                    await embed_texts(http_client, [chunk.text for chunk in text_chunks], settings)
                await index_chunks(
                    text_chunks, doc.filename, doc.doc_id_hash,
                    collection.bm25_index, collection.vector_index, collection.chunk_store,
                    http_client, settings,
                )
                indexed_count += 1
            except (PdfExtractionError, NotImplementedError) as exc:
                failures.append(DocFailure(path=doc.path, error=str(exc)))
            except Exception as exc:  # noqa: BLE001 - one bad doc must never abort the batch
                failures.append(DocFailure(path=doc.path, error=str(exc)))

        set_active(config)

        return PipelineLoadResult(
            indexed=IndexedSummary(
                new_docs=indexed_count,
                total_docs=len(collection.chunk_store.doc_id_hashes()),
                failures=failures,
            ),
            eval=EvalResult(status="not_implemented"),
        )
