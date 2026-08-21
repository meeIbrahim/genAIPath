from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from app.archive.pdf_extractor import extract_pdf_text
from app.archive.scanner import scan_archive
from app.config import Settings, settings as default_settings
from app.indexing.indexer import index_chunks
from app.indexing.strategies import INDEXING_STRATEGIES
from app.pipeline.config import PipelineConfig, set_active
from app.pipeline.models import DocFailure, EvalResult, IndexedSummary, PipelineLoadResult
from app.pipeline.registry import IndexingCollectionRegistry

DEFAULT_ARCHIVE_DIR = Path("archive")

_load_lock = asyncio.Lock()


def run_gold_eval(config: PipelineConfig) -> EvalResult:
    """Seam for piece C: run the gold-set evaluation for a freshly loaded pipeline."""
    return EvalResult(status="not_implemented")


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

        seen_in_batch: set[str] = set()
        failures: list[DocFailure] = []
        indexed_count = 0
        for doc in new_docs:
            if doc.doc_id_hash in seen_in_batch:
                continue  # identical content already indexed earlier in this same batch
            seen_in_batch.add(doc.doc_id_hash)

            try:
                text = extract_pdf_text(Path(doc.path))
            except Exception as exc:  # noqa: BLE001 - one unreadable doc must never abort the batch
                failures.append(DocFailure(path=doc.path, error=str(exc)))
                continue

            # Deliberately uncaught: an unimplemented indexing strategy raises
            # NotImplementedError, which must propagate so /pipeline/load can answer
            # 501 instead of silently activating a pipeline that can never index.
            text_chunks = chunk_fn(text, settings)

            try:
                result = await index_chunks(
                    text_chunks, doc.filename, doc.doc_id_hash,
                    collection.bm25_index, collection.vector_index, collection.chunk_store,
                    http_client, settings,
                )
            except Exception as exc:  # noqa: BLE001 - one bad doc must never abort the batch
                failures.append(DocFailure(path=doc.path, error=str(exc)))
                continue

            if result.chunk_count == 0:
                # index_chunks returns before writing to chunk_store, so this doc's hash is
                # never recorded; count it as a visible failure rather than a phantom new doc.
                failures.append(DocFailure(path=doc.path, error="no extractable chunks"))
            else:
                indexed_count += 1

        set_active(config)

        return PipelineLoadResult(
            indexed=IndexedSummary(
                new_docs=indexed_count,
                total_docs=len(collection.chunk_store.doc_id_hashes()),
                failures=failures,
            ),
            eval=run_gold_eval(config),
        )
