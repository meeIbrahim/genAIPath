import asyncio
import json
from pathlib import Path

import httpx
import pytest

import app.pipeline.loader as loader_module
from app.config import Settings
from app.indexing.models import IndexResult
from app.pipeline.config import PipelineConfig, get_active, set_active
from app.pipeline.loader import load_pipeline
from app.pipeline.registry import IndexingCollectionRegistry


@pytest.fixture(autouse=True)
def _reset_active_pipeline():
    yield
    set_active(None)


def _settings(tmp_path) -> Settings:
    return Settings(
        qdrant_path=str(tmp_path / "qdrant"), qdrant_collection="ploader", vector_size=2,
        chunk_size_tokens=6, chunk_overlap_tokens=0,
    )


def _embed_client() -> httpx.AsyncClient:
    def handler(request):
        body = json.loads(request.read())
        return httpx.Response(200, json={"embeddings": [[0.1, 0.2] for _ in body["input"]]})
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def _make_archive(tmp_path, names: list[str]) -> Path:
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    for name in names:
        (archive_dir / name).write_bytes(name.encode())
    return archive_dir


def _config(**overrides) -> PipelineConfig:
    defaults = dict(indexing_strategy="fixed_window", retrieval_strategy="hybrid_rrf", post_retrieval_strategy="none")
    defaults.update(overrides)
    return PipelineConfig(**defaults)


async def test_load_pipeline_indexes_new_docs_and_sets_active(tmp_path, monkeypatch):
    monkeypatch.setattr(loader_module, "extract_pdf_text", lambda path: "one two three. four five six.")
    settings = _settings(tmp_path)
    registry = IndexingCollectionRegistry(settings)
    archive_dir = _make_archive(tmp_path, ["a.pdf", "b.pdf"])
    config = _config()

    client = _embed_client()
    result = await load_pipeline(config, registry, client, settings, archive_dir=archive_dir)
    await client.aclose()
    registry.close_all()

    assert result.indexed.new_docs == 2
    assert result.indexed.total_docs == 2
    assert result.indexed.failures == []
    assert get_active() == config


async def test_load_pipeline_skips_already_indexed_docs(tmp_path, monkeypatch):
    monkeypatch.setattr(loader_module, "extract_pdf_text", lambda path: "one two three. four five six.")
    settings = _settings(tmp_path)
    registry = IndexingCollectionRegistry(settings)
    archive_dir = _make_archive(tmp_path, ["a.pdf"])
    config = _config()

    client = _embed_client()
    first = await load_pipeline(config, registry, client, settings, archive_dir=archive_dir)
    second = await load_pipeline(config, registry, client, settings, archive_dir=archive_dir)
    await client.aclose()
    registry.close_all()

    assert first.indexed.new_docs == 1
    assert second.indexed.new_docs == 0
    assert second.indexed.total_docs == 1


async def test_load_pipeline_isolates_strategies_in_separate_collections(tmp_path, monkeypatch):
    monkeypatch.setattr(loader_module, "extract_pdf_text", lambda path: "one two three. four five six.")
    settings = _settings(tmp_path)
    registry = IndexingCollectionRegistry(settings)
    archive_dir = _make_archive(tmp_path, ["a.pdf"])

    client = _embed_client()
    await load_pipeline(_config(indexing_strategy="fixed_window"), registry, client, settings, archive_dir=archive_dir)
    await client.aclose()
    registry.close_all()

    settings2 = _settings(tmp_path)
    registry2 = IndexingCollectionRegistry(settings2)
    assert registry2.doc_count("fixed_window") == 1
    assert registry2.doc_count("semantic") == 0
    registry2.close_all()


async def test_load_pipeline_records_per_doc_failure_without_stopping_batch(tmp_path, monkeypatch):
    def flaky_extract(path):
        if path.name == "bad.pdf":
            raise RuntimeError("corrupt pdf")
        return "one two three. four five six."

    monkeypatch.setattr(loader_module, "extract_pdf_text", flaky_extract)
    settings = _settings(tmp_path)
    registry = IndexingCollectionRegistry(settings)
    archive_dir = _make_archive(tmp_path, ["bad.pdf", "good.pdf"])
    config = _config()

    client = _embed_client()
    result = await load_pipeline(config, registry, client, settings, archive_dir=archive_dir)
    await client.aclose()
    registry.close_all()

    assert result.indexed.new_docs == 1
    assert len(result.indexed.failures) == 1
    assert result.indexed.failures[0].path.endswith("bad.pdf")
    assert result.indexed.failures[0].error == "corrupt pdf"


async def test_load_pipeline_empty_archive_returns_zero_without_error(tmp_path):
    settings = _settings(tmp_path)
    registry = IndexingCollectionRegistry(settings)
    config = _config()

    client = _embed_client()
    result = await load_pipeline(config, registry, client, settings, archive_dir=tmp_path / "missing")
    await client.aclose()
    registry.close_all()

    assert result.indexed.new_docs == 0
    assert result.indexed.total_docs == 0


async def test_load_pipeline_propagates_not_implemented_for_stub_indexing_strategy(tmp_path, monkeypatch):
    monkeypatch.setattr(loader_module, "extract_pdf_text", lambda path: "one two three. four five six.")
    settings = _settings(tmp_path)
    registry = IndexingCollectionRegistry(settings)
    archive_dir = _make_archive(tmp_path, ["a.pdf"])
    config = _config(indexing_strategy="semantic")

    client = _embed_client()
    with pytest.raises(NotImplementedError):
        await load_pipeline(config, registry, client, settings, archive_dir=archive_dir)
    await client.aclose()
    registry.close_all()

    assert get_active() is None


async def test_load_pipeline_records_zero_chunk_doc_as_failure_and_retries_it_next_time(tmp_path, monkeypatch):
    monkeypatch.setattr(loader_module, "extract_pdf_text", lambda path: "one two three. four five six.")

    async def empty_index_chunks(*args, **kwargs):
        return IndexResult(doc_id="d0", status="indexed", chunk_count=0)

    monkeypatch.setattr(loader_module, "index_chunks", empty_index_chunks)

    settings = _settings(tmp_path)
    registry = IndexingCollectionRegistry(settings)
    archive_dir = _make_archive(tmp_path, ["blank.pdf"])
    config = _config()

    client = _embed_client()
    first = await load_pipeline(config, registry, client, settings, archive_dir=archive_dir)
    second = await load_pipeline(config, registry, client, settings, archive_dir=archive_dir)
    await client.aclose()
    registry.close_all()

    assert first.indexed.new_docs == 0
    assert first.indexed.total_docs == 0
    assert len(first.indexed.failures) == 1
    assert first.indexed.failures[0].path.endswith("blank.pdf")
    assert first.indexed.failures[0].error == "no extractable chunks"

    # The doc's hash was never stored, so the next Load must retry it rather than
    # silently skipping it or counting it as already-indexed.
    assert second.indexed.new_docs == 0
    assert len(second.indexed.failures) == 1
    assert second.indexed.failures[0].path.endswith("blank.pdf")


async def test_load_pipeline_dedupes_identical_content_within_one_batch(tmp_path, monkeypatch):
    monkeypatch.setattr(loader_module, "extract_pdf_text", lambda path: "one two three. four five six.")
    settings = _settings(tmp_path)
    registry = IndexingCollectionRegistry(settings)

    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    (archive_dir / "original.pdf").write_bytes(b"identical bytes")
    (archive_dir / "copy.pdf").write_bytes(b"identical bytes")
    config = _config()

    client = _embed_client()
    result = await load_pipeline(config, registry, client, settings, archive_dir=archive_dir)
    await client.aclose()
    registry.close_all()

    assert result.indexed.new_docs == 1
    assert result.indexed.total_docs == 1
    assert result.indexed.failures == []


async def test_load_pipeline_serializes_concurrent_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(loader_module, "extract_pdf_text", lambda path: "one two three. four five six.")
    settings = _settings(tmp_path)
    registry = IndexingCollectionRegistry(settings)
    archive_dir = _make_archive(tmp_path, ["a.pdf"])
    config = _config()

    concurrent = {"active": 0, "max": 0}
    real_index_chunks = loader_module.index_chunks

    async def slow_index_chunks(*args, **kwargs):
        concurrent["active"] += 1
        concurrent["max"] = max(concurrent["max"], concurrent["active"])
        await asyncio.sleep(0.05)
        result = await real_index_chunks(*args, **kwargs)
        concurrent["active"] -= 1
        return result

    monkeypatch.setattr(loader_module, "index_chunks", slow_index_chunks)

    client = _embed_client()
    await asyncio.gather(
        load_pipeline(config, registry, client, settings, archive_dir=archive_dir),
        load_pipeline(config, registry, client, settings, archive_dir=archive_dir),
    )
    await client.aclose()
    registry.close_all()

    assert concurrent["max"] == 1
