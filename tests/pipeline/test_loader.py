import asyncio
import json
from pathlib import Path

import httpx
import pytest

import app.pipeline.loader as loader_module
from app.config import Settings
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


async def test_load_pipeline_serializes_concurrent_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(loader_module, "extract_pdf_text", lambda path: "one two three. four five six.")
    settings = _settings(tmp_path)
    registry = IndexingCollectionRegistry(settings)
    archive_dir = _make_archive(tmp_path, ["a.pdf"])
    config = _config()

    concurrent = {"active": 0, "max": 0}

    async def slow_embed(client, texts, settings):
        concurrent["active"] += 1
        concurrent["max"] = max(concurrent["max"], concurrent["active"])
        await asyncio.sleep(0.05)
        concurrent["active"] -= 1
        return [[0.1, 0.2] for _ in texts]

    monkeypatch.setattr(loader_module, "embed_texts", slow_embed)

    client = _embed_client()
    await asyncio.gather(
        load_pipeline(config, registry, client, settings, archive_dir=archive_dir),
        load_pipeline(config, registry, client, settings, archive_dir=archive_dir),
    )
    await client.aclose()
    registry.close_all()

    assert concurrent["max"] == 1
