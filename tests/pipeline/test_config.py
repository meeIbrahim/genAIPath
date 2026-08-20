import pytest
from pydantic import ValidationError

from app.pipeline.config import PipelineConfig, collection_name_for, get_active, set_active


@pytest.fixture(autouse=True)
def _reset_active_pipeline():
    yield
    set_active(None)


def test_get_active_defaults_to_none():
    assert get_active() is None


def test_set_active_then_get_active_round_trips():
    config = PipelineConfig(indexing_strategy="fixed_window", retrieval_strategy="hybrid_rrf", post_retrieval_strategy="none")
    set_active(config)
    assert get_active() == config


def test_pipeline_config_rejects_unknown_strategy_id():
    with pytest.raises(ValidationError):
        PipelineConfig(indexing_strategy="not_a_real_strategy", retrieval_strategy="hybrid_rrf", post_retrieval_strategy="none")


def test_collection_name_for_namespaces_by_strategy():
    assert collection_name_for("rag_chunks", "fixed_window") == "rag_chunks__fixed_window"
