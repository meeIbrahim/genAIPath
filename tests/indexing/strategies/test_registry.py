import pytest

from app.config import Settings
from app.indexing.chunker import chunk_text
from app.indexing.strategies import INDEXING_STRATEGIES


def test_registry_has_exactly_the_four_expected_strategies():
    assert set(INDEXING_STRATEGIES.keys()) == {
        "fixed_window", "semantic", "hierarchical", "hierarchical_summary",
    }


def test_fixed_window_matches_chunk_text_output():
    settings = Settings(chunk_size_tokens=6, chunk_overlap_tokens=0)
    text = "one two three. four five six."
    assert INDEXING_STRATEGIES["fixed_window"](text, settings) == chunk_text(text, 6, 0)


@pytest.mark.parametrize("strategy_id", ["semantic", "hierarchical", "hierarchical_summary"])
def test_stub_strategies_raise_not_implemented(strategy_id):
    with pytest.raises(NotImplementedError):
        INDEXING_STRATEGIES[strategy_id]("some text", Settings())
