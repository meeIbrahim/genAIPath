from app.retrieval.strategies import RETRIEVAL_STRATEGIES


def test_registry_has_exactly_the_three_expected_strategies():
    assert set(RETRIEVAL_STRATEGIES.keys()) == {"bm25_only", "semantic_only", "hybrid_rrf"}
