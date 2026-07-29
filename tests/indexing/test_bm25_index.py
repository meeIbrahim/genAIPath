from app.indexing.bm25_index import InMemoryBM25Index


def test_search_returns_best_matching_chunk_first():
    index = InMemoryBM25Index()
    index.add_documents(
        ["c1", "c2", "c3"],
        [
            "the quick brown fox jumps over the lazy dog",
            "completely unrelated text about kubernetes deployments",
            "another fox story, a second fox appears here",
        ],
    )
    results = index.search("fox", top_k=2)
    result_ids = [chunk_id for chunk_id, _score in results]
    assert result_ids[0] == "c3"  # two mentions of "fox" outranks one
    assert "c2" not in result_ids


def test_search_on_empty_index_returns_empty():
    index = InMemoryBM25Index()
    assert index.search("anything", top_k=5) == []


def test_remove_documents_excludes_them_from_future_searches():
    index = InMemoryBM25Index()
    index.add_documents(["c1", "c2"], ["fox fox fox", "fox"])
    index.remove_documents({"c1"})
    results = index.search("fox", top_k=5)
    assert [chunk_id for chunk_id, _score in results] == ["c2"]


def test_search_excludes_zero_score_non_matches_even_with_large_top_k():
    """Regression test: non-matching docs (score=0) must be excluded from results even when top_k exceeds true match count."""
    index = InMemoryBM25Index()
    index.add_documents(
        ["c1", "c2", "c3"],
        [
            "the quick brown fox jumps over the lazy dog",
            "completely unrelated text about kubernetes deployments",
            "another fox story, a second fox appears here",
        ],
    )
    # top_k=3 exceeds the 2 matching documents (c1, c3)
    results = index.search("fox", top_k=3)
    result_ids = [chunk_id for chunk_id, _score in results]
    # c2 (with score 0.0, no "fox" match) must NOT be in results
    assert "c2" not in result_ids
    # Only the two matching documents should be returned
    assert len(result_ids) == 2
    assert set(result_ids) == {"c1", "c3"}
