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
