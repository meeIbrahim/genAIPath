from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class InMemoryBM25Index:
    def __init__(self) -> None:
        self._chunk_ids: list[str] = []
        self._tokenized_docs: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    def add_documents(self, chunk_ids: list[str], texts: list[str]) -> None:
        for chunk_id, text in zip(chunk_ids, texts):
            self._chunk_ids.append(chunk_id)
            self._tokenized_docs.append(tokenize(text))
        self._rebuild()

    def remove_documents(self, chunk_ids: set[str]) -> None:
        keep = [i for i, cid in enumerate(self._chunk_ids) if cid not in chunk_ids]
        self._chunk_ids = [self._chunk_ids[i] for i in keep]
        self._tokenized_docs = [self._tokenized_docs[i] for i in keep]
        self._rebuild()

    def _rebuild(self) -> None:
        self._bm25 = BM25Okapi(self._tokenized_docs) if self._tokenized_docs else None

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self._chunk_ids, scores), key=lambda pair: pair[1], reverse=True)
        return [(chunk_id, float(score)) for chunk_id, score in ranked[:top_k] if score != 0]
