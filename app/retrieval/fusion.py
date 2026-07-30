from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RankedHit:
    chunk_id: str
    rank: int
    score: float


@dataclass(frozen=True)
class FusedHit:
    chunk_id: str
    fused_rank: int
    rrf_score: float
    bm25_rank: int | None
    bm25_score: float | None
    semantic_rank: int | None
    semantic_score: float | None
    matched_methods: list[str]


def reciprocal_rank_fusion(
    bm25_hits: list[RankedHit],
    semantic_hits: list[RankedHit],
    k: int,
    top_k: int,
) -> list[FusedHit]:
    bm25_by_id = {hit.chunk_id: hit for hit in bm25_hits}
    semantic_by_id = {hit.chunk_id: hit for hit in semantic_hits}
    ordered_ids = list(dict.fromkeys([h.chunk_id for h in bm25_hits] + [h.chunk_id for h in semantic_hits]))

    scored = []
    for chunk_id in ordered_ids:
        bm25_hit = bm25_by_id.get(chunk_id)
        semantic_hit = semantic_by_id.get(chunk_id)
        rrf_score = 0.0
        matched_methods = []
        if bm25_hit:
            rrf_score += 1.0 / (k + bm25_hit.rank)
            matched_methods.append("bm25")
        if semantic_hit:
            rrf_score += 1.0 / (k + semantic_hit.rank)
            matched_methods.append("semantic")
        scored.append((chunk_id, rrf_score, bm25_hit, semantic_hit, matched_methods))

    scored.sort(key=lambda item: item[1], reverse=True)

    return [
        FusedHit(
            chunk_id=chunk_id,
            fused_rank=index + 1,
            rrf_score=rrf_score,
            bm25_rank=bm25_hit.rank if bm25_hit else None,
            bm25_score=bm25_hit.score if bm25_hit else None,
            semantic_rank=semantic_hit.rank if semantic_hit else None,
            semantic_score=semantic_hit.score if semantic_hit else None,
            matched_methods=matched_methods,
        )
        for index, (chunk_id, rrf_score, bm25_hit, semantic_hit, matched_methods) in enumerate(scored[:top_k])
    ]
