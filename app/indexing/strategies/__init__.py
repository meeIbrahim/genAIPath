from __future__ import annotations

from typing import Callable

from app.config import Settings
from app.indexing.chunker import TextChunk
from app.indexing.strategies import fixed_window, hierarchical, hierarchical_summary, semantic

IndexingStrategyFn = Callable[[str, Settings], list[TextChunk]]

INDEXING_STRATEGIES: dict[str, IndexingStrategyFn] = {
    "fixed_window": fixed_window.chunk,
    "semantic": semantic.chunk,
    "hierarchical": hierarchical.chunk,
    "hierarchical_summary": hierarchical_summary.chunk,
}
