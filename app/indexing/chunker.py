from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

import tiktoken

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
_ENCODING = tiktoken.get_encoding("cl100k_base")


@dataclass(frozen=True)
class TextChunk:
    text: str
    char_start: int
    char_end: int
    overlap_with_prev: int


def sentence_spans(text: str) -> list[tuple[int, int]]:
    if not text:
        return []
    spans = []
    start = 0
    for match in _SENTENCE_BOUNDARY_RE.finditer(text):
        spans.append((start, match.start()))
        start = match.end()
    spans.append((start, len(text)))
    return [span for span in spans if span[1] > span[0]]


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
    token_counter: Callable[[str], int] = count_tokens,
) -> list[TextChunk]:
    sentences = [(s, e, token_counter(text[s:e])) for s, e in sentence_spans(text)]
    n = len(sentences)
    chunks: list[TextChunk] = []
    i = 0
    prev_overlap = 0

    while i < n:
        window_end = i
        window_tokens = 0
        while window_end < n:
            tok = sentences[window_end][2]
            if window_end > i and window_tokens + tok > chunk_size:
                break
            window_tokens += tok
            window_end += 1

        chunk_start = sentences[i][0]
        chunk_end = sentences[window_end - 1][1]
        chunks.append(TextChunk(text[chunk_start:chunk_end], chunk_start, chunk_end, prev_overlap))

        if window_end >= n:
            break

        overlap_tokens = 0
        k = window_end - 1
        while k >= i and overlap_tokens < overlap:
            overlap_tokens += sentences[k][2]
            k -= 1
        next_i = k + 1
        if next_i <= i:
            next_i = window_end
            overlap_tokens = 0
        prev_overlap = overlap_tokens
        i = next_i

    return chunks
