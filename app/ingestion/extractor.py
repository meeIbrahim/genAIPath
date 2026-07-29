from __future__ import annotations

import re

import trafilatura
from bs4 import BeautifulSoup
from readability import Document

from app.config import Settings


class ExtractionError(Exception):
    pass


def _normalize(text: str) -> str:
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in text.split("\n\n")]
    paragraphs = [p for p in paragraphs if p]
    return "\n\n".join(paragraphs)


def _extract_with_trafilatura(html: str) -> str | None:
    return trafilatura.extract(html, include_comments=False, include_tables=False)


def _extract_with_readability(html: str) -> str | None:
    try:
        summary_html = Document(html).summary()
    except Exception:
        return None
    return BeautifulSoup(summary_html, "lxml").get_text("\n\n")


def _preprocess_html(html: str) -> str:
    """Remove nav (site navigation) before extraction.

    nav elements are reliably boilerplate (site header/footer nav).
    Other tags like header/footer/aside may contain article content
    (e.g., article header with title/byline, aside with pull-quotes),
    so we leave them for trafilatura's content-scoring heuristics.
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup.find_all("nav"):
        tag.decompose()
    return str(soup)


def extract_main_text(html: str, settings: Settings) -> str:
    preprocessed = _preprocess_html(html)
    text = _extract_with_trafilatura(preprocessed)
    if not text:
        text = _extract_with_readability(preprocessed)

    normalized = _normalize(text or "")
    if len(normalized) < settings.min_extract_length:
        raise ExtractionError("no extractable content")
    return normalized
