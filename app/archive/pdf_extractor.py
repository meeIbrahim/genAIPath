from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader


class PdfExtractionError(Exception):
    pass


def extract_pdf_text(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
    except Exception as exc:
        raise PdfExtractionError(f"failed to extract text from {path}: {exc}") from exc

    text = "\n\n".join(page for page in pages if page.strip())
    if not text.strip():
        raise PdfExtractionError(f"no extractable text in {path}")
    return text
