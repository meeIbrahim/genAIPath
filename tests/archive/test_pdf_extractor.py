from pathlib import Path

import pytest

from app.archive.pdf_extractor import PdfExtractionError, extract_pdf_text


def test_extract_pdf_text_returns_nonempty_text_for_real_pdf():
    text = extract_pdf_text(Path("archive/LungPaper.pdf"))
    assert len(text.strip()) > 0


def test_extract_pdf_text_raises_on_corrupt_file(tmp_path):
    corrupt = tmp_path / "not_a_pdf.pdf"
    corrupt.write_bytes(b"this is not a valid pdf file at all")
    with pytest.raises(PdfExtractionError):
        extract_pdf_text(corrupt)
