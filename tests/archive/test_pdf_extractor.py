from pathlib import Path

import pytest

from app.archive.pdf_extractor import PdfExtractionError, extract_pdf_text


def _write_pdf_with_text(path: Path, text: str) -> None:
    """Write a minimal, valid single-page PDF whose only content is `text`.

    Hand-authored rather than committed as a binary fixture so the happy-path
    test depends on nothing outside the test run (the repo's archive/ directory
    is gitignored and untracked).
    """
    content = f"BT /F1 24 Tf 20 100 Td ({text}) Tj ET\n".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200] "
        b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"endstream",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += str(number).encode() + b" 0 obj\n" + body + b"\nendobj\n"

    xref_offset = len(out)
    out += b"xref\n0 " + str(len(objects) + 1).encode() + b"\n"
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += b"trailer\n<< /Size " + str(len(objects) + 1).encode() + b" /Root 1 0 R >>\n"
    out += b"startxref\n" + str(xref_offset).encode() + b"\n%%EOF\n"

    path.write_bytes(bytes(out))


def test_extract_pdf_text_returns_page_text_for_real_pdf(tmp_path):
    pdf_path = tmp_path / "generated.pdf"
    _write_pdf_with_text(pdf_path, "Lung nodule classification results")

    text = extract_pdf_text(pdf_path)

    assert "Lung nodule classification results" in text


def test_extract_pdf_text_raises_on_corrupt_file(tmp_path):
    corrupt = tmp_path / "not_a_pdf.pdf"
    corrupt.write_bytes(b"this is not a valid pdf file at all")
    with pytest.raises(PdfExtractionError):
        extract_pdf_text(corrupt)
