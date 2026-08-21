from pathlib import Path

from app.archive.scanner import scan_archive


def test_scan_archive_returns_empty_list_for_missing_directory(tmp_path):
    assert scan_archive(tmp_path / "does-not-exist") == []


def test_scan_archive_lists_pdfs_with_stable_hash(tmp_path):
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    (archive_dir / "a.pdf").write_bytes(b"pdf-content-a")
    (archive_dir / "b.pdf").write_bytes(b"pdf-content-b")
    (archive_dir / "notes.txt").write_bytes(b"ignore me")

    first_scan = scan_archive(archive_dir)
    second_scan = scan_archive(archive_dir)

    assert {doc.filename for doc in first_scan} == {"a.pdf", "b.pdf"}
    assert {doc.doc_id_hash for doc in first_scan} == {doc.doc_id_hash for doc in second_scan}
    assert len({doc.doc_id_hash for doc in first_scan}) == 2


def test_scan_archive_matches_pdf_extension_case_insensitively(tmp_path):
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    (archive_dir / "lower.pdf").write_bytes(b"pdf-content-lower")
    (archive_dir / "UPPER.PDF").write_bytes(b"pdf-content-upper")
    (archive_dir / "Mixed.Pdf").write_bytes(b"pdf-content-mixed")
    (archive_dir / "notes.txt").write_bytes(b"ignore me")

    docs = scan_archive(archive_dir)

    assert {doc.filename for doc in docs} == {"lower.pdf", "UPPER.PDF", "Mixed.Pdf"}


def test_scan_archive_hash_changes_when_file_content_changes(tmp_path):
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    path = archive_dir / "a.pdf"
    path.write_bytes(b"version one")
    first_hash = scan_archive(archive_dir)[0].doc_id_hash
    path.write_bytes(b"version two")
    second_hash = scan_archive(archive_dir)[0].doc_id_hash
    assert first_hash != second_hash
