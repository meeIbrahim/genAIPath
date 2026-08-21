from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel


class ArchiveDoc(BaseModel):
    doc_id_hash: str
    path: str
    filename: str


def scan_archive(archive_dir: Path = Path("archive")) -> list[ArchiveDoc]:
    if not archive_dir.is_dir():
        return []
    docs = []
    pdf_paths = (p for p in archive_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")
    for path in sorted(pdf_paths):
        doc_id_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        docs.append(ArchiveDoc(doc_id_hash=doc_id_hash, path=str(path), filename=path.name))
    return docs
