"""Local PDF directory adapter. Only plain *.pdf files directly inside PDF_DIR are ever read."""

from __future__ import annotations

import logging
from pathlib import Path

from app.domain.errors import DocumentTooLarge, InvalidFileName
from app.domain.models import SourceFile, SourceFingerprint
from app.domain.ports import DocumentSource
from app.domain.text import doc_id_from_file, sha256_hex
from common.errors import NotFound

logger = logging.getLogger(__name__)


class LocalDocumentSource(DocumentSource):
    def __init__(self, pdf_dir: Path, *, max_bytes: int = 20 * 1024 * 1024) -> None:
        self.pdf_dir = pdf_dir
        self.max_bytes = max_bytes

    def exists(self) -> bool:
        return self.pdf_dir.is_dir()

    def _paths(self) -> list[Path]:
        if not self.pdf_dir.is_dir():
            return []
        return sorted(
            (
                p
                for p in self.pdf_dir.iterdir()
                # symlinks are skipped: they could point outside PDF_DIR
                if p.suffix.lower() == ".pdf" and not p.name.startswith(".") and not p.is_symlink() and p.is_file()
            ),
            key=lambda p: p.name,
        )

    def list_files(self) -> list[SourceFile]:
        files: list[SourceFile] = []
        used: set[str] = set()
        for path in self._paths():
            doc_id = doc_id_from_file(path.name)
            if doc_id in used:  # 'a b.pdf' and 'a_b.pdf' would collide: disambiguate deterministically
                doc_id = f"{doc_id}_{sha256_hex(path.name)[:8]}"
                logger.warning("doc_id_collision", extra={"file": path.name, "doc_id": doc_id})
            used.add(doc_id)
            files.append(SourceFile(file=path.name, doc_id=doc_id))
        return files

    def resolve(self, file: str) -> SourceFile:
        if (
            not file
            or any(ch in file for ch in ("/", "\\", "\x00", ":"))
            or file in (".", "..")
            or file.startswith(".")
            or not file.lower().endswith(".pdf")
        ):
            raise InvalidFileName(details={"file": file[:100]})
        for source in self.list_files():
            if source.file == file:
                return source
        raise NotFound(f"No PDF named '{file}' in the PDF directory.")

    def read(self, source: SourceFile) -> bytes:
        path = self.pdf_dir / source.file
        size = path.stat().st_size
        if size > self.max_bytes:
            raise DocumentTooLarge(f"{source.file} is {size} bytes; the limit is {self.max_bytes}.")
        return path.read_bytes()

    def fingerprints(self) -> list[SourceFingerprint]:
        out: list[SourceFingerprint] = []
        for path in self._paths():
            try:
                stat = path.stat()
            except FileNotFoundError:  # deleted between listing and stat
                continue
            out.append(SourceFingerprint(file=path.name, size=stat.st_size, mtime_ns=stat.st_mtime_ns))
        return out
