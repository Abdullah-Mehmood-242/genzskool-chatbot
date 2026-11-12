"""PDF loader adapter: pdfplumber (text outside tables + tables), pypdf text-only fallback."""

from __future__ import annotations

import io
import logging
from typing import Any

import pdfplumber
import pypdf

from app.domain.errors import CorruptDocument, EmptyDocument
from app.domain.models import PageContent, ParsedDocument, SourceFile, Table
from app.domain.ports import DocumentLoader

logger = logging.getLogger(__name__)


def _outside(bboxes: list[tuple[float, float, float, float]]) -> Any:
    def keep(obj: dict[str, Any]) -> bool:
        if obj.get("object_type") != "char":
            return True
        cx = (obj["x0"] + obj["x1"]) / 2
        cy = (obj["top"] + obj["bottom"]) / 2
        return not any(x0 <= cx <= x1 and top <= cy <= bottom for x0, top, x1, bottom in bboxes)

    return keep


class PdfPlumberLoader(DocumentLoader):
    def load(self, data: bytes, source: SourceFile, content_hash: str) -> ParsedDocument:
        if not data:
            raise EmptyDocument(f"{source.file} is empty (0 bytes).")
        if b"%PDF-" not in data[:1024]:
            raise CorruptDocument(f"{source.file} is not a PDF (missing %PDF header).")
        warnings: list[str] = []
        try:
            pages = self._with_pdfplumber(data)
        except Exception as exc:
            logger.warning("pdfplumber_failed", extra={"file": source.file, "error_type": type(exc).__name__})
            warnings.append(f"pdfplumber failed ({type(exc).__name__}); used pypdf text fallback without tables.")
            pages = self._with_pypdf(data, source)
        if not pages:
            raise EmptyDocument(f"{source.file} has no pages.")
        return ParsedDocument(
            doc_id=source.doc_id,
            source_file=source.file,
            content_hash=content_hash,
            pages=pages,
            warnings=warnings,
        )

    @staticmethod
    def _with_pdfplumber(data: bytes) -> list[PageContent]:
        pages: list[PageContent] = []
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for number, page in enumerate(pdf.pages, start=1):
                found = page.find_tables()
                tables: list[Table] = [t.extract() for t in found]
                text = page.filter(_outside([t.bbox for t in found])).extract_text() or ""
                pages.append(PageContent(page=number, text=text, tables=[t for t in tables if t]))
        return pages

    @staticmethod
    def _with_pypdf(data: bytes, source: SourceFile) -> list[PageContent]:
        try:
            reader = pypdf.PdfReader(io.BytesIO(data))
            if reader.is_encrypted:
                raise CorruptDocument(f"{source.file} is encrypted.")
            return [PageContent(page=i, text=p.extract_text() or "") for i, p in enumerate(reader.pages, start=1)]
        except CorruptDocument:
            raise
        except Exception as exc:
            raise CorruptDocument(f"{source.file} could not be parsed ({type(exc).__name__}).") from None
