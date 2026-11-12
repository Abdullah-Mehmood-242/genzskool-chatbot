"""Text cleaner: Unicode normalisation, page-furniture removal, table-cell normalisation."""

from __future__ import annotations

import re
import unicodedata

from app.domain.models import CleanedDocument, Line, ParsedDocument, Table
from app.domain.ports import TextCleaner
from app.domain.text import collapse_ws

_INVISIBLE = dict.fromkeys(map(ord, "​‌‍﻿­"), None)


def normalize_text(text: str) -> str:
    """NFKC (ligatures, full-width), drop zero-width chars, unify dashes/quotes, collapse spaces."""
    text = unicodedata.normalize("NFKC", text).translate(_INVISIBLE)
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = text.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"[ \t ]+", " ", text).strip()


class StandardTextCleaner(TextCleaner):
    def __init__(self, strip_patterns: tuple[str, ...] = ()) -> None:
        self._strip = [re.compile(p) for p in strip_patterns]

    def _keep(self, line: str) -> bool:
        return bool(line) and not any(p.search(line) for p in self._strip)

    def clean(self, doc: ParsedDocument) -> CleanedDocument:
        lines: list[Line] = []
        tables: list[tuple[int, Table]] = []
        for page in doc.pages:
            for raw in page.text.splitlines():
                line = normalize_text(raw)
                if self._keep(line):
                    lines.append(Line(page=page.page, text=line))
            for table in page.tables:
                cleaned: Table = [
                    [collapse_ws(normalize_text(cell)) if cell is not None else None for cell in row] for row in table
                ]
                cleaned = [row for row in cleaned if any(c for c in row)]
                if cleaned:
                    tables.append((page.page, cleaned))
        return CleanedDocument(
            doc_id=doc.doc_id,
            source_file=doc.source_file,
            content_hash=doc.content_hash,
            lines=lines,
            tables=tables,
            warnings=list(doc.warnings),
        )
