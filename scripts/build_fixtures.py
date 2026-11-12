#!/usr/bin/env python
"""Regenerate tests/fixtures/kb/ from the real PDF using the ingestion service's own pipeline.

The fixture is what retrieval (and later advisor/evaluation) tests search over, so it must be
exactly what ingestion produces. services/ingestion/tests/test_fixture_current.py fails when the
fixture is stale. Run via:  python scripts/dev.py fixtures
(requires PYTHONPATH=libs:services/ingestion, which dev.py sets).
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from app.api.dependencies import ChunkingFile
from app.domain.chunking import StructureAwareChunker
from app.domain.cleaning import StandardTextCleaner
from app.domain.models import SourceFile
from app.domain.text import doc_id_from_file, sha256_hex
from app.infrastructure.pdf_loader import PdfPlumberLoader
from common.config import load_yaml_model
from common.contracts import Chunk

ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "data" / "pdfs" / "genzskool_knowledge_base.pdf"
OUT = ROOT / "tests" / "fixtures" / "kb"
INGESTED_AT = datetime(2026, 11, 12, tzinfo=UTC)
VERSION = 1


def build_chunks() -> list[Chunk]:
    cfg = load_yaml_model(ROOT / "config" / "chunking.yaml", ChunkingFile).to_config()
    data = PDF.read_bytes()
    source = SourceFile(file=PDF.name, doc_id=doc_id_from_file(PDF.name))
    doc = PdfPlumberLoader().load(data, source, sha256_hex(data))
    cleaned = StandardTextCleaner(cfg.strip_patterns).clean(doc)
    result = StructureAwareChunker(cfg).chunk(cleaned, index_version=VERSION, ingested_at=INGESTED_AT)
    if result.warnings:
        raise SystemExit(f"chunker warnings, refusing to write fixture: {result.warnings}")
    return result.chunks


def render(chunks: list[Chunk]) -> dict[str, str]:
    doc_id = chunks[0].metadata.doc_id
    registry = {
        "schema": 1,
        "documents": {
            doc_id: {
                "source_file": chunks[0].metadata.source_file,
                "content_hash": sha256_hex(PDF.read_bytes()),
                "active_index_version": VERSION,
                "chunks": len(chunks),
                "ingested_at": INGESTED_AT.isoformat(),
            }
        },
    }
    return {
        "registry.json": json.dumps(registry, indent=2) + "\n",
        f"{doc_id}.v{VERSION}.chunks.jsonl": "".join(c.model_dump_json() + "\n" for c in chunks),
    }


def main() -> int:
    files = render(build_chunks())
    OUT.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        (OUT / name).write_text(content, encoding="utf-8", newline="\n")
        print(f"wrote {(OUT / name).relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
