"""Shared fixtures for the ingestion service tests."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.application.ingest import IngestionPolicy, IngestionService
from app.domain.chunking import StructureAwareChunker
from app.domain.cleaning import StandardTextCleaner
from app.domain.models import ChunkingConfig
from app.domain.ports import Clock, EmbeddingProvider, Vector
from app.infrastructure.adapters import InMemoryChunkArtifactStore, InMemoryRegistry
from app.infrastructure.pdf_loader import PdfPlumberLoader
from app.infrastructure.source import LocalDocumentSource
from app.infrastructure.vector_store import InMemoryVectorStore
from common.embeddings import FakeEmbedder
from common.resilience import RetryPolicy

REPO = Path(__file__).resolve().parents[3]
REAL_PDF = REPO / "data" / "pdfs" / "genzskool_knowledge_base.pdf"
CONFIG_DIR = REPO / "config"
FOOTER = r"^Genz Skool knowledge base - crawled .* Page \d+$"


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No test touches real Qdrant, the real data dir, or starts the watcher unless it opts in."""
    (tmp_path / "pdfs").mkdir()
    (tmp_path / "processed").mkdir()
    monkeypatch.setenv("CONFIG_DIR", str(CONFIG_DIR))
    monkeypatch.setenv("PDF_DIR", str(tmp_path / "pdfs"))
    monkeypatch.setenv("PROCESSED_DIR", str(tmp_path / "processed"))
    monkeypatch.setenv("VECTOR_STORE", "memory")
    monkeypatch.setenv("EMBEDDING_PROVIDER", "fake")
    monkeypatch.setenv("EMBEDDING_DIM", "64")
    monkeypatch.setenv("AUTO_INGEST_ON_STARTUP", "false")
    monkeypatch.setenv("AUTO_INGEST_INTERVAL_S", "0")


# ------------------------------------------------------------------------------------- PDFs


def _esc(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def make_pdf(pages: Sequence[Sequence[str]]) -> bytes:
    """Build a small valid PDF (Helvetica text, one line per entry). ``[[]]`` gives one blank page."""
    objects: list[bytes] = []
    page_ids = [4 + 2 * i for i in range(len(pages))]
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
    for i, lines in enumerate(pages):
        ops = "".join(f"({_esc(line)}) Tj T* " for line in lines)
        stream = f"BT /F1 10 Tf 14 TL 50 760 Td {ops}ET".encode("latin-1")
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> "
            f"/Contents {5 + 2 * i} 0 R >>".encode()
        )
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for n, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets).encode()
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


KB_LIKE_PAGES: list[list[str]] = [
    [
        "Test Academy - Knowledge Base",
        "1. About Test Academy",
        "Test Academy teaches practical skills online.",
        "2. Contact and location",
        "- Phone / WhatsApp: 0300-0000000",
        "- Email: hello@example.org",
        "3. Mentors",
        "Every mentor reviews assignments personally.",
        "- Ayesha Khan - Design mentor - teaches Graphic design.",
        "4. Enrollment",
        "- No payment is taken on the website.",
    ]
]


# ------------------------------------------------------------------------------------- fakes


class StepClock(Clock):
    """Deterministic clock: each call advances one second."""

    def __init__(self, start: datetime | None = None) -> None:
        self.t = start or datetime(2026, 11, 12, 9, 0, tzinfo=UTC)

    def now(self) -> datetime:
        self.t += timedelta(seconds=1)
        return self.t


class ScriptedEmbedder(EmbeddingProvider):
    """Fake embedder whose calls can be made to fail in a scripted way."""

    def __init__(self, dim: int = 32, failures: Sequence[Exception] = ()) -> None:
        self._dim = dim
        self._inner = FakeEmbedder(dim)
        self.failures = list(failures)
        self.calls: list[int] = []

    @property
    def dim(self) -> int:
        return self._dim

    async def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        self.calls.append(len(texts))
        if self.failures:
            raise self.failures.pop(0)
        return await self._inner.embed_documents(texts)


async def _no_sleep(_: float) -> None:
    return None


def build_service(
    pdf_dir: Path,
    *,
    embedder: EmbeddingProvider | None = None,
    store: InMemoryVectorStore | None = None,
    registry: InMemoryRegistry | None = None,
    artifacts: InMemoryChunkArtifactStore | None = None,
    policy: IngestionPolicy | None = None,
    chunking: ChunkingConfig | None = None,
) -> IngestionService:
    chunking = chunking or ChunkingConfig(strip_patterns=(FOOTER,))
    return IngestionService(
        source=LocalDocumentSource(pdf_dir),
        loader=PdfPlumberLoader(),
        cleaner=StandardTextCleaner(chunking.strip_patterns),
        chunker=StructureAwareChunker(chunking),
        embedder=embedder or ScriptedEmbedder(),
        store=store or InMemoryVectorStore(),
        registry=registry or InMemoryRegistry(),
        artifacts=artifacts or InMemoryChunkArtifactStore(),
        clock=StepClock(),
        policy=policy
        or IngestionPolicy(retry=RetryPolicy(attempts=3, base_delay_s=0, max_delay_s=0, retry_on=(Exception,))),
        sleep=_no_sleep,
    )


@pytest.fixture
def pdf_dir(tmp_path: Path) -> Path:
    return tmp_path / "pdfs"


@pytest.fixture
def real_pdf_dir(pdf_dir: Path) -> Path:
    (pdf_dir / REAL_PDF.name).write_bytes(REAL_PDF.read_bytes())
    return pdf_dir
