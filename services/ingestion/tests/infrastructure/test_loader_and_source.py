"""PDF loader (pdfplumber + pypdf fallback) and local PDF directory source."""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import pypdf
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from conftest import REAL_PDF, make_pdf

from app.domain.errors import CorruptDocument, DocumentTooLarge, EmptyDocument, InvalidFileName
from app.domain.models import SourceFile
from app.infrastructure import pdf_loader
from app.infrastructure.pdf_loader import PdfPlumberLoader
from app.infrastructure.source import LocalDocumentSource
from common.errors import NotFound

SRC = SourceFile(file="x.pdf", doc_id="x")


# ---------- loader ----------


def test_real_pdf_pages_and_tables() -> None:
    doc = PdfPlumberLoader().load(REAL_PDF.read_bytes(), SourceFile(REAL_PDF.name, "kb"), "hash")
    assert [p.page for p in doc.pages] == [1, 2, 3, 4, 5]
    assert len(doc.pages[0].tables) == 1
    assert len(doc.pages[1].tables) == 1
    assert doc.pages[0].tables[0][0][0] == "Course"
    assert doc.warnings == []
    assert doc.content_hash == "hash"
    # Table text is NOT duplicated into page text (no garbled table rows).
    assert "Hamza Ali Rs. 30,000" not in doc.pages[0].text
    assert "1. About Genz Skool" in doc.pages[0].text


def test_synthetic_pdf_text_extracted() -> None:
    doc = PdfPlumberLoader().load(make_pdf([["Hello (world)", "Line two"], ["Page 2"]]), SRC, "h")
    assert doc.pages[0].text.splitlines() == ["Hello (world)", "Line two"]
    assert doc.pages[1].text == "Page 2"
    assert doc.doc_id == "x"
    assert doc.source_file == "x.pdf"


def test_blank_pages_load_with_no_text() -> None:
    doc = PdfPlumberLoader().load(make_pdf([[], []]), SRC, "h")
    assert [p.text for p in doc.pages] == ["", ""]


def test_zero_page_pdf_is_empty() -> None:
    with pytest.raises(EmptyDocument):
        PdfPlumberLoader().load(make_pdf([]), SRC, "h")


def test_empty_bytes() -> None:
    with pytest.raises(EmptyDocument, match="0 bytes"):
        PdfPlumberLoader().load(b"", SRC, "h")


@pytest.mark.parametrize("data", [b"hello world", b"\x89PNG\r\n\x1a\n" + b"0" * 100, b"<html>not a pdf</html>"])
def test_non_pdf_bytes(data: bytes) -> None:
    with pytest.raises(CorruptDocument, match="not a PDF"):
        PdfPlumberLoader().load(data, SRC, "h")


@pytest.mark.parametrize("cut", [20, 200, 400])
def test_truncated_pdf_is_corrupt_or_recovered(cut: int) -> None:
    data = make_pdf([["Some text here"]])[:cut]
    try:
        doc = PdfPlumberLoader().load(data, SRC, "h")
    except (CorruptDocument, EmptyDocument):
        return
    assert doc.pages  # a parser may recover a truncated file; it must never crash differently


def test_garbage_after_header_is_corrupt() -> None:
    with pytest.raises((CorruptDocument, EmptyDocument)):
        PdfPlumberLoader().load(b"%PDF-1.4\n" + os.urandom(2000), SRC, "h")


def test_encrypted_pdf_rejected() -> None:
    writer = pypdf.PdfWriter(clone_from=io.BytesIO(make_pdf([["secret text"]])))
    writer.encrypt(user_password="pw", owner_password="owner", algorithm="RC4-128")
    buf = io.BytesIO()
    writer.write(buf)
    with pytest.raises(CorruptDocument):
        PdfPlumberLoader().load(buf.getvalue(), SRC, "h")


def test_pypdf_fallback_when_pdfplumber_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*_: object, **__: object) -> None:
        raise RuntimeError("pdfplumber bug")

    monkeypatch.setattr(pdf_loader.pdfplumber, "open", broken)
    doc = PdfPlumberLoader().load(make_pdf([["Fallback text"]]), SRC, "h")
    assert "Fallback text" in doc.pages[0].text
    assert doc.pages[0].tables == []
    assert any("pypdf text fallback" in w for w in doc.warnings)


def test_both_parsers_failing_is_corrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(*_: object, **__: object) -> None:
        raise RuntimeError("bug")

    monkeypatch.setattr(pdf_loader.pdfplumber, "open", broken)
    monkeypatch.setattr(pdf_loader.pypdf, "PdfReader", broken)
    with pytest.raises(CorruptDocument, match="could not be parsed"):
        PdfPlumberLoader().load(make_pdf([["x"]]), SRC, "h")


def test_header_may_follow_junk_bytes() -> None:
    doc = PdfPlumberLoader().load(b"\n\n" + make_pdf([["ok"]]), SRC, "h")
    assert doc.pages[0].text == "ok"


# ---------- source ----------


@pytest.fixture
def src_dir(tmp_path: Path) -> Path:
    d = tmp_path / "src"
    d.mkdir()
    return d


def test_lists_only_visible_pdf_files_sorted(src_dir: Path) -> None:
    for name in ("b.pdf", "A.PDF", "notes.txt", ".hidden.pdf", "c.pdf.bak"):
        (src_dir / name).write_bytes(b"%PDF-")
    (src_dir / "dir.pdf").mkdir()
    files = LocalDocumentSource(src_dir).list_files()
    assert [f.file for f in files] == ["A.PDF", "b.pdf"]
    assert [f.doc_id for f in files] == ["a", "b"]


def test_doc_id_collisions_are_disambiguated(src_dir: Path) -> None:
    (src_dir / "my file.pdf").write_bytes(b"%PDF-")
    (src_dir / "my_file.pdf").write_bytes(b"%PDF-")
    ids = [f.doc_id for f in LocalDocumentSource(src_dir).list_files()]
    assert len(set(ids)) == 2
    assert ids[0] == "my_file"
    assert ids[1].startswith("my_file_")
    assert ids == [f.doc_id for f in LocalDocumentSource(src_dir).list_files()]  # deterministic


@pytest.mark.skipif(sys.platform == "win32", reason="symlinks need admin rights on Windows")
def test_symlinks_are_ignored(src_dir: Path, tmp_path: Path) -> None:
    outside = tmp_path / "secret.pdf"
    outside.write_bytes(b"%PDF-")
    (src_dir / "link.pdf").symlink_to(outside)
    assert LocalDocumentSource(src_dir).list_files() == []


def test_missing_directory_lists_nothing(tmp_path: Path) -> None:
    source = LocalDocumentSource(tmp_path / "missing")
    assert source.list_files() == []
    assert source.fingerprints() == []
    assert source.exists() is False


@pytest.mark.parametrize(
    "name",
    [
        "../etc/passwd.pdf",
        "..\\x.pdf",
        "sub/a.pdf",
        "C:a.pdf",
        "/abs.pdf",
        "a.txt",
        "",
        ".",
        "..",
        ".hidden.pdf",
        "a\x00.pdf",
    ],
)
def test_resolve_rejects_unsafe_names(src_dir: Path, name: str) -> None:
    with pytest.raises(InvalidFileName):
        LocalDocumentSource(src_dir).resolve(name)


def test_resolve_unknown_file(src_dir: Path) -> None:
    with pytest.raises(NotFound):
        LocalDocumentSource(src_dir).resolve("missing.pdf")


def test_resolve_and_read(src_dir: Path) -> None:
    (src_dir / "kb.pdf").write_bytes(b"%PDF-abc")
    source = LocalDocumentSource(src_dir)
    sf = source.resolve("kb.pdf")
    assert sf == SourceFile("kb.pdf", "kb")
    assert source.read(sf) == b"%PDF-abc"


def test_read_enforces_size_limit(src_dir: Path) -> None:
    (src_dir / "big.pdf").write_bytes(b"%PDF-" + b"0" * 2000)
    source = LocalDocumentSource(src_dir, max_bytes=1000)
    with pytest.raises(DocumentTooLarge):
        source.read(source.resolve("big.pdf"))


def test_fingerprints_change_on_modification(src_dir: Path) -> None:
    f = src_dir / "kb.pdf"
    f.write_bytes(b"%PDF-1")
    source = LocalDocumentSource(src_dir)
    before = source.fingerprints()
    f.write_bytes(b"%PDF-22")
    os.utime(f, ns=(before[0].mtime_ns + 10**9, before[0].mtime_ns + 10**9))
    after = source.fingerprints()
    assert before != after
    assert after[0].size == 7
    assert source.fingerprints() == after  # stable when nothing changes
