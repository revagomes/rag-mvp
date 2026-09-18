"""Unit tests for document loading and chunking (rag_mvp.ingest)."""

from __future__ import annotations

import pytest

from rag_mvp.ingest import (
    SUPPORTED_EXTENSIONS,
    Chunk,
    chunk_text,
    discover_files,
    load_and_chunk_file,
    load_text,
)


class TestChunkText:
    def test_empty_text_returns_no_chunks(self):
        assert chunk_text("", source="s", chunk_size=100, chunk_overlap=10) == []

    def test_whitespace_only_returns_no_chunks(self):
        assert chunk_text("   \n\n  \t ", source="s", chunk_size=100, chunk_overlap=10) == []

    def test_short_text_is_a_single_chunk(self):
        chunks = chunk_text("hello world", source="s", chunk_size=100, chunk_overlap=10)
        assert len(chunks) == 1
        assert chunks[0].text == "hello world"
        assert chunks[0].chunk_index == 0
        assert chunks[0].source == "s"

    def test_long_text_splits_into_multiple_chunks(self):
        text = " ".join(f"word{i}" for i in range(200))
        chunks = chunk_text(text, source="doc", chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 1
        # Indices are sequential starting at 0.
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_chunks_respect_max_size(self):
        text = " ".join(f"word{i}" for i in range(200))
        chunk_size = 100
        chunks = chunk_text(text, source="doc", chunk_size=chunk_size, chunk_overlap=20)
        for c in chunks:
            assert len(c.text) <= chunk_size

    def test_consecutive_chunks_overlap(self):
        # Build text with distinct tokens so we can detect shared content.
        text = " ".join(f"token{i:03d}" for i in range(100))
        chunks = chunk_text(text, source="doc", chunk_size=120, chunk_overlap=40)
        assert len(chunks) >= 2
        # The tail of chunk N should share at least one token with chunk N+1.
        for a, b in zip(chunks, chunks[1:]):
            a_tokens = set(a.text.split())
            b_tokens = set(b.text.split())
            assert a_tokens & b_tokens, "expected overlap between consecutive chunks"

    def test_no_chunk_splits_a_word_when_possible(self):
        text = " ".join(f"alpha{i}" for i in range(80))
        chunks = chunk_text(text, source="doc", chunk_size=100, chunk_overlap=20)
        for c in chunks:
            # Every token should be a complete "alphaN" token, never truncated.
            for tok in c.text.split():
                assert tok.startswith("alpha")
                assert tok[len("alpha"):].isdigit()

    def test_overlap_equal_to_size_raises(self):
        with pytest.raises(ValueError):
            chunk_text("some text", source="s", chunk_size=50, chunk_overlap=50)

    def test_overlap_greater_than_size_raises(self):
        with pytest.raises(ValueError):
            chunk_text("some text", source="s", chunk_size=50, chunk_overlap=60)

    def test_ids_are_stable_for_same_input(self):
        a = chunk_text("repeatable content here", source="s", chunk_size=100, chunk_overlap=10)
        b = chunk_text("repeatable content here", source="s", chunk_size=100, chunk_overlap=10)
        assert [c.id for c in a] == [c.id for c in b]

    def test_ids_differ_across_sources(self):
        a = chunk_text("same content", source="fileA.txt", chunk_size=100, chunk_overlap=10)
        b = chunk_text("same content", source="fileB.txt", chunk_size=100, chunk_overlap=10)
        assert a[0].id != b[0].id

    def test_blank_lines_are_collapsed(self):
        text = "line one\n\n\n\n\nline two"
        chunks = chunk_text(text, source="s", chunk_size=1000, chunk_overlap=10)
        assert len(chunks) == 1
        assert "\n\n\n" not in chunks[0].text

    def test_hard_split_when_no_whitespace(self):
        # A single long "word" with no spaces must still be split by hard windows.
        text = "x" * 250
        chunks = chunk_text(text, source="s", chunk_size=100, chunk_overlap=20)
        assert len(chunks) >= 3
        assert all(len(c.text) <= 100 for c in chunks)


class TestLoadText:
    def test_load_txt(self, tmp_path):
        p = tmp_path / "a.txt"
        p.write_text("plain text content", encoding="utf-8")
        assert load_text(p) == "plain text content"

    def test_load_markdown(self, tmp_path):
        p = tmp_path / "a.md"
        p.write_text("# Heading\n\nBody", encoding="utf-8")
        assert "Heading" in load_text(p)

    def test_unsupported_extension_raises(self, tmp_path):
        p = tmp_path / "a.docx"
        p.write_text("x", encoding="utf-8")
        with pytest.raises(ValueError):
            load_text(p)

    def test_load_pdf(self, tmp_path):
        # Generate a minimal real PDF with pypdf (already a project dependency)
        # so the PDF path is exercised end-to-end, not mocked.
        pytest.importorskip("pypdf")
        from pypdf import PdfWriter

        pdf_path = tmp_path / "doc.pdf"
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        with pdf_path.open("wb") as fh:
            writer.write(fh)

        # A blank page yields little/no text, but load_text must run without
        # error and return a string.
        result = load_text(pdf_path)
        assert isinstance(result, str)

    def test_load_and_chunk_file(self, tmp_path):
        p = tmp_path / "doc.txt"
        p.write_text("some content to chunk", encoding="utf-8")
        chunks = load_and_chunk_file(p, chunk_size=100, chunk_overlap=10)
        assert len(chunks) == 1
        assert isinstance(chunks[0], Chunk)
        assert chunks[0].source == str(p)


class TestDiscoverFiles:
    def test_finds_supported_files_recursively(self, tmp_path):
        (tmp_path / "a.txt").write_text("a", encoding="utf-8")
        (tmp_path / "b.md").write_text("b", encoding="utf-8")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "c.markdown").write_text("c", encoding="utf-8")
        (tmp_path / "ignore.docx").write_text("x", encoding="utf-8")
        (tmp_path / "notes.log").write_text("x", encoding="utf-8")

        found = discover_files(tmp_path)
        names = {p.name for p in found}
        assert names == {"a.txt", "b.md", "c.markdown"}

    def test_results_are_sorted(self, tmp_path):
        for name in ["z.txt", "a.txt", "m.txt"]:
            (tmp_path / name).write_text("x", encoding="utf-8")
        found = discover_files(tmp_path)
        assert found == sorted(found)

    def test_empty_directory_returns_empty(self, tmp_path):
        assert discover_files(tmp_path) == []


def test_supported_extensions_include_expected():
    assert {".txt", ".md", ".markdown", ".pdf"} <= SUPPORTED_EXTENSIONS
