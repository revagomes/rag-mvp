"""Tests for RagPipeline orchestration using injected fakes (offline)."""

from __future__ import annotations

from pathlib import Path

from rag_mvp.config import Settings
from rag_mvp.pipeline import RagPipeline


def _pipeline(fake_embedder, fake_store, fake_llm, **overrides):
    settings = Settings(chunk_size=100, chunk_overlap=20, top_k=3, **overrides)
    return RagPipeline(
        settings, embedder=fake_embedder, store=fake_store, llm=fake_llm
    )


class TestIngest:
    def test_ingest_text_adds_chunks(self, fake_embedder, fake_store, fake_llm):
        p = _pipeline(fake_embedder, fake_store, fake_llm)
        result = p.ingest_text("hello world content", source="inline")
        assert result.chunks_added >= 1
        assert result.files_processed == 1
        assert result.sources == ["inline"]
        assert fake_store.count() == result.chunks_added

    def test_ingest_empty_text_adds_nothing(self, fake_embedder, fake_store, fake_llm):
        p = _pipeline(fake_embedder, fake_store, fake_llm)
        result = p.ingest_text("   ", source="inline")
        assert result.chunks_added == 0
        assert result.files_processed == 0
        assert result.sources == []
        assert fake_store.count() == 0

    def test_ingest_file(self, tmp_path, fake_embedder, fake_store, fake_llm):
        f = tmp_path / "doc.txt"
        f.write_text("some content here", encoding="utf-8")
        p = _pipeline(fake_embedder, fake_store, fake_llm)
        result = p.ingest_file(f)
        assert result.chunks_added >= 1
        assert result.sources == [str(f)]

    def test_ingest_directory_processes_all_files(
        self, tmp_path, fake_embedder, fake_store, fake_llm
    ):
        (tmp_path / "a.txt").write_text("alpha content", encoding="utf-8")
        (tmp_path / "b.md").write_text("beta content", encoding="utf-8")
        p = _pipeline(fake_embedder, fake_store, fake_llm)
        result = p.ingest_directory(tmp_path)
        assert result.files_processed == 2
        assert result.chunks_added >= 2

    def test_ingest_embeds_the_chunk_text(self, fake_embedder, fake_store, fake_llm):
        p = _pipeline(fake_embedder, fake_store, fake_llm)
        p.ingest_text("hello world", source="inline")
        # The embedder was asked to embed the documents.
        assert fake_embedder.embed_documents_calls
        assert "hello world" in fake_embedder.embed_documents_calls[0][0]


class TestQuery:
    def test_query_empty_store_returns_message(self, fake_embedder, fake_store, fake_llm):
        p = _pipeline(fake_embedder, fake_store, fake_llm)
        result = p.query("anything")
        assert result.chunks == []
        assert "no documents" in result.answer.lower()
        # The LLM must not be invoked when there is nothing to ground on.
        assert fake_llm.calls == []

    def test_query_retrieves_and_calls_llm(self, fake_embedder, fake_store, fake_llm):
        p = _pipeline(fake_embedder, fake_store, fake_llm)
        p.ingest_text("the capital of France is Paris", source="geo")
        result = p.query("capital of France")
        assert result.chunks, "expected at least one retrieved chunk"
        assert result.answer == "ANSWER[capital of France]"
        # The LLM was given a context string containing the source text.
        assert fake_llm.calls
        _q, context = fake_llm.calls[0]
        assert "Paris" in context

    def test_query_respects_explicit_top_k(self, fake_embedder, fake_store, fake_llm):
        p = _pipeline(fake_embedder, fake_store, fake_llm)
        for i in range(6):
            p.ingest_text(f"document number {i}", source=f"d{i}")
        result = p.query("document", top_k=2)
        assert len(result.chunks) == 2

    def test_query_defaults_to_settings_top_k(self, fake_embedder, fake_store, fake_llm):
        p = _pipeline(fake_embedder, fake_store, fake_llm)  # top_k=3
        for i in range(6):
            p.ingest_text(f"document number {i}", source=f"d{i}")
        result = p.query("document")
        assert len(result.chunks) == 3

    def test_context_includes_source_filename(self, fake_embedder, fake_store, fake_llm):
        p = _pipeline(fake_embedder, fake_store, fake_llm)
        p.ingest_text("grounded fact", source="/some/path/notes.txt")
        p.query("fact")
        _q, context = fake_llm.calls[0]
        assert "notes.txt" in context

    def test_query_result_reports_llm_backend(self, fake_embedder, fake_store, fake_llm):
        p = _pipeline(fake_embedder, fake_store, fake_llm, llm_backend="ollama")
        p.ingest_text("content", source="s")
        result = p.query("content")
        assert result.llm_backend == "ollama"


class TestMaintenance:
    def test_document_count(self, fake_embedder, fake_store, fake_llm):
        p = _pipeline(fake_embedder, fake_store, fake_llm)
        assert p.document_count() == 0
        p.ingest_text("x y z", source="s")
        assert p.document_count() == fake_store.count()

    def test_reset_delegates_to_store(self, fake_embedder, fake_store, fake_llm):
        p = _pipeline(fake_embedder, fake_store, fake_llm)
        p.ingest_text("content", source="s")
        p.reset()
        assert fake_store.reset_called
        assert p.document_count() == 0
