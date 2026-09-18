"""Unit tests for the embedding and LLM backend factories and their selection
and error-handling logic. These avoid any real model loading or network I/O."""

from __future__ import annotations

import pytest

from rag_mvp import embeddings, llm
from rag_mvp.config import Settings


class TestEmbedderFactory:
    def test_local_backend_returns_local_embedder(self):
        s = Settings(embedding_backend="local", embedding_model="dummy-model")
        emb = embeddings.build_embedder(s)
        assert isinstance(emb, embeddings.LocalEmbedder)
        # Constructing a LocalEmbedder must NOT load the model eagerly.
        assert emb._model is None

    def test_openai_backend_requires_api_key(self):
        s = Settings(embedding_backend="openai", openai_api_key=None)
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            embeddings.build_embedder(s)

    def test_unknown_backend_raises(self):
        s = Settings(embedding_backend="banana")
        with pytest.raises(ValueError, match="Unknown embedding backend"):
            embeddings.build_embedder(s)

    def test_backend_selection_is_case_insensitive(self):
        s = Settings(embedding_backend="LOCAL")
        assert isinstance(embeddings.build_embedder(s), embeddings.LocalEmbedder)

    def test_local_embedder_embed_query_delegates_to_documents(self, monkeypatch):
        emb = embeddings.LocalEmbedder("dummy")
        captured = {}

        def fake_embed_documents(texts):
            captured["texts"] = texts
            return [[0.1, 0.2, 0.3]]

        monkeypatch.setattr(emb, "embed_documents", fake_embed_documents)
        vec = emb.embed_query("hello")
        assert vec == [0.1, 0.2, 0.3]
        assert captured["texts"] == ["hello"]

    def test_local_embedder_empty_documents_short_circuits(self):
        # Must return [] without ever loading a model.
        emb = embeddings.LocalEmbedder("dummy")
        assert emb.embed_documents([]) == []
        assert emb._model is None


class TestLLMFactory:
    def test_none_backend_returns_none_llm(self):
        s = Settings(llm_backend="none")
        assert isinstance(llm.build_llm(s), llm.NoneLLM)

    def test_ollama_backend_returns_ollama_llm(self):
        s = Settings(llm_backend="ollama", ollama_url="http://x:1", ollama_model="m")
        obj = llm.build_llm(s)
        assert isinstance(obj, llm.OllamaLLM)

    def test_openai_backend_requires_api_key(self):
        s = Settings(llm_backend="openai", openai_api_key=None)
        with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
            llm.build_llm(s)

    def test_unknown_backend_raises(self):
        s = Settings(llm_backend="banana")
        with pytest.raises(ValueError, match="Unknown LLM backend"):
            llm.build_llm(s)

    def test_selection_is_case_insensitive(self):
        s = Settings(llm_backend="NONE")
        assert isinstance(llm.build_llm(s), llm.NoneLLM)


class TestNoneLLM:
    def test_generate_reports_no_answer(self):
        out = llm.NoneLLM().generate("q", "ctx")
        assert "none" in out.lower()
        assert "no answer" in out.lower()


class TestPromptBuilding:
    def test_user_prompt_contains_question_and_context(self):
        prompt = llm._build_user_prompt("What is X?", "Some context here")
        assert "What is X?" in prompt
        assert "Some context here" in prompt

    def test_system_prompt_instructs_grounding(self):
        assert "context" in llm.SYSTEM_PROMPT.lower()


class TestOllamaLLM:
    def test_generate_parses_response_field(self, monkeypatch):
        obj = llm.OllamaLLM("http://localhost:11434", "llama3.2")

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {"response": "  the answer  "}

        def fake_post(url, json, timeout):  # noqa: A002
            assert url.endswith("/api/generate")
            assert json["model"] == "llama3.2"
            assert json["stream"] is False
            return FakeResponse()

        monkeypatch.setattr(llm.httpx, "post", fake_post)
        assert obj.generate("q", "ctx") == "the answer"

    def test_generate_wraps_http_errors(self, monkeypatch):
        obj = llm.OllamaLLM("http://localhost:11434", "llama3.2")

        def fake_post(url, json, timeout):  # noqa: A002
            raise llm.httpx.ConnectError("refused")

        monkeypatch.setattr(llm.httpx, "post", fake_post)
        with pytest.raises(RuntimeError, match="Ollama request failed"):
            obj.generate("q", "ctx")

    def test_base_url_trailing_slash_is_stripped(self):
        obj = llm.OllamaLLM("http://localhost:11434/", "m")
        assert obj._base_url == "http://localhost:11434"
