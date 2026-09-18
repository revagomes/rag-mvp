"""HTTP API tests. The server's process-wide pipeline is replaced with one
built from offline fakes, so these exercise routing, schemas, and error mapping
without loading models or touching the network."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from rag_mvp import server
from rag_mvp.config import Settings
from rag_mvp.pipeline import RagPipeline


@pytest.fixture
def client(fake_embedder, fake_store, fake_llm):
    settings = Settings(chunk_size=100, chunk_overlap=20, top_k=3, llm_backend="none")
    pipeline = RagPipeline(
        settings, embedder=fake_embedder, store=fake_store, llm=fake_llm
    )
    server._pipeline = pipeline  # inject before lifespan/get_pipeline runs
    with TestClient(server.app) as c:
        yield c
    server._pipeline = None  # reset global for isolation


class TestHealth:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["embedding_backend"] == "local"
        assert body["llm_backend"] == "none"
        assert body["documents_in_store"] == 0


class TestIngestText:
    def test_ingest_text(self, client):
        resp = client.post(
            "/ingest/text", json={"text": "hello world content", "source": "inline"}
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["chunks_added"] >= 1
        assert body["total_chunks_in_store"] == body["chunks_added"]

    def test_ingest_text_missing_field_is_422(self, client):
        resp = client.post("/ingest/text", json={"source": "x"})
        assert resp.status_code == 422


class TestIngestPath:
    def test_ingest_path_missing_returns_404(self, client):
        resp = client.post("/ingest/path", json={"path": "/no/such/path/here"})
        assert resp.status_code == 404

    def test_ingest_path_file(self, client, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("file content to ingest", encoding="utf-8")
        resp = client.post("/ingest/path", json={"path": str(f)})
        assert resp.status_code == 200
        assert resp.json()["chunks_added"] >= 1

    def test_ingest_path_unsupported_type_is_400(self, client, tmp_path):
        f = tmp_path / "bad.docx"
        f.write_text("x", encoding="utf-8")
        resp = client.post("/ingest/path", json={"path": str(f)})
        assert resp.status_code == 400


class TestQuery:
    def test_query_empty_store(self, client):
        resp = client.post("/query", json={"question": "anything"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["chunks"] == []
        assert "no documents" in body["answer"].lower()

    def test_query_after_ingest(self, client):
        client.post(
            "/ingest/text",
            json={"text": "the capital of France is Paris", "source": "geo"},
        )
        resp = client.post("/query", json={"question": "capital of France", "top_k": 2})
        assert resp.status_code == 200
        body = resp.json()
        assert body["question"] == "capital of France"
        assert body["chunks"]
        # scores are rounded floats in the response model
        assert all(0.0 <= c["score"] <= 1.0 for c in body["chunks"])

    def test_query_top_k_validation(self, client):
        # top_k must be >= 1
        resp = client.post("/query", json={"question": "x", "top_k": 0})
        assert resp.status_code == 422

    def test_query_maps_backend_runtime_error_to_502(
        self, client, monkeypatch
    ):
        client.post("/ingest/text", json={"text": "content", "source": "s"})

        def boom(question, top_k=None):  # noqa: ARG001
            raise RuntimeError("Ollama request failed (connection refused)")

        monkeypatch.setattr(server.get_pipeline(), "query", boom)
        resp = client.post("/query", json={"question": "x"})
        assert resp.status_code == 502
        assert "Ollama" in resp.json()["detail"]


class TestReset:
    def test_reset_clears_store(self, client):
        client.post("/ingest/text", json={"text": "a b c", "source": "s"})
        assert client.get("/health").json()["documents_in_store"] > 0
        resp = client.post("/reset")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cleared"
        assert client.get("/health").json()["documents_in_store"] == 0
