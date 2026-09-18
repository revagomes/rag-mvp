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
def client(fake_embedder, fake_store, fake_llm, tmp_path, monkeypatch):
    # Confine /ingest/path to a temp root the path-based tests write into.
    ingest_root = tmp_path / "ingest_root"
    ingest_root.mkdir()
    settings = Settings(
        chunk_size=100,
        chunk_overlap=20,
        top_k=3,
        llm_backend="none",
        ingest_root=str(ingest_root),
    )
    # The /ingest/path handler reads settings via get_settings(); point that at
    # our test settings so the containment root is the temp dir.
    monkeypatch.setattr(server, "get_settings", lambda: settings)
    pipeline = RagPipeline(
        settings, embedder=fake_embedder, store=fake_store, llm=fake_llm
    )
    server._pipeline = pipeline  # inject before lifespan/get_pipeline runs
    c = TestClient(server.app)
    c.ingest_root = ingest_root  # expose for tests that need an in-root path
    with c:
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

    def test_ingest_text_oversized_is_422(self, client):
        resp = client.post(
            "/ingest/text", json={"text": "x" * 1_000_001, "source": "big"}
        )
        assert resp.status_code == 422


class TestIngestPath:
    def test_ingest_path_missing_absolute_outside_root_is_403(self, client):
        # An absolute path outside the root is rejected before existence check.
        resp = client.post("/ingest/path", json={"path": "/no/such/path/here"})
        assert resp.status_code == 403

    def test_ingest_path_file(self, client):
        f = client.ingest_root / "doc.txt"
        f.write_text("file content to ingest", encoding="utf-8")
        resp = client.post("/ingest/path", json={"path": str(f)})
        assert resp.status_code == 200
        assert resp.json()["chunks_added"] >= 1

    def test_ingest_path_unsupported_type_is_400(self, client):
        f = client.ingest_root / "bad.docx"
        f.write_text("x", encoding="utf-8")
        resp = client.post("/ingest/path", json={"path": str(f)})
        assert resp.status_code == 400

    def test_ingest_path_outside_root_is_403(self, client):
        # Default ingest_root is "." (repo cwd); /etc/passwd is outside it.
        resp = client.post("/ingest/path", json={"path": "/etc/passwd"})
        assert resp.status_code == 403

    def test_ingest_path_dotdot_traversal_is_403(self, client):
        resp = client.post(
            "/ingest/path", json={"path": "../../../../etc/passwd"}
        )
        assert resp.status_code == 403

    def test_ingest_path_404_does_not_echo_path(self, client):
        # A missing path inside the root should 404 without reflecting the input.
        resp = client.post("/ingest/path", json={"path": "no_such_file_here.txt"})
        assert resp.status_code == 404
        assert "no_such_file_here" not in resp.json()["detail"]


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

    def test_query_oversized_question_is_422(self, client):
        resp = client.post("/query", json={"question": "x" * 4_001})
        assert resp.status_code == 422

    def test_query_maps_backend_runtime_error_to_502(
        self, client, monkeypatch
    ):
        client.post("/ingest/text", json={"text": "content", "source": "s"})

        def boom(question, top_k=None):  # noqa: ARG001
            raise RuntimeError("Ollama request failed (connection refused) at http://localhost:11434")

        monkeypatch.setattr(server.get_pipeline(), "query", boom)
        resp = client.post("/query", json={"question": "x"})
        assert resp.status_code == 502
        detail = resp.json()["detail"]
        # Generic message; internal details (URL, backend name) must not leak.
        assert detail == "The language model backend is unavailable."
        assert "Ollama" not in detail
        assert "11434" not in detail


class TestReset:
    def test_reset_clears_store(self, client):
        client.post("/ingest/text", json={"text": "a b c", "source": "s"})
        assert client.get("/health").json()["documents_in_store"] > 0
        resp = client.post("/reset")
        assert resp.status_code == 200
        assert resp.json()["status"] == "cleared"
        assert client.get("/health").json()["documents_in_store"] == 0


class TestSecretNonLeakage:
    """Lock in that the OpenAI API key is never reflected to clients."""

    _FAKE_KEY = "sk-test-SECRET-must-not-leak-123456"

    @pytest.fixture
    def client_with_key(self, fake_embedder, fake_store, fake_llm, monkeypatch):
        settings = Settings(
            chunk_size=100,
            chunk_overlap=20,
            top_k=3,
            llm_backend="none",
            openai_api_key=self._FAKE_KEY,
        )
        monkeypatch.setattr(server, "get_settings", lambda: settings)
        pipeline = RagPipeline(
            settings, embedder=fake_embedder, store=fake_store, llm=fake_llm
        )
        server._pipeline = pipeline
        with TestClient(server.app) as c:
            yield c
        server._pipeline = None

    def test_health_does_not_expose_key(self, client_with_key):
        body = client_with_key.get("/health").text
        assert self._FAKE_KEY not in body

    def test_query_response_does_not_expose_key(self, client_with_key):
        client_with_key.post("/ingest/text", json={"text": "content", "source": "s"})
        body = client_with_key.post("/query", json={"question": "content"}).text
        assert self._FAKE_KEY not in body



class TestAuth:
    """Bearer API-key authentication on protected endpoints."""

    _KEYS = "key-one,key-two"

    @pytest.fixture
    def keyed_client(self, fake_embedder, fake_store, fake_llm, tmp_path, monkeypatch):
        ingest_root = tmp_path / "root"
        ingest_root.mkdir()
        settings = Settings(
            chunk_size=100,
            chunk_overlap=20,
            top_k=3,
            llm_backend="none",
            ingest_root=str(ingest_root),
            api_keys=self._KEYS,
        )
        monkeypatch.setattr(server, "get_settings", lambda: settings)
        pipeline = RagPipeline(
            settings, embedder=fake_embedder, store=fake_store, llm=fake_llm
        )
        server._pipeline = pipeline
        with TestClient(server.app) as c:
            yield c
        server._pipeline = None

    def test_settings_auth_enabled(self):
        assert Settings(api_keys="k1,k2").auth_enabled is True
        assert Settings(api_keys="").auth_enabled is False
        assert Settings(api_keys="  ,  ").auth_enabled is False

    def test_settings_parsed_api_keys_trims_and_dedupes(self):
        assert Settings(api_keys=" a , b ,a ").parsed_api_keys() == {"a", "b"}

    # --- auth disabled (default client fixture has no keys) ---

    def test_query_open_when_auth_disabled(self, client):
        resp = client.post("/query", json={"question": "anything"})
        assert resp.status_code == 200

    # --- auth enabled ---

    def test_query_without_header_is_401(self, keyed_client):
        resp = keyed_client.post("/query", json={"question": "x"})
        assert resp.status_code == 401
        assert resp.headers.get("WWW-Authenticate") == "Bearer"

    def test_query_with_wrong_key_is_403(self, keyed_client):
        resp = keyed_client.post(
            "/query",
            json={"question": "x"},
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status_code == 403

    def test_query_with_valid_key_is_200(self, keyed_client):
        resp = keyed_client.post(
            "/query",
            json={"question": "x"},
            headers={"Authorization": "Bearer key-one"},
        )
        assert resp.status_code == 200

    def test_second_valid_key_also_works(self, keyed_client):
        resp = keyed_client.post(
            "/query",
            json={"question": "x"},
            headers={"Authorization": "Bearer key-two"},
        )
        assert resp.status_code == 200

    def test_malformed_scheme_is_401(self, keyed_client):
        resp = keyed_client.post(
            "/query",
            json={"question": "x"},
            headers={"Authorization": "Basic key-one"},
        )
        assert resp.status_code == 401

    def test_all_mutating_endpoints_protected(self, keyed_client):
        # Each protected endpoint rejects an unauthenticated request.
        assert keyed_client.post(
            "/ingest/text", json={"text": "x", "source": "s"}
        ).status_code == 401
        assert keyed_client.post(
            "/ingest/path", json={"path": "x.txt"}
        ).status_code == 401
        assert keyed_client.post("/reset").status_code == 401

    def test_ingest_and_reset_work_with_valid_key(self, keyed_client):
        h = {"Authorization": "Bearer key-one"}
        assert keyed_client.post(
            "/ingest/text", json={"text": "hello world", "source": "s"}, headers=h
        ).status_code == 200
        assert keyed_client.post("/reset", headers=h).status_code == 200

    def test_health_open_even_when_auth_enabled(self, keyed_client):
        # /health must stay reachable without a key (liveness probes).
        assert keyed_client.get("/health").status_code == 200
