"""Shared test fakes: deterministic, offline stand-ins for the embedder, vector
store, and LLM so the pipeline and API can be tested without models or network."""

from __future__ import annotations

import pytest

from rag_mvp.ingest import Chunk
from rag_mvp.vectorstore import RetrievedChunk


class FakeEmbedder:
    """Deterministic embedder: maps text to a tiny bag-of-chars vector.

    The exact geometry does not matter for pipeline tests; we only need it to be
    deterministic and to produce distinct vectors for distinct text.
    """

    def __init__(self) -> None:
        self.embed_documents_calls: list[list[str]] = []
        self.embed_query_calls: list[str] = []

    def _vec(self, text: str) -> list[float]:
        # 4-dim vector from character-class counts; stable and cheap.
        letters = sum(c.isalpha() for c in text)
        digits = sum(c.isdigit() for c in text)
        spaces = sum(c.isspace() for c in text)
        return [float(len(text)), float(letters), float(digits), float(spaces)]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embed_documents_calls.append(list(texts))
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.embed_query_calls.append(text)
        return self._vec(text)


class FakeStore:
    """In-memory stand-in for VectorStore with the same interface."""

    def __init__(self) -> None:
        self._items: dict[str, tuple[Chunk, list[float]]] = {}
        self.reset_called = False

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("length mismatch")
        for chunk, emb in zip(chunks, embeddings):
            self._items[chunk.id] = (chunk, emb)

    def query(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        # Rank by negative L2 distance so nearest is first; score is a simple
        # inverse-distance mapping into (0, 1].
        scored = []
        for chunk, emb in self._items.values():
            dist = sum((a - b) ** 2 for a, b in zip(query_embedding, emb)) ** 0.5
            score = 1.0 / (1.0 + dist)
            scored.append((score, chunk))
        scored.sort(key=lambda t: t[0], reverse=True)
        return [
            RetrievedChunk(
                id=c.id, text=c.text, source=c.source,
                chunk_index=c.chunk_index, score=s,
            )
            for s, c in scored[:top_k]
        ]

    def count(self) -> int:
        return len(self._items)

    def reset(self) -> None:
        self._items.clear()
        self.reset_called = True


class FakeLLM:
    """LLM that echoes what it received, so tests can assert wiring."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def generate(self, question: str, context: str) -> str:
        self.calls.append((question, context))
        return f"ANSWER[{question}]"


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def fake_store() -> FakeStore:
    return FakeStore()


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()
