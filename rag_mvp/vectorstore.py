"""Persistent vector store backed by ChromaDB.

We pass embeddings in explicitly (rather than using Chroma's built-in embedding
functions) so the embedding backend is controlled entirely by ``embeddings.py``.
Distances are cosine; we convert them to a 0..1 similarity score for readability.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from .ingest import Chunk


@dataclass
class RetrievedChunk:
    """A chunk returned from a similarity search."""

    id: str
    text: str
    source: str
    chunk_index: int
    score: float  # higher is more similar (1.0 == identical)


class VectorStore:
    """Thin wrapper around a persistent Chroma collection."""

    def __init__(self, storage_dir: str, collection_name: str) -> None:
        Path(storage_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=storage_dir,
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Upsert chunks with their precomputed embeddings."""
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings length mismatch")
        self._collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {"source": c.source, "chunk_index": c.chunk_index} for c in chunks
            ],
        )

    def query(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        """Return the ``top_k`` most similar chunks."""
        if self.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        retrieved: list[RetrievedChunk] = []
        for _id, doc, meta, dist in zip(ids, documents, metadatas, distances):
            retrieved.append(
                RetrievedChunk(
                    id=_id,
                    text=doc,
                    source=str(meta.get("source", "")),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    score=_distance_to_score(dist),
                )
            )
        return retrieved

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        """Delete all documents in the collection."""
        name = self._collection.name
        metadata = self._collection.metadata
        self._client.delete_collection(name)
        self._collection = self._client.get_or_create_collection(
            name=name, metadata=metadata
        )


def _distance_to_score(distance: float) -> float:
    """Convert cosine distance (0..2) to a similarity score (1..-1 -> clamp 0..1)."""
    score = 1.0 - float(distance)
    return max(0.0, min(1.0, score))
