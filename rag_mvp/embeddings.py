"""Embedding backends.

Two interchangeable backends are provided:

* ``LocalEmbedder`` — uses ``sentence-transformers`` and runs fully offline
  after the model is downloaded once. This is the default and requires no API key.
* ``OpenAIEmbedder`` — uses the OpenAI embeddings API (requires ``OPENAI_API_KEY``).

Both expose ``embed_documents`` and ``embed_query`` returning lists of float vectors.
"""

from __future__ import annotations

from typing import Protocol

from .config import Settings


class Embedder(Protocol):
    """Common interface for embedding backends."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class LocalEmbedder:
    """Local sentence-transformers embedder. Lazily loads the model."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            # Imported lazily so the package can be imported without the heavy
            # dependency being loaded until embeddings are actually needed.
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._ensure_model()
        vectors = model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        )
        return [v.tolist() for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class OpenAIEmbedder:
    """OpenAI API embedder."""

    def __init__(self, model_name: str, api_key: str | None) -> None:
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required for the OpenAI embedding backend."
            )
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(
            model=self._model_name, input=texts
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def build_embedder(settings: Settings) -> Embedder:
    """Factory: construct the configured embedding backend."""
    backend = settings.embedding_backend.lower()
    if backend == "local":
        return LocalEmbedder(settings.embedding_model)
    if backend == "openai":
        return OpenAIEmbedder(settings.openai_embedding_model, settings.openai_api_key)
    raise ValueError(f"Unknown embedding backend: {settings.embedding_backend!r}")
