"""LLM backends for answer generation.

Three backends:

* ``none``   — no LLM. The pipeline returns retrieved chunks only. Useful for
  testing retrieval quality without running any model.
* ``ollama`` — a local Ollama server (http://localhost:11434 by default).
* ``openai`` — the OpenAI chat completions API (requires ``OPENAI_API_KEY``).
"""

from __future__ import annotations

from typing import Protocol

import httpx

from .config import Settings

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using only the provided "
    "context. If the answer is not contained in the context, say you don't know. "
    "Be concise and cite the source filenames you used."
)


class LLM(Protocol):
    """Common interface for answer-generating backends."""

    def generate(self, question: str, context: str) -> str:
        ...


class NoneLLM:
    """No-op backend: signals that only retrieval was performed."""

    def generate(self, question: str, context: str) -> str:  # noqa: ARG002
        return (
            "[LLM backend is 'none'] No answer generated. "
            "Returning retrieved context only. Set RAG_LLM_BACKEND=ollama or "
            "openai to generate answers."
        )


def _build_user_prompt(question: str, context: str) -> str:
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above."
    )


class OllamaLLM:
    """Generate answers with a local Ollama model."""

    def __init__(self, base_url: str, model: str, timeout: float = 120.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    def generate(self, question: str, context: str) -> str:
        payload = {
            "model": self._model,
            "system": SYSTEM_PROMPT,
            "prompt": _build_user_prompt(question, context),
            "stream": False,
        }
        try:
            response = httpx.post(
                f"{self._base_url}/api/generate",
                json=payload,
                timeout=self._timeout,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise RuntimeError(
                f"Ollama request failed ({exc}). Is Ollama running at "
                f"{self._base_url} with model '{self._model}' pulled?"
            ) from exc
        return response.json().get("response", "").strip()


class OpenAILLM:
    """Generate answers with the OpenAI chat completions API."""

    def __init__(self, model: str, api_key: str | None) -> None:
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required for the OpenAI LLM backend."
            )
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self._model = model

    def generate(self, question: str, context: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(question, context)},
            ],
        )
        return (response.choices[0].message.content or "").strip()


def build_llm(settings: Settings) -> LLM:
    """Factory: construct the configured LLM backend."""
    backend = settings.llm_backend.lower()
    if backend == "none":
        return NoneLLM()
    if backend == "ollama":
        return OllamaLLM(settings.ollama_url, settings.ollama_model)
    if backend == "openai":
        return OpenAILLM(settings.openai_chat_model, settings.openai_api_key)
    raise ValueError(f"Unknown LLM backend: {settings.llm_backend!r}")
