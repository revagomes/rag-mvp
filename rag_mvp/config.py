"""Application configuration, loaded from environment variables / .env file."""

from __future__ import annotations

from functools import lru_cache

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the RAG server.

    Values are read from environment variables or a local ``.env`` file.
    Most fields use the ``RAG_`` prefix (e.g. ``RAG_TOP_K``). Every field has a
    sensible default so the server runs out of the box with no configuration
    and no API keys.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="RAG_",
        extra="ignore",
    )

    # ---- Embeddings ----
    embedding_backend: str = "local"  # "local" | "openai"
    embedding_model: str = "all-MiniLM-L6-v2"
    openai_embedding_model: str = "text-embedding-3-small"

    # ---- LLM ----
    llm_backend: str = "none"  # "none" | "ollama" | "openai"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"
    openai_chat_model: str = "gpt-4o-mini"

    # ---- OpenAI credential (no RAG_ prefix; accepts the standard var name) ----
    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "RAG_OPENAI_API_KEY"),
    )

    # ---- Chunking ----
    chunk_size: int = 800
    chunk_overlap: int = 120

    # ---- Retrieval ----
    top_k: int = 4

    # ---- Storage ----
    storage_dir: str = "./storage"
    collection_name: str = "documents"

    # ---- Security ----
    # Filesystem root that /ingest/path is confined to. Requests to ingest a
    # path outside this root are rejected. Defaults to the current working
    # directory, which is where the server is normally launched.
    ingest_root: str = "."

    # Comma-separated API keys accepted via `Authorization: Bearer <key>` on
    # protected endpoints. Empty (the default) disables auth entirely, which is
    # fine for local/CLI use but MUST NOT be used on a network interface.
    api_keys: str = ""

    def parsed_api_keys(self) -> set[str]:
        """Return the configured API keys as a set of non-empty, trimmed strings."""
        return {k.strip() for k in self.api_keys.split(",") if k.strip()}

    @property
    def auth_enabled(self) -> bool:
        return bool(self.parsed_api_keys())


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
