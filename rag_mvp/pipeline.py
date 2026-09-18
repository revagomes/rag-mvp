"""The RAG pipeline: ties ingestion, embeddings, vector store, and LLM together.

A single ``RagPipeline`` instance holds the embedder, vector store and LLM and
exposes high-level ``ingest_*`` and ``query`` methods used by both the HTTP API
and the CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import Settings, get_settings
from .embeddings import Embedder, build_embedder
from .ingest import (
    Chunk,
    chunk_text,
    discover_files,
    load_and_chunk_file,
)
from .llm import LLM, build_llm
from .vectorstore import RetrievedChunk, VectorStore


@dataclass
class IngestResult:
    files_processed: int
    chunks_added: int
    sources: list[str]


@dataclass
class QueryResult:
    question: str
    answer: str
    chunks: list[RetrievedChunk]
    llm_backend: str


class RagPipeline:
    """Orchestrates the full retrieval-augmented generation flow."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        llm: LLM | None = None,
    ) -> None:
        """Construct the pipeline.

        Dependencies are built from ``settings`` by default, but each may be
        supplied directly. Injecting them keeps tests fast and offline: a fake
        embedder and an in-memory store avoid model downloads and network I/O.
        """
        self.settings = settings or get_settings()
        self.embedder: Embedder = embedder or build_embedder(self.settings)
        self.store = store or VectorStore(
            self.settings.storage_dir, self.settings.collection_name
        )
        self.llm: LLM = llm or build_llm(self.settings)

    # ---- Ingestion ----

    def ingest_chunks(self, chunks: list[Chunk]) -> int:
        """Embed and store a list of chunks. Returns the number stored."""
        if not chunks:
            return 0
        embeddings = self.embedder.embed_documents([c.text for c in chunks])
        self.store.add(chunks, embeddings)
        return len(chunks)

    def ingest_text(self, text: str, *, source: str) -> IngestResult:
        chunks = chunk_text(
            text,
            source=source,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        added = self.ingest_chunks(chunks)
        return IngestResult(
            files_processed=1 if added else 0,
            chunks_added=added,
            sources=[source] if added else [],
        )

    def ingest_file(self, path: Path) -> IngestResult:
        chunks = load_and_chunk_file(
            path,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )
        added = self.ingest_chunks(chunks)
        return IngestResult(
            files_processed=1 if added else 0,
            chunks_added=added,
            sources=[str(path)] if added else [],
        )

    def ingest_directory(self, directory: Path) -> IngestResult:
        files = discover_files(directory)
        total_chunks = 0
        sources: list[str] = []
        for path in files:
            result = self.ingest_file(path)
            total_chunks += result.chunks_added
            sources.extend(result.sources)
        return IngestResult(
            files_processed=len(files),
            chunks_added=total_chunks,
            sources=sources,
        )

    # ---- Query ----

    def query(self, question: str, top_k: int | None = None) -> QueryResult:
        k = top_k or self.settings.top_k
        query_embedding = self.embedder.embed_query(question)
        chunks = self.store.query(query_embedding, k)

        if not chunks:
            return QueryResult(
                question=question,
                answer="No documents have been ingested yet, so I can't answer.",
                chunks=[],
                llm_backend=self.settings.llm_backend,
            )

        context = self._format_context(chunks)
        answer = self.llm.generate(question, context)
        return QueryResult(
            question=question,
            answer=answer,
            chunks=chunks,
            llm_backend=self.settings.llm_backend,
        )

    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        blocks: list[str] = []
        for chunk in chunks:
            name = Path(chunk.source).name
            blocks.append(f"[source: {name} #{chunk.chunk_index}]\n{chunk.text}")
        return "\n\n---\n\n".join(blocks)

    # ---- Maintenance ----

    def document_count(self) -> int:
        return self.store.count()

    def reset(self) -> None:
        self.store.reset()
