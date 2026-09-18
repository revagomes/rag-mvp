"""FastAPI application exposing the RAG pipeline over HTTP.

Endpoints:
    GET  /health          — liveness + configuration summary
    POST /ingest/text     — ingest raw text
    POST /ingest/path     — ingest a file or directory from the server's disk
    POST /query           — retrieve + (optionally) generate an answer
    POST /reset           — clear the vector store

Run with:  uvicorn rag_mvp.server:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from .config import get_settings
from .pipeline import RagPipeline
from .schemas import (
    HealthResponse,
    IngestPathRequest,
    IngestResponse,
    IngestTextRequest,
    QueryRequest,
    QueryResponse,
    RetrievedChunkModel,
)

_pipeline: RagPipeline | None = None


def get_pipeline() -> RagPipeline:
    """Return the process-wide pipeline, constructing it on first use."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RagPipeline()
    return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    # Warm up the pipeline (loads embedding model) at startup.
    get_pipeline()
    yield


app = FastAPI(
    title="rag-mvp",
    version="0.1.0",
    description="A minimal local-first RAG server.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    pipeline = get_pipeline()
    return HealthResponse(
        status="ok",
        embedding_backend=settings.embedding_backend,
        llm_backend=settings.llm_backend,
        documents_in_store=pipeline.document_count(),
    )


@app.post("/ingest/text", response_model=IngestResponse)
def ingest_text(req: IngestTextRequest) -> IngestResponse:
    pipeline = get_pipeline()
    result = pipeline.ingest_text(req.text, source=req.source)
    return IngestResponse(
        files_processed=result.files_processed,
        chunks_added=result.chunks_added,
        sources=result.sources,
        total_chunks_in_store=pipeline.document_count(),
    )


@app.post("/ingest/path", response_model=IngestResponse)
def ingest_path(req: IngestPathRequest) -> IngestResponse:
    pipeline = get_pipeline()
    path = Path(req.path).expanduser()
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")
    try:
        if path.is_dir():
            result = pipeline.ingest_directory(path)
        else:
            result = pipeline.ingest_file(path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return IngestResponse(
        files_processed=result.files_processed,
        chunks_added=result.chunks_added,
        sources=result.sources,
        total_chunks_in_store=pipeline.document_count(),
    )


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    pipeline = get_pipeline()
    try:
        result = pipeline.query(req.question, top_k=req.top_k)
    except RuntimeError as exc:
        # e.g. Ollama not running / OpenAI key missing
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return QueryResponse(
        question=result.question,
        answer=result.answer,
        llm_backend=result.llm_backend,
        chunks=[
            RetrievedChunkModel(
                source=c.source,
                chunk_index=c.chunk_index,
                score=round(c.score, 4),
                text=c.text,
            )
            for c in result.chunks
        ],
    )


@app.post("/reset")
def reset() -> dict[str, str]:
    pipeline = get_pipeline()
    pipeline.reset()
    return {"status": "cleared"}
