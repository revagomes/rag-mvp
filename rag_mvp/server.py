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
import logging

from fastapi import Depends, FastAPI, Header, HTTPException

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
from .security import (
    PathNotAllowedError,
    extract_bearer_token,
    is_authorized,
    resolve_within_root,
)

logger = logging.getLogger("rag_mvp")

_pipeline: RagPipeline | None = None


def require_api_key(authorization: str | None = Header(default=None)) -> None:
    """FastAPI dependency enforcing Bearer API-key auth on protected endpoints.

    No-op when no keys are configured (auth disabled). Otherwise requires a
    valid ``Authorization: Bearer <key>`` header: 401 if missing/malformed,
    403 if a token is present but not recognized.
    """
    settings = get_settings()
    allowed = settings.parsed_api_keys()
    if not allowed:
        return  # auth disabled
    token = extract_bearer_token(authorization)
    if token is None:
        raise HTTPException(
            status_code=401,
            detail="Missing or malformed Authorization header.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not is_authorized(token, allowed):
        raise HTTPException(status_code=403, detail="Invalid API key.")


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
    if not get_settings().auth_enabled:
        logger.warning(
            "API authentication is DISABLED (RAG_API_KEYS is empty). "
            "This is fine for local use, but do not expose this server on a "
            "network interface without setting RAG_API_KEYS."
        )
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


@app.post(
    "/ingest/text",
    response_model=IngestResponse,
    dependencies=[Depends(require_api_key)],
)
def ingest_text(req: IngestTextRequest) -> IngestResponse:
    pipeline = get_pipeline()
    result = pipeline.ingest_text(req.text, source=req.source)
    return IngestResponse(
        files_processed=result.files_processed,
        chunks_added=result.chunks_added,
        sources=result.sources,
        total_chunks_in_store=pipeline.document_count(),
    )


@app.post(
    "/ingest/path",
    response_model=IngestResponse,
    dependencies=[Depends(require_api_key)],
)
def ingest_path(req: IngestPathRequest) -> IngestResponse:
    pipeline = get_pipeline()
    settings = get_settings()
    # Confine ingestion to the configured root to prevent path traversal and
    # arbitrary file reads (which could then be exfiltrated via /query).
    try:
        path = resolve_within_root(req.path, settings.ingest_root)
    except PathNotAllowedError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if not path.exists():
        raise HTTPException(status_code=404, detail="Path not found.")
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


@app.post(
    "/query",
    response_model=QueryResponse,
    dependencies=[Depends(require_api_key)],
)
def query(req: QueryRequest) -> QueryResponse:
    pipeline = get_pipeline()
    try:
        result = pipeline.query(req.question, top_k=req.top_k)
    except RuntimeError as exc:
        # Log the detailed cause server-side, but do not expose internal
        # details (backend URLs, config hints) to the client.
        logger.warning("LLM backend error during /query: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="The language model backend is unavailable.",
        ) from exc
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


@app.post("/reset", dependencies=[Depends(require_api_key)])
def reset() -> dict[str, str]:
    pipeline = get_pipeline()
    pipeline.reset()
    return {"status": "cleared"}
