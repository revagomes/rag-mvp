"""Pydantic request/response models for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class IngestTextRequest(BaseModel):
    text: str = Field(..., description="Raw document text to ingest.")
    source: str = Field("inline", description="A label identifying the source.")


class IngestPathRequest(BaseModel):
    path: str = Field(..., description="Absolute or relative path to a file or directory.")


class IngestResponse(BaseModel):
    files_processed: int
    chunks_added: int
    sources: list[str]
    total_chunks_in_store: int


class QueryRequest(BaseModel):
    question: str = Field(..., description="The user's question.")
    top_k: int | None = Field(None, ge=1, le=50, description="Number of chunks to retrieve.")


class RetrievedChunkModel(BaseModel):
    source: str
    chunk_index: int
    score: float
    text: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    llm_backend: str
    chunks: list[RetrievedChunkModel]


class HealthResponse(BaseModel):
    status: str
    embedding_backend: str
    llm_backend: str
    documents_in_store: int
