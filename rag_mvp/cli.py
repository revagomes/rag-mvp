"""Command-line interface for ingesting documents and querying the RAG store.

Usage:
    python -m rag_mvp.cli ingest <path> [<path> ...]
    python -m rag_mvp.cli query "your question here" [--top-k N]
    python -m rag_mvp.cli info
    python -m rag_mvp.cli reset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .pipeline import RagPipeline


def _cmd_ingest(pipeline: RagPipeline, paths: list[str]) -> int:
    total_files = 0
    total_chunks = 0
    for raw in paths:
        path = Path(raw).expanduser()
        if not path.exists():
            print(f"  ! skipping (not found): {path}", file=sys.stderr)
            continue
        if path.is_dir():
            result = pipeline.ingest_directory(path)
        else:
            try:
                result = pipeline.ingest_file(path)
            except ValueError as exc:
                print(f"  ! skipping ({exc})", file=sys.stderr)
                continue
        total_files += result.files_processed
        total_chunks += result.chunks_added
        for source in result.sources:
            print(f"  + {source}")
    print(
        f"\nIngested {total_chunks} chunk(s) from {total_files} file(s). "
        f"Store now holds {pipeline.document_count()} chunk(s)."
    )
    return 0


def _cmd_query(pipeline: RagPipeline, question: str, top_k: int | None) -> int:
    result = pipeline.query(question, top_k=top_k)
    print(f"\nQ: {result.question}\n")
    print(f"A: {result.answer}\n")
    print(f"--- retrieved {len(result.chunks)} chunk(s) "
          f"(llm backend: {result.llm_backend}) ---")
    for chunk in result.chunks:
        name = Path(chunk.source).name
        preview = chunk.text.replace("\n", " ")
        if len(preview) > 160:
            preview = preview[:157] + "..."
        print(f"  [{chunk.score:.3f}] {name} #{chunk.chunk_index}: {preview}")
    return 0


def _cmd_info(pipeline: RagPipeline) -> int:
    s = pipeline.settings
    print("rag-mvp configuration")
    print(f"  embedding backend : {s.embedding_backend} ({s.embedding_model})")
    print(f"  llm backend       : {s.llm_backend}")
    print(f"  chunk size/overlap: {s.chunk_size}/{s.chunk_overlap}")
    print(f"  top_k             : {s.top_k}")
    print(f"  storage dir       : {s.storage_dir}")
    print(f"  chunks in store   : {pipeline.document_count()}")
    return 0


def _cmd_reset(pipeline: RagPipeline) -> int:
    pipeline.reset()
    print("Vector store cleared.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-mvp", description="Local RAG MVP CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest files or directories")
    p_ingest.add_argument("paths", nargs="+", help="Files or directories to ingest")

    p_query = sub.add_parser("query", help="Ask a question")
    p_query.add_argument("question", help="The question to ask")
    p_query.add_argument("--top-k", type=int, default=None, help="Chunks to retrieve")

    sub.add_parser("info", help="Show configuration and store size")
    sub.add_parser("reset", help="Clear the vector store")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pipeline = RagPipeline()

    if args.command == "ingest":
        return _cmd_ingest(pipeline, args.paths)
    if args.command == "query":
        return _cmd_query(pipeline, args.question, args.top_k)
    if args.command == "info":
        return _cmd_info(pipeline)
    if args.command == "reset":
        return _cmd_reset(pipeline)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
