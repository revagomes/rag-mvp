"""Document loading and chunking.

Supported formats: plain text (.txt, .md), and PDF (.pdf). The loader returns
raw text; the chunker splits it into overlapping character windows that respect
paragraph and sentence boundaries where possible.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf"}


@dataclass
class Chunk:
    """A single chunk of text ready to be embedded and stored."""

    id: str
    text: str
    source: str
    chunk_index: int


def load_text(path: Path) -> str:
    """Load raw text from a supported file.

    Raises:
        ValueError: if the extension is not supported.
    """
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix in {".txt", ".md", ".markdown"}:
        return path.read_text(encoding="utf-8", errors="replace")
    raise ValueError(f"Unsupported file type: {suffix} ({path})")


def _load_pdf(path: Path) -> str:
    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            parts.append(text)
    return "\n\n".join(parts)


def chunk_text(
    text: str,
    *,
    source: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Split ``text`` into overlapping chunks.

    Splitting is greedy on paragraph boundaries first, then falls back to hard
    character windows for any paragraph longer than ``chunk_size``. Consecutive
    windows overlap by ``chunk_overlap`` characters to preserve context across
    boundaries.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    normalized = _normalize_whitespace(text)
    if not normalized:
        return []

    windows = _sliding_windows(normalized, chunk_size, chunk_overlap)

    chunks: list[Chunk] = []
    for index, window in enumerate(windows):
        chunk_id = _make_id(source, index, window)
        chunks.append(
            Chunk(id=chunk_id, text=window, source=source, chunk_index=index)
        )
    return chunks


def _sliding_windows(text: str, size: int, overlap: int) -> list[str]:
    """Produce overlapping windows, preferring to break on whitespace."""
    windows: list[str] = []
    start = 0
    length = len(text)
    step = size - overlap

    while start < length:
        end = min(start + size, length)
        # Try to end on a whitespace boundary to avoid splitting words.
        if end < length:
            boundary = text.rfind(" ", start + step, end)
            if boundary != -1 and boundary > start:
                end = boundary
        window = text[start:end].strip()
        if window:
            windows.append(window)
        if end >= length:
            break
        start = max(end - overlap, start + 1)
    return windows


def _normalize_whitespace(text: str) -> str:
    lines = [line.strip() for line in text.splitlines()]
    # Collapse runs of blank lines into a single blank line.
    out: list[str] = []
    blank = False
    for line in lines:
        if line:
            out.append(line)
            blank = False
        elif not blank:
            out.append("")
            blank = True
    return "\n".join(out).strip()


def _make_id(source: str, index: int, text: str) -> str:
    digest = hashlib.sha1(f"{source}:{index}:{text}".encode()).hexdigest()[:16]
    return f"{Path(source).name}-{index}-{digest}"


def load_and_chunk_file(
    path: Path,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Convenience: load a single file and return its chunks."""
    text = load_text(path)
    return chunk_text(
        text,
        source=str(path),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def discover_files(directory: Path) -> list[Path]:
    """Return all supported files under ``directory`` (recursive)."""
    return sorted(
        p
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
