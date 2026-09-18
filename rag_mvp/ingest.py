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
    """Produce overlapping windows built from whole words.

    The text is split on whitespace and windows are assembled word-by-word so a
    word is never split across a boundary. ``size`` is a character budget per
    window; ``overlap`` is the approximate number of trailing characters carried
    into the next window. A single token longer than ``size`` is emitted on its
    own (and hard-split only as an unavoidable last resort).
    """
    words = text.split()
    if not words:
        return []

    windows: list[str] = []
    i = 0
    n = len(words)

    while i < n:
        current: list[str] = []
        current_len = 0
        j = i
        while j < n:
            word = words[j]
            added = len(word) + (1 if current else 0)
            if current and current_len + added > size:
                break
            # A lone word longer than the whole budget: hard-split it.
            if not current and len(word) > size:
                for piece_start in range(0, len(word), size):
                    windows.append(word[piece_start:piece_start + size])
                j += 1
                i = j
                current = []
                current_len = 0
                break
            current.append(word)
            current_len += added
            j += 1
        else:
            # inner loop exhausted words without breaking
            if current:
                windows.append(" ".join(current))
            break

        if current:
            windows.append(" ".join(current))
            # Advance start so the next window overlaps by ~overlap characters
            # worth of trailing words, without ever moving backwards.
            back = 0
            k = j
            while k > i + 1 and back < overlap:
                back += len(words[k - 1]) + 1
                k -= 1
            i = k
        # If current was empty we already advanced i during the hard-split.

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
