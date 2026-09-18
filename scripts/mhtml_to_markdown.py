"""Convert MHTML (.mhtml) web-archive files to clean Markdown.

MHTML is a MIME multipart container (RFC 2557) holding an HTML document plus its
resources (CSS, images) inline. Feeding the raw file to a RAG pipeline would
pollute embeddings with base64 blobs, quoted-printable escapes, and markup.

This script extracts the primary text/html part, decodes it, removes non-content
noise (scripts, styles, nav/chrome), and converts to Markdown so that document
structure (headings, lists) and — importantly — hyperlinks are preserved. A
retrieved chunk therefore keeps its `[text](url)` links.

Usage:
    python scripts/mhtml_to_markdown.py <src_dir_or_file> <out_dir>
"""

from __future__ import annotations

import re
import sys
from email import policy
from email.parser import BytesParser
from pathlib import Path

from bs4 import BeautifulSoup
from markdownify import markdownify as md

# Tags whose contents are never useful body text / would add noise.
_STRIP_TAGS = ["script", "style", "noscript", "head", "meta", "link", "svg",
               "img", "nav", "footer", "header", "button"]


def extract_html_from_mhtml(path: Path) -> str:
    """Return the decoded HTML of the largest text/html part in an MHTML file."""
    with path.open("rb") as fh:
        message = BytesParser(policy=policy.default).parse(fh)

    html_parts: list[str] = []
    for part in message.walk():
        if part.get_content_type() == "text/html":
            # get_content() handles quoted-printable/base64 + charset decoding.
            try:
                html_parts.append(part.get_content())
            except (LookupError, ValueError):
                payload = part.get_payload(decode=True) or b""
                html_parts.append(payload.decode("utf-8", errors="replace"))

    if not html_parts:
        return ""
    # The main document is typically the largest HTML part.
    return max(html_parts, key=len)


def clean_html(html: str) -> str:
    """Remove noise tags and links that are not real, resolvable URLs."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(_STRIP_TAGS):
        tag.decompose()

    # Drop hrefs that are inline resource refs or empty anchors so markdownify
    # does not emit useless [text](cid:...) / [text](#) links. Keep http(s)
    # and mailto links, which are the ones worth preserving for RAG context.
    for a in soup.find_all("a"):
        href = (a.get("href") or "").strip()
        if not href or href.startswith(("cid:", "#", "javascript:")):
            a.replace_with(a.get_text())
    return str(soup)


def html_to_markdown(html: str) -> str:
    cleaned = clean_html(html)
    markdown = md(cleaned, heading_style="ATX", strip=["span"])
    return _normalize(markdown)


def _normalize(text: str) -> str:
    # Drop U+FFFD replacement chars and zero-width junk left by the source page.
    text = text.replace("\ufffd", "").replace("\u200b", "").replace("\u200e", "")
    # Trim trailing spaces, collapse 3+ blank lines to a single blank line.
    lines = [re.sub(r"[ \t\u00a0]+$", "", ln.replace("\u00a0", " ")) for ln in text.splitlines()]
    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip() + "\n"


def safe_stem(name: str) -> str:
    """Make a filesystem- and RAG-friendly stem from an MHTML filename."""
    base = Path(name).name
    # Drop a trailing .ext only when a real name precedes the dot; this avoids
    # Path.stem's quirk of treating ".mhtml" (extension only) as a full stem.
    dot = base.rfind(".")
    stem = base[:dot] if dot > 0 else base
    stem = stem.replace("_", " ")
    stem = re.sub(r"\s+", "_", stem.strip())
    stem = re.sub(r"[^A-Za-z0-9._-]", "", stem)
    stem = stem.strip("._-")
    return stem or "document"


def convert_file(src: Path, out_dir: Path) -> tuple[Path, int]:
    html = extract_html_from_mhtml(src)
    markdown = html_to_markdown(html)
    out_path = out_dir / f"{safe_stem(src.name)}.md"
    out_path.write_text(markdown, encoding="utf-8")
    return out_path, len(markdown)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    src = Path(argv[1])
    out_dir = Path(argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)

    sources = sorted(src.glob("*.mhtml")) if src.is_dir() else [src]
    if not sources:
        print(f"No .mhtml files found under {src}")
        return 1

    total = 0
    for path in sources:
        out_path, size = convert_file(path, out_dir)
        total += 1
        print(f"  {path.name}  ->  {out_path.name}  ({size:,} chars)")
    print(f"\nConverted {total} file(s) into {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
