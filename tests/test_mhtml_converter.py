"""Offline tests for the MHTML->Markdown converter (scripts/mhtml_to_markdown.py).

The script is a standalone utility, not part of the rag_mvp package, so it is
imported by file path. A synthetic MHTML document (built inline, no network) is
used to assert the important behaviors: links and structure are preserved while
noise is stripped.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# bs4/markdownify are optional (the "convert" extra); skip cleanly if absent.
pytest.importorskip("bs4")
pytest.importorskip("markdownify")

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "mhtml_to_markdown.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("mhtml_to_markdown", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


mod = _load_module()


def _make_mhtml(html_body: str) -> bytes:
    """Wrap an HTML fragment in a minimal, valid MHTML MIME container."""
    boundary = "----TestBoundary----"
    html_doc = (
        "<!DOCTYPE html><html><head>"
        "<style>.x{color:red}</style><title>ignored</title></head>"
        f"<body>{html_body}</body></html>"
    )
    return (
        "From: <Saved by Test>\r\n"
        "Subject: Test\r\n"
        "MIME-Version: 1.0\r\n"
        f'Content-Type: multipart/related; boundary="{boundary}"\r\n'
        "\r\n"
        f"--{boundary}\r\n"
        "Content-Type: text/html\r\n"
        "Content-Transfer-Encoding: 8bit\r\n"
        "\r\n"
        f"{html_doc}\r\n"
        f"--{boundary}--\r\n"
    ).encode("utf-8")


class TestExtractHtml:
    def test_extracts_html_part(self, tmp_path):
        p = tmp_path / "a.mhtml"
        p.write_bytes(_make_mhtml("<p>Hello world</p>"))
        html = mod.extract_html_from_mhtml(p)
        assert "Hello world" in html

    def test_returns_empty_when_no_html_part(self, tmp_path):
        p = tmp_path / "b.mhtml"
        p.write_bytes(
            b"From: x\r\nMIME-Version: 1.0\r\n"
            b"Content-Type: text/plain\r\n\r\njust text\r\n"
        )
        assert mod.extract_html_from_mhtml(p) == ""


class TestHtmlToMarkdown:
    def test_http_link_is_preserved(self):
        html = '<p>See <a href="https://example.com/doc">the doc</a>.</p>'
        md = mod.html_to_markdown(html)
        assert "[the doc](https://example.com/doc)" in md

    def test_mailto_link_is_preserved(self):
        html = '<p>Mail <a href="mailto:x@example.com">us</a>.</p>'
        md = mod.html_to_markdown(html)
        assert "[us](mailto:x@example.com)" in md

    def test_heading_becomes_atx(self):
        md = mod.html_to_markdown("<h1>Title</h1><h2>Sub</h2>")
        assert "# Title" in md
        assert "## Sub" in md

    def test_list_structure_preserved(self):
        md = mod.html_to_markdown("<ul><li>one</li><li>two</li></ul>")
        assert "one" in md and "two" in md
        assert md.count("one") == 1

    def test_script_and_style_stripped(self):
        html = (
            "<style>.a{color:red}</style>"
            "<script>alert('x')</script>"
            "<p>real content</p>"
        )
        md = mod.html_to_markdown(html)
        assert "real content" in md
        assert "color:red" not in md
        assert "alert" not in md

    def test_cid_link_is_dropped_but_text_kept(self):
        html = '<p><a href="cid:image001@x">logo</a> caption</p>'
        md = mod.html_to_markdown(html)
        assert "logo" in md
        assert "cid:" not in md
        assert "](cid" not in md

    def test_anchor_only_link_is_dropped(self):
        html = '<p><a href="#top">back to top</a></p>'
        md = mod.html_to_markdown(html)
        assert "back to top" in md
        assert "](#top)" not in md

    def test_javascript_link_is_dropped(self):
        html = "<p><a href=\"javascript:void(0)\">click</a></p>"
        md = mod.html_to_markdown(html)
        assert "click" in md
        assert "javascript:" not in md

    def test_replacement_char_mojibake_removed(self):
        md = mod.html_to_markdown("<p>\ufffd\ufffdclean text</p>")
        assert "\ufffd" not in md
        assert "clean text" in md

    def test_output_ends_with_single_newline(self):
        md = mod.html_to_markdown("<p>content</p>")
        assert md.endswith("\n")
        assert not md.endswith("\n\n")

    def test_no_triple_blank_lines(self):
        html = "<p>a</p><br><br><br><br><p>b</p>"
        md = mod.html_to_markdown(html)
        assert "\n\n\n" not in md


class TestSafeStem:
    def test_spaces_become_underscores(self):
        assert mod.safe_stem("Some Report part 1.mhtml") == "Some_Report_part_1"

    def test_strips_unsafe_chars(self):
        assert mod.safe_stem("a:b c*d.mhtml") == "ab_cd"

    def test_underscores_normalized(self):
        # Underscores are treated as spaces then re-joined.
        assert mod.safe_stem("USER_GUIDE.mhtml") == "USER_GUIDE"

    def test_empty_falls_back(self):
        # A name whose usable characters are all stripped falls back to "document".
        assert mod.safe_stem("___.mhtml") == "document"
        assert mod.safe_stem("###.mhtml") == "document"


class TestConvertFileEndToEnd:
    def test_convert_file_writes_markdown_with_links(self, tmp_path):
        src = tmp_path / "page.mhtml"
        src.write_bytes(
            _make_mhtml(
                "<h1>Guide</h1>"
                '<p>Read <a href="https://example.com/x">this</a>.</p>'
                "<script>bad()</script>"
            )
        )
        out_dir = tmp_path / "out"
        out_dir.mkdir()
        out_path, size = mod.convert_file(src, out_dir)

        assert out_path.name == "page.md"
        assert size > 0
        text = out_path.read_text(encoding="utf-8")
        assert "# Guide" in text
        assert "[this](https://example.com/x)" in text
        assert "bad()" not in text
