"""Tests for filesystem containment (rag_mvp.security.resolve_within_root)."""

from __future__ import annotations

import pytest

from rag_mvp.security import PathNotAllowedError, resolve_within_root


class TestResolveWithinRoot:
    def test_file_inside_root_is_allowed(self, tmp_path):
        (tmp_path / "doc.txt").write_text("x", encoding="utf-8")
        resolved = resolve_within_root("doc.txt", tmp_path)
        assert resolved == (tmp_path / "doc.txt").resolve()

    def test_nested_file_inside_root_is_allowed(self, tmp_path):
        sub = tmp_path / "a" / "b"
        sub.mkdir(parents=True)
        (sub / "doc.txt").write_text("x", encoding="utf-8")
        resolved = resolve_within_root("a/b/doc.txt", tmp_path)
        assert resolved == (sub / "doc.txt").resolve()

    def test_root_itself_is_allowed(self, tmp_path):
        assert resolve_within_root(".", tmp_path) == tmp_path.resolve()

    def test_absolute_path_inside_root_is_allowed(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("x", encoding="utf-8")
        assert resolve_within_root(str(f), tmp_path) == f.resolve()

    def test_dotdot_traversal_is_blocked(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        with pytest.raises(PathNotAllowedError):
            resolve_within_root("../secret.txt", root)

    def test_deep_dotdot_traversal_is_blocked(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        with pytest.raises(PathNotAllowedError):
            resolve_within_root("a/../../../../etc/passwd", root)

    def test_absolute_path_outside_root_is_blocked(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        with pytest.raises(PathNotAllowedError):
            resolve_within_root("/etc/passwd", root)

    def test_symlink_escaping_root_is_blocked(self, tmp_path):
        root = tmp_path / "root"
        root.mkdir()
        outside = tmp_path / "outside_secret.txt"
        outside.write_text("secret", encoding="utf-8")
        link = root / "link.txt"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks not supported on this platform")
        with pytest.raises(PathNotAllowedError):
            resolve_within_root("link.txt", root)

    def test_tilde_is_not_expanded_to_escape(self, tmp_path):
        # "~" must be treated as a literal path segment under root, never the
        # user's real home directory.
        root = tmp_path / "root"
        root.mkdir()
        # Either it resolves to a (non-existent) path *inside* root, or raises;
        # it must never resolve to the real home directory.
        try:
            resolved = resolve_within_root("~/.ssh/id_rsa", root)
        except PathNotAllowedError:
            return
        assert root.resolve() in resolved.parents or resolved == root.resolve()
