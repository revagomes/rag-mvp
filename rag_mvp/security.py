"""Security helpers for the RAG server.

Currently: filesystem containment for the /ingest/path endpoint, to prevent
path traversal and arbitrary file reads (e.g. /etc/passwd, ~/.ssh/id_rsa, .env)
whose contents could then be exfiltrated verbatim through /query.
"""

from __future__ import annotations

from pathlib import Path


class PathNotAllowedError(Exception):
    """Raised when a requested path resolves outside the permitted ingest root."""


def resolve_within_root(requested: str | Path, root: str | Path) -> Path:
    """Resolve ``requested`` and ensure it stays inside ``root``.

    Both paths are fully resolved (symlinks included) before comparison, so
    neither ``..`` traversal nor a symlink pointing outside the root can escape.

    Returns:
        The resolved absolute path, guaranteed to be ``root`` itself or a
        descendant of it.

    Raises:
        PathNotAllowedError: if the resolved path is outside ``root``.
    """
    root_resolved = Path(root).expanduser().resolve()
    # Note: we intentionally do NOT expanduser() the requested path — a caller
    # should not be able to reach ~ outside the configured root via "~".
    target = Path(requested)
    if not target.is_absolute():
        target = root_resolved / target
    target_resolved = target.resolve()

    if target_resolved != root_resolved and root_resolved not in target_resolved.parents:
        raise PathNotAllowedError(
            f"Path is outside the permitted ingest root: {requested}"
        )
    return target_resolved
