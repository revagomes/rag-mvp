"""Security helpers for the RAG server.

Currently: filesystem containment for the /ingest/path endpoint, to prevent
path traversal and arbitrary file reads (e.g. /etc/passwd, ~/.ssh/id_rsa, .env)
whose contents could then be exfiltrated verbatim through /query.
"""

from __future__ import annotations

import hmac
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



def extract_bearer_token(authorization_header: str | None) -> str | None:
    """Return the token from an ``Authorization: Bearer <token>`` header.

    Returns None when the header is missing or not a well-formed Bearer header.
    The scheme comparison is case-insensitive per RFC 7235.
    """
    if not authorization_header:
        return None
    parts = authorization_header.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    token = parts[1].strip()
    return token or None


def is_authorized(token: str | None, allowed_keys: set[str]) -> bool:
    """Constant-time check that ``token`` matches one of ``allowed_keys``.

    Uses ``hmac.compare_digest`` against every configured key so the comparison
    time does not reveal how many characters matched (timing-attack resistant).
    Returns False for a missing token or when no keys are configured.
    """
    if not token or not allowed_keys:
        return False
    # Evaluate all keys (no short-circuit) to keep timing independent of order.
    matched = False
    for key in allowed_keys:
        if hmac.compare_digest(token, key):
            matched = True
    return matched
