"""Content revision for one atomic LeapBoard server generation.

A Board process imports Python once but historically read YAML/JS/CSS from disk on every
request. Editing a dashboard file while the process lived therefore created an impossible
mixture: new template/static assets plus old builder code. This module gives the launcher
and the server one content-addressed generation boundary.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


_DASHBOARD_ROOT = Path(__file__).parent
_REVISION_SUFFIXES = frozenset({".py", ".yaml", ".js", ".css", ".html"})


def board_revision(root: Path | None = None) -> str:
    """Return a stable digest of dashboard code, templates and static assets.

    File names and bytes both enter the digest, in sorted order. Transient files and
    `__pycache__` do not participate, so one machine's imports cannot make another
    machine's Board generation look incompatible.
    """
    directory = root or _DASHBOARD_ROOT
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix not in _REVISION_SUFFIXES:
            continue
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(directory).as_posix().encode("utf-8")
        try:
            content = path.read_bytes()
        except OSError:
            # A file changing during revision collection is itself a generation mismatch.
            # Use a deterministic marker rather than raising from a launcher/status path.
            content = b"<unreadable>"
        digest.update(relative)
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest()[:16]


__all__ = ["board_revision"]
