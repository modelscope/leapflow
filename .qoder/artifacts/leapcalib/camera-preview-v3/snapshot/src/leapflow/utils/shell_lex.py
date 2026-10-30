# Copyright (c) Alibaba, Inc. and its affiliates.
"""Shell-style argument tokenization that survives Windows paths."""

from __future__ import annotations

import os
import shlex


def split_args(text: str) -> list[str]:
    """Tokenize a command/argument string like ``shlex.split``, keeping Windows paths.

    POSIX shlex treats ``\\`` as an escape and silently strips the separators
    out of ``C:\\a\\b.yaml``; on Windows ``\\`` is a path separator, so double
    it before lexing. Quote semantics are otherwise unchanged. Raises
    ``ValueError`` on unbalanced quoting, exactly like ``shlex.split``.
    """
    lexable = text.replace("\\", "\\\\") if os.name == "nt" else text
    return shlex.split(lexable)
