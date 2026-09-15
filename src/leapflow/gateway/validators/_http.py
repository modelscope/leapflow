# Copyright (c) Alibaba, Inc. and its affiliates.
"""Shared HTTP helper for credential validators."""

from __future__ import annotations

from typing import Any


def make_client_timeout() -> Any:
    """Create a strict per-request timeout for credential validation.

    ``aiohttp`` is imported lazily so this package stays importable without the
    optional dependency.
    """
    import aiohttp

    return aiohttp.ClientTimeout(total=8, connect=5)


__all__ = ["make_client_timeout"]
