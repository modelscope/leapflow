# Copyright (c) Alibaba, Inc. and its affiliates.
"""Fixtures for the real end-to-end journey layer.

Every journey runs against a real ``leapd`` subprocess with the LLM boundary
served by a local cassette proxy. Mode selection is environment-driven so the
same journey bodies serve all four lanes:

- ``replay`` (default, and what CI PR/main lanes use) — offline, deterministic
- ``seed`` — author committed cassettes from a journey's declared script
- ``record`` — capture real provider traffic into cassettes
- ``live`` — run against a real provider, persisting nothing
"""

from __future__ import annotations

from typing import Iterator

import pytest

from tests._harness.cassette_proxy import LIVE, RECORD, resolve_mode, upstream_from_env
from tests._harness.journey import JourneyFactory


@pytest.fixture(scope="session")
def journey_mode() -> str:
    """Resolve and validate the cassette mode once per session."""
    mode = resolve_mode()
    if mode in (RECORD, LIVE):
        base_url, api_key, model = upstream_from_env()
        missing = [
            name
            for name, value in (
                ("LEAPFLOW_LLM_BASE_URL", base_url),
                ("LEAPFLOW_LLM_API_KEY", api_key),
                ("LEAPFLOW_LLM_MODEL", model),
            )
            if not value
        ]
        if missing:
            pytest.skip(
                f"mode {mode!r} needs a real provider; missing {missing} "
                "(the live lane injects them from secrets)"
            )
    return mode


@pytest.fixture
def journeys(journey_mode: str) -> Iterator[JourneyFactory]:
    """Factory that starts journeys and tears them down after the test.

    Note it does not take ``tmp_path``: the factory allocates a deliberately
    short scratch root, because a daemon Unix socket cannot live under pytest's
    deep temp directory on macOS.
    """
    factory = JourneyFactory(journey_mode)
    try:
        yield factory
    finally:
        factory.close()
