# Copyright (c) Alibaba, Inc. and its affiliates.
"""Extension contract for dynamic prompt context providers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class VolatileContextProvider(Protocol):
    """Render bounded per-turn context outside the stable prompt prefix."""

    @property
    def provider_id(self) -> str:
        """Return a globally unique provider identifier."""
        ...

    @property
    def categories(self) -> frozenset[str]:
        """Return tool categories that make this provider relevant."""
        ...

    def render_volatile_context(
        self,
        *,
        session_id: str,
        disclosure_level: str,
    ) -> str:
        """Return bounded prompt text for the active session and PCD level."""
        ...


__all__ = ["VolatileContextProvider"]
