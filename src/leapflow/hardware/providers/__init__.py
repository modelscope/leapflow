"""Provider factory table -- the context half of the pluggability mechanism.

A provider answers "where does device knowledge come from". Adding an upstream
standard's descriptor import is a new module plus one row here; it records
anything it could not map in ``ContextProvenance.lossy_fields`` so that a drop in
fidelity is visible in the reference document instead of being absorbed silently.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Callable, Mapping, Protocol, runtime_checkable

from leapflow.hardware.context import HardwareContext

logger = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    """Raised when a provider cannot be constructed or configured.

    Never frozen and never a dataclass: CPython assigns ``__traceback__`` on
    every re-raise, and a frozen exception would replace the real failure with a
    complaint about that assignment.
    """

    def __init__(self, message: str, *, failure_code: str = "provider_error") -> None:
        super().__init__(message)
        self.failure_code = failure_code


@runtime_checkable
class HardwareContextProvider(Protocol):
    """Supplies hardware contexts from one source."""

    kind: str

    def discover(self) -> tuple[HardwareContext, ...]:
        """Return all contexts this provider can supply.

        Must not connect to any device: discovery has to work with the hardware
        powered off, and boot must never block on device I/O.
        """
        ...


_PROVIDERS: dict[str, str] = {
    "yaml": "leapflow.hardware.providers.yaml_provider:build_provider",
}


def available_providers() -> tuple[str, ...]:
    """Return the registered provider kinds, sorted for stable reporting."""
    return tuple(sorted(_PROVIDERS))


def register_provider(kind: str, target: str) -> Callable[[], None]:
    """Register a provider factory as ``"module:factory"``.

    Returns the undo callable for the same reason ``register_transport`` does: the
    table is process-global, so a plugin registers through its ``EffectScope`` and a
    reload cannot leave a stale factory behind.
    """
    key = str(kind).strip()
    if not key:
        raise ValueError("provider kind must be a non-empty string")
    if key in _PROVIDERS and _PROVIDERS[key] != target:
        raise ValueError(f"provider kind {key!r} is already registered")
    previous = _PROVIDERS.get(key)
    _PROVIDERS[key] = str(target)

    def _undo() -> None:
        if previous is None:
            _PROVIDERS.pop(key, None)
        else:
            _PROVIDERS[key] = previous

    return _undo


def build_provider(kind: str, config: Mapping[str, Any] | None = None) -> HardwareContextProvider:
    """Instantiate the provider registered for *kind*."""
    target = _PROVIDERS.get(str(kind).strip())
    if target is None:
        raise ProviderError(
            f"unknown provider kind {kind!r}; available: {', '.join(available_providers())}",
            failure_code="unknown_provider_kind",
        )
    module_path, _, factory_name = target.partition(":")
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ProviderError(
            f"provider kind {kind!r} is registered but not importable: {exc}",
            failure_code="provider_import_failed",
        ) from exc
    factory = getattr(module, factory_name, None)
    if not callable(factory):
        raise ProviderError(
            f"provider kind {kind!r} resolves to a non-callable factory",
            failure_code="provider_factory_missing",
        )
    return factory(config or {})


__all__ = [
    "HardwareContextProvider",
    "ProviderError",
    "available_providers",
    "build_provider",
    "register_provider",
]
