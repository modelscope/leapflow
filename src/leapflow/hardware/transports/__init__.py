"""Transport factory table -- the transport half of the pluggability mechanism.

Adding support for a new southbound standard is a new module plus one row here.
Nothing upstream of this table knows which transports exist.

Deliberately absent: a factory for any unpublished standard. Pluggability means a
Protocol plus a lookup row, not an empty file waiting to be filled in; a
placeholder that returns nothing would be indistinguishable from a broken driver
at the moment it mattered.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping

from leapflow.hardware.transport import HardwareTransport, TransportError

logger = logging.getLogger(__name__)

TransportFactory = Callable[[Mapping[str, Any]], HardwareTransport]

_TRANSPORTS: dict[str, str] = {
    # kind -> "module:factory", imported lazily so that an optional dependency in
    # one transport cannot break registry loading for the others.
    "mock": "leapflow.hardware.transports.mock:build_transport",
    "python": "leapflow.hardware.transports.python_callable:build_transport",
    "mcp": "leapflow.hardware.transports.mcp:build_transport",
}


def available_transports() -> tuple[str, ...]:
    """Return the registered transport kinds, sorted for stable reporting."""
    return tuple(sorted(_TRANSPORTS))


def register_transport(kind: str, target: str) -> Callable[[], None]:
    """Register a transport factory as ``"module:factory"``.

    Exposed so a plugin can contribute a transport without editing this table.
    Re-registering an existing kind is refused rather than silently overriding: a
    device quietly switching transports is not a debuggable state.

    Returns the undo callable, because the table is process-global and this is a
    mutation with a lifetime. A plugin registers through its ``EffectScope`` so a
    hot reload cannot leave a factory pointing at a module that no longer exists:

        scope.effect(register_transport("my_rig", "my_pkg.driver:build"))
    """
    key = str(kind).strip()
    if not key:
        raise ValueError("transport kind must be a non-empty string")
    if key in _TRANSPORTS and _TRANSPORTS[key] != target:
        raise ValueError(f"transport kind {key!r} is already registered")
    previous = _TRANSPORTS.get(key)
    _TRANSPORTS[key] = str(target)

    def _undo() -> None:
        if previous is None:
            _TRANSPORTS.pop(key, None)
        else:
            _TRANSPORTS[key] = previous

    return _undo


def build_transport(kind: str, config: Mapping[str, Any] | None = None) -> HardwareTransport:
    """Instantiate the transport registered for *kind*."""
    target = _TRANSPORTS.get(str(kind).strip())
    if target is None:
        raise TransportError(
            f"unknown transport kind {kind!r}; available: {', '.join(available_transports())}",
            failure_code="unknown_transport_kind",
        )
    module_path, _, factory_name = target.partition(":")
    import importlib

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise TransportError(
            f"transport kind {kind!r} is registered but not importable: {exc}",
            failure_code="transport_import_failed",
        ) from exc
    factory = getattr(module, factory_name, None)
    if not callable(factory):
        raise TransportError(
            f"transport kind {kind!r} resolves to a non-callable factory",
            failure_code="transport_factory_missing",
        )
    return factory(config or {})


__all__ = ["available_transports", "build_transport", "register_transport"]
