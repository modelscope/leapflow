"""Transport factory table -- the transport half of the pluggability mechanism.

Adding support for a new southbound standard is a new module plus one row here.
Nothing upstream of this table knows which transports exist.

Deliberately absent: a factory for any unpublished standard. Pluggability means a
Protocol plus a lookup row, not an empty file waiting to be filled in; a
placeholder that returns nothing would be indistinguishable from a broken driver
at the moment it mattered.

Out-of-tree transports are discovered through the ``leapflow.hardware.transports``
entry-point group.  ``pip install -e my-driver`` is enough to make the transport
available to every profile without editing this file.
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
    "simulated": "leapflow.hardware.transports.simulated:build_transport",
    "python": "leapflow.hardware.transports.python_callable:build_transport",
    "mcp": "leapflow.hardware.transports.mcp:build_transport",
}

_EP_GROUP = "leapflow.hardware.transports"
_ep_scanned: bool = False


def _discover_entry_points() -> None:
    """Merge entry-point declared transports into ``_TRANSPORTS``, once.

    Idempotent: the scan runs only on the first call, guarded by ``_ep_scanned``.
    Manual registrations (via ``register_transport``) and the built-in table take
    precedence -- an entry-point whose name collides with an existing key is
    silently skipped so that an installed package cannot hijack a core transport.
    """
    global _ep_scanned  # noqa: PLW0603
    if _ep_scanned:
        return
    _ep_scanned = True

    try:
        from importlib.metadata import entry_points
    except ImportError:  # defensive: should never happen on 3.11+
        return

    try:
        # Python 3.12+ accepts ``group`` as a keyword directly.
        # Python 3.9-3.11 returns a dict-like SelectableGroups when called with
        # no arguments; the ``.select()`` method is the portable path (3.9.10+
        # / 3.10.2+).  Since the project requires >=3.11, ``.select()`` is
        # always available.
        eps = entry_points(group=_EP_GROUP)
    except TypeError:
        # Truly ancient importlib.metadata (should not be reachable on >=3.11).
        eps = entry_points().get(_EP_GROUP, [])  # type: ignore[arg-type,union-attr]

    for ep in eps:
        key = ep.name
        if key in _TRANSPORTS:
            logger.debug(
                "entry-point transport %r skipped: already registered", key
            )
            continue
        # Store as "module:attr" so the lazy import path in build_transport()
        # handles it identically to a built-in row.
        _TRANSPORTS[key] = f"{ep.value}"
        logger.debug("entry-point transport %r discovered -> %s", key, ep.value)


def available_transports() -> tuple[str, ...]:
    """Return the registered transport kinds, sorted for stable reporting."""
    _discover_entry_points()
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
    _discover_entry_points()
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
