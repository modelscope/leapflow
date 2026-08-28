"""Transport that delegates to an externally supplied Python driver.

This is the hardware-neutral escape hatch. A vendor SDK, a serial library, or a
single-board-computer GPIO wrapper lives *outside* this repository and is named
by the declaration:

    transport:
      kind: python
      config:
        module: my_lab_drivers.bench_node
        factory: build_transport
        options: {port: /dev/ttyUSB0}

The referenced factory returns any object satisfying ``HardwareTransport``. No
device-specific code belongs in this file, or in this package: the whole point of
the transport seam is that adding a device is a declaration plus an external
module, never a change here.
"""

from __future__ import annotations

import importlib
from typing import Any, Mapping

from leapflow.hardware.transport import HardwareTransport, TransportError


def build_transport(config: Mapping[str, Any] | None = None) -> HardwareTransport:
    """Import and instantiate the declared driver factory.

    Fails closed with a structured reason rather than a bare ImportError: a
    missing driver must be reported as an unusable device, not crash registry
    loading for every other device in the profile.
    """
    config = config or {}
    module_path = str(config.get("module") or "").strip()
    factory_name = str(config.get("factory") or "build_transport").strip()
    if not module_path:
        raise TransportError(
            "python transport requires config.module naming an importable driver",
            failure_code="driver_module_missing",
        )

    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise TransportError(
            f"cannot import driver module {module_path!r}: {exc}",
            failure_code="driver_import_failed",
        ) from exc

    factory = getattr(module, factory_name, None)
    if factory is None or not callable(factory):
        raise TransportError(
            f"driver module {module_path!r} has no callable {factory_name!r}",
            failure_code="driver_factory_missing",
        )

    options = config.get("options")
    try:
        transport = factory(**dict(options)) if isinstance(options, Mapping) else factory()
    except TypeError as exc:
        raise TransportError(
            f"driver factory {module_path}.{factory_name} rejected its options: {exc}",
            failure_code="driver_factory_signature",
        ) from exc

    if not isinstance(transport, HardwareTransport):
        raise TransportError(
            f"{module_path}.{factory_name} returned {type(transport).__name__}, "
            "which does not satisfy the HardwareTransport protocol",
            failure_code="driver_protocol_mismatch",
        )
    return transport


__all__ = ["build_transport"]
