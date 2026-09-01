"""Host provider: declares the machine LeapFlow runs on as one device.

The host is not a special case in the protocol -- it is a device whose channel set
happens to be discovered rather than written down. Everything downstream (risk
classification, sampling, the reading store, the board) treats it exactly like an
instrument on a bench, which is the whole point: no parallel "system metrics" path
exists to keep in sync.

Two properties are deliberate and load-bearing:

``DISCOVERED`` provenance with no verifier
    Under the default ``deny_write`` policy that demotes every writable channel to
    read-only. Since this provider declares nothing writable anyway, the effect is
    belt-and-braces -- but it means a future host control channel cannot become
    commandable without a human confirming the declaration first.

``halt_supported=False``
    Honest rather than convenient: there is no emergency stop for a CPU. Admission
    (V5) reads this and refuses to expose writable channels, which is the correct
    consequence and needs no exception here.
"""

from __future__ import annotations

import logging
import platform
from typing import Any, Mapping

from leapflow.hardware.context import (
    ContextProvenance,
    ContextSource,
    HardwareContext,
    TransportRef,
)
from leapflow.hardware.host_metrics import DEVICE_CLASS, DEVICE_ID, HostMetrics

logger = logging.getLogger(__name__)


class HostContextProvider:
    """Supplies one context describing this host's resource channels."""

    kind = "host"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self._config = dict(config or {})
        self._metrics = HostMetrics(self._config)

    def discover(self) -> tuple[HardwareContext, ...]:
        """Return the host context, or nothing when no probe is available.

        Never connects to anything: enumeration reads in-process counters, the mount
        table and sensor names. An empty tuple rather than a channel-less context
        when nothing can be read, because admission rejects a context with no
        channels and would report that rejection as a user error -- when the real
        situation is simply an unsupported platform.
        """
        channels = self._metrics.channels()
        if not channels:
            logger.debug("host provider: no readable probes on this platform")
            return ()
        return (
            HardwareContext(
                device_id=DEVICE_ID,
                display_name=_display_name(),
                device_class=DEVICE_CLASS,
                transport=TransportRef(
                    kind="host",
                    config={
                        "sample_interval_s": self._metrics.interval_s,
                        "include": sorted(_as_list(self._config.get("include"))),
                        "exclude": sorted(_as_list(self._config.get("exclude"))),
                    },
                ),
                channels=channels,
                vendor=platform.system(),
                model=platform.machine(),
                location="local",
                halt_supported=False,
                notes=(
                    "Discovered host resources. Collection backends: "
                    f"{', '.join(self._metrics.backends)}."
                ),
                provenance=ContextProvenance(
                    source=ContextSource.DISCOVERED.value,
                    notes=(
                        "Enumerated from this host at startup. Channel set reflects the "
                        "mounts, interfaces and sensors present when discovery ran."
                    ),
                ),
            ),
        )


def _display_name() -> str:
    """Return a human-facing host label that does not leak more than a hostname."""
    node = platform.node().split(".")[0]
    return f"{node or 'localhost'} ({platform.system()})"


def _as_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def build_provider(config: Mapping[str, Any] | None = None) -> HostContextProvider:
    """Factory registered in the provider table."""
    return HostContextProvider(config)


__all__ = ["HostContextProvider", "build_provider"]
