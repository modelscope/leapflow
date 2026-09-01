"""Physical-signal observability: the board's view of the bench.

Four files, four reasons to change: ``series`` when the payload shape changes,
``digest`` when the cycle analysis changes, ``inventory`` when the on-demand device
view changes, ``producer`` when the runtime wiring changes. Nothing here samples or
writes to a device -- the registry does that, and an observation surface that could
actuate would not be one.

``digest`` and ``inventory`` answer different questions on purpose. The digest is a
byte-capped cycle payload pushed on the monitor cadence and capped at eight charted
channels; the inventory answers "what is attached" and "what is this one device
doing" for somebody who just clicked on it.
"""

from leapflow.hardware.observability.digest import ALERT_KINDS, build_digest
from leapflow.hardware.observability.exporter import (
    HardwareMetricsExporter,
    MetricSample,
    build_exporter,
)
from leapflow.hardware.observability.inventory import build_device_view, build_inventory
from leapflow.hardware.observability.producer import DOMAIN, HardwareObservationProducer
from leapflow.hardware.observability.series import (
    MAX_PAYLOAD_BYTES,
    MAX_POINTS,
    MAX_SERIES,
    SERIES_SCHEMA_VERSION,
    WALL_CLOCK,
    ChannelSeries,
    EnvelopeBand,
    HardwareDigest,
    SeriesPoint,
)

__all__ = [
    "ALERT_KINDS",
    "DOMAIN",
    "HardwareMetricsExporter",
    "MAX_PAYLOAD_BYTES",
    "MAX_POINTS",
    "MAX_SERIES",
    "MetricSample",
    "SERIES_SCHEMA_VERSION",
    "WALL_CLOCK",
    "ChannelSeries",
    "EnvelopeBand",
    "HardwareDigest",
    "HardwareObservationProducer",
    "SeriesPoint",
    "build_device_view",
    "build_digest",
    "build_exporter",
    "build_inventory",
]
