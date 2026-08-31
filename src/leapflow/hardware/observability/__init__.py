"""Physical-signal observability: the board's view of the bench.

Three files, three reasons to change: ``series`` when the payload shape changes,
``digest`` when the analysis changes, ``producer`` when the runtime wiring changes.
Nothing here samples or writes to a device -- the registry does that, and an
observation surface that could actuate would not be one.
"""

from leapflow.hardware.observability.digest import build_digest
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
    "DOMAIN",
    "MAX_PAYLOAD_BYTES",
    "MAX_POINTS",
    "MAX_SERIES",
    "SERIES_SCHEMA_VERSION",
    "WALL_CLOCK",
    "ChannelSeries",
    "EnvelopeBand",
    "HardwareDigest",
    "HardwareObservationProducer",
    "SeriesPoint",
    "build_digest",
]
