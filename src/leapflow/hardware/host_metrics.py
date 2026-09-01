"""Host resource probes: the one table both the host provider and transport read.

The machine LeapFlow runs on is a device like any other -- it has quantities,
units, and physical ranges -- so it enters through the ordinary Hardware Context
Protocol rather than a parallel "system stats" feature. What makes it different is
only that its channel *set* is discovered at runtime instead of declared: how many
disks are mounted and which thermal sensors exist are facts about this host.

Provider and transport share this module deliberately. Held as two tables they
drifted immediately in every design sketch: the provider would declare a channel
the transport had no reader for, and the board would show a permanently empty
trace with no error anywhere. Here, a probe *is* its reader, so declaring one
without being able to read it is not expressible.

Every dependency is optional. ``psutil`` widens coverage substantially; without it
the table shrinks to what the standard library can answer (load average, disk
usage, cpu count) and the rest of the system behaves as though those channels do
not exist -- which is the honest report, not a zero.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from leapflow.hardware.context import (
    Channel,
    Direction,
    Envelope,
    HardwareEffect,
    PrivacyTier,
    Representation,
)

logger = logging.getLogger(__name__)

try:  # optional: widens coverage from "load average and disk usage" to the full set
    import psutil

    _HAS_PSUTIL = True
except ImportError:  # pragma: no cover - exercised by the degradation test
    psutil = None  # type: ignore[assignment]
    _HAS_PSUTIL = False


DEVICE_ID = "host"
"""Reserved device id. Namespaced channels keep the host one device, not seven.

A host with cpu, memory, four mounts and three sensors would otherwise consume most
of ``hardware.max_devices`` before a single real peripheral was admitted.
"""

DEVICE_CLASS = "compute"

DEFAULT_INTERVAL_S = 5.0
"""Sampling period for fast-moving quantities (cpu, memory, network).

Five seconds, not one: every sample becomes a row in the raw NDJSON segment and a
contribution to a downsample window, and nobody diagnoses a laptop from
second-resolution CPU history. Slower quantities multiply this.
"""

_SLOW_MULTIPLIER = 4
"""Applied to quantities that move on a human timescale (disk, battery, thermal)."""

DEFAULT_MAX_INTERFACES = 3
"""Network interfaces charted by default, busiest first.

A macOS host enumerates around two dozen interfaces -- ``utun*`` tunnels,
``awdl0``, ``anpi*``, ``llw0``, bridges -- and every one of them would become an
asyncio sampling task, a raw NDJSON stream and two more channels competing for the
eight series the board can chart. Ranking by observed traffic keeps the ones that
carry the host's actual network activity without a per-platform name list, which
would be wrong on the first machine that named things differently.
"""

DEFAULT_MAX_CHANNELS = 48
"""Upper bound on the discovered channel set.

A safety valve, not a design target: a host with fifty mounts should degrade to a
partial view rather than start fifty sampling loops. Exceeding it is logged so the
truncation is never silent.
"""

DEFAULT_MAX_MOUNTS = 4
"""Filesystems charted by default.

Applied after de-duplication, so it bounds *distinct* filesystems rather than mount
points.
"""


@dataclass(frozen=True)
class HostProbe:
    """One discovered host quantity, together with the reader that answers it."""

    channel_id: str
    quantity: str
    unit: str
    reader: Callable[[], Any]
    description: str = ""
    min_value: float | None = None
    max_value: float | None = None
    representation: str = Representation.SCALAR.value
    interval_multiplier: int = 1

    def to_channel(self, *, interval_s: float) -> Channel:
        """Return the declared channel for this probe.

        Read-only and ``effect=read`` unconditionally: nothing in this module can
        change the host, and a writable host channel would need an envelope a
        person is accountable for -- which is exactly what a discovered
        declaration cannot supply.
        """
        period = max(0.1, interval_s * max(1, self.interval_multiplier))
        declared = self.min_value is not None or self.max_value is not None
        return Channel(
            channel_id=self.channel_id,
            direction=Direction.READ.value,
            quantity=self.quantity,
            unit=self.unit,
            effect=HardwareEffect.READ.value,
            envelope=Envelope(
                declared=declared,
                min_value=self.min_value,
                max_value=self.max_value,
            ),
            sample_rate_hz=1.0 / period,
            description=self.description,
            representation=self.representation,
            privacy=PrivacyTier.NONE.value,
        )


class HostMetrics:
    """Enumerates and reads this host's resource channels.

    One instance owns the state that rate-derived quantities need (a throughput is
    a difference between two samples, so the previous sample has to live
    somewhere). The provider builds a throwaway instance to enumerate; the
    transport keeps one for the life of the connection.
    """

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        config = config or {}
        self._interval_s = _as_float(config.get("sample_interval_s"), DEFAULT_INTERVAL_S)
        self._include = _as_str_set(config.get("include"))
        self._exclude = _as_str_set(config.get("exclude"))
        self._max_interfaces = _as_int(config.get("max_interfaces"), DEFAULT_MAX_INTERFACES)
        self._max_mounts = _as_int(config.get("max_mounts"), DEFAULT_MAX_MOUNTS)
        self._max_channels = _as_int(config.get("max_channels"), DEFAULT_MAX_CHANNELS)
        self._counters: dict[str, tuple[float, float]] = {}
        self._probes: dict[str, HostProbe] = {}
        self._enumerated_at: float = 0.0

    @property
    def interval_s(self) -> float:
        return self._interval_s

    @property
    def backends(self) -> tuple[str, ...]:
        """Return which collection backends are active, for status reporting."""
        return ("psutil", "stdlib") if _HAS_PSUTIL else ("stdlib",)

    def probes(self) -> tuple[HostProbe, ...]:
        """Return every probe available on this host right now.

        Cheap and local: reads mount tables, interface lists and sensor names, all
        in-process. Deliberately never shells out to a system profiler -- discovery
        runs during daemon boot, and a multi-second subprocess there would make
        startup latency a function of how many peripherals are attached.
        """
        probes: list[HostProbe] = []
        for build in (self._cpu, self._memory, self._disk, self._network, self._battery, self._thermal):
            try:
                probes.extend(build())
            except Exception as exc:  # noqa: BLE001 - one absent subsystem is not a failure
                logger.debug("host metrics: %s enumeration unavailable: %s", build.__name__, exc)
        selected = tuple(p for p in probes if self._selected(p.channel_id))
        if len(selected) > self._max_channels:
            logger.warning(
                "host metrics: %d channels discovered, charting the first %d; "
                "narrow the set with hardware.host_include/exclude",
                len(selected),
                self._max_channels,
            )
            selected = selected[: self._max_channels]
        self._probes = {probe.channel_id: probe for probe in selected}
        self._enumerated_at = time.time()
        return selected

    def channels(self) -> tuple[Channel, ...]:
        """Return the declared channels for every available probe."""
        return tuple(probe.to_channel(interval_s=self._interval_s) for probe in self.probes())

    def read(self, channel_id: str) -> Any:
        """Return the current value of one channel, or None when unavailable.

        None rather than an exception for a probe that momentarily cannot answer (a
        sensor that disappeared, an unmounted volume): the sampling loop treats a
        missing value as a quality problem on that channel, which is the accurate
        report and leaves every other channel running.
        """
        if not self._probes:
            self.probes()
        probe = self._probes.get(str(channel_id))
        if probe is None:
            return None
        try:
            return probe.reader()
        except Exception as exc:  # noqa: BLE001 - a probe must not fail the sampling loop
            logger.debug("host metrics: %s unreadable: %s", channel_id, exc)
            return None

    def known_channels(self) -> frozenset[str]:
        if not self._probes:
            self.probes()
        return frozenset(self._probes)

    # ── Selection ──

    def _selected(self, channel_id: str) -> bool:
        """Apply include/exclude prefixes so a profile can narrow the channel set.

        Prefix matching rather than exact ids: mounts and interfaces are discovered,
        so a person configuring this cannot know their full channel ids in advance
        but can reasonably say ``net`` or ``disk``.
        """
        if self._exclude and any(channel_id.startswith(prefix) for prefix in self._exclude):
            return False
        if self._include:
            return any(channel_id.startswith(prefix) for prefix in self._include)
        return True

    # ── CPU ──

    def _cpu(self) -> list[HostProbe]:
        probes: list[HostProbe] = [
            HostProbe(
                channel_id="cpu.count",
                quantity="processor_count",
                unit="cores",
                reader=lambda: os.cpu_count() or 0,
                description="Logical processors visible to this host.",
                representation=Representation.STATE.value,
                interval_multiplier=_SLOW_MULTIPLIER,
            )
        ]
        if hasattr(os, "getloadavg"):
            probes.append(
                HostProbe(
                    channel_id="cpu.load1",
                    quantity="load_average",
                    unit="runnable",
                    reader=lambda: round(os.getloadavg()[0], 3),
                    description="One-minute load average (runnable threads).",
                    min_value=0.0,
                )
            )
        if _HAS_PSUTIL:
            probes.append(
                HostProbe(
                    channel_id="cpu.utilization",
                    quantity="cpu_utilization",
                    unit="%",
                    # interval=None returns the mean since the previous call, which is
                    # exactly the sampling period. Passing an interval would block the
                    # sampling loop for that long.
                    reader=lambda: round(float(psutil.cpu_percent(interval=None)), 2),
                    description="Processor busy time since the previous sample.",
                    min_value=0.0,
                    max_value=100.0,
                )
            )
        return probes

    # ── Memory ──

    def _memory(self) -> list[HostProbe]:
        if not _HAS_PSUTIL:
            return []
        total = float(psutil.virtual_memory().total)
        probes = [
            HostProbe(
                channel_id="memory.used_percent",
                quantity="memory_utilization",
                unit="%",
                reader=lambda: round(float(psutil.virtual_memory().percent), 2),
                description="Share of physical memory in use.",
                min_value=0.0,
                max_value=100.0,
            ),
            HostProbe(
                channel_id="memory.available_bytes",
                quantity="memory_available",
                unit="B",
                reader=lambda: int(psutil.virtual_memory().available),
                description="Physical memory available without swapping.",
                min_value=0.0,
                max_value=total,
            ),
        ]
        swap_total = float(psutil.swap_memory().total)
        if swap_total > 0:
            probes.append(
                HostProbe(
                    channel_id="memory.swap_used_bytes",
                    quantity="swap_used",
                    unit="B",
                    reader=lambda: int(psutil.swap_memory().used),
                    description="Swap in use; sustained growth means memory pressure.",
                    min_value=0.0,
                    max_value=swap_total,
                )
            )
        return probes

    # ── Disk ──

    def _disk(self) -> list[HostProbe]:
        probes: list[HostProbe] = []
        for mount, usage in _filesystems(self._max_mounts):
            label = _slug(mount)
            total = float(usage.total)
            probes.append(
                HostProbe(
                    channel_id=f"disk.{label}.free_bytes",
                    quantity="disk_free",
                    unit="B",
                    reader=_disk_reader(mount, "free"),
                    description=f"Free capacity on {mount}.",
                    min_value=0.0,
                    max_value=total,
                    interval_multiplier=_SLOW_MULTIPLIER,
                )
            )
            probes.append(
                HostProbe(
                    channel_id=f"disk.{label}.used_percent",
                    quantity="disk_utilization",
                    unit="%",
                    reader=_disk_reader(mount, "percent"),
                    description=f"Share of {mount} consumed.",
                    min_value=0.0,
                    max_value=100.0,
                    interval_multiplier=_SLOW_MULTIPLIER,
                )
            )
        return probes

    # ── Network ──

    def _network(self) -> list[HostProbe]:
        if not _HAS_PSUTIL:
            return []
        probes: list[HostProbe] = []
        for name in self._active_interfaces():
            label = _slug(name)
            for field_name, direction in (("bytes_recv", "rx"), ("bytes_sent", "tx")):
                probes.append(
                    HostProbe(
                        channel_id=f"net.{label}.{direction}_bytes_per_s",
                        quantity=f"network_{direction}_throughput",
                        unit="B/s",
                        reader=self._throughput_reader(name, field_name),
                        description=f"{direction.upper()} throughput on {name}.",
                        min_value=0.0,
                    )
                )
        return probes

    def _active_interfaces(self) -> tuple[str, ...]:
        """Return the busiest real interfaces, at most ``max_interfaces`` of them.

        Three filters, each earning its place: an interface that is down produces a
        flat zero trace forever; a loopback interface measures the host talking to
        itself; and one that has never carried a byte is configured but unused. What
        survives is ranked by observed traffic, so the selection follows what the
        machine actually does rather than what it happens to have enumerated.
        """
        counters = psutil.net_io_counters(pernic=True)
        stats = psutil.net_if_stats()
        ranked: list[tuple[float, str]] = []
        for name, counter in counters.items():
            stat = stats.get(name)
            if not getattr(stat, "isup", False) or _is_loopback(name, stat):
                continue
            total = float(getattr(counter, "bytes_recv", 0) + getattr(counter, "bytes_sent", 0))
            if total <= 0:
                continue
            ranked.append((total, name))
        # Name breaks ties so the channel set is stable across rediscovery on an idle
        # host; an unstable set would orphan and restart sampling loops for no reason.
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return tuple(name for _total, name in ranked[: self._max_interfaces])

    def _throughput_reader(self, nic: str, field_name: str) -> Callable[[], Any]:
        """Return a reader that differentiates a monotonic byte counter.

        The counter itself is useless on a chart -- it only ever rises, so every
        trace looks like the same straight line regardless of what the link is
        doing. The first call has no predecessor and reports None rather than a
        fabricated zero, which would read as "no traffic".
        """
        key = f"{nic}:{field_name}"

        def _read() -> Any:
            counters = psutil.net_io_counters(pernic=True).get(nic)
            if counters is None:
                return None
            total = float(getattr(counters, field_name, 0.0))
            now = time.monotonic()
            previous = self._counters.get(key)
            self._counters[key] = (total, now)
            if previous is None:
                return None
            last_total, last_at = previous
            elapsed = now - last_at
            if elapsed <= 0.0:
                return None
            # A counter reset (interface reconfigured, wrap) would otherwise produce a
            # large negative rate that no link ever achieved.
            delta = total - last_total
            return round(delta / elapsed, 2) if delta >= 0 else None

        return _read

    # ── Battery and thermal ──

    def _battery(self) -> list[HostProbe]:
        if not _HAS_PSUTIL or getattr(psutil, "sensors_battery", None) is None:
            return []
        if psutil.sensors_battery() is None:
            return []
        return [
            HostProbe(
                channel_id="battery.percent",
                quantity="battery_charge",
                unit="%",
                reader=lambda: _battery_field("percent"),
                description="Remaining battery charge.",
                min_value=0.0,
                max_value=100.0,
                interval_multiplier=_SLOW_MULTIPLIER,
            ),
            HostProbe(
                channel_id="battery.power_plugged",
                quantity="power_source",
                unit="",
                reader=lambda: _battery_field("power_plugged"),
                description="Whether the host is on external power.",
                representation=Representation.STATE.value,
                interval_multiplier=_SLOW_MULTIPLIER,
            ),
        ]

    def _thermal(self) -> list[HostProbe]:
        if not _HAS_PSUTIL or getattr(psutil, "sensors_temperatures", None) is None:
            return []
        readings = psutil.sensors_temperatures() or {}
        probes: list[HostProbe] = []
        for group in sorted(readings):
            for index, entry in enumerate(readings[group]):
                label = _slug(getattr(entry, "label", "") or f"{group}_{index}")
                # The vendor's own critical threshold is the only defensible upper
                # bound; inventing one would make every breach event a guess.
                critical = getattr(entry, "critical", None)
                probes.append(
                    HostProbe(
                        channel_id=f"thermal.{_slug(group)}.{label}.celsius",
                        quantity="temperature",
                        unit="degC",
                        reader=_thermal_reader(group, index),
                        description=f"{group} sensor {label}.",
                        min_value=0.0,
                        max_value=float(critical) if critical else None,
                        interval_multiplier=2,
                    )
                )
        return probes


# ── Module helpers ──


def _disk_reader(mount: str, field_name: str) -> Callable[[], Any]:
    def _read() -> Any:
        usage = shutil.disk_usage(mount)
        if field_name == "free":
            return int(usage.free)
        if usage.total <= 0:
            return None
        return round(usage.used / usage.total * 100.0, 2)

    return _read


def _thermal_reader(group: str, index: int) -> Callable[[], Any]:
    def _read() -> Any:
        entries = (psutil.sensors_temperatures() or {}).get(group) or ()
        if index >= len(entries):
            return None
        return round(float(entries[index].current), 2)

    return _read


def _battery_field(name: str) -> Any:
    """Return one field of the battery reading, or None when there is no battery."""
    battery = psutil.sensors_battery()
    if battery is None:
        return None
    value = getattr(battery, name, None)
    if name == "percent" and isinstance(value, (int, float)):
        return round(float(value), 2)
    return value


def _is_loopback(name: str, stat: Any) -> bool:
    """Return whether an interface only carries traffic to the host itself.

    Prefers the interface flags psutil exposes on Linux and macOS; falls back to the
    POSIX loopback naming convention, which is the one interface name that is
    genuinely portable.
    """
    flags = str(getattr(stat, "flags", "") or "")
    if "loopback" in flags.lower():
        return True
    return name in {"lo", "lo0"}


def _filesystems(limit: int) -> tuple[tuple[str, Any], ...]:
    """Return distinct writable filesystems as ``(mountpoint, usage)``, capped at *limit*.

    De-duplicated by observed capacity, which is what makes this usable on macOS: an
    APFS container presents ``Data``, ``Preboot``, ``Update``, ``VM`` and several
    helper volumes as separate mounts that all report the *same* total and free
    bytes, because the free space belongs to the container rather than to any one
    volume. Charting all of them repeats a single trace under eight names and buries
    every other channel on the board.

    Two filesystems having byte-identical total *and* free capacity at the same
    instant is possible in principle; the cost of that collision is one fewer
    redundant-looking row, which is why capacity is preferred over a per-platform
    volume-name list that would be wrong on the next OS release.

    Within a group the shallowest, alphabetically-first mount point wins, so the
    selection is stable across rediscovery -- an unstable choice would stop and
    restart sampling loops for no reason.
    """
    groups: dict[tuple[int, int], list[str]] = {}
    for mount in _mount_points():
        try:
            usage = shutil.disk_usage(mount)
        except OSError:
            continue
        if usage.total <= 0:
            continue
        groups.setdefault((usage.total, usage.free), []).append(mount)

    chosen: list[tuple[str, Any]] = []
    for signature, mounts in groups.items():
        mounts.sort(key=lambda path: (path.count(os.sep), path))
        keeper = mounts[0]
        if len(mounts) > 1:
            logger.debug(
                "host metrics: mounts %s share one filesystem (%d bytes); charting %s",
                mounts,
                signature[0],
                keeper,
            )
        chosen.append((keeper, shutil.disk_usage(keeper)))

    # Largest first: the volume holding the user's data is the one worth a chart, and
    # it is reliably the biggest among a container's helper volumes.
    chosen.sort(key=lambda item: (-item[1].total, item[0]))
    return tuple(chosen[:limit])


def _mount_points() -> tuple[str, ...]:
    """Return the mount points worth charting.

    Physical, writable filesystems only. Every read-only system snapshot, every
    container overlay and every disk image mount is a fixed-size volume whose free
    space never changes, so charting them adds rows and no information.
    """
    if _HAS_PSUTIL:
        mounts: list[str] = []
        for part in psutil.disk_partitions(all=False):
            options = str(getattr(part, "opts", ""))
            if "ro" in options.split(","):
                continue
            mounts.append(part.mountpoint)
        if mounts:
            return tuple(sorted(set(mounts)))
    return (os.path.abspath(os.sep),)


def _slug(value: str) -> str:
    """Normalise a discovered name into a channel-id fragment.

    Channel ids appear in cache keys, DuckDB rows and board paths, so a mount point
    like ``/System/Volumes/Data`` cannot be used verbatim. Dots are the channel
    namespace separator and must not survive inside a fragment.
    """
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(value).strip().lower())
    collapsed = "_".join(part for part in cleaned.split("_") if part)
    return collapsed or "root"


def _as_float(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if result > 0 else default


def _as_int(value: Any, default: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError):
        return default
    return result if result > 0 else default


def _as_str_set(value: Any) -> frozenset[str]:
    if isinstance(value, str):
        items = value.split(",")
    elif isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
    else:
        return frozenset()
    return frozenset(str(item).strip() for item in items if str(item).strip())


__all__ = [
    "DEFAULT_INTERVAL_S",
    "DEFAULT_MAX_CHANNELS",
    "DEFAULT_MAX_INTERFACES",
    "DEFAULT_MAX_MOUNTS",
    "DEVICE_CLASS",
    "DEVICE_ID",
    "HostMetrics",
    "HostProbe",
]
