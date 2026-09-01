"""Hardware registry: providers in, admitted contexts and transports out.

Structurally the same shape as ``ToolPluginRegistry`` -- discover, validate,
assemble -- because the problem is the same and the philosophy has already been
settled: one declaration is the single source of truth, and everything else is
derived from it deterministically.

Admission is fail-closed and returns a structured report rather than raising. One
malformed declaration must not make every other device in the profile disappear,
and a device that fails a *write* precondition is demoted to read-only rather than
removed, because reads stay valuable for diagnosis exactly when something is wrong.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Mapping, Sequence

from leapflow.hardware.context import (
    SUPPORTED_HC_VERSIONS,
    HardwareContext,
)
from leapflow.hardware.providers import (
    HardwareContextProvider,
    ProviderError,
    build_provider,
)
from leapflow.hardware.transport import HardwareTransport, TransportError
from leapflow.hardware.transports import available_transports, build_transport

logger = logging.getLogger(__name__)

_DEVICE_ID_ALLOWED = set("abcdefghijklmnopqrstuvwxyz0123456789_")


class UnverifiedContextPolicy:
    """How to treat a context no human has confirmed."""

    DENY_WRITE = "deny_write"
    PROMPT = "prompt"
    ALLOW = "allow"


@dataclass(frozen=True)
class HardwareSettings:
    """Runtime policy for the hardware subsystem.

    Passive local discovery starts by default: host metrics and media enumeration give the
    board an honest inventory without opening a camera, a microphone, a radio or a network
    connection. Observation remains gated per declared privacy tier, and explicitly
    disabling this setting remains a hard disable.
    """

    enabled: bool = True
    providers: tuple[tuple[str, Mapping[str, Any]], ...] = ()
    max_devices: int = 16
    unverified_context_policy: str = UnverifiedContextPolicy.DENY_WRITE
    require_describe_before_write: bool = True
    envelope_grant: bool = True
    trust_skip_enabled: bool = False
    stream_enabled: bool = True
    stream_ring_capacity: int = 4096
    persist_readings: bool = True
    downsample_interval_s: float = 60.0
    raw_retention_days: float = 7.0
    history_retention_days: float = 90.0
    raw_segment_mb: float = 32.0
    reading_store_sensitive: bool = True
    readings_dir: str = ""
    instrument_db_path: str = ""
    workspace_id: str = ""
    rediscover_interval_s: float = 0.0
    preview_idle_timeout_s: float = 15.0
    # Hard ceilings. The page's balanced default is lower (960px / 8fps / JPEG 75); it
    # may select any profile within these bounds but cannot make a durable config edit or
    # exceed them.
    preview_max_fps: float = 12.0
    preview_max_width: int = 1280
    preview_quality: int = 85

    @property
    def denies_unverified_writes(self) -> bool:
        return self.unverified_context_policy == UnverifiedContextPolicy.DENY_WRITE

    @classmethod
    def from_settings(cls, settings: Any) -> "HardwareSettings":
        """Derive runtime policy from the loaded ``Settings``.

        The declarations directory defaults to the active profile's, resolved through
        the layout rather than assembled here: a managed path must be declared by a
        layout object, never joined together at the point of use.
        """
        profile_layout = getattr(settings, "profile_layout", None)
        configured = str(getattr(settings, "hardware_devices_dir", "") or "").strip()
        if configured:
            devices_dir = Path(configured).expanduser()
            verified_path = devices_dir.parent / "verified.json"
        elif profile_layout is not None:
            devices_dir = profile_layout.hardware.devices_dir
            verified_path = profile_layout.hardware.verified_path
        else:
            # Tests and small in-process callers may deliberately provide no profile
            # layout. They still own the explicit enable/disable decision; returning the
            # class default here would silently re-enable a subsystem a caller set False.
            return cls(enabled=bool(getattr(settings, "hardware_enabled", True)))
        return cls(
            enabled=bool(getattr(settings, "hardware_enabled", True)),
            providers=_provider_rows(
                settings, devices_dir=devices_dir, verified_path=verified_path
            ),
            max_devices=int(getattr(settings, "hardware_max_devices", 16) or 16),
            unverified_context_policy=str(
                getattr(settings, "hardware_unverified_policy", UnverifiedContextPolicy.DENY_WRITE)
                or UnverifiedContextPolicy.DENY_WRITE
            ),
            require_describe_before_write=bool(
                getattr(settings, "hardware_require_describe", True)
            ),
            envelope_grant=bool(getattr(settings, "hardware_envelope_grant", True)),
            trust_skip_enabled=bool(
                getattr(settings, "hardware_trust_skip_enabled", False)
            ),
            stream_enabled=bool(getattr(settings, "hardware_stream_enabled", True)),
            stream_ring_capacity=int(
                getattr(settings, "hardware_stream_ring_capacity", 4096) or 4096
            ),
            persist_readings=bool(getattr(settings, "hardware_persist_readings", True)),
            downsample_interval_s=float(
                getattr(settings, "hardware_downsample_interval_s", 60.0) or 60.0
            ),
            raw_retention_days=float(
                getattr(settings, "hardware_raw_retention_days", 7.0) or 7.0
            ),
            history_retention_days=float(
                getattr(settings, "hardware_history_retention_days", 90.0) or 90.0
            ),
            raw_segment_mb=float(getattr(settings, "hardware_raw_segment_mb", 32.0) or 32.0),
            reading_store_sensitive=bool(
                getattr(settings, "hardware_reading_store_sensitive", True)
            ),
            instrument_db_path=(
                str(profile_layout.instrument_db_path) if profile_layout is not None else ""
            ),
            rediscover_interval_s=float(
                getattr(settings, "hardware_rediscover_interval_s", 0.0) or 0.0
            ),
            preview_idle_timeout_s=float(
                getattr(settings, "hardware_preview_idle_timeout_s", 15.0) or 15.0
            ),
            preview_max_fps=float(
                getattr(settings, "hardware_preview_max_fps", 12.0) or 12.0
            ),
            preview_max_width=int(
                getattr(settings, "hardware_preview_max_width", 1280) or 1280
            ),
            preview_quality=int(getattr(settings, "hardware_preview_quality", 85) or 85),
        )


DEFAULT_PROVIDER_KINDS = ("yaml", "host", "media")
"""Discovery sources enabled when hardware is on and nothing else is configured.

``yaml`` supplies declarations a person wrote and is accountable for. ``host`` costs
nothing to enumerate, needs no permission, and answers the question every operator
asks first -- what is this machine doing. ``media`` lists cameras and microphones
without opening any of them, which is why it can be on by default: enumeration needs
no consent, and the reads that *do* disclose something are refused until a grant
exists. Discovering a camera and disclosing what it sees are different acts.

Scanners that transmit (Bluetooth) or leave the host (mDNS) are deliberately absent:
discovery should never be the reason a radio starts up or a packet leaves the machine.
"""


def _provider_rows(
    settings: Any, *, devices_dir: Path, verified_path: Path
) -> tuple[tuple[str, Mapping[str, Any]], ...]:
    """Build the ``(kind, config)`` rows for every configured provider.

    Config-driven rather than a fixed row, so adding a discovery source is a
    configuration change and an out-of-tree provider needs no core edit. Each kind
    receives only the config it understands; an unknown kind is passed through with
    an empty mapping and reported by ``build_provider`` as an admission note rather
    than crashing the load.
    """
    raw = getattr(settings, "hardware_providers", "") or ""
    if isinstance(raw, str):
        kinds = [item.strip() for item in raw.split(",") if item.strip()]
    else:
        kinds = [str(item).strip() for item in raw if str(item).strip()]
    if not kinds:
        kinds = list(DEFAULT_PROVIDER_KINDS)

    configs: dict[str, Mapping[str, Any]] = {
        "yaml": {"devices_dir": str(devices_dir), "verified_path": str(verified_path)},
        "host": {
            "sample_interval_s": float(
                getattr(settings, "hardware_host_interval_s", 0.0) or 0.0
            ),
            "include": str(getattr(settings, "hardware_host_include", "") or ""),
            "exclude": str(getattr(settings, "hardware_host_exclude", "") or ""),
        },
        "media": {
            "include_screens": bool(getattr(settings, "hardware_media_screens", False)),
            "include_microphones": bool(
                getattr(settings, "hardware_media_microphones", True)
            ),
            "max_fps": float(getattr(settings, "hardware_preview_max_fps", 0.0) or 0.0),
        },
    }
    # Deduplicated while preserving order: two rows of one kind would discover every
    # device twice and have the second copy rejected as a duplicate device_id, which
    # reads as a declaration error rather than a configuration one.
    seen: set[str] = set()
    rows: list[tuple[str, Mapping[str, Any]]] = []
    for kind in kinds:
        if kind in seen:
            continue
        seen.add(kind)
        rows.append((kind, configs.get(kind, {})))
    return tuple(rows)


def build_registry(settings: Any) -> HardwareRegistry | None:
    """Return a loaded registry, or None when hardware is disabled.

    Returning None rather than an empty registry is the whole default-off contract:
    callers use it to decide whether to compose a hardware risk classifier at all, so
    a disabled profile keeps the unmodified ``DefaultRiskClassifier`` and behaves
    byte-for-byte as it did before this subsystem existed.

    Never raises: a malformed declaration directory must not prevent the process from
    starting. Failures are reported through the returned registry's load report.
    """
    policy = HardwareSettings.from_settings(settings)
    if not policy.enabled:
        return None
    registry = HardwareRegistry(policy)
    try:
        report = registry.load()
    except (OSError, ValueError) as exc:
        logger.error("Hardware registry failed to load: %s", exc, exc_info=True)
        return registry
    for note in report.notes:
        logger.warning(
            "Hardware admission %s device=%s rule=%s: %s",
            note.outcome,
            note.device_id or "-",
            note.rule,
            note.detail,
        )
    return registry


@dataclass(frozen=True)
class AdmissionNote:
    """One admission decision, reported rather than logged and forgotten."""

    device_id: str
    rule: str
    outcome: str  # "rejected" | "demoted" | "warning"
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {
            "device_id": self.device_id,
            "rule": self.rule,
            "outcome": self.outcome,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class LoadReport:
    """Outcome of a registry load pass."""

    admitted: tuple[str, ...] = ()
    rejected: tuple[str, ...] = ()
    notes: tuple[AdmissionNote, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "admitted": list(self.admitted),
            "rejected": list(self.rejected),
            "notes": [note.to_dict() for note in self.notes],
        }


class HardwareRegistry:
    """Owns admitted hardware contexts and their lazily opened transports."""

    def __init__(
        self,
        settings: HardwareSettings | None = None,
        *,
        providers: Sequence[HardwareContextProvider] = (),
    ) -> None:
        self._settings = settings or HardwareSettings()
        self._explicit_providers = tuple(providers)
        self._contexts: dict[str, HardwareContext] = {}
        self._transports: dict[str, HardwareTransport] = {}
        self._open_locks: dict[str, asyncio.Lock] = {}
        self._io_locks: dict[str, asyncio.Lock] = {}
        self._report = LoadReport()
        self._described: set[tuple[str, str]] = set()
        self._last_command: dict[tuple[str, str], tuple[float, float]] = {}
        self._stream_sources: tuple[Any, ...] | None = None
        self._reading_store: Any = None
        self._calibration_store: Any = None
        self._preview_broker: Any = None
        # One holder for instrument.duckdb, shared by the reading and calibration stores.
        # A single process must not open two independent read-write connections to the
        # same DuckDB file, so the registry owns the holder and injects it into both.
        self._instrument_conn: Any = None
        self._outcome_recorder: Any = None
        self._cache_manager: Any = None
        self._session_id: str = ""
        # Bounded on purpose: an unbounded event log on a long-running bench is a leak
        # with a schedule, and hw_status only ever shows a recent tail.
        self._recent_events: Deque[Any] = deque(maxlen=200)
        self._event_emitter: Any = None

    # ── Loading ──

    @property
    def settings(self) -> HardwareSettings:
        return self._settings

    @property
    def report(self) -> LoadReport:
        return self._report

    def load(self) -> LoadReport:
        """Run every configured provider, apply admission rules, and store results.

        Idempotent: calling it again re-runs discovery and replaces the admitted
        set, which is what makes a declaration edit visible without a restart.
        """
        self._contexts.clear()
        # Discarded so the next request rebuilds them against the new declarations; a
        # cached source would keep sampling a channel that no longer exists.
        self._stream_sources = None
        if not self._settings.enabled:
            self._report = LoadReport()
            return self._report

        discovered: list[HardwareContext] = []
        notes: list[AdmissionNote] = []
        for provider in self._resolve_providers(notes):
            try:
                discovered.extend(provider.discover())
            except (ProviderError, OSError, ValueError) as exc:
                logger.warning(
                    "Hardware provider %r discovery failed: %s",
                    getattr(provider, "kind", "?"),
                    exc,
                    exc_info=True,
                )
                notes.append(
                    AdmissionNote(
                        device_id="",
                        rule="provider",
                        outcome="warning",
                        detail=f"provider {getattr(provider, 'kind', '?')!r} failed: {exc}",
                    )
                )

        admitted: list[str] = []
        rejected: list[str] = []
        for context in discovered:
            verdict = self._admit(context, notes)
            if verdict is None:
                rejected.append(context.device_id or "<unnamed>")
                continue
            if len(admitted) >= self._settings.max_devices:
                notes.append(
                    AdmissionNote(
                        device_id=verdict.device_id,
                        rule="V8",
                        outcome="rejected",
                        detail=f"exceeds max_devices={self._settings.max_devices}",
                    )
                )
                rejected.append(verdict.device_id)
                continue
            self._contexts[verdict.device_id] = verdict
            admitted.append(verdict.device_id)

        self._report = LoadReport(
            admitted=tuple(admitted), rejected=tuple(rejected), notes=tuple(notes)
        )
        logger.info(
            "Hardware registry loaded: %d admitted, %d rejected", len(admitted), len(rejected)
        )
        return self._report

    def _resolve_providers(self, notes: list[AdmissionNote]) -> tuple[HardwareContextProvider, ...]:
        if self._explicit_providers:
            return self._explicit_providers
        resolved: list[HardwareContextProvider] = []
        for kind, config in self._settings.providers:
            try:
                resolved.append(build_provider(kind, config))
            except ProviderError as exc:
                notes.append(
                    AdmissionNote(
                        device_id="",
                        rule="provider",
                        outcome="warning",
                        detail=f"provider {kind!r} unavailable: {exc}",
                    )
                )
        return tuple(resolved)

    async def reconcile(self) -> LoadReport:
        """Re-run discovery and converge on the new device set without disrupting the old.

        The hot-plug path, and deliberately *not* ``load()``. ``load()`` clears the
        admitted contexts and discards ``_stream_sources`` while the sampling tasks
        those objects own are still running: the tasks keep reading, keep writing to
        the store, and are no longer the objects ``stop_streams()`` can stop. One
        rediscovery would leak a task per streaming channel, every time.

        So the difference is computed and the old sources are captured before the
        reload. Devices that vanished have their transports closed; a device whose
        declaration is byte-identical keeps running untouched; only new or altered
        devices cause a restart. Comparing whole contexts is what catches an edited
        envelope -- the event detector holds the old bounds, so it must be rebuilt.

        Never raises: rediscovery runs on a schedule and from an RPC, and a scanner
        that fails must leave the previous fleet running.
        """
        if not self._settings.enabled:
            return self._report

        previous = dict(self._contexts)
        # Captured before load() drops the attribute: without this reference the only
        # thing still holding the running tasks would be the event loop.
        retired = self._stream_sources or ()
        was_streaming = self._stream_sources is not None

        report = self.load()

        current = dict(self._contexts)
        removed = sorted(set(previous) - set(current))
        added = sorted(set(current) - set(previous))
        altered = sorted(
            device_id
            for device_id in set(previous) & set(current)
            if previous[device_id] != current[device_id]
        )

        if not (removed or added or altered):
            # Nothing changed, so the discarded source tuple has to be put back or the
            # next call to stream_sources() would build a second set of loops beside
            # the ones already running.
            self._stream_sources = retired or self._stream_sources
            return report

        for device_id in (*removed, *altered):
            # An altered device's transport was configured from a declaration that no
            # longer exists, so it is dropped and rebuilt on next use.
            await self.drop_transport(device_id)

        if was_streaming:
            await self._stop_sources(retired)
            await self.start_streams()

        logger.info(
            "Hardware reconcile: +%d -%d ~%d device(s)", len(added), len(removed), len(altered)
        )
        return report

    @staticmethod
    async def _stop_sources(sources: Sequence[Any]) -> None:
        """Stop the given sampling loops, isolating failures. Never raises."""
        for source in sources:
            try:
                await source.stop()
            except Exception as exc:  # noqa: BLE001 - teardown must not propagate
                logger.warning(
                    "Hardware stream %s stop failed during reconcile: %s",
                    getattr(source, "source_id", "?"),
                    exc,
                    exc_info=True,
                )

    # ── Admission rules ──

    def _admit(
        self, context: HardwareContext, notes: list[AdmissionNote]
    ) -> HardwareContext | None:
        """Apply V1-V7 to *context*, returning the admitted form or None.

        Rules that concern the ability to command a device demote it to read-only;
        rules that concern whether the declaration can be understood at all reject
        it. The distinction matters: an unusable declaration is a user error to
        report, while an untrusted device is still worth observing.
        """
        device_id = context.device_id

        # V2 -- identity must be usable as a stable key and safe in a path.
        if not device_id or any(ch not in _DEVICE_ID_ALLOWED for ch in device_id):
            notes.append(
                AdmissionNote(
                    device_id=device_id or "<unnamed>",
                    rule="V2",
                    outcome="rejected",
                    detail="device_id must be non-empty and match [a-z0-9_]+",
                )
            )
            return None
        if device_id in self._contexts:
            notes.append(
                AdmissionNote(
                    device_id=device_id,
                    rule="V2",
                    outcome="rejected",
                    detail="duplicate device_id",
                )
            )
            return None

        # V1 -- an unknown protocol version is refused, never migrated silently.
        if context.hc_version not in SUPPORTED_HC_VERSIONS:
            notes.append(
                AdmissionNote(
                    device_id=device_id,
                    rule="V1",
                    outcome="rejected",
                    detail=(
                        f"unsupported hc_version {context.hc_version!r}; "
                        f"supported: {', '.join(sorted(SUPPORTED_HC_VERSIONS))}"
                    ),
                )
            )
            return None

        # V4 -- a transport we cannot build makes the device unusable entirely.
        if context.transport.kind not in available_transports():
            notes.append(
                AdmissionNote(
                    device_id=device_id,
                    rule="V4",
                    outcome="rejected",
                    detail=(
                        f"unknown transport kind {context.transport.kind!r}; "
                        f"available: {', '.join(available_transports())}"
                    ),
                )
            )
            return None

        if not context.channels:
            notes.append(
                AdmissionNote(
                    device_id=device_id,
                    rule="V2",
                    outcome="rejected",
                    detail="declaration has no channels",
                )
            )
            return None

        channels = list(context.channels)
        seen: set[str] = set()
        for index, channel in enumerate(channels):
            if not channel.channel_id or channel.channel_id in seen:
                notes.append(
                    AdmissionNote(
                        device_id=device_id,
                        rule="V2",
                        outcome="rejected",
                        detail=f"channel #{index} has a missing or duplicate channel_id",
                    )
                )
                return None
            seen.add(channel.channel_id)

        # V5 -- a device that cannot be stopped may not be commanded at all.
        if context.halt_supported is False and any(c.is_writable for c in channels):
            notes.append(
                AdmissionNote(
                    device_id=device_id,
                    rule="V5",
                    outcome="demoted",
                    detail="halt_supported is false; all writable channels demoted to read-only",
                )
            )
            channels = [c.without_write() for c in channels]

        # V7 -- an unconfirmed context cannot authorize a physical change.
        if (
            self._settings.denies_unverified_writes
            and not context.provenance.is_verified
            and any(c.is_writable for c in channels)
        ):
            notes.append(
                AdmissionNote(
                    device_id=device_id,
                    rule="V7",
                    outcome="demoted",
                    detail=(
                        "context is not verified by a human; writable channels demoted "
                        "to read-only (policy unverified_context_policy=deny_write)"
                    ),
                )
            )
            channels = [c.without_write() for c in channels]

        # V3 -- an undeclared envelope is not an unbounded one.
        for index, channel in enumerate(channels):
            if channel.is_writable and not channel.envelope.declared:
                notes.append(
                    AdmissionNote(
                        device_id=device_id,
                        rule="V3",
                        outcome="demoted",
                        detail=(
                            f"channel {channel.channel_id!r} is writable without a declared "
                            "envelope; demoted to read-only"
                        ),
                    )
                )
                channels[index] = channel.without_write()

        # V9 -- a frame channel is an observation, never a command.
        #
        # Its declared rate is deliberately left alone: on a media channel the rate is
        # a capture ceiling rather than a sampling cadence, and ``Channel.is_streaming``
        # already excludes media, so no sampling loop is built for it. Zeroing it here
        # would destroy the ceiling the preview path needs.
        for index, channel in enumerate(channels):
            if channel.is_media and channel.is_writable:
                notes.append(
                    AdmissionNote(
                        device_id=device_id,
                        rule="V9",
                        outcome="demoted",
                        detail=(
                            f"channel {channel.channel_id!r} carries frames and cannot be "
                            "written; demoted to read-only"
                        ),
                    )
                )
                channels[index] = channel.without_write()

        # V10 -- a read that discloses the environment must say so before it happens.
        for channel in channels:
            if channel.is_privacy_gated:
                notes.append(
                    AdmissionNote(
                        device_id=device_id,
                        rule="V10",
                        outcome="warning",
                        detail=(
                            f"channel {channel.channel_id!r} declares privacy "
                            f"{channel.privacy!r}; reads require consent before the first "
                            "observation. Start its preview on LeapBoard and answer the "
                            "in-panel prompt, or run `/board preview <device>` in the TUI. "
                            "A session grant then covers the continuous preview."
                        ),
                    )
                )

        # V6 -- an interlock we cannot evaluate must block, not be ignored.
        readable = {c.channel_id for c in channels if c.is_readable}
        broken = {
            lock.interlock_id
            for lock in context.interlocks
            if lock.channel_id not in readable or not lock.interlock_id
        }
        for lock_id in sorted(broken):
            notes.append(
                AdmissionNote(
                    device_id=device_id,
                    rule="V6",
                    outcome="warning",
                    detail=(
                        f"interlock {lock_id!r} references a channel that is not readable; "
                        "it will evaluate as unsatisfied and hardline-deny guarded writes"
                    ),
                )
            )
        for index, channel in enumerate(channels):
            required = set(channel.envelope.requires_interlocks)
            missing = {
                name
                for name in required
                if context.interlock(name) is None
            }
            if channel.is_writable and (missing or (required & broken)):
                notes.append(
                    AdmissionNote(
                        device_id=device_id,
                        rule="V6",
                        outcome="warning",
                        detail=(
                            f"channel {channel.channel_id!r} requires interlocks that cannot be "
                            f"evaluated ({', '.join(sorted(missing | (required & broken)))}); "
                            "writes will be hardline-denied"
                        ),
                    )
                )
                # Left writable on purpose: the classifier denies with a specific,
                # repairable reason, which teaches more than the channel vanishing.

        return context.with_channels(tuple(channels))

    # ── Access ──

    def contexts(self) -> tuple[HardwareContext, ...]:
        return tuple(self._contexts[key] for key in sorted(self._contexts))

    def context(self, device_id: str) -> HardwareContext | None:
        return self._contexts.get(str(device_id))

    def channel_is_writable(self, device_id: str, channel_id: str) -> bool:
        context = self.context(device_id)
        channel = context.channel(channel_id) if context is not None else None
        return bool(channel is not None and channel.is_writable)

    async def transport(self, device_id: str) -> HardwareTransport:
        """Return the open transport for *device_id*, opening it on first use.

        Lazy on purpose: ``load()`` must not touch hardware, so the connection is
        established the first time an operation actually needs it.

        Guarded by a per-device lock because two concurrent first calls would
        otherwise each build and open a transport, leaving one connected but
        unreferenced -- a leaked serial port or socket that nothing will ever
        close. ``open()`` being idempotent does not help here: the idempotence is
        per instance, and this race produces two instances.
        """
        context = self.context(device_id)
        if context is None:
            raise TransportError(
                f"unknown device {device_id!r}", failure_code="unknown_device"
            )
        existing = self._transports.get(device_id)
        if existing is not None:
            return existing
        lock = self._open_locks.setdefault(device_id, asyncio.Lock())
        async with lock:
            # Re-check under the lock: another coroutine may have opened it while
            # this one waited.
            existing = self._transports.get(device_id)
            if existing is not None:
                return existing
            transport = build_transport(context.transport.kind, context.transport.config)
            status = await transport.open(context)
            if not status.connected:
                raise TransportError(
                    f"transport for {device_id!r} did not connect: {status.detail}",
                    failure_code="transport_open_failed",
                )
            self._transports[device_id] = transport
            return transport

    def opened_devices(self) -> tuple[str, ...]:
        return tuple(sorted(self._transports))

    # ── Streaming ──

    def stream_sources(self) -> tuple[Any, ...]:
        """Return one signal source per streaming channel, building them once.

        Built lazily and cached because the manager they register with rejects
        registration after it has started: handing out fresh instances on a second call
        would leave the manager driving sources nobody else can see.
        """
        if not self._settings.stream_enabled:
            return ()
        if self._stream_sources is None:
            from leapflow.hardware.stream import build_stream_sources

            self._stream_sources = build_stream_sources(
                self,
                ring_capacity=self._settings.stream_ring_capacity,
                event_sink=self.record_event,
                reading_store=self.reading_store,
            )
        return self._stream_sources

    @property
    def reading_store(self) -> Any:
        """Return the durable reading store, or None when persistence is off.

        Built on first use rather than at construction so that a profile with hardware
        enabled but no streaming channels never touches the filesystem.

        Without this store samples exist only in a bounded in-memory ring and vanish with
        the process, which makes every form of learning from physical experience
        impossible -- there is nothing to learn *from*.
        """
        if not self._settings.persist_readings:
            return None
        if self._reading_store is None:
            from leapflow.hardware.reading_store import ReadingStore

            self._reading_store = ReadingStore(
                raw_dir=Path(self._settings.readings_dir) if self._settings.readings_dir else None,
                db_path=(
                    Path(self._settings.instrument_db_path)
                    if self._settings.instrument_db_path
                    else None
                ),
                connection_holder=self._instrument_holder(),
                cache_manager=self._cache_manager,
                workspace_id=self._settings.workspace_id,
                session_id=self._session_id,
                reading_store_sensitive=self._settings.reading_store_sensitive,
                raw_ttl_s=self._settings.raw_retention_days * 24 * 3600.0,
                downsample_interval_s=self._settings.downsample_interval_s,
                history_ttl_s=self._settings.history_retention_days * 24 * 3600.0,
                raw_segment_bytes=int(self._settings.raw_segment_mb * 1024 * 1024),
            )
        return self._reading_store

    def _instrument_holder(self) -> Any:
        """Return the shared holder for ``instrument.duckdb``, building it once.

        Profile-scoped, so it outlives session rebinds: unlike the reading store it is
        never rebuilt in ``bind_persistence``, which lets the rebuilt store reattach to
        the same connection rather than opening a second one to the same file. Returns
        None when no instrument database is configured, so both stores fall back to
        their bare-path behaviour.
        """
        if not self._settings.instrument_db_path:
            return None
        if self._instrument_conn is None:
            from leapflow.storage.connection import LocalConnectionHolder

            self._instrument_conn = LocalConnectionHolder(
                Path(self._settings.instrument_db_path)
            )
        return self._instrument_conn

    @property
    def calibration_store(self) -> Any:
        """Return the versioned calibration store, or None without an instrument database.

        Independent of ``persist_readings``: a bench can want its calibration history
        durable without streaming sample history, so this is gated on the database path
        alone. Shares the reading store's connection and its sensitivity posture, since
        both tiers live in the one file.
        """
        if not self._settings.instrument_db_path:
            return None
        if self._calibration_store is None:
            from leapflow.hardware.calibration_store import CalibrationStore

            self._calibration_store = CalibrationStore(
                db_path=Path(self._settings.instrument_db_path),
                connection_holder=self._instrument_holder(),
                cache_manager=self._cache_manager,
                sensitive=self._settings.reading_store_sensitive,
            )
        return self._calibration_store

    @property
    def preview_broker(self) -> Any:
        """Return the shared preview broker, building it on first use.

        Lazy because most profiles have no previewable channel, and a broker that
        exists starts a sweeper task the moment it holds a lease. Built here rather
        than injected so there is exactly one per registry: two brokers would each
        open the device and the second would be refused, since most cameras admit a
        single reader.
        """
        if self._preview_broker is None:
            from leapflow.hardware.preview import PreviewBroker

            self._preview_broker = PreviewBroker(
                self,
                idle_timeout_s=self._settings.preview_idle_timeout_s,
                max_fps=self._settings.preview_max_fps,
                max_width=self._settings.preview_max_width,
                quality=self._settings.preview_quality,
            )
        return self._preview_broker

    def bind_persistence(
        self,
        *,
        cache_manager: Any = None,
        readings_dir: Any = None,
        session_id: str = "",
        experience_store: Any = None,
    ) -> None:
        """Attach session-scoped persistence and learning targets before sampling starts.

        Separate from construction because the raw sample directory is session-scoped and
        the session id is not known when the registry is built -- the registry is created
        during context construction, well before any session exists. The experience store
        arrives late for the same reason: it is wired during deferred initialization.
        """
        from dataclasses import replace

        if experience_store is not None:
            from leapflow.hardware.outcome import HardwareOutcomeRecorder

            self._outcome_recorder = HardwareOutcomeRecorder(experience_store)

        targets_changed = False
        if cache_manager is not None:
            self._cache_manager = cache_manager
            targets_changed = True
        if session_id and session_id != self._session_id:
            self._session_id = session_id
            targets_changed = True
        if readings_dir is not None and str(readings_dir) != self._settings.readings_dir:
            self._settings = replace(self._settings, readings_dir=str(readings_dir))
            targets_changed = True

        if not targets_changed:
            # Attaching only the experience store must not discard the stream sources.
            # This call arrives during deferred initialization, by which time sampling has
            # already started; rebuilding the sources here would leave the running ones
            # orphaned -- still reading, but no longer the objects stop_streams() stops.
            return
        self._reading_store = None
        # The calibration store is left in place: its file is profile-scoped and it
        # shares the instrument holder, which is deliberately not rebuilt here. Only its
        # cache_manager could go stale, and it re-registers idempotently on next write.
        self._stream_sources = None

    @property
    def outcome_recorder(self) -> Any:
        """Return the physical outcome recorder, or None when no store is bound.

        None rather than a no-op object so callers can skip the work entirely: with no
        experience store there is nowhere for a numeric delta to go, and pretending
        otherwise would accumulate pending commands nothing will ever resolve.
        """
        return self._outcome_recorder

    def channel_history(
        self, device_id: str, channel_id: str, *, limit: int = 200
    ) -> tuple[dict[str, Any], ...]:
        """Return downsampled history windows for a channel, oldest first."""
        store = self.reading_store
        if store is None:
            return ()
        return store.history(device_id, channel_id, limit=limit)

    def channel_summary(self, device_id: str, channel_id: str) -> dict[str, Any] | None:
        """Return the sampled history summary for a streaming channel, if any.

        This is the only form in which sampled history is disclosed. The raw series
        stays in the ring: handing it to a model is what makes an overnight run
        unaffordable, and a summary is what a decision actually needs.
        """
        for source in self.stream_sources():
            if (
                source.source_id == f"hw:{device_id}:{channel_id}"
                and len(source.ring) > 0
            ):
                return source.ring.summary()
        return None

    async def start_streams(self) -> int:
        """Start sampling every streaming channel, returning how many started.

        The registry owns this lifecycle rather than delegating it to
        ``ActiveSourceManager``, because that manager still has no production
        caller: relying on it would mean shipping a sampling loop that never runs.

        Events go to whatever ``set_event_emitter`` installed. Without it they are
        still recorded for ``hw_status``, but nothing reacts to them -- a bench can
        leave its declared envelope overnight with no watch, board or turn ever
        hearing about it. The sink is read from the registry rather than taken as an
        argument so that omitting it is one fact about the process instead of a
        mistake each caller can make separately.
        """
        sources = self.stream_sources()
        if not sources:
            return 0
        started = 0
        for source in sources:
            try:
                await source.start(self._event_emitter)
                started += 1
            except Exception as exc:  # noqa: BLE001 - one source must not stop the rest
                logger.warning(
                    "Hardware stream %s failed to start: %s",
                    source.source_id,
                    exc,
                    exc_info=True,
                )
        logger.info("Hardware streaming started: %d channel(s)", started)
        return started

    async def stop_streams(self) -> None:
        """Stop every sampling loop. Idempotent, and isolates failures."""
        for source in self.stream_sources():
            try:
                await source.stop()
            except Exception as exc:  # noqa: BLE001 - teardown must not propagate
                logger.warning(
                    "Hardware stream %s stop failed: %s", source.source_id, exc, exc_info=True
                )

    def record_event(self, event: Any) -> None:
        """Keep a bounded tail of derived events for hw_status.

        An alert-severity event also tightens the reading store's downsample window,
        so the excursion that raised it is captured at finer resolution than the coarse
        steady-state interval -- the shape of a breach is exactly what a later analysis
        needs and what a sixty-second mean would average away. Routed through the same
        sink the sampling loop already uses, and guarded so an unbuilt store (persistence
        off) is simply skipped.
        """
        self._recent_events.append(event)
        store = self._reading_store
        if store is None:
            return
        # Lazy import, matching how the rest of the registry reaches observability, and
        # so the alert taxonomy has one home (the digest) rather than a second copy here.
        from leapflow.hardware.observability.digest import ALERT_KINDS

        if str(getattr(event, "kind", "")) in ALERT_KINDS:
            store.note_alert()

    def set_event_emitter(self, emit: Any) -> None:
        """Install the sink that carries hardware events onto the signal path.

        One installation point rather than an argument threaded to each producer.
        The sampling loop used to take its emitter as a parameter, which made
        "forgot to pass it" a per-callsite mistake that no test could see from the
        inside -- and it shipped exactly that way once: six detection rules produced
        events that reached nothing. With the sink held here, whether anything is
        listening is a single testable fact, and the command path can report through
        the same channel as the sampling loop.
        """
        self._event_emitter = emit

    def publish_event(self, event: Any) -> None:
        """Record an event and put it on the signal path.

        Used by the command path. The sampling loop keeps its own dispatch because it
        must pace per channel first, and that state belongs to the source.

        Not paced here: a command is a human-scale act and each refusal is one the
        operator asked for, so suppressing repeats would hide the very thing that
        makes a stalled bench visible. Downstream deduplication remains free to
        collapse them.
        """
        self.record_event(event)
        emit = self._event_emitter
        if emit is None:
            return
        try:
            emit(event)
        except Exception as exc:  # noqa: BLE001 - a sink must not fail the command
            logger.warning("Hardware event emit raised: %s", exc, exc_info=True)

    def recent_events(self, device_id: str = "", limit: int = 10) -> tuple[Any, ...]:
        """Return the most recent derived events, newest last."""
        items = [
            event
            for event in self._recent_events
            if not device_id or getattr(event, "device_id", "") == device_id
        ]
        return tuple(items[-max(0, limit):])

    # ── Describe-before-write bookkeeping ──

    def mark_described(self, session_id: str, device_id: str) -> None:
        self._described.add((str(session_id), str(device_id)))

    def was_described(self, session_id: str, device_id: str) -> bool:
        return (str(session_id), str(device_id)) in self._described

    def device_io(self, device_id: str) -> Any:
        """Return an async context manager serialising data-plane access to one device.

        Per device, not per channel: a serial line, an I2C bus or a GPIB address is a
        single conversation, and two coroutines reading different channels of the same
        instrument interleave request and response frames. The result is not a failed
        read -- it is a *plausible* reading carrying the wrong channel's value, which
        no downstream check can detect. Streaming makes this the common case rather
        than an edge one, because one task per channel starts automatically.

        ``halt`` deliberately does not take this lock. Emergency stop must preempt a
        queued read, not wait behind it.

        A shared bus hosting several addresses still needs a coarser lock; that needs
        a real device to size (see the transport conformance suite).
        """
        key = str(device_id)
        lock = self._io_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._io_locks[key] = lock
        return lock

    # ── Rate-limit baseline ──

    def record_command(self, device_id: str, channel_id: str, value: float) -> None:
        """Remember a value that reached the device, for rate limiting.

        Recorded only after a write is known to have succeeded. A denied command
        must not move the baseline, or refusing one command would relax the limit
        on the next; and a failed command must not either, because its effect is
        by definition unknown -- claiming it as the current value would be a guess
        presented as a measurement.
        """
        self._last_command[(str(device_id), str(channel_id))] = (
            float(value),
            time.monotonic(),
        )

    def last_command(self, device_id: str, channel_id: str) -> tuple[float, float] | None:
        """Return ``(value, monotonic_timestamp)`` of the last accepted command."""
        return self._last_command.get((str(device_id), str(channel_id)))

    # ── Teardown ──

    async def drop_transport(self, device_id: str) -> None:
        """Forget the cached transport for *device_id* so the next call reconnects.

        ``transport()`` caches, and the cache is what makes a dead connection
        permanent: a server that restarted leaves a session object that answers every
        probe with "not connected", and since the instance is cached ``open()`` is
        never called again. Without this, one transient outage would make a device
        unusable for the life of the process.

        Never raises. It is called from failure paths, where an exception would
        replace the original diagnosis with a teardown error.
        """
        transport = self._transports.pop(device_id, None)
        if transport is None:
            return
        try:
            await transport.close()
        except Exception as exc:  # noqa: BLE001 - as above
            logger.warning(
                "Hardware transport %r close failed while dropping it: %s",
                device_id,
                exc,
                exc_info=True,
            )

    async def close_all(self) -> None:
        """Stop sampling, then close every open transport, isolating failures.

        Registered as an async effect on the owning scope so connections unwind in
        reverse order of opening. Sampling stops first: a loop still reading from a
        transport that is being closed would log a failure for every channel on the way
        down, burying whatever actually caused the teardown.

        Previews are released before transports for the same reason and one more: a
        preview holds a device claimed, and releasing it here is what powers a camera
        down on shutdown rather than at interpreter exit.
        """
        await self.stop_streams()
        if self._preview_broker is not None:
            await self._preview_broker.close()
        store = self._reading_store
        if store is not None:
            # Flushed before transports close: the last interval of a long run is exactly
            # the data somebody will want, and it is still only buffered at this point.
            store.close()
        calibration = self._calibration_store
        if calibration is not None:
            calibration.close()
        # The stores share this holder and neither owns it, so it is closed here, last of
        # the instrument.duckdb writers. It reopens lazily if history is read afterwards.
        if self._instrument_conn is not None:
            try:
                self._instrument_conn.close()
            except Exception as exc:  # noqa: BLE001 - teardown must not propagate
                logger.debug("instrument holder close failed: %s", exc, exc_info=True)
        for device_id, transport in list(self._transports.items()):
            try:
                await transport.close()
            except Exception as exc:  # noqa: BLE001 - teardown must not propagate
                logger.warning(
                    "Hardware transport %r close failed: %s", device_id, exc, exc_info=True
                )
        self._transports.clear()


__all__ = [
    "AdmissionNote",
    "HardwareRegistry",
    "HardwareSettings",
    "LoadReport",
    "UnverifiedContextPolicy",
    "build_registry",
]
