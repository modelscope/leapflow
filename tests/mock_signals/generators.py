"""Signal generators for mock injection testing.

Each generator class produces events conforming to the LeapFlow observer
payload contract.  All generators share a common SignalConfig and yield
(event_type, payload) tuples on demand.
"""

from __future__ import annotations

import random
import string
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class SignalConfig:
    """Common configuration for all signal generators."""

    frequency_hz: float = 1.0
    burst_size: int = 1
    burst_interval_s: float = 0.0
    duration_s: float = 5.0
    jitter_ms: float = 50.0


class BaseGenerator:
    """Base class for signal generators.

    Subclasses override ``_make_payload()`` to produce type-specific payloads.
    The ``generate()`` iterator handles timing, bursting, and jitter uniformly.
    """

    event_type: str = "event.unknown"

    def __init__(self, config: SignalConfig) -> None:
        self.config = config

    def generate(self) -> Iterator[tuple[str, Dict[str, Any]]]:
        """Yield (event_type, payload) tuples according to config timing."""
        cfg = self.config
        interval = 1.0 / cfg.frequency_hz if cfg.frequency_hz > 0 else cfg.duration_s
        start = time.monotonic()
        deadline = start + cfg.duration_s

        while time.monotonic() < deadline:
            for _ in range(cfg.burst_size):
                if time.monotonic() >= deadline:
                    return
                yield (self.event_type, self._make_payload())
            # Simulate inter-event wait as a yield marker (actual sleeping
            # is done by the runner to allow async interleaving)
            jitter = random.uniform(0, cfg.jitter_ms / 1000.0)
            wait = interval + jitter
            if cfg.burst_interval_s > 0 and cfg.burst_size > 1:
                wait = cfg.burst_interval_s + jitter
            # Cap wait at remaining duration to avoid overshooting the deadline
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            wait = min(wait, remaining)
            yield ("__wait__", {"seconds": wait})

    def _make_payload(self) -> Dict[str, Any]:
        """Override in subclass to produce type-specific payload."""
        raise NotImplementedError


# ─── Concrete generators ─────────────────────────────────────────────────


class FsChangeGenerator(BaseGenerator):
    """File system change events."""

    event_type = "event.fs_change"

    def __init__(
        self,
        config: SignalConfig,
        *,
        paths: Optional[List[str]] = None,
        actions: Optional[List[str]] = None,
    ) -> None:
        super().__init__(config)
        self.paths = paths or [
            f"/tmp/test/project/src/module_{i}.py" for i in range(20)
        ]
        self.actions = actions or ["created", "modified", "deleted", "modified", "modified"]

    def _make_payload(self) -> Dict[str, Any]:
        return {
            "path": random.choice(self.paths),
            "action": random.choice(self.actions),
            "is_dir": False,
            "_mono_ts": time.monotonic(),
        }


class AppFocusGenerator(BaseGenerator):
    """Application focus change events."""

    event_type = "event.app_focus_change"

    _DEFAULT_APPS = [
        {"bundle_id": "com.apple.Safari", "app_name": "Safari"},
        {"bundle_id": "com.microsoft.VSCode", "app_name": "Visual Studio Code"},
        {"bundle_id": "com.apple.Terminal", "app_name": "Terminal"},
        {"bundle_id": "com.tinyspeck.slackmacgap", "app_name": "Slack"},
        {"bundle_id": "com.apple.finder", "app_name": "Finder"},
    ]

    def __init__(
        self,
        config: SignalConfig,
        *,
        apps: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        super().__init__(config)
        self.apps = apps or self._DEFAULT_APPS

    def _make_payload(self) -> Dict[str, Any]:
        app = random.choice(self.apps)
        return {
            "bundle_id": app["bundle_id"],
            "app_name": app["app_name"],
            "pid": random.randint(1000, 99999),
            "window_title": f"{app['app_name']} — Document {random.randint(1, 10)}",
            "ts": time.time(),
            "_mono_ts": time.monotonic(),
        }


class ClipboardGenerator(BaseGenerator):
    """Clipboard change events."""

    event_type = "event.clipboard_change"

    _SAMPLE_TEXTS = [
        "def hello_world():\n    print('Hello, World!')",
        "SELECT * FROM users WHERE active = true;",
        "https://github.com/leapflow/leapflow/pull/42",
        "The quick brown fox jumps over the lazy dog.",
        "import asyncio\n\nasync def main():\n    await asyncio.sleep(1)",
        "ERROR: Connection refused at 127.0.0.1:5432",
        '{"status": "ok", "data": [1, 2, 3]}',
    ]

    def __init__(
        self,
        config: SignalConfig,
        *,
        texts: Optional[List[str]] = None,
        content_type: str = "text",
    ) -> None:
        super().__init__(config)
        self.texts = texts or self._SAMPLE_TEXTS
        self.content_type = content_type

    def _make_payload(self) -> Dict[str, Any]:
        text = random.choice(self.texts)
        return {
            "text": text,
            "content_type": self.content_type,
            "source_app": "",
            "change_ts": time.time(),
            "_mono_ts": time.monotonic(),
        }


class InputGenerator(BaseGenerator):
    """User input events (click, type, scroll)."""

    event_type = "event.ui_action"

    def __init__(
        self,
        config: SignalConfig,
        *,
        action_type: str = "click",
        app_bundle_id: str = "com.apple.Safari",
    ) -> None:
        super().__init__(config)
        self.action_type = action_type
        self.app_bundle_id = app_bundle_id

    def _make_payload(self) -> Dict[str, Any]:
        base: Dict[str, Any] = {
            "action": self.action_type,
            "app_bundle_id": self.app_bundle_id,
            "timestamp": time.time(),
            "_mono_ts": time.monotonic(),
        }
        if self.action_type == "click":
            base["mouse_x"] = random.randint(0, 1920)
            base["mouse_y"] = random.randint(0, 1080)
        elif self.action_type == "type":
            char = random.choice(string.ascii_lowercase + string.digits + " ")
            base["key_code"] = random.randint(0, 127)
            base["char"] = char
        elif self.action_type == "scroll":
            base["delta_x"] = 0
            base["delta_y"] = random.choice([-3, -2, -1, 1, 2, 3])
            base["mouse_x"] = random.randint(0, 1920)
            base["mouse_y"] = random.randint(0, 1080)
        return base


class GatewaySignalGenerator(BaseGenerator):
    """External platform signal events."""

    event_type = "gateway.signal"

    _SIGNAL_TYPES = ["message", "reaction", "mention", "status_change", "callback"]

    def __init__(
        self,
        config: SignalConfig,
        *,
        platform_id: str = "feishu",
        signal_type: str = "message",
    ) -> None:
        super().__init__(config)
        self.platform_id = platform_id
        self.signal_type = signal_type

    def _make_payload(self) -> Dict[str, Any]:
        return {
            "event_id": uuid.uuid4().hex,
            "platform_id": self.platform_id,
            "signal_type": self.signal_type,
            "_platform": self.platform_id,
            "_mono_ts": time.monotonic(),
        }


class GatewayMessageGenerator(BaseGenerator):
    """External platform message events."""

    event_type = "gateway.message.received"

    _SAMPLE_MESSAGES = [
        "Hey, can you check the latest PR?",
        "Meeting at 3pm today",
        "The deployment failed again",
        "LGTM, ship it!",
        "Can you help me debug this issue?",
        "Updated the design doc, please review",
    ]

    def __init__(
        self,
        config: SignalConfig,
        *,
        platform_id: str = "feishu",
        sender: str = "test_user",
    ) -> None:
        super().__init__(config)
        self.platform_id = platform_id
        self.sender = sender

    def _make_payload(self) -> Dict[str, Any]:
        return {
            "event_id": uuid.uuid4().hex,
            "platform_id": self.platform_id,
            "sender": self.sender,
            "text": random.choice(self._SAMPLE_MESSAGES),
            "_platform": self.platform_id,
            "_mono_ts": time.monotonic(),
        }


# ─── Hardware channel configuration ──────────────────────────────────────


_QUALITY_OK = "ok"
_QUALITY_SUSPECT = "suspect"
_QUALITY_STALE = "stale"
_QUALITY_SATURATED = "saturated"
_DEGRADED_QUALITIES: List[str] = [_QUALITY_SUSPECT, _QUALITY_STALE, _QUALITY_SATURATED]


@dataclass
class HardwareChannelSpec:
    """Configuration for one simulated hardware channel.

    Mirrors the physically meaningful fields of
    ``leapflow.hardware.context.Channel`` without importing it, so the
    mock framework stays dependency-free from ``src/``.
    """

    channel_id: str = "ch0"
    quantity: str = "temperature"
    unit: str = "°C"
    center: float = 25.0
    amplitude: float = 5.0
    """Half-range of the simulated noise envelope around *center*."""
    min_threshold: float = 15.0
    max_threshold: float = 35.0
    quality_degradation_rate: float = 0.05
    """Per-reading probability of producing a non-OK quality flag."""


_DEFAULT_CHANNELS: List[HardwareChannelSpec] = [
    HardwareChannelSpec(
        channel_id="ch_temp",
        quantity="temperature",
        unit="°C",
        center=25.0,
        amplitude=5.0,
        min_threshold=15.0,
        max_threshold=35.0,
    ),
    HardwareChannelSpec(
        channel_id="ch_voltage",
        quantity="voltage",
        unit="V",
        center=3.3,
        amplitude=0.2,
        min_threshold=3.0,
        max_threshold=3.6,
    ),
]


class HardwareSignalGenerator(BaseGenerator):
    """Hardware reading and event generator.

    Produces two families of ``(event_type, payload)`` tuples:

    * **hw.reading** — one sampled reading whose payload aligns with
      ``Reading.to_dict()`` plus ``monotonic_at`` (needed by the test
      pipeline for monotonic ordering even though ``Reading.to_dict()``
      omits it for persistence).
    * **hw.<kind>** — derived hardware events (``threshold_exceeded``,
      ``quality_degraded``, ``sample_loss``, ``rate_exceeded``, ``stale``,
      ``settled``) whose payload aligns with ``HardwareEvent.to_payload()``.

    Overrides ``generate()`` because the base implementation always yields a
    single ``event_type``; this generator interleaves readings with
    probabilistic event injections.
    """

    event_type: str = "hw.reading"

    # Supported hardware event kinds mirroring ``stream.EventKind``.
    _EVENT_KINDS: List[str] = [
        "threshold_exceeded",
        "quality_degraded",
        "sample_loss",
        "rate_exceeded",
        "stale",
        "settled",
    ]

    def __init__(
        self,
        config: SignalConfig,
        *,
        device_id: str = "mock_device_0",
        channels: Optional[List[HardwareChannelSpec]] = None,
        event_kinds: Optional[List[str]] = None,
        event_probability: float = 0.08,
    ) -> None:
        super().__init__(config)
        self.device_id = device_id
        self.channels: List[HardwareChannelSpec] = (
            channels if channels is not None else list(_DEFAULT_CHANNELS)
        )
        self.event_kinds: List[str] = (
            event_kinds if event_kinds is not None else list(self._EVENT_KINDS[:2])
        )
        self.event_probability = max(0.0, min(1.0, event_probability))
        self._seq: Dict[str, int] = {}

    # ── generate (multi-type override) ──────────────────────────────────

    def generate(self) -> Iterator[tuple[str, Dict[str, Any]]]:
        """Yield ``(event_type, payload)`` tuples per config timing.

        Inherits the burst / jitter / duration contract from ``BaseGenerator``
        but yields two event families: readings and hardware events.
        """
        cfg = self.config
        interval = 1.0 / cfg.frequency_hz if cfg.frequency_hz > 0 else cfg.duration_s
        start = time.monotonic()
        deadline = start + cfg.duration_s

        while time.monotonic() < deadline:
            for _ in range(cfg.burst_size):
                if time.monotonic() >= deadline:
                    return
                channel = random.choice(self.channels)
                yield (self.event_type, self._make_reading(channel))
                # Probabilistic hardware event injection
                if self.event_kinds and random.random() < self.event_probability:
                    kind = random.choice(self.event_kinds)
                    yield (f"hw.{kind}", self._make_hw_event(channel, kind))

            jitter = random.uniform(0, cfg.jitter_ms / 1000.0)
            wait = interval + jitter
            if cfg.burst_interval_s > 0 and cfg.burst_size > 1:
                wait = cfg.burst_interval_s + jitter
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            wait = min(wait, remaining)
            yield ("__wait__", {"seconds": wait})

    # ── payload builders ────────────────────────────────────────────────

    def _make_reading(self, ch: HardwareChannelSpec) -> Dict[str, Any]:
        """Build a reading payload aligned with ``Reading.to_dict()`` + ``monotonic_at``."""
        seq = self._seq.get(ch.channel_id, 0)
        self._seq[ch.channel_id] = seq + 1

        noise = random.gauss(0, ch.amplitude * 0.3)
        value = round(ch.center + noise, 4)

        quality: str = _QUALITY_OK
        if random.random() < ch.quality_degradation_rate:
            quality = random.choice(_DEGRADED_QUALITIES)

        return {
            "device_id": self.device_id,
            "channel_id": ch.channel_id,
            "value": value,
            "quantity": ch.quantity,
            "unit": ch.unit,
            "observed_at": time.time(),
            "sequence": seq,
            "quality": quality,
            # Deliberately included for test-pipeline monotonic ordering
            # even though Reading.to_dict() omits it for persistence.
            "monotonic_at": time.monotonic(),
        }

    def _make_hw_event(
        self, ch: HardwareChannelSpec, kind: str
    ) -> Dict[str, Any]:
        """Build a hardware event payload aligned with ``HardwareEvent.to_payload()``."""
        now_wall = time.time()
        value = round(ch.center + random.gauss(0, ch.amplitude), 4)

        detail = self._event_detail(ch, kind, value)
        return {
            "kind": kind,
            "source": f"{self.device_id}.{ch.channel_id}",
            "device_id": self.device_id,
            "channel_id": ch.channel_id,
            "quantity": ch.quantity,
            "detail": detail,
            "value": value,
            "unit": ch.unit,
            "ts": now_wall,
            "_mono_ts": time.monotonic(),
        }

    @staticmethod
    def _event_detail(
        ch: HardwareChannelSpec, kind: str, value: float
    ) -> str:
        """Return a human-readable detail string for a hardware event."""
        details: Dict[str, str] = {
            "threshold_exceeded": (
                f"left the declared range ({ch.min_threshold:g}..{ch.max_threshold:g})"
            ),
            "quality_degraded": (
                "quality has been 'suspect' for 3 consecutive samples"
            ),
            "sample_loss": "2 sample(s) missing from the transport sequence",
            "rate_exceeded": "changing at 15/s, above the declared 10/s",
            "stale": "no sample for 2.50s on a 10 Hz channel",
            "settled": "returned to the declared range",
        }
        return details.get(kind, f"hardware event: {kind}")

    def _make_payload(self) -> Dict[str, Any]:
        """Fallback for direct base-class callers; yields a reading."""
        return self._make_reading(self.channels[0])


# Generator class registry for dynamic resolution by name
GENERATOR_REGISTRY: Dict[str, type] = {
    "FsChangeGenerator": FsChangeGenerator,
    "AppFocusGenerator": AppFocusGenerator,
    "ClipboardGenerator": ClipboardGenerator,
    "InputGenerator": InputGenerator,
    "GatewaySignalGenerator": GatewaySignalGenerator,
    "GatewayMessageGenerator": GatewayMessageGenerator,
    "HardwareSignalGenerator": HardwareSignalGenerator,
}

__all__ = [
    "SignalConfig",
    "BaseGenerator",
    "FsChangeGenerator",
    "AppFocusGenerator",
    "ClipboardGenerator",
    "InputGenerator",
    "GatewaySignalGenerator",
    "GatewayMessageGenerator",
    "HardwareChannelSpec",
    "HardwareSignalGenerator",
    "GENERATOR_REGISTRY",
]
