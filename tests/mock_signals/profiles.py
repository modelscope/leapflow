"""Predefined signal injection scenarios (profiles).

Each profile describes a complete scenario with a mix of generators and their
configuration.  Profiles are resolved by name in the runner and CLI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from tests.mock_signals.generators import HardwareChannelSpec, SignalConfig


@dataclass
class ScenarioProfile:
    """A complete signal injection scenario."""

    name: str
    description: str
    generators: List[Tuple[str, Dict[str, Any]]] = field(default_factory=list)


PROFILES: Dict[str, ScenarioProfile] = {
    "normal": ScenarioProfile(
        name="normal",
        description="Normal user workflow: periodic fs changes, occasional app switches, typing",
        generators=[
            ("FsChangeGenerator", {"config": SignalConfig(frequency_hz=0.5, duration_s=10)}),
            ("AppFocusGenerator", {"config": SignalConfig(frequency_hz=0.1, duration_s=10)}),
            ("InputGenerator", {"config": SignalConfig(frequency_hz=2.0, duration_s=10), "action_type": "type"}),
        ],
    ),
    "burst": ScenarioProfile(
        name="burst",
        description="High-frequency burst: rapid file saves, fast typing",
        generators=[
            ("FsChangeGenerator", {"config": SignalConfig(frequency_hz=20.0, burst_size=5, duration_s=5)}),
            ("InputGenerator", {"config": SignalConfig(frequency_hz=10.0, duration_s=5), "action_type": "type"}),
        ],
    ),
    "mixed": ScenarioProfile(
        name="mixed",
        description="Mixed signals: all types simultaneously",
        generators=[
            ("FsChangeGenerator", {"config": SignalConfig(frequency_hz=1.0, duration_s=8)}),
            ("AppFocusGenerator", {"config": SignalConfig(frequency_hz=0.3, duration_s=8)}),
            ("ClipboardGenerator", {"config": SignalConfig(frequency_hz=0.2, duration_s=8)}),
            ("InputGenerator", {"config": SignalConfig(frequency_hz=3.0, duration_s=8), "action_type": "click"}),
            ("GatewaySignalGenerator", {"config": SignalConfig(frequency_hz=0.5, duration_s=8)}),
        ],
    ),
    "stress": ScenarioProfile(
        name="stress",
        description="Stress test: maximum throughput across all signal types",
        generators=[
            ("FsChangeGenerator", {"config": SignalConfig(frequency_hz=50.0, burst_size=10, duration_s=10)}),
            ("InputGenerator", {"config": SignalConfig(frequency_hz=20.0, duration_s=10), "action_type": "click"}),
            ("ClipboardGenerator", {"config": SignalConfig(frequency_hz=5.0, duration_s=10)}),
            ("GatewaySignalGenerator", {"config": SignalConfig(frequency_hz=10.0, duration_s=10)}),
        ],
    ),
    "gateway": ScenarioProfile(
        name="gateway",
        description="Gateway focused: external platform signals and messages",
        generators=[
            ("GatewaySignalGenerator", {"config": SignalConfig(frequency_hz=2.0, duration_s=10)}),
            ("GatewayMessageGenerator", {"config": SignalConfig(frequency_hz=1.0, duration_s=10)}),
        ],
    ),
    "hardware": ScenarioProfile(
        name="hardware",
        description="Hardware sensor stream: readings with threshold and quality events",
        generators=[
            (
                "HardwareSignalGenerator",
                {
                    "config": SignalConfig(frequency_hz=5.0, duration_s=10),
                    "device_id": "mock_bench_0",
                    "channels": [
                        HardwareChannelSpec(
                            channel_id="ch_temp",
                            quantity="temperature",
                            unit="\u00b0C",
                            center=25.0,
                            amplitude=6.0,
                            min_threshold=15.0,
                            max_threshold=35.0,
                            quality_degradation_rate=0.03,
                        ),
                        HardwareChannelSpec(
                            channel_id="ch_voltage",
                            quantity="voltage",
                            unit="V",
                            center=3.3,
                            amplitude=0.3,
                            min_threshold=3.0,
                            max_threshold=3.6,
                            quality_degradation_rate=0.02,
                        ),
                    ],
                    "event_kinds": ["threshold_exceeded", "quality_degraded"],
                    "event_probability": 0.10,
                },
            ),
            (
                "HardwareSignalGenerator",
                {
                    "config": SignalConfig(frequency_hz=2.0, duration_s=10),
                    "device_id": "mock_bench_1",
                    "channels": [
                        HardwareChannelSpec(
                            channel_id="ch_pressure",
                            quantity="pressure",
                            unit="kPa",
                            center=101.3,
                            amplitude=2.0,
                            min_threshold=95.0,
                            max_threshold=110.0,
                            quality_degradation_rate=0.01,
                        ),
                    ],
                    "event_kinds": ["threshold_exceeded", "sample_loss", "stale"],
                    "event_probability": 0.05,
                },
            ),
        ],
    ),
}


__all__ = ["ScenarioProfile", "PROFILES"]
