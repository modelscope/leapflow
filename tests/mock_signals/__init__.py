"""Mock signal injection framework for LeapFlow end-to-end testing."""

from tests.mock_signals.generators import (
    BaseGenerator,
    SignalConfig,
    FsChangeGenerator,
    AppFocusGenerator,
    ClipboardGenerator,
    InputGenerator,
    GatewaySignalGenerator,
    GatewayMessageGenerator,
    HardwareChannelSpec,
    HardwareSignalGenerator,
)
from tests.mock_signals.profiles import PROFILES, ScenarioProfile
from tests.mock_signals.runner import MockSignalRunner, RunResult

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
    "PROFILES",
    "ScenarioProfile",
    "MockSignalRunner",
    "RunResult",
]
