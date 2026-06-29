"""Platform capability discovery and manifest types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet


class PlatformID(Enum):
    """Identifies the host platform variant."""

    DARWIN_15 = "darwin_15"
    DARWIN_26 = "darwin_26"
    LINUX_GNOME = "linux_gnome"
    LINUX_KDE = "linux_kde"
    UNKNOWN = "unknown"


class Capability(Enum):
    """Atomic capability that a host may or may not support."""

    # Perception
    FS_WATCH = "fs.watch"
    FS_SEMANTIC_INDEX = "fs.semantic_index"
    AX_TREE_READ = "ax.tree_read"
    AX_PERFORM_ACTION = "ax.perform_action"
    APP_INTENTS_DISCOVER = "app_intents.discover"
    APP_INTENTS_PERFORM = "app_intents.perform"
    CLIPBOARD_READ = "clipboard.read"
    CLIPBOARD_WATCH = "clipboard.watch"
    SCREEN_CAPTURE = "screen.capture"
    SCREEN_CAPTURE_GPU = "screen.capture_gpu"

    # Execution
    FILE_OPS = "file.ops"
    APP_LAUNCH = "app.launch"
    APP_ACTIVATE = "app.activate"
    SHELL_EXEC = "shell.exec"
    NOTIFICATION_SEND = "notification.send"

    # Linux (reserved)
    DBUS_CALL = "dbus.call"
    AT_SPI_READ = "at_spi.read"
    PIPEWIRE_CAPTURE = "pipewire.capture"


# Reverse lookup: capability string value → enum member
_VALUE_TO_CAP = {c.value: c for c in Capability}


def capability_from_str(value: str) -> Capability | None:
    """Resolve a capability string to its enum member, or None if unknown."""
    return _VALUE_TO_CAP.get(value)


# Default capability set assumed for legacy hosts that don't support system.manifest
DEFAULT_DARWIN_CAPABILITIES: FrozenSet[Capability] = frozenset(
    {
        Capability.FS_WATCH,
        Capability.AX_TREE_READ,
        Capability.AX_PERFORM_ACTION,
        Capability.CLIPBOARD_READ,
        Capability.CLIPBOARD_WATCH,
        Capability.FILE_OPS,
        Capability.APP_LAUNCH,
        Capability.APP_ACTIVATE,
        Capability.SHELL_EXEC,
    }
)


@dataclass(frozen=True)
class PlatformManifest:
    """Immutable snapshot of host capabilities obtained during handshake."""

    platform_id: PlatformID
    os_version: str
    capabilities: FrozenSet[Capability]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def supports(self, cap: Capability) -> bool:
        return cap in self.capabilities

    def supports_all(self, *caps: Capability) -> bool:
        return all(c in self.capabilities for c in caps)

    def supports_any(self, *caps: Capability) -> bool:
        return any(c in self.capabilities for c in caps)

    @staticmethod
    def default_darwin() -> PlatformManifest:
        """Fallback manifest when host lacks system.manifest support."""
        return PlatformManifest(
            platform_id=PlatformID.DARWIN_15,
            os_version="15.0.0",
            capabilities=DEFAULT_DARWIN_CAPABILITIES,
        )
