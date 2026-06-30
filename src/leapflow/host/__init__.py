"""OS Host lifecycle management package.

Public surface:
    HostManager       — start/stop/status/install/launchd coordinator
    HostStatus        — immutable status snapshot
    HostState         — RUNNING / STOPPED / STALE
    PermissionStatus  — Accessibility / Screen Recording / FDA snapshot
    LaunchdService    — LaunchAgent plist generation + launchctl wrapper
    check_permissions — convenience aggregator
"""

from leapflow.host.launchd import LaunchdError, LaunchdService
from leapflow.host.manager import HostDiagnosis, HostManager, HostState, HostStatus
from leapflow.host.permissions import (
    PermissionStatus,
    check_accessibility,
    check_full_disk_access,
    check_permissions,
    check_screen_recording,
    open_accessibility_settings,
    open_full_disk_access_settings,
    open_screen_recording_settings,
)

__all__ = [
    "HostManager",
    "HostStatus",
    "HostState",
    "HostDiagnosis",
    "LaunchdService",
    "LaunchdError",
    "PermissionStatus",
    "check_permissions",
    "check_accessibility",
    "check_screen_recording",
    "check_full_disk_access",
    "open_accessibility_settings",
    "open_screen_recording_settings",
    "open_full_disk_access_settings",
]
