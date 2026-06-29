"""macOS permission detection and settings deep-links.

Detects Accessibility / Screen Recording / Full Disk Access status without
requiring elevated privileges. Some permissions cannot be reliably probed
from Python; in those cases the corresponding field is ``None`` (unknown).
"""

from __future__ import annotations

import logging
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PermissionStatus:
    """Snapshot of macOS TCC permission state.

    A value of ``None`` denotes "cannot be determined from the Brain side"
    (typically because the public API is only callable from a Cocoa process).
    """

    accessibility: Optional[bool]
    screen_recording: Optional[bool]
    full_disk_access: Optional[bool] = None

    def all_granted(self) -> bool:
        """True only when every known field is explicitly True."""
        return all(
            v is True
            for v in (self.accessibility, self.screen_recording, self.full_disk_access)
        )

    def to_dict(self) -> dict[str, Optional[bool]]:
        return {
            "accessibility": self.accessibility,
            "screen_recording": self.screen_recording,
            "full_disk_access": self.full_disk_access,
        }


# ─── Detection ──────────────────────────────────────────────────────────


def _is_macos() -> bool:
    return sys.platform == "darwin"


def check_accessibility() -> Optional[bool]:
    """Probe Accessibility (AX) trust via PyObjC if available, else None.

    The TCC database is SIP-protected and we deliberately avoid reading it.
    """
    if not _is_macos():
        return False
    try:
        # PyObjC is an optional dependency; fall back to unknown if absent.
        from ApplicationServices import AXIsProcessTrusted  # type: ignore
    except Exception:
        return None
    try:
        return bool(AXIsProcessTrusted())
    except Exception as exc:  # pragma: no cover — defensive
        logger.debug("AXIsProcessTrusted failed: %s", exc)
        return None


def check_screen_recording() -> Optional[bool]:
    """Probe Screen Recording permission via CoreGraphics preflight.

    macOS exposes ``CGPreflightScreenCaptureAccess`` (10.15+) which returns a
    bool without prompting. Without PyObjC we cannot reliably detect — return
    ``None`` so the caller can guide the user explicitly.
    """
    if not _is_macos():
        return False
    try:
        from Quartz import CGPreflightScreenCaptureAccess  # type: ignore
    except Exception:
        return None
    try:
        return bool(CGPreflightScreenCaptureAccess())
    except Exception as exc:  # pragma: no cover
        logger.debug("CGPreflightScreenCaptureAccess failed: %s", exc)
        return None


def check_full_disk_access() -> Optional[bool]:
    """Probe Full Disk Access via a read attempt on a TCC-protected path.

    Reading ``~/Library/Application Support/com.apple.TCC/TCC.db`` requires
    FDA. A successful ``open()`` (even read-only) implies the permission is
    granted to the calling process.
    """
    if not _is_macos():
        return False
    from pathlib import Path

    probe = Path.home() / "Library" / "Application Support" / "com.apple.TCC" / "TCC.db"
    if not probe.exists():
        return None
    try:
        with open(probe, "rb") as fh:
            fh.read(1)
        return True
    except PermissionError:
        return False
    except OSError as exc:
        logger.debug("FDA probe failed: %s", exc)
        return None


def check_permissions() -> PermissionStatus:
    """Aggregate snapshot of all detectable permissions."""
    return PermissionStatus(
        accessibility=check_accessibility(),
        screen_recording=check_screen_recording(),
        full_disk_access=check_full_disk_access(),
    )


# ─── Settings deep-links ────────────────────────────────────────────────


_PRIVACY_URLS = {
    "accessibility": "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
    "screen_recording": "x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture",
    "full_disk_access": "x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles",
}


def _open_url(url: str) -> bool:
    if not _is_macos():
        logger.debug("Skip opening %s on non-macOS host", url)
        return False
    try:
        subprocess.run(["open", url], check=False, timeout=5)
        return True
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("open(%s) failed: %s", url, exc)
        return False


def open_accessibility_settings() -> bool:
    """Open System Settings → Privacy → Accessibility."""
    return _open_url(_PRIVACY_URLS["accessibility"])


def open_screen_recording_settings() -> bool:
    """Open System Settings → Privacy → Screen Recording."""
    return _open_url(_PRIVACY_URLS["screen_recording"])


def open_full_disk_access_settings() -> bool:
    """Open System Settings → Privacy → Full Disk Access."""
    return _open_url(_PRIVACY_URLS["full_disk_access"])
