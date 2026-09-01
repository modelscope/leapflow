"""Shared leapspace helpers: sandbox image presets."""

import os
from enum import Enum
from pathlib import Path
from typing import Any


def write_atomic(path: Path, text: str) -> None:
    """Write text atomically via a sibling tmp file + os.replace."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def check(name: str, cond: bool, detail: str = "") -> bool:
    """Report one named assertion as a PASS/FAIL line; returns cond.

    Shared convention for task expect() functions: accumulate with
    ``ok &= check(...)`` so every check runs and reports, then exit
    non-zero if any failed.
    """
    print(f"{'PASS' if cond else 'FAIL'} {name}: {detail}")
    return cond


class LeapAppImage(str, Enum):
    """Sandbox image presets, selectable by name from CLI or task config."""

    LINUX = "linux"
    # WINDOWS = "windows"
    # MACOS = "macos"


CUA_MCP_PORT = 3000  # Port exposed by the CUA driver for MCP access

# Image spec dicts consumed by cua.Image.from_dict(). Install layers
# (apt_install, env, ...) stay empty until the app UI stack is chosen.
# Host feasibility: LINUX runs under local QEMU+KVM; WINDOWS is untested;
# MACOS requires an Apple Silicon host (Lume).
LINUX_IMAGE_CONFIG: dict[str, Any] = {
    "os_type": "linux",
    "distro": "ubuntu",
    "version": "24.04",
    "kind": "vm",
}

IMAGE_CONFIGS: dict[LeapAppImage, dict[str, Any]] = {
    LeapAppImage.LINUX: LINUX_IMAGE_CONFIG,
}


def get_image_config(image: LeapAppImage) -> dict[str, Any]:
    """Return a copy of the preset spec for cua.Image.from_dict().

    Shared conventions (e.g. exposing the MCP port) are applied by the caller.
    """
    return IMAGE_CONFIGS[image].copy()
