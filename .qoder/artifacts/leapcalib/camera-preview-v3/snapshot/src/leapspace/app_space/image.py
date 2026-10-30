# Copyright (c) Alibaba, Inc. and its affiliates.
"""Host-only leapspace image construction.

This is the half of the former ``utils.py`` that must import the host SDK
(``cua_sandbox``). It is kept strictly separate from ``state.py`` (EVO-02 LS-1)
so that importing the in-box verdict, the pure state helpers, or the PyQt6 apps
never drags in ``cua_sandbox`` -- the coupling that made a headless/off-sandbox
run impossible. Only the harness (which already needs the sandbox to boot a VM)
imports this module.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from cua_sandbox import Image

from leapspace.app_space.state import (
    CUA_MCP_PORT,
    LEAPFLOW_ARCHIVE_DST,
    LINUX_LEAPFLOW_PATH,
    PYQT_SYSTEM_LIBS,
    LeapAppImage,
)


def _archive_checkout() -> Path:
    """Pack this checkout's committed state for the image to copy in.

    The build VM reaches GitHub only through the host's flaky link, so the
    repo travels as a host-built archive: no in-box clone, and the box runs
    exactly the code under test (HEAD, not some remote ref).
    """
    repo_root = Path(__file__).resolve().parents[3]
    fd, name = tempfile.mkstemp(suffix=".tar.gz")
    os.close(fd)
    result = subprocess.run(
        ["git", "archive", "--format=tar.gz", "-o", name, "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git archive failed: {result.stderr.strip()}")
    return Path(name)


def get_image(image: LeapAppImage) -> Image:
    """Return the preset image spec, rooted in an archive of this checkout.

    Image is frozen and chainable -- every mutation returns a new instance --
    so callers get a fresh spec (and a fresh archive) per run. needrestart
    is removed before apt: installing python3-dev upgrades service
    libraries, and needrestart's service restarts SIGTERM the layer's own
    command transport. Host feasibility: LINUX runs under local QEMU+KVM;
    WINDOWS is untested; MACOS requires an Apple Silicon host (Lume).
    """
    archive = _archive_checkout()
    if image == LeapAppImage.LINUX:
        return (
            Image.linux(distro="ubuntu", version="24.04", kind="vm")
            .expose(CUA_MCP_PORT)
            .run("sudo apt-get remove -y needrestart")
            .apt_install("python3-pyatspi", "python3-dev", *PYQT_SYSTEM_LIBS, "git", "make")
            .pip_install("PyQt6", "uv")
            .copy(str(archive), LEAPFLOW_ARCHIVE_DST)
            .run(
                f"mkdir -p {LINUX_LEAPFLOW_PATH} && "
                f"tar xzf {LEAPFLOW_ARCHIVE_DST} -C {LINUX_LEAPFLOW_PATH} && "
                f"rm {LEAPFLOW_ARCHIVE_DST}"
            )
            .run(f"cd {LINUX_LEAPFLOW_PATH} && make space-sync")
        )
    else:
        raise NotImplementedError(f"image preset not defined for {image}")
