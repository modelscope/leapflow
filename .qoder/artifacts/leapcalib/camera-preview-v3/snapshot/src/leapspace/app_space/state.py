# Copyright (c) Alibaba, Inc. and its affiliates.
"""In-box-safe leapspace helpers: the state-dir convention, atomic writes,
the verdict ``check`` line, and task action loading.

Split out from the former ``utils.py`` (EVO-02 LS-1). Everything here is
stdlib-only and imports no host SDK, so the in-box verdict program
(``action.py``'s ``expect()``), the pure state helpers, and the PyQt6 apps
(``apps/_base``) are importable with ``cua_sandbox`` absent -- which is what
lets a real app run headless (offscreen Qt) or a verdict run on any host.
Host-only image construction lives in the sibling ``image.py``.
"""

from __future__ import annotations

import importlib.util
import os
import platform
from enum import Enum
from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Awaitable, Callable, Literal

if TYPE_CHECKING:
    from leapspace.app_space.actor import LeapAppActor


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


def load_action(
    action_path: Path | str,
) -> tuple[Callable[["LeapAppActor"], Awaitable[None]], Callable[..., int]]:
    """Import a task's action.py once; return its (reference, expect) pair.

    Module-level imports are lint-guaranteed in-sandbox-safe (stdlib /
    PyQt6 / leapspace) -- a set the host import satisfies as well -- and
    exec_module honors the __main__ guard, so loading never triggers the
    verdict.
    """
    action_path = Path(action_path)
    spec = importlib.util.spec_from_file_location(
        f"leapspace_task_{action_path.stem}", str(action_path)
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"{action_path}: cannot be imported as a Python module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.reference, module.expect


class LeapAppImage(str, Enum):
    """Sandbox image presets, selectable by name from CLI or task config.

    Values are the sandbox OS names, which get_sandbox_state_dir keys on.
    """

    LINUX = "linux"
    # WINDOWS = "windows"
    # MACOS = "macos"


# Fixed per-OS roots, never resolved and never env-overridden: the harness
# and the in-sandbox apps compute this independently and must land on the
# same literal path.
_STATE_DIR_BY_SYSTEM: dict[str, PurePath] = {
    "linux": PurePosixPath("/tmp/leapspace"),
    "macos": PurePosixPath("/tmp/leapspace"),
    "windows": PureWindowsPath("C:/ProgramData/leapspace"),
}


def get_sandbox_state_dir(
    in_sandbox: bool,
    system: Literal["linux", "macos", "windows"] | None = None,
) -> PurePath:
    """The apps' state root -- the one path harness and apps must agree on.

    Two consumers, two ways to know the sandbox OS: apps run inside and
    detect it with platform.system(); the harness runs on the host, where
    detection would answer the host's OS, so it names the sandbox's OS
    (an image preset's value) instead. Callers append the app_id. No env
    override -- hermetic tests monkeypatch this function.
    """
    if in_sandbox:
        system = platform.system().lower()
        system = "macos" if system == "darwin" else system
    else:
        if system is None:
            raise ValueError("system is required when resolving from the host")
    return _STATE_DIR_BY_SYSTEM[system]


CUA_MCP_PORT = 3000  # Port exposed by the CUA driver for MCP access

# System X/GL libraries the pip PyQt6 wheel needs (it bundles Qt itself).
PYQT_SYSTEM_LIBS = [
    "libegl1",
    "libgl1",
    "libxkbcommon0",
    "libxkbcommon-x11-0",
    "libfontconfig1",
    "libglib2.0-0t64",
    "libdbus-1-3",
    "libxcb-cursor0",
    "libxcb-icccm4",
    "libxcb-image0",
    "libxcb-keysyms1",
    "libxcb-randr0",
    "libxcb-render-util0",
    "libxcb-shape0",
    "libxcb-xinerama0",
    "libxcb-xkb1",
    "libx11-xcb1",
]

# Repo checkout path inside the sandbox image.
LINUX_LEAPFLOW_PATH = "/opt/leapflow"

# Source tree inside the checkout -- PYTHONPATH for interpreters outside the
# repo venv (the OS python's apt stack: pyatspi).
LINUX_LEAPFLOW_SRC = f"{LINUX_LEAPFLOW_PATH}/src"

# Host-side archive landing spot: copied into the image, untarred, removed.
LEAPFLOW_ARCHIVE_DST = "/tmp/leapflow-checkout.tar.gz"


def get_image_venv_python(system: Literal["linux", "macos", "windows"]) -> str:
    """Interpreter inside the image's repo venv, keyed by sandbox OS.

    The venv is built by the image's `make space-sync` step (uv sync,
    hence the .venv name). The path must be spelled out: the shell cwd
    is not the repo checkout, so interpreter discovery (uv run, bare
    python) would resolve elsewhere. The harness launches apps and
    runs the verdict program through this interpreter.
    """
    if system == "linux":
        return f"{LINUX_LEAPFLOW_PATH}/.venv/bin/python"
    raise NotImplementedError(f"image venv python path not defined for {system}")


def get_image_system_python(system: Literal["linux", "macos", "windows"]) -> str:
    """Interpreter carrying the image's OS-level Python packages.

    apt installs land on the system interpreter (pyatspi, ...) while pip
    installs land in the repo venv (pynput, python-Xlib); staged helper
    scripts must pick the interpreter that owns their imports.
    """
    if system == "linux":
        return "/usr/bin/python3"
    raise NotImplementedError(f"image system python not defined for {system}")


def get_actor_stage_dir(system: Literal["linux", "macos", "windows"]) -> PurePath:
    """Staging dir for the actor's in-sandbox run payloads (element dumps).

    Sits under the state root but outside every app's state dir, so the
    signal watch surface (the apps' ground-truth writes) never records
    the actor's own traffic as signal. Helper scripts are not staged:
    they ship with the package and run via ``python -m action_utils``.
    """
    return get_sandbox_state_dir(in_sandbox=False, system=system) / ".actor"
