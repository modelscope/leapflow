# Copyright (c) Alibaba, Inc. and its affiliates.
"""Offline workcell admission: would these declarations be admitted on a node?

The node admits workcells against its live device declarations when it starts
(``SkillRuntime.reload_workcells``). This runs the same admission over a
directory of device YAML without opening any transport, so a workcell can be
checked on a laptop before it is deployed. Device verification is judged
separately (``leap hw validate <device>``): unverified declarations are taken
as written here, so a missing human confirmation does not mask a workcell
problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from leapflow.hardware.registry import (
    HardwareRegistry,
    HardwareSettings,
    LoadReport,
    UnverifiedContextPolicy,
)
from leapflow.hardware.skills.runtime import SkillRuntime
from leapflow.hardware.skills.runtime_settings import SkillRuntimeSettings
from leapflow.hardware.skills.workcell import WorkcellLoadReport


@dataclass(frozen=True)
class WorkcellValidation:
    """Device admission plus workcell admission for one pair of directories."""

    workcells_dir: str
    devices_dir: str
    devices: LoadReport
    workcells: WorkcellLoadReport

    @property
    def ok(self) -> bool:
        return bool(self.workcells.admitted) and not self.workcells.rejected

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "workcells_dir": self.workcells_dir,
            "devices_dir": self.devices_dir,
            "devices": self.devices.to_dict(),
            "workcells": self.workcells.to_dict(),
        }


def validate_workcells(workcells_dir: Path, devices_dir: Path) -> WorkcellValidation:
    """Admit every workcell in *workcells_dir* against the devices in *devices_dir*.

    Nothing is opened, persisted, or streamed; an empty or missing workcell
    directory validates as not ok (there is nothing to deploy).
    """
    settings = HardwareSettings(
        # No verified_path: verification is out of scope here (see module doc).
        providers=(("yaml", {"devices_dir": str(devices_dir)}),),
        unverified_context_policy=UnverifiedContextPolicy.ALLOW,
        persist_readings=False,
        stream_enabled=False,
    )
    registry = HardwareRegistry(settings)
    devices = registry.load()
    runtime = SkillRuntime(registry, settings=SkillRuntimeSettings(workcells_dir=workcells_dir))
    return WorkcellValidation(
        workcells_dir=str(workcells_dir),
        devices_dir=str(devices_dir),
        devices=devices,
        workcells=runtime.load_report,
    )
