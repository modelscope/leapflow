# Copyright (c) Alibaba, Inc. and its affiliates.
"""`leap hw validate --workcells` — offline workcell admission.

Runs the node's own workcell admission over a directory of device YAML, so a
workcell can be checked before it is deployed. Nothing is opened: hardware may
be powered off or absent, and the host need not have hardware enabled.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_RESET = "\033[0m"
_DIM = "\033[2m"
_GREEN = "\033[32m"
_RED = "\033[31m"
_YELLOW = "\033[33m"
_CYAN = "\033[1;36m"


def run_workcell_validation(args: argparse.Namespace, settings: Any, json_mode: bool) -> int:
    """Validate ``--workcells`` against ``--devices`` (both default to the profile)."""
    from leapflow.hardware.workcell_validation import validate_workcells
    from leapflow.layout import HardwareLayout

    layout = HardwareLayout(settings.profile_dir / "hardware")
    workcells = Path(str(args.workcells)).expanduser() if args.workcells else layout.workcells_dir
    devices = Path(str(args.devices)).expanduser() if args.devices else layout.devices_dir
    result = validate_workcells(workcells, devices).to_dict()
    if json_mode:
        print(json.dumps(result, indent=2, default=str, ensure_ascii=False))
    else:
        _render(result)
    return 0 if result["ok"] else 1


def _render(result: dict[str, Any]) -> None:
    print(f"{_CYAN}LEAP Hardware \u2014 validate workcells{_RESET}")
    print(f"  {_DIM}workcells: {result['workcells_dir']}{_RESET}")
    print(f"  {_DIM}devices:   {result['devices_dir']}{_RESET}")
    devices = result["devices"]
    for note in devices["notes"]:
        if note["outcome"] == "rejected":
            print(
                f"  {_YELLOW}!{_RESET} device {note['device_id']} ({note['rule']}): {note['detail']}"
            )
    workcells = result["workcells"]
    for workcell_id in workcells["admitted"]:
        print(f"  {_GREEN}\u2713{_RESET} {workcell_id}: admitted")
    for note in workcells["rejected"]:
        name = note["workcell"] or Path(note["source"]).name
        print(f"  {_RED}\u2717{_RESET} {name}: {note['detail']}")
    if not workcells["admitted"] and not workcells["rejected"]:
        print(f"  {_RED}\u2717{_RESET} no workcell declarations found")
