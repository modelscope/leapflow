# Copyright (c) Alibaba, Inc. and its affiliates.
"""AppTaskConfig — config.yaml loading and validation (harness-side).

config.yaml is pure declaration: wiring and metadata only, no behavior
bodies — hook functions live in the task's action.py, config only names
which function hangs on which hook point. Validation is pydantic's type
system plus ``extra="forbid"`` (typo protection); semantic mistakes (a
hook wired to an unlaunched app, a missing function) surface where they
are consumed — app-side hook registration, static precheck, expect.

Harness-side only: apps and in-sandbox code must not import this module
(pydantic/PyYAML are host dependencies; the in-sandbox hook registry reads
hooks.json with stdlib json).
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict


class AppTaskConfig(BaseModel):
    """One task's config.yaml.

    hooks:     {app_id: {hook_point: function_name}} — wiring only; the
               named functions live in the task's action file.
    action_path: full path to the task's action file. Optional in the
               yaml: a relative value resolves against the config file's
               directory; when absent, load() defaults to action.py beside
               the config.
    interface: {app_id: (bound_names,)} — the semantic surface the task
               addresses; the static precheck compares it against the app's
               persisted ``interface`` (the mutation budget).
    timeout_s / max_steps: budget; None means the harness default applies.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    title: str
    app_ids: tuple[str, ...]
    instruction: str
    action_path: Path
    hooks: dict[str, dict[str, str]] = {}
    interface: dict[str, tuple[str, ...]] = {}
    timeout_s: float | None = None
    max_steps: int | None = None

    @classmethod
    def load(cls, config_path: str | Path) -> AppTaskConfig:
        """Load from a config.yaml path; action_path resolves to a full path."""
        config_path = Path(config_path)
        raw = yaml.safe_load(config_path.read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"{config_path}: top level must be a mapping")
        base = config_path.parent
        declared = raw.get("action_path")
        if declared is None:
            raw["action_path"] = base / "action.py"
        else:
            action = Path(declared)
            raw["action_path"] = action if action.is_absolute() else base / action
        return cls.model_validate(raw)
