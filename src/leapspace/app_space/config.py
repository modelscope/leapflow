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
               named functions live in the task's action.py.
    interface: {app_id: (bound_names,)} — the semantic surface the task
               addresses; the static precheck compares it against the app's
               persisted ``interface`` (the mutation budget).
    timeout_s / max_steps: budget; None means the harness default applies.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    title: str
    apps: tuple[str, ...]
    instruction: str
    hooks: dict[str, dict[str, str]] = {}
    interface: dict[str, tuple[str, ...]] = {}
    timeout_s: float | None = None
    max_steps: int | None = None

    @classmethod
    def load(cls, path: str | Path) -> AppTaskConfig:
        """Load from a config.yaml path, or a task directory containing one."""
        path = Path(path)
        if path.is_dir():
            path = path / "config.yaml"
        raw = yaml.safe_load(path.read_text())
        if not isinstance(raw, dict):
            raise ValueError(f"{path}: top level must be a mapping")
        return cls.model_validate(raw)
