# Copyright (c) Alibaba, Inc. and its affiliates.
"""Configuration diagnostic checks (profile, LLM config, directory layout)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from leapflow.cli.doctor.protocol import Finding


class ProfileConfigCheck:
    """Verify that the active profile exists and has a valid manifest."""

    name = "Profile config"
    section = "config"

    def __init__(self, profile_layout: Any) -> None:
        self._layout = profile_layout

    async def check(self, should_fix: bool = False) -> Finding:
        f = Finding()
        root: Path = self._layout.root
        if not root.is_dir():
            if should_fix:
                root.mkdir(parents=True, exist_ok=True)
                f.fix(f"Created missing profile directory: {root}")
            else:
                f.error(f"Profile directory missing: {root}")
                return f

        manifest = self._layout.manifest_path
        if not manifest.is_file():
            if should_fix:
                self._layout.ensure()
                f.fix(f"Bootstrapped profile manifest: {manifest}")
            else:
                f.error(f"Profile manifest missing: {manifest}")
        else:
            f.pass_()
        return f


class LLMConfigCheck:
    """Verify that LLM provider, model, and API key reference are configured."""

    name = "LLM configuration"
    section = "config"

    def __init__(self, settings: Any) -> None:
        self._settings = settings

    async def check(self, should_fix: bool = False) -> Finding:
        f = Finding()
        s = self._settings

        if not s.llm_model:
            f.error("llm.model is not configured — run `leap config llm set --model <name>`")
        else:
            f.pass_()

        if not s.llm_base_url:
            f.error("llm.base_url is not configured — run `leap config llm set --base-url <url>`")
        else:
            f.pass_()

        if not s.has_llm_credentials:
            f.warn("LLM API key is empty — run `leap config llm key` to set it")
        else:
            f.pass_()

        return f


class PathLayoutCheck:
    """Verify that critical profile directories exist (optionally create them)."""

    name = "Path layout"
    section = "config"

    def __init__(self, profile_layout: Any) -> None:
        self._layout = profile_layout

    async def check(self, should_fix: bool = False) -> Finding:
        f = Finding()
        required_dirs: list[tuple[str, Path]] = [
            ("config", self._layout.config_dir),
            ("db", self._layout.db_dir),
            ("memory", self._layout.memory_dir),
            ("skills", self._layout.skills_dir),
            ("plugins", self._layout.plugins_dir),
            ("audit", self._layout.audit_dir),
            ("runtime", self._layout.runtime_dir),
        ]
        for label, path in required_dirs:
            if path.is_dir():
                f.pass_()
            elif should_fix:
                path.mkdir(parents=True, exist_ok=True)
                f.fix(f"Created missing directory: {label} ({path})")
            else:
                f.error(f"Missing directory: {label} ({path})")
        return f
