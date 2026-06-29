"""Bundle file I/O — writes auxiliary Knowledge Bundle artifacts alongside SKILL.md.

Each method creates parent directories as needed. All writes are idempotent.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class BundleFiles:
    """Auxiliary files to write alongside SKILL.md during save."""

    anchors: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None
    recovery_scripts: Dict[str, str] = field(default_factory=dict)
    changelog_entries: List[str] = field(default_factory=list)
    trajectory_summaries: Dict[str, str] = field(default_factory=dict)


@dataclass
class BundleContext:
    """Read-back view of auxiliary bundle files in a skill folder."""

    folder_path: Path = field(default_factory=lambda: Path("."))
    anchors_yaml: str = ""
    meta_yaml: str = ""
    recovery_scripts: List[Path] = field(default_factory=list)
    verification_tests: List[Path] = field(default_factory=list)
    visual_templates: List[Path] = field(default_factory=list)
    has_bundle: bool = False


class BundleWriter:
    """Writes auxiliary Knowledge Bundle files into a skill folder."""

    def write_bundle(self, folder: Path, bundle: BundleFiles) -> None:
        if bundle.anchors is not None:
            self.write_anchors(folder, bundle.anchors)
        if bundle.meta is not None:
            self.write_meta(folder, bundle.meta)
        for name, content in bundle.recovery_scripts.items():
            self.write_recovery_script(folder, name, content)
        if bundle.changelog_entries:
            version = (bundle.meta or {}).get("version", 1)
            self.append_changelog(folder, version, bundle.changelog_entries)
        for traj_id, summary in bundle.trajectory_summaries.items():
            self.write_trajectory_summary(folder, traj_id, summary)

    def write_anchors(self, folder: Path, anchors: Dict[str, Any]) -> None:
        path = folder / "resources" / "anchors.yaml"
        self._write_yaml(path, anchors)

    def write_meta(self, folder: Path, meta: Dict[str, Any]) -> None:
        path = folder / "meta.yaml"
        self._write_yaml(path, meta)

    def write_recovery_script(self, folder: Path, name: str, content: str) -> None:
        path = folder / "scripts" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        path.chmod(0o755)

    def write_verification_test(self, folder: Path, step: int, content: str) -> None:
        path = folder / "tests" / f"verify_step_{step}.sh"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        path.chmod(0o755)

    def save_visual_template(self, folder: Path, name: str, image_data: bytes) -> None:
        path = folder / "assets" / "visual_templates" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(image_data)

    def save_step_screenshot(self, folder: Path, step: int, image_data: bytes) -> None:
        path = folder / "assets" / "step_states" / f"step_{step}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(image_data)

    def append_changelog(self, folder: Path, version: int, entries: List[str]) -> None:
        path = folder / "CHANGELOG.md"
        block = f"## v{version}\n\n" + "\n".join(f"- {e}" for e in entries) + "\n\n"
        if path.exists():
            existing = path.read_text(encoding="utf-8")
            path.write_text(block + existing, encoding="utf-8")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(block, encoding="utf-8")

    def write_trajectory_summary(self, folder: Path, traj_id: str, summary: str) -> None:
        path = folder / "references" / f"{traj_id}_summary.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(summary, encoding="utf-8")

    @staticmethod
    def load_bundle_context(folder: Path) -> BundleContext:
        """Scan a skill folder for auxiliary bundle files."""
        ctx = BundleContext(folder_path=folder)

        anchors_path = folder / "resources" / "anchors.yaml"
        if anchors_path.exists():
            ctx.anchors_yaml = anchors_path.read_text(encoding="utf-8")
            ctx.has_bundle = True

        meta_path = folder / "meta.yaml"
        if meta_path.exists():
            ctx.meta_yaml = meta_path.read_text(encoding="utf-8")
            ctx.has_bundle = True

        scripts_dir = folder / "scripts"
        if scripts_dir.is_dir():
            ctx.recovery_scripts = sorted(scripts_dir.glob("recover_*"))
            if ctx.recovery_scripts:
                ctx.has_bundle = True

        tests_dir = folder / "tests"
        if tests_dir.is_dir():
            ctx.verification_tests = sorted(tests_dir.glob("verify_*"))
            if ctx.verification_tests:
                ctx.has_bundle = True

        templates_dir = folder / "assets" / "visual_templates"
        if templates_dir.is_dir():
            ctx.visual_templates = sorted(templates_dir.iterdir())
            if ctx.visual_templates:
                ctx.has_bundle = True

        return ctx

    @staticmethod
    def _write_yaml(path: Path, data: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        text = yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)
        path.write_text(text, encoding="utf-8")
