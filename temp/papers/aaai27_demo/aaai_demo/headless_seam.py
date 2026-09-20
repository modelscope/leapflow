"""Export a write-once record of the real offscreen Leapspace structural seam.

The demo uses this helper instead of presenting a fixture as if it were a real UI.
It creates two actual PyQt6 ``BaseLeapApp`` surfaces, lets their normal persistence
write the envelopes, and routes the observed rename through the production
``CapabilityObservationService``.  It has no engine, registry, approval, or plugin
mutation capability.
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

from aaai_demo.evidence import EvidenceBundleError, SCHEMA_VERSION, sha256_file


def _write_once(path: Path, value: dict[str, Any]) -> None:
    """Write one JSON record atomically without replacing an existing artifact."""
    if path.exists():
        raise EvidenceBundleError(f"write-once artifact already exists: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise EvidenceBundleError(f"cannot write headless seam artifact: {path}") from exc


def _artifact(path: Path, *, root: Path) -> dict[str, Any]:
    """Describe one local artifact relative to the seam run root."""
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _require_pyqt() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    """Load optional GUI dependencies only for the explicit headless-seam command."""
    try:
        from PyQt6.QtWidgets import QApplication, QLineEdit, QPushButton, QVBoxLayout, QWidget
        from leapspace.app_space.apps import _base as base_module
        from leapspace.app_space.apps._base import BaseLeapApp
    except ImportError as exc:
        raise EvidenceBundleError(
            "headless-seam requires the leapspace extra with PyQt6; run it in the PyQt-enabled environment"
        ) from exc
    return QApplication, QLineEdit, QPushButton, QVBoxLayout, QWidget, base_module, BaseLeapApp


def export_headless_seam(output_dir: Path) -> dict[str, Any]:
    """Run the real offscreen rename/negative-control seam and export its evidence.

    ``output_dir`` is write-once. The record contains no widget values or message
    contents: only structural identities, persisted evidence identifiers, and hashes.
    """
    target = output_dir.expanduser().resolve()
    if target.exists():
        raise EvidenceBundleError(f"write-once headless seam directory already exists: {target}")
    QApplication, QLineEdit, QPushButton, QVBoxLayout, QWidget, base_module, BaseLeapApp = _require_pyqt()
    target.mkdir(parents=True, exist_ok=False)
    original_state_dir = base_module.get_sandbox_state_dir
    try:
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        app = QApplication.instance() or QApplication([])

        class _ChatV1(BaseLeapApp):
            app_id = "chat_probe"
            app_title = "ChatProbe"
            version = "1.0"

            def build_ui(self) -> None:
                central = QWidget()
                layout = QVBoxLayout(central)
                layout.addWidget(self.bind(QLineEdit(), "message_input"))
                layout.addWidget(self.bind(QPushButton("Send"), self._send_name()))
                self.setCentralWidget(central)

            def _send_name(self) -> str:
                return "send_button"

            def reset(self, data: dict[str, Any]) -> None:
                return None

            def state(self) -> dict[str, Any]:
                return {}

        class _ChatV2(_ChatV1):
            version = "1.1"

            def _send_name(self) -> str:
                return "dispatch_button"

        class _ChatBenign(_ChatV1):
            version = "1.1-benign"

            def build_ui(self) -> None:
                central = QWidget()
                layout = QVBoxLayout(central)
                layout.addWidget(self.bind(QLineEdit(), "message_input"))
                layout.addWidget(self.bind(QPushButton("Send"), "send_button"))
                layout.addWidget(self.bind(QPushButton("Emoji"), "emoji_picker"))
                self.setCentralWidget(central)

        def _write_state(app_type: type[Any], name: str) -> Path:
            state_root = target / "states" / name
            base_module.get_sandbox_state_dir = lambda in_sandbox=True, system=None: state_root
            window = app_type()
            window.close()
            window.deleteLater()
            app.processEvents()
            return state_root / "chat_probe" / "state.json"

        before_path = _write_state(_ChatV1, "before")
        renamed_path = _write_state(_ChatV2, "renamed")
        benign_path = _write_state(_ChatBenign, "benign")

        from leapflow.learning.capability_observation import (
            CapabilityEvidenceClassifier,
            CapabilityObservationService,
        )
        from leapflow.storage.capability_observation_store import JsonCapabilityObservationStore
        from leapexp2.contracts import CapabilityPrecondition
        from leapexp2.leapspace_source import LeapSpaceEnvironmentSource

        precondition = CapabilityPrecondition(
            capability="chat.reply", bound_names=frozenset({"send_button"})
        )
        before = LeapSpaceEnvironmentSource(target / "states" / "before").read_snapshot("chat_probe")
        renamed_source = LeapSpaceEnvironmentSource(
            target / "states" / "renamed", workspace_root="demo-headless-seam"
        )
        renamed = renamed_source.read_snapshot("chat_probe")
        evidence = renamed_source.detect(before, renamed, precondition)
        if len(evidence) != 1 or evidence[0].evidence_kind != "interface_drift":
            raise EvidenceBundleError("real rename did not produce exactly one interface_drift observation")

        store = JsonCapabilityObservationStore(target / "observations.json")
        admitted_service = CapabilityObservationService(
            store,
            classifier=CapabilityEvidenceClassifier.from_kinds(
                ["unknown_tool", "interface_drift", "affordance_removed"]
            ),
        )
        observation = renamed_source.emit(admitted_service, evidence[0])
        requirements = admitted_service.requirements(min_count=1)
        if observation is None or len(requirements) != 1:
            raise EvidenceBundleError("admitted structural evidence did not produce one requirement")
        requirement = requirements[0]

        benign_source = LeapSpaceEnvironmentSource(target / "states" / "benign")
        benign = benign_source.read_snapshot("chat_probe")
        benign_evidence = benign_source.detect(before, benign, precondition)
        if benign_evidence:
            raise EvidenceBundleError("benign structural change unexpectedly produced evidence")

        default_store = JsonCapabilityObservationStore(target / "default_observations.json")
        default_service = CapabilityObservationService(default_store)
        default_record = renamed_source.emit(default_service, evidence[0])
        if default_record is not None:
            raise EvidenceBundleError("default classifier unexpectedly admitted environment evidence")

        store_artifacts = [_artifact(target / "observations.json", root=target)]
        default_store_path = target / "default_observations.json"
        if default_store_path.is_file():
            store_artifacts.append(_artifact(default_store_path, root=target))

        record = {
            "schema_version": SCHEMA_VERSION,
            "kind": "aaai_demo_headless_seam",
            "run_id": f"headless-seam-{uuid.uuid4().hex[:12]}",
            "created_at": time.time(),
            "evidence_level": "real_headless_structural_seam",
            "limitations": [
                "No framebuffer or pixel evidence is available in the offscreen lane.",
                "The seam creates an observation and requirement; it does not execute an agent, install a plugin, or approve a mutation.",
            ],
            "rename": {
                "before": _artifact(before_path, root=target),
                "after": _artifact(renamed_path, root=target),
                "evidence_kind": evidence[0].evidence_kind,
                "recovery_hint": evidence[0].recovery_hint,
                "suggestions": list(evidence[0].suggestions),
                "observation_id": str(observation.get("observation_id") or ""),
            },
            "benign_control": {
                "state": _artifact(benign_path, root=target),
                "evidence_count": 0,
            },
            # A refused observation does not need to create a store file; that absence
            # is itself part of the default-off evidence, not an export failure.
            "default_refusal": {
                "admitted": False,
                "store_created": default_store_path.is_file(),
            },
            "requirement": {
                "capability": requirement.capability,
                "origin": requirement.origin,
                "requirement_id": requirement.requirement_id,
            },
            "stores": store_artifacts,
        }
        _write_once(target / "headless_seam.json", record)
        return record
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise
    finally:
        base_module.get_sandbox_state_dir = original_state_dir


__all__ = ["export_headless_seam"]
