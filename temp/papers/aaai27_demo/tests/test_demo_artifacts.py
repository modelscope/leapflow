"""Tests for AAAI demo evidence packaging without a live GUI or LLM."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


_DEMO_ROOT = Path(__file__).resolve().parents[1]
_EVO2_ROOT = _DEMO_ROOT.parents[1] / "leapspace_exp" / "evo-02"
for path in (str(_DEMO_ROOT), str(_EVO2_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from aaai_demo.evidence import (  # noqa: E402
    EvidenceBundleError,
    build_evidence_bundle,
    export_evidence_bundle,
)
from aaai_demo.headless_seam import export_headless_seam  # noqa: E402
from aaai_demo.poster import render_poster_source  # noqa: E402
from aaai_demo.render import render_demo_package  # noqa: E402


_ARMS = (
    ("unchanged_baseline", "none", "none", 0),
    ("irrelevant_delta_rejected", "none", "none", 0),
    ("satisfied_by_catalog", "reuse", "reuse", 0),
    ("unmet_traverses_lifecycle", "acquire", "acquire", 1),
)


def _run(tmp_path: Path) -> Path:
    run = tmp_path / "c2-test"
    run.mkdir()
    manifest = {
        "run_id": "c2-test",
        "lane": "L1",
        "family": "C2_coevolution_counterfactual",
        "protocol_id": "LEAPSPACE-EVO-02",
        "experiment_version": "v1",
        "subject": {"commit": "abc123"},
        "substitutions": {"not_exercised": "approval prompt, real install, daemon"},
    }
    summary = {"arms_total": 4, "arms_correct": 4, "mutations": 1, "effect_confirmed": 1}
    records = []
    for index, (arm, expected, action, verified) in enumerate(_ARMS):
        records.append({
            "arm": arm,
            "expected_action": expected,
            "action": action,
            "ok": True,
            "proposed": 1 if arm == "unmet_traverses_lifecycle" else 0,
            "admitted": 1 if arm == "unmet_traverses_lifecycle" else 0,
            "requirements": 1 if arm in {"satisfied_by_catalog", "unmet_traverses_lifecycle"} else 0,
            "authorised": arm != "irrelevant_delta_rejected",
            "effect_verified": verified,
            "effect_refuted": 0,
            "profile_root": str(run / "isolated" / f"profile-{index}"),
            "notes": f"evidence for {arm}",
        })
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run / "records.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return run


def _fixture() -> Path:
    return _DEMO_ROOT / "demo_fixtures" / "task-001-structural-drifts.json"


def _signal_archive(tmp_path: Path, *, complete: bool = True) -> Path:
    archive = tmp_path / "signal-archive"
    archive.mkdir()
    paths = {
        "recording": "recording.mp4",
        "cursor_path": "cursor.jsonl",
        "state_snapshot": "chat/state.json",
        "event_log": "chat/events.jsonl",
        "expect_stdout": "expect.stdout",
    }
    for path in paths.values():
        artifact = archive / path
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("evidence", encoding="utf-8")
    artifacts = [
        {"kind": kind, "path": path, "sensitive": kind in {"recording", "state_snapshot"}}
        for kind, path in paths.items()
        if complete or kind != "event_log"
    ]
    manifest = {
        "schema_version": 1,
        "kind": "leapspace_signal_archive",
        "run_id": "signal-test",
        "expect": {"verdict": "pass", "exit_code": 0},
        "artifacts": artifacts,
    }
    path = archive / "signal_archive.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def _headless_seam(tmp_path: Path) -> Path:
    path = tmp_path / "headless_seam.json"
    path.write_text(json.dumps({
        "schema_version": 1,
        "kind": "aaai_demo_headless_seam",
        "run_id": "headless-test",
        "rename": {"evidence_kind": "interface_drift"},
        "benign_control": {"evidence_count": 0},
        "default_refusal": {"admitted": False},
        "requirement": {"capability": "chat.reply", "origin": "environment_probe"},
    }), encoding="utf-8")
    return path


def test_bundle_preserves_l1_boundaries_and_counterfactual_order(tmp_path: Path) -> None:
    bundle = build_evidence_bundle(_run(tmp_path), drift_fixture=_fixture())

    assert [arm["arm"] for arm in bundle["arms"]] == [item[0] for item in _ARMS]
    unmet = bundle["arms"][-1]
    assert unmet["outcome"] == "acquire_hypothesis"
    assert unmet["approval"] == "not_exercised"
    assert unmet["registry_mutation"] == "not_exercised"
    assert unmet["effect"]["independent_oracle"] == "not_available"
    assert "End-to-end Leapspace agent execution." in bundle["claims"]["not_supported"]
    assert len(bundle["drift_fixture"]["drifts"]) == 3
    assert bundle["signal_archive"]["status"] == "not_supplied"
    assert bundle["headless_seam"]["status"] == "not_supplied"


def test_bundle_binds_complete_signal_archive_and_headless_seam(tmp_path: Path) -> None:
    bundle = build_evidence_bundle(
        _run(tmp_path),
        signal_archive=_signal_archive(tmp_path),
        headless_seam=_headless_seam(tmp_path),
    )

    assert bundle["signal_archive"]["status"] == "available"
    assert bundle["signal_archive"]["expect"] == {"verdict": "pass", "exit_code": 0}
    assert len(bundle["signal_archive"]["artifacts"]) == 5
    assert bundle["headless_seam"]["status"] == "available"
    assert bundle["headless_seam"]["requirement"]["origin"] == "environment_probe"


def test_incomplete_signal_archive_is_visible_not_upgraded_to_pass(tmp_path: Path) -> None:
    bundle = build_evidence_bundle(_run(tmp_path), signal_archive=_signal_archive(tmp_path, complete=False))

    assert bundle["signal_archive"]["status"] == "incomplete"
    assert "event_log" in bundle["signal_archive"]["missing"]


def test_headless_seam_export_records_real_widget_evidence_when_pyqt_is_available(tmp_path: Path) -> None:
    pytest.importorskip("PyQt6")

    record = export_headless_seam(tmp_path / "headless")

    assert record["rename"]["evidence_kind"] == "interface_drift"
    assert record["benign_control"]["evidence_count"] == 0
    assert record["default_refusal"]["admitted"] is False
    assert record["requirement"]["origin"] == "environment_probe"
    seam_path = tmp_path / "headless" / "headless_seam.json"
    assert seam_path.is_file()
    assert build_evidence_bundle(_run(tmp_path), headless_seam=seam_path)["headless_seam"]["status"] == "available"


def test_export_and_render_are_write_once(tmp_path: Path) -> None:
    source = _run(tmp_path)
    bundle_dir = tmp_path / "bundle"
    bundle = export_evidence_bundle(
        source,
        bundle_dir,
        drift_fixture=_fixture(),
        signal_archive=_signal_archive(tmp_path),
        headless_seam=_headless_seam(tmp_path),
    )

    assert (bundle_dir / "bundle.json").is_file()
    assert (bundle_dir / "checksums.sha256").is_file()
    assert (bundle_dir / "raw" / "records.jsonl").is_file()
    assert (bundle_dir / "raw" / "signal_archive.json").is_file()
    assert (bundle_dir / "raw" / "headless_seam.json").is_file()
    with pytest.raises(EvidenceBundleError, match="already exists"):
        export_evidence_bundle(source, bundle_dir, drift_fixture=_fixture())

    render_dir = tmp_path / "render"
    paths = render_demo_package(bundle, render_dir)
    html = paths["index"].read_text(encoding="utf-8")
    assert "CE-X four-arm matrix" in html
    assert "End-to-end Leapspace agent execution." in html
    storyboard = json.loads(paths["storyboard"].read_text(encoding="utf-8"))
    assert storyboard["duration_s"] == 300
    assert storyboard["segments"][-1]["end_s"] == 300
    assert "Signal archive" in html
    assert "Headless seam" in html
    poster_dir = tmp_path / "poster"
    poster = render_poster_source(bundle, poster_dir)
    assert "PROVISIONAL SOURCE" in poster["poster"].read_text(encoding="utf-8")
    assert json.loads(poster["format"].read_text(encoding="utf-8"))["official_size"] == (
        "pending_chairs_exhibit_format_information"
    )
    with pytest.raises(EvidenceBundleError, match="already exists"):
        render_poster_source(bundle, poster_dir)
    with pytest.raises(EvidenceBundleError, match="already exists"):
        render_demo_package(bundle, render_dir)
