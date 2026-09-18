"""Build a conservative, immutable evidence bundle for the AAAI demo.

This module is deliberately publication-local. It reads existing experiment
artifacts and trajectory recordings; it never drives a model, approves a
mutation, or changes a registry. Its output distinguishes what was observed
from what the L1 CE-X instrument intentionally did not exercise.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 1
CE_X_ARMS = (
    "unchanged_baseline",
    "irrelevant_delta_rejected",
    "satisfied_by_catalog",
    "unmet_traverses_lifecycle",
)
_REQUIRED_RECORD_FIELDS = frozenset({"arm", "expected_action", "action", "ok"})


class EvidenceBundleError(ValueError):
    """Raised when input cannot support a truthful, replayable demo bundle."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a regular file without loading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_mapping(path: Path) -> dict[str, Any]:
    """Read one JSON object or raise an evidence-specific error."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceBundleError(f"cannot read JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise EvidenceBundleError(f"expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL records and reject malformed or non-object rows."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise EvidenceBundleError(f"cannot read records: {path}") from exc
    records: list[dict[str, Any]] = []
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvidenceBundleError(f"invalid JSONL at {path}:{number}") from exc
        if not isinstance(record, dict):
            raise EvidenceBundleError(f"non-object JSONL record at {path}:{number}")
        records.append(record)
    return records


def _require_file(directory: Path, name: str) -> Path:
    """Return a required regular file beneath an experiment run directory."""
    path = directory / name
    if not path.is_file():
        raise EvidenceBundleError(f"missing required run artifact: {path}")
    return path


def _artifact(path: Path, *, root: Path) -> dict[str, Any]:
    """Return immutable metadata for a source artifact without copying it."""
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = Path(path.name)
    return {
        "path": relative.as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _validate_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate the canonical CE-X arm set and preserve its declared order."""
    by_arm: dict[str, dict[str, Any]] = {}
    for raw in records:
        missing = _REQUIRED_RECORD_FIELDS - set(raw)
        if missing:
            raise EvidenceBundleError(f"counterfactual record misses fields: {sorted(missing)}")
        arm = str(raw["arm"])
        if arm in by_arm:
            raise EvidenceBundleError(f"duplicate CE-X arm record: {arm}")
        by_arm[arm] = dict(raw)
    unknown = set(by_arm) - set(CE_X_ARMS)
    missing = set(CE_X_ARMS) - set(by_arm)
    if unknown or missing:
        raise EvidenceBundleError(
            f"CE-X must contain exactly {CE_X_ARMS}; missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    return [by_arm[arm] for arm in CE_X_ARMS]


def _effect_status(record: Mapping[str, Any]) -> dict[str, str]:
    """Describe effect evidence without upgrading handler reports into an oracle."""
    if int(record.get("effect_verified") or 0) > 0:
        return {
            "status": "handler_reported_effect_confirmed",
            "independent_oracle": "not_available",
            "note": "The CE-X L1 runner reports a handler-observed effect; no Leapspace expect() oracle was run.",
        }
    if int(record.get("effect_refuted") or 0) > 0:
        return {
            "status": "effect_refuted",
            "independent_oracle": "not_available",
            "note": "The instrument reported a refuted effect; this arm must not be presented as successful.",
        }
    return {
        "status": "not_exercised",
        "independent_oracle": "not_available",
        "note": "No effect verification exists for this arm in the L1 CE-X run.",
    }


def _arm_label(record: Mapping[str, Any]) -> str:
    """Name the non-mutation outcome in a form suitable for a compact matrix."""
    arm = str(record["arm"])
    if arm == "unchanged_baseline":
        return "no_op"
    if arm == "irrelevant_delta_rejected":
        return "rejected"
    if arm == "satisfied_by_catalog":
        return "catalog_reuse"
    return "acquire_hypothesis"


def _normalise_arm(record: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Map one CE-X record to a demo row while retaining its evidentiary boundary."""
    substitutions = manifest.get("substitutions")
    substitutions = dict(substitutions) if isinstance(substitutions, Mapping) else {}
    not_exercised = str(substitutions.get("not_exercised") or "")
    arm = str(record["arm"])
    action = str(record["action"])
    return {
        "arm": arm,
        "outcome": _arm_label(record),
        "expected_action": str(record["expected_action"]),
        "actual_action": action,
        "matches_oracle": bool(record["ok"]),
        "drift_relevance": (
            "unchanged" if arm == "unchanged_baseline"
            else "irrelevant" if arm == "irrelevant_delta_rejected"
            else "task_relevant"
        ),
        "resolution": (
            "not_required" if arm in {"unchanged_baseline", "irrelevant_delta_rejected"}
            else "incumbent_satisfies" if arm == "satisfied_by_catalog"
            else "unmet_requirement"
        ),
        "proposal_count": int(record.get("proposed") or 0),
        "admitted_count": int(record.get("admitted") or 0),
        "requirement_count": int(record.get("requirements") or 0),
        "authorised": bool(record.get("authorised", False)),
        "profile_root": str(record.get("profile_root") or ""),
        "approval": "not_exercised",
        "registry_mutation": "not_exercised",
        "lifecycle": "mechanism_level_only" if action == "acquire" else "not_applicable",
        "effect": _effect_status(record),
        "notes": str(record.get("notes") or ""),
        "boundary": not_exercised,
    }


def _trajectory_metadata(trajectory_dir: Path | None) -> dict[str, Any]:
    """Index legacy optional trajectory assets without asserting an oracle verdict."""
    if trajectory_dir is None:
        return {"status": "not_supplied", "media": [], "turns": []}
    root = trajectory_dir.expanduser().resolve()
    if not root.is_dir():
        raise EvidenceBundleError(f"trajectory directory does not exist: {root}")
    media: list[dict[str, Any]] = []
    for name in ("recording.mp4", "cursor.jsonl"):
        path = root / name
        if path.is_file():
            media.append(_artifact(path, root=root))
    turns: list[dict[str, Any]] = []
    for turn_dir in sorted(path for path in root.glob("turn-*") if path.is_dir()):
        files = [
            _artifact(path, root=root)
            for path in sorted(turn_dir.iterdir())
            if path.is_file() and path.name in {"action.json", "app_state.json", "screenshot.png"}
        ]
        turns.append({"turn": turn_dir.name, "artifacts": files})
    return {
        "status": "available" if media or turns else "empty",
        "pixel_capture": "available" if any(item["path"] == "recording.mp4" for item in media) else "not_available",
        "media": media,
        "turns": turns,
    }


_SIGNAL_ARCHIVE_REQUIRED_KINDS = frozenset({
    "recording", "cursor_path", "state_snapshot", "event_log", "expect_stdout",
})


def _safe_archive_artifact(root: Path, raw: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """Hash one declared signal artifact without allowing an archive-path escape."""
    kind = str(raw.get("kind") or "").strip()
    relative = Path(str(raw.get("path") or ""))
    if not kind or not relative.parts or relative.is_absolute() or ".." in relative.parts:
        return None, kind or "invalid_artifact"
    path = root / relative
    if not path.is_file():
        return None, kind
    artifact = _artifact(path, root=root)
    artifact.update({"kind": kind, "sensitive": bool(raw.get("sensitive", False))})
    return artifact, ""


def load_signal_archive(path: Path | None) -> dict[str, Any]:
    """Load a declared signal-mode archive without upgrading missing evidence to PASS.

    The manifest is deliberately file-oriented. It binds a recording, state/event
    evidence, and the independent ``expect()`` verdict to one run while allowing a
    renderer to show an incomplete archive honestly instead of failing the whole CE-X
    bundle.
    """
    if path is None:
        return {"status": "not_supplied", "artifacts": [], "missing": []}
    manifest_path = path.expanduser().resolve()
    manifest = _read_mapping(manifest_path)
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("kind") != "leapspace_signal_archive":
        raise EvidenceBundleError("signal archive must declare the current schema and kind")
    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise EvidenceBundleError("signal archive artifacts must be a list")
    artifacts: list[dict[str, Any]] = []
    missing: list[str] = []
    for raw in raw_artifacts:
        if not isinstance(raw, Mapping):
            raise EvidenceBundleError("signal archive contains a non-object artifact")
        artifact, problem = _safe_archive_artifact(manifest_path.parent, raw)
        if artifact is None:
            missing.append(problem)
        else:
            artifacts.append(artifact)
    present = {str(item["kind"]) for item in artifacts}
    missing.extend(sorted(_SIGNAL_ARCHIVE_REQUIRED_KINDS - present))
    expect = manifest.get("expect")
    expect = dict(expect) if isinstance(expect, Mapping) else {}
    verdict = str(expect.get("verdict") or "not_recorded")
    exit_code = expect.get("exit_code")
    passed = verdict == "pass" and exit_code == 0 and not missing
    return {
        "status": "available" if passed else "incomplete",
        "run_id": str(manifest.get("run_id") or ""),
        "expect": {"verdict": verdict, "exit_code": exit_code},
        "artifacts": artifacts,
        "missing": sorted(set(missing)),
        "pixel_capture": "available" if "recording" in present else "not_available",
    }


def load_headless_seam(path: Path | None) -> dict[str, Any]:
    """Load a write-once real-widget seam record, or name its absence explicitly."""
    if path is None:
        return {"status": "not_supplied"}
    record_path = path.expanduser().resolve()
    record = _read_mapping(record_path)
    if record.get("schema_version") != SCHEMA_VERSION or record.get("kind") != "aaai_demo_headless_seam":
        raise EvidenceBundleError("headless seam record must declare the current schema and kind")
    required = {"rename", "benign_control", "default_refusal", "requirement"}
    missing = sorted(key for key in required if key not in record)
    return {
        "status": "available" if not missing else "incomplete",
        "artifact": _artifact(record_path, root=record_path.parent),
        "run_id": str(record.get("run_id") or ""),
        "rename": dict(record.get("rename") or {}),
        "benign_control": dict(record.get("benign_control") or {}),
        "default_refusal": dict(record.get("default_refusal") or {}),
        "requirement": dict(record.get("requirement") or {}),
        "missing": missing,
    }


def load_drift_fixture(path: Path | None) -> dict[str, Any] | None:
    """Load and validate a declarative, reversible interface-drift fixture."""
    if path is None:
        return None
    fixture = _read_mapping(path)
    drifts = fixture.get("drifts")
    if fixture.get("schema_version") != SCHEMA_VERSION or not isinstance(drifts, list) or not drifts:
        raise EvidenceBundleError("drift fixture must declare schema_version and a non-empty drifts list")
    seen: set[str] = set()
    for drift in drifts:
        if not isinstance(drift, Mapping):
            raise EvidenceBundleError("drift fixture contains a non-object drift")
        drift_id = str(drift.get("drift_id") or "")
        if not drift_id or drift_id in seen:
            raise EvidenceBundleError("each drift fixture entry requires a unique drift_id")
        seen.add(drift_id)
        if not isinstance(drift.get("before"), Mapping) or not isinstance(drift.get("after"), Mapping):
            raise EvidenceBundleError(f"drift {drift_id} must contain before and after snapshots")
    return fixture


def build_evidence_bundle(
    run_dir: Path,
    *,
    drift_fixture: Path | None = None,
    trajectory_dir: Path | None = None,
    signal_archive: Path | None = None,
    headless_seam: Path | None = None,
) -> dict[str, Any]:
    """Normalise a completed CE-X run into a conservative publication artifact."""
    root = run_dir.expanduser().resolve()
    if not root.is_dir():
        raise EvidenceBundleError(f"CE-X run directory does not exist: {root}")
    manifest_path = _require_file(root, "manifest.json")
    records_path = _require_file(root, "records.jsonl")
    summary_path = _require_file(root, "summary.json")
    manifest = _read_mapping(manifest_path)
    summary = _read_mapping(summary_path)
    records = _validate_records(_read_jsonl(records_path))
    lane = str(manifest.get("lane") or "")
    if lane != "L1":
        raise EvidenceBundleError(f"this demo normalizer currently accepts L1 CE-X runs, got {lane!r}")
    profiles = {str(record.get("profile_root") or "") for record in records}
    if "" in profiles or len(profiles) != len(CE_X_ARMS):
        raise EvidenceBundleError("CE-X arms must retain four distinct non-empty profile roots")
    fixture = load_drift_fixture(drift_fixture)
    arms = [_normalise_arm(record, manifest) for record in records]
    trace = [
        {
            "stage": "observe",
            "arm": arm["arm"],
            "title": f"OBSERVE · {arm['drift_relevance']}",
            "summary": arm["notes"],
            "severity": "info" if arm["matches_oracle"] else "alert",
        }
        for arm in arms
    ] + [
        {
            "stage": "decide",
            "arm": arm["arm"],
            "title": f"DECIDE · {arm['outcome']}",
            "summary": f"expected={arm['expected_action']}; actual={arm['actual_action']}",
            "severity": "notable" if arm["outcome"] in {"rejected", "catalog_reuse"} else "info",
        }
        for arm in arms
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "aaai_demo_evidence_bundle",
        "created_at": time.time(),
        "evidence_level": "L1",
        "run": {
            "run_id": str(manifest.get("run_id") or root.name),
            "family": str(manifest.get("family") or ""),
            "protocol_id": str(manifest.get("protocol_id") or ""),
            "experiment_version": str(manifest.get("experiment_version") or ""),
            "subject": dict(manifest.get("subject") or {}),
            "manifest": _artifact(manifest_path, root=root),
            "records": _artifact(records_path, root=root),
            "summary": _artifact(summary_path, root=root),
            "substitutions": dict(manifest.get("substitutions") or {}),
        },
        "claims": {
            "supported": [
                "The controlled L1 CE-X instrument distinguishes no-op, rejection, catalog reuse, and an acquisition hypothesis.",
                "Each counterfactual arm uses an isolated profile root and emits a durable run record.",
            ],
            "not_supported": [
                "End-to-end Leapspace agent execution.",
                "A real plugin installation or approval decision.",
                "Independent expect() oracle confirmation for the L1 CE-X acquisition arm.",
                "Longitudinal governed self-evolution in the daemon runtime.",
            ],
        },
        "summary": summary,
        "arms": arms,
        "causal_trace": trace,
        # ``trajectory`` is a legacy media index. Only ``signal_archive`` binds media
        # to state/event evidence and an independent expect() verdict.
        "trajectory": _trajectory_metadata(trajectory_dir),
        "signal_archive": load_signal_archive(signal_archive),
        "headless_seam": load_headless_seam(headless_seam),
        "drift_fixture": fixture,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Write one JSON document atomically without overwriting an existing artifact."""
    if path.exists():
        raise EvidenceBundleError(f"write-once artifact already exists: {path}")
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise EvidenceBundleError(f"cannot write evidence artifact: {path}") from exc


def _copy_raw(path: Path, destination: Path) -> None:
    """Copy one selected source artifact, refusing to replace an existing copy."""
    if destination.exists():
        raise EvidenceBundleError(f"write-once raw artifact already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy2(path, destination)
    except OSError as exc:
        raise EvidenceBundleError(f"cannot copy evidence artifact: {path}") from exc


def export_evidence_bundle(
    run_dir: Path,
    output_dir: Path,
    *,
    drift_fixture: Path | None = None,
    trajectory_dir: Path | None = None,
    signal_archive: Path | None = None,
    headless_seam: Path | None = None,
    include_media: bool = False,
) -> dict[str, Any]:
    """Create a write-once derived bundle and return its normalized content."""
    target = output_dir.expanduser().resolve()
    if target.exists():
        raise EvidenceBundleError(f"write-once bundle directory already exists: {target}")
    bundle = build_evidence_bundle(
        run_dir,
        drift_fixture=drift_fixture,
        trajectory_dir=trajectory_dir,
        signal_archive=signal_archive,
        headless_seam=headless_seam,
    )
    target.mkdir(parents=True, exist_ok=False)
    try:
        source = Path(run_dir).expanduser().resolve()
        raw_dir = target / "raw"
        for name in ("manifest.json", "records.jsonl", "summary.json"):
            _copy_raw(source / name, raw_dir / name)
        if drift_fixture is not None:
            _copy_raw(drift_fixture.expanduser().resolve(), raw_dir / "drift_fixture.json")
        if signal_archive is not None:
            _copy_raw(signal_archive.expanduser().resolve(), raw_dir / "signal_archive.json")
        if headless_seam is not None:
            _copy_raw(headless_seam.expanduser().resolve(), raw_dir / "headless_seam.json")
        if include_media and trajectory_dir is not None:
            trajectory = trajectory_dir.expanduser().resolve()
            for path in sorted(trajectory.rglob("*")):
                if not path.is_file():
                    continue
                relative = path.relative_to(trajectory)
                if path.name in {"recording.mp4", "cursor.jsonl", "action.json", "app_state.json", "screenshot.png"}:
                    _copy_raw(path, raw_dir / "trajectory" / relative)
        bundle["packaged_files"] = [
            _artifact(path, root=target)
            for path in sorted(target.rglob("*"))
            if path.is_file()
        ]
        _write_json(target / "bundle.json", bundle)
        lines = [
            f"{sha256_file(path)}  {path.relative_to(target).as_posix()}"
            for path in sorted(target.rglob("*"))
            if path.is_file()
        ]
        (target / "checksums.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise
    return bundle


def viewspec_data(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Project bundle facts into the existing read-only Causal Trace template shape."""
    arms = [dict(item) for item in bundle.get("arms") or [] if isinstance(item, Mapping)]
    trace = [dict(item) for item in bundle.get("causal_trace") or [] if isinstance(item, Mapping)]
    summary = dict(bundle.get("summary") or {})
    run = dict(bundle.get("run") or {})
    return {
        "title": f"AAAI Causal Trace · {run.get('run_id') or 'unknown run'}",
        "causal_trace": {
            "evidence_level": str(bundle.get("evidence_level") or ""),
            "arms": arms,
            "timeline": trace,
            "summary": {
                "arms_total": summary.get("arms_total", len(arms)),
                "arms_correct": summary.get("arms_correct", 0),
                "mutations": summary.get("mutations", 0),
                "effect_confirmed": summary.get("effect_confirmed", 0),
            },
            "manifest": {
                "run_id": run.get("run_id", ""),
                "protocol_id": run.get("protocol_id", ""),
                "subject_commit": dict(run.get("subject") or {}).get("commit", ""),
                "records_sha256": dict(run.get("records") or {}).get("sha256", ""),
            },
            "claims": dict(bundle.get("claims") or {}),
            "signal_archive": dict(bundle.get("signal_archive") or {}),
            "headless_seam": dict(bundle.get("headless_seam") or {}),
        },
    }


__all__ = [
    "CE_X_ARMS",
    "SCHEMA_VERSION",
    "EvidenceBundleError",
    "build_evidence_bundle",
    "export_evidence_bundle",
    "load_drift_fixture",
    "load_headless_seam",
    "load_signal_archive",
    "sha256_file",
    "viewspec_data",
]
