# Copyright (c) Alibaba, Inc. and its affiliates.
"""End-to-end CLI subcommand, exit code, and JSON output tests."""

from __future__ import annotations

import io
import json
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from benchmarks import cli
from benchmarks.cli import (
    EXIT_BLOCKED,
    EXIT_CONDITIONAL,
    EXIT_READY,
    EXIT_USAGE,
    main,
)


def _invoke(*argv: str) -> tuple[int, str, str]:
    out = io.StringIO()
    err = io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


def _json(stdout: str) -> dict:
    return json.loads(stdout)


@pytest.fixture(autouse=True)
def _isolate_benchmark_evidence(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Keep CLI benchmark evidence inside pytest's temporary directory."""
    monkeypatch.setattr(cli, "_profile_evidence_root", lambda: str(tmp_path / "evidence"))


MANIFEST_TEMPLATE = """\
id: {bench_id}
version: "1.0.0"
adapter: {adapter}
tier: 0
required: {required}
seeds: [42]
scenarios:
  - scenario_id: one
    name: Scenario one
"""


def _write_manifest(tmp_path: Path, bench_id: str, adapter: str, required: bool = False) -> Path:
    manifest = tmp_path / f"{bench_id}.yaml"
    manifest.write_text(
        MANIFEST_TEMPLATE.format(
            bench_id=bench_id,
            adapter=adapter,
            required=str(required).lower(),
        ),
        encoding="utf-8",
    )
    return manifest


def test_list_lists_adapters_and_native_benchmarks_in_json() -> None:
    code, stdout, _ = _invoke("list", "--profile", "tier0", "--json")

    assert code == EXIT_READY
    data = _json(stdout)
    assert "native_harmful_instruction" in data["adapters"]
    assert any(bench["id"] == "native_harmful_instruction" for bench in data["benchmarks"])
    assert [profile["id"] for profile in data["profiles"]][0] == "tier0"


def test_doctor_tier0_is_ready_in_json() -> None:
    code, stdout, _ = _invoke("doctor", "--profile", "tier0", "--json")

    assert code == EXIT_READY
    data = _json(stdout)
    assert data["ready"] is True


def test_run_native_benchmark_returns_ready_with_json(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    code, stdout, _ = _invoke(
        "run",
        "--profile",
        "tier0",
        "--benchmark",
        "native_harmful_instruction",
        "--json",
        "--output",
        str(output),
    )

    assert code == EXIT_READY
    data = _json(stdout)
    assert data["results"][0]["benchmark_id"] == "native_harmful_instruction"
    assert data["results"][0]["passed"] == data["results"][0]["total_trials"]
    assert output.exists()
    stored = json.loads(output.read_text(encoding="utf-8"))
    assert stored["schema_version"] == 1
    config = stored["results"][0]["config"]
    assert config["run_id"].startswith("run-")
    assert config["evidence_root"] == str(tmp_path / "evidence")


def test_resume_skips_completed_trials(tmp_path: Path) -> None:
    first_output = tmp_path / "run1.json"
    first_code, _, _ = _invoke(
        "run",
        "--profile",
        "tier0",
        "--benchmark",
        "native_harmful_instruction",
        "--json",
        "--output",
        str(first_output),
    )
    assert first_code == EXIT_READY

    code, stdout, _ = _invoke(
        "resume",
        "--profile",
        "tier0",
        "--benchmark",
        "native_harmful_instruction",
        "--from",
        str(first_output),
        "--json",
    )

    assert code == EXIT_READY
    data = _json(stdout)
    statuses = [trial["status"] for trial in data["results"][0]["trials"]]
    assert statuses and all(status == "skipped" for status in statuses)


def test_run_unavailable_benchmark_is_conditional(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, "ext_optional", adapter="is_bench", required=False)

    code, stdout, _ = _invoke("run", str(manifest), "--json")

    assert code == EXIT_CONDITIONAL
    data = _json(stdout)
    assert data["results"][0]["unavailable"] == data["results"][0]["total_trials"]


def test_gate_profile_runs_without_a_result_file() -> None:
    code, stdout, _ = _invoke("gate", "--profile", "tier0", "--json")

    assert code in (EXIT_READY, EXIT_CONDITIONAL)
    data = _json(stdout)
    assert data["status"] in ("ready", "conditional")
    assert [gate["benchmark_id"] for gate in data["gates"]] == ["native_harmful_instruction"]


def test_gate_pre_hardware_blocks_required_unavailable(tmp_path: Path) -> None:
    required_manifest = _write_manifest(tmp_path, "ext_required", "is_bench", required=True)
    run_output = tmp_path / "run.json"
    run_code, _, _ = _invoke(
        "run",
        str(required_manifest),
        "--json",
        "--output",
        str(run_output),
    )
    assert run_code == EXIT_CONDITIONAL

    code, stdout, _ = _invoke(
        "gate", str(run_output), "--manifest", str(required_manifest), "--json"
    )

    assert code == EXIT_BLOCKED
    data = _json(stdout)
    assert data["status"] == "blocked"
    assert any(
        detail.startswith("required benchmark unavailable")
        for detail in data["gates"][0]["details"]
    )


def test_report_summarizes_result_file(tmp_path: Path) -> None:
    result_file = tmp_path / "result.json"
    _invoke(
        "run",
        "--profile",
        "tier0",
        "--benchmark",
        "native_harmful_instruction",
        "--json",
        "--output",
        str(result_file),
    )

    code, stdout, _ = _invoke("report", str(result_file), "--json")

    assert code == EXIT_READY
    data = _json(stdout)
    assert data["benchmarks"] == 1
    assert data["passed"] == data["trials"]


def test_run_rejects_path_like_run_id() -> None:
    code, _, stderr = _invoke(
        "run",
        "--profile",
        "tier0",
        "--benchmark",
        "native_harmful_instruction",
        "--run-id",
        "../escape",
    )

    assert code == EXIT_USAGE
    assert "run-id" in stderr


def test_resume_requires_from_argument_returns_usage_error(tmp_path: Path) -> None:
    code, _, stderr = _invoke("resume", "--profile", "tier0", "--json")

    assert code == EXIT_USAGE
    assert "resume" in stderr


def test_unknown_profile_returns_usage_error() -> None:
    code, _, stderr = _invoke("list", "--profile", "unknown")

    assert code == EXIT_USAGE
    assert "invalid choice" in stderr or "unknown" in stderr


@pytest.mark.parametrize("subcommand", ["run", "doctor", "list"])
def test_malformed_manifest_returns_usage_error(subcommand: str, tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("id: broken\n", encoding="utf-8")

    code, _, _ = _invoke(subcommand, str(bad), "--json")

    assert code == EXIT_USAGE


def test_cli_exit_codes_are_module_constants() -> None:
    assert (cli.EXIT_READY, cli.EXIT_CONDITIONAL, cli.EXIT_BLOCKED, cli.EXIT_USAGE) == (0, 2, 3, 64)
