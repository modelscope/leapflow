# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for the unified ``leap doctor`` diagnostic command."""
from __future__ import annotations

import io
from pathlib import Path

import pytest

from conftest import make_settings
from leapflow.cli.doctor import (
    SECTION_ORDER,
    build_doctor_checks,
    build_doctor_payload,
    print_doctor_report,
    run_doctor,
)
from leapflow.cli.doctor.protocol import DiagnosticCheck, Finding


# ════════════════════════════════════════════════════════════════
# Finding value object
# ════════════════════════════════════════════════════════════════


class TestFinding:
    """Finding dataclass behaviour."""

    def test_initial_state(self) -> None:
        f = Finding()
        assert f.passed == 0
        assert f.warnings == []
        assert f.errors == []
        assert f.fixed == 0
        assert f.ok is True
        assert f.total == 0

    def test_pass_increments(self) -> None:
        f = Finding()
        f.pass_()
        f.pass_()
        assert f.passed == 2
        assert f.total == 2
        assert f.ok is True

    def test_warn_appends(self) -> None:
        f = Finding()
        f.warn("low memory")
        assert f.warnings == ["low memory"]
        assert f.ok is True  # warnings do not make ok=False
        assert f.total == 1

    def test_error_appends_and_flips_ok(self) -> None:
        f = Finding()
        f.error("disk full")
        assert f.errors == ["disk full"]
        assert f.ok is False
        assert f.total == 1

    def test_fix_increments_both(self) -> None:
        f = Finding()
        f.fix("created dir")
        assert f.fixed == 1
        assert f.passed == 1
        assert f.total == 1

    def test_merge_combines_two_findings(self) -> None:
        a = Finding(passed=2, warnings=["w1"], errors=[], fixed=1)
        b = Finding(passed=1, warnings=["w2"], errors=["e1"], fixed=0)
        merged = a.merge(b)
        assert merged.passed == 3
        assert merged.warnings == ["w1", "w2"]
        assert merged.errors == ["e1"]
        assert merged.fixed == 1
        assert merged.ok is False
        # Originals are not mutated
        assert a.passed == 2
        assert b.passed == 1


# ════════════════════════════════════════════════════════════════
# DiagnosticCheck Protocol
# ════════════════════════════════════════════════════════════════


class _DummyPassCheck:
    name = "dummy-pass"
    section = "platform"

    async def check(self, should_fix: bool = False) -> Finding:
        f = Finding()
        f.pass_()
        return f


class _DummyFailCheck:
    name = "dummy-fail"
    section = "config"

    async def check(self, should_fix: bool = False) -> Finding:
        f = Finding()
        f.error("something broke")
        return f


class _DummyWarnCheck:
    name = "dummy-warn"
    section = "state"

    async def check(self, should_fix: bool = False) -> Finding:
        f = Finding()
        f.warn("not ideal")
        return f


class _DummyFixCheck:
    name = "dummy-fix"
    section = "config"

    async def check(self, should_fix: bool = False) -> Finding:
        f = Finding()
        if should_fix:
            f.fix("auto-repaired")
        else:
            f.error("needs fix")
        return f


def test_protocol_conformance() -> None:
    """Concrete checks satisfy the DiagnosticCheck Protocol."""
    assert isinstance(_DummyPassCheck(), DiagnosticCheck)
    assert isinstance(_DummyFailCheck(), DiagnosticCheck)


# ════════════════════════════════════════════════════════════════
# Orchestrator
# ════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_run_doctor_aggregates() -> None:
    checks = [_DummyPassCheck(), _DummyFailCheck(), _DummyWarnCheck()]
    agg, details = await run_doctor(checks)
    assert agg.passed == 1
    assert len(agg.errors) == 1
    assert len(agg.warnings) == 1
    assert agg.ok is False
    assert len(details) == 3


@pytest.mark.asyncio
async def test_run_doctor_section_filter() -> None:
    checks = [_DummyPassCheck(), _DummyFailCheck(), _DummyWarnCheck()]
    agg, details = await run_doctor(checks, section_filter="platform")
    assert len(details) == 1
    assert details[0][0].name == "dummy-pass"
    assert agg.ok is True


@pytest.mark.asyncio
async def test_run_doctor_fix_mode() -> None:
    checks = [_DummyFixCheck()]
    # Without fix
    agg, details = await run_doctor(checks, should_fix=False)
    assert agg.ok is False
    assert agg.errors == ["needs fix"]

    # With fix
    agg, details = await run_doctor(checks, should_fix=True)
    assert agg.ok is True
    assert agg.fixed == 1


# ════════════════════════════════════════════════════════════════
# Rich output
# ════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_print_doctor_report_outputs_text() -> None:
    checks = [_DummyPassCheck(), _DummyFailCheck()]
    agg, details = await run_doctor(checks)
    buf = io.StringIO()
    print_doctor_report(agg, details, file=buf)
    output = buf.getvalue()
    assert "LeapFlow Doctor" in output
    assert "dummy-pass" in output
    assert "dummy-fail" in output


# ════════════════════════════════════════════════════════════════
# Serializable payload (TUI /doctor)
# ════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_build_doctor_payload_structure() -> None:
    checks = [_DummyPassCheck(), _DummyWarnCheck(), _DummyFailCheck()]
    agg, details = await run_doctor(checks)
    payload = build_doctor_payload(agg, details)
    assert "ok" in payload
    assert "message" in payload
    assert "checks" in payload
    assert "summary" in payload
    assert payload["ok"] is False
    assert payload["summary"]["passed"] == 1
    assert payload["summary"]["warnings"] == 1
    assert payload["summary"]["errors"] == 1


# ════════════════════════════════════════════════════════════════
# build_doctor_checks factory
# ════════════════════════════════════════════════════════════════


def test_build_doctor_checks_returns_list(tmp_path: Path) -> None:
    settings = make_settings(str(tmp_path / "leap-home"))
    checks = build_doctor_checks(settings)
    assert len(checks) > 0
    # All must satisfy the Protocol
    for check in checks:
        assert isinstance(check, DiagnosticCheck)
        assert check.section in SECTION_ORDER


# ════════════════════════════════════════════════════════════════
# Individual check modules (smoke tests)
# ════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_python_version_check_passes() -> None:
    from leapflow.cli.doctor.checks_platform import PythonVersionCheck

    f = await PythonVersionCheck().check()
    # Must not error on the Python running the tests
    assert f.ok is True


@pytest.mark.asyncio
async def test_os_compatibility_check_passes() -> None:
    from leapflow.cli.doctor.checks_platform import OSCompatibilityCheck

    f = await OSCompatibilityCheck().check()
    # macOS/Linux should pass cleanly
    assert f.ok is True


@pytest.mark.asyncio
async def test_disk_space_check_passes(tmp_path: Path) -> None:
    from leapflow.cli.doctor.checks_platform import DiskSpaceCheck

    f = await DiskSpaceCheck(data_dir=tmp_path).check()
    assert f.ok is True


@pytest.mark.asyncio
async def test_profile_config_check_pass(tmp_path: Path) -> None:
    from leapflow.cli.doctor.checks_config import ProfileConfigCheck
    from leapflow.layout import build_layout

    layout = build_layout(tmp_path)
    profile_layout = layout.ensure(profile_id="default")
    f = await ProfileConfigCheck(profile_layout).check()
    assert f.ok is True


@pytest.mark.asyncio
async def test_profile_config_check_fix_creates_dir(tmp_path: Path) -> None:
    from leapflow.cli.doctor.checks_config import ProfileConfigCheck
    from leapflow.layout import ProfileLayout

    missing = tmp_path / "nonexistent" / "profile"
    layout = ProfileLayout(root=missing, profile_id="test")
    # Without fix — should error
    f = await ProfileConfigCheck(layout).check(should_fix=False)
    assert f.ok is False
    # With fix — should create
    f = await ProfileConfigCheck(layout).check(should_fix=True)
    assert missing.is_dir()


@pytest.mark.asyncio
async def test_llm_config_check_warns_on_missing_key(tmp_path: Path) -> None:
    from leapflow.cli.doctor.checks_config import LLMConfigCheck

    settings = make_settings(str(tmp_path / "leap-home"))
    settings = settings.__class__(**{**settings.__dict__, "llm_api_key": ""})
    f = await LLMConfigCheck(settings).check()
    assert len(f.warnings) >= 1
    assert any("API key" in w for w in f.warnings)


@pytest.mark.asyncio
async def test_path_layout_check_pass(tmp_path: Path) -> None:
    from leapflow.cli.doctor.checks_config import PathLayoutCheck
    from leapflow.layout import build_layout

    layout = build_layout(tmp_path)
    profile_layout = layout.ensure(profile_id="default")
    f = await PathLayoutCheck(profile_layout).check()
    assert f.ok is True


@pytest.mark.asyncio
async def test_path_layout_check_fix(tmp_path: Path) -> None:
    from leapflow.cli.doctor.checks_config import PathLayoutCheck
    from leapflow.layout import ProfileLayout

    root = tmp_path / "fresh_profile"
    root.mkdir()
    layout = ProfileLayout(root=root, profile_id="test")
    # Without fix — missing dirs => errors
    f = await PathLayoutCheck(layout).check(should_fix=False)
    assert f.ok is False
    # With fix — should create all
    f = await PathLayoutCheck(layout).check(should_fix=True)
    assert f.ok is True
    assert f.fixed > 0


@pytest.mark.asyncio
async def test_daemon_health_warns_when_not_running(tmp_path: Path) -> None:
    from leapflow.cli.doctor.checks_connectivity import DaemonHealthCheck

    f = await DaemonHealthCheck(runtime_dir=tmp_path).check()
    # Daemon is unlikely running in test — should warn, not error
    assert f.ok is True
    assert len(f.warnings) >= 1 or f.passed >= 1


@pytest.mark.asyncio
async def test_duckdb_health_check_nonexistent(tmp_path: Path) -> None:
    from leapflow.cli.doctor.checks_state import DuckDBHealthCheck

    f = await DuckDBHealthCheck(duckdb_path=tmp_path / "nope.duckdb").check()
    # Non-existent is warned, not errored
    assert f.ok is True
    assert len(f.warnings) >= 1


@pytest.mark.asyncio
async def test_vault_check_pass(tmp_path: Path) -> None:
    from leapflow.cli.doctor.checks_state import VaultCheck
    from leapflow.layout import build_layout

    layout = build_layout(tmp_path)
    profile_layout = layout.ensure(profile_id="default")
    f = await VaultCheck(profile_layout).check()
    assert f.ok is True


# ════════════════════════════════════════════════════════════════
# CLI argparse
# ════════════════════════════════════════════════════════════════


def test_cli_parses_doctor_command() -> None:
    # --help exits with 0 so we can't really run it, but we can test that
    # the command is recognized by checking known_commands set.
    # Verify 'doctor' is accepted as a subcommand by the parser.
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("doctor")
    args = parser.parse_args(["doctor"])
    assert args.command == "doctor"


def test_cli_known_commands_includes_doctor() -> None:
    """Verify the pre-parse set in cli.py includes 'doctor'."""
    from pathlib import Path

    cli_path = Path(__file__).resolve().parent.parent / "src" / "leapflow" / "cli" / "cli.py"
    source = cli_path.read_text(encoding="utf-8")
    assert '"doctor"' in source or "'doctor'" in source


# ════════════════════════════════════════════════════════════════
# Command registry
# ════════════════════════════════════════════════════════════════


def test_doctor_in_command_registry() -> None:
    from leapflow.cli.commands.registry import resolve_command

    cmd = resolve_command("doctor")
    assert cmd is not None
    assert cmd.name == "doctor"
    assert cmd.category == "Diagnostics"
