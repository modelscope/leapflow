"""Tests for ReadingReplay and HardwareAuditLog (Phase 2.5).

Covers:
- Deterministic replay: same file → identical events on two runs.
- Audit entries for read, write, and estop operations.
- CLI replay rendering paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from leapflow.hardware.audit import AuditEntry, HardwareAuditLog
from leapflow.hardware.context import Channel, HardwareContext
from leapflow.hardware.replay import (
    _build_replay_detector,
    _reading_from_dict,
    replay_segment,
    run_replay,
)
from leapflow.hardware.stream import HardwareEvent
from leapflow.hardware.transport import Reading


# ════════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════════


def _make_readings_ndjson(readings: list[dict[str, Any]]) -> str:
    """Return NDJSON text from a list of reading dicts."""
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in readings) + "\n"


def _sample_readings() -> list[dict[str, Any]]:
    """Return a series of readings that produce at least one event (quality degraded)."""
    base_ts = 1_000_000.0
    readings = []
    for i in range(6):
        readings.append({
            "device_id": "dev_a",
            "channel_id": "temp",
            "value": 25.0 + i,
            "quantity": "temperature",
            "unit": "C",
            "observed_at": base_ts + i,
            "sequence": i,
            "quality": "suspect" if i >= 2 and i <= 4 else "ok",
        })
    return readings


def _sample_readings_with_gap() -> list[dict[str, Any]]:
    """Return readings where seq 3 is missing, causing a sample_loss event."""
    base_ts = 2_000_000.0
    seqs = [0, 1, 2, 5, 6]  # gap: 3, 4 missing
    readings = []
    for idx, seq in enumerate(seqs):
        readings.append({
            "device_id": "dev_b",
            "channel_id": "pressure",
            "value": 100.0 + idx,
            "quantity": "pressure",
            "unit": "bar",
            "observed_at": base_ts + idx,
            "sequence": seq,
            "quality": "ok",
        })
    return readings


# ════════════════════════════════════════════════════════════════
# ReadingReplay tests
# ════════════════════════════════════════════════════════════════


class TestReadingFromDict:
    def test_roundtrip_via_to_dict(self) -> None:
        original = Reading(
            device_id="dev",
            channel_id="ch",
            value=42.0,
            quantity="temperature",
            unit="C",
            observed_at=1_000.0,
            monotonic_at=500.0,
            sequence=7,
            quality="ok",
        )
        d = original.to_dict()
        restored = _reading_from_dict(d)
        assert restored.device_id == original.device_id
        assert restored.channel_id == original.channel_id
        assert restored.value == original.value
        assert restored.sequence == original.sequence
        # monotonic_at is derived from observed_at in replay
        assert restored.monotonic_at == original.observed_at

    def test_missing_fields_default_gracefully(self) -> None:
        r = _reading_from_dict({"device_id": "x", "channel_id": "y"})
        assert r.device_id == "x"
        assert r.value is None
        assert r.sequence == 0


class TestReplaySegment:
    def test_deterministic_replay(self, tmp_path: Path) -> None:
        """Same file replayed twice produces identical event sequences."""
        readings = _sample_readings()
        seg = tmp_path / "seg.ndjson"
        seg.write_text(_make_readings_ndjson(readings), encoding="utf-8")

        def _run_once() -> list[HardwareEvent]:
            det = _build_replay_detector("dev_a", "temp", "temperature", "C")
            return replay_segment(seg, det)

        events_1 = _run_once()
        events_2 = _run_once()

        assert len(events_1) > 0, "Expected at least one event from quality-degraded run"
        assert len(events_1) == len(events_2)
        for e1, e2 in zip(events_1, events_2):
            assert e1 == e2

    def test_gap_detection(self, tmp_path: Path) -> None:
        """A sequence gap produces a SAMPLE_LOSS event."""
        readings = _sample_readings_with_gap()
        seg = tmp_path / "gap.ndjson"
        seg.write_text(_make_readings_ndjson(readings), encoding="utf-8")

        det = _build_replay_detector("dev_b", "pressure", "pressure", "bar")
        events = replay_segment(seg, det)

        loss_events = [e for e in events if e.kind == "sample_loss"]
        assert len(loss_events) >= 1

    def test_empty_file(self, tmp_path: Path) -> None:
        seg = tmp_path / "empty.ndjson"
        seg.write_text("", encoding="utf-8")
        det = _build_replay_detector("x", "y")
        assert replay_segment(seg, det) == []

    def test_corrupt_line_skipped(self, tmp_path: Path) -> None:
        """A corrupt line does not abort the rest of the replay."""
        readings = _sample_readings()[:2]
        content = json.dumps(readings[0]) + "\n" + "NOT JSON\n" + json.dumps(readings[1]) + "\n"
        seg = tmp_path / "corrupt.ndjson"
        seg.write_text(content, encoding="utf-8")

        det = _build_replay_detector("dev_a", "temp")
        events = replay_segment(seg, det)
        # Should not raise; may or may not produce events, but parsing succeeded.
        assert isinstance(events, list)

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        seg = tmp_path / "nonexistent.ndjson"
        det = _build_replay_detector("x", "y")
        assert replay_segment(seg, det) == []


class TestRunReplay:
    def test_end_to_end(self, tmp_path: Path) -> None:
        readings = _sample_readings()
        seg = tmp_path / "full.ndjson"
        seg.write_text(_make_readings_ndjson(readings), encoding="utf-8")

        events = run_replay(seg)
        assert isinstance(events, list)
        assert len(events) > 0

    def test_empty_file(self, tmp_path: Path) -> None:
        seg = tmp_path / "empty.ndjson"
        seg.write_text("", encoding="utf-8")
        assert run_replay(seg) == []


# ════════════════════════════════════════════════════════════════
# HardwareAuditLog tests
# ════════════════════════════════════════════════════════════════


class TestAuditEntry:
    def test_roundtrip(self) -> None:
        entry = AuditEntry(
            ts=1_000.0, action="read", device="dev", channel="ch",
            value=42.0, outcome="ok", identity="sess",
        )
        d = entry.to_dict()
        restored = AuditEntry.from_dict(d)
        assert restored == entry


class TestHardwareAuditLog:
    def test_record_and_read(self, tmp_path: Path) -> None:
        log_path = tmp_path / "audit" / "hardware_audit.ndjson"
        audit = HardwareAuditLog(log_path)

        audit.record(action="read", device="d1", channel="ch1", value=1.0, outcome="ok")
        audit.record(action="write", device="d1", channel="ch1", value=2.0, outcome="ok")
        audit.record(action="estop", device="d1", outcome="ok")

        entries = audit.read_entries()
        assert len(entries) == 3
        assert entries[0].action == "read"
        assert entries[1].action == "write"
        assert entries[2].action == "estop"

    def test_all_three_actions_present(self, tmp_path: Path) -> None:
        """Verify read/write/estop audit entries are parseable."""
        log_path = tmp_path / "hw.ndjson"
        audit = HardwareAuditLog(log_path)

        audit.record(action="read", device="sensor", channel="temp", value=25.0)
        audit.record(action="write", device="pump", channel="duty", value=50)
        audit.record(action="estop", device="robot")

        entries = audit.read_entries()
        actions = {e.action for e in entries}
        assert actions == {"read", "write", "estop"}
        for entry in entries:
            d = entry.to_dict()
            assert "ts" in d
            assert "action" in d
            assert "device" in d

    def test_no_path_degrades_gracefully(self) -> None:
        audit = HardwareAuditLog(None)
        entry = audit.record(action="read", device="d", channel="c")
        assert entry is not None
        assert entry.action == "read"
        assert audit.read_entries() == []


# ════════════════════════════════════════════════════════════════
# tools.py audit wiring integration test
# ════════════════════════════════════════════════════════════════


class _FakeTransport:
    kind = "fake"

    async def open(self, context: Any) -> Any:
        return SimpleNamespace(connected=True, halt_supported=True, detail="", latency_ms=0)

    async def close(self) -> Any:
        return SimpleNamespace(connected=False, halt_supported=True, detail="", latency_ms=0)

    async def read(self, channel_id: str) -> Reading:
        return Reading(
            device_id="dev",
            channel_id=channel_id,
            value=25.0,
            quantity="temperature",
            unit="C",
        )

    async def write(self, channel_id: str, value: Any) -> Any:
        from leapflow.hardware.transport import WriteOutcome, SIDE_EFFECT_COMMITTED
        return WriteOutcome(ok=True, side_effect_state=SIDE_EFFECT_COMMITTED, settled=True)

    async def probe(self) -> Any:
        return SimpleNamespace(connected=True, halt_supported=True, detail="", latency_ms=0, to_dict=lambda: {})

    async def halt(self) -> Any:
        return SimpleNamespace(connected=True, halt_supported=True, detail="stopped", latency_ms=0, to_dict=lambda: {})


class _FakeRegistry:
    """Minimal registry stand-in for audit wiring tests."""

    def __init__(self) -> None:
        self._transport = _FakeTransport()
        self._context = HardwareContext(
            device_id="dev",
            channels=(
                Channel(channel_id="temp", direction="read", quantity="temperature", unit="C"),
            ),
        )
        self.outcome_recorder = None

    def context(self, device_id: str) -> HardwareContext | None:
        return self._context if device_id == "dev" else None

    def contexts(self) -> list[HardwareContext]:
        return [self._context]

    def channel(self, device_id: str, channel_id: str) -> Channel | None:
        if device_id == "dev" and channel_id == "temp":
            return self._context.channels[0]
        return None

    async def transport(self, device_id: str) -> _FakeTransport:
        return self._transport

    def device_io(self, device_id: str) -> Any:
        import contextlib
        @contextlib.asynccontextmanager
        async def _ctx():
            yield
        return _ctx()

    def mark_described(self, session_id: str, device_id: str) -> None:
        pass

    def channel_summary(self, device_id: str, channel_id: str) -> dict:
        return {}

    def channel_history(self, device_id: str, channel_id: str, limit: int = 10) -> tuple:
        return ()

    def recent_events(self, device_id: str) -> list:
        return []


@pytest.mark.asyncio
async def test_hw_read_produces_audit_entry(tmp_path: Path) -> None:
    from leapflow.hardware.tools import HardwareTools

    audit_path = tmp_path / "audit.ndjson"
    audit = HardwareAuditLog(audit_path)
    tools = HardwareTools(_FakeRegistry(), audit_log=audit, session_id="test-sess")

    result = await tools.hw_read(device_id="dev", channel_id="temp")
    assert result["ok"] is True

    entries = audit.read_entries()
    assert len(entries) == 1
    assert entries[0].action == "read"
    assert entries[0].device == "dev"
    assert entries[0].channel == "temp"


@pytest.mark.asyncio
async def test_hw_estop_produces_audit_entry(tmp_path: Path) -> None:
    from leapflow.hardware.tools import HardwareTools

    audit_path = tmp_path / "audit.ndjson"
    audit = HardwareAuditLog(audit_path)
    tools = HardwareTools(_FakeRegistry(), audit_log=audit, session_id="test-sess")

    result = await tools.hw_estop(device_id="dev")
    assert result["ok"] is True

    entries = audit.read_entries()
    assert len(entries) == 1
    assert entries[0].action == "estop"
    assert entries[0].device == "dev"


# ════════════════════════════════════════════════════════════════
# CLI replay subcommand test
# ════════════════════════════════════════════════════════════════


def test_cli_replay_renders(tmp_path: Path) -> None:
    """The replay CLI path runs and produces a result."""
    from leapflow.cli.commands.hardware import _run_replay

    readings = _sample_readings()
    seg = tmp_path / "seg.ndjson"
    seg.write_text(_make_readings_ndjson(readings), encoding="utf-8")

    args = SimpleNamespace(segment_path=str(seg), json=False)
    exit_code = _run_replay(args, json_mode=False)
    assert exit_code == 0


def test_cli_replay_missing_file(tmp_path: Path) -> None:
    from leapflow.cli.commands.hardware import _run_replay

    args = SimpleNamespace(segment_path=str(tmp_path / "nope.ndjson"), json=False)
    exit_code = _run_replay(args, json_mode=False)
    assert exit_code == 1
