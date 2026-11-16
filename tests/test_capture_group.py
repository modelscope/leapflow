"""Capture groups: concurrent members, bounded laggard re-capture, honest evidence."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from leapflow.hardware.vision import CaptureGroup, bounded_capture


@dataclass(frozen=True)
class _Frame:
    monotonic_at: float
    tag: str = ""


class _Camera:
    """Answers with scripted stamps, one per call, recording the bound it was given."""

    def __init__(self, *stamps: float, fail: Exception | None = None, delay: float = 0.0):
        self._stamps = list(stamps)
        self._fail = fail
        self._delay = delay
        self.bounds: list[float] = []

    async def __call__(self, after: float) -> _Frame:
        self.bounds.append(after)
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._fail is not None and (not self._stamps or len(self.bounds) > 1):
            raise self._fail
        stamp = self._stamps[min(len(self.bounds), len(self._stamps)) - 1]
        return _Frame(stamp, tag=f"call{len(self.bounds)}")


async def test_members_run_concurrently_against_one_bound() -> None:
    top, wrist = _Camera(10.0, delay=0.05), _Camera(10.01, delay=0.05)
    loop = asyncio.get_running_loop()
    started = loop.time()
    captured = await CaptureGroup({"top": top, "wrist": wrist}).capture(after_monotonic=9.0)
    assert loop.time() - started < 0.09  # concurrent, not 0.1 s in sequence
    assert top.bounds == wrist.bounds == [9.0]
    assert captured.within_tolerance and captured.attempts == 1
    assert captured.skew_ms == pytest.approx(10.0)
    assert captured.summary()["members"] == ["top", "wrist"]
    assert captured.summary()["sync"] == "timestamp"


async def test_a_laggard_is_recaptured_newer_than_the_tolerance_window() -> None:
    top, wrist = _Camera(10.0), _Camera(9.8, 9.99)
    captured = await CaptureGroup({"top": top, "wrist": wrist}, tolerance_ms=50).capture(
        after_monotonic=9.0
    )
    assert wrist.bounds == [9.0, pytest.approx(9.95)]
    assert top.bounds == [9.0]
    assert captured.recaptured == ("wrist",) and captured.attempts == 2
    assert captured.frames["wrist"].tag == "call2" and captured.within_tolerance


async def test_recapture_is_bounded_and_reports_the_remaining_skew() -> None:
    wrist = _Camera(9.0, 9.1, 9.2, 9.3)
    captured = await CaptureGroup(
        {"top": _Camera(10.0), "wrist": wrist}, tolerance_ms=50, attempts=3
    ).capture(after_monotonic=8.0)
    assert len(wrist.bounds) == 3 and captured.attempts == 3
    assert not captured.within_tolerance
    assert captured.skew_ms == pytest.approx(800.0)
    assert captured.summary()["within_tolerance"] is False


async def test_a_failed_recapture_keeps_the_first_frame() -> None:
    wrist = _Camera(9.0, fail=OSError("usb reset"))
    captured = await CaptureGroup({"top": _Camera(10.0), "wrist": wrist}).capture()
    assert captured.frames["wrist"].tag == "call1"
    assert not captured.errors and captured.recaptured == ()


async def test_member_failures_are_collected_not_raised() -> None:
    error = RuntimeError("no frame")
    captured = await CaptureGroup({"top": _Camera(10.0), "wrist": _Camera(fail=error)}).capture()
    assert set(captured.frames) == {"top"} and captured.errors == {"wrist": error}
    assert captured.skew_ms == 0.0 and captured.summary()["failed"] == ["wrist"]


async def test_cancellation_is_never_a_member_failure() -> None:
    async def cancelled(_after: float) -> _Frame:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await CaptureGroup({"top": _Camera(10.0), "wrist": cancelled}).capture()


def test_a_group_needs_members() -> None:
    with pytest.raises(ValueError, match="at least one member"):
        CaptureGroup({})


async def test_bounded_capture_passes_the_bound_only_when_accepted() -> None:
    seen: list[object] = []

    async def fresh(*, after_monotonic: float | None = None) -> _Frame:
        seen.append(after_monotonic)
        return _Frame(1.0)

    async def legacy() -> _Frame:
        seen.append("legacy")
        return _Frame(2.0)

    assert (await bounded_capture(fresh)(5.0)).monotonic_at == 1.0
    assert (await bounded_capture(legacy)(5.0)).monotonic_at == 2.0
    assert seen == [5.0, "legacy"]
