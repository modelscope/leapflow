# Copyright (c) Alibaba, Inc. and its affiliates.
"""Regressions for the freshness contract and the evolution axis.

Each test here corresponds to a defect measured on a live board, and states the
symptom rather than the implementation, so a refactor that keeps the behaviour keeps
the test.
"""
from __future__ import annotations

import time

from leapflow.dashboard.service import _evidence_trend, _provenance


# ════════════════════════════════════════════════════════════════
# Provenance: two instants, never merged
# ════════════════════════════════════════════════════════════════


def test_content_age_and_check_age_are_reported_separately() -> None:
    """The defect: one instant could not answer both questions.

    A finding-backed board renders the newest *persisted* finding and findings dedup
    on content, so the content can be much older than the last completed cycle.
    Reporting only the content instant made a healthy system look frozen; reporting
    only the cycle instant would age a stale page from the clock.
    """
    now = time.time()
    prov = _provenance(
        {"observed_at": now - 600},
        {"last_run_at": now - 5, "interval_seconds": 120, "watch_id": "w1"},
    )

    assert prov["age_seconds"] > 590, "content age must reflect the payload instant"
    assert prov["checked_age_seconds"] < 10, "check age must reflect the last cycle"
    assert prov["unchanged_for_seconds"] > 590


def test_old_but_freshly_checked_content_is_not_stale() -> None:
    """Stable is not stale, and conflating them is what made refresh look broken.

    The verdict is about the watch: a cycle completed 5s ago, so the page reflects
    the present even though its content has not changed for ten minutes.
    """
    now = time.time()
    prov = _provenance(
        {"observed_at": now - 600},
        {"last_run_at": now - 5, "interval_seconds": 120},
    )
    assert prov["stale"] is False


def test_a_watch_that_stopped_running_is_stale() -> None:
    """Two cadences without a completed cycle means nothing on the page is current."""
    now = time.time()
    prov = _provenance(
        {"observed_at": now - 900},
        {"last_run_at": now - 500, "interval_seconds": 120},
    )
    assert prov["stale"] is True


def test_one_missed_cycle_is_jitter_not_staleness() -> None:
    """The threshold is two cadences on purpose: scheduling is not exact."""
    now = time.time()
    prov = _provenance(
        {"observed_at": now - 200},
        {"last_run_at": now - 150, "interval_seconds": 120},
    )
    assert prov["stale"] is False


def test_a_watch_with_no_declared_cadence_gets_no_staleness_verdict() -> None:
    """An event-driven watch has no interval, so judging it would mean inventing one."""
    now = time.time()
    prov = _provenance(
        {"observed_at": now - 100000},
        {"last_run_at": now - 100000, "interval_seconds": 0},
    )
    assert prov["stale"] is False
    assert prov["cadence_seconds"] == 0.0


def test_a_never_run_watch_reports_absent_rather_than_zero_age() -> None:
    """Absent is not fresh. "0s ago" on a board that never observed invents a fact."""
    prov = _provenance({}, {})
    assert prov["observed"] is False
    assert prov["checked"] is False
    assert prov["age_seconds"] == 0.0
    assert prov["stale"] is False


def test_provenance_carries_the_watch_id_so_the_bar_can_refresh() -> None:
    """Without it the bar can report the page is behind but not do anything about it."""
    prov = _provenance({"observed_at": 1.0}, {"watch_id": "w-42", "last_run_at": 2.0})
    assert prov["watch_id"] == "w-42"


def test_a_payload_without_its_own_instant_is_dated_by_the_cycle() -> None:
    now = time.time()
    prov = _provenance({}, {"last_run_at": now - 30, "interval_seconds": 120})
    assert prov["observed"] is True
    assert 25 < prov["age_seconds"] < 35


# ════════════════════════════════════════════════════════════════
# Evolution axis: the only panel that shows direction
# ════════════════════════════════════════════════════════════════


def _finding(observed_at: float, with_evidence: int, total: int = 10) -> dict:
    return {
        "ts": observed_at,
        "payload": {
            "observed_at": observed_at,
            "summary": {
                "segments_with_evidence": with_evidence,
                "segments_total": total,
            },
        },
    }


def test_a_single_sample_still_produces_a_series() -> None:
    """The defect this exists to prevent.

    Suppressing a one-point series hides the beginning of every evolution the board
    is for -- and on a new profile that is the only reading there is.
    """
    trend = _evidence_trend([_finding(100.0, 2)])
    assert trend["samples"] == 1
    assert trend["series"], "one sample must still yield a drawable series"
    assert trend["series"][0]["points"] == [{"x": 100.0, "y": 2, "at": 100.0}]


def test_the_axis_reads_oldest_first() -> None:
    """Findings are persisted newest-first; a time axis must not be."""
    trend = _evidence_trend([_finding(300.0, 5), _finding(200.0, 3), _finding(100.0, 2)])
    assert [p["y"] for p in trend["series"][0]["points"]] == [2, 3, 5]
    assert trend["first_at"] == 100.0
    assert trend["last_at"] == 300.0


def test_net_change_states_the_direction() -> None:
    """Direction is stated rather than left to the reader's eye on a short line."""
    assert _evidence_trend([_finding(300.0, 5), _finding(100.0, 2)])["delta"] == 3
    assert _evidence_trend([_finding(300.0, 1), _finding(100.0, 4)])["delta"] == -3


def test_a_single_sample_declares_no_direction() -> None:
    """One point is a reading, not a trend."""
    assert _evidence_trend([_finding(100.0, 2)])["delta"] == 0


def test_no_findings_yields_no_series_rather_than_a_fake_zero() -> None:
    trend = _evidence_trend([])
    assert trend["series"] == []
    assert trend["samples"] == 0


def test_malformed_findings_are_skipped_not_charted_as_zero() -> None:
    """A payload with no summary is unknown, and unknown is not zero evidence."""
    trend = _evidence_trend([
        {"ts": 1.0, "payload": None},
        {"ts": 2.0, "payload": {"observed_at": 2.0}},
        {"ts": 3.0, "payload": {"observed_at": 3.0, "summary": {}}},
        _finding(4.0, 3),
    ])
    assert trend["samples"] == 1
    assert trend["series"][0]["points"][0]["y"] == 3
