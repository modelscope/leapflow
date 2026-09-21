# Copyright (c) Alibaba, Inc. and its affiliates.
"""Focused tests for the --list-unused cassette diagnostic in sync_fixtures.py."""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

# Import the functions under test. The tool script adds REPO_ROOT to sys.path
# at import time, which is fine — we only need the pure-logic helpers.
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from tools.sync_fixtures import (  # noqa: E402
    _declared_journey_ids,
    _list_unused,
    main,
)


# ════════════════════════════════════════════════════════════════
# _declared_journey_ids
# ════════════════════════════════════════════════════════════════


def test_declared_journey_ids_parses_real_journeys() -> None:
    """Smoke: the parser finds at least the known journey IDs."""
    ids = _declared_journey_ids()
    assert "r1_conversation" in ids
    assert "r8_hardware" in ids
    assert len(ids) >= 8


def test_declared_journey_ids_extracts_string_literal(tmp_path: Path) -> None:
    """Only string-literal arguments to ``journeys(...)`` are collected."""
    journey_dir = tmp_path / "journeys"
    journey_dir.mkdir()
    (journey_dir / "test_alpha.py").write_text(
        textwrap.dedent("""\
            def test_something(journeys):
                j = journeys("alpha_journey", script=None)
        """),
        encoding="utf-8",
    )
    (journey_dir / "test_beta.py").write_text(
        textwrap.dedent("""\
            def test_other(journeys):
                name = "beta"
                j = journeys(name, script=None)  # dynamic — must NOT be collected
        """),
        encoding="utf-8",
    )
    with patch("tools.sync_fixtures.JOURNEY_DIR", journey_dir):
        ids = _declared_journey_ids()
    assert ids == {"alpha_journey"}


def test_declared_journey_ids_empty_when_no_dir(tmp_path: Path) -> None:
    """Graceful when the journey directory does not exist."""
    with patch("tools.sync_fixtures.JOURNEY_DIR", tmp_path / "nonexistent"):
        ids = _declared_journey_ids()
    assert ids == set()


# ════════════════════════════════════════════════════════════════
# _list_unused / --list-unused
# ════════════════════════════════════════════════════════════════


def _setup_cassette_dirs(
    root: Path,
    *dir_names: str,
    subdirs: tuple[str, ...] = ("cassettes",),
) -> None:
    """Create cassette-like directory structures under *root*."""
    for sub in subdirs:
        for name in dir_names:
            (root / sub / name).mkdir(parents=True, exist_ok=True)


def test_list_unused_reports_unmatched_directories(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Directories whose names don't match any journey ID are reported."""
    cassette_root = tmp_path / "cassettes"
    recording_root = tmp_path / "recordings"
    journey_dir = tmp_path / "journeys"
    journey_dir.mkdir()
    (journey_dir / "test_a.py").write_text(
        'def test(journeys):\n    journeys("alpha", script=None)\n',
        encoding="utf-8",
    )
    _setup_cassette_dirs(tmp_path, "alpha", "beta_stale", subdirs=("cassettes",))
    _setup_cassette_dirs(tmp_path, "alpha", subdirs=("recordings",))

    with patch("tools.sync_fixtures.CASSETTE_ROOT", cassette_root), \
         patch("tools.sync_fixtures.RECORDING_ROOT", recording_root), \
         patch("tools.sync_fixtures.JOURNEY_DIR", journey_dir), \
         patch("tools.sync_fixtures.REPO_ROOT", tmp_path):
        rc = _list_unused()

    assert rc == 0
    captured = capsys.readouterr().out
    assert "beta_stale" in captured
    assert "1 cassette director" in captured


def test_list_unused_clean_when_all_match(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """No output when every directory matches a declared journey."""
    cassette_root = tmp_path / "cassettes"
    journey_dir = tmp_path / "journeys"
    journey_dir.mkdir()
    (journey_dir / "test_x.py").write_text(
        'def test(journeys):\n    journeys("x_journey", script=None)\n',
        encoding="utf-8",
    )
    _setup_cassette_dirs(tmp_path, "x_journey", subdirs=("cassettes",))

    with patch("tools.sync_fixtures.CASSETTE_ROOT", cassette_root), \
         patch("tools.sync_fixtures.RECORDING_ROOT", tmp_path / "no_recordings"), \
         patch("tools.sync_fixtures.JOURNEY_DIR", journey_dir), \
         patch("tools.sync_fixtures.REPO_ROOT", tmp_path):
        rc = _list_unused()

    assert rc == 0
    captured = capsys.readouterr().out
    assert "all 1 cassette" in captured


def test_list_unused_warns_when_no_journey_ids(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """Warns rather than crashing when no journey IDs can be parsed."""
    with patch("tools.sync_fixtures.CASSETTE_ROOT", tmp_path / "c"), \
         patch("tools.sync_fixtures.RECORDING_ROOT", tmp_path / "r"), \
         patch("tools.sync_fixtures.JOURNEY_DIR", tmp_path / "nope"), \
         patch("tools.sync_fixtures.REPO_ROOT", tmp_path):
        rc = _list_unused()

    assert rc == 0
    assert "warning" in capsys.readouterr().out.lower()


def test_main_list_unused_exits_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    """``main(["--list-unused"])`` delegates and exits 0."""
    cassette_root = tmp_path / "cassettes"
    cassette_root.mkdir()
    journey_dir = tmp_path / "journeys"
    journey_dir.mkdir()

    with patch("tools.sync_fixtures.CASSETTE_ROOT", cassette_root), \
         patch("tools.sync_fixtures.RECORDING_ROOT", tmp_path / "rec"), \
         patch("tools.sync_fixtures.JOURNEY_DIR", journey_dir), \
         patch("tools.sync_fixtures.REPO_ROOT", tmp_path):
        rc = main(["--list-unused"])

    assert rc == 0
