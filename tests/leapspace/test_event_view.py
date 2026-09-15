# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for event_view — timeline rendering over a temp evidence db."""

import json

import duckdb
import pytest

from leapspace.app_space.event_view import main, view_events

SCHEMA = (
    """
    CREATE TABLE leap_trajectory (
        id TEXT, user_id TEXT, start_time DOUBLE, end_time DOUBLE,
        step_count INTEGER, metadata TEXT, created_at DOUBLE)
    """,
    """
    CREATE TABLE leap_trajectory_step (
        trajectory_id TEXT, step_idx INTEGER, timestamp DOUBLE,
        action_type TEXT, target TEXT, target_label TEXT, target_role TEXT,
        app_bundle_id TEXT, app_name TEXT, params TEXT,
        state_focused_app TEXT, state_ax_digest TEXT, state_clipboard TEXT,
        visual_frame_ref TEXT, state_ax_tree TEXT, state_snapshot_level TEXT)
    """,
    """
    CREATE TABLE leap_episode (
        id TEXT, trajectory_id TEXT, start_idx INTEGER, end_idx INTEGER,
        inferred_goal TEXT, app_sequence TEXT, semantic_actions TEXT,
        confidence DOUBLE, created_at DOUBLE)
    """,
)


def make_db(tmp_path):
    path = tmp_path / "eval.duckdb"
    con = duckdb.connect(str(path))
    for statement in SCHEMA:
        con.execute(statement)
    return path, con


def seed_run(con, trajectory_id="traj-1", user_id="default"):
    con.execute(
        "INSERT INTO leap_trajectory VALUES (?, ?, ?, ?, ?, '{}', 0.0)",
        [trajectory_id, user_id, 1000.0, 1002.0, 3],
    )
    con.execute(
        "INSERT INTO leap_trajectory_step VALUES (?, ?, ?, ?, ?, '', '', '', '', ?, '', '', '', '', '', '')",
        [trajectory_id, 0, 1000.5, "ui.click", "",
         json.dumps({"mouse_x": 128, "mouse_y": 87})],
    )
    con.execute(
        "INSERT INTO leap_trajectory_step VALUES (?, ?, ?, ?, ?, '', '', '', '', ?, '', '', '', '', '', '')",
        [trajectory_id, 1, 1000.75, "ui.type", "",
         json.dumps({"key_code": 0, "char": "G", "modifiers": ["shift"]})],
    )
    con.execute(
        "INSERT INTO leap_trajectory_step VALUES (?, ?, ?, ?, ?, '', '', '', '', ?, '', '', '', '', '', '')",
        [trajectory_id, 2, 1001.0, "file.modify", "/tmp/leapspace/chat/state.json",
         json.dumps({"path": "/tmp/leapspace/chat/state.json", "action": "modified"})],
    )
    con.execute(
        "INSERT INTO leap_episode VALUES ('ep-1', ?, 0, 3, 'reply to the boss', '[]', ?, 1.0, 0.0)",
        [trajectory_id,
         json.dumps([{"action_name": "ui.click"}, {"action_name": "type_text"}])],
    )


def test_renders_timeline_with_offsets_and_descriptions(tmp_path):
    path, con = make_db(tmp_path)
    seed_run(con)
    con.close()
    text = view_events(path)
    assert "trajectory traj-1  user=default  steps=3" in text
    assert "episode 1  steps 0-3  goal: reply to the boss" in text
    assert "semantic: ui.click → type_text" in text
    # offsets are relative to the trajectory start (1000.0)
    assert "t+  0.500s  #0    ui.click     (128, 87)" in text
    assert "t+  0.750s  #1    ui.type      'G' ['shift']" in text
    assert "t+  1.000s  #2    file.modify  modified /tmp/leapspace/chat/state.json" in text


def test_empty_db_reports_no_trajectories(tmp_path):
    path, con = make_db(tmp_path)
    con.close()
    assert view_events(path) == "no trajectories recorded"


def test_missing_tables_rejected(tmp_path):
    path = tmp_path / "plain.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE other (x INTEGER)")
    con.close()
    with pytest.raises(ValueError, match="not a signal evidence db"):
        view_events(path)


def test_cli_prints_timeline_and_exits_zero(tmp_path, capsys):
    path, con = make_db(tmp_path)
    seed_run(con)
    con.close()
    assert main([str(path)]) == 0
    assert "trajectory traj-1" in capsys.readouterr().out


def test_cli_missing_file_exits_nonzero(tmp_path, capsys):
    assert main([str(tmp_path / "absent.duckdb")]) == 1
    assert "absent.duckdb" in capsys.readouterr().err


def test_cli_rejects_foreign_db(tmp_path, capsys):
    path = tmp_path / "plain.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE other (x INTEGER)")
    con.close()
    assert main([str(path)]) == 1
    assert "not a signal evidence db" in capsys.readouterr().err
