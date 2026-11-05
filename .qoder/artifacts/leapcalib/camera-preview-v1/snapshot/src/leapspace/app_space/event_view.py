# Copyright (c) Alibaba, Inc. and its affiliates.
"""event_view — render a signal evidence DuckDB as a human-readable timeline.

Read-only viewer for the eval.duckdb a signal-mode run leaves behind (the
LeapSignal writer's schema: leap_trajectory, leap_trajectory_step,
leap_episode). Prints every recorded signal in timeline order — one line
per step with its offset from the trajectory start, episodes as section
headers carrying their inferred goal and semantic action chain — so a
pulled evidence dir can be inspected without hand-written SQL:

    python -m leapspace.app_space.event_view <path/to/eval.duckdb>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import duckdb


def view_events(db_path: str | Path) -> str:
    """Return the rendered timeline of every trajectory in one evidence db."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        _require_signal_tables(con)
        lines: list[str] = []
        trajectories = con.execute(
            "SELECT id, user_id, start_time, end_time, step_count "
            "FROM leap_trajectory ORDER BY start_time"
        ).fetchall()
        if not trajectories:
            return "no trajectories recorded"
        for traj_id, user_id, start, end, step_count in trajectories:
            lines.extend(_render_trajectory(con, traj_id, user_id, start, end, step_count))
        return "\n".join(lines)
    finally:
        con.close()


def _require_signal_tables(con: duckdb.DuckDBPyConnection) -> None:
    tables = {
        row[0]
        for row in con.execute(
            "SELECT table_name FROM information_schema.tables"
        ).fetchall()
    }
    missing = {"leap_trajectory", "leap_trajectory_step", "leap_episode"} - tables
    if missing:
        raise ValueError(
            f"missing signal tables {sorted(missing)} — not a signal evidence db"
        )


def _render_trajectory(
    con: duckdb.DuckDBPyConnection,
    traj_id: str,
    user_id: str,
    start: float,
    end: float,
    step_count: int,
) -> list[str]:
    steps = con.execute(
        "SELECT step_idx, timestamp, action_type, target, params "
        "FROM leap_trajectory_step WHERE trajectory_id = ? ORDER BY step_idx",
        [traj_id],
    ).fetchall()
    episodes = con.execute(
        "SELECT start_idx, end_idx, inferred_goal, semantic_actions "
        "FROM leap_episode WHERE trajectory_id = ? ORDER BY start_idx",
        [traj_id],
    ).fetchall()
    width = max([len(action_type) for _, _, action_type, _, _ in steps] + [len("file.modify")])

    lines = [
        f"trajectory {traj_id}  user={user_id}  steps={step_count}  "
        f"{_fmt_time(start)} → {_fmt_time(end)} ({end - start:.2f}s)"
    ]
    episode_no = 0
    for step_idx, timestamp, action_type, target, params in steps:
        while episode_no < len(episodes) and episodes[episode_no][0] <= step_idx:
            lines.append("")
            lines.extend(_render_episode(episode_no + 1, episodes[episode_no]))
            episode_no += 1
        offset = timestamp - start
        description = _describe_step(action_type, target, _parse_params(params))
        lines.append(f"  t+{offset:7.3f}s  #{step_idx:<3d}  {action_type:{width}}  {description}")
    while episode_no < len(episodes):
        lines.append("")
        lines.extend(_render_episode(episode_no + 1, episodes[episode_no]))
        episode_no += 1
    return lines


def _render_episode(episode_no: int, episode: tuple[Any, ...]) -> list[str]:
    start_idx, end_idx, goal, semantic_actions = episode
    lines = [f"episode {episode_no}  steps {start_idx}-{end_idx}  goal: {goal}"]
    semantic = _parse_params(semantic_actions)
    if isinstance(semantic, list) and semantic:
        chain = " → ".join(str(action.get("action_name", "?")) for action in semantic)
        lines.append(f"  semantic: {chain}")
    return lines


def _describe_step(
    action_type: str, target: str, params: dict[str, Any] | None
) -> str:
    if action_type == "ui.click" and params:
        return f"({params.get('mouse_x')}, {params.get('mouse_y')})"
    if action_type == "ui.type" and params:
        text = repr(params.get("char", ""))
        modifiers = params.get("modifiers") or []
        return f"{text} {modifiers}" if modifiers else text
    if action_type == "file.modify" and params:
        return f"{params.get('action')} {target}"
    if target:
        return target
    if params:
        return json.dumps(params, ensure_ascii=False)
    return ""


def _parse_params(raw: str | None) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _fmt_time(timestamp: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="event_view",
        description="Render a signal evidence DuckDB as a human-readable timeline.",
    )
    parser.add_argument("db_path", type=Path, help="path to the evidence eval.duckdb")
    args = parser.parse_args(argv)
    try:
        print(view_events(args.db_path))
    except (duckdb.Error, ValueError) as exc:
        print(f"{args.db_path}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
