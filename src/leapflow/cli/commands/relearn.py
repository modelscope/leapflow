"""Relearn subcommand — re-run learning pipeline on a saved trajectory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from leapflow.cli.helpers import finish_learn_progress, install_learn_progress, require_initialized

if TYPE_CHECKING:
    from leapflow.cli.context import Context


async def cmd_relearn(ctx: "Context", trajectory_id: str) -> int:
    """Re-run learning pipeline on a saved trajectory (requires LLM)."""
    require_initialized(ctx)
    if not ctx.settings.has_llm_credentials:
        print(
            "Error: LLM credentials required for learning. "
            "Configure profile secrets or use LEAPFLOW_LLM_API_KEY as a process override."
        )
        return 1

    traj = ctx.imitation.get_trajectory(trajectory_id)
    if traj is None:
        print(f"Trajectory '{trajectory_id}' not found.")
        print("Use 'leap skills sessions' to list recorded trajectories.")
        return 1

    print(f"[ RELEARNING from trajectory: {trajectory_id} ]")
    print(f"  Steps: {traj.step_count} | Duration: {traj.duration:.1f}s")
    print()

    install_learn_progress(ctx)
    candidates = await ctx.imitation.distill(trajectory_id)
    finish_learn_progress()

    if not candidates:
        print()
        print("[ NO SKILLS EXTRACTED ]")
        print("  The trajectory may be too short or ambiguous.")
        return 0

    activated = set()
    if ctx.session and hasattr(ctx.session, '_observer') and ctx.session._observer:
        try:
            activated = await ctx.session._observer.await_activations()
        except Exception:
            pass

    confirmed = [c for c in candidates if c.title in activated]
    suggestions = [c for c in candidates if c.title not in activated]

    if confirmed:
        print()
        print(f"[ NEW SKILLS ({len(confirmed)}) ]")
        for c in confirmed:
            print(f"  * {c.title}")
            print(f"      confidence: {c.confidence:.0%}")
            print(f"      steps:      {len(c.steps)}")
    if suggestions:
        print()
        print(f"[ CANDIDATES ({len(suggestions)}) ]")
        for c in suggestions:
            print(f"  - {c.title} ({c.confidence:.0%})")

    return 0
