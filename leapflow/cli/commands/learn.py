"""Learn subcommand — interactive learning mode (record → distill)."""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING, List, Optional

from leapflow.cli.helpers import (
    blink_recording,
    finish_learn_progress,
    get_perceptual_field_filter,
    install_learn_progress,
    install_stop_progress,
    require_initialized,
)

if TYPE_CHECKING:
    from leapflow.cli.context import Context


_VALID_LEVELS = frozenset({"full", "structural", "opaque", "deny"})


def parse_field_args(field_args: List[str]) -> "List[Any]":
    """Parse --field CLI arguments into FieldRule objects.

    Format: "app_pattern:context_pattern[:level]"
    Level defaults to 'full' if omitted.
    """
    from leapflow.domain.perception import FieldRule, PerceptionLevel

    rules = []
    for arg in field_args:
        first_colon = arg.find(":")
        if first_colon < 0:
            print(f"  Warning: ignoring malformed --field arg: {arg!r}")
            continue
        app_pat = arg[:first_colon]
        remainder = arg[first_colon + 1:]

        last_colon = remainder.rfind(":")
        if last_colon >= 0 and remainder[last_colon + 1:] in _VALID_LEVELS:
            ctx_pat = remainder[:last_colon]
            level_str = remainder[last_colon + 1:]
        else:
            ctx_pat = remainder
            level_str = "full"

        level = PerceptionLevel(level_str)
        rules.append(FieldRule(app_pat, ctx_pat, level, "user", 500))
    return rules


def _handle_field_command(ctx: "Context", subcmd: str) -> None:
    """Handle 'field <subcmd>' during learn mode."""
    from leapflow.domain.perception import FieldRule, PerceptionLevel

    pf_filter = get_perceptual_field_filter(ctx)
    if not pf_filter:
        print("  Perceptual field not enabled. Set LEAPFLOW_PERCEPTUAL_FIELD_ENABLED=true")
        return

    current_ctx = pf_filter.extractor.current_context

    if subcmd == "status" or subcmd == "":
        if current_ctx:
            level = pf_filter.policy.evaluate(current_ctx)
            app_short = current_ctx.app_bundle_id.split(".")[-1]
            print(f"  Context: {app_short}/{current_ctx.context_value}")
            print(f"  Level:   {level.value}")
        else:
            print("  No context detected yet.")

    elif subcmd == "opaque":
        if current_ctx:
            pf_filter.policy.add_rule(FieldRule(
                current_ctx.app_bundle_id, current_ctx.context_value,
                PerceptionLevel.OPAQUE, "user", 500,
            ))
            print(f"  Marked opaque: {current_ctx.context_value}")
        else:
            print("  No active context to mark.")

    elif subcmd == "deny":
        if current_ctx:
            pf_filter.policy.add_rule(FieldRule(
                current_ctx.app_bundle_id, current_ctx.context_value,
                PerceptionLevel.DENY, "user", 500,
            ))
            retracted = ctx.imitation.recorder.retract_context(
                current_ctx.app_bundle_id, f"*{current_ctx.context_value}*",
            )
            print(f"  Denied: {current_ctx.context_value} ({retracted} events retracted)")
        else:
            print("  No active context to deny.")

    elif subcmd == "full":
        if current_ctx:
            pf_filter.policy.add_rule(FieldRule(
                current_ctx.app_bundle_id, current_ctx.context_value,
                PerceptionLevel.FULL, "user", 500,
            ))
            print(f"  Set FULL: {current_ctx.context_value}")
        else:
            print("  No active context.")

    elif subcmd == "list":
        observed = pf_filter.get_observed_contexts()
        if not observed:
            print("  No contexts observed yet.")
        else:
            print(f"  {'Context':<40} {'Level':<12} Events")
            print("  " + "-" * 60)
            for ctx_id, info in observed.items():
                level = pf_filter.policy.evaluate(ctx_id)
                app_short = ctx_id.app_bundle_id.split(".")[-1]
                label = f"{app_short}/{ctx_id.context_value[:30]}"
                print(f"  {label:<40} {level.value:<12} {info['count']}")

    else:
        print(f"  Unknown field command: {subcmd}")
        print("  Available: status | opaque | deny | full | list")


def _print_perception_summary(ctx: "Context") -> None:
    """Print perception summary after recording (if perceptual field is enabled)."""
    pf_filter = get_perceptual_field_filter(ctx)
    if not pf_filter:
        return

    observed = pf_filter.get_observed_contexts()
    if not observed:
        return

    print()
    print("[ PERCEPTION SUMMARY ]")

    by_level: dict = {"full": [], "structural": [], "opaque": [], "deny": []}
    for ctx_id, info in observed.items():
        level = pf_filter.policy.evaluate(ctx_id)
        app_short = ctx_id.app_bundle_id.split(".")[-1]
        label = f"{app_short}/{ctx_id.context_value}"
        source = info.get("rule_source", "")
        by_level[level.value].append((label, info["count"], source))

    level_display = {"full": "FULL", "structural": "STRUCTURAL", "opaque": "OPAQUE", "deny": "DENIED"}
    for level_name in ("full", "structural", "opaque", "deny"):
        for label, count, source in by_level[level_name]:
            suffix = f" ({source})" if source else ""
            print(f"  {level_display[level_name]:<12} {label:<35} {count} events{suffix}")

    print()


async def cmd_learn(ctx: "Context", goal: str, timeout: Optional[float], field_args: Optional[List[str]] = None) -> int:
    require_initialized(ctx)
    if timeout:
        ctx.session.idle_timeout = timeout

    pf_filter = get_perceptual_field_filter(ctx) if ctx.settings.perceptual_field_enabled else None

    if field_args and pf_filter:
        for rule in parse_field_args(field_args):
            pf_filter.policy.add_rule(rule)

    if not ctx.settings.has_llm_credentials:
        sys.stderr.write(
            "\033[33m⚠ LEAPFLOW_LLM_API_KEY not set — "
            "recording will be saved but skill learning requires LLM.\033[0m\n"
        )
        sys.stderr.flush()

    # Check Bridge connection before starting recording
    if not ctx.effective_mock and hasattr(ctx.rpc, 'connected') and not ctx.rpc.connected:
        sys.stderr.write(
            "\033[33m⚠ Warning: OS Host bridge not connected. "
            "Recording may not capture real events.\033[0m\n"
        )
        sys.stderr.write(
            "\033[2m  Check: is OS Host running? "
            "Try 'leap host dev' in another terminal.\033[0m\n"
        )
        sys.stderr.flush()

    session = await ctx.session.enter_learning(goal=goal)
    if ctx.effective_mock:
        mode = "mock"
    else:
        mode = ctx.imitation.recorder.recording_mode.value

    print("[ LEARNING STARTED ]")
    print(f"  Trajectory: {session.trajectory_id}")
    print(f"  Mode:       {mode}")
    print()
    print("What LEAP records:")
    print("  • File operations    — create / move / rename / delete in Finder or terminal")
    print("  • App switches       — bringing apps to foreground")
    print("  • UI interactions    — clicks, typing, focus changes (full mode only)")
    print("  • Clipboard          — copy / paste events")
    if ctx.settings.visual_track_enabled:
        print("  • Screen regions     — visual frames at keyframes (visual track enabled)")
    if pf_filter:
        print("  • Perceptual field   — context-level perception control active")
        goal_rules = [r for r in pf_filter.policy.get_all_rules() if r.source == "goal"]
        user_rules = [r for r in pf_filter.policy.get_all_rules() if r.source == "user"]
        if goal_rules:
            print(f"    Auto-rules from goal: {len(goal_rules)} contexts scoped")
            for r in goal_rules[:5]:
                app_short = r.app_pattern.rstrip("*").rsplit(".", 1)[-1]
                print(f"      {app_short}/{r.context_pattern} -> {r.level.value.upper()}")
        if user_rules:
            for r in user_rules:
                app_short = r.app_pattern.rstrip("*").rsplit(".", 1)[-1]
                print(f"    User rule: {app_short}/{r.context_pattern} -> {r.level.value.upper()}")
        print("    Commands during recording: field status | opaque | deny | full | list")
    print()
    print("Now perform your task naturally on this machine.")
    print("Type 'stop' (or 'done' / 'finish') here when complete.")
    cmds = "discard | pause | resume | annotate <text>"
    if pf_filter:
        cmds += " | field <cmd>"
    print(f"Other commands: {cmds}")
    if ctx.effective_mock:
        print("(mock mode: events are simulated, real OS interactions are not captured)")
    print()
    sys.stdout.flush()

    consent_notifier = None
    if pf_filter:
        from leapflow.recording.perceptual_field import ConsentNotifier
        consent_notifier = ConsentNotifier()
        pf_filter.set_consent_callback(consent_notifier.maybe_notify)

    health_monitor = _build_health_monitor(ctx)
    stop_blink = asyncio.Event()
    blink_task = asyncio.create_task(blink_recording(stop_blink))
    health_task = asyncio.create_task(_health_check_loop(health_monitor, stop_blink))

    user_command: Optional[str] = None
    try:
        while True:
            if consent_notifier:
                consent_notifier.flush()
            try:
                line = (await asyncio.to_thread(input)).strip()
            except (EOFError, KeyboardInterrupt):
                line = "stop"
                print()

            if line in ("stop", "done", "finish", "停止", "结束"):
                user_command = "stop"
                break

            if line in ("discard", "quit", "exit", "q", "放弃"):
                user_command = "quit"
                break

            if not line:
                continue

            if line in ("pause", "暂停"):
                ctx.session.pause_learning()
                print("Paused.")
                continue

            if line in ("resume", "继续"):
                ctx.session.resume_learning()
                print("Resumed.")
                continue

            if line.startswith("annotate ") or line.startswith("标注 "):
                text = line.split(" ", 1)[1] if " " in line else ""
                ctx.session.annotate(text)
                print("Annotation added.")
                continue

            if line.startswith("field"):
                _handle_field_command(ctx, line[5:].strip())
                continue

            print(f"Unknown command: {line}")
    finally:
        stop_blink.set()
        await blink_task
        health_task.cancel()
        try:
            await health_task
        except asyncio.CancelledError:
            pass

    _print_perception_summary(ctx)

    if user_command == "quit":
        await ctx.session.discard_learning()
        print("Learning discarded. No skill generated.")
        return 0

    # Note: auto_learn is controlled by LEAPFLOW_LEARN_AUTO_DISTILL config.
    # If LLM is not configured, the distillation will use heuristic path
    # or gracefully report the issue at line 343.

    print()
    print("[ STOPPING RECORDING ]")
    sys.stdout.flush()
    install_stop_progress(ctx)
    if ctx.settings.has_llm_credentials:
        install_learn_progress(ctx)

    try:
        result = await ctx.session.exit_learning()
    except Exception as e:
        print(f"Error stopping: {e}")
        return 0

    print()
    print("[ LEARNING STOPPED ]")
    print(f"  Trajectory: {result.trajectory_id}")
    print(f"  Steps:      {result.step_count}")
    print(f"  Duration:   {result.duration:.1f}s")
    if result.event_stats:
        stats_str = ", ".join(f"{k}={v}" for k, v in result.event_stats.items())
        print(f"  Events:     {stats_str}")

    if ctx.perception_session:
        ps = ctx.perception_session
        stats = getattr(ps, 'capture_stats', None)
        if stats:
            total = stats.get("success", 0) + stats.get("fail", 0)
            if total > 0:
                pct = stats["success"] * 100 // total
                frame_count = getattr(ps, 'frame_count', 0)
                print(f"  Captures:   {stats['success']}/{total} ({pct}% success), {frame_count} frames stored")

    if ctx.imitation and ctx.imitation.recorder.visual_degraded:
        print("  Fallback:   structural events recorded (visual channel degraded)")

    # Check learnability assessment
    report = getattr(result, 'learnability_report', None)
    if report:
        from leapflow.learning.learnability import LearnabilityDecision
        if report.decision == LearnabilityDecision.SKIP:
            print()
            print("[ SKIPPED \u2014 NOT WORTH LEARNING ]")
            print(f"  Reason: {report.reason}")
            print(f"  Score:  {report.score:.2f}")
            return 0
        elif report.decision == LearnabilityDecision.ASK:
            print()
            print(f"[ UNCERTAIN \u2014 Score: {report.score:.2f} ]")
            print(f"  {report.reason}")
            try:
                answer = (await asyncio.to_thread(
                    lambda: input("  Learn this operation? [y/N]: ").strip().lower()
                ))
            except (EOFError, KeyboardInterrupt):
                answer = "n"
            if answer not in ("y", "yes"):
                ctx.session.reject_learning()
                print("  Skipped.")
                return 0
            ctx.session.confirm_learning()

    if result.step_count == 0:
        print()
        print("[ NO STEPS RECORDED ]")
        if ctx.perception_session and hasattr(ctx.perception_session, 'capture_stats') and ctx.perception_session.capture_stats and ctx.perception_session.capture_stats.get("fail", 0) > 0:
            print("  Visual capture failed — check bridge connection and screen recording permission.")
        else:
            print("  No actions were observed. Try again and perform the task before stopping.")
        return 0

    if not ctx.settings.has_llm_credentials:
        print()
        print("[ WARNING: LLM NOT CONFIGURED ]")
        print(f"  Recording saved (trajectory: {result.trajectory_id}, {result.step_count} steps).")
        print("  Learning requires LEAPFLOW_LLM_API_KEY to produce useful skills.")
        print("  Configure it in .env, then run:")
        print(f"    leap relearn {result.trajectory_id}")
        return 0

    print()
    print("[ ANALYZING TRAJECTORY ]")
    sys.stdout.flush()
    print("[ DISTILLING ... ]")
    sys.stdout.flush()
    final = await ctx.session.await_learning() or result
    finish_learn_progress()

    candidates = list(final.candidates) if final and final.candidates else []
    activated = (
        set(final.activated_skill_names)
        if final and final.activated_skill_names
        else set()
    )
    confirmed = [c for c in candidates if c.title in activated]
    suggestions = [c for c in candidates if c.title not in activated]

    if confirmed:
        print()
        print(f"[ NEW SKILLS ({len(confirmed)}) ]")
        for c in confirmed:
            print(f"  * {c.title}")
            print(f"      confidence: {c.confidence:.0%}")
            triggers = (
                ", ".join(c.trigger_phrases[:3])
                if c.trigger_phrases else "(none)"
            )
            print(f"      triggers:   {triggers}")
            print(f"      steps:      {len(c.steps)}")
            if getattr(c, "pre_conditions", None):
                preconds = ", ".join(c.pre_conditions[:2])
                print(f"      preconds:   {preconds}")
    if suggestions:
        print()
        print(f"[ SUGGESTIONS ({len(suggestions)}) — pending review ]")
        for c in suggestions:
            print(f"  - {c.title} ({c.confidence:.0%})")

    if final.storage_path or final.audit_log_path:
        print()
        print("[ STORED AT ]")
        if final.storage_path:
            print(f"  Skills:  {final.storage_path}")
        if final.audit_log_path:
            print(f"  Audit:   {final.audit_log_path}")

    if confirmed and confirmed[0].trigger_phrases:
        print()
        print("[ TRY IT ]")
        print(f'  leap run "{confirmed[0].trigger_phrases[0]}"')
    return 0


def _build_health_monitor(ctx: "Context") -> "RecordingHealthMonitor":
    """Construct a RecordingHealthMonitor from the current context."""
    from leapflow.recording.health import RecordingHealthMonitor

    perception = ctx.perception_session if hasattr(ctx, "perception_session") else None
    recorder = ctx.imitation.recorder if ctx.imitation else None

    return RecordingHealthMonitor(
        perception=perception,
        recorder=recorder,
        visual_enabled=ctx.settings.visual_track_enabled,
        recording_mode=ctx.settings.recording_mode,
    )


async def _health_check_loop(
    monitor: "RecordingHealthMonitor",
    stop_event: asyncio.Event,
    interval_s: float = 10.0,
) -> None:
    """Periodic health check coroutine — runs until stop_event is set."""
    await asyncio.sleep(5.0)
    while not stop_event.is_set():
        try:
            health = await monitor.check()
            for warning in health.warnings:
                sys.stderr.write(f"\033[33m  \u26a0 {warning}\033[0m\n")
                sys.stderr.flush()
        except Exception:
            pass
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_s)
            return
        except asyncio.TimeoutError:
            pass


async def cmd_learn_resume(ctx: "Context", resume_id: str, timeout: Optional[float]) -> int:
    """Resume a previous learning session."""
    require_initialized(ctx)
    if timeout:
        ctx.session.idle_timeout = timeout

    try:
        session = await ctx.session.resume_session(resume_id)
    except Exception as e:
        print(f"Error resuming session: {e}")
        return 1

    traj = ctx.imitation.recorder.current_trajectory
    prior_steps = traj.step_count if traj else 0

    print("[ LEARNING RESUMED ]")
    print(f"  Session:    {session.session_id}")
    print(f"  Trajectory: {session.trajectory_id}")
    if session.goal:
        print(f"  Goal:       {session.goal}")
    print(f"  Prior steps: {prior_steps}")
    print()
    print("Continue performing the task. Type 'stop' when complete.")
    print("Commands: stop | discard | pause | resume | annotate <text>")
    print()
    sys.stdout.flush()

    stop_blink = asyncio.Event()
    blink_task = asyncio.create_task(blink_recording(stop_blink))

    user_command: Optional[str] = None
    try:
        while True:
            try:
                line = (await asyncio.to_thread(input)).strip()
            except (EOFError, KeyboardInterrupt):
                line = "quit"
                print()

            if line in ("stop", "done", "finish", "停止", "结束"):
                user_command = "stop"
                break

            if line in ("discard", "quit", "exit", "q", "放弃"):
                user_command = "quit"
                break

            if not line:
                continue

            if line in ("pause", "暂停"):
                ctx.session.pause_learning()
                print("Paused.")
                continue

            if line in ("resume", "继续"):
                ctx.session.resume_learning()
                print("Resumed.")
                continue

            if line.startswith("annotate ") or line.startswith("标注 "):
                text = line.split(" ", 1)[1] if " " in line else ""
                ctx.session.annotate(text)
                print("Annotation added.")
                continue

            if line.startswith("field"):
                _handle_field_command(ctx, line[5:].strip())
                continue

            print(f"Unknown command: {line}")
    finally:
        stop_blink.set()
        await blink_task

    _print_perception_summary(ctx)

    if user_command == "quit":
        await ctx.session.discard_learning()
        print("Learning discarded. No skill generated.")
        return 0

    print()
    print("[ STOPPING RECORDING ]")
    sys.stdout.flush()
    install_stop_progress(ctx)
    install_learn_progress(ctx)

    try:
        result = await ctx.session.exit_learning()
    except Exception as e:
        print(f"Error stopping: {e}")
        return 0

    print()
    print("[ LEARNING STOPPED ]")
    print(f"  Trajectory: {result.trajectory_id}")
    print(f"  Steps:      {result.step_count}")
    print(f"  Duration:   {result.duration:.1f}s")

    # Check learnability assessment
    report = getattr(result, 'learnability_report', None)
    if report:
        from leapflow.learning.learnability import LearnabilityDecision
        if report.decision == LearnabilityDecision.SKIP:
            print()
            print("[ SKIPPED \u2014 NOT WORTH LEARNING ]")
            print(f"  Reason: {report.reason}")
            print(f"  Score:  {report.score:.2f}")
            return 0
        elif report.decision == LearnabilityDecision.ASK:
            print()
            print(f"[ UNCERTAIN \u2014 Score: {report.score:.2f} ]")
            print(f"  {report.reason}")
            try:
                answer = (await asyncio.to_thread(
                    lambda: input("  Learn this operation? [y/N]: ").strip().lower()
                ))
            except (EOFError, KeyboardInterrupt):
                answer = "n"
            if answer not in ("y", "yes"):
                ctx.session.reject_learning()
                print("  Skipped.")
                return 0
            ctx.session.confirm_learning()

    if result.step_count == 0:
        print()
        print("[ NO STEPS RECORDED ]")
        print("  No actions were observed. Try again and perform the task before stopping.")
        return 0

    print()
    print("[ ANALYZING TRAJECTORY ]")
    sys.stdout.flush()
    print("[ DISTILLING ... ]")
    sys.stdout.flush()
    final = await ctx.session.await_learning() or result
    finish_learn_progress()

    candidates = list(final.candidates) if final and final.candidates else []
    activated = (
        set(final.activated_skill_names)
        if final and final.activated_skill_names
        else set()
    )
    confirmed = [c for c in candidates if c.title in activated]
    if confirmed:
        print()
        print(f"[ NEW SKILLS ({len(confirmed)}) ]")
        for c in confirmed:
            print(f"  * {c.title} ({c.confidence:.0%})")
    return 0
