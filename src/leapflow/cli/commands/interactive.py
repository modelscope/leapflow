"""Interactive subcommand — persistent REPL session."""

from __future__ import annotations

import asyncio
import sys
import time
from typing import TYPE_CHECKING, Optional

from leapflow.cli.banner import BRIGHT_CYAN, DIM, RESET, VERSION
from leapflow.cli.commands.run import _print_execution_result
from leapflow.cli.helpers import require_initialized
from leapflow.cli.tui import finish_input_frame, render_input_frame
from leapflow.engine import StreamEvent

if TYPE_CHECKING:
    from leapflow.cli.context import Context
    from leapflow.copilot.types import PredictionCandidate

# ── Constants ─────────────────────────────────────────────────────────────────
_CLI_INTERACTION_EVENT = "cli.interaction"
_CLI_EVENT_SOURCE = "interactive_repl"

# Module-level state for tracking the last displayed ghost hint
_last_hint: Optional["PredictionCandidate"] = None


async def cmd_interactive(ctx: "Context") -> int:
    """Persistent REPL session supporting teach/run/skills/chat switching."""
    require_initialized(ctx)
    from leapflow.utils.terminal_io import TerminalIOProvider
    from leapflow.engine.session import SessionMode

    io = TerminalIOProvider()

    def _on_progress(stage: str, current: int, total: int) -> None:
        sys.stderr.write(f"\r\033[2m  [{stage}] {current}/{total}\033[0m  ")
        sys.stderr.flush()

    def _on_complete(result) -> None:
        if result and result.new_skills:
            sys.stderr.write(
                f"\n\033[2m[LeapFlow] Learning complete \u2014 "
                f"{len(result.new_skills)} new skill(s)\033[0m\n"
            )
            for name in result.new_skills:
                sys.stderr.write(f"\033[2m        -> {name}\033[0m\n")
            sys.stderr.flush()

    def _on_step(idx: int, total: int, step_desc: str) -> None:
        print(f"  [{idx + 1}/{total}] {step_desc}")

    ctx.session.set_on_learn_progress(_on_progress)
    if hasattr(ctx.session, "set_on_learn_complete"):
        ctx.session.set_on_learn_complete(_on_complete)
    ctx.session.set_on_execute_step(_on_step)

    print(f"\n{BRIGHT_CYAN}LEAP{RESET} {DIM}v{VERSION}{RESET} \u2502 Interactive Mode")
    print(f"{DIM}Type 'help' for commands, 'exit' to quit{RESET}\n")

    loop = asyncio.get_event_loop()

    def _mode_name() -> str:
        if ctx.session and ctx.session.mode == SessionMode.LEARNING:
            if ctx.imitation and ctx.imitation.recorder.state.name == "PAUSED":
                return "paused"
            return "learning"
        elif ctx.session and ctx.session.mode == SessionMode.EXECUTING:
            return "executing"
        return "idle"

    def _skill_count() -> int:
        # Combine: SkillIndex (learned/manual/hub SKILL.md) + Registry (builtin)
        index_count = len(ctx.skill_index.get_entries()) if hasattr(ctx, 'skill_index') and ctx.skill_index else 0
        registry_count = len(ctx.registry.list_all()) if ctx.registry else 0
        return index_count + registry_count

    def _bridge_online() -> bool:
        return hasattr(ctx.rpc, "connected") and ctx.rpc.connected

    while True:
        _learning = ctx.session and ctx.session.mode == SessionMode.LEARNING
        if _learning:
            ctx.imitation.end_control_input()

        try:
            prompt = render_input_frame(
                _mode_name(), _skill_count(), _bridge_online()
            )
            # ── Ghost Hint: show Copilot suggestion before prompt ──
            global _last_hint
            if (
                not _learning
                and ctx.copilot_pipeline is not None
                and ctx.copilot_config is not None
            ):
                best = ctx.copilot_pipeline.get_best(
                    min_confidence=ctx.copilot_config.min_confidence_display
                )
                if best is not None:
                    _render_ghost_hint(best)
                    _last_hint = best
                    # Track shown for feedback collection
                    if ctx.copilot_feedback is not None and ctx.copilot_encoder is not None:
                        ctx.copilot_feedback.track_shown(
                            best, ctx.copilot_encoder.current_state
                        )
                else:
                    _last_hint = None
            else:
                _last_hint = None

            line = await loop.run_in_executor(None, lambda: input(prompt).strip())
            finish_input_frame()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not line:
            continue

        # ── Copilot: Idle timestamp update ──
        if ctx.copilot_idle is not None:
            ctx.copilot_idle.on_event_timestamp(time.time())

        # ── Copilot: Feedback collection ──
        if ctx.copilot_feedback is not None and _last_hint is not None:
            if _is_hint_accepted(line, _last_hint):
                signal = ctx.copilot_feedback.on_accept()
            elif ctx.copilot_encoder is not None:
                signal = ctx.copilot_feedback.on_next_action(
                    line, ctx.copilot_encoder.current_state
                )
            else:
                signal = None

            if signal and ctx.copilot_evolution:
                await ctx.copilot_evolution.process_feedback(signal)

            _last_hint = None

        if _learning:
            ctx.imitation.mark_control_input()

        # ── Exit ──
        if line in ("exit", "quit", "q", "退出"):
            if ctx.session and ctx.session.mode == SessionMode.LEARNING:
                try:
                    await ctx.session.exit_learning()
                    print("Learning stopped.")
                except Exception:
                    pass
            elif _learning:
                ctx.imitation.end_control_input()
            print("Bye!")
            break

        # ── Help ──
        if line in ("help", "帮助", "?"):
            print("Commands:")
            print("  teach start [goal]    — Start teaching mode")
            print("  teach stop            — Stop teaching and distill")
            print("  teach discard         — Discard recording (no skill generated)")
            print("  teach save            — Save session for later resume")
            print("  teach pause           — Pause recording")
            print("  teach resume          — Resume recording")
            print("  teach resume <id>     — Resume a saved session")
            print("  annotate <text>       — Add annotation during teaching")
            print("  skip [n]              — Mark last n steps as noise")
            print("  run <trigger>         — Execute a skill by trigger")
            print("  run --skill <name>    — Execute a skill by name")
            print("  skills                — List all skills")
            print("  skills show <name>    — Show skill details")
            print("  skills sessions       — List teaching sessions")
            print("  skills disable <name> — Disable a learned skill")
            print("  skills delete <name>  — Delete a learned skill")
            print("  hub <subcommand>      — Hub operations (push/pull/sync/search)")
            print("  arm <skill> --trigger — Arm a skill for scheduled execution")
            print("  tasks                 — List/manage scheduled tasks")
            print("  shortcut list         — List quick-reply shortcuts")
            print("  shortcut add <p>=<r>  — Add shortcut (pattern = reply)")
            print("  shortcut remove <p>   — Remove a shortcut")
            print("  <text>                — Chat / natural language")
            print("  help                  — Show this help")
            print("  exit                  — Quit")
            print()
            continue

        # ── Teach/Learn commands ──
        if line.startswith("teach start") or line.startswith("教学开始") or line == "teach":
            if ctx.session and ctx.session.mode == SessionMode.LEARNING:
                print("Already in teaching mode. Say 'teach stop' to end.")
                continue
            goal = ""
            if line.startswith("teach start "):
                goal = line[len("teach start "):]
            elif line.startswith("教学开始 "):
                goal = line[len("教学开始 "):]
            try:
                session = await ctx.session.enter_learning(goal=goal)
                print(f"Teaching started. Session: {session.session_id}")
                print(f"Trajectory: {session.trajectory_id}")
                if goal:
                    print(f"Goal: {goal}")
                print("Commands: stop | discard | pause | resume | annotate <text> | skip [n]")
            except Exception as e:
                print(f"Error: {e}")
            continue

        if line in ("teach stop", "stop", "done", "教学结束", "结束"):
            if not ctx.session or ctx.session.mode != SessionMode.LEARNING:
                if _learning:
                    ctx.imitation.end_control_input()
                print("Not in teaching mode.")
                continue
            try:
                print("\n[ STOPPING RECORDING ... ]")
                sys.stdout.flush()
                result = await ctx.session.exit_learning()
                print(f"  Recording stopped. Steps: {result.step_count} | Duration: {result.duration:.1f}s")

                # Check learnability assessment
                report = getattr(result, 'learnability_report', None)
                if report:
                    from leapflow.learning.learnability import LearnabilityDecision
                    if report.decision == LearnabilityDecision.SKIP:
                        print(f"\n[ SKIPPED \u2014 NOT WORTH LEARNING ]")
                        print(f"  Reason: {report.reason}")
                        print(f"  Score: {report.score:.2f}")
                        continue
                    elif report.decision == LearnabilityDecision.ASK:
                        print(f"\n[ UNCERTAIN \u2014 Score: {report.score:.2f} ]")
                        print(f"  {report.reason}")
                        answer = await loop.run_in_executor(
                            None, lambda: input("  Learn this operation? [y/N]: ").strip().lower()
                        )
                        if answer not in ("y", "yes"):
                            ctx.session.reject_learning()
                            print("  Skipped.")
                            continue
                        ctx.session.confirm_learning()

                # Trigger distillation if steps recorded and LLM available
                if result.step_count > 0 and ctx.settings.has_llm_credentials:
                    print("\n[ ANALYZING VIDEO + EVENTS ... ]")
                    sys.stdout.flush()
                    print("\n[ DISTILLING ... ]")
                    sys.stdout.flush()
                    final = await ctx.session.await_learning()
                    if final and final.candidates:
                        candidates = list(final.candidates)
                        activated = set(final.activated_skill_names) if final.activated_skill_names else set()
                        print(f"  Candidates: {len(candidates)}")
                        if activated:
                            print(f"  Activated:  {', '.join(activated)}")
                    else:
                        print("  No skills distilled (insufficient signal).")
                elif result.step_count > 0:
                    print("\n[ SKIPPED: LLM not configured — run 'leap relearn' later ]")

                if result.new_skills:
                    print(f"New skills: {', '.join(result.new_skills)}")
                if result.suggestions > 0:
                    print(f"Suggestions pending: {result.suggestions}")
            except Exception as e:
                print(f"Error: {e}")
            continue

        if line in ("teach quit", "teach discard", "quit", "discard", "退出教学", "放弃"):
            if not ctx.session or ctx.session.mode != SessionMode.LEARNING:
                print("Not in teaching mode.")
                continue
            try:
                await ctx.session.discard_learning()
                print("\nTeaching discarded. No skill generated.")
            except Exception as e:
                print(f"Error: {e}")
            continue

        if line in ("teach save", "save", "保存"):
            if not ctx.session or ctx.session.mode != SessionMode.LEARNING:
                print("Not in teaching mode.")
                continue
            try:
                learning = ctx.session.current_session
                await ctx.session.abandon_learning()
                print("\nTeaching paused. Session saved for later resume.")
                if learning:
                    print(f"  Session ID: {learning.session_id}")
                    print(f"  Resume with: teach resume {learning.session_id}")
            except Exception as e:
                print(f"Error: {e}")
            continue

        # All remaining commands: end control input before processing
        if _learning:
            ctx.imitation.end_control_input()

        if line in ("teach pause", "pause", "暂停"):
            if ctx.session:
                ctx.session.pause_learning()
                print("Recording paused.")
            continue

        if line in ("teach resume", "resume", "继续"):
            if ctx.session and ctx.session.mode == SessionMode.LEARNING:
                ctx.session.resume_learning()
                print("Recording resumed.")
            else:
                print("Not in teaching mode. Use 'teach resume <id>' to resume a saved session.")
            continue

        if line.startswith("teach resume "):
            resume_id = line[len("teach resume "):].strip()
            if not resume_id:
                print("Usage: teach resume <session_id>")
                continue
            if ctx.session and ctx.session.mode == SessionMode.LEARNING:
                print("Already in teaching mode. Stop current session first.")
                continue
            try:
                session = await ctx.session.resume_session(resume_id)
                traj = ctx.imitation.get_trajectory(session.trajectory_id)
                step_count = traj.step_count if traj else 0
                print(f"Resumed session: {session.session_id}")
                print(f"Trajectory: {session.trajectory_id} ({step_count} existing steps)")
                if session.goal:
                    print(f"Goal: {session.goal}")
                print("Commands: stop | discard | save | pause | resume | annotate <text>")
            except Exception as e:
                print(f"Error resuming: {e}")
            continue

        if line.startswith("annotate ") or line.startswith("标注 "):
            text = line.split(" ", 1)[1] if " " in line else ""
            if ctx.session and text:
                ctx.session.annotate(text)
                print("Annotation added.")
            continue

        # ── Skip (noise marking during learning) ──
        if line.startswith("skip") or line.startswith("跳过"):
            parts = line.split()
            n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
            if ctx.session:
                count = ctx.session.mark_skip(n)
                print(f"Marked {count} step(s) as noise.")
            else:
                print("Not in learning mode.")
            continue

        # ── Skills commands ──
        if line in ("skills", "skills list", "技能列表"):
            skills = ctx.registry.list_all() if ctx.registry else []
            if not skills:
                print("No skills registered.")
            else:
                print(f"{'Name':<25} {'Version':<10} {'Confidence':<12} Description")
                print("-" * 80)
                for s in skills:
                    m = s.metadata
                    print(f"{s.name:<25} v{m.version:<9} {m.confidence:<11.0%} {s.description[:40]}")
            continue

        if line.startswith("skills show "):
            name = line[len("skills show "):]
            skill = ctx.registry.get(name) if ctx.registry else None
            if skill is None:
                print(f"Skill '{name}' not found.")
            else:
                m = skill.metadata
                print(f"Name:        {skill.name}")
                print(f"Description: {skill.description}")
                print(f"Version:     v{m.version}")
                print(f"Confidence:  {m.confidence:.0%}")
                if skill.triggers:
                    print(f"Triggers:    {', '.join(skill.triggers)}")
            continue

        if line.startswith("skills disable "):
            name = line[len("skills disable "):]
            found = False
            if ctx.skill_lib and ctx.skill_lib.deactivate_parameterized(name):
                found = True
            if ctx.registry and ctx.registry.unregister(name):
                found = True
            print(f"Skill '{name}' disabled." if found else f"Skill '{name}' not found.")
            continue

        if line.startswith("skills delete "):
            name = line[len("skills delete "):]
            found = False
            if ctx.skill_lib:
                stored = ctx.skill_lib.load_skill_by_title(name)
                if stored:
                    stored.status = "deleted"
                    ctx.skill_lib.update_skill(stored)
                    found = True
            if ctx.registry and ctx.registry.unregister(name):
                found = True
            print(f"Skill '{name}' deleted." if found else f"Skill '{name}' not found.")
            continue

        # ── Hub commands ──
        if line.startswith("hub"):
            from leapflow.cli.commands.hub import cmd_hub
            hub_args = line.split()[1:] if len(line.split()) > 1 else []
            await cmd_hub(ctx, hub_args)
            continue

        # ── Scheduler commands ──
        if line.startswith("arm"):
            from leapflow.cli.commands.scheduler import cmd_arm
            arm_args = line.split()[1:] if len(line.split()) > 1 else []
            await cmd_arm(ctx, arm_args)
            continue

        if line.startswith("tasks"):
            from leapflow.cli.commands.scheduler import cmd_tasks
            tasks_args = line.split()[1:] if len(line.split()) > 1 else []
            await cmd_tasks(ctx, tasks_args)
            continue

        # ── Run command ──
        if line.startswith("run "):
            trigger_or_name = line[4:].strip()
            if trigger_or_name.startswith("--skill "):
                skill_name = trigger_or_name[len("--skill "):]
                result = await ctx.session.execute_skill(
                    skill_name, io=io
                )
            else:
                matched = ctx.session.find_skill(trigger_or_name)
                if matched:
                    result = await ctx.session.execute_skill(
                        matched, io=io
                    )
                else:
                    _streamed = False
                    async for event in ctx.engine.run_stream(trigger_or_name):
                        if isinstance(event, StreamEvent):
                            if event.type == "chunk":
                                print(event.content, end="", flush=True)
                                _streamed = True
                            elif event.type == "final" and not _streamed:
                                print(event.content, end="", flush=True)
                        else:
                            print(event, end="", flush=True)
                    print()
                    continue

            _print_execution_result(result)
            continue

        # ── Shortcut commands ──
        if line in ("shortcut list", "shortcut", "快捷短语"):
            shortcuts = ctx.shortcuts.list_all()
            if not shortcuts:
                print("No shortcuts configured.")
            else:
                print(f"Shortcuts ({len(shortcuts)}):")
                for pattern, reply in shortcuts.items():
                    print(f"  {pattern} → {reply}")
            print()
            continue

        if line.startswith("shortcut add ") or line.startswith("快捷短语 添加 "):
            rest = line.split(" ", 2)[-1] if line.startswith("shortcut add ") else line.split(" ", 2)[-1]
            if "=" not in rest:
                print("Usage: shortcut add <pattern> = <reply>")
                continue
            pattern, reply = rest.split("=", 1)
            pattern, reply = pattern.strip(), reply.strip()
            if not pattern or not reply:
                print("Usage: shortcut add <pattern> = <reply>")
                continue
            ctx.shortcuts.add(pattern, reply)
            print(f"Shortcut added: {pattern} → {reply}")
            continue

        if line.startswith("shortcut remove ") or line.startswith("快捷短语 删除 "):
            pattern = line.split(" ", 2)[-1].strip()
            if ctx.shortcuts.remove(pattern):
                print(f"Shortcut removed: {pattern}")
            else:
                print(f"Shortcut not found: {pattern}")
            continue

        # ── Default: Natural language (try skill trigger, then chat) ──
        if ctx.session and ctx.session.mode == SessionMode.LEARNING:
            ctx.session.annotate(line)
            print("(Noted as annotation during learning)")
            await _inject_copilot_event(ctx, line, _mode_name)
            continue

        matched = ctx.session.find_skill(line) if ctx.session else None
        if matched:
            result = await ctx.session.execute_skill(matched, io=io)
            _print_execution_result(result)
        else:
            streamed = False
            async for event in ctx.engine.run_stream(line):
                if isinstance(event, StreamEvent):
                    if event.type == "chunk":
                        print(event.content, end="", flush=True)
                        streamed = True
                    elif event.type == "final" and not streamed:
                        print(event.content, end="", flush=True)
                else:
                    print(event, end="", flush=True)
            print()

        # ── Copilot: Synthetic event injection after command processing ──
        await _inject_copilot_event(ctx, line, _mode_name)

    return 0


# ── Copilot helper functions ──────────────────────────────────────────────────


def _render_ghost_hint(candidate: "PredictionCandidate") -> None:
    """Render a ghost hint below the prompt line.

    Uses dim ANSI escape codes for unobtrusive display.
    Only emits ANSI when stdout is a TTY.
    """
    hint_text = f"  \U0001f4a1 {candidate.action_description}"
    confidence_pct = int(candidate.confidence * 100)
    if sys.stdout.isatty():
        print(f"\033[2m{hint_text} ({confidence_pct}% confidence \u2014 Tab to accept)\033[0m")
    else:
        print(f"{hint_text} ({confidence_pct}% confidence)")


def _is_hint_accepted(user_input: str, hint: "PredictionCandidate") -> bool:
    """Check if the user input effectively accepts the ghost hint.

    MVP heuristic: exact match or first-word prefix match.
    Limitations: may false-positive on commands sharing the same verb
    (e.g., hint="run skill_a" vs input="run skill_b").
    Future: replace with explicit Tab-to-accept or high-similarity scoring.
    """
    desc = hint.action_description.lower().strip()
    inp = user_input.lower().strip()
    # Exact match or the input starts with the suggestion's key verb
    return inp == desc or (desc and inp.startswith(desc.split()[0]) and len(inp) > 2)


async def _inject_copilot_event(
    ctx: "Context", line: str, mode_fn
) -> None:
    """Synthesize a CLI interaction event and inject into EventBus for Copilot."""
    if ctx.copilot_pipeline is None or ctx.copilot_encoder is None:
        return
    from leapflow.domain.events import PRIORITY_NORMAL, SystemEvent

    synth_event = SystemEvent(
        event_type=_CLI_INTERACTION_EVENT,
        source=_CLI_EVENT_SOURCE,
        payload={"input": line, "mode": mode_fn()},
        timestamp=time.time(),
        priority=PRIORITY_NORMAL,
    )
    # 1) Inject event into EventBus → CopilotEventSubscriber → ContextEncoder
    await ctx.event_bus.handle_event(synth_event.event_type, synth_event.payload)

    # 2) Drive SpeculativePipeline with updated context (snapshot to avoid shared state)
    await ctx.copilot_pipeline.on_action_observed(ctx.copilot_encoder.snapshot())
