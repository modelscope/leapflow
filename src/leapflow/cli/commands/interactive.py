"""Interactive subcommand — persistent REPL session with rich TUI."""

from __future__ import annotations

import sys
import time
from typing import TYPE_CHECKING, Optional

from leapflow.cli.helpers import require_initialized
from leapflow.cli.commands.run import _print_execution_result
from leapflow.engine import StreamEvent

if TYPE_CHECKING:
    from leapflow.cli.context import Context
    from leapflow.copilot.types import PredictionCandidate

_CLI_INTERACTION_EVENT = "cli.interaction"
_CLI_EVENT_SOURCE = "interactive_repl"

_last_hint: Optional["PredictionCandidate"] = None


async def cmd_interactive(ctx: "Context") -> int:
    """Persistent REPL session with rich TUI experience."""
    require_initialized(ctx)

    # ── TUI setup ──
    from leapflow.cli.tui_app import detect_theme, LeapConsole, LeapInput, StreamRenderer
    from leapflow.cli.tui_app.status import StatusBar
    from leapflow.cli.banner import display_welcome
    from leapflow.utils.terminal_io import TerminalIOProvider
    from leapflow.engine.session import SessionMode

    theme = detect_theme()
    console = LeapConsole(theme)
    status = StatusBar()
    io = TerminalIOProvider()

    leap_input = LeapInput(
        theme,
        data_dir=ctx.settings.data_dir if hasattr(ctx.settings, "data_dir") else None,
    )

    # ── Callbacks ──
    def _on_progress(stage: str, current: int, total: int) -> None:
        console.system(f"[{stage}] {current}/{total}")

    def _on_complete(result) -> None:
        if result and result.new_skills:
            console.success(
                f"Learning complete — {len(result.new_skills)} new skill(s)"
            )
            for name in result.new_skills:
                console.system(f"  → {name}")

    def _on_step(idx: int, total: int, step_desc: str) -> None:
        console.system(f"[{idx + 1}/{total}] {step_desc}")

    ctx.session.set_on_learn_progress(_on_progress)
    if hasattr(ctx.session, "set_on_learn_complete"):
        ctx.session.set_on_learn_complete(_on_complete)
    ctx.session.set_on_execute_step(_on_step)

    # ── Banner ──
    display_welcome()
    console.rule()
    console.print()

    # ── Helpers ──
    def _mode_name() -> str:
        if ctx.session and ctx.session.mode == SessionMode.LEARNING:
            if ctx.imitation and ctx.imitation.recorder.state.name == "PAUSED":
                return "paused"
            return "learning"
        elif ctx.session and ctx.session.mode == SessionMode.EXECUTING:
            return "executing"
        return "idle"

    def _skill_count() -> int:
        index_count = len(ctx.skill_index.get_entries()) if hasattr(ctx, "skill_index") and ctx.skill_index else 0
        registry_count = len(ctx.registry.list_all()) if ctx.registry else 0
        return index_count + registry_count

    def _platform_online() -> bool:
        return hasattr(ctx.rpc, "connected") and ctx.rpc.connected

    def _update_status() -> None:
        status.update(
            mode=_mode_name(),
            skill_count=_skill_count(),
            platform_online=_platform_online(),
            model_name=getattr(ctx.settings, "model", ""),
            session_turns=getattr(ctx.engine, "turn_count", 0),
        )

    async def _stream_response(prompt_text: str) -> None:
        """Stream LLM response with rich rendering."""
        renderer = StreamRenderer(console._console, theme)
        renderer.start()
        try:
            async for event in ctx.engine.run_stream(prompt_text):
                if isinstance(event, StreamEvent):
                    if event.type == "chunk":
                        renderer.feed(event.content)
                    elif event.type == "thinking":
                        renderer.feed_thinking(event.content)
                    elif event.type == "tool_start":
                        renderer.tool_started(event.content)
                    elif event.type == "tool_complete":
                        renderer.tool_finished(event.content)
                    elif event.type == "final":
                        if not renderer.text:
                            renderer.feed(event.content)
                else:
                    renderer.feed(str(event))
        finally:
            renderer.finish()

        final_text = renderer.text.strip()
        if final_text:
            console.markdown(final_text)
        console.print()

    # ── Main loop ──
    while True:
        _learning = ctx.session and ctx.session.mode == SessionMode.LEARNING
        if _learning:
            ctx.imitation.end_control_input()

        _update_status()

        # Show ghost hint from Copilot
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
                _render_ghost_hint(console, best)
                _last_hint = best
                if ctx.copilot_feedback is not None and ctx.copilot_encoder is not None:
                    ctx.copilot_feedback.track_shown(
                        best, ctx.copilot_encoder.current_state
                    )
            else:
                _last_hint = None
        else:
            _last_hint = None

        try:
            line = await leap_input.prompt(
                _mode_name(),
                bottom_toolbar=status,
            )
        except (EOFError, KeyboardInterrupt):
            console.print()
            console.system("Bye!")
            break

        if not line:
            continue

        # Copilot idle timestamp
        if ctx.copilot_idle is not None:
            ctx.copilot_idle.on_event_timestamp(time.time())

        # Copilot feedback
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
                    console.system("Learning stopped.")
                except Exception:
                    pass
            elif _learning:
                ctx.imitation.end_control_input()
            console.system("Bye!")
            break

        # ── Help ──
        if line in ("help", "帮助", "?"):
            _show_help(console)
            continue

        # ── Teach commands ──
        if await _handle_teach(ctx, console, line, _learning):
            continue

        if _learning:
            ctx.imitation.end_control_input()

        # ── Skills commands ──
        if _handle_skills(ctx, console, line):
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
            await cmd_arm(ctx, line.split()[1:] if len(line.split()) > 1 else [])
            continue

        if line.startswith("tasks"):
            from leapflow.cli.commands.scheduler import cmd_tasks
            await cmd_tasks(ctx, line.split()[1:] if len(line.split()) > 1 else [])
            continue

        # ── Run command ──
        if line.startswith("run "):
            trigger_or_name = line[4:].strip()
            if trigger_or_name.startswith("--skill "):
                skill_name = trigger_or_name[len("--skill "):]
                result = await ctx.session.execute_skill(skill_name, io=io)
            else:
                matched = ctx.session.find_skill(trigger_or_name)
                if matched:
                    result = await ctx.session.execute_skill(matched, io=io)
                else:
                    await _stream_response(trigger_or_name)
                    await _inject_copilot_event(ctx, line, _mode_name)
                    continue
            _print_execution_result(result)
            continue

        # ── Shortcut commands ──
        if _handle_shortcuts(ctx, console, line):
            continue

        # ── Default: Natural language ──
        if ctx.session and ctx.session.mode == SessionMode.LEARNING:
            ctx.session.annotate(line)
            console.system("(Noted as annotation during learning)")
            await _inject_copilot_event(ctx, line, _mode_name)
            continue

        matched = ctx.session.find_skill(line) if ctx.session else None
        if matched:
            result = await ctx.session.execute_skill(matched, io=io)
            _print_execution_result(result)
        else:
            await _stream_response(line)

        await _inject_copilot_event(ctx, line, _mode_name)

    return 0


# ── Command handlers ──────────────────────────────────────────────────────────


async def _handle_teach(ctx: "Context", console, line: str, learning: bool) -> bool:
    """Handle teach/learn commands. Returns True if handled."""
    from leapflow.engine.session import SessionMode
    import asyncio

    if line.startswith("teach start") or line.startswith("教学开始") or line == "teach":
        if ctx.session and ctx.session.mode == SessionMode.LEARNING:
            console.warning("Already in teaching mode. Say 'teach stop' to end.")
            return True
        goal = ""
        if line.startswith("teach start "):
            goal = line[len("teach start "):]
        elif line.startswith("教学开始 "):
            goal = line[len("教学开始 "):]
        try:
            session = await ctx.session.enter_learning(goal=goal)
            console.success(f"Teaching started — session {session.session_id}")
            if goal:
                console.system(f"Goal: {goal}")
            console.system("Commands: stop │ discard │ pause │ resume │ annotate <text> │ skip [n]")
        except Exception as e:
            console.error(str(e))
        return True

    if line in ("teach stop", "stop", "done", "教学结束", "结束"):
        if not ctx.session or ctx.session.mode != SessionMode.LEARNING:
            if learning:
                ctx.imitation.end_control_input()
            console.warning("Not in teaching mode.")
            return True
        try:
            console.system("Stopping recording…")
            result = await ctx.session.exit_learning()
            console.success(f"Recording stopped — {result.step_count} steps, {result.duration:.1f}s")

            report = getattr(result, "learnability_report", None)
            if report:
                from leapflow.learning.learnability import LearnabilityDecision
                if report.decision == LearnabilityDecision.SKIP:
                    console.warning(f"Not worth learning — {report.reason} (score: {report.score:.2f})")
                    return True
                elif report.decision == LearnabilityDecision.ASK:
                    console.system(f"Uncertain (score: {report.score:.2f}) — {report.reason}")
                    from leapflow.cli.tui_app.input import LeapInput
                    # Use simple input for yes/no
                    answer = input("  Learn this? [y/N]: ").strip().lower()
                    if answer not in ("y", "yes"):
                        ctx.session.reject_learning()
                        console.system("Skipped.")
                        return True
                    ctx.session.confirm_learning()

            if result.step_count > 0 and ctx.settings.has_llm_credentials:
                console.system("Analyzing and distilling…")
                final = await ctx.session.await_learning()
                if final and final.candidates:
                    candidates = list(final.candidates)
                    activated = set(final.activated_skill_names) if final.activated_skill_names else set()
                    console.success(f"Distilled {len(candidates)} candidate(s)")
                    if activated:
                        for name in activated:
                            console.system(f"  → {name}")
                else:
                    console.system("No skills distilled (insufficient signal).")
            elif result.step_count > 0:
                console.warning("LLM not configured — run 'leap relearn' later")

            if result.new_skills:
                console.success(f"New skills: {', '.join(result.new_skills)}")
            if result.suggestions > 0:
                console.system(f"Suggestions pending: {result.suggestions}")
        except Exception as e:
            console.error(str(e))
        return True

    if line in ("teach quit", "teach discard", "quit", "discard", "退出教学", "放弃"):
        if not ctx.session or ctx.session.mode != SessionMode.LEARNING:
            console.warning("Not in teaching mode.")
            return True
        try:
            await ctx.session.discard_learning()
            console.system("Teaching discarded. No skill generated.")
        except Exception as e:
            console.error(str(e))
        return True

    if line in ("teach save", "save", "保存"):
        if not ctx.session or ctx.session.mode != SessionMode.LEARNING:
            console.warning("Not in teaching mode.")
            return True
        try:
            learning_session = ctx.session.current_session
            await ctx.session.abandon_learning()
            console.success("Session saved for later resume.")
            if learning_session:
                console.system(f"  Resume with: teach resume {learning_session.session_id}")
        except Exception as e:
            console.error(str(e))
        return True

    if line in ("teach pause", "pause", "暂停"):
        if ctx.session:
            ctx.session.pause_learning()
            console.system("Recording paused.")
        return True

    if line in ("teach resume", "resume", "继续"):
        from leapflow.engine.session import SessionMode as SM
        if ctx.session and ctx.session.mode == SM.LEARNING:
            ctx.session.resume_learning()
            console.system("Recording resumed.")
        else:
            console.warning("Not in teaching mode. Use 'teach resume <id>'.")
        return True

    if line.startswith("teach resume "):
        resume_id = line[len("teach resume "):].strip()
        if not resume_id:
            console.warning("Usage: teach resume <session_id>")
            return True
        if ctx.session and ctx.session.mode == SessionMode.LEARNING:
            console.warning("Already in teaching mode. Stop first.")
            return True
        try:
            session = await ctx.session.resume_session(resume_id)
            traj = ctx.imitation.get_trajectory(session.trajectory_id)
            step_count = traj.step_count if traj else 0
            console.success(f"Resumed session {session.session_id} ({step_count} existing steps)")
            if session.goal:
                console.system(f"Goal: {session.goal}")
        except Exception as e:
            console.error(f"Resume failed: {e}")
        return True

    if line.startswith("annotate ") or line.startswith("标注 "):
        text = line.split(" ", 1)[1] if " " in line else ""
        if ctx.session and text:
            ctx.session.annotate(text)
            console.system("Annotation added.")
        return True

    if line.startswith("skip") or line.startswith("跳过"):
        parts = line.split()
        n = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
        if ctx.session:
            count = ctx.session.mark_skip(n)
            console.system(f"Marked {count} step(s) as noise.")
        else:
            console.warning("Not in learning mode.")
        return True

    return False


def _handle_skills(ctx: "Context", console, line: str) -> bool:
    """Handle skills commands. Returns True if handled."""
    if line in ("skills", "skills list", "技能列表"):
        skills = ctx.registry.list_all() if ctx.registry else []
        if not skills:
            console.system("No skills registered.")
        else:
            from rich.table import Table
            table = Table(show_header=True, header_style="bold", border_style="dim")
            table.add_column("Name", style="cyan", max_width=30)
            table.add_column("Version", justify="center")
            table.add_column("Confidence", justify="center")
            table.add_column("Description", max_width=40)
            for s in skills:
                m = s.metadata
                table.add_row(
                    s.name,
                    f"v{m.version}",
                    f"{m.confidence:.0%}",
                    s.description[:40],
                )
            console._console.print(table)
        return True

    if line.startswith("skills show "):
        name = line[len("skills show "):]
        skill = ctx.registry.get(name) if ctx.registry else None
        if skill is None:
            console.warning(f"Skill '{name}' not found.")
        else:
            from rich.panel import Panel
            from rich.text import Text
            m = skill.metadata
            info = Text()
            info.append(f"Name:        {skill.name}\n")
            info.append(f"Description: {skill.description}\n")
            info.append(f"Version:     v{m.version}\n")
            info.append(f"Confidence:  {m.confidence:.0%}\n")
            if skill.triggers:
                info.append(f"Triggers:    {', '.join(skill.triggers)}")
            console._console.print(Panel(info, title=skill.name, border_style="cyan"))
        return True

    if line.startswith("skills disable "):
        name = line[len("skills disable "):]
        found = False
        if ctx.skill_lib and ctx.skill_lib.deactivate_parameterized(name):
            found = True
        if ctx.registry and ctx.registry.unregister(name):
            found = True
        if found:
            console.success(f"Skill '{name}' disabled.")
        else:
            console.warning(f"Skill '{name}' not found.")
        return True

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
        if found:
            console.success(f"Skill '{name}' deleted.")
        else:
            console.warning(f"Skill '{name}' not found.")
        return True

    return False


def _handle_shortcuts(ctx: "Context", console, line: str) -> bool:
    """Handle shortcut commands. Returns True if handled."""
    if line in ("shortcut list", "shortcut", "快捷短语"):
        shortcuts = ctx.shortcuts.list_all()
        if not shortcuts:
            console.system("No shortcuts configured.")
        else:
            console.system(f"Shortcuts ({len(shortcuts)}):")
            for pattern, reply in shortcuts.items():
                console.system(f"  {pattern} → {reply}")
        return True

    if line.startswith("shortcut add ") or line.startswith("快捷短语 添加 "):
        rest = line.split(" ", 2)[-1]
        if "=" not in rest:
            console.warning("Usage: shortcut add <pattern> = <reply>")
            return True
        pattern, reply = rest.split("=", 1)
        pattern, reply = pattern.strip(), reply.strip()
        if not pattern or not reply:
            console.warning("Usage: shortcut add <pattern> = <reply>")
            return True
        ctx.shortcuts.add(pattern, reply)
        console.success(f"Shortcut added: {pattern} → {reply}")
        return True

    if line.startswith("shortcut remove ") or line.startswith("快捷短语 删除 "):
        pattern = line.split(" ", 2)[-1].strip()
        if ctx.shortcuts.remove(pattern):
            console.success(f"Shortcut removed: {pattern}")
        else:
            console.warning(f"Shortcut not found: {pattern}")
        return True

    return False


def _show_help(console) -> None:
    """Display help with rich formatting."""
    from rich.table import Table

    table = Table(
        title="Commands",
        show_header=True,
        header_style="bold",
        border_style="dim",
        title_style="bold cyan",
        padding=(0, 1),
    )
    table.add_column("Command", style="cyan", no_wrap=True)
    table.add_column("Description")

    commands = [
        ("teach start [goal]", "Start teaching mode"),
        ("teach stop / stop", "Stop and distill skill"),
        ("teach pause / resume", "Pause/resume recording"),
        ("teach discard", "Discard recording"),
        ("annotate <text>", "Add annotation during teaching"),
        ("skip [n]", "Mark last n steps as noise"),
        ("run <trigger>", "Execute a skill by trigger"),
        ("run --skill <name>", "Execute by exact name"),
        ("skills", "List all skills"),
        ("skills show <name>", "Show skill details"),
        ("hub push/pull/search", "Hub operations"),
        ("shortcut list/add/remove", "Manage quick-reply shortcuts"),
        ("<text>", "Chat / natural language"),
        ("exit", "Quit"),
    ]
    for cmd, desc in commands:
        table.add_row(cmd, desc)

    console._console.print(table)
    console.print()


# ── Copilot helpers ───────────────────────────────────────────────────────────


def _render_ghost_hint(console, candidate: "PredictionCandidate") -> None:
    """Render a ghost hint in the console."""
    confidence_pct = int(candidate.confidence * 100)
    console.system(
        f"💡 {candidate.action_description} ({confidence_pct}% — Tab to accept)"
    )


def _is_hint_accepted(user_input: str, hint: "PredictionCandidate") -> bool:
    desc = hint.action_description.lower().strip()
    inp = user_input.lower().strip()
    return inp == desc or (desc and inp.startswith(desc.split()[0]) and len(inp) > 2)


async def _inject_copilot_event(ctx: "Context", line: str, mode_fn) -> None:
    """Synthesize a CLI interaction event for the Copilot pipeline."""
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
    await ctx.event_bus.handle_event(synth_event.event_type, synth_event.payload)
    await ctx.copilot_pipeline.on_action_observed(ctx.copilot_encoder.snapshot())
