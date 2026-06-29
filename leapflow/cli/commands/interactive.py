"""Interactive subcommand — persistent REPL session."""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING

from leapflow.cli.banner import BRIGHT_CYAN, DIM, RESET, VERSION
from leapflow.cli.commands.run import _print_execution_result
from leapflow.cli.helpers import require_initialized
from leapflow.cli.tui import finish_input_frame, render_input_frame

if TYPE_CHECKING:
    from leapflow.cli.context import Context


async def cmd_interactive(ctx: "Context") -> int:
    """Persistent REPL session supporting learn/run/skills/chat switching."""
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
                f"\n\033[2m[LEAP] Learning complete — "
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
            line = await loop.run_in_executor(None, lambda: input(prompt).strip())
            finish_input_frame()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not line:
            continue

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
            print("  learn start [goal]    — Start learning mode")
            print("  learn stop            — Stop learning and distill")
            print("  learn discard         — Discard recording (no skill generated)")
            print("  learn save            — Save session for later resume")
            print("  learn pause           — Pause recording")
            print("  learn resume          — Resume recording")
            print("  learn resume <id>     — Resume a saved session")
            print("  annotate <text>       — Add annotation during learning")
            print("  skip [n]              — Mark last n steps as noise")
            print("  run <trigger>         — Execute a skill by trigger")
            print("  run --skill <name>    — Execute a skill by name")
            print("  skills                — List all skills")
            print("  skills show <name>    — Show skill details")
            print("  skills sessions       — List learning sessions")
            print("  skills disable <name> — Disable a learned skill")
            print("  skills delete <name>  — Delete a learned skill")
            print("  shortcut list         — List quick-reply shortcuts")
            print("  shortcut add <p>=<r>  — Add shortcut (pattern = reply)")
            print("  shortcut remove <p>   — Remove a shortcut")
            print("  <text>                — Chat / natural language")
            print("  help                  — Show this help")
            print("  exit                  — Quit")
            print()
            continue

        # ── Learn commands ──
        if line.startswith("learn start") or line.startswith("教学开始") or line == "learn":
            if ctx.session and ctx.session.mode == SessionMode.LEARNING:
                print("Already in learning mode. Say 'learn stop' to end.")
                continue
            goal = ""
            if line.startswith("learn start "):
                goal = line[len("learn start "):]
            elif line.startswith("教学开始 "):
                goal = line[len("教学开始 "):]
            try:
                session = await ctx.session.enter_learning(goal=goal)
                print(f"Learning started. Session: {session.session_id}")
                print(f"Trajectory: {session.trajectory_id}")
                if goal:
                    print(f"Goal: {goal}")
                print("Commands: stop | discard | pause | resume | annotate <text> | skip [n]")
            except Exception as e:
                print(f"Error: {e}")
            continue

        if line in ("learn stop", "stop", "done", "教学结束", "结束"):
            if not ctx.session or ctx.session.mode != SessionMode.LEARNING:
                if _learning:
                    ctx.imitation.end_control_input()
                print("Not in learning mode.")
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

        if line in ("learn quit", "learn discard", "quit", "discard", "退出教学", "放弃"):
            if not ctx.session or ctx.session.mode != SessionMode.LEARNING:
                print("Not in learning mode.")
                continue
            try:
                await ctx.session.discard_learning()
                print("\nLearning discarded. No skill generated.")
            except Exception as e:
                print(f"Error: {e}")
            continue

        if line in ("learn save", "save", "保存"):
            if not ctx.session or ctx.session.mode != SessionMode.LEARNING:
                print("Not in learning mode.")
                continue
            try:
                learning = ctx.session.current_session
                await ctx.session.abandon_learning()
                print("\nLearning paused. Session saved for later resume.")
                if learning:
                    print(f"  Session ID: {learning.session_id}")
                    print(f"  Resume with: learn resume {learning.session_id}")
            except Exception as e:
                print(f"Error: {e}")
            continue

        # All remaining commands: end control input before processing
        if _learning:
            ctx.imitation.end_control_input()

        if line in ("learn pause", "pause", "暂停"):
            if ctx.session:
                ctx.session.pause_learning()
                print("Recording paused.")
            continue

        if line in ("learn resume", "resume", "继续"):
            if ctx.session and ctx.session.mode == SessionMode.LEARNING:
                ctx.session.resume_learning()
                print("Recording resumed.")
            else:
                print("Not in learning mode. Use 'learn resume <id>' to resume a saved session.")
            continue

        if line.startswith("learn resume "):
            resume_id = line[len("learn resume "):].strip()
            if not resume_id:
                print("Usage: learn resume <session_id>")
                continue
            if ctx.session and ctx.session.mode == SessionMode.LEARNING:
                print("Already in learning mode. Stop current session first.")
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
                    async for chunk in ctx.engine.run_stream(trigger_or_name):
                        print(chunk, end="", flush=True)
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
            continue

        matched = ctx.session.find_skill(line) if ctx.session else None
        if matched:
            result = await ctx.session.execute_skill(matched, io=io)
            _print_execution_result(result)
        else:
            async for chunk in ctx.engine.run_stream(line):
                print(chunk, end="", flush=True)
            print()

    return 0
