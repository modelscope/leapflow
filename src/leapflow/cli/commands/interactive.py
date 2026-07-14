"""Interactive subcommand — persistent REPL with hybrid Application TUI.

Uses ``LeapApp`` (prompt_toolkit Application + Rich) for a Hermes-style
fixed-input experience: status bar and input are pinned at the terminal
bottom while Rich-formatted output scrolls above.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from typing import TYPE_CHECKING, Any, Optional

from leapflow.cli.commands.run import _print_execution_result
from leapflow.cli.helpers import require_initialized
from leapflow.engine import StreamEvent

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from leapflow.cli.context import Context
    from leapflow.copilot.types import PredictionCandidate
    from leapflow.daemon.client import DaemonClient

_CLI_INTERACTION_EVENT = "cli.interaction"
_CLI_EVENT_SOURCE = "interactive_repl"

_last_hint: Optional["PredictionCandidate"] = None


def _is_app_command(canonical: str) -> bool:
    """Return true only for `/app` or `/app ...`, not `/apple`."""
    return canonical == "app" or canonical.startswith("app ")


async def _prompt_stop_daemon_on_exit(
    client: "DaemonClient",
    settings: Any,
    console: Any,
) -> None:
    """Ask whether to stop leapd after a daemon-backed TUI exits."""
    try:
        daemon_status = await client.status()
    except Exception:
        daemon_status = {}
    pid = daemon_status.get("pid") or "unknown"
    console.system(
        "leapd runs in the background; stop/restart it after reinstalling LeapFlow "
        "to load new code."
    )
    connected_clients = daemon_status.get("connected_clients")
    other_clients = 0
    try:
        other_clients = max(0, int(connected_clients or 0))
    except (TypeError, ValueError):
        other_clients = 0
    if other_clients > 0:
        console.system(
            f"Detected {other_clients} other Leap client(s); keeping leapd running by default."
        )
        stop = await _ask_yes_no_default_no(f"Stop leapd anyway (pid={pid})? [y/N]: ")
    else:
        stop = await _ask_yes_no_default_yes(f"Stop leapd now (pid={pid})? [Y/n]: ")
    if not stop:
        console.system("leapd kept running. Use `leap daemon restart` when needed.")
        return

    from leapflow.daemon.lifecycle import stop_daemon

    run_dir = settings.profile_dir / "run"
    console.system(f"Stopping leapd (pid={pid})...")
    graceful_requested = False
    try:
        await asyncio.wait_for(client.shutdown(), timeout=2.0)
        graceful_requested = True
    except Exception:
        graceful_requested = False
    result = await asyncio.to_thread(
        stop_daemon,
        run_dir,
        timeout_s=5.0,
        grace_timeout_s=1.0 if graceful_requested else 0.0,
    )
    if result.stopped:
        console.system("leapd stopped.")
    else:
        console.warning(
            f"leapd did not stop within the exit window (pid={result.pid}). "
            "Run `leap daemon stop --force` if it remains unhealthy."
        )


async def _ask_yes_no_default_yes(prompt: str) -> bool:
    """Return True by default, including non-interactive or interrupted prompts."""
    if not sys.stdin.isatty():
        return True
    try:
        answer = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: input(prompt).strip().lower(),
        )
    except (EOFError, KeyboardInterrupt):
        return True
    return answer not in {"n", "no"}


async def _ask_yes_no_default_no(prompt: str) -> bool:
    """Return False by default, including non-interactive or interrupted prompts."""
    if not sys.stdin.isatty():
        return False
    try:
        answer = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: input(prompt).strip().lower(),
        )
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in {"y", "yes"}


def _host_started(status: dict[str, Any]) -> bool:
    return bool(status.get("started")) and str(status.get("backend") or "") != "mock"


def _print_host_status(console: Any, host: dict[str, Any]) -> None:
    backend = str(host.get("backend") or "unknown")
    if _host_started(host):
        console.success("Host is on — CuaDriver OS control is connected.")
    else:
        console.system("Host is off — CuaDriver is not running for this session.")
    console.system(
        "Host/CuaDriver lets LeapFlow see and control the desktop: screenshots, UI automation, "
        "app/clipboard actions."
    )
    console.system(
        "Stopping it releases the background CuaDriver process; chat, memory, approvals, "
        "skills, and non-OS tools keep working."
    )
    tools = host.get("tools_count")
    extra = f" tools={tools}" if tools is not None else ""
    console.system(f"backend={backend} started={host.get('started')}{extra}")
    if host.get("last_error"):
        console.warning(f"host error: {host['last_error']}")


def _host_action(args: str) -> str:
    action = (args or "status").strip().lower()
    if action in {"on", "up", "enable"}:
        return "start"
    if action in {"off", "down", "disable"}:
        return "stop"
    return action


def _format_queue_elapsed(seconds: float) -> str:
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    return f"{minutes}m{seconds - minutes * 60:.0f}s"


class _DaemonRuntimeBridge:
    """Reconnectable bridge for daemon-backed TUI runtime calls."""

    def __init__(
        self,
        client: "DaemonClient",
        settings: Any,
        console: Any,
        *,
        session_id_getter: Any,
        session_id_setter: Any,
        metadata_applier: Any,
        lease: Any | None = None,
        mock_host: bool = False,
        client_factory: Any | None = None,
    ) -> None:
        self.client = client
        self._settings = settings
        self._console = console
        self._session_id_getter = session_id_getter
        self._session_id_setter = session_id_setter
        self._metadata_applier = metadata_applier
        self._lease = lease
        self._mock_host = mock_host
        self._client_factory = client_factory

    async def call(self, operation: Any, *, description: str) -> Any:
        """Run one RPC operation and recover the daemon once on connection failure."""
        from leapflow.daemon.client import DaemonUnavailableError

        try:
            return await operation(self.client)
        except DaemonUnavailableError as exc:
            await self.recover(f"{description} failed: {exc}")
            return await operation(self.client)

    async def recover(self, reason: str) -> None:
        """Reconnect or restart leapd, then restore runtime metadata and session."""
        from leapflow.daemon.client import recover_daemon_client

        self._console.warning(f"Lost connection to leapd; attempting recovery. {reason}")

        def _status(message: str) -> None:
            self._console.system(message)

        factory = self._client_factory or recover_daemon_client
        self.client = await factory(
            self._settings,
            mock_host=self._mock_host,
            status_callback=_status,
        )
        status = await self.client.status()
        self._metadata_applier(status)
        session_id = str(self._session_id_getter() or "")
        if session_id:
            result = await self.client.session_resume(session_id)
            if result.get("found"):
                restored = str(result.get("session_id") or session_id)
                self._session_id_setter(restored)
                self._console.success(f"Reconnected to leapd and resumed session {restored}")
            else:
                self._console.warning(
                    f"Reconnected to leapd, but session '{session_id}' was not found."
                )
        else:
            self._console.success("Reconnected to leapd.")
        if self._lease is not None:
            await self._lease.touch(state="idle", session_id=str(self._session_id_getter() or ""))


async def cmd_interactive(ctx: "Context", *, resume_id: Optional[str] = None) -> int:
    """Persistent REPL session with hybrid TUI (Application + Rich)."""
    require_initialized(ctx)

    from leapflow.cli.tui_app import (
        LeapApp,
        LeapConsole,
        SessionExitStats,
        StreamRenderer,
        build_exit_summary_lines,
        detect_theme,
        summarize_messages,
    )
    from leapflow.cli.tui_app.status import StatusBar
    from leapflow.cli.banner import display_rich_banner
    from leapflow.cli.commands.registry import completion_entries
    from leapflow.cli.commands.router import CommandRouter, render_command_result
    from leapflow.cli.commands.slash_handlers import (
        handle_status,
        handle_tools,
        handle_usage,
        handle_model,
        handle_clear,
        handle_gateway,
        handle_app,
    )
    from leapflow.utils.terminal_io import TerminalIOProvider
    from leapflow.engine.session import SessionMode
    from leapflow.tools.registry_bootstrap import TOOL_DEFINITIONS

    theme = detect_theme()
    console = LeapConsole(theme)
    status = StatusBar(theme)
    io = TerminalIOProvider()
    exit_stats = SessionExitStats()
    command_router = CommandRouter("in_process")
    active_resume_id = ""
    storage_volatile = bool(getattr(ctx, "storage_volatile", False))

    # ── Session callbacks ────────────────────────────────────────────

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

    # ── Helpers ──────────────────────────────────────────────────────

    def _mode_name() -> str:
        if ctx.session and ctx.session.mode == SessionMode.LEARNING:
            if ctx.imitation and ctx.imitation.recorder.state.name == "PAUSED":
                return "paused"
            return "learning"
        elif ctx.session and ctx.session.mode == SessionMode.EXECUTING:
            return "executing"
        return "idle"

    def _skill_count() -> int:
        index_count = (
            len(ctx.skill_index.get_entries())
            if hasattr(ctx, "skill_index") and ctx.skill_index
            else 0
        )
        registry_count = len(ctx.registry.list_all()) if ctx.registry else 0
        return index_count + registry_count

    def _platform_online() -> bool:
        return hasattr(ctx.rpc, "connected") and ctx.rpc.connected

    def _update_status() -> None:
        ctx_used = 0
        ctx_max = ctx.settings.llm_context_length
        ctx_state = "baseline"
        engine = ctx.engine
        if engine is not None:
            ctx_used = getattr(engine, "context_token_count", 0)
            snapshot = getattr(engine, "context_budget_snapshot", {})
            if callable(snapshot):
                snapshot = snapshot()
            if isinstance(snapshot, dict):
                ctx_state = str(snapshot.get("context_posture") or "baseline")
        mode = _mode_name()
        status.update(
            mode=mode,
            skill_count=_skill_count(),
            platform_online=_platform_online(),
            model_name=ctx.settings.llm_model,
            session_turns=getattr(engine, "turn_count", 0) if engine else 0,
            context_used=ctx_used,
            context_max=ctx_max,
            context_state=ctx_state,
        )
        app.prompt_mode = mode

    # ── Banner ───────────────────────────────────────────────────────

    all_skills = ctx.registry.list_all() if ctx.registry else []
    mcp_count = (
        len(getattr(ctx, "platform_tools", []))
        if hasattr(ctx, "platform_tools")
        else 0
    )
    ctx_len = ctx.settings.llm_context_length

    def _gateway_connected_names() -> list[str]:
        gw = getattr(ctx, "gateway_server", None)
        if gw is None:
            return []
        return [
            (gw.manifests[s.platform_id].display_name
             if s.platform_id in gw.manifests
             else s.platform_id)
            for s in gw.platform_status()
            if s.connected
        ]

    def _render_banner() -> None:
        display_rich_banner(
            model=ctx.settings.llm_model,
            cwd=os.getcwd(),
            session_id=getattr(ctx.session, "session_id", ""),
            platform_online=_platform_online(),
            tool_defs=TOOL_DEFINITIONS,
            skills=all_skills,
            context_length=ctx_len,
            mcp_tools=mcp_count,
            gateway_connected=_gateway_connected_names(),
            theme=theme,
        )
        if storage_volatile:
            console.warning(
                "Primary database is locked by another LeapFlow instance; "
                "this window is using volatile storage."
            )
            console.system("New memory, session history, and learned skills will not persist here.")

    def _active_chat_session_id() -> str:
        engine = getattr(ctx, "engine", None)
        current = getattr(engine, "_current_session_id", "") if engine else ""
        return str(current or active_resume_id or "")

    def _stored_message_counts(session_id: str) -> tuple[int, int, int] | None:
        if not session_id:
            return None
        store = getattr(ctx, "_conversation_store", None)
        if store is None:
            engine = getattr(ctx, "engine", None)
            store = getattr(engine, "_conversation_store", None) if engine else None
        if store is None:
            return None
        try:
            messages = store.get_messages(session_id, limit=10_000)
        except Exception:
            logger.debug("session summary message lookup failed", exc_info=True)
            return None
        return summarize_messages(messages)

    def _print_exit_summary() -> None:
        session_id = _active_chat_session_id()
        counts = _stored_message_counts(session_id)
        if counts is None:
            message_count = exit_stats.message_count
            user_messages = exit_stats.user_messages
            tool_calls = exit_stats.tool_calls
        else:
            message_count, user_messages, stored_tool_calls = counts
            tool_calls = max(stored_tool_calls, exit_stats.tool_calls)
        if not session_id and message_count == 0:
            return
        console.newline()
        for line in build_exit_summary_lines(
            session_id=session_id,
            duration_s=exit_stats.duration_s,
            message_count=message_count,
            user_messages=user_messages,
            tool_calls=tool_calls,
            resumable=not storage_volatile,
        ):
            console.print(line)

    # ── Stream response ──────────────────────────────────────────────

    async def _stream_response(prompt_text: str) -> None:
        exit_stats.record_user_message()
        status.mark_turn_start()
        app.agent_running = True
        app.spinner_text = "Thinking…"

        renderer = StreamRenderer(console)
        renderer.start()
        turn_completed = False
        try:
            async for event in ctx.engine.run_stream(prompt_text):
                if isinstance(event, StreamEvent):
                    if event.type == "chunk":
                        renderer.feed(event.content)
                    elif event.type == "thinking":
                        renderer.feed_thinking(event.content)
                    elif event.type == "tool_start":
                        app.spinner_text = renderer.tool_started(
                            event.content,
                            metadata=event.metadata or {},
                        )
                    elif event.type == "tool_complete":
                        renderer.tool_finished(event.content, metadata=event.metadata or {})
                        app.spinner_text = "Thinking…"
                    elif event.type == "final":
                        if not renderer.text:
                            renderer.feed(event.content)
                else:
                    renderer.feed(str(event))
            turn_completed = True
        finally:
            command = app.complete_active_command_in_response() if turn_completed else None
            renderer.finish(command=command)
            if renderer.has_output:
                exit_stats.record_assistant_message()
            exit_stats.record_tool_calls(renderer.tool_count)
            app.spinner_text = ""
            app.agent_running = False
            status.mark_turn_end()
            _update_status()

    # ── Input handler (business logic) ───────────────────────────────

    async def handle_input(text: str) -> None:
        """Dispatch one user input — slash commands or natural language."""
        global _last_hint

        try:
            if ctx.reload_runtime_config_if_changed():
                console.success(
                    "Configuration reloaded — LLM settings updated for this session."
                )
        except Exception as exc:
            logger.warning("Runtime config reload failed: %s", exc)
            console.warning(f"Configuration reload failed: {exc}")

        _learning = ctx.session and ctx.session.mode == SessionMode.LEARNING
        if _learning:
            ctx.imitation.end_control_input()

        _update_status()

        # Copilot feedback on previous hint
        if ctx.copilot_idle is not None:
            ctx.copilot_idle.on_event_timestamp(time.time())

        if ctx.copilot_feedback is not None and _last_hint is not None:
            if _is_hint_accepted(text, _last_hint):
                signal = ctx.copilot_feedback.on_accept()
            elif ctx.copilot_encoder is not None:
                signal = ctx.copilot_feedback.on_next_action(
                    text, ctx.copilot_encoder.current_state
                )
            else:
                signal = None
            if signal and ctx.copilot_evolution:
                await ctx.copilot_evolution.process_feedback(signal)
            _last_hint = None

        if _learning:
            ctx.imitation.mark_control_input()

        console.rule()

        # ── Slash command dispatch ──
        invocation = command_router.parse(text)
        if invocation is not None:
            unsupported = command_router.unsupported_result(invocation)
            if unsupported is not None:
                render_command_result(console, unsupported)
                return
            cmd_text = invocation.text
            cmd_def = invocation.command
            canonical = cmd_def.name
            cmd_args = invocation.args

            if canonical == "exit":
                if ctx.session and ctx.session.mode == SessionMode.LEARNING:
                    try:
                        await ctx.session.exit_learning()
                        console.system("Learning stopped.")
                    except Exception:
                        pass
                elif _learning:
                    ctx.imitation.end_control_input()
                app.exit()
                return

            if canonical == "help":
                _show_help(console, runtime="in_process")
                return

            if canonical == "status":
                handle_status(ctx, console, cmd_args)
                return

            if canonical == "host":
                action = _host_action(cmd_args)
                if action == "status":
                    result = await ctx.host_backend_status()
                elif action == "start":
                    console.system("Starting CuaDriver OS control for this session…")
                    result = await ctx.host_backend_start()
                elif action == "stop":
                    console.system("Stopping CuaDriver OS control; chat and memory stay available…")
                    result = await ctx.host_backend_stop()
                elif action == "restart":
                    console.system("Restarting CuaDriver OS control…")
                    result = await ctx.host_backend_restart()
                else:
                    console.warning("Usage: /host [status|start|stop|restart]")
                    return
                _print_host_status(console, result)
                _update_status()
                return

            if canonical == "clear":
                handle_clear(ctx, console, cmd_args)
                _render_banner()
                return

            if canonical == "tools":
                handle_tools(ctx, console, cmd_args)
                return

            if canonical == "gateway":
                handle_gateway(ctx, console, cmd_args)
                return

            if _is_app_command(canonical):
                app_args = cmd_text[len("app"):].strip()
                await handle_app(ctx, console, app_args)
                _update_status()
                return

            if canonical == "usage":
                handle_usage(ctx, console, cmd_args)
                return

            if canonical == "model":
                handle_model(ctx, console, cmd_args)
                return

            if canonical.startswith("teach") or canonical == "annotate":
                if await _handle_teach(ctx, console, cmd_text, _learning):
                    await _after_dispatch(text)
                    return

            if canonical.startswith("skills"):
                if _handle_skills(ctx, console, cmd_text):
                    return

            if canonical.startswith("hub"):
                from leapflow.cli.commands.hub import cmd_hub

                hub_args = (
                    cmd_text.split()[1:] if len(cmd_text.split()) > 1 else []
                )
                await cmd_hub(ctx, hub_args)
                return

            if canonical == "run":
                trigger_or_name = cmd_args
                if trigger_or_name.startswith("--skill "):
                    skill_name = trigger_or_name[len("--skill "):]
                    result = await ctx.session.execute_skill(skill_name, io=io)
                else:
                    matched = ctx.session.find_skill(trigger_or_name)
                    if matched:
                        result = await ctx.session.execute_skill(
                            matched, io=io
                        )
                    else:
                        await _stream_response(trigger_or_name)
                        await _after_dispatch(text)
                        return
                _print_execution_result(result)
                return

            if canonical == "arm":
                from leapflow.cli.commands.scheduler import cmd_arm

                await cmd_arm(
                    ctx,
                    cmd_text.split()[1:] if len(cmd_text.split()) > 1 else [],
                )
                return

            if canonical == "tasks":
                from leapflow.cli.commands.scheduler import cmd_tasks

                await cmd_tasks(
                    ctx,
                    cmd_text.split()[1:] if len(cmd_text.split()) > 1 else [],
                )
                return

        if _learning:
            ctx.imitation.end_control_input()

        # ── Natural language input ──
        if ctx.session and ctx.session.mode == SessionMode.LEARNING:
            ctx.session.annotate(text)
            console.system("(Noted as annotation during learning)")
            await _after_dispatch(text)
            return

        matched = ctx.session.find_skill(text) if ctx.session else None
        if matched:
            result = await ctx.session.execute_skill(matched, io=io)
            _print_execution_result(result)
        else:
            await _stream_response(text)

        await _after_dispatch(text)

    async def _after_dispatch(text: str) -> None:
        """Post-dispatch: inject copilot event and display ghost hint."""
        global _last_hint

        await _inject_copilot_event(ctx, text, _mode_name)

        _learning = ctx.session and ctx.session.mode == SessionMode.LEARNING
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
                if (
                    ctx.copilot_feedback is not None
                    and ctx.copilot_encoder is not None
                ):
                    ctx.copilot_feedback.track_shown(
                        best, ctx.copilot_encoder.current_state
                    )
            else:
                _last_hint = None
        else:
            _last_hint = None

    def _render_queue_state() -> None:
        active = app.active_command
        queued = app.queued_commands()
        if active is None and not queued:
            console.system("Queue is empty.")
            return
        if active is not None:
            console.system(f"{active.label} {active.status.value} {_format_queue_elapsed(active.elapsed_s)}  {active.summary()}")
        for command in queued:
            console.system(f"{command.label} {command.status.value}  {command.summary()}")
        if app.queue_paused:
            console.system("Queue is paused — use /resume to continue.")

    def _handle_task_control(text: str) -> bool:
        invocation = command_router.parse(text)
        if invocation is None:
            return False
        canonical = invocation.command.name
        args = invocation.args.strip()
        if canonical not in {"cancel", "skip", "pause", "resume", "queue", "drop"}:
            return False
        if canonical == "cancel":
            cancelled = app.request_cancel_active("cancelled by user")
            if cancelled is None:
                console.system("No running task to cancel.")
            else:
                engine = getattr(ctx, "engine", None)
                if engine is not None and hasattr(engine, "cancel"):
                    engine.cancel()
                console.warning(f"Cancelled {cancelled.label}. Continuing queued work.")
            return True
        if canonical == "skip":
            skipped = app.request_skip_active("skipped by user")
            if skipped is None:
                console.system("No running task to skip.")
            else:
                engine = getattr(ctx, "engine", None)
                if engine is not None and hasattr(engine, "cancel"):
                    engine.cancel()
                console.warning(f"Skipped {skipped.label}. Continuing queued work.")
            return True
        if canonical == "pause":
            changed = app.pause_queue()
            console.system("Queue paused. Current running task continues; new queued tasks will wait." if changed else "Queue is already paused.")
            return True
        if canonical == "resume":
            changed = app.resume_queue()
            console.system("Queue resumed." if changed else "Queue is not paused.")
            return True
        if canonical == "queue":
            if args.lower() == "clear":
                dropped = app.clear_queued_commands("cleared by user")
                console.system(f"Cleared {len(dropped)} queued task(s).")
            else:
                _render_queue_state()
            return True
        if canonical == "drop":
            try:
                command_id = int(args)
            except ValueError:
                console.warning("Usage: /drop <queued_task_id>")
                return True
            dropped = app.drop_queued_command(command_id, "dropped by user")
            if dropped is None:
                console.warning(f"No queued task #{command_id}.")
            else:
                console.system(f"Dropped {dropped.label}.")
            return True
        return False

    # ── Create and run the Application ───────────────────────────────

    app = LeapApp(
        console=console,
        theme=theme,
        status=status,
        commands=completion_entries(),
        data_dir=(
            ctx.settings.data_dir
            if hasattr(ctx.settings, "data_dir")
            else None
        ),
        on_input=handle_input,
        on_control=_handle_task_control,
    )
    ctx.set_approval_handler(app.request_approval)

    # Auto-connect previously configured gateway platforms
    gw = getattr(ctx, "gateway_server", None)
    if gw is not None:
        try:
            gw_count = await gw.start()
            if gw_count > 0:
                console.system(f"  Gateway: {gw_count} platform(s) reconnected")
        except Exception:
            logger.debug("Gateway auto-connect failed", exc_info=True)

    if resume_id and ctx.engine is not None:
        if ctx.engine.load_session(resume_id):
            active_resume_id = resume_id
            console.success(f"Resumed session {resume_id}")
        else:
            console.warning(f"Session '{resume_id}' not found; starting a new session.")

    _render_banner()
    _update_status()
    exit_code = 0
    try:
        exit_code = await app.run()
    finally:
        ctx.set_approval_handler(None)
        _print_exit_summary()
    return exit_code


async def cmd_interactive_daemon(
    client: "DaemonClient",
    settings: Any,
    *,
    resume_id: Optional[str] = None,
    mock_host: bool = False,
) -> int:
    """Persistent REPL backed by leapd thin-client RPC."""
    from leapflow.cli.banner import display_rich_banner
    from leapflow.cli.commands.registry import completion_entries
    from leapflow.cli.commands.router import CommandRouter, render_command_result
    from leapflow.cli.commands.slash_handlers import (
        render_app_payload,
        render_model_payload,
        render_tools_payload,
        render_usage_payload,
    )
    from leapflow.cli.tui_app import (
        LeapApp,
        LeapConsole,
        SessionExitStats,
        StreamRenderer,
        build_exit_summary_lines,
        detect_theme,
    )
    from leapflow.cli.tui_app.status import StatusBar
    from leapflow.daemon.lease import ClientLease
    from leapflow.tools.registry_bootstrap import TOOL_DEFINITIONS

    theme = detect_theme()
    console = LeapConsole(theme)
    status = StatusBar(theme)
    exit_stats = SessionExitStats()
    command_router = CommandRouter("daemon")
    active_session_id = str(resume_id or "")
    turn_count = 0
    runtime_model_name = str(getattr(settings, "llm_model", ""))
    runtime_context_length = int(getattr(settings, "llm_context_length", 0) or 0)
    runtime_context_used = 0
    runtime_context_state = "baseline"
    runtime_daemon_pid = ""
    runtime_host_online = False
    client_lease = ClientLease(
        settings.profile_dir / "run",
        kind="tui",
        session_id=active_session_id,
    )

    def _apply_daemon_runtime_metadata(metadata: dict[str, Any]) -> None:
        nonlocal active_session_id, runtime_model_name, runtime_context_length
        nonlocal runtime_context_used, runtime_context_state, runtime_daemon_pid, runtime_host_online
        if metadata.get("pid"):
            runtime_daemon_pid = str(metadata["pid"])
        if metadata.get("session_id"):
            active_session_id = str(metadata["session_id"])
        if metadata.get("model"):
            runtime_model_name = str(metadata["model"])
        if metadata.get("llm_model"):
            runtime_model_name = str(metadata["llm_model"])
        if metadata.get("llm_context_length") is not None:
            try:
                runtime_context_length = max(1, int(metadata["llm_context_length"]))
            except (TypeError, ValueError):
                pass
        if metadata.get("context_used") is not None:
            try:
                runtime_context_used = max(0, int(metadata["context_used"]))
            except (TypeError, ValueError):
                pass
        if metadata.get("context_posture"):
            runtime_context_state = str(metadata["context_posture"])
        elif isinstance(metadata.get("context_budget_snapshot"), dict):
            snapshot = metadata["context_budget_snapshot"]
            runtime_context_state = str(snapshot.get("context_posture") or runtime_context_state)
        host = metadata.get("host_backend")
        if isinstance(host, dict):
            runtime_host_online = _host_started(host)

    def _set_active_session_id(session_id: str) -> None:
        nonlocal active_session_id
        active_session_id = session_id

    bridge = _DaemonRuntimeBridge(
        client,
        settings,
        console,
        session_id_getter=lambda: active_session_id,
        session_id_setter=_set_active_session_id,
        metadata_applier=_apply_daemon_runtime_metadata,
        lease=client_lease,
        mock_host=mock_host,
    )

    def _update_status() -> None:
        status.update(
            mode="daemon",
            skill_count=0,
            platform_online=runtime_host_online,
            model_name=runtime_model_name,
            session_turns=turn_count,
            context_used=runtime_context_used,
            context_max=runtime_context_length,
            context_state=runtime_context_state,
        )
        app.prompt_mode = "daemon"

    def _render_banner() -> None:
        display_rich_banner(
            model=runtime_model_name,
            cwd=os.getcwd(),
            session_id=active_session_id,
            platform_online=runtime_host_online,
            tool_defs=TOOL_DEFINITIONS,
            skills=[],
            context_length=runtime_context_length,
            mcp_tools=0,
            gateway_connected=[],
            theme=theme,
        )
        daemon_suffix = f" pid={runtime_daemon_pid}" if runtime_daemon_pid else ""
        console.system(f"Daemon mode{daemon_suffix}: shared runtime across terminals.")
        console.system("After reinstalling LeapFlow, use `leap daemon restart` to load new code.")

    async def _print_daemon_status() -> None:
        try:
            daemon_status = await bridge.call(
                lambda current_client: current_client.status(),
                description="daemon status",
            )
        except Exception as exc:
            console.warning(f"Daemon status unavailable: {exc}")
            return
        _apply_daemon_runtime_metadata(daemon_status)
        _update_status()
        console.system(
            "leapd "
            f"pid={daemon_status.get('pid')} "
            f"profile={daemon_status.get('profile')} "
            f"clients={daemon_status.get('active_clients')} "
            f"connected={daemon_status.get('connected_clients', 0)} "
            f"volatile={daemon_status.get('volatile')}"
        )
        db_path = daemon_status.get("db_path")
        if db_path:
            console.system(f"DB: {db_path}")
        config_path = daemon_status.get("config_path")
        if config_path:
            console.system(f"Config: {config_path}")
        project_env_path = daemon_status.get("project_env_path")
        if project_env_path:
            console.system(f"Project override: {project_env_path}")
        runtime_version = daemon_status.get("runtime_version")
        if runtime_version:
            console.system(f"Runtime version: {runtime_version}")
        runtime_source = daemon_status.get("runtime_source")
        if runtime_source:
            console.system(f"Runtime source: {runtime_source}")
        runtime_executable = daemon_status.get("runtime_executable")
        if runtime_executable:
            console.system(f"Python: {runtime_executable}")
        context_length = daemon_status.get("llm_context_length")
        if context_length:
            console.system(f"Context budget: {int(context_length):,} tokens")
        context_used = daemon_status.get("context_used")
        if context_used is not None:
            console.system(f"Context used: {int(context_used):,} tokens")
        host = daemon_status.get("host_backend")
        if isinstance(host, dict):
            _print_host_status(console, host)

    async def _handle_daemon_approval(event: StreamEvent) -> None:
        from leapflow.security.approval import ApprovalDecision, ApprovalRequest

        metadata = event.metadata or {}
        payload = metadata.get("approval")
        if not isinstance(payload, dict):
            return
        pending_id = str(payload.get("pending_id") or "")
        if not pending_id:
            return
        app.spinner_text = "Waiting for approval…"
        request = ApprovalRequest.from_dict(payload)
        decision = await app.request_approval(request)
        value = decision.value if isinstance(decision, ApprovalDecision) else str(decision)
        await bridge.call(
            lambda current_client: current_client.approval_resolve(pending_id, value),
            description="approval resolve",
        )
        app.spinner_text = "Thinking…"

    async def _stream_response(
        prompt_text: str,
        *,
        allow_retry: bool = True,
        record_user: bool = True,
    ) -> None:
        nonlocal turn_count
        from leapflow.daemon.client import DaemonUnavailableError

        if record_user:
            exit_stats.record_user_message()
        status.mark_turn_start()
        app.agent_running = True
        app.spinner_text = "Thinking…"
        await client_lease.touch(state="streaming", session_id=active_session_id)

        renderer = StreamRenderer(console)
        renderer.start()
        retry_error: DaemonUnavailableError | None = None
        saw_real_event = False
        turn_completed = False
        try:
            async for event in bridge.client.engine_chat(prompt_text):
                metadata = event.metadata or {}
                is_heartbeat = event.type == "status" and metadata.get("heartbeat")
                if not is_heartbeat:
                    saw_real_event = True
                _apply_daemon_runtime_metadata(metadata)
                if event.type == "chunk":
                    renderer.feed(event.content)
                elif event.type == "thinking":
                    renderer.feed_thinking(event.content)
                elif event.type == "tool_start":
                    app.spinner_text = renderer.tool_started(
                        event.content,
                        metadata=metadata,
                    )
                elif event.type == "tool_complete":
                    renderer.tool_finished(event.content, metadata=metadata)
                    app.spinner_text = "Thinking…"
                elif event.type == "final":
                    if not renderer.text:
                        renderer.feed(event.content)
                elif event.type == "error":
                    renderer.feed(event.content)
                elif event.type == "approval_request":
                    await client_lease.touch(state="approval", session_id=active_session_id)
                    await _handle_daemon_approval(event)
                    await client_lease.touch(state="streaming", session_id=active_session_id)
                elif event.type == "status":
                    if not metadata.get("heartbeat"):
                        console.system(event.content)
            turn_completed = True
        except DaemonUnavailableError as exc:
            if allow_retry and not saw_real_event:
                retry_error = exc
            else:
                console.warning(
                    "Lost connection to leapd after the turn started; "
                    "the command was not replayed to avoid duplicate side effects."
                )
                raise
        finally:
            command = app.complete_active_command_in_response() if turn_completed else None
            renderer.finish(command=command)
            if renderer.has_output:
                exit_stats.record_assistant_message()
            exit_stats.record_tool_calls(renderer.tool_count)
            if turn_completed:
                turn_count += 1
            app.spinner_text = ""
            app.agent_running = False
            status.mark_turn_end()
            await client_lease.touch(state="idle", session_id=active_session_id)
            _update_status()

        if retry_error is not None:
            await bridge.recover(f"chat stream failed before output: {retry_error}")
            console.system("Retrying the interrupted request once after reconnecting to leapd.")
            await _stream_response(prompt_text, allow_retry=False, record_user=False)

    async def handle_input(text: str) -> None:
        invocation = command_router.parse(text)
        if invocation is not None:
            unsupported = command_router.unsupported_result(invocation)
            if unsupported is not None:
                render_command_result(console, unsupported)
                return
            canonical = invocation.command.name
            cmd_args = invocation.args
            if canonical == "exit":
                app.exit()
                return
            if canonical == "help":
                _show_help(console, runtime="daemon")
                return
            if canonical == "status":
                await _print_daemon_status()
                return
            if canonical == "host":
                action = _host_action(cmd_args)
                try:
                    if action == "status":
                        result = await bridge.call(
                            lambda current_client: current_client.host_status(),
                            description="host status",
                        )
                    elif action == "start":
                        console.system("Starting CuaDriver OS control for this session…")
                        result = await bridge.call(
                            lambda current_client: current_client.host_start(),
                            description="host start",
                        )
                    elif action == "stop":
                        console.system("Stopping CuaDriver OS control; chat and memory stay available…")
                        result = await bridge.call(
                            lambda current_client: current_client.host_stop(),
                            description="host stop",
                        )
                    elif action == "restart":
                        console.system("Restarting CuaDriver OS control…")
                        result = await bridge.call(
                            lambda current_client: current_client.host_restart(),
                            description="host restart",
                        )
                    else:
                        console.warning("Usage: /host [status|start|stop|restart]")
                        return
                except Exception as exc:
                    console.warning(f"Host control failed: {exc}")
                    return
                _apply_daemon_runtime_metadata({"host_backend": result})
                _print_host_status(console, result)
                _update_status()
                return
            if canonical == "clear":
                _render_banner()
                return
            if canonical == "tools":
                try:
                    payload = await bridge.call(
                        lambda current_client: current_client.tools_list(),
                        description="tools list",
                    )
                except Exception as exc:
                    console.warning(f"Tools unavailable: {exc}")
                    return
                render_tools_payload(console, payload)
                return
            if canonical == "usage":
                try:
                    payload = await bridge.call(
                        lambda current_client: current_client.usage_summary(),
                        description="usage summary",
                    )
                except Exception as exc:
                    console.warning(f"Usage unavailable: {exc}")
                    return
                render_usage_payload(console, payload)
                return
            if canonical == "model":
                try:
                    payload = await bridge.call(
                        lambda current_client: current_client.model_info(cmd_args),
                        description="model info",
                    )
                except Exception as exc:
                    console.warning(f"Model info unavailable: {exc}")
                    return
                render_model_payload(console, payload)
                return
            if _is_app_command(canonical):
                app_args = invocation.text[len("app"):].strip()
                try:
                    payload = await bridge.call(
                        lambda current_client: current_client.app_command(app_args),
                        description="app command",
                    )
                except Exception as exc:
                    console.warning(f"App Connector unavailable: {exc}")
                    return
                render_app_payload(console, payload)
                return
            if canonical == "run":
                if not cmd_args:
                    console.warning("Usage: /run <trigger>")
                    return
                console.system("Running through daemon chat stream; approvals and host state stay visible.")
                await _stream_response(cmd_args)
                return
            console.warning(
                f"/{canonical} is not available in daemon mode yet. "
                "Use --no-daemon for legacy in-process commands."
            )
            return

        console.rule()
        await _stream_response(text)

    def _print_exit_summary() -> None:
        if not active_session_id and exit_stats.message_count == 0:
            return
        console.newline()
        for line in build_exit_summary_lines(
            session_id=active_session_id,
            duration_s=exit_stats.duration_s,
            message_count=exit_stats.message_count,
            user_messages=exit_stats.user_messages,
            tool_calls=exit_stats.tool_calls,
            resumable=True,
        ):
            console.print(line)

    def _render_queue_state() -> None:
        active = app.active_command
        queued = app.queued_commands()
        if active is None and not queued:
            console.system("Queue is empty.")
            return
        if active is not None:
            console.system(f"{active.label} {active.status.value} {_format_queue_elapsed(active.elapsed_s)}  {active.summary()}")
        for command in queued:
            console.system(f"{command.label} {command.status.value}  {command.summary()}")
        if app.queue_paused:
            console.system("Queue is paused — use /resume to continue.")

    def _request_daemon_cancel() -> None:
        async def cancel_remote() -> None:
            try:
                await bridge.call(
                    lambda current_client: current_client.engine_cancel(),
                    description="engine cancel",
                )
            except Exception as exc:
                console.warning(f"Daemon cancel failed: {exc}")
        asyncio.create_task(cancel_remote())

    def _handle_task_control(text: str) -> bool:
        invocation = command_router.parse(text)
        if invocation is None:
            return False
        canonical = invocation.command.name
        args = invocation.args.strip()
        if canonical not in {"cancel", "skip", "pause", "resume", "queue", "drop"}:
            return False
        if canonical == "cancel":
            cancelled = app.request_cancel_active("cancelled by user")
            if cancelled is None:
                console.system("No running task to cancel.")
            else:
                _request_daemon_cancel()
                console.warning(f"Cancelled {cancelled.label}. Continuing queued work.")
            return True
        if canonical == "skip":
            skipped = app.request_skip_active("skipped by user")
            if skipped is None:
                console.system("No running task to skip.")
            else:
                _request_daemon_cancel()
                console.warning(f"Skipped {skipped.label}. Continuing queued work.")
            return True
        if canonical == "pause":
            changed = app.pause_queue()
            console.system("Queue paused. Current running task continues; new queued tasks will wait." if changed else "Queue is already paused.")
            return True
        if canonical == "resume":
            changed = app.resume_queue()
            console.system("Queue resumed." if changed else "Queue is not paused.")
            return True
        if canonical == "queue":
            if args.lower() == "clear":
                dropped = app.clear_queued_commands("cleared by user")
                console.system(f"Cleared {len(dropped)} queued task(s).")
            else:
                _render_queue_state()
            return True
        if canonical == "drop":
            try:
                command_id = int(args)
            except ValueError:
                console.warning("Usage: /drop <queued_task_id>")
                return True
            dropped = app.drop_queued_command(command_id, "dropped by user")
            if dropped is None:
                console.warning(f"No queued task #{command_id}.")
            else:
                console.system(f"Dropped {dropped.label}.")
            return True
        return False

    app = LeapApp(
        console=console,
        theme=theme,
        status=status,
        commands=completion_entries(),
        data_dir=getattr(settings, "data_dir", None),
        on_input=handle_input,
        on_control=_handle_task_control,
    )

    if resume_id:
        result = await bridge.call(
            lambda current_client: current_client.session_resume(resume_id),
            description="session resume",
        )
        if result.get("found"):
            active_session_id = str(result.get("session_id") or resume_id)
            console.success(f"Resumed session {active_session_id}")
        else:
            console.warning(f"Session '{resume_id}' not found; starting a new session.")
            active_session_id = ""

    try:
        _apply_daemon_runtime_metadata(await bridge.call(
            lambda current_client: current_client.status(),
            description="daemon status",
        ))
    except Exception as exc:
        console.warning(f"Daemon status unavailable: {exc}")

    _render_banner()
    _update_status()
    exit_code = 0
    try:
        await client_lease.start()
        exit_code = await app.run()
    finally:
        await client_lease.stop()
        _print_exit_summary()
        await _prompt_stop_daemon_on_exit(bridge.client, settings, console)
    return exit_code


# ── Command handlers ─────────────────────────────────────────────────


async def _handle_teach(
    ctx: "Context", console, line: str, learning: bool
) -> bool:
    """Handle teach/learn commands. Returns True if handled."""
    from leapflow.engine.session import SessionMode

    if (
        line.startswith("teach start")
        or line.startswith("教学开始")
        or line == "teach"
    ):
        if ctx.session and ctx.session.mode == SessionMode.LEARNING:
            console.warning("Already in teaching mode. Say '/teach stop' to end.")
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
            console.system(
                "Commands: /teach stop │ /teach discard │ /teach pause │ "
                "/teach resume │ /teach skip [n] │ /annotate <text>"
            )
        except Exception as e:
            console.error(str(e))
        return True

    if line == "teach stop":
        if not ctx.session or ctx.session.mode != SessionMode.LEARNING:
            if learning:
                ctx.imitation.end_control_input()
            console.warning("Not in teaching mode.")
            return True
        try:
            console.system("Stopping recording…")
            result = await ctx.session.exit_learning()
            console.success(
                f"Recording stopped — {result.step_count} steps, "
                f"{result.duration:.1f}s"
            )

            report = getattr(result, "learnability_report", None)
            if report:
                from leapflow.learning.learnability import LearnabilityDecision

                if report.decision == LearnabilityDecision.SKIP:
                    console.warning(
                        f"Not worth learning — {report.reason} "
                        f"(score: {report.score:.2f})"
                    )
                    return True
                elif report.decision == LearnabilityDecision.ASK:
                    console.system(
                        f"Uncertain (score: {report.score:.2f}) — {report.reason}"
                    )
                    answer = await asyncio.get_running_loop().run_in_executor(
                        None, lambda: input("  Learn this? [y/N]: ").strip().lower()
                    )
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
                    activated = (
                        set(final.activated_skill_names)
                        if final.activated_skill_names
                        else set()
                    )
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

    if line in ("teach quit", "teach discard"):
        if not ctx.session or ctx.session.mode != SessionMode.LEARNING:
            console.warning("Not in teaching mode.")
            return True
        try:
            await ctx.session.discard_learning()
            console.system("Teaching discarded. No skill generated.")
        except Exception as e:
            console.error(str(e))
        return True

    if line == "teach save":
        if not ctx.session or ctx.session.mode != SessionMode.LEARNING:
            console.warning("Not in teaching mode.")
            return True
        try:
            learning_session = ctx.session.current_session
            await ctx.session.abandon_learning()
            console.success("Session saved for later resume.")
            if learning_session:
                console.system(
                    f"  Resume with: /teach resume {learning_session.session_id}"
                )
        except Exception as e:
            console.error(str(e))
        return True

    if line == "teach pause":
        if ctx.session:
            ctx.session.pause_learning()
            console.system("Recording paused.")
        return True

    if line == "teach resume":
        from leapflow.engine.session import SessionMode as SM

        if ctx.session and ctx.session.mode == SM.LEARNING:
            ctx.session.resume_learning()
            console.system("Recording resumed.")
        else:
            console.warning("Not in teaching mode. Use '/teach resume <id>'.")
        return True

    if line.startswith("teach resume "):
        resume_id = line[len("teach resume "):].strip()
        if not resume_id:
            console.warning("Usage: /teach resume <session_id>")
            return True
        if ctx.session and ctx.session.mode == SessionMode.LEARNING:
            console.warning("Already in teaching mode. Stop first.")
            return True
        try:
            session = await ctx.session.resume_session(resume_id)
            traj = ctx.imitation.get_trajectory(session.trajectory_id)
            step_count = traj.step_count if traj else 0
            console.success(
                f"Resumed session {session.session_id} "
                f"({step_count} existing steps)"
            )
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

    if line.startswith("teach skip"):
        parts = line.split()
        n = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 1
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

            table = Table(
                show_header=True, header_style="bold", border_style="dim"
            )
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
            console.print(table)
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
            console.print(
                Panel(info, title=skill.name, border_style="cyan")
            )
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


def _show_help(console, runtime: str = "in_process") -> None:
    """Display categorized help using the command registry."""
    from leapflow.cli.commands.registry import commands_by_category

    categories = commands_by_category(runtime=runtime)  # type: ignore[arg-type]

    console.print()
    for category, cmds in categories.items():
        console.print(f"  [bold]── {category} ──[/]")
        for cmd in cmds:
            name = f"/{cmd.name}"
            if cmd.args_hint:
                name = f"{name} {cmd.args_hint}"
            aliases = ""
            if cmd.aliases:
                visible = [
                    a
                    for a in cmd.aliases
                    if not a.startswith("教") and a != "?" and a not in ("teach",)
                ]
                if visible:
                    aliases = f" [dim]({', '.join(visible)})[/]"
            support = ""
            if not cmd.supports_runtime(runtime):  # type: ignore[arg-type]
                support = " [dim]not in daemon[/]" if runtime == "daemon" else " [dim]not in this mode[/]"
            effect = ""
            if cmd.requires_host:
                effect = " [dim]host[/]"
            elif cmd.requires_llm:
                effect = " [dim]llm[/]"
            console.print(
                f"    [cyan]{name:<28}[/] {cmd.description}{aliases}{effect}{support}"
            )
        console.print()

    console.system(
        "Type your message to chat · Alt+Enter for multiline · Tab for completion"
    )


# ── Copilot helpers ──────────────────────────────────────────────────


def _render_ghost_hint(console, candidate: "PredictionCandidate") -> None:
    """Render a ghost hint in the console."""
    confidence_pct = int(candidate.confidence * 100)
    console.system(
        f"💡 {candidate.action_description} ({confidence_pct}% — Tab to accept)"
    )


def _is_hint_accepted(user_input: str, hint: "PredictionCandidate") -> bool:
    desc = hint.action_description.lower().strip()
    inp = user_input.lower().strip()
    return inp == desc or (
        desc and inp.startswith(desc.split()[0]) and len(inp) > 2
    )


async def _inject_copilot_event(
    ctx: "Context", line: str, mode_fn
) -> None:
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
