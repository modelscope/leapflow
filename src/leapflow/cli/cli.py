"""CLI entrypoint for LeapFlow.

Usage:
    leap              — Show banner, enter interactive REPL
    leap "question"   — Single-turn chat (one answer, then exit)
    leap teach        — Interactive teaching mode (record → distill)
    leap run          — Execute a skill by trigger match or explicit name
    leap skills       — List / inspect learned skills
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

try:
    import termios
except ImportError:  # pragma: no cover - non-POSIX platforms
    termios = None

try:
    import gnureadline as readline  # noqa: F401
except ImportError:
    import readline  # noqa: F401

from leapflow.cli.context import Context
from leapflow.config import load_config


async def _async_main(args: argparse.Namespace) -> int:
    settings = load_config()
    mock_host = getattr(args, "mock_host", False)
    sys.stderr.write("\033[2m→ Initializing LeapFlow...\033[0m\n")
    sys.stderr.flush()
    ctx = Context(settings, mock_host)
    await ctx.initialize()
    sys.stderr.write("\033[2m→ Ready\033[0m\n")
    sys.stderr.flush()

    try:
        cmd = args.command

        if cmd == "interactive":
            from leapflow.cli.commands.interactive import cmd_interactive
            return await cmd_interactive(ctx, resume_id=getattr(args, "resume", None))
        elif cmd == "chat":
            from leapflow.cli.commands.chat import cmd_chat
            return await cmd_chat(ctx, args.prompt, getattr(args, "thinking", False))
        elif cmd == "teach":
            resume_id = getattr(args, "resume", None)
            if resume_id:
                from leapflow.cli.commands.teach import cmd_teach_resume
                return await cmd_teach_resume(ctx, resume_id, getattr(args, "timeout", None))
            from leapflow.cli.commands.teach import cmd_teach
            return await cmd_teach(
                ctx, getattr(args, "goal", ""), getattr(args, "timeout", None),
                field_args=getattr(args, "field", []),
            )
        elif cmd == "run":
            from leapflow.cli.commands.run import cmd_run
            prompt = getattr(args, "prompt", "")
            return await cmd_run(
                ctx, prompt, getattr(args, "skill", None),
                step=getattr(args, "step", False),
                auto=getattr(args, "auto", False),
            )
        elif cmd == "relearn":
            from leapflow.cli.commands.relearn import cmd_relearn
            return await cmd_relearn(ctx, args.trajectory_id)
        elif cmd == "skills":
            from leapflow.cli.commands.skills import cmd_skills
            return await cmd_skills(
                ctx,
                getattr(args, "action", "list"),
                getattr(args, "name", None),
                getattr(args, "output", None),
                getattr(args, "limit", 20),
                include_suggestions=getattr(args, "include_suggestions", False),
            )
        else:
            print("Unknown command. Use 'leap --help' for usage.")
            return 1
    finally:
        await ctx.cleanup()


async def _async_host(args: argparse.Namespace) -> int:
    """Run host subcommand without Context initialization."""
    from leapflow.cli.commands.host import cmd_host
    return await cmd_host(args)


class _StdinEchoGuard:
    """Temporarily hide pre-TUI stdin echo while leapd starts."""

    def __init__(self) -> None:
        self._fd: int | None = None
        self._attrs: list | None = None

    def __enter__(self) -> "_StdinEchoGuard":
        if termios is None or not sys.stdin.isatty():
            return self
        try:
            self._fd = sys.stdin.fileno()
            self._attrs = termios.tcgetattr(self._fd)
            muted = list(self._attrs)
            muted[3] = muted[3] & ~termios.ECHO
            termios.tcsetattr(self._fd, termios.TCSADRAIN, muted)
        except termios.error:
            self._fd = None
            self._attrs = None
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if termios is None or self._fd is None or self._attrs is None:
            return
        try:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._attrs)
            termios.tcflush(self._fd, termios.TCIFLUSH)
        except termios.error:
            return


async def _async_daemon_main(args: argparse.Namespace) -> int:
    """Run chat/interactive through a shared leapd daemon."""
    from leapflow.daemon.client import DaemonUnavailableError, ensure_daemon_client

    settings = load_config()
    mock_host = getattr(args, "mock_host", False)

    def _status(message: str) -> None:
        sys.stderr.write(f"\033[2m→ {message}\033[0m\n")
        sys.stderr.flush()

    try:
        with _StdinEchoGuard():
            client = await ensure_daemon_client(
                settings,
                mock_host=mock_host,
                status_callback=_status,
            )
    except DaemonUnavailableError as exc:
        sys.stderr.write(
            "\033[33m→ leapd unavailable; falling back to local volatile-capable mode.\033[0m\n"
        )
        sys.stderr.write(f"\033[2m  {exc}\033[0m\n")
        sys.stderr.flush()
        return await _async_main(args)

    cmd = args.command
    if cmd == "interactive":
        from leapflow.cli.commands.interactive import cmd_interactive_daemon

        return await cmd_interactive_daemon(
            client,
            settings,
            resume_id=getattr(args, "resume", None),
            mock_host=mock_host,
        )
    if cmd == "chat":
        from leapflow.cli.commands.chat import cmd_chat_daemon

        return await cmd_chat_daemon(client, args.prompt, getattr(args, "thinking", False))

    return await _async_main(args)


def _daemon_enabled(args: argparse.Namespace) -> bool:
    """Return whether chat/interactive should use leapd."""
    if getattr(args, "no_daemon", False):
        return False
    raw = os.getenv("LEAPFLOW_DAEMON", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _suggest_known_command(token: str, known_commands: set[str]) -> str | None:
    """Return the closest known command when a token looks like a typo, else None.

    Uses a conservative similarity cutoff so genuine free-text prompts (e.g.
    ``leap what is a daemon``) still route to chat, while near-miss command
    typos (e.g. ``deamon`` -> ``daemon``) are caught and surfaced as a
    suggestion instead of silently becoming an LLM chat turn that also spawns a
    daemon.
    """
    import difflib

    matches = difflib.get_close_matches(
        token.lower(), sorted(known_commands), n=1, cutoff=0.82
    )
    return matches[0] if matches else None


def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--mock-host",
        action="store_true",
        help="Force in-process mock platform (overrides LEAPFLOW_MOCK_HOST).",
    )
    common.add_argument(
        "--no-daemon",
        action="store_true",
        help="Run chat/interactive in this process instead of leapd.",
    )

    parser = argparse.ArgumentParser(
        prog="leap",
        parents=[common],
        description="LeapFlow — Learning and Evolving from Actual Practice",
    )

    parser.add_argument("--thinking", action="store_true", help="Enable LLM reasoning mode")
    parser.add_argument("--resume", metavar="ID", help="Resume a previous chat session")

    subparsers = parser.add_subparsers(dest="command")

    # leap teach
    teach_parser = subparsers.add_parser("teach", parents=[common], help="Interactive teaching mode")
    teach_parser.add_argument("goal", nargs="?", default="", help="Goal description")
    teach_parser.add_argument("--prompt", dest="goal_flag", help="Goal description (alternative)")
    teach_parser.add_argument("--goal", dest="goal_opt", default="", help="Goal description (alternative)")
    teach_parser.add_argument("--timeout", type=float, help="Idle timeout in seconds")
    teach_parser.add_argument("--resume", metavar="ID", help="Resume a previous teaching session")
    teach_parser.add_argument(
        "--field", action="append", default=[],
        help="Session-scoped perceptual field rule (app:context[:level])",
    )

    # leap run
    run_parser = subparsers.add_parser("run", parents=[common], help="Execute a skill")
    run_parser.add_argument("prompt", nargs="?", default="", help="Natural language trigger")
    run_parser.add_argument("--skill", help="Explicit skill name to execute")
    run_parser.add_argument("--step", action="store_true", help="Step-through execution")
    run_parser.add_argument("--auto", action="store_true", help="Skip confirmation, execute directly")

    # leap skills
    skills_parser = subparsers.add_parser("skills", parents=[common], help="Skill management")
    skills_parser.add_argument("action", nargs="?", default="list", choices=["list", "show", "export", "import", "disable", "delete", "audit", "sessions"])
    skills_parser.add_argument("name", nargs="?", help="Skill name (for 'show'/'export'/'audit') or file path (for 'import')")
    skills_parser.add_argument("-o", "--output", help="Output file path (for 'export')")
    skills_parser.add_argument("--limit", type=int, default=20, help="Limit rows (for 'audit')")
    skills_parser.add_argument(
        "--include-suggestions",
        action="store_true",
        help="Include pending skill update suggestions when listing (action='list').",
    )

    # leap relearn
    relearn_parser = subparsers.add_parser("relearn", parents=[common], help="Re-learn from a saved trajectory")
    relearn_parser.add_argument("trajectory_id", help="Trajectory ID to re-process")

    # leap host
    host_parser = subparsers.add_parser("host", help="Manage cua-driver and ObservationDaemon")
    host_sub = host_parser.add_subparsers(dest="host_action")
    host_sub.add_parser("start", help="Start ObservationDaemon (background observers)")
    host_sub.add_parser("stop", help="Stop ObservationDaemon")
    host_sub.add_parser("status", help="Show cua-driver and daemon status")
    host_sub.add_parser("doctor", help="Run cua-driver connectivity health check")
    host_sub.add_parser("install", help="Install cua-driver")

    # leap daemon
    daemon_parser = subparsers.add_parser("daemon", help="Manage leapd daemon")
    daemon_sub = daemon_parser.add_subparsers(dest="daemon_action")
    daemon_sub.add_parser("status", help="Show daemon status")
    daemon_sub.add_parser("start", help="Start daemon for the active profile")
    stop_parser = daemon_sub.add_parser("stop", help="Stop running daemon")
    stop_parser.add_argument("--force", action="store_true", help="Escalate to SIGKILL if graceful stop times out")
    restart_parser = daemon_sub.add_parser("restart", help="Restart daemon for the active profile")
    restart_parser.add_argument("--force", action="store_true", help="Force old daemon shutdown before restart")
    serve_parser = daemon_sub.add_parser("serve", help=argparse.SUPPRESS)
    serve_parser.add_argument("--internal", action="store_true", help=argparse.SUPPRESS)

    # leap board (LeapBoard monitoring dashboard)
    dashboard_parser = subparsers.add_parser("board", help="Open the LeapBoard session analysis board")
    dashboard_parser.add_argument("template", nargs="?", default="", help="Template lens to render (default: generic)")
    dashboard_parser.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    dashboard_parser.add_argument("--token", default="", help=argparse.SUPPRESS)
    dashboard_parser.add_argument("--port", type=int, default=0, help="Override the dashboard port")
    dashboard_parser.add_argument("--bind", default="", help="Override the dashboard bind address")
    dashboard_parser.add_argument("--no-open", action="store_true", help="Print the URL instead of opening a browser")

    # leap config
    config_parser = subparsers.add_parser("config", help="View and update LeapFlow configuration")
    config_sub = config_parser.add_subparsers(dest="config_action")
    config_show = config_sub.add_parser("show", help="Show effective configuration or one config field")
    config_show.add_argument("key", nargs="?", help="Optional config key, e.g. llm.model")
    config_sub.add_parser("keys", help="List writable configuration keys")
    config_list = config_sub.add_parser("list", help="List writable configuration fields with descriptions")
    config_list.add_argument("category", nargs="?", help="Optional category or key prefix filter, e.g. llm or memory")
    config_list.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    config_sub.add_parser("sources", help="List configuration sources")
    config_get = config_sub.add_parser("get", help="Show one effective config value")
    config_get.add_argument("key", help="Config key, e.g. llm.model")
    config_set = config_sub.add_parser("set", help="Set a writable config value")
    config_set.add_argument("key", help="Config key, e.g. llm.model")
    config_set.add_argument("value", help="Config value")
    config_set.add_argument("--scope", choices=["profile", "workspace"], default="profile", help="Durable config scope")
    config_unset = config_sub.add_parser("unset", help="Unset a writable config value")
    config_unset.add_argument("key", help="Config key, e.g. llm.model")
    config_unset.add_argument("--scope", choices=["profile", "workspace"], default="profile", help="Durable config scope")

    config_llm = config_sub.add_parser("llm", help="Configure the primary LLM provider")
    config_llm_sub = config_llm.add_subparsers(dest="llm_action")
    config_llm_sub.add_parser("show", help="Show LLM configuration")
    config_llm_set = config_llm_sub.add_parser("set", help="Set LLM provider fields")
    config_llm_set.add_argument("--api-key", help="LLM API key; prefer --ask-api-key for shell history safety")
    config_llm_set.add_argument("--ask-api-key", action="store_true", help="Prompt securely for the LLM API key")
    config_llm_set.add_argument("--base-url", help="OpenAI-compatible base URL")
    config_llm_set.add_argument("--model", help="Model name")
    config_llm_set.add_argument("--context-length", type=int, help="Runtime context budget in tokens")
    config_llm_set.add_argument("--max-retries", type=int, help="Provider retry count")
    config_llm_set.add_argument("--scope", choices=["profile", "workspace"], default="profile", help="Durable config scope")
    config_llm_key = config_llm_sub.add_parser("key", help="Prompt securely for the LLM API key")
    config_llm_key.add_argument("--scope", choices=["profile", "workspace"], default="profile", help="Durable config scope")

    config_secret = config_sub.add_parser("secret", help="Manage profile/global secret refs")
    config_secret_sub = config_secret.add_subparsers(dest="secret_action")
    config_secret_set = config_secret_sub.add_parser("set", help="Store a secret value")
    config_secret_set.add_argument("ref", help="secret:// ref or shorthand like llm.primary.api_key")
    config_secret_set.add_argument("value", nargs="?", help="Secret value; prompts securely when omitted")
    config_secret_set.add_argument("--scope", choices=["profile", "global"], default="profile", help="Scope for shorthand refs")
    config_secret_get = config_secret_sub.add_parser("get", help="Check or reveal a secret value")
    config_secret_get.add_argument("ref", help="secret:// ref or shorthand like llm.primary.api_key")
    config_secret_get.add_argument("--scope", choices=["profile", "global"], default="profile", help="Scope for shorthand refs")
    config_secret_get.add_argument("--reveal", action="store_true", help="Print the plaintext secret value")
    config_secret_delete = config_secret_sub.add_parser("delete", help="Delete a secret ref")
    config_secret_delete.add_argument("ref", help="secret:// ref or shorthand like llm.primary.api_key")
    config_secret_delete.add_argument("--scope", choices=["profile", "global"], default="profile", help="Scope for shorthand refs")
    config_secret_sub.add_parser("list", help="List stored secret refs without revealing values")

    # ── Pre-parse: detect if first non-flag arg is a known subcommand ──
    # If not, treat everything non-flag as a chat prompt.
    known_commands = {"teach", "run", "skills", "relearn", "host", "daemon", "config", "board"}
    effective_argv = list(argv) if argv is not None else sys.argv[1:]

    # Find first non-flag argument, skipping values owned by global options.
    value_options = {"--resume"}
    prompt_words: list[str] = []
    first_pos = None
    skip_next = False
    for i, tok in enumerate(effective_argv):
        if skip_next:
            skip_next = False
            continue
        if tok in value_options:
            skip_next = True
            continue
        if tok.startswith("-"):
            continue
        first_pos = i
        break

    if first_pos is not None and effective_argv[first_pos] not in known_commands:
        first_token = effective_argv[first_pos]
        non_flag_tokens = [tok for tok in effective_argv if not tok.startswith("-")]
        # A short, command-like invocation whose first word is a near-miss of a
        # known command is almost certainly a typo (e.g. `leap deamon status`).
        # Surface a suggestion instead of silently spawning a daemon + LLM chat.
        if len(non_flag_tokens) <= 3:
            suggestion = _suggest_known_command(first_token, known_commands)
            if suggestion is not None:
                corrected = " ".join(["leap", suggestion, *effective_argv[first_pos + 1:]])
                sys.stderr.write(
                    f"leap: '{first_token}' is not a leap command. "
                    f"Did you mean '{suggestion}'?\n"
                    f"Try: {corrected}\n"
                )
                return 2
        # Collect all non-option prompt tokens while preserving global option values.
        flags: list[str] = []
        prompt_words = []
        skip_next = False
        for tok in effective_argv:
            if skip_next:
                flags.append(tok)
                skip_next = False
                continue
            if tok in value_options:
                flags.append(tok)
                skip_next = True
                continue
            if tok.startswith("-"):
                flags.append(tok)
            else:
                prompt_words.append(tok)
        effective_argv = flags

    args = parser.parse_args(effective_argv)

    # Resolve teach goal from positional, --prompt, or --goal
    if args.command == "teach":
        goal = getattr(args, "goal", "") or ""
        if not goal:
            goal = getattr(args, "goal_flag", "") or getattr(args, "goal_opt", "") or ""
        args.goal = goal

    # ── Route ──
    if not args.command:
        if prompt_words:
            # leap "question" / leap some question here → single-turn chat
            args.command = "chat"
            args.prompt = " ".join(prompt_words)
        else:
            # leap → interactive REPL (Rich banner rendered inside cmd_interactive)
            args.command = "interactive"

    if args.command == "config":
        from leapflow.cli.commands.config import cmd_config
        return cmd_config(args)

    # Host command does not need Context initialization
    if args.command == "host":
        try:
            return asyncio.run(_async_host(args))
        except KeyboardInterrupt:
            sys.stderr.write("\n\033[2m→ Interrupted\033[0m\n")
            return 130

    # Daemon command does not need Context initialization
    if args.command == "daemon":
        from leapflow.cli.commands.daemon import cmd_daemon
        return cmd_daemon(args)

    # Dashboard is a view client (connects to leapd); no Context initialization
    if args.command == "board":
        from leapflow.cli.commands.dashboard import cmd_dashboard
        return cmd_dashboard(args)

    try:
        if args.command in {"interactive", "chat"} and _daemon_enabled(args):
            return asyncio.run(_async_daemon_main(args))
        return asyncio.run(_async_main(args))
    except KeyboardInterrupt:
        sys.stderr.write("\n\033[2m→ Interrupted\033[0m\n")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
