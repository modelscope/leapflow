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
import sys

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
            return await cmd_interactive(ctx)
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


def main(argv: list[str] | None = None) -> int:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--mock-host",
        action="store_true",
        help="Force in-process mock OSHost (overrides LEAPFLOW_MOCK_HOST).",
    )

    parser = argparse.ArgumentParser(
        prog="leap",
        parents=[common],
        description="LeapFlow — Learning and Evolving from Actual Practice",
    )

    parser.add_argument("--thinking", action="store_true", help="Enable LLM reasoning mode")

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
    host_parser = subparsers.add_parser("host", help="Manage OS Host lifecycle")
    host_sub = host_parser.add_subparsers(dest="host_action")
    host_sub.add_parser("start", help="Start the OS Host daemon")
    host_sub.add_parser("stop", help="Gracefully stop the OS Host")
    host_sub.add_parser("restart", help="Restart the OS Host")
    host_sub.add_parser("status", help="Show host status and permissions")
    logs_parser = host_sub.add_parser("logs", help="View host logs")
    logs_parser.add_argument("--follow", "-f", action="store_true", help="Stream logs in real-time")
    host_sub.add_parser("install", help="Build and deploy .app bundle")
    host_sub.add_parser("setup", help="Install + register launchd + permission guidance")
    host_sub.add_parser("uninstall", help="Stop, unregister, and remove bundle")
    host_sub.add_parser("dev", help="Development mode (auto-rebuild on changes)")

    # ── Pre-parse: detect if first non-flag arg is a known subcommand ──
    # If not, treat everything non-flag as a chat prompt.
    known_commands = {"teach", "run", "skills", "relearn", "host"}
    effective_argv = list(argv) if argv is not None else sys.argv[1:]

    # Find first non-flag argument
    prompt_words: list[str] = []
    first_pos = None
    for i, tok in enumerate(effective_argv):
        if tok.startswith("-"):
            continue
        first_pos = i
        break

    if first_pos is not None and effective_argv[first_pos] not in known_commands:
        # Collect all non-flag tokens as prompt, remove them from argv for argparse
        flags = [t for t in effective_argv if t.startswith("-")]
        prompt_words = [t for t in effective_argv if not t.startswith("-")]
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
            # leap → banner + interactive REPL
            from leapflow.cli.banner import display_welcome
            try:
                display_welcome()
            except KeyboardInterrupt:
                return 130
            args.command = "interactive"

    # Host command does not need Context initialization
    if args.command == "host":
        try:
            return asyncio.run(_async_host(args))
        except KeyboardInterrupt:
            sys.stderr.write("\n\033[2m→ Interrupted\033[0m\n")
            return 130

    try:
        return asyncio.run(_async_main(args))
    except KeyboardInterrupt:
        sys.stderr.write("\n\033[2m→ Interrupted\033[0m\n")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
