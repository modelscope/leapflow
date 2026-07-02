"""CLI entrypoint for LeapFlow.

Subcommands:
    leap teach  — Interactive teaching mode (record → distill)
    leap run    — Execute a skill by trigger match or explicit name
    leap skills — List / inspect learned skills
    leap chat   — Conversational mode (original single-turn or interactive)
"""

from leapflow.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
