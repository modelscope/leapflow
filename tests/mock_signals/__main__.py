# Copyright (c) Alibaba, Inc. and its affiliates.
"""Mock signal injection for LeapFlow end-to-end testing.

Usage:
    python -m tests.mock_signals                    # Run default "normal" profile
    python -m tests.mock_signals --profile burst    # Run "burst" profile
    python -m tests.mock_signals --profile stress   # Run "stress" profile
    python -m tests.mock_signals --daemon           # Inject into running leapd
    python -m tests.mock_signals --list             # List available profiles
    python -m tests.mock_signals --profile normal --duration 30  # Override duration
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from tests.mock_signals.profiles import PROFILES
from tests.mock_signals.runner import MockSignalRunner


def _list_profiles() -> None:
    """Print all available profiles with descriptions."""
    print("\nAvailable signal profiles:\n")
    for name, profile in sorted(PROFILES.items()):
        print(f"  {name:<12} {profile.description}")
        gen_names = [g[0] for g in profile.generators]
        print(f"  {'':12} generators: {', '.join(gen_names)}")
        print()


async def _run_profile(
    profile_name: str,
    *,
    daemon_mode: bool = False,
    duration_override: float | None = None,
    freq_multiplier: float | None = None,
) -> int:
    """Run a profile and print results. Returns exit code."""
    mode_label = "DAEMON (injecting into running leapd)" if daemon_mode else "LOCAL (in-process pipeline)"
    print(f"  Mode: {mode_label}")
    runner = MockSignalRunner(
        profile_name,
        daemon_mode=daemon_mode,
        duration_override=duration_override,
        freq_multiplier=freq_multiplier,
    )
    result = await runner.run()
    print(result.summary())
    return 0 if not result.errors else 1


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="LeapFlow mock signal injection framework"
    )
    parser.add_argument(
        "--profile", "-p", default="normal", help="Signal profile name"
    )
    parser.add_argument(
        "--list", "-l", action="store_true", help="List available profiles"
    )
    parser.add_argument(
        "--duration", "-d", type=float, help="Override duration (seconds)"
    )
    parser.add_argument(
        "--frequency", "-f", type=float, help="Override frequency multiplier"
    )
    parser.add_argument(
        "--daemon", action="store_true",
        help="Inject signals into running leapd via RPC (affects TUI/LeapBoard)",
    )
    args = parser.parse_args()

    if args.list:
        _list_profiles()
        return

    if args.profile not in PROFILES:
        print(f"Error: unknown profile {args.profile!r}", file=sys.stderr)
        print(f"Available: {', '.join(sorted(PROFILES))}", file=sys.stderr)
        sys.exit(1)

    exit_code = asyncio.run(
        _run_profile(
            args.profile,
            daemon_mode=args.daemon,
            duration_override=args.duration,
            freq_multiplier=args.frequency,
        )
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
