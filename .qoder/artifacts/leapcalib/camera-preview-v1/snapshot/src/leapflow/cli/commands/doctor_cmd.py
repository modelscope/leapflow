# Copyright (c) Alibaba, Inc. and its affiliates.
"""CLI handler for ``leap doctor``."""
from __future__ import annotations

import argparse
import asyncio


def cmd_doctor(args: argparse.Namespace) -> int:
    """Synchronous entry point for ``leap doctor``."""
    try:
        return asyncio.run(_async_doctor(args))
    except KeyboardInterrupt:
        import sys

        sys.stderr.write("\n\033[2m→ Interrupted\033[0m\n")
        return 130


async def _async_doctor(args: argparse.Namespace) -> int:
    """Run all diagnostic checks and print the report."""
    from leapflow.config import load_config

    settings = load_config()

    from leapflow.cli.doctor import (
        build_doctor_checks,
        print_doctor_report,
        run_doctor,
    )

    should_fix = getattr(args, "fix", False)
    section_filter = getattr(args, "section", None)

    checks = build_doctor_checks(settings)
    aggregate, details = await run_doctor(
        checks,
        should_fix=should_fix,
        section_filter=section_filter,
    )
    print_doctor_report(aggregate, details)
    return 0 if aggregate.ok else 1
