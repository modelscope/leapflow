# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unified ``leap doctor`` orchestrator.

Discovers and runs all registered :class:`DiagnosticCheck` instances,
presenting results through Rich console output.  The public entry points are:

* :func:`run_doctor` — async orchestrator returning an aggregate :class:`Finding`.
* :func:`build_doctor_checks` — factory that builds the default check list from
  a :class:`Settings` object.
* :func:`print_doctor_report` — Rich-formatted terminal output.
"""
from __future__ import annotations

import sys
from typing import Any, Sequence

from leapflow.cli.doctor.protocol import DiagnosticCheck, Finding

# ── Section display order ───────────────────────────────────────────
SECTION_ORDER = ("platform", "config", "connectivity", "state", "tools")


# ── Check factory ───────────────────────────────────────────────────

def build_doctor_checks(settings: Any) -> list[DiagnosticCheck]:
    """Build the default diagnostic check list from runtime settings."""
    from leapflow.cli.doctor.checks_config import (
        LLMConfigCheck,
        PathLayoutCheck,
        ProfileConfigCheck,
    )
    from leapflow.cli.doctor.checks_connectivity import (
        DaemonHealthCheck,
        GatewayConnectivityCheck,
        LLMConnectivityCheck,
    )
    from leapflow.cli.doctor.checks_platform import (
        DiskSpaceCheck,
        OSCompatibilityCheck,
        PythonVersionCheck,
    )
    from leapflow.cli.doctor.checks_state import (
        DuckDBHealthCheck,
        SchedulerHealthCheck,
        VaultCheck,
    )
    from leapflow.cli.doctor.checks_tools import (
        CoreToolsCheck,
        MCPServerCheck,
        PluginRegistryCheck,
    )

    layout = settings.profile_layout
    return [
        # Platform
        PythonVersionCheck(),
        OSCompatibilityCheck(),
        DiskSpaceCheck(data_dir=settings.data_dir),
        # Config
        ProfileConfigCheck(profile_layout=layout),
        LLMConfigCheck(settings=settings),
        PathLayoutCheck(profile_layout=layout),
        # Connectivity
        DaemonHealthCheck(runtime_dir=settings.runtime_dir),
        LLMConnectivityCheck(settings=settings),
        GatewayConnectivityCheck(profile_layout=layout),
        # State
        DuckDBHealthCheck(duckdb_path=settings.duckdb_path),
        VaultCheck(profile_layout=layout),
        SchedulerHealthCheck(profile_layout=layout),
        # Tools
        PluginRegistryCheck(),
        CoreToolsCheck(),
        MCPServerCheck(settings=settings),
    ]


# ── Orchestrator ────────────────────────────────────────────────────

async def run_doctor(
    checks: Sequence[DiagnosticCheck],
    *,
    should_fix: bool = False,
    section_filter: str | None = None,
) -> tuple[Finding, list[tuple[DiagnosticCheck, Finding]]]:
    """Execute all *checks* and return (aggregate, per-check details).

    Parameters
    ----------
    checks:
        The list of diagnostic checks to run.
    should_fix:
        When ``True``, checks may attempt auto-remediation.
    section_filter:
        When set, only run checks whose ``section`` matches.

    Returns
    -------
    A 2-tuple of the merged :class:`Finding` and per-check details.
    """
    aggregate = Finding()
    details: list[tuple[DiagnosticCheck, Finding]] = []

    for check in checks:
        if section_filter and check.section != section_filter:
            continue
        result = await check.check(should_fix=should_fix)
        details.append((check, result))
        aggregate = aggregate.merge(result)

    return aggregate, details


# ── Rich output ─────────────────────────────────────────────────────

def print_doctor_report(
    aggregate: Finding,
    details: list[tuple[DiagnosticCheck, Finding]],
    *,
    file: Any = None,
) -> None:
    """Render the diagnostic report to the terminal with Rich formatting."""
    from rich.console import Console

    console = Console(file=file or sys.stdout)
    console.print()
    console.print("[bold cyan]LeapFlow Doctor[/bold cyan]")
    console.print()

    # Group by section
    sections: dict[str, list[tuple[DiagnosticCheck, Finding]]] = {}
    for check, finding in details:
        sections.setdefault(check.section, []).append((check, finding))

    for section in SECTION_ORDER:
        items = sections.get(section)
        if not items:
            continue
        console.print(f"  [bold]{section.upper()}[/bold]")
        for check, finding in items:
            _print_check_line(console, check.name, finding)
        console.print()

    # Summary
    _print_summary(console, aggregate)


def _print_check_line(console: Any, name: str, finding: Finding) -> None:
    """Print a single check result line."""
    if finding.errors:
        icon = "[red]✗[/red]"
        detail = finding.errors[0]
        console.print(f"    {icon} {name}: [red]{detail}[/red]")
        for extra in finding.errors[1:]:
            console.print(f"        [red]{extra}[/red]")
    elif finding.warnings:
        icon = "[yellow]![/yellow]"
        detail = finding.warnings[0]
        console.print(f"    {icon} {name}: [yellow]{detail}[/yellow]")
        for extra in finding.warnings[1:]:
            console.print(f"        [yellow]{extra}[/yellow]")
    elif finding.fixed:
        icon = "[cyan]🔧[/cyan]"
        console.print(f"    {icon} {name}: [cyan]fixed ({finding.fixed} action(s))[/cyan]")
    else:
        icon = "[green]✓[/green]"
        console.print(f"    {icon} {name}")


def _print_summary(console: Any, aggregate: Finding) -> None:
    """Print the final summary line."""
    parts: list[str] = []
    parts.append(f"[green]{aggregate.passed} passed[/green]")
    if aggregate.warnings:
        parts.append(f"[yellow]{len(aggregate.warnings)} warning(s)[/yellow]")
    if aggregate.errors:
        parts.append(f"[red]{len(aggregate.errors)} error(s)[/red]")
    if aggregate.fixed:
        parts.append(f"[cyan]{aggregate.fixed} fixed[/cyan]")

    summary = ", ".join(parts)
    if aggregate.ok:
        console.print(f"  [bold green]Summary:[/bold green] {summary}")
    else:
        console.print(f"  [bold red]Summary:[/bold red] {summary}")


# ── Serializable payload (for TUI /doctor command) ──────────────────

def build_doctor_payload(
    aggregate: Finding,
    details: list[tuple[DiagnosticCheck, Finding]],
) -> dict[str, Any]:
    """Build a serializable dict for TUI rendering."""
    checks_data: list[dict[str, Any]] = []
    for check, finding in details:
        status = "pass"
        if finding.errors:
            status = "error"
        elif finding.warnings:
            status = "warning"
        elif finding.fixed:
            status = "fixed"
        checks_data.append({
            "name": check.name,
            "section": check.section,
            "status": status,
            "errors": finding.errors,
            "warnings": finding.warnings,
            "passed": finding.passed,
            "fixed": finding.fixed,
        })
    lines = []
    lines.append("LeapFlow Doctor Report")
    lines.append(f"  {aggregate.passed} passed, "
                 f"{len(aggregate.warnings)} warning(s), "
                 f"{len(aggregate.errors)} error(s), "
                 f"{aggregate.fixed} fixed")
    for entry in checks_data:
        if entry["status"] == "error":
            lines.append(f"  ✗ {entry['name']}: {entry['errors'][0]}")
        elif entry["status"] == "warning":
            lines.append(f"  ! {entry['name']}: {entry['warnings'][0]}")
        elif entry["status"] == "fixed":
            lines.append(f"  🔧 {entry['name']}: fixed")
        else:
            lines.append(f"  ✓ {entry['name']}")

    return {
        "ok": aggregate.ok,
        "message": "\n".join(lines),
        "checks": checks_data,
        "summary": {
            "passed": aggregate.passed,
            "warnings": len(aggregate.warnings),
            "errors": len(aggregate.errors),
            "fixed": aggregate.fixed,
        },
    }
