"""Hub CLI commands — push, pull, sync, search, list, login, whoami.

Provides the ``leap hub`` subcommand family for cloud skill collaboration
through configured Hub backends (ModelScope, HuggingFace, local, etc.).
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from leapflow.cli.context import Context


# ─── Human-in-the-loop Confirmation ─────────────────────────────────────────


async def _confirm(prompt: str, *, dangerous: bool = False, skill_name: str = "") -> bool:
    """Human-in-the-loop confirmation.

    Standard: "Proceed? [y/N]"
    Dangerous: "Type skill name to confirm: " (exact match required)
    """
    import asyncio

    loop = asyncio.get_event_loop()
    if dangerous and skill_name:
        display = f"  \033[1;31m{prompt}\033[0m\n  Type '{skill_name}' to confirm: "
        answer = await loop.run_in_executor(None, lambda: input(display).strip())
        return answer == skill_name
    else:
        display = f"  {prompt} [y/N]: "
        answer = await loop.run_in_executor(None, lambda: input(display).strip().lower())
        return answer in ("y", "yes")


# ─── Utility Helpers ─────────────────────────────────────────────────────────


def _build_hub_client(ctx: "Context"):
    """Create HubClient from context settings."""
    from leapflow.hub import HubClient

    return HubClient(
        hub_type=ctx.settings.hub_type,
        default_owner=ctx.settings.hub_default_owner,
        default_visibility=ctx.settings.hub_default_visibility,
        repo_prefix=ctx.settings.hub_repo_prefix,
        search_sources=ctx.settings.hub_search_sources,
    )


def _severity_icon(severity: str) -> str:
    """Return severity indicator icon."""
    icons = {"high": "\033[1;31m!\033[0m", "medium": "\033[1;33m~\033[0m", "low": "\033[2m-\033[0m"}
    return icons.get(severity, "-")


def _print_warnings(warnings, label: str = "Warnings") -> None:
    """Print sanitization/audit warnings."""
    if not warnings:
        return
    high = sum(1 for w in warnings if w.severity == "high")
    med = sum(1 for w in warnings if w.severity == "medium")
    low = sum(1 for w in warnings if w.severity == "low")
    print(f"\n  {label}: {high} high, {med} medium, {low} low")
    for w in warnings[:10]:
        print(f"    {_severity_icon(w.severity)} [{w.severity}] {w.detail}")
    if len(warnings) > 10:
        print(f"    ... and {len(warnings) - 10} more")


# ─── Subcommand Router ───────────────────────────────────────────────────────


async def cmd_hub(ctx: "Context", args: List[str]) -> int:
    """Route hub subcommands."""
    if not args:
        _print_hub_help()
        return 0

    subcmd = args[0].lower()
    rest = args[1:]

    dispatch = {
        "login": _hub_login,
        "whoami": _hub_whoami,
        "push": _hub_push,
        "pull": _hub_pull,
        "sync": _hub_sync,
        "search": _hub_search,
        "list": _hub_list,
        "info": _hub_info,
    }

    handler = dispatch.get(subcmd)
    if handler is None:
        print(f"Unknown hub command: '{subcmd}'")
        _print_hub_help()
        return 1

    try:
        return await handler(ctx, rest)
    except ValueError as e:
        print(f"  Error: {e}")
        return 1
    except ImportError as e:
        print(f"  SDK not available: {e}")
        print("  Install the required Hub SDK (e.g. 'pip install modelscope').")
        return 1
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")
        return 1


def _print_hub_help() -> None:
    """Print hub subcommand help."""
    print("Hub commands:")
    print("  hub login                           — Authenticate with Hub")
    print("  hub whoami                          — Show current user")
    print("  hub push <skill> [--visibility V]   — Push skill to Hub")
    print("  hub pull <repo-id> [--version V]    — Pull skill from Hub")
    print("  hub sync [--push-only|--pull-only]  — Sync local/remote skills")
    print("  hub search <query>                  — Search skills on Hub")
    print("  hub list [--mine]                   — List remote skills")
    print("  hub info <repo-id>                  — Show skill details")
    print()


# ─── Login ───────────────────────────────────────────────────────────────────


async def _hub_login(ctx: "Context", args: List[str]) -> int:
    """Authenticate with the hub backend."""
    client = _build_hub_client(ctx)
    print(f"  Authenticating with {client.hub_type}...")
    user = await client.login()
    print(f"  Logged in as: {user.username}")
    if user.email:
        print(f"  Email: {user.email}")
    return 0


# ─── Whoami ──────────────────────────────────────────────────────────────────


async def _hub_whoami(ctx: "Context", args: List[str]) -> int:
    """Show current authenticated user."""
    client = _build_hub_client(ctx)
    user = await client.whoami()
    print(f"  User: {user.username}")
    if user.email:
        print(f"  Email: {user.email}")
    print(f"  Hub: {client.hub_type}")
    return 0


# ─── Push ────────────────────────────────────────────────────────────────────


async def _hub_push(ctx: "Context", args: List[str]) -> int:
    """Push a skill to the Hub."""
    from leapflow.hub import ContentSanitizer, SkillSerializer, Visibility
    from leapflow.hub.protocol import VersionConflictError

    if not args:
        print("  Usage: hub push <skill-name> [--visibility private|public] [--version v1.0.0]")
        return 1

    skill_name = args[0]
    visibility_str = "private"
    version = None

    # Parse flags
    i = 1
    while i < len(args):
        if args[i] == "--visibility" and i + 1 < len(args):
            visibility_str = args[i + 1]
            i += 2
        elif args[i] == "--version" and i + 1 < len(args):
            version = args[i + 1]
            i += 2
        else:
            i += 1

    # Step 1: Export skill from local store
    if not ctx.skill_lib:
        print("  Error: Skill library not initialized.")
        return 1

    stored = ctx.skill_lib.load_skill_by_title(skill_name)
    if stored is None:
        print(f"  Error: Skill '{skill_name}' not found in local library.")
        return 1

    # Step 2: Serialize to bundle
    serializer = SkillSerializer()
    stored_dict = {
        "name": stored.title if hasattr(stored, "title") else skill_name,
        "version": version or getattr(stored, "version", "0.1.0"),
        "description": getattr(stored, "description", ""),
        "source_code": getattr(stored, "source_code", ""),
        "parameters": getattr(stored, "parameters", []),
        "triggers": list(getattr(stored, "trigger_phrases", [])),
        "trajectory_skeleton": getattr(stored, "trajectory_skeleton", ""),
        "copilot_prior": getattr(stored, "copilot_prior", ""),
        "readme": getattr(stored, "readme", ""),
        "source_tag": getattr(stored, "source_tag", "learned"),
        "tier": getattr(stored, "tier", 1),
    }
    bundle = serializer.export_skill(stored_dict)

    # Step 3: Sanitize content
    sanitizer = ContentSanitizer()
    warnings = sanitizer.scan(bundle)
    _print_warnings(warnings, "Sanitization")

    # Step 3b: Check for blocking findings
    if sanitizer.has_blocking_findings(warnings) and "--force" not in args:
        print("  HIGH severity findings detected. Use --force to push anyway.")
        confirmed = await _confirm("Push despite high-severity warnings?", dangerous=True, skill_name=skill_name)
        if not confirmed:
            print("  Push aborted.")
            return 0

    # Step 4: Show summary
    visibility = Visibility(visibility_str)
    client = _build_hub_client(ctx)
    repo_id = client._build_repo_id(bundle.manifest.name)

    print(f"\n  Push Summary:")
    print(f"    Skill:      {bundle.manifest.name}")
    print(f"    Version:    {bundle.manifest.version}")
    print(f"    Visibility: {visibility.value}")
    print(f"    Target:     {repo_id} ({client.hub_type})")

    # Step 5: Confirm (dangerous if public)
    is_public = visibility == Visibility.PUBLIC
    if is_public:
        confirmed = await _confirm(
            "Publishing publicly is irreversible.",
            dangerous=True,
            skill_name=bundle.manifest.name,
        )
    else:
        confirmed = await _confirm("Proceed with push?")

    if not confirmed:
        print("  Cancelled.")
        return 0

    # Step 6: Auto-fill author info
    import dataclasses
    from datetime import datetime, timezone

    try:
        user_info = await client.backend.authenticate()
        username = user_info.username
    except Exception:
        username = ""

    if username:
        manifest = dataclasses.replace(
            bundle.manifest,
            author=bundle.manifest.author or username,
            updated_by=username,
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        bundle = dataclasses.replace(bundle, manifest=manifest)

    # Step 7: Push
    force = "--force" in args
    try:
        result = await client.push(bundle, skill_name=bundle.manifest.name, visibility=visibility, force=force)
    except VersionConflictError as e:
        print(f"  Version conflict: {e}")
        confirmed = await _confirm("Force push anyway?")
        if confirmed:
            result = await client.push(bundle, skill_name=bundle.manifest.name, visibility=visibility, force=True)
        else:
            print("  Push aborted.")
            return 0

    print(f"\n  Pushed successfully!")
    print(f"    Repo: {result.repo_id}")
    print(f"    Version: {result.version}")
    print(f"    URL: {result.url}")
    return 0


# ─── Pull ────────────────────────────────────────────────────────────────────


async def _hub_pull(ctx: "Context", args: List[str]) -> int:
    """Pull a skill from the Hub."""
    from leapflow.hub import SecurityAuditor, SkillSerializer

    if not args:
        print("  Usage: hub pull <repo-id> [--version v1.0.0] [--trust]")
        return 1

    repo_id = args[0]
    version = None
    trust = False

    i = 1
    while i < len(args):
        if args[i] == "--version" and i + 1 < len(args):
            version = args[i + 1]
            i += 2
        elif args[i] == "--trust":
            trust = True
            i += 1
        else:
            i += 1

    # Step 1: Pull bundle
    client = _build_hub_client(ctx)
    print(f"  Pulling {repo_id}...")
    bundle = await client.pull(repo_id, version=version)

    # Step 2: Security audit
    auditor = SecurityAuditor()
    findings = auditor.audit(bundle)
    _print_warnings(findings, "Security Audit")

    # Step 3: Check high-risk findings
    high_risk = [f for f in findings if f.severity == "high"]
    if high_risk and not trust:
        print(f"\n  \033[1;31m{len(high_risk)} high-risk finding(s) detected.\033[0m")
        confirmed = await _confirm(
            "Install despite high-risk findings?",
            dangerous=True,
            skill_name=bundle.manifest.name,
        )
        if not confirmed:
            print("  Cancelled.")
            return 0

    # Step 4: Show summary
    print(f"\n  Pull Summary:")
    print(f"    Skill:   {bundle.manifest.name}")
    print(f"    Version: {bundle.manifest.version}")
    print(f"    Source:  {repo_id} ({client.hub_type})")

    if not trust and not high_risk:
        confirmed = await _confirm("Install this skill locally?")
        if not confirmed:
            print("  Cancelled.")
            return 0

    # Step 5: Import locally
    serializer = SkillSerializer()
    skill_data = serializer.import_skill(bundle)

    if ctx.skill_lib:
        ctx.skill_lib.save_from_hub(skill_data)
        print(f"\n  Installed '{bundle.manifest.name}' successfully.")
    else:
        print("  Warning: Skill library not available. Bundle downloaded but not installed.")

    return 0


# ─── Sync ────────────────────────────────────────────────────────────────────


async def _hub_sync(ctx: "Context", args: List[str]) -> int:
    """Sync local and remote skills."""
    push_only = "--push-only" in args
    pull_only = "--pull-only" in args
    dry_run = "--dry-run" in args

    if push_only and pull_only:
        print("  Error: Cannot use --push-only and --pull-only together.")
        return 1

    client = _build_hub_client(ctx)

    # Gather local skills
    local_manifests = []
    if ctx.skill_lib:
        from leapflow.hub.protocol import SkillManifest

        stored_skills = ctx.skill_lib.load_all_active()
        for s in stored_skills:
            local_manifests.append(SkillManifest(
                name=s.title if hasattr(s, "title") else "",
                version=getattr(s, "version", "0.1.0"),
                description=getattr(s, "description", ""),
            ))

    # Compute sync plan
    print(f"  Sync strategy: {ctx.settings.hub_sync_strategy}")
    print("  Computing sync plan...")
    plan = await client.sync_skills(local_manifests)

    if plan.is_empty:
        print("  Everything is in sync.")
        return 0

    # Display plan
    print(f"\n  Sync Plan:")
    if plan.to_push and not pull_only:
        print(f"    Push ({len(plan.to_push)}):")
        for m in plan.to_push:
            print(f"      -> {m.name} v{m.version}")
    if plan.to_pull and not push_only:
        print(f"    Pull ({len(plan.to_pull)}):")
        for s in plan.to_pull:
            print(f"      <- {s.name} v{s.version}")
    if plan.conflicts:
        print(f"    Conflicts ({len(plan.conflicts)}):")
        for name in plan.conflicts:
            print(f"      !! {name}")

    if dry_run:
        print("\n  (dry-run — no changes applied)")
        return 0

    confirmed = await _confirm("Execute sync plan?")
    if not confirmed:
        print("  Cancelled.")
        return 0

    # Execute plan
    errors = 0
    if plan.to_push and not pull_only:
        from leapflow.hub import SkillSerializer

        serializer = SkillSerializer()
        for manifest in plan.to_push:
            try:
                stored = ctx.skill_lib.load_skill_by_title(manifest.name) if ctx.skill_lib else None
                if stored is None:
                    print(f"    Skip push '{manifest.name}': not found locally")
                    continue
                stored_dict = {
                    "name": manifest.name,
                    "version": manifest.version,
                    "description": manifest.description,
                    "source_code": getattr(stored, "source_code", ""),
                    "triggers": list(getattr(stored, "trigger_phrases", [])),
                    "trajectory_skeleton": getattr(stored, "trajectory_skeleton", ""),
                    "copilot_prior": getattr(stored, "copilot_prior", ""),
                    "readme": getattr(stored, "readme", ""),
                }
                bundle = serializer.export_skill(stored_dict)
                await client.push(bundle, skill_name=manifest.name)
                print(f"    Pushed: {manifest.name}")
            except Exception as e:
                print(f"    Error pushing '{manifest.name}': {e}")
                errors += 1

    if plan.to_pull and not push_only:
        from leapflow.hub import SkillSerializer

        serializer = SkillSerializer()
        for summary in plan.to_pull:
            try:
                bundle = await client.pull(summary.repo_id)
                skill_data = serializer.import_skill(bundle)
                if ctx.skill_lib:
                    ctx.skill_lib.save_from_hub(skill_data)
                print(f"    Pulled: {summary.name}")
            except Exception as e:
                print(f"    Error pulling '{summary.name}': {e}")
                errors += 1

    if errors:
        print(f"\n  Sync completed with {errors} error(s).")
    else:
        print("\n  Sync completed successfully.")
    return 0 if not errors else 1


# ─── Search ──────────────────────────────────────────────────────────────────


async def _hub_search(ctx: "Context", args: List[str]) -> int:
    """Search for skills on the Hub."""
    search_all = "--all" in args
    filtered_args = [a for a in args if a != "--all"]

    if not filtered_args:
        print("  Usage: hub search <query> [--all]")
        return 1

    query = " ".join(filtered_args)
    client = _build_hub_client(ctx)

    if search_all:
        sources = client._search_sources
        print(f"  Searching '{query}' across {', '.join(sources)}...")
        results = await client.search_all(query)
    else:
        print(f"  Searching '{query}' on {client.hub_type}...")
        results = await client.search(query)

    if not results:
        print("  No results found.")
        return 0

    print(f"\n  {'Name':<30} {'Source':<12} {'Version':<10} Description")
    print(f"  {'-'*80}")
    for r in results:
        desc = (r.description[:30] + "...") if len(r.description) > 33 else r.description
        source = r.hub_type or client.hub_type
        print(f"  {r.name:<30} {source:<12} {r.version:<10} {desc}")

    print(f"\n  {len(results)} result(s) found.")
    return 0


# ─── List ────────────────────────────────────────────────────────────────────


async def _hub_list(ctx: "Context", args: List[str]) -> int:
    """List skills on the Hub."""
    mine = "--mine" in args
    client = _build_hub_client(ctx)
    owner = ctx.settings.hub_default_owner if mine else None

    print(f"  Listing skills on {client.hub_type}...")
    results = await client.search("", owner=owner)

    if not results:
        print("  No skills found.")
        return 0

    print(f"\n  {'Repo ID':<35} {'Version':<10} Description")
    print(f"  {'-'*70}")
    for r in results:
        desc = (r.description[:30] + "...") if len(r.description) > 33 else r.description
        print(f"  {r.repo_id:<35} {r.version:<10} {desc}")

    print(f"\n  {len(results)} skill(s).")
    return 0


# ─── Info ────────────────────────────────────────────────────────────────────


async def _hub_info(ctx: "Context", args: List[str]) -> int:
    """Show detailed info for a skill on the Hub."""
    if not args:
        print("  Usage: hub info <repo-id>")
        return 1

    repo_id = args[0]
    client = _build_hub_client(ctx)

    # Get versions
    versions = await client.backend.get_skill_versions(repo_id)
    if not versions:
        print(f"  No info found for '{repo_id}'.")
        return 1

    print(f"\n  Skill: {repo_id}")
    print(f"  Hub:   {client.hub_type}")
    print(f"\n  Versions ({len(versions)}):")
    for v in versions:
        created = f" ({v.created_at})" if v.created_at else ""
        sha = f" [{v.commit_sha[:8]}]" if v.commit_sha else ""
        print(f"    {v.version}{created}{sha}")

    return 0
