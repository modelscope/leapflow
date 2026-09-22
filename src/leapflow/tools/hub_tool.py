# Copyright (c) Alibaba, Inc. and its affiliates.
"""Hub operations as an Agent Tool — enables natural language hub interaction.

Registered as agent-callable tools so the AgentEngine can push, pull, search,
and sync skills with the configured Hub backend during chat-mode conversations.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ─── Tool Implementations ────────────────────────────────────────────────────


async def hub_push_tool(
    skill_name: str = "",
    visibility: str = "private",
    version: str = "",
    **kwargs: Any,
) -> str:
    """Push a local skill to the Hub. Returns status message.

    Requires a Context with skill_lib to load real skill data.

    Args:
        skill_name: Name of the local skill to push.
        visibility: Target visibility ('private', 'public', 'internal').
        version: Optional version override.
    """
    from leapflow.config import get_settings
    from leapflow.hub import ContentSanitizer, HubClient, SkillSerializer, Visibility

    if not skill_name:
        return "Error: skill_name is required."

    ctx = kwargs.get("ctx")

    # Try to load real skill from library
    stored_dict: Dict[str, Any] = {"name": skill_name}
    if ctx is not None and hasattr(ctx, "skill_lib") and ctx.skill_lib is not None:
        try:
            stored = ctx.skill_lib.load_skill_by_title(skill_name)
            if stored is None:
                return f"Error: Skill '{skill_name}' not found in local library."
            stored_dict = {
                "name": getattr(stored, "title", skill_name),
                "version": version or getattr(stored, "version", "0.1.0"),
                "description": getattr(stored, "description", ""),
                "source_code": getattr(stored, "source_code", ""),
                "parameters": getattr(stored, "parameters", []),
                "triggers": list(getattr(stored, "trigger_phrases", [])),
                "trajectory_skeleton": getattr(stored, "trajectory_skeleton", ""),
                "copilot_prior": getattr(stored, "copilot_prior", ""),
                "readme": getattr(stored, "readme", f"# {skill_name}\n"),
                "source_tag": getattr(stored, "source_tag", "learned"),
                "tier": getattr(stored, "tier", 1),
            }
        except Exception as e:
            return f"Error loading skill '{skill_name}': {e}"
    else:
        return "Error: Skill library context not available. Use 'leap hub push' CLI command instead."

    settings = get_settings()
    client = HubClient(
        hub_type=settings.hub_type,
        default_owner=settings.hub_default_owner,
        default_visibility=settings.hub_default_visibility,
        repo_prefix=settings.hub_repo_prefix,
    )

    # Serialize to bundle
    serializer = SkillSerializer()
    bundle = serializer.export_skill(stored_dict)

    # Sanitize
    sanitizer = ContentSanitizer()
    warnings = sanitizer.scan(bundle)
    warning_text = ""
    if warnings:
        high = sum(1 for w in warnings if w.severity == "high")
        if high > 0:
            warning_text = f" ({high} high-risk warnings detected — review before publishing)"

    # Push
    vis = Visibility(visibility)
    try:
        result = await client.push(bundle, skill_name=skill_name, visibility=vis)
        return (
            f"Pushed '{skill_name}' to {result.repo_id} "
            f"(v{result.version}, {visibility}).{warning_text}\n"
            f"URL: {result.url}"
        )
    except Exception as e:
        return f"Push failed: {type(e).__name__}: {e}"


async def hub_pull_tool(
    repo_id: str = "",
    version: str = "",
    **kwargs: Any,
) -> str:
    """Pull a skill from the Hub. Returns status message.

    Args:
        repo_id: Repository identifier (e.g. 'owner/leapflow-skill-name').
        version: Optional specific version to pull.
    """
    from leapflow.config import get_settings
    from leapflow.hub import HubClient, SecurityAuditor, SkillSerializer

    if not repo_id:
        return "Error: repo_id is required."

    settings = get_settings()
    client = HubClient(
        hub_type=settings.hub_type,
        default_owner=settings.hub_default_owner,
        default_visibility=settings.hub_default_visibility,
        repo_prefix=settings.hub_repo_prefix,
    )

    try:
        bundle = await client.pull(repo_id, version=version or None)
    except Exception as e:
        return f"Pull failed: {type(e).__name__}: {e}"

    # Security audit
    auditor = SecurityAuditor()
    findings = auditor.audit(bundle)
    high_risk = [f for f in findings if f.severity == "high"]

    finding_text = ""
    if high_risk:
        finding_text = (
            f"\n\nWARNING: {len(high_risk)} high-risk finding(s):\n"
            + "\n".join(f"  - {f.detail}" for f in high_risk[:5])
        )

    # Import to local
    serializer = SkillSerializer()
    skill_data = serializer.import_skill(bundle)

    # Attempt to save to local skill library if context available
    ctx = kwargs.get("ctx")
    if ctx is not None and hasattr(ctx, "skill_lib") and ctx.skill_lib is not None:
        if high_risk:
            return (
                f"Pulled '{bundle.manifest.name}' v{bundle.manifest.version} "
                f"from {repo_id} but NOT installed due to {len(high_risk)} "
                f"high-risk finding(s).{finding_text}\n"
                f"Use CLI 'leap hub pull {repo_id} --trust' to install with risks accepted."
            )
        try:
            ctx.skill_lib.save_from_hub(skill_data)
            return (
                f"Pulled and installed '{bundle.manifest.name}' "
                f"v{bundle.manifest.version} from {repo_id}.{finding_text}"
            )
        except Exception as e:
            return (
                f"Pulled '{bundle.manifest.name}' v{bundle.manifest.version} "
                f"from {repo_id} but install failed: {e}.{finding_text}"
            )

    return (
        f"Pulled '{bundle.manifest.name}' v{bundle.manifest.version} "
        f"from {repo_id}.{finding_text}\n"
        f"Skill library context not available; skill was not installed. "
        f"Use 'leap hub pull {repo_id}' in CLI to install."
    )


async def hub_search_tool(
    query: str = "",
    federated: bool = False,
    **kwargs: Any,
) -> str:
    """Search for skills on the Hub. Returns formatted results.

    Args:
        query: Free-text search query.
        federated: If True, search across all registered backends in parallel.
    """
    from leapflow.config import get_settings
    from leapflow.hub import HubClient

    if not query:
        return "Error: query is required."

    settings = get_settings()
    client = HubClient(
        hub_type=settings.hub_type,
        default_owner=settings.hub_default_owner,
        default_visibility=settings.hub_default_visibility,
        repo_prefix=settings.hub_repo_prefix,
        search_sources=getattr(settings, "hub_search_sources", ""),
    )

    try:
        if federated:
            fed_result = await client.federated_search(query)
            results = list(fed_result.results)
            extra = ""
            if fed_result.failed_backends:
                extra = (
                    f"\n(backends failed: {', '.join(fed_result.failed_backends)})"
                )
        else:
            results = await client.search(query)
            extra = ""
    except Exception as e:
        return f"Search failed: {type(e).__name__}: {e}"

    if not results:
        return f"No skills found for '{query}'."

    mode_label = "federated " if federated else ""
    lines = [f"Found {len(results)} skill(s) via {mode_label}search for '{query}':"]
    for r in results:
        desc = f" \u2014 {r.description}" if r.description else ""
        hub_tag = f" [{r.hub_type}]" if r.hub_type else ""
        lines.append(f"  {r.repo_id} v{r.version}{hub_tag}{desc}")
    if extra:
        lines.append(extra)

    return "\n".join(lines)


async def hub_sync_tool(
    mode: str = "preview",
    dry_run: bool = True,
    **kwargs: Any,
) -> str:
    """Preview skill sync plan between local and remote Hub.

    Note: Actual sync execution is available via 'leap hub sync' CLI command.
    This tool provides a preview of what would be synchronized.

    Args:
        mode: 'full', 'push-only', or 'pull-only'.
        dry_run: If True, only shows the plan without executing.
    """
    from leapflow.config import get_settings
    from leapflow.hub import HubClient
    from leapflow.hub.protocol import SkillManifest

    settings = get_settings()
    client = HubClient(
        hub_type=settings.hub_type,
        default_owner=settings.hub_default_owner,
        default_visibility=settings.hub_default_visibility,
        repo_prefix=settings.hub_repo_prefix,
    )

    # Load actual local skills from context if available
    ctx = kwargs.get("ctx")
    local_manifests: list = []
    if ctx is not None and hasattr(ctx, "skill_lib") and ctx.skill_lib is not None:
        try:
            stored_skills = ctx.skill_lib.load_all_active()
            for s in stored_skills:
                local_manifests.append(SkillManifest(
                    name=getattr(s, "title", ""),
                    version=str(getattr(s, "version", "0.1.0")),
                ))
        except Exception:
            pass  # Fall through to empty manifests

    try:
        plan = await client.sync_skills(local_manifests)
    except Exception as e:
        return f"Sync failed: {type(e).__name__}: {e}"

    if plan.is_empty:
        return "Everything is in sync — no actions needed."

    lines = ["Sync Plan:"]
    if plan.to_push and mode != "pull-only":
        lines.append(f"\n  Push ({len(plan.to_push)}):")
        for m in plan.to_push:
            lines.append(f"    -> {m.name} v{m.version}")
    if plan.to_pull and mode != "push-only":
        lines.append(f"\n  Pull ({len(plan.to_pull)}):")
        for s in plan.to_pull:
            lines.append(f"    <- {s.name} v{s.version}")
    if plan.conflicts:
        lines.append(f"\n  Conflicts ({len(plan.conflicts)}):")
        for name in plan.conflicts:
            lines.append(f"    !! {name}")

    if dry_run:
        lines.append("\n(preview only — use 'leap hub sync' CLI command to execute actual sync)")

    return "\n".join(lines)


async def hub_federated_search_tool(
    query: str = "",
    **kwargs: Any,
) -> str:
    """Search for skills across ALL registered Hub backends in parallel.

    Queries every configured backend concurrently, deduplicates by skill name
    (preferring higher version / more downloads), and returns a unified listing.

    Args:
        query: Free-text search query.
    """
    return await hub_search_tool(query=query, federated=True, **kwargs)


# ─── Marketplace & Contribution Tool Implementations ─────────────────────────


async def hub_marketplace_browse_tool(
    category: str = "",
    featured: bool = False,
    trending: bool = False,
    limit: int = 10,
    **kwargs: Any,
) -> str:
    """Browse marketplace categories and featured/trending skills.

    Args:
        category: Filter by category name (empty = show categories overview).
        featured: If True, show only featured/trending entries.
        trending: If True, show top skills by installs/downloads.
        limit: Maximum number of results (default 10).
    """
    marketplace = _get_marketplace(**kwargs)
    if marketplace is None:
        return "Error: Hub client not available."

    if not category and not featured and not trending:
        # Show categories overview
        cats = marketplace.get_categories()
        lines = ["Skill Marketplace Categories:"]
        for c in cats:
            count = f" ({c.skill_count} skills)" if c.skill_count else ""
            lines.append(f"  {c.icon} {c.name}{count} — {c.description}")
        lines.append("\nUse category filter to browse skills in a specific category.")
        return "\n".join(lines)

    if featured:
        entries = marketplace.get_featured()
        label = "Featured & Trending"
    elif trending:
        entries = marketplace.get_trending(limit=limit)
        label = f"Top {limit} Trending"
    elif category:
        entries = marketplace.get_by_category(category)
        label = f"Category: {category}"
    else:
        entries = []
        label = "Browse"

    if not entries:
        return f"No skills found for {label}."

    lines = [f"{label} ({len(entries)} skill{'s' if len(entries) != 1 else ''}):\n"]
    for e in entries[:limit]:
        flags = []
        if e.featured:
            flags.append("★ featured")
        if e.trending:
            flags.append("🔥 trending")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        desc = f" — {e.description}" if e.description else ""
        lines.append(f"  {e.repo_id} v{e.version}{flag_str}{desc}")
    return "\n".join(lines)


async def hub_marketplace_install_tool(
    name: str = "",
    **kwargs: Any,
) -> str:
    """Install a skill from the marketplace.

    Args:
        name: Skill name or repo_id to install.
    """
    if not name:
        return "Error: name is required."

    marketplace = _get_marketplace(**kwargs)
    if marketplace is None:
        return "Error: Hub client not available."

    return await marketplace.install(name)


async def hub_contribute_tool(
    skill_name: str = "",
    action: str = "submit",
    hub_type: str = "github",
    **kwargs: Any,
) -> str:
    """Submit a skill to the community or check contribution status.

    Args:
        skill_name: Name of the local skill.
        action: 'prepare', 'submit', 'status', or 'list'.
        hub_type: Target hub backend for submission (default: 'github').
    """
    contributor = _get_contributor(**kwargs)
    if contributor is None:
        return "Error: Hub client not available."

    ctx = kwargs.get("ctx")

    if action == "list":
        records = contributor.list_my_contributions()
        if not records:
            return "No contributions found."
        lines = [f"Your Contributions ({len(records)}):"]
        for r in records:
            lines.append(f"  {r.skill_name} — {r.status} ({r.hub_type or 'local'})")
        return "\n".join(lines)

    if not skill_name:
        return "Error: skill_name is required."

    if action == "prepare":
        return await contributor.prepare(skill_name, ctx=ctx)
    elif action == "submit":
        return await contributor.submit(skill_name, hub_type=hub_type, ctx=ctx)
    elif action == "status":
        try:
            status = contributor.check_status(skill_name)
            return f"Contribution status for '{skill_name}': {status.value}"
        except KeyError as exc:
            return str(exc)
    else:
        return f"Unknown action '{action}'. Use 'prepare', 'submit', 'status', or 'list'."


def _get_marketplace(**kwargs: Any) -> Any:
    """Build a SkillMarketplace from runtime context."""
    from leapflow.config import get_settings
    from leapflow.hub import HubClient
    from leapflow.hub.marketplace import SkillMarketplace

    try:
        settings = get_settings()
        client = HubClient(
            hub_type=settings.hub_type,
            default_owner=settings.hub_default_owner,
            default_visibility=settings.hub_default_visibility,
            repo_prefix=settings.hub_repo_prefix,
            search_sources=getattr(settings, "hub_search_sources", ""),
        )
        profile_layout = getattr(settings, "profile_layout", None)
        cache_path = None
        if profile_layout is not None:
            cache_path = profile_layout.root / "marketplace_cache.json"
        return SkillMarketplace(client, cache_path=cache_path)
    except Exception as exc:
        logger.warning("Failed to create marketplace: %s", exc)
        return None


def _get_contributor(**kwargs: Any) -> Any:
    """Build a CommunityContributor from runtime context."""
    from leapflow.config import get_settings
    from leapflow.hub import HubClient
    from leapflow.hub.contribute import CommunityContributor

    try:
        settings = get_settings()
        client = HubClient(
            hub_type=settings.hub_type,
            default_owner=settings.hub_default_owner,
            default_visibility=settings.hub_default_visibility,
            repo_prefix=settings.hub_repo_prefix,
        )
        profile_layout = getattr(settings, "profile_layout", None)
        store_path = None
        if profile_layout is not None:
            store_path = profile_layout.root / "contributions.json"
        return CommunityContributor(client, store_path=store_path)
    except Exception as exc:
        logger.warning("Failed to create contributor: %s", exc)
        return None


# ─── Tool Definitions (OpenAI function calling schema) ───────────────────────


HUB_TOOL_DEFINITIONS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "hub_push",
            "description": "Push a local skill to the ModelScope Hub for sharing or backup.",
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "Name of the local skill to push",
                    },
                    "visibility": {
                        "type": "string",
                        "enum": ["private", "public", "internal"],
                        "description": "Repository visibility (default: private)",
                    },
                    "version": {
                        "type": "string",
                        "description": "Version string (default: auto-detect from skill)",
                    },
                },
                "required": ["skill_name"],
            },
            "x_leapflow": {
                "category": "hub",
                "risk_level": "medium",
                "schema_cost": "high",
                "requires_approval": True,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hub_pull",
            "description": "Pull a skill from the ModelScope Hub to install locally.",
            "parameters": {
                "type": "object",
                "properties": {
                    "repo_id": {
                        "type": "string",
                        "description": "Repository identifier (e.g. 'owner/leapflow-skill-name')",
                    },
                    "version": {
                        "type": "string",
                        "description": "Specific version to pull (default: latest)",
                    },
                },
                "required": ["repo_id"],
            },
            "x_leapflow": {
                "category": "hub",
                "risk_level": "medium",
                "schema_cost": "high",
                "requires_approval": True,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hub_search",
            "description": "Search for skills on the Hub by keyword or description.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Free-text search query for finding skills",
                    },
                    "federated": {
                        "type": "boolean",
                        "description": (
                            "If true, search across all registered backends "
                            "in parallel (default: false)"
                        ),
                    },
                },
                "required": ["query"],
            },
            "x_leapflow": {
                "category": "hub",
                "risk_level": "read_only",
                "schema_cost": "high",
                "requires_approval": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hub_federated_search",
            "description": (
                "Search for skills across ALL registered Hub backends in parallel. "
                "Queries every configured backend concurrently, deduplicates results, "
                "and returns a unified listing sorted by relevance."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Free-text search query for finding skills",
                    },
                },
                "required": ["query"],
            },
            "x_leapflow": {
                "category": "hub",
                "risk_level": "read_only",
                "schema_cost": "high",
                "requires_approval": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hub_sync",
            "description": "Preview or execute sync between local skills and Hub.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["full", "push-only", "pull-only"],
                        "description": "Sync mode (default: full)",
                    },
                    "dry_run": {
                        "type": "boolean",
                        "description": "If true, only shows the plan (default: true)",
                    },
                },
            },
            "x_leapflow": {
                "category": "hub",
                "risk_level": "medium",
                "schema_cost": "high",
                "requires_approval": True,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hub_marketplace_browse",
            "description": (
                "Browse the skill marketplace — view categories, featured skills, "
                "and trending skills. Supports category filtering."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": (
                            "Filter by category name "
                            "(development, research, automation, productivity, "
                            "security, operations, analysis, integration). "
                            "Empty shows overview."
                        ),
                    },
                    "featured": {
                        "type": "boolean",
                        "description": "Show only featured/trending entries (default: false)",
                    },
                    "trending": {
                        "type": "boolean",
                        "description": "Show top skills by installs/downloads (default: false)",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results to return (default: 10)",
                    },
                },
            },
            "x_leapflow": {
                "category": "hub",
                "risk_level": "read_only",
                "schema_cost": "high",
                "requires_approval": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hub_marketplace_install",
            "description": "Install a skill from the marketplace by name or repo_id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Skill name or repo_id to install",
                    },
                },
                "required": ["name"],
            },
            "x_leapflow": {
                "category": "hub",
                "risk_level": "medium",
                "schema_cost": "high",
                "requires_approval": True,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hub_contribute",
            "description": (
                "Submit a local skill to the community hub, check contribution "
                "status, or list all your contributions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": "Name of the local skill to contribute",
                    },
                    "action": {
                        "type": "string",
                        "enum": ["prepare", "submit", "status", "list"],
                        "description": "Contribution action (default: submit)",
                    },
                    "hub_type": {
                        "type": "string",
                        "description": "Target hub backend (default: github)",
                    },
                },
            },
            "x_leapflow": {
                "category": "hub",
                "risk_level": "medium",
                "schema_cost": "high",
                "requires_approval": True,
                "mutates_state": True,
            },
        },
    },
]


# ─── Bridge Registration Table ───────────────────────────────────────────────

HUB_BRIDGE_TOOLS = [
    {
        "name": "hub_push",
        "description": "Push a local skill to the Hub for sharing or backup.",
        "parameters": {
            "skill_name": "string (required) \u2014 name of the skill to push",
            "visibility": "string (optional) \u2014 'private' (default), 'public', or 'internal'",
            "version": "string (optional) \u2014 version override",
        },
        "handler": hub_push_tool,
        "mutates_state": True,
    },
    {
        "name": "hub_pull",
        "description": "Pull a skill from the Hub to install locally.",
        "parameters": {
            "repo_id": "string (required) \u2014 repository identifier",
            "version": "string (optional) \u2014 specific version to pull",
        },
        "handler": hub_pull_tool,
        "mutates_state": True,
    },
    {
        "name": "hub_search",
        "description": "Search for skills on the Hub by keyword.",
        "parameters": {
            "query": "string (required) \u2014 search query",
            "federated": "boolean (optional) \u2014 search all backends in parallel (default: false)",
        },
        "handler": hub_search_tool,
    },
    {
        "name": "hub_federated_search",
        "description": "Search for skills across ALL registered Hub backends in parallel.",
        "parameters": {
            "query": "string (required) \u2014 search query",
        },
        "handler": hub_federated_search_tool,
    },
    {
        "name": "hub_sync",
        "description": "Preview or execute skill sync between local and Hub.",
        "parameters": {
            "mode": "string (optional) \u2014 'full' (default), 'push-only', or 'pull-only'",
            "dry_run": "boolean (optional) \u2014 if true, only show plan (default: true)",
        },
        "handler": hub_sync_tool,
    },
    {
        "name": "hub_marketplace_browse",
        "description": "Browse marketplace categories, featured and trending skills.",
        "parameters": {
            "category": "string (optional) — filter by category name",
            "featured": "boolean (optional) — show featured/trending only (default: false)",
            "trending": "boolean (optional) — show top skills by installs (default: false)",
            "limit": "integer (optional) — max results (default: 10)",
        },
        "handler": hub_marketplace_browse_tool,
    },
    {
        "name": "hub_marketplace_install",
        "description": "Install a skill from the marketplace.",
        "parameters": {
            "name": "string (required) — skill name or repo_id",
        },
        "handler": hub_marketplace_install_tool,
        "mutates_state": True,
    },
    {
        "name": "hub_contribute",
        "description": "Submit a skill to the community or check contribution status.",
        "parameters": {
            "skill_name": "string — name of the skill",
            "action": "string (optional) — 'prepare', 'submit', 'status', or 'list'",
            "hub_type": "string (optional) — target hub backend (default: 'github')",
        },
        "handler": hub_contribute_tool,
        "mutates_state": True,
    },
]


# ─── Handler Map (for TOOL_HANDLERS integration) ─────────────────────────────

HUB_TOOL_HANDLERS: Dict[str, Any] = {t["name"]: t["handler"] for t in HUB_BRIDGE_TOOLS}
