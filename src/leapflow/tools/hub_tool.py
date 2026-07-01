"""Hub operations as an Agent Tool — enables natural language hub interaction.

Registered as agent-callable tools so the AgentEngine can push, pull, search,
and sync skills with the configured Hub backend during chat-mode conversations.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List

if TYPE_CHECKING:
    from leapflow.cli.context import Context

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
    **kwargs: Any,
) -> str:
    """Search for skills on the Hub. Returns formatted results.

    Args:
        query: Free-text search query.
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
    )

    try:
        results = await client.search(query)
    except Exception as e:
        return f"Search failed: {type(e).__name__}: {e}"

    if not results:
        return f"No skills found for '{query}'."

    lines = [f"Found {len(results)} skill(s) for '{query}':\n"]
    for r in results:
        desc = f" — {r.description}" if r.description else ""
        lines.append(f"  {r.repo_id} v{r.version}{desc}")

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
                },
                "required": ["query"],
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
        },
    },
]


# ─── Bridge Registration Table ───────────────────────────────────────────────

HUB_BRIDGE_TOOLS = [
    {
        "name": "hub_push",
        "description": "Push a local skill to the Hub for sharing or backup.",
        "parameters": {
            "skill_name": "string (required) — name of the skill to push",
            "visibility": "string (optional) — 'private' (default), 'public', or 'internal'",
            "version": "string (optional) — version override",
        },
        "handler": hub_push_tool,
        "mutates_state": True,
    },
    {
        "name": "hub_pull",
        "description": "Pull a skill from the Hub to install locally.",
        "parameters": {
            "repo_id": "string (required) — repository identifier",
            "version": "string (optional) — specific version to pull",
        },
        "handler": hub_pull_tool,
        "mutates_state": True,
    },
    {
        "name": "hub_search",
        "description": "Search for skills on the Hub by keyword.",
        "parameters": {
            "query": "string (required) — search query",
        },
        "handler": hub_search_tool,
    },
    {
        "name": "hub_sync",
        "description": "Preview or execute skill sync between local and Hub.",
        "parameters": {
            "mode": "string (optional) — 'full' (default), 'push-only', or 'pull-only'",
            "dry_run": "boolean (optional) — if true, only show plan (default: true)",
        },
        "handler": hub_sync_tool,
    },
]


# ─── Handler Map (for TOOL_HANDLERS integration) ─────────────────────────────

HUB_TOOL_HANDLERS: Dict[str, Any] = {t["name"]: t["handler"] for t in HUB_BRIDGE_TOOLS}
