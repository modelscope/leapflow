"""Skill discovery tools — exposed to LLM for progressive disclosure.

Provides two tool handlers registered into the unified tool system:
- skills_list: Layer 1 — compact metadata listing with optional keyword filter
- skill_view: Layer 2 — full SKILL.md content for a specific skill

Module-level configuration pattern: call configure() at startup to inject
SkillIndex and SkillInjector instances without coupling to DI framework.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Module-level references (set by bootstrap via configure())
_skill_index: Optional[Any] = None
_skill_injector: Optional[Any] = None
_skill_registry: Optional[Any] = None
_skill_view_max_chars: int = 5000  # Configurable cap for skill_view content


def configure(index: Any, injector: Any, *, registry: Any = None, skill_view_max_chars: int = 5000) -> None:
    """Configure module with SkillIndex and SkillInjector instances.

    Called once during CLI context initialization. Both arguments must
    implement the public APIs defined in leapflow.skills.index and
    leapflow.skills.injector respectively.
    """
    global _skill_index, _skill_injector, _skill_registry, _skill_view_max_chars
    _skill_index = index
    _skill_injector = injector
    _skill_registry = registry
    _skill_view_max_chars = skill_view_max_chars
    logger.debug("skill_discovery.configured index=%s injector=%s registry=%s max_chars=%d", type(index).__name__, type(injector).__name__, type(registry).__name__ if registry else 'None', skill_view_max_chars)


# ------------------------------------------------------------------
# Tool handlers (conform to async handler(params: Dict) -> Dict protocol)
# ------------------------------------------------------------------


async def skills_list(params: Dict[str, Any]) -> Dict[str, Any]:
    """Layer 1: List available skills with compact metadata.

    Params:
        query (str, optional): Keyword filter applied to name and description.
        category (str, optional): Filter by category.
        source (str, optional): Filter by source ("learned"|"manual"|"hub").

    Returns dict with ok, count, skills[] fields.
    """
    if _skill_index is None:
        return {"ok": False, "error": "Skill index not initialized"}

    query = str(params.get("query", "")).strip()
    category_filter = str(params.get("category", "")).strip()
    source_filter = str(params.get("source", "")).strip()
    entries = _skill_index.get_entries()

    # Optional keyword filter (case-insensitive substring match)
    if query:
        query_lower = query.lower()
        entries = [
            e for e in entries
            if query_lower in e.name.lower() or query_lower in e.description.lower()
        ]

    # Category filter
    if category_filter:
        cat_lower = category_filter.lower()
        entries = [e for e in entries if e.category.lower() == cat_lower]

    # Source filter
    if source_filter:
        src_lower = source_filter.lower()
        entries = [e for e in entries if e.source.lower() == src_lower]

    skills = [
        {
            "name": e.name,
            "description": e.description,
            "tags": list(e.tags),
            "category": e.category,
            "source": e.source,
        }
        for e in entries
    ]

    # Include builtin skills from registry (always available)
    if _skill_registry and hasattr(_skill_registry, 'list_all'):
        for s in _skill_registry.list_all():
            reg_entry = {
                "name": s.name,
                "description": getattr(s, 'description', s.name),
                "tags": [],
                "category": "builtin",
                "source": "builtin",
            }
            # Apply filters
            if query and query.lower() not in reg_entry["name"].lower() and query.lower() not in reg_entry["description"].lower():
                continue
            if source_filter and source_filter.lower() != "builtin":
                continue
            if category_filter and category_filter.lower() != "builtin":
                continue
            skills.append(reg_entry)

    return {"ok": True, "count": len(skills), "skills": skills}


async def skill_view(params: Dict[str, Any]) -> Dict[str, Any]:
    """Layer 2: View full SKILL.md content for a specific skill.

    Params:
        name (str, required): Skill name to look up.

    Returns dict with ok, name, content, path fields.
    """
    if _skill_injector is None:
        return {"ok": False, "error": "Skill injector not initialized"}

    name = str(params.get("name", "")).strip()
    if not name:
        return {"ok": False, "error": "Skill name required"}

    skill_dir = _skill_injector.find_skill_dir(name)
    if skill_dir is None:
        return {"ok": False, "error": f"Skill '{name}' not found"}

    skill_md = skill_dir / "SKILL.md"
    if not skill_md.exists():
        return {"ok": False, "error": f"SKILL.md not found in {skill_dir}"}

    content = skill_md.read_text(errors="replace")
    # Cap content to prevent oversized tool results (configurable)
    max_chars = _skill_view_max_chars
    truncated = len(content) > max_chars
    content_out = content[:max_chars]

    result: Dict[str, Any] = {
        "ok": True,
        "name": name,
        "content": content_out,
        "path": str(skill_dir),
    }
    if truncated:
        result["truncated"] = True
        result["total_chars"] = len(content)

    return result
