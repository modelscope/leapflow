"""Skill content injection as user message (Hermes pattern).

Protects system prompt cache by injecting SKILL.md content into
user messages rather than modifying the system prompt.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# Injection markers (LLM recognizes these as skill context)
_SKILL_MARKER = (
    "[IMPORTANT: The user has invoked skill '{name}'. "
    "Follow the instructions below.]"
)
_SKILL_END_MARKER = "[END OF SKILL DOCUMENT]"


class SkillInjector:
    """Injects SKILL.md content as user message (protects prompt cache).

    5-phase injection layout:
    1. Activation marker
    2. Full SKILL.md content
    3. Skill directory listing (support files)
    4. Config values (extensible)
    5. User instruction
    """

    def __init__(self, skills_dir: Path):
        self._skills_dir = Path(skills_dir).expanduser()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_injection_message(
        self, skill_name: str, user_instruction: str = ""
    ) -> Optional[str]:
        """Build injection message with 5 phases.

        Returns None if skill not found.
        """
        skill_dir = self.find_skill_dir(skill_name)
        if skill_dir is None:
            return None

        skill_md = skill_dir / "SKILL.md"
        if not skill_md.exists():
            return None

        content = skill_md.read_text(errors="replace")

        # Build 5-phase injection
        parts: List[str] = []

        # Phase 1: Activation marker
        parts.append(_SKILL_MARKER.format(name=skill_name))
        parts.append("")

        # Phase 2: Full SKILL.md content
        parts.append(content)
        parts.append("")

        # Phase 3: Skill directory listing
        support_files = self._discover_support_files(skill_dir)
        if support_files:
            parts.append(f"Skill directory: {skill_dir}")
            parts.append("Support files: " + ", ".join(support_files))
            parts.append("")

        # Phase 4: Config values (extensible hook — no-op for now)

        # Phase 5: User instruction
        parts.append(_SKILL_END_MARKER)
        if user_instruction:
            parts.append("")
            parts.append(f"User instruction: {user_instruction}")

        return "\n".join(parts)

    def find_skill_dir(self, name: str) -> Optional[Path]:
        """Resolve skill name -> directory path.

        Attempts direct match first, then normalized (hyphen-based) lookup.
        """
        if not self._skills_dir.exists():
            return None

        # Direct match
        direct = self._skills_dir / name
        if direct.is_dir() and (direct / "SKILL.md").exists():
            return direct

        # Normalized: replace spaces/underscores with hyphens
        normalized = name.lower().replace(" ", "-").replace("_", "-")
        for d in self._skills_dir.iterdir():
            if not d.is_dir():
                continue
            if d.name.lower().replace("_", "-") == normalized:
                if (d / "SKILL.md").exists():
                    return d

        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _discover_support_files(
        self, skill_dir: Path, *, limit: int = 10
    ) -> List[str]:
        """Find support files in skill directory (excludes SKILL.md)."""
        support: List[str] = []
        for item in sorted(skill_dir.iterdir()):
            if item.name == "SKILL.md" or item.name.startswith("."):
                continue
            if item.is_file():
                support.append(item.name)
            elif item.is_dir():
                support.append(f"{item.name}/")
            if len(support) >= limit:
                break
        return support
