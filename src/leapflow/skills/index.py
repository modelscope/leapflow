# Copyright (c) Alibaba, Inc. and its affiliates.
"""Skill index with three-layer caching and conditional filtering.

Hermes-inspired design: skills are discovered from SKILL.md files,
indexed by frontmatter metadata, and filtered by runtime conditions.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Package-bundled builtin SKILL.md skills — always scanned in addition to
# the user-profile skills_dir so they ship with the installed package.
_BUILTIN_SKILLS_DIR = Path(__file__).resolve().parent / "builtin_skills"

# Bumped when the snapshot layout or signing scheme changes; a mismatch forces
# a fresh L3 rescan so an old on-disk snapshot can never mask new skills.
_SNAPSHOT_SCHEMA = 2


@dataclass(frozen=True)
class SkillEntry:
    """Compact skill metadata parsed from SKILL.md frontmatter."""

    name: str
    description: str
    category: str = ""
    source: str = "manual"  # "learned" | "manual" | "hub"
    confidence: float = 1.0
    quality_score: float = 1.0
    tags: Tuple[str, ...] = ()
    requires_tools: Tuple[str, ...] = ()
    platforms: Tuple[str, ...] = ()
    skill_dir: str = ""


class SkillIndex:
    """Three-layer cached skill index with conditional filtering.

    Cache layers:
    - L1: In-memory (instant, invalidated on explicit call)
    - L2: Disk snapshot (.skills_index.json, ~50ms load)
    - L3: Full directory scan (~300ms, parses all SKILL.md files)
    """

    def __init__(
        self,
        skills_dir: Path,
        *,
        min_quality: float = 0.5,
    ):
        self._skills_dir = Path(skills_dir).expanduser()
        self._min_quality = min_quality
        self._entries: Optional[List[SkillEntry]] = None
        self._cache_time: float = 0
        self._snapshot_path = self._skills_dir / ".skills_index.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_entries(
        self,
        *,
        platform: Optional[str] = None,
        available_tools: Optional[Set[str]] = None,
        disabled: Optional[Set[str]] = None,
        archived: Optional[Set[str]] = None,
        include_archived: bool = False,
    ) -> List[SkillEntry]:
        """Get filtered skill entries (L1 -> L2 -> L3).

        Args:
            archived: Set of archived skill names (from SkillCurator).
                      Excluded from results unless *include_archived* is True.
            include_archived: If True, archived skills are included in results.
        """
        entries = self._load_entries()
        return self._apply_filters(
            entries, platform, available_tools, disabled,
            archived=archived, include_archived=include_archived,
        )

    def get_entry(self, name: str) -> Optional[SkillEntry]:
        """Get single entry by exact name."""
        for entry in self._load_entries():
            if entry.name == name:
                return entry
        return None

    def invalidate(self) -> None:
        """Clear all cache layers."""
        self._entries = None
        self._cache_time = 0
        if self._snapshot_path.exists():
            self._snapshot_path.unlink(missing_ok=True)
        logger.debug("skill_index.invalidated")

    def compact_index_text(self, entries: Optional[List[SkillEntry]] = None) -> str:
        """Generate compact index for system prompt (~100 bytes/skill)."""
        if entries is None:
            entries = self.get_entries()
        if not entries:
            return "(no skills available)"
        lines: List[str] = []
        for e in entries:
            tags_str = f" [{', '.join(e.tags[:3])}]" if e.tags else ""
            lines.append(f"- {e.name}: {e.description[:80]}{tags_str}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Cache hierarchy
    # ------------------------------------------------------------------

    def _load_entries(self) -> List[SkillEntry]:
        """Load entries using cache hierarchy: L1 -> L2 -> L3."""
        # L1: Memory cache
        if self._entries is not None:
            return self._entries

        # L2: Disk snapshot
        entries = self._load_from_snapshot()
        if entries is not None:
            self._entries = entries
            self._cache_time = time.monotonic()
            return entries

        # L3: Full scan
        entries = self._scan_skills_dir()
        self._entries = entries
        self._cache_time = time.monotonic()
        self._save_snapshot(entries)
        return entries

    def _scan_skills_dir(self) -> List[SkillEntry]:
        """L3: Full directory scan, parse all SKILL.md files.

        Scans both the package-bundled builtin_skills directory and the
        user-profile skills_dir.  Builtin entries are scanned first;
        user entries with the same name silently win (last-write-wins
        in the name→entry mapping) so users can override builtins.
        """
        seen_names: Dict[str, SkillEntry] = {}

        # 1) Package-bundled builtin skills
        for entry in self._scan_single_dir(_BUILTIN_SKILLS_DIR):
            seen_names[entry.name] = entry

        # 2) User-profile skills — may override builtins
        for entry in self._scan_single_dir(self._skills_dir):
            seen_names[entry.name] = entry

        # 3) MCP-bridged skills (auto-generated under _mcp_skills/)
        mcp_skills_dir = self._skills_dir / "_mcp_skills"
        for entry in self._scan_single_dir(mcp_skills_dir):
            seen_names[entry.name] = entry

        entries = list(seen_names.values())
        logger.info(
            "skill_index.scanned count=%d user_dir=%s builtin_dir=%s",
            len(entries), self._skills_dir, _BUILTIN_SKILLS_DIR,
        )
        return entries

    def _scan_single_dir(self, directory: Path) -> List[SkillEntry]:
        """Scan one directory for SKILL.md subdirectories."""
        entries: List[SkillEntry] = []
        if not directory.exists():
            return entries
        for skill_dir in sorted(directory.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue
            entry = self._parse_skill_md(skill_md, skill_dir)
            if entry is not None:
                entries.append(entry)
        return entries

    def _parse_skill_md(self, path: Path, skill_dir: Path) -> Optional[SkillEntry]:
        """Parse SKILL.md frontmatter YAML into SkillEntry."""
        try:
            content = path.read_text(encoding="utf-8", errors="replace")

            # No frontmatter — use directory name and first heading
            if not content.startswith("---"):
                name = skill_dir.name
                desc = self._extract_first_heading(content) or name
                return SkillEntry(
                    name=name, description=desc, skill_dir=str(skill_dir)
                )

            parts = content.split("---", 2)
            if len(parts) < 3:
                return SkillEntry(
                    name=skill_dir.name,
                    description=skill_dir.name,
                    skill_dir=str(skill_dir),
                )

            # Parse YAML frontmatter
            import yaml  # noqa: PLC0415

            try:
                fm: Dict[str, Any] = yaml.safe_load(parts[1]) or {}
            except Exception:
                fm = {}

            metadata = fm.get("metadata", {})
            hermes_meta = metadata.get("hermes", {}) if isinstance(metadata, dict) else {}
            leapflow_meta = (
                metadata.get("leapflow", {}) if isinstance(metadata, dict) else {}
            )

            return SkillEntry(
                name=fm.get("name", skill_dir.name),
                description=fm.get("description", ""),
                category=hermes_meta.get("category", ""),
                source=leapflow_meta.get("source", "manual"),
                confidence=float(leapflow_meta.get("confidence", 1.0)),
                quality_score=float(leapflow_meta.get("quality_score", 1.0)),
                tags=tuple(hermes_meta.get("tags", [])),
                requires_tools=tuple(hermes_meta.get("requires_tools", [])),
                platforms=tuple(fm.get("platforms", [])),
                skill_dir=str(skill_dir),
            )
        except Exception as exc:
            logger.debug("skill_index.parse_failed path=%s error=%s", path, exc)
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_first_heading(content: str) -> Optional[str]:
        """Extract first markdown heading as fallback description."""
        for line in content.splitlines():
            if line.startswith("# "):
                return line[2:].strip()
        return None

    def _compute_source_signature(self) -> str:
        """Fingerprint the scanned skill sources so a stale snapshot is detected.

        Covers add / remove / edit of any SKILL.md across the package builtin_skills
        dir, the user skills_dir, and the MCP-bridged dir — the exact sources
        ``_scan_skills_dir`` reads — using only ``stat()`` (no file reads), so it
        stays cheap on the L2 hot path. Without it, an ``.skills_index.json``
        written while a source dir was empty (e.g. before builtin skills shipped,
        or before a package upgrade) is trusted forever and silently hides every
        skill added since.
        """
        parts: List[str] = []
        for label, directory in (
            ("builtin", _BUILTIN_SKILLS_DIR),
            ("user", self._skills_dir),
            ("mcp", self._skills_dir / "_mcp_skills"),
        ):
            if not directory.exists():
                continue
            for skill_dir in sorted(directory.iterdir()):
                skill_md = skill_dir / "SKILL.md"
                try:
                    st = skill_md.stat()
                except OSError:
                    continue
                parts.append(f"{label}/{skill_dir.name}:{st.st_mtime_ns}:{st.st_size}")
        digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
        return f"{_SNAPSHOT_SCHEMA}:{digest}"

    def _load_from_snapshot(self) -> Optional[List[SkillEntry]]:
        """L2: Load from disk snapshot, rejecting a legacy or stale one.

        Returning ``None`` forces an L3 rescan that rewrites a fresh, signed
        snapshot, so this path is self-healing across package upgrades and after
        skills are added, edited, or removed.
        """
        if not self._snapshot_path.exists():
            return None
        try:
            data = json.loads(self._snapshot_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.debug("skill_index.snapshot_load_failed path=%s error=%s", self._snapshot_path, exc)
            return None
        # Legacy bare-list snapshots carry no signature — treat as stale and rescan.
        if not isinstance(data, dict) or data.get("schema") != _SNAPSHOT_SCHEMA:
            logger.info("skill_index.snapshot_legacy_or_schema_drift; rescanning path=%s", self._snapshot_path)
            return None
        if data.get("signature") != self._compute_source_signature():
            logger.info("skill_index.snapshot_stale (skill sources changed); rescanning path=%s", self._snapshot_path)
            return None
        try:
            entries: List[SkillEntry] = []
            for raw in data.get("entries", []):
                raw["tags"] = tuple(raw.get("tags", ()))
                raw["requires_tools"] = tuple(raw.get("requires_tools", ()))
                raw["platforms"] = tuple(raw.get("platforms", ()))
                entries.append(SkillEntry(**raw))
            return entries
        except Exception as exc:
            logger.debug("skill_index.snapshot_parse_failed path=%s error=%s", self._snapshot_path, exc)
            return None

    def _save_snapshot(self, entries: List[SkillEntry]) -> None:
        """Save a signed snapshot for the L2 cache."""
        try:
            self._skills_dir.mkdir(parents=True, exist_ok=True)
            payload = {
                "schema": _SNAPSHOT_SCHEMA,
                "signature": self._compute_source_signature(),
                "entries": [dataclasses.asdict(e) for e in entries],
            }
            self._snapshot_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass  # Non-critical — next scan will rebuild

    def _apply_filters(
        self,
        entries: List[SkillEntry],
        platform: Optional[str],
        available_tools: Optional[Set[str]],
        disabled: Optional[Set[str]],
        *,
        archived: Optional[Set[str]] = None,
        include_archived: bool = False,
    ) -> List[SkillEntry]:
        """Conditional filtering (Hermes-style) with curation awareness."""
        result: List[SkillEntry] = []
        for entry in entries:
            if disabled and entry.name in disabled:
                continue
            # Curation: exclude archived skills unless explicitly requested
            if not include_archived and archived and entry.name in archived:
                continue
            if platform and entry.platforms and platform not in entry.platforms:
                continue
            if available_tools and entry.requires_tools:
                if not all(t in available_tools for t in entry.requires_tools):
                    continue
            if entry.quality_score < self._min_quality:
                continue
            result.append(entry)
        return result
