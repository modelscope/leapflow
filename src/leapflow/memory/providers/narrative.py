"""Narrative memory provider — pure-text Markdown for LLM-readable knowledge.

Implements the narrative layer of the dual memory architecture:

- **Narrative (this)**: Low-frequency, stable, LLM-readable text (MEMORY.md files)
- **Signal (SemanticMemoryProvider)**: High-frequency DuckDB signal/analytical data

Storage layout (inside ``profiles/<profile>/memory/``)::

    global/
        MEMORY.md            # Global index (always loaded)
        topics/<slug>.md     # Detail files (lazy-loaded on search)
    workspaces/<hash>/
        MEMORY.md            # Project-specific index
        topics/<slug>.md

Token budget control:

1. **Index always loaded** — ``MEMORY.md`` from global + current workspace
2. **Topics lazy-loaded** — ``topics/<slug>.md`` fetched on keyword match
3. **Conditional** — workspace scoping by cwd git root hash

Access pattern: client-direct read/write (advisory file lock). No daemon needed.
"""
from __future__ import annotations

import fcntl
import hashlib
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from leapflow.memory.protocol import (
    MemoryEntry,
    MemoryKind,
    MemoryQuery,
    MemoryToolSchema,
)

logger = logging.getLogger(__name__)

_NARRATIVE_KINDS = frozenset({
    MemoryKind.USER_PREFERENCE,
    MemoryKind.FACT,
    MemoryKind.SESSION_SUMMARY,
})

_KIND_SECTION = {
    MemoryKind.USER_PREFERENCE: "User Preferences",
    MemoryKind.FACT: "Facts",
    MemoryKind.SESSION_SUMMARY: "Session Summaries",
}

_MEMORY_HEADER = "# Memory\n\n"


def _slug(text: str) -> str:
    """Convert text to a filesystem-safe slug."""
    s = re.sub(r"[^\w\s-]", "", text.lower().strip())
    return re.sub(r"[\s_]+", "-", s)[:60] or "untitled"


def _workspace_hash(workspace_path: str) -> str:
    """Deterministic short hash for workspace directory isolation."""
    return hashlib.sha256(workspace_path.encode()).hexdigest()[:12]


def _atomic_write(path: Path, content: str) -> None:
    """Write with advisory file lock for multi-process safety."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(path)
    except OSError:
        tmp.unlink(missing_ok=True)
        raise


def _read_text(path: Path) -> str:
    """Read file content, returning empty string if missing."""
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


class NarrativeProvider:
    """Pure-text Markdown memory for LLM-readable knowledge.

    Accepts ``USER_PREFERENCE``, ``FACT``, and ``SESSION_SUMMARY``
    memory kinds. Entries are appended as bullet points to the
    appropriate section in ``MEMORY.md``.
    """

    def __init__(
        self,
        memory_dir: Path,
        *,
        workspace_path: Optional[str] = None,
    ) -> None:
        self._memory_dir = memory_dir
        self._workspace_path = workspace_path
        self._workspace_hash = _workspace_hash(workspace_path) if workspace_path else None

    @property
    def name(self) -> str:
        return "narrative"

    @property
    def _global_dir(self) -> Path:
        return self._memory_dir / "global"

    @property
    def _workspace_dir(self) -> Optional[Path]:
        if self._workspace_hash is None:
            return None
        return self._memory_dir / "workspaces" / self._workspace_hash

    async def initialize(self, **kwargs: Any) -> None:
        self._global_dir.mkdir(parents=True, exist_ok=True)
        index = self._global_dir / "MEMORY.md"
        if not index.exists():
            _atomic_write(index, _MEMORY_HEADER)
        (self._global_dir / "topics").mkdir(exist_ok=True)

        if self._workspace_dir is not None:
            self._workspace_dir.mkdir(parents=True, exist_ok=True)
            ws_index = self._workspace_dir / "MEMORY.md"
            if not ws_index.exists():
                _atomic_write(ws_index, _MEMORY_HEADER)
            (self._workspace_dir / "topics").mkdir(exist_ok=True)

    async def shutdown(self) -> None:
        pass

    def accepts(self, entry: MemoryEntry) -> bool:
        return entry.kind in _NARRATIVE_KINDS

    async def insert(self, entry: MemoryEntry) -> str:
        """Append entry as a bullet point to the appropriate MEMORY.md section."""
        section_name = _KIND_SECTION.get(entry.kind, "Notes")
        scope = entry.metadata.get("scope", "global")
        target_dir = (
            self._workspace_dir if scope == "workspace" and self._workspace_dir
            else self._global_dir
        )
        index_path = target_dir / "MEMORY.md"
        content = _read_text(index_path) or _MEMORY_HEADER

        section_header = f"## {section_name}"
        bullet = f"- {entry.content.strip()}"

        if section_header in content:
            # Avoid duplicates
            if bullet in content:
                return entry.entry_id
            # Append to existing section (before next ## or EOF)
            parts = content.split(section_header, 1)
            after = parts[1]
            next_section = after.find("\n## ")
            if next_section == -1:
                content = parts[0] + section_header + after.rstrip() + "\n" + bullet + "\n"
            else:
                before_next = after[:next_section].rstrip()
                rest = after[next_section:]
                content = parts[0] + section_header + before_next + "\n" + bullet + "\n" + rest
        else:
            content = content.rstrip() + "\n\n" + section_header + "\n" + bullet + "\n"

        _atomic_write(index_path, content)
        logger.debug("narrative: inserted %s into %s/%s", entry.entry_id, scope, section_name)

        # Long entries also get a topic file for detail
        if len(entry.content) > 200:
            slug = _slug(entry.content[:50])
            topic_path = target_dir / "topics" / f"{slug}.md"
            if not topic_path.exists():
                topic_content = f"# {entry.content[:80]}\n\n{entry.content}\n"
                _atomic_write(topic_path, topic_content)

        return entry.entry_id

    async def search(self, query: MemoryQuery) -> List[MemoryEntry]:
        """Keyword search across MEMORY.md files and topic files."""
        if not query.keywords:
            return []

        results: List[MemoryEntry] = []
        seen_content: set = set()

        # Search index files first (higher priority)
        for scope, directory in self._iter_scopes():
            index_path = directory / "MEMORY.md"
            index_text = _read_text(index_path)
            if index_text:
                self._search_text(
                    index_text, query.keywords, results, seen_content,
                    source=f"{scope}/MEMORY.md",
                )

            # Search topic files (lazy load)
            topics_dir = directory / "topics"
            if topics_dir.exists():
                for topic_file in sorted(topics_dir.glob("*.md")):
                    topic_text = _read_text(topic_file)
                    if topic_text:
                        self._search_text(
                            topic_text, query.keywords, results, seen_content,
                            source=f"{scope}/topics/{topic_file.name}",
                        )

        results.sort(key=lambda e: e.score, reverse=True)
        return results[:query.limit]

    async def delete(self, entry_id: str) -> bool:
        return False

    def context_block(self) -> str:
        """Return text for system prompt injection.

        Loads the always-on index files (global + workspace MEMORY.md).
        This is called at session start to provide persistent context.
        """
        parts: List[str] = []

        global_index = _read_text(self._global_dir / "MEMORY.md")
        if global_index.strip() and global_index.strip() != _MEMORY_HEADER.strip():
            parts.append(global_index.strip())

        if self._workspace_dir is not None:
            ws_index = _read_text(self._workspace_dir / "MEMORY.md")
            if ws_index.strip() and ws_index.strip() != _MEMORY_HEADER.strip():
                parts.append(f"## Project Memory\n{ws_index.strip()}")

        return "\n\n".join(parts)

    # ── Lifecycle hooks (no-op for narrative) ──

    def on_turn_start(self, turn: int, user_message: str) -> None:
        pass

    def on_promoted(self, entry: MemoryEntry, source_provider: str) -> None:
        pass

    def on_inserted(self, entry: MemoryEntry) -> None:
        pass

    def on_accessed(self, entry: MemoryEntry) -> None:
        pass

    # ── Tool interface ──

    def get_tool_schemas(self) -> List[MemoryToolSchema]:
        return [
            MemoryToolSchema(
                name="narrative_read",
                description="Read persistent narrative memory (user preferences, facts, project notes).",
                parameters={
                    "type": "object",
                    "properties": {
                        "scope": {
                            "type": "string",
                            "enum": ["global", "workspace"],
                            "description": "Memory scope to read",
                        },
                    },
                },
                provider_name="narrative",
            ),
            MemoryToolSchema(
                name="narrative_write",
                description="Write to persistent narrative memory (user preferences, facts, lessons learned).",
                parameters={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string", "description": "What to remember"},
                        "kind": {
                            "type": "string",
                            "enum": ["user_pref", "fact", "session_summary"],
                        },
                        "scope": {
                            "type": "string",
                            "enum": ["global", "workspace"],
                            "default": "global",
                        },
                    },
                    "required": ["content"],
                },
                provider_name="narrative",
            ),
        ]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any]) -> str:
        import asyncio
        import json

        if tool_name == "narrative_read":
            scope = args.get("scope", "global")
            directory = (
                self._workspace_dir if scope == "workspace" and self._workspace_dir
                else self._global_dir
            )
            text = _read_text(directory / "MEMORY.md")
            return json.dumps({"content": text or "(empty)"}, ensure_ascii=False)

        if tool_name == "narrative_write":
            content = args.get("content", "")
            kind_str = args.get("kind", "fact")
            scope = args.get("scope", "global")
            try:
                kind = MemoryKind(kind_str)
            except ValueError:
                kind = MemoryKind.FACT

            entry = MemoryEntry(
                kind=kind,
                content=content[:2000],
                metadata={"scope": scope},
            )
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self.insert(entry))
            except RuntimeError:
                asyncio.run(self.insert(entry))
            return json.dumps({"success": True, "id": entry.entry_id})

        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    # ── Internal ──

    def _iter_scopes(self):
        """Yield (scope_name, directory) pairs for search iteration."""
        yield "global", self._global_dir
        if self._workspace_dir is not None and self._workspace_dir.exists():
            yield "workspace", self._workspace_dir

    def _search_text(
        self,
        text: str,
        keywords: List[str],
        results: List[MemoryEntry],
        seen: set,
        *,
        source: str,
    ) -> None:
        """Extract matching lines/paragraphs from text content."""
        text_lower = text.lower()
        matched_keywords = [kw for kw in keywords if kw.lower() in text_lower]
        if not matched_keywords:
            return

        score = len(matched_keywords) / max(len(keywords), 1)

        for line in text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            # Remove markdown bullet prefix
            clean = stripped.lstrip("- ").strip()
            if not clean or clean in seen:
                continue

            line_lower = clean.lower()
            if any(kw.lower() in line_lower for kw in matched_keywords):
                seen.add(clean)
                results.append(MemoryEntry(
                    entry_id=uuid.uuid4().hex[:12],
                    kind=MemoryKind.FACT,
                    content=clean,
                    score=score,
                    metadata={"source": source},
                ))
