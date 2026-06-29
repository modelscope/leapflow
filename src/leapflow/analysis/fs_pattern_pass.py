"""File system event pattern recognition pass.

Identifies high-level file operation patterns from low-level FS events,
improving skill quality when the visual/video signal is unavailable.

Recognized patterns (extensible via PATTERN_MATCHERS):
    organize_files_to_folder : file.create(dir) + N x file.move(into dir)
    batch_delete             : N consecutive file.delete actions
    rename_file              : file.delete + file.create (same dir, same ext)
    create_document          : file.create + file.modify (same path)

Design constraints:
    - Pure-functional: depends only on the input action list, no external state.
    - Configurable: matcher list is data-driven; new patterns plug in without
      touching the dispatch loop.
    - Structural typing: only relies on the SemanticAction-shaped Protocol so
      it can be unit-tested with plain stubs.
    - Context-preserving: parameters from the matched window are carried into
      the synthesized high-level action.
"""

from __future__ import annotations

import logging
import os.path
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

from leapflow.analysis.abstractor import AbstractionPass
from leapflow.domain.trajectory import SemanticAction, TrajectoryStep

logger = logging.getLogger(__name__)


# ── Structural protocol ────────────────────────────────────────────────────


class SemanticActionLike(Protocol):
    """Minimal shape consumed by the pattern matchers.

    Declared as a Protocol so unit tests can use lightweight stand-ins
    without depending on the concrete ``SemanticAction`` dataclass.
    """

    action_name: str
    description: str
    parameters: Dict[str, Any]
    raw_action_range: Tuple[int, int]
    confidence: float


# ── Pattern data model ────────────────────────────────────────────────────


@dataclass(frozen=True)
class FSPattern:
    """A recognized file system operation pattern."""

    name: str                       # e.g. "organize_files_to_folder"
    description: str                # Human-readable description
    parameters: Dict[str, Any] = field(default_factory=dict)
    source_indices: Tuple[int, ...] = field(default_factory=tuple)


# Matcher signature: (actions, start_idx) -> Optional[FSPattern]
PatternMatcher = Callable[[Sequence[SemanticActionLike], int], Optional[FSPattern]]


# ── Path helpers ──────────────────────────────────────────────────────────


def _path_of(action: SemanticActionLike) -> str:
    """Best-effort path extraction from a semantic action's parameters."""
    p = action.parameters
    return p.get("path") or p.get("target") or p.get("destination") or ""


def _source_of(action: SemanticActionLike) -> str:
    return action.parameters.get("source") or _path_of(action)


def _parent(path: str) -> str:
    return os.path.dirname(path) if path else ""


def _basename(path: str) -> str:
    return os.path.basename(path) if path else ""


def _ext(path: str) -> str:
    if not path:
        return ""
    return os.path.splitext(os.path.basename(path))[1].lower()


def _looks_like_directory(path: str) -> bool:
    """Heuristic: paths without a file extension are treated as directories."""
    if not path:
        return False
    base = os.path.basename(path.rstrip("/"))
    if not base:
        return False
    return "." not in base


# ── Pattern matchers ──────────────────────────────────────────────────────


def _match_organize_to_folder(
    actions: Sequence[SemanticActionLike], start: int
) -> Optional[FSPattern]:
    """file.create(dir) followed by N (≥2) file.move actions into that dir."""
    if start >= len(actions):
        return None

    head = actions[start]
    if head.action_name != "file.create":
        return None

    folder_path = _path_of(head)
    if not folder_path or not _looks_like_directory(folder_path):
        return None

    moved_files: List[str] = []
    indices: List[int] = [start]
    j = start + 1
    while j < len(actions):
        nxt = actions[j]
        if nxt.action_name != "file.move":
            break
        dest = _path_of(nxt)
        # A move "into" the created folder: destination's parent equals folder_path,
        # or destination starts with folder_path + sep (nested move into subtree).
        if _parent(dest) != folder_path and not dest.startswith(folder_path + os.sep):
            break
        moved_files.append(_source_of(nxt))
        indices.append(j)
        j += 1

    if len(moved_files) < 2:
        return None

    # Compute the dominant file extension for a richer description.
    exts = [_ext(p) for p in moved_files if _ext(p)]
    ext_label = ""
    if exts:
        dominant = max(set(exts), key=exts.count).lstrip(".")
        if dominant and exts.count("." + dominant) == len(exts):
            ext_label = f"{dominant.upper()} "

    return FSPattern(
        name="organize_files_to_folder",
        description=(
            f"Move {len(moved_files)} {ext_label}files to "
            f"{_basename(folder_path) or folder_path} folder"
        ),
        parameters={
            "folder": folder_path,
            "folder_name": _basename(folder_path),
            "files": moved_files,
            "file_count": len(moved_files),
            "file_extension": ext_label.strip().lower() or None,
        },
        source_indices=tuple(indices),
    )


def _match_batch_delete(
    actions: Sequence[SemanticActionLike], start: int
) -> Optional[FSPattern]:
    """N (≥2) consecutive file.delete actions."""
    if start >= len(actions) or actions[start].action_name != "file.delete":
        return None

    deleted: List[str] = []
    indices: List[int] = []
    j = start
    while j < len(actions) and actions[j].action_name == "file.delete":
        deleted.append(_path_of(actions[j]))
        indices.append(j)
        j += 1

    if len(deleted) < 2:
        return None

    parents = {_parent(p) for p in deleted if p}
    common_parent = parents.pop() if len(parents) == 1 else ""

    return FSPattern(
        name="batch_delete",
        description=(
            f"Delete {len(deleted)} files"
            + (f" in {_basename(common_parent) or common_parent}" if common_parent else "")
        ),
        parameters={
            "files": deleted,
            "file_count": len(deleted),
            "common_parent": common_parent or None,
        },
        source_indices=tuple(indices),
    )


def _match_rename(
    actions: Sequence[SemanticActionLike], start: int
) -> Optional[FSPattern]:
    """file.delete + file.create in the same directory, same extension."""
    if start + 1 >= len(actions):
        return None

    a, b = actions[start], actions[start + 1]
    if a.action_name != "file.delete" or b.action_name != "file.create":
        return None

    src, dst = _path_of(a), _path_of(b)
    if not src or not dst:
        return None
    if _parent(src) != _parent(dst):
        return None
    if _ext(src) != _ext(dst):
        return None
    if _basename(src) == _basename(dst):
        return None

    return FSPattern(
        name="rename_file",
        description=f"Rename {_basename(src)} → {_basename(dst)}",
        parameters={
            "source": src,
            "destination": dst,
            "directory": _parent(src),
            "extension": _ext(src),
        },
        source_indices=(start, start + 1),
    )


def _match_create_document(
    actions: Sequence[SemanticActionLike], start: int
) -> Optional[FSPattern]:
    """file.create + file.modify on the same path."""
    if start + 1 >= len(actions):
        return None

    a, b = actions[start], actions[start + 1]
    if a.action_name != "file.create" or b.action_name != "file.modify":
        return None

    path = _path_of(a)
    if not path or path != _path_of(b):
        return None
    # Skip directory-shaped paths — those belong to organize_files_to_folder.
    if _looks_like_directory(path):
        return None

    return FSPattern(
        name="create_document",
        description=f"Create document {_basename(path)}",
        parameters={
            "path": path,
            "filename": _basename(path),
            "directory": _parent(path),
            "extension": _ext(path),
        },
        source_indices=(start, start + 1),
    )


# Ordered registry: higher-priority / more-specific matchers come first so
# they "win" over weaker overlaps (e.g. organize wins over create_document).
DEFAULT_MATCHERS: Tuple[PatternMatcher, ...] = (
    _match_organize_to_folder,
    _match_batch_delete,
    _match_rename,
    _match_create_document,
)


# ── Pass implementation ───────────────────────────────────────────────────


class FileSystemPatternPass(AbstractionPass):
    """Recognize high-level file operation patterns from low-level FS events.

    Sits between ``GroupingPass`` and ``PatternPass``.  When the visual signal
    is unavailable, the synthesized high-level actions become the primary
    source of meaning for downstream skill distillation.
    """

    DEFAULT_CONFIDENCE: float = 0.85

    def __init__(
        self,
        matchers: Optional[Sequence[PatternMatcher]] = None,
        *,
        confidence: Optional[float] = None,
    ) -> None:
        self._matchers: Tuple[PatternMatcher, ...] = (
            tuple(matchers) if matchers is not None else DEFAULT_MATCHERS
        )
        self._confidence: float = (
            confidence if confidence is not None else self.DEFAULT_CONFIDENCE
        )

    def apply(
        self,
        actions: List[SemanticAction],
        steps: Optional[List[TrajectoryStep]] = None,
    ) -> List[SemanticAction]:
        """Replace matched FS-event sub-sequences with single high-level actions."""
        if not actions:
            return actions

        result: List[SemanticAction] = []
        i = 0
        n = len(actions)
        while i < n:
            pattern = self._try_match(actions, i)
            if pattern is None:
                result.append(actions[i])
                i += 1
                continue
            result.append(self._pattern_to_action(pattern, actions))
            # source_indices is contiguous by construction; advance past last consumed.
            consumed = len(pattern.source_indices)
            i += consumed if consumed > 0 else 1
        return result

    def _try_match(
        self, actions: Sequence[SemanticActionLike], start_idx: int
    ) -> Optional[FSPattern]:
        for matcher in self._matchers:
            try:
                pattern = matcher(actions, start_idx)
            except Exception:  # pragma: no cover - defensive
                logger.exception("FS pattern matcher %s failed", getattr(matcher, "__name__", matcher))
                continue
            if pattern is not None:
                return pattern
        return None

    def _pattern_to_action(
        self,
        pattern: FSPattern,
        actions: Sequence[SemanticActionLike],
    ) -> SemanticAction:
        """Synthesize a single high-level SemanticAction from a matched pattern."""
        indices = pattern.source_indices
        first_idx = indices[0]
        last_idx = indices[-1]

        first_range = actions[first_idx].raw_action_range or (first_idx, first_idx + 1)
        last_range = actions[last_idx].raw_action_range or (last_idx, last_idx + 1)

        # Carry context from the matched window — earlier params first, then
        # overlay matcher-extracted params (which are more specific).
        merged: Dict[str, Any] = {}
        for idx in indices:
            merged.update(actions[idx].parameters)
        merged.update(pattern.parameters)

        return SemanticAction(
            action_name=pattern.name,
            description=pattern.description,
            parameters=merged,
            raw_action_range=(first_range[0], last_range[1]),
            confidence=self._confidence,
        )


__all__ = [
    "FSPattern",
    "FileSystemPatternPass",
    "PatternMatcher",
    "SemanticActionLike",
    "DEFAULT_MATCHERS",
]
