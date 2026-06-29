"""Platform-aware event synthesis — merges low-level OS events into high-level operations.

Sits between DenoisePass and GroupingPass in the abstraction pipeline.

Responsibilities:
    1. Rename-pair synthesis: two FSEvent renames → one file.move(source, destination)
    2. Batch aggregation: consecutive same-destination moves → batch_move
    3. System noise suppression: filter platform-internal metadata operations

Architecture:
    SynthesisRule (ABC)   — single-responsibility transformation rule
    PlatformSynthesisPass — orchestrates rules in priority order (OCP-compliant)

Rules are composable: new platform-specific rules can be added via
PlatformSynthesisPass(rules=[...]) without modifying existing code.
"""

from __future__ import annotations

import logging
import os.path
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from leapflow.analysis.abstractor import AbstractionPass
from leapflow.domain.trajectory import SemanticAction, TrajectoryStep

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# SynthesisRule protocol
# ══════════════════════════════════════════════════════════════════════


class SynthesisRule(ABC):
    """A single event synthesis transformation.

    Each rule consumes a list of SemanticActions and produces a
    (usually shorter) list with synthesized high-level actions.
    """

    @abstractmethod
    def apply(self, actions: List[SemanticAction]) -> List[SemanticAction]:
        """Transform action sequence. Must not mutate the input list."""


# ══════════════════════════════════════════════════════════════════════
# Rule 1: Rename Pair Synthesis
# ══════════════════════════════════════════════════════════════════════


class RenamePairRule(SynthesisRule):
    """Pair consecutive file.rename events into file.move operations.

    macOS FSEvents emits two RENAMED events for a single mv/move:
      - Event A: old path (file "renamed away" from here)
      - Event B: new path (file "renamed to" here)

    Pairing heuristics (all must hold):
      1. Both actions are file.rename
      2. They are temporally adjacent (within max_gap steps)
      3. Basename matches (same filename, different directory) — this is a move
         OR directory differs and filenames differ — this is a move+rename

    After pairing, the two renames become one file.move with source and destination.
    """

    def __init__(self, *, max_gap: int = 3) -> None:
        self._max_gap = max_gap

    def apply(self, actions: List[SemanticAction]) -> List[SemanticAction]:
        if len(actions) < 2:
            return actions

        result: List[SemanticAction] = []
        consumed: Set[int] = set()

        for i, action in enumerate(actions):
            if i in consumed:
                continue

            if action.action_name != "file.rename":
                result.append(action)
                continue

            pair_idx = self._find_pair(actions, i, consumed)
            if pair_idx is not None:
                consumed.add(pair_idx)
                result.append(self._synthesize_move(action, actions[pair_idx]))
            else:
                result.append(action)

        return result

    def _find_pair(
        self, actions: List[SemanticAction], idx: int, consumed: Set[int]
    ) -> Optional[int]:
        """Find the paired rename event for the rename at idx.

        FSEvents emits two RENAMED events for any mv operation:
          - Same basename, different dir → cross-directory move
          - Different basename, same dir → in-place rename
          - Different basename, different dir → move + rename
        All three cases produce a pair when temporally adjacent.
        """
        source_path = self._get_path(actions[idx])
        if not source_path:
            return None

        source_basename = os.path.basename(source_path)
        source_dir = os.path.dirname(source_path)

        for j in range(idx + 1, min(idx + 1 + self._max_gap, len(actions))):
            if j in consumed:
                continue
            candidate = actions[j]
            if candidate.action_name != "file.rename":
                continue

            candidate_path = self._get_path(candidate)
            if not candidate_path:
                continue

            candidate_dir = os.path.dirname(candidate_path)
            candidate_basename = os.path.basename(candidate_path)

            # Identical paths are not a pair (duplicate event)
            if source_path == candidate_path:
                continue

            # Same basename, different directory → cross-directory move
            if source_basename == candidate_basename and source_dir != candidate_dir:
                return j

            # Different basename (same or different dir) → rename or move+rename
            if source_basename != candidate_basename and self._is_temporal_pair(actions[idx], candidate):
                return j

        return None

    def _synthesize_move(
        self, source_action: SemanticAction, dest_action: SemanticAction
    ) -> SemanticAction:
        """Create a file.move or file.rename from paired rename events.

        If source and destination share the same directory (only basename changed),
        this is an in-place rename. Otherwise it's a move (possibly with rename).
        """
        source_path = self._get_path(source_action)
        dest_path = self._get_path(dest_action)
        source_dir = os.path.dirname(source_path)
        dest_dir = os.path.dirname(dest_path)

        if source_dir == dest_dir:
            return SemanticAction(
                action_name="file.rename",
                description=f"Rename {os.path.basename(source_path)} → {os.path.basename(dest_path)}",
                parameters={
                    "source": source_path,
                    "destination": dest_path,
                    "path": dest_path,
                    "target": dest_path,
                },
                raw_action_range=(
                    source_action.raw_action_range[0],
                    dest_action.raw_action_range[1],
                ),
                confidence=0.95,
            )

        return SemanticAction(
            action_name="file.move",
            description=f"Move {os.path.basename(source_path)} to {os.path.basename(dest_dir) or dest_dir}",
            parameters={
                "source": source_path,
                "destination": dest_path,
                "path": dest_path,
                "target": dest_path,
            },
            raw_action_range=(
                source_action.raw_action_range[0],
                dest_action.raw_action_range[1],
            ),
            confidence=0.95,
        )

    @staticmethod
    def _get_path(action: SemanticAction) -> str:
        return action.parameters.get("path", "") or action.parameters.get("target", "")

    @staticmethod
    def _is_temporal_pair(a: SemanticAction, b: SemanticAction) -> bool:
        """Check if two actions are close enough in raw step indices to be a pair."""
        a_end = a.raw_action_range[1] if a.raw_action_range else 0
        b_start = b.raw_action_range[0] if b.raw_action_range else 0
        return abs(b_start - a_end) <= 2


# ══════════════════════════════════════════════════════════════════════
# Rule 2: App Launch Noise Suppression
# ══════════════════════════════════════════════════════════════════════


class AppLaunchNoiseRule(SynthesisRule):
    """Filter file operations that are side effects of app launching.

    When an app activates, the OS reads/writes its bundle resources,
    preferences, and caches. These appear as file.modify/create events
    within a short window after app.switch. They are never user intent.

    The window breaks on substantive user actions (click, type, shortcut)
    to avoid incorrectly filtering file ops caused by user interaction.
    """

    _APP_NOISE_PATTERNS: Tuple[re.Pattern[str], ...] = (
        re.compile(r"/Application Support/"),
        re.compile(r"/Preferences/"),
        re.compile(r"/Caches/"),
        re.compile(r"\.app/"),
        re.compile(r"/Containers/.+/Data/"),
        re.compile(r"\.plist$"),
    )

    _WINDOW_BREAKERS = frozenset({"ui.click", "ui.type", "ui.shortcut"})

    def __init__(self, *, window_steps: int = 3) -> None:
        self._window = window_steps

    def apply(self, actions: List[SemanticAction]) -> List[SemanticAction]:
        if len(actions) < 2:
            return actions

        switch_indices = [
            i for i, a in enumerate(actions) if a.action_name == "app.switch"
        ]
        if not switch_indices:
            return actions

        noise_indices: Set[int] = set()
        for si in switch_indices:
            for j in range(si + 1, min(si + 1 + self._window, len(actions))):
                if j in noise_indices:
                    continue
                action = actions[j]
                if action.action_name in self._WINDOW_BREAKERS:
                    break
                if not action.action_name.startswith("file."):
                    continue
                path = action.parameters.get("path", "") or action.parameters.get("target", "")
                if self._is_launch_noise(path):
                    noise_indices.add(j)

        if not noise_indices:
            return actions
        return [a for i, a in enumerate(actions) if i not in noise_indices]

    def _is_launch_noise(self, path: str) -> bool:
        if not path:
            return False
        return any(p.search(path) for p in self._APP_NOISE_PATTERNS)


# ══════════════════════════════════════════════════════════════════════
# Rule 3: Duplicate Focus Deduplication
# ══════════════════════════════════════════════════════════════════════


class DuplicateFocusRule(SynthesisRule):
    """Deduplicate consecutive app.switch events to the same target.

    macOS fires multiple focus notifications for a single app activation
    (window ready, menu bar update, etc.). Keep only the first in a
    consecutive run.
    """

    def apply(self, actions: List[SemanticAction]) -> List[SemanticAction]:
        if len(actions) < 2:
            return actions

        result: List[SemanticAction] = [actions[0]]
        for i in range(1, len(actions)):
            current = actions[i]
            prev = result[-1]
            if (current.action_name == "app.switch"
                    and prev.action_name == "app.switch"
                    and self._same_target(current, prev)):
                continue
            result.append(current)
        return result

    @staticmethod
    def _same_target(a: SemanticAction, b: SemanticAction) -> bool:
        target_a = a.parameters.get("target", "") or a.parameters.get("app_bundle_id", "")
        target_b = b.parameters.get("target", "") or b.parameters.get("app_bundle_id", "")
        return target_a == target_b and target_a != ""


# ══════════════════════════════════════════════════════════════════════
# Rule 3b: Modify-Pair Move Synthesis (macOS FSEvents)
# ══════════════════════════════════════════════════════════════════════


class ModifyPairMoveRule(SynthesisRule):
    """Pair consecutive file.modify events into file.move on macOS.

    macOS FSEvents reports a cross-directory `mv` as TWO file.modify events:
      - Event A: source path (file "modified" = removed from here)
      - Event B: destination path (file "modified" = appeared here)

    Pairing heuristics (all must hold):
      1. Both actions are file.modify
      2. They are temporally adjacent (within max_gap steps)
      3. Basename matches (same filename, different directory)
      4. Source path no longer contains the file OR destination wasn't tracked before
         (approximated by: source directory != destination directory)

    After pairing, the two modifies become one file.move(source, destination).
    """

    def __init__(self, *, max_gap: int = 3) -> None:
        self._max_gap = max_gap

    def apply(self, actions: List[SemanticAction]) -> List[SemanticAction]:
        if len(actions) < 2:
            return actions

        result: List[SemanticAction] = []
        consumed: Set[int] = set()

        for i, action in enumerate(actions):
            if i in consumed:
                continue

            if action.action_name != "file.modify":
                result.append(action)
                continue

            pair_idx = self._find_pair(actions, i, consumed)
            if pair_idx is not None:
                consumed.add(pair_idx)
                result.append(self._synthesize_move(action, actions[pair_idx]))
            else:
                result.append(action)

        return result

    def _find_pair(
        self, actions: List[SemanticAction], idx: int, consumed: Set[int]
    ) -> Optional[int]:
        """Find the paired modify event for the modify at idx."""
        source_path = self._get_path(actions[idx])
        if not source_path:
            return None

        source_basename = os.path.basename(source_path)
        source_dir = os.path.dirname(source_path)

        if not source_basename:
            return None

        for j in range(idx + 1, min(idx + 1 + self._max_gap, len(actions))):
            if j in consumed:
                continue
            candidate = actions[j]
            if candidate.action_name != "file.modify":
                continue

            candidate_path = self._get_path(candidate)
            if not candidate_path:
                continue

            candidate_basename = os.path.basename(candidate_path)
            candidate_dir = os.path.dirname(candidate_path)

            # Must be same file, different directory
            if source_basename == candidate_basename and source_dir != candidate_dir:
                return j

        return None

    def _synthesize_move(
        self, source_action: SemanticAction, dest_action: SemanticAction
    ) -> SemanticAction:
        source_path = self._get_path(source_action)
        dest_path = self._get_path(dest_action)
        dest_dir = os.path.dirname(dest_path)

        return SemanticAction(
            action_name="file.move",
            description=f"Move {os.path.basename(source_path)} to {os.path.basename(dest_dir) or dest_dir}",
            parameters={
                "source": source_path,
                "destination": dest_path,
                "path": dest_path,
                "target": dest_path,
            },
            raw_action_range=(
                source_action.raw_action_range[0],
                dest_action.raw_action_range[1],
            ),
            confidence=0.9,
        )

    @staticmethod
    def _get_path(action: SemanticAction) -> str:
        return action.parameters.get("path", "") or action.parameters.get("target", "")


# ══════════════════════════════════════════════════════════════════════
# Rule 4: Batch Move Aggregation
# ══════════════════════════════════════════════════════════════════════


class BatchMoveRule(SynthesisRule):
    """Aggregate consecutive file.move operations sharing a destination directory.

    Detects patterns like:
        move(a.pdf → dir/a.pdf)
        move(b.pdf → dir/b.pdf)
        move(c.pdf → dir/c.pdf)

    And synthesizes into:
        batch_move(files=[a.pdf, b.pdf, c.pdf], destination=dir/, count=3)

    Only aggregates when 2+ consecutive moves share the same destination directory.
    """

    def __init__(self, *, min_batch_size: int = 2) -> None:
        self._min_batch = min_batch_size

    def apply(self, actions: List[SemanticAction]) -> List[SemanticAction]:
        if len(actions) < self._min_batch:
            return actions

        result: List[SemanticAction] = []
        i = 0

        while i < len(actions):
            if actions[i].action_name == "file.move":
                batch_end = self._find_batch_end(actions, i)
                if batch_end - i >= self._min_batch:
                    result.append(self._synthesize_batch(actions[i:batch_end]))
                    i = batch_end
                    continue
            result.append(actions[i])
            i += 1

        return result

    def _find_batch_end(self, actions: List[SemanticAction], start: int) -> int:
        """Find the end of a consecutive same-destination move run."""
        dest_dir = self._dest_dir(actions[start])
        if not dest_dir:
            return start + 1

        end = start + 1
        while end < len(actions):
            if actions[end].action_name != "file.move":
                break
            if self._dest_dir(actions[end]) != dest_dir:
                break
            end += 1
        return end

    def _synthesize_batch(self, moves: List[SemanticAction]) -> SemanticAction:
        """Create a batch_move from consecutive same-destination moves."""
        dest_dir = self._dest_dir(moves[0])
        sources = [
            m.parameters.get("source", "") for m in moves
        ]
        destinations = [
            m.parameters.get("destination", "") for m in moves
        ]

        # Detect common extension
        extensions = {
            os.path.splitext(s)[1].lower()
            for s in sources if os.path.splitext(s)[1]
        }
        common_ext = next(iter(extensions)) if len(extensions) == 1 else ""
        ext_desc = f" ({common_ext})" if common_ext else ""

        first = moves[0]
        last = moves[-1]
        count = len(moves)

        return SemanticAction(
            action_name="batch_move",
            description=(
                f"Move {count} file{'' if count == 1 else 's'}{ext_desc} "
                f"to {os.path.basename(dest_dir.rstrip('/')) or dest_dir}"
            ),
            parameters={
                "target_dir": dest_dir,
                "sources": sources,
                "destinations": destinations,
                "count": count,
                "file_pattern": f"*{common_ext}" if common_ext else "*",
                "first_file": sources[0] if sources else "",
                "path": dest_dir,
            },
            raw_action_range=(
                first.raw_action_range[0],
                last.raw_action_range[1],
            ),
            confidence=min(m.confidence for m in moves),
        )

    @staticmethod
    def _dest_dir(action: SemanticAction) -> str:
        dest = action.parameters.get("destination", "")
        return os.path.dirname(dest) if dest else ""


# ══════════════════════════════════════════════════════════════════════
# Rule 5: System Noise Suppression
# ══════════════════════════════════════════════════════════════════════


_DEFAULT_SYSTEM_NOISE = (
    re.compile(r"/\.fseventsd/"),
    re.compile(r"/\.DS_Store$"),
    re.compile(r"/\.Spotlight-V100/"),
    re.compile(r"\.lock$"),
    re.compile(r"/\.DocumentRevisions-V100/"),
    re.compile(r"/\.TemporaryItems/"),
    re.compile(r"/\.apdisk$"),
    re.compile(r"/\.com\.apple\."),
    re.compile(r"/\.vol/"),
    re.compile(r"/__MACOSX/"),
)

_LINUX_NOISE_PATTERNS = (
    re.compile(r"/\.gvfs/"),
    re.compile(r"/\.local/share/Trash/"),
    re.compile(r"/\.cache/"),
    re.compile(r"/proc/"),
    re.compile(r"/sys/"),
    re.compile(r"/run/"),
    re.compile(r"/\.dbus/"),
)


class SystemNoiseRule(SynthesisRule):
    """Filter out file operations on system/platform-internal paths.

    Targets paths that are platform infrastructure (FSEvents daemon,
    Spotlight indexer, lock files, document revision stores) — never
    part of user intent.

    Patterns are configurable via constructor for cross-platform support.
    """

    def __init__(self, patterns: Optional[Sequence[re.Pattern]] = None) -> None:  # type: ignore[type-arg]
        self._patterns = list(patterns) if patterns is not None else list(_DEFAULT_SYSTEM_NOISE)

    def apply(self, actions: List[SemanticAction]) -> List[SemanticAction]:
        return [a for a in actions if not self._is_system_noise(a)]

    def _is_system_noise(self, action: SemanticAction) -> bool:
        if not action.action_name.startswith("file."):
            return False

        path = action.parameters.get("path", "") or action.parameters.get("target", "")
        if not path:
            return False

        return any(p.search(path) for p in self._patterns)


# ══════════════════════════════════════════════════════════════════════
# Rule 6: Unpaired Rename Dedup
# ══════════════════════════════════════════════════════════════════════


class UnpairedRenameDedupRule(SynthesisRule):
    """Deduplicate leftover file.rename events that represent the same operation.

    After RenamePairRule runs, some rename events may remain unpaired
    (e.g., when one of the pair was filtered by attention). This rule
    deduplicates consecutive renames on the same basename within a window,
    keeping only the one with the longer (more informative) path.
    """

    def __init__(self, *, max_gap: int = 2) -> None:
        self._max_gap = max_gap

    def apply(self, actions: List[SemanticAction]) -> List[SemanticAction]:
        if len(actions) < 2:
            return actions

        result: List[SemanticAction] = []
        skip: Set[int] = set()

        for i, action in enumerate(actions):
            if i in skip:
                continue
            if action.action_name != "file.rename":
                result.append(action)
                continue

            best = action
            for j in range(i + 1, min(i + 1 + self._max_gap, len(actions))):
                if j in skip:
                    continue
                candidate = actions[j]
                if candidate.action_name != "file.rename":
                    break
                if self._is_duplicate(action, candidate):
                    skip.add(j)
                    if len(self._get_path(candidate)) > len(self._get_path(best)):
                        best = candidate

            result.append(best)

        return result

    def _is_duplicate(self, a: SemanticAction, b: SemanticAction) -> bool:
        path_a = self._get_path(a)
        path_b = self._get_path(b)
        if not path_a or not path_b:
            return False
        return os.path.basename(path_a) == os.path.basename(path_b)

    @staticmethod
    def _get_path(action: SemanticAction) -> str:
        return action.parameters.get("path", "") or action.parameters.get("target", "")


# ══════════════════════════════════════════════════════════════════════
# Rule 7: Window Management Noise (L1 — UI domain)
# ══════════════════════════════════════════════════════════════════════


class WindowManagementRule(SynthesisRule):
    """Filter or collapse consecutive window move/resize events.

    Window management produces many intermediate-state events:
        ui.move(window) × N, ui.resize(window) × N
    These are noise unless they represent a functional operation (e.g., snap).

    Strategy: collapse consecutive move/resize events on the same target
    window into a single operation, or filter entirely when they appear
    as positioning noise between substantive actions.
    """

    _WINDOW_ACTIONS = frozenset({"ui.move", "ui.resize"})

    def __init__(self, *, min_run: int = 2) -> None:
        self._min_run = min_run

    def apply(self, actions: List[SemanticAction]) -> List[SemanticAction]:
        if len(actions) < self._min_run:
            return actions

        result: List[SemanticAction] = []
        i = 0

        while i < len(actions):
            if actions[i].action_name not in self._WINDOW_ACTIONS:
                result.append(actions[i])
                i += 1
                continue

            run_end = self._find_run_end(actions, i)
            run_length = run_end - i

            if run_length >= self._min_run:
                result.append(self._collapse_run(actions[i:run_end]))
            else:
                for j in range(i, run_end):
                    result.append(actions[j])
            i = run_end

        return result

    def _find_run_end(self, actions: List[SemanticAction], start: int) -> int:
        """Find end of consecutive move/resize events on the same window."""
        target = self._window_target(actions[start])
        end = start + 1
        while end < len(actions):
            if actions[end].action_name not in self._WINDOW_ACTIONS:
                break
            if target and self._window_target(actions[end]) != target:
                break
            end += 1
        return end

    def _collapse_run(self, run: List[SemanticAction]) -> SemanticAction:
        """Collapse a window management run into a single action."""
        first = run[0]
        last = run[-1]
        target = self._window_target(first) or "window"
        has_resize = any(a.action_name == "ui.resize" for a in run)
        action_name = "window.arrange" if has_resize else "window.move"

        return SemanticAction(
            action_name=action_name,
            description=f"Arrange {target}",
            parameters={
                "target": target,
                "event_count": len(run),
                "has_resize": has_resize,
                "app_bundle_id": first.parameters.get("app_bundle_id", ""),
            },
            raw_action_range=(
                first.raw_action_range[0],
                last.raw_action_range[1],
            ),
            confidence=0.8,
        )

    @staticmethod
    def _window_target(action: SemanticAction) -> str:
        return action.parameters.get("target", "") or action.parameters.get("target_label", "")


# ══════════════════════════════════════════════════════════════════════
# Rule 8: Drag-and-Drop Synthesis (L1 — UI domain)
# ══════════════════════════════════════════════════════════════════════


class DragDropRule(SynthesisRule):
    """Synthesize drag-and-drop sequences into a single drag_drop action.

    Pattern: ui.drag + file.move (or file.rename pair that became file.move).
    The ui.drag action is the user's gesture; the file.move is the OS effect.
    When they appear in close proximity, synthesize into drag_drop.
    """

    def __init__(self, *, window_steps: int = 3) -> None:
        self._window = window_steps

    def apply(self, actions: List[SemanticAction]) -> List[SemanticAction]:
        if len(actions) < 2:
            return actions

        result: List[SemanticAction] = []
        consumed: Set[int] = set()

        for i, action in enumerate(actions):
            if i in consumed:
                continue
            if action.action_name == "ui.drag":
                move_idx = self._find_associated_move(actions, i, consumed)
                if move_idx is not None:
                    consumed.add(move_idx)
                    result.append(self._synthesize_drag_drop(action, actions[move_idx]))
                else:
                    result.append(action)
            else:
                result.append(action)

        return result

    def _find_associated_move(
        self, actions: List[SemanticAction], drag_idx: int, consumed: Set[int]
    ) -> Optional[int]:
        """Find a file.move within window_steps after a drag action."""
        for j in range(drag_idx + 1, min(drag_idx + 1 + self._window, len(actions))):
            if j in consumed:
                continue
            if actions[j].action_name in ("file.move", "batch_move"):
                return j
        return None

    @staticmethod
    def _synthesize_drag_drop(drag: SemanticAction, move: SemanticAction) -> SemanticAction:
        source = move.parameters.get("source", "")
        destination = move.parameters.get("destination", "")
        return SemanticAction(
            action_name="drag_drop",
            description=f"Drag {os.path.basename(source)} to {os.path.basename(os.path.dirname(destination))}",
            parameters={
                "source": source,
                "destination": destination,
                "drag_target": drag.parameters.get("target", ""),
                "app_bundle_id": drag.parameters.get("app_bundle_id", ""),
            },
            raw_action_range=(
                drag.raw_action_range[0],
                move.raw_action_range[1],
            ),
            confidence=0.85,
        )


# ══════════════════════════════════════════════════════════════════════
# Rule 9: Download Synthesis (L2 — cross-domain: UI + FS)
# ══════════════════════════════════════════════════════════════════════


_DOWNLOAD_TMP_SUFFIXES = frozenset({
    ".crdownload", ".download", ".part", ".partial", ".tmp",
    ".crswap", ".opdownload",
})


class DownloadSynthesisRule(SynthesisRule):
    """Synthesize browser download sequences into download_file actions.

    Pattern:
        file.create(path.crdownload) → [file.modify × N] → file.rename(→ final_path)

    The temporary file creation + modifications + final rename represent a
    single download operation. Synthesize into download_file(path).
    """

    def apply(self, actions: List[SemanticAction]) -> List[SemanticAction]:
        if len(actions) < 2:
            return actions

        result: List[SemanticAction] = []
        consumed: Set[int] = set()

        for i, action in enumerate(actions):
            if i in consumed:
                continue

            if action.action_name == "file.create" and self._is_temp_download(action):
                end_idx = self._find_download_end(actions, i, consumed)
                if end_idx is not None:
                    for j in range(i + 1, end_idx + 1):
                        consumed.add(j)
                    result.append(self._synthesize_download(action, actions[end_idx]))
                else:
                    consumed.add(i)
                    continue
            elif action.action_name == "file.modify" and self._is_temp_download(action):
                consumed.add(i)
                continue
            else:
                result.append(action)

        return result

    def _is_temp_download(self, action: SemanticAction) -> bool:
        path = action.parameters.get("path", "") or action.parameters.get("target", "")
        if not path:
            return False
        _, ext = os.path.splitext(path.lower())
        return ext in _DOWNLOAD_TMP_SUFFIXES

    def _find_download_end(
        self, actions: List[SemanticAction], start: int, consumed: Set[int]
    ) -> Optional[int]:
        """Find the file.rename/file.move that completes a download sequence."""
        tmp_path = actions[start].parameters.get("path", "") or actions[start].parameters.get("target", "")
        tmp_stem = os.path.splitext(tmp_path)[0]

        for j in range(start + 1, min(start + 20, len(actions))):
            if j in consumed:
                continue
            action = actions[j]
            if action.action_name in ("file.rename", "file.move"):
                source = action.parameters.get("source", "") or action.parameters.get("path", "")
                if source == tmp_path or os.path.splitext(source)[0] == tmp_stem:
                    return j
                dest = action.parameters.get("destination", "")
                if dest and os.path.splitext(dest)[0] == tmp_stem:
                    return j
            elif action.action_name == "file.modify" and self._is_temp_download(action):
                continue
        return None

    @staticmethod
    def _synthesize_download(create_action: SemanticAction, end_action: SemanticAction) -> SemanticAction:
        final_path = (
            end_action.parameters.get("destination", "")
            or end_action.parameters.get("path", "")
            or end_action.parameters.get("target", "")
        )
        return SemanticAction(
            action_name="download_file",
            description=f"Download {os.path.basename(final_path)}",
            parameters={
                "path": final_path,
                "target": final_path,
            },
            raw_action_range=(
                create_action.raw_action_range[0],
                end_action.raw_action_range[1],
            ),
            confidence=0.9,
        )


# ══════════════════════════════════════════════════════════════════════
# Rule 10: Permission/Auth Noise (L2 — cross-domain: App + UI)
# ══════════════════════════════════════════════════════════════════════


_PERMISSION_BUNDLES = frozenset({
    "com.apple.SecurityAgent",
    "com.apple.systempreferences",
    "com.apple.SystemPreferences",
    "com.apple.Passwords",
    "com.apple.UserNotificationCenter",
})


class PermissionNoiseRule(SynthesisRule):
    """Filter permission/auth popup sequences that interrupt the user's workflow.

    Pattern:
        app.switch(SecurityAgent/SystemPreferences) → [ui.click × N] → app.switch(original_app)

    These are environment configuration, not skill steps. The entire sequence
    (switch to system app + interaction + return) is filtered.
    """

    def apply(self, actions: List[SemanticAction]) -> List[SemanticAction]:
        if len(actions) < 3:
            return actions

        noise_indices: Set[int] = set()

        for i, action in enumerate(actions):
            if i in noise_indices:
                continue
            if action.action_name != "app.switch":
                continue
            target_app = self._get_switch_target(action)
            if target_app not in _PERMISSION_BUNDLES:
                continue

            return_idx = self._find_return_switch(actions, i, noise_indices)
            if return_idx is not None:
                for j in range(i, return_idx + 1):
                    noise_indices.add(j)

        if not noise_indices:
            return actions
        return [a for i, a in enumerate(actions) if i not in noise_indices]

    def _find_return_switch(
        self, actions: List[SemanticAction], start: int, noise: Set[int]
    ) -> Optional[int]:
        """Find the app.switch back to the original app (or any non-system app)."""
        for j in range(start + 1, min(start + 10, len(actions))):
            if j in noise:
                continue
            if actions[j].action_name == "app.switch":
                target = self._get_switch_target(actions[j])
                if target not in _PERMISSION_BUNDLES:
                    return j
        return None

    @staticmethod
    def _get_switch_target(action: SemanticAction) -> str:
        return (
            action.parameters.get("app_bundle_id", "")
            or action.parameters.get("target", "")
        )


# ══════════════════════════════════════════════════════════════════════
# Rule 11: Open-In-App Synthesis (L2 — cross-domain: UI/FS + App)
# ══════════════════════════════════════════════════════════════════════


class OpenInAppSynthesisRule(SynthesisRule):
    """Synthesize 'open file in app' sequences.

    Pattern:
        ui.click(file_icon) [or file.create] + app.switch(new_app)
        within a close window → open_in_app(file, app)

    Triggers when a file-initiating action immediately precedes an app switch
    to a different app, indicating the user opened a file which launched/
    switched to the target app.

    NOTE: file.modify is intentionally excluded — a save+switch sequence
    is normal workflow, not an "open in app" operation.
    """

    _FILE_ACTIONS = frozenset({"ui.click", "file.create"})

    def __init__(self, *, window_steps: int = 2) -> None:
        self._window = window_steps

    def apply(self, actions: List[SemanticAction]) -> List[SemanticAction]:
        if len(actions) < 2:
            return actions

        result: List[SemanticAction] = []
        consumed: Set[int] = set()

        for i, action in enumerate(actions):
            if i in consumed:
                continue

            if action.action_name in self._FILE_ACTIONS and self._has_file_target(action):
                switch_idx = self._find_app_switch(actions, i, consumed)
                if switch_idx is not None:
                    consumed.add(switch_idx)
                    result.append(self._synthesize_open_in_app(action, actions[switch_idx]))
                else:
                    result.append(action)
            else:
                result.append(action)

        return result

    def _has_file_target(self, action: SemanticAction) -> bool:
        target = action.parameters.get("path", "") or action.parameters.get("target", "")
        if not target:
            return False
        _, ext = os.path.splitext(target)
        return bool(ext) and not target.endswith("/")

    def _find_app_switch(
        self, actions: List[SemanticAction], file_idx: int, consumed: Set[int]
    ) -> Optional[int]:
        file_app = actions[file_idx].parameters.get("app_bundle_id", "")
        for j in range(file_idx + 1, min(file_idx + 1 + self._window, len(actions))):
            if j in consumed:
                continue
            if actions[j].action_name == "app.switch":
                switch_app = self._get_switch_target(actions[j])
                if switch_app and switch_app != file_app:
                    return j
        return None

    @staticmethod
    def _get_switch_target(action: SemanticAction) -> str:
        return (
            action.parameters.get("app_bundle_id", "")
            or action.parameters.get("target", "")
        )

    @staticmethod
    def _synthesize_open_in_app(
        file_action: SemanticAction, switch_action: SemanticAction
    ) -> SemanticAction:
        file_path = file_action.parameters.get("path", "") or file_action.parameters.get("target", "")
        app = switch_action.parameters.get("app_bundle_id", "") or switch_action.parameters.get("target", "")
        return SemanticAction(
            action_name="open_in_app",
            description=f"Open {os.path.basename(file_path)} in {app.split('.')[-1]}",
            parameters={
                "path": file_path,
                "app_bundle_id": app,
                "target": file_path,
            },
            raw_action_range=(
                file_action.raw_action_range[0],
                switch_action.raw_action_range[1],
            ),
            confidence=0.85,
        )


# ══════════════════════════════════════════════════════════════════════
# Rule 12: Gather-and-Compose Synthesis (L2 — cross-domain: Clipboard + App)
# ══════════════════════════════════════════════════════════════════════


class GatherComposeSynthesisRule(SynthesisRule):
    """Synthesize multi-source copy-paste sequences into gather_and_compose.

    Pattern:
        [app.switch(src1) + clipboard.copy] + [app.switch(target) + ui.shortcut(paste)]
        repeated 2+ times → gather_and_compose(sources, target, fragments)

    Detects when a user is gathering content from multiple sources into
    a single target app via copy-paste cycles.
    """

    def __init__(self, *, min_fragments: int = 2) -> None:
        self._min_fragments = min_fragments

    def apply(self, actions: List[SemanticAction]) -> List[SemanticAction]:
        if len(actions) < 4:
            return actions

        fragments = self._detect_gather_fragments(actions)
        if len(fragments) < self._min_fragments:
            return actions

        target_app = fragments[0].get("target_app", "")
        if not all(f.get("target_app", "") == target_app for f in fragments):
            return actions

        # Consume all indices from first fragment start to last fragment end
        first_idx = min(f["copy_idx"] for f in fragments)
        last_idx = max(f["paste_idx"] for f in fragments)
        consumed: Set[int] = set(range(first_idx, last_idx + 1))

        result: List[SemanticAction] = []
        gather_inserted = False

        for i, action in enumerate(actions):
            if i in consumed:
                if not gather_inserted:
                    gather_inserted = True
                    result.append(self._synthesize_gather(fragments, actions))
                continue
            result.append(action)

        return result

    def _detect_gather_fragments(
        self, actions: List[SemanticAction]
    ) -> List[Dict[str, Any]]:
        """Detect copy-paste cycles targeting the same destination app."""
        fragments: List[Dict[str, Any]] = []
        i = 0

        while i < len(actions):
            copy_idx = self._find_copy(actions, i)
            if copy_idx is None:
                i += 1
                continue

            paste_info = self._find_paste_after_copy(actions, copy_idx)
            if paste_info is not None:
                paste_idx, target_app = paste_info
                indices = set(range(copy_idx, paste_idx + 1))
                source_app = actions[copy_idx].parameters.get("app_bundle_id", "")
                fragments.append({
                    "indices": indices,
                    "source_app": source_app,
                    "target_app": target_app,
                    "copy_idx": copy_idx,
                    "paste_idx": paste_idx,
                })
                i = paste_idx + 1
            else:
                i = copy_idx + 1

        return fragments

    def _find_copy(self, actions: List[SemanticAction], start: int) -> Optional[int]:
        for i in range(start, len(actions)):
            if actions[i].action_name == "clipboard.copy":
                return i
        return None

    def _find_paste_after_copy(
        self, actions: List[SemanticAction], copy_idx: int
    ) -> Optional[Tuple[int, str]]:
        """Find a paste action (Cmd+V shortcut) after a copy, return (idx, target_app)."""
        for j in range(copy_idx + 1, min(copy_idx + 6, len(actions))):
            action = actions[j]
            if action.action_name == "ui.shortcut" and self._is_paste(action):
                target_app = action.parameters.get("app_bundle_id", "")
                return (j, target_app)
        return None

    @staticmethod
    def _is_paste(action: SemanticAction) -> bool:
        label = action.parameters.get("target_label", "").lower()
        target = action.parameters.get("target", "").lower()
        return "paste" in label or "paste" in target or "⌘v" in target or "cmd+v" in target

    def _synthesize_gather(
        self, fragments: List[Dict[str, Any]], actions: List[SemanticAction]
    ) -> SemanticAction:
        sources = list({f["source_app"] for f in fragments if f["source_app"]})
        target_app = fragments[0]["target_app"]
        first_idx = min(f["copy_idx"] for f in fragments)
        last_idx = max(f["paste_idx"] for f in fragments)

        return SemanticAction(
            action_name="gather_and_compose",
            description=f"Gather from {len(sources)} source(s) into {target_app.split('.')[-1]}",
            parameters={
                "sources": sources,
                "target_app": target_app,
                "fragment_count": len(fragments),
            },
            raw_action_range=(
                actions[first_idx].raw_action_range[0],
                actions[last_idx].raw_action_range[1],
            ),
            confidence=0.8,
        )


# ══════════════════════════════════════════════════════════════════════
# Rule 13: Inotify Cookie Pair (Linux-specific, L1 — FS domain)
# ══════════════════════════════════════════════════════════════════════


class InotifyCookiePairRule(SynthesisRule):
    """Pair inotify MOVED_FROM + MOVED_TO events using cookie matching.

    Linux inotify emits IN_MOVED_FROM and IN_MOVED_TO events with a shared
    cookie for each rename/move operation. This is more precise than the
    heuristic pairing used on macOS (RenamePairRule).

    Falls back to RenamePairRule behavior when cookie is unavailable.
    """

    def apply(self, actions: List[SemanticAction]) -> List[SemanticAction]:
        if len(actions) < 2:
            return actions

        cookie_map: Dict[str, List[int]] = {}
        for i, action in enumerate(actions):
            if action.action_name == "file.rename":
                cookie = action.parameters.get("inotify_cookie", "")
                if cookie:
                    cookie_map.setdefault(cookie, []).append(i)

        if not cookie_map:
            return RenamePairRule().apply(actions)

        consumed: Set[int] = set()
        result: List[SemanticAction] = []

        for i, action in enumerate(actions):
            if i in consumed:
                continue
            if action.action_name == "file.rename":
                cookie = action.parameters.get("inotify_cookie", "")
                if cookie and cookie in cookie_map:
                    pair_indices = cookie_map[cookie]
                    if len(pair_indices) == 2 and i == pair_indices[0]:
                        j = pair_indices[1]
                        consumed.add(j)
                        result.append(self._synthesize_move(action, actions[j]))
                        continue
            result.append(action)

        return result

    @staticmethod
    def _synthesize_move(source_action: SemanticAction, dest_action: SemanticAction) -> SemanticAction:
        source_path = source_action.parameters.get("path", "") or source_action.parameters.get("target", "")
        dest_path = dest_action.parameters.get("path", "") or dest_action.parameters.get("target", "")
        return SemanticAction(
            action_name="file.move",
            description=f"Move {os.path.basename(source_path)} to {os.path.basename(os.path.dirname(dest_path))}",
            parameters={
                "source": source_path,
                "destination": dest_path,
                "path": dest_path,
                "target": dest_path,
            },
            raw_action_range=(
                source_action.raw_action_range[0],
                dest_action.raw_action_range[1],
            ),
            confidence=0.98,
        )


# ══════════════════════════════════════════════════════════════════════
# PlatformSynthesisPass — orchestrator
# ══════════════════════════════════════════════════════════════════════


class PlatformSynthesisPass(AbstractionPass):
    """Synthesize platform-specific low-level events into high-level operations.

    Runs between DenoisePass and GroupingPass in the abstraction pipeline.
    Rules are applied in declared order; each rule's output feeds the next.

    Two-layer rule chain:
        L1 (domain-specific cleanup):
            SystemNoiseRule → AppLaunchNoiseRule → DuplicateFocusRule
            → WindowManagementRule → RenamePairRule → UnpairedRenameDedupRule → BatchMoveRule
        L2 (cross-domain synthesis):
            PermissionNoiseRule → DownloadSynthesisRule → OpenInAppSynthesisRule
            → DragDropRule → GatherComposeSynthesisRule

    Order rationale:
        L1 runs first to clean each domain independently.
        L2 runs on the cleaned stream to detect cross-domain patterns.
    """

    def __init__(self, rules: Optional[Sequence[SynthesisRule]] = None) -> None:
        if rules is not None:
            self._rules = list(rules)
        else:
            self._rules = self._default_rules()

    @staticmethod
    def _default_rules() -> List[SynthesisRule]:
        return [
            # L1: Domain-specific cleanup
            SystemNoiseRule(),
            AppLaunchNoiseRule(),
            DuplicateFocusRule(),
            WindowManagementRule(),
            RenamePairRule(),
            ModifyPairMoveRule(),
            UnpairedRenameDedupRule(),
            BatchMoveRule(),
            # L2: Cross-domain synthesis
            PermissionNoiseRule(),
            DownloadSynthesisRule(),
            OpenInAppSynthesisRule(),
            DragDropRule(),
            GatherComposeSynthesisRule(),
        ]

    @staticmethod
    def for_platform(platform_hint: str) -> "PlatformSynthesisPass":
        """Create a platform-optimized synthesis pass."""
        if platform_hint.startswith("darwin"):
            return PlatformSynthesisPass()
        elif platform_hint.startswith("linux"):
            return PlatformSynthesisPass(rules=[
                # L1
                SystemNoiseRule(patterns=_LINUX_NOISE_PATTERNS),
                DuplicateFocusRule(),
                WindowManagementRule(),
                InotifyCookiePairRule(),
                UnpairedRenameDedupRule(),
                BatchMoveRule(),
                # L2
                PermissionNoiseRule(),
                DownloadSynthesisRule(),
                OpenInAppSynthesisRule(),
                DragDropRule(),
                GatherComposeSynthesisRule(),
            ])
        return PlatformSynthesisPass(rules=[
            # Conservative cross-platform set
            DuplicateFocusRule(),
            WindowManagementRule(),
            RenamePairRule(),
            ModifyPairMoveRule(),
            BatchMoveRule(),
            PermissionNoiseRule(),
            DownloadSynthesisRule(),
            GatherComposeSynthesisRule(),
        ])

    def apply(
        self,
        actions: List[SemanticAction],
        steps: Optional[List[TrajectoryStep]] = None,
    ) -> List[SemanticAction]:
        for rule in self._rules:
            before = len(actions)
            actions = rule.apply(actions)
            if len(actions) < before:
                logger.debug(
                    "synthesis.%s: %d → %d actions",
                    type(rule).__name__, before, len(actions),
                )
        return actions
