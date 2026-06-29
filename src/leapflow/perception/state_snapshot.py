"""Multi-fidelity environment state snapshot service.

Captures environment state at varying levels of detail for use by
the prediction loop's pre/post action comparison.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Dict, Optional, Tuple

from leapflow.domain.events import UINode
from leapflow.memory.providers.episodic import EpisodicMemoryProvider
from leapflow.platform.protocol import HostRpc

logger = logging.getLogger(__name__)


class SnapshotFidelity(IntEnum):
    """Trade-off between information richness and latency/cost.

    Integer ordering guarantees correct ``>=`` comparisons across levels.
    """

    MINIMAL = 0    # ~0ms, zero RPC
    LIGHT = 1      # ~2ms, zero RPC
    STANDARD = 2   # ~50ms, 1 RPC (ax.tree)
    FULL = 3       # ~200ms, 2 RPC (ax.tree + screenshot)


@dataclass(frozen=True)
class StateSnapshot:
    """Immutable point-in-time environment state at a given fidelity."""

    timestamp: float
    fidelity: SnapshotFidelity
    app_bundle_id: str
    window_title: str
    clipboard_text: str
    recent_events: Tuple[str, ...]
    ax_digest: str
    ax_summary: str
    screenshot_phash: str

    def to_prompt_context(self, budget_tokens: int = 300) -> str:
        """Render as an LLM-consumable state description."""
        parts = [f"App: {self.app_bundle_id} | Window: {self.window_title}"]
        if self.recent_events:
            events_str = "; ".join(self.recent_events[:5])
            parts.append(f"Recent: {events_str}")
        if self.ax_summary:
            parts.append(f"UI: {self.ax_summary[:budget_tokens]}")
        if self.clipboard_text:
            parts.append(f"Clipboard: {self.clipboard_text[:100]}")
        return "\n".join(parts)

    def semantic_distance(self, other: StateSnapshot) -> float:
        """Compute weighted semantic distance ∈ [0, 1] between two snapshots."""
        components: list[tuple[float, float]] = []

        # App context change (binary, high weight)
        app_diff = 1.0 if self.app_bundle_id != other.app_bundle_id else 0.0
        components.append((app_diff, 0.30))

        # AX structural diff
        if self.ax_digest and other.ax_digest:
            ax_diff = 0.0 if self.ax_digest == other.ax_digest else 1.0
            components.append((ax_diff, 0.35))

        # Visual diff via perceptual hash
        if self.screenshot_phash and other.screenshot_phash:
            hamming = _hamming_distance(self.screenshot_phash, other.screenshot_phash)
            components.append((min(1.0, hamming / 20.0), 0.25))

        # Clipboard change
        clip_diff = float(self.clipboard_text != other.clipboard_text)
        components.append((clip_diff, 0.10))

        total_weight = sum(w for _, w in components)
        if total_weight == 0:
            return 0.0
        return sum(s * w for s, w in components) / total_weight


class StateSnapshotService:
    """Captures environment snapshots at configurable fidelity levels."""

    def __init__(
        self,
        rpc: HostRpc,
        imm: EpisodicMemoryProvider,
        *,
        default_fidelity: SnapshotFidelity = SnapshotFidelity.LIGHT,
    ) -> None:
        self._rpc = rpc
        self._imm = imm
        self._default_fidelity = default_fidelity
        self._last_focus: Dict[str, str] = {"app": "", "title": ""}

    async def capture(
        self,
        fidelity: Optional[SnapshotFidelity] = None,
    ) -> StateSnapshot:
        """Capture an environment snapshot at the requested fidelity."""
        fid = fidelity or self._default_fidelity
        ts = time.time()

        app_id, title = self._get_focus_info()
        clipboard = ""
        recent: Tuple[str, ...] = ()
        ax_digest = ""
        ax_summary = ""
        phash = ""

        if fid >= SnapshotFidelity.LIGHT:
            recent = self._get_recent_events(limit=5)
            clipboard = await self._get_clipboard()

        if fid >= SnapshotFidelity.STANDARD:
            ax_digest, ax_summary = await self._get_ax_info(app_id)

        if fid >= SnapshotFidelity.FULL:
            phash = await self._get_screenshot_phash()

        return StateSnapshot(
            timestamp=ts,
            fidelity=fid,
            app_bundle_id=app_id,
            window_title=title,
            clipboard_text=clipboard[:256],
            recent_events=recent,
            ax_digest=ax_digest,
            ax_summary=ax_summary,
            screenshot_phash=phash,
        )

    def update_focus(self, app_id: str, title: str) -> None:
        """Called by event bus on focus changes."""
        self._last_focus = {"app": app_id, "title": title}

    def _get_focus_info(self) -> tuple[str, str]:
        if self._last_focus["app"]:
            return self._last_focus["app"], self._last_focus["title"]
        fragments = self._imm.search_fragments(["app.focus_change", "focus"], limit=1)
        if fragments:
            meta = fragments[0].metadata
            return meta.get("app_bundle_id", ""), meta.get("window_title", "")
        return "unknown", "unknown"

    def _get_recent_events(self, limit: int = 5) -> Tuple[str, ...]:
        frags = self._imm.recent(limit=limit)
        return tuple(f"{f.event_type}: {f.content[:80]}" for f in frags)

    async def _get_clipboard(self) -> str:
        try:
            result = await self._rpc.call("clipboard.get")
            if isinstance(result, dict):
                return str(result.get("text", ""))[:256]
            return str(result)[:256] if result else ""
        except Exception:
            return ""

    async def _get_ax_info(self, app_id: str) -> tuple[str, str]:
        try:
            tree = await self._rpc.call("ax.tree", {"app_id": app_id} if app_id else None)
            if isinstance(tree, dict):
                node = _dict_to_ui_node(tree)
                digest = _compute_ax_digest(node)
                summary = _compute_ax_summary(node)
                return digest, summary
        except Exception:
            logger.debug("ax.tree failed for snapshot", exc_info=True)
        return "", ""

    async def _get_screenshot_phash(self) -> str:
        try:
            result = await self._rpc.call("screen.capture_frame")
            if isinstance(result, dict) and "phash" in result:
                return str(result["phash"])
        except Exception:
            logger.debug("screen.capture_frame failed for snapshot", exc_info=True)
        return ""


def _compute_ax_digest(node: UINode, max_depth: int = 3, max_width: int = 5) -> str:
    """Compress AX tree into a structural fingerprint."""
    parts: list[str] = []

    def _walk(n: UINode, depth: int = 0) -> None:
        if depth > max_depth:
            return
        label_part = n.label[:20] if n.label else ""
        parts.append(f"{n.role}:{label_part}")
        for child in (n.children or [])[:max_width]:
            _walk(child, depth + 1)

    _walk(node)
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:16]


def _compute_ax_summary(node: UINode, max_items: int = 8) -> str:
    """Generate a brief natural-language summary of the AX tree."""
    items: list[str] = []

    def _walk(n: UINode, depth: int = 0) -> None:
        if len(items) >= max_items or depth > 2:
            return
        if n.label:
            items.append(f"{n.role}({n.label})")
        for child in (n.children or [])[:4]:
            _walk(child, depth + 1)

    _walk(node)
    return ", ".join(items)


def _dict_to_ui_node(d: Any) -> UINode:
    """Recursively convert a dict to UINode."""
    children = [_dict_to_ui_node(c) for c in (d.get("children") or [])]
    return UINode(
        node_id=d.get("node_id", ""),
        role=d.get("role", ""),
        label=d.get("label", ""),
        value=d.get("value", ""),
        children=children,
    )


def _hamming_distance(a: str, b: str) -> int:
    """Hamming distance between two hex-encoded hashes."""
    if len(a) != len(b):
        return max(len(a), len(b)) * 4
    dist = 0
    for ca, cb in zip(a, b):
        x = int(ca, 16) ^ int(cb, 16)
        dist += bin(x).count("1")
    return dist
