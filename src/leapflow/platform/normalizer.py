"""Event normalization engine — transforms platform-specific raw events into SystemEvent."""

from __future__ import annotations

import time
from dataclasses import replace
from fnmatch import fnmatch
from typing import Any, Callable, Dict, List, Tuple

from leapflow.domain.events import (
    PRIORITY_CRITICAL,
    PRIORITY_DEFERRED,
    PRIORITY_HIGH,
    PRIORITY_LOW,
    PRIORITY_NORMAL,
    SystemEvent,
)
from leapflow.domain.platform import PlatformID, PlatformManifest

# ── macOS FSEvent flag constants (per CoreServices/FSEvents.h) ──
# Item-level flags surfaced by FSEventStream. We model 13 of them — enough
# to recover full semantic intent when Swift forwards a raw flags bitmap.
_FSEVENT_CREATED = 0x00000100         # kFSEventStreamEventFlagItemCreated
_FSEVENT_REMOVED = 0x00000200         # kFSEventStreamEventFlagItemRemoved
_FSEVENT_INODE_META = 0x00000400      # kFSEventStreamEventFlagItemInodeMetaMod
_FSEVENT_RENAMED = 0x00000800         # kFSEventStreamEventFlagItemRenamed
_FSEVENT_MODIFIED = 0x00001000        # kFSEventStreamEventFlagItemModified
_FSEVENT_FINDER_INFO = 0x00002000     # kFSEventStreamEventFlagItemFinderInfoMod
_FSEVENT_CHANGE_OWNER = 0x00004000    # kFSEventStreamEventFlagItemChangeOwner
_FSEVENT_XATTR_MOD = 0x00008000       # kFSEventStreamEventFlagItemXattrMod
_FSEVENT_IS_FILE = 0x00010000         # kFSEventStreamEventFlagItemIsFile
_FSEVENT_IS_DIR = 0x00020000          # kFSEventStreamEventFlagItemIsDir
_FSEVENT_IS_SYMLINK = 0x00040000      # kFSEventStreamEventFlagItemIsSymlink
_FSEVENT_IS_HARDLINK = 0x00100000     # kFSEventStreamEventFlagItemIsHardlink
_FSEVENT_CLONED = 0x00400000          # kFSEventStreamEventFlagItemCloned

# Ordered (flag, semantic_action) table. Order defines preference for the
# legacy single-action API — the first matching flag wins. ``_decode_fs_flags``
# walks the full list and collects every match, preserving this order.
_FS_FLAG_ACTIONS: Tuple[Tuple[int, str], ...] = (
    (_FSEVENT_CREATED, "created"),
    (_FSEVENT_REMOVED, "deleted"),
    (_FSEVENT_RENAMED, "renamed"),
    (_FSEVENT_MODIFIED, "modified"),
    (_FSEVENT_INODE_META, "metadata_modified"),
    (_FSEVENT_FINDER_INFO, "finder_info_modified"),
    (_FSEVENT_CHANGE_OWNER, "owner_changed"),
    (_FSEVENT_XATTR_MOD, "xattr_modified"),
    (_FSEVENT_CLONED, "cloned"),
)

# Linux inotify mask constants (subset, used in the legacy fallback path).
_INOTIFY_CREATE = 0x00000100
_INOTIFY_DELETE = 0x00000200
_INOTIFY_MOVED_TO = 0x00000040

# ── Event-type → base priority lookup table ──
# A pure data-driven map keeps the policy declarative; ``_resolve_priority``
# applies sub-type refinements on top (e.g. ui.action click → CRITICAL).
_EVENT_PRIORITY_MAP: Dict[str, int] = {
    "ui.action": PRIORITY_CRITICAL,
    "app.focus_change": PRIORITY_HIGH,
    "context.change": PRIORITY_HIGH,
    "intent.signal": PRIORITY_HIGH,
    "clipboard.change": PRIORITY_NORMAL,
    "ui.scroll": PRIORITY_NORMAL,
    "fs.change": PRIORITY_LOW,
}

# UI sub-types that always count as critical user interaction.
_UI_CRITICAL_SUBTYPES = frozenset({
    "click", "double_click", "shortcut", "drag", "type", "keyboard",
})
# UI sub-types that are explicitly demoted to NORMAL (high-frequency, low signal).
_UI_NORMAL_SUBTYPES = frozenset({"scroll"})


def _resolve_priority(event_type: str, payload: Dict[str, Any]) -> int:
    """Compute the dispatch priority for a normalized event.

    Strategy:
    1. Look up base priority in :data:`_EVENT_PRIORITY_MAP`.
    2. For ``ui.action``, refine via the ``sub_type`` field: critical clicks
       and keystrokes stay CRITICAL; passive scrolls drop to NORMAL.
    3. Unknown event types fall back to :data:`PRIORITY_DEFERRED`.
    """
    base = _EVENT_PRIORITY_MAP.get(event_type)
    if base is None:
        return PRIORITY_DEFERRED
    if event_type == "ui.action":
        sub_type = str(payload.get("sub_type", ""))
        if sub_type in _UI_CRITICAL_SUBTYPES:
            return PRIORITY_CRITICAL
        if sub_type in _UI_NORMAL_SUBTYPES:
            return PRIORITY_NORMAL
    return base


def _decode_fs_flags(flags: int) -> List[str]:
    """Decode an FSEvent ``flags`` bitmap into the list of semantic actions.

    Returns every action whose flag bit is set, in declaration order. An
    empty list signals a no-op or pure type-indicator bitmap (IsFile/IsDir).
    """
    if not isinstance(flags, int) or flags <= 0:
        return []
    return [action for bit, action in _FS_FLAG_ACTIONS if flags & bit]


class EventNormalizer:
    """Converts raw platform events into unified SystemEvent instances.

    The normalizer is stateless per-event; platform identity only affects
    interpretation heuristics (e.g. FSEvent flags vs inotify masks).
    """

    def __init__(
        self,
        manifest: PlatformManifest,
        *,
        ax_enhancer: "AppSemanticEnhancer | None" = None,
    ) -> None:
        self._manifest = manifest
        self._ax_enhancer = ax_enhancer

    @property
    def platform_id(self) -> PlatformID:
        return self._manifest.platform_id

    def normalize(self, event_type: str, payload: Dict[str, Any]) -> SystemEvent:
        """Route to the appropriate normalizer based on event type."""
        if event_type in ("event.fs_change", "fs_change"):
            event = self._normalize_fs(payload)
        elif event_type in ("event.clipboard_change", "clipboard_change"):
            event = self._normalize_clipboard(payload)
        elif event_type in ("event.app_focus_change", "app_focus_change"):
            event = self._normalize_focus(payload)
        elif event_type in ("event.ui_action", "ui_action"):
            event = self._normalize_ui_action(payload)
        elif event_type in ("event.context_change", "context_change"):
            event = self._normalize_context(payload)
        elif event_type == "event.intent_signal":
            event = self._normalize_intent(payload)
        else:
            event = SystemEvent(
                event_type="internal.unmapped",
                source=str(payload.get("source", event_type)),
                payload={"_original_type": event_type, **payload},
                timestamp=payload.get("ts", time.time()),
                platform_hint=self._manifest.platform_id.value,
            )
        # Attach computed priority via dataclasses.replace (SystemEvent is frozen).
        priority = _resolve_priority(event.event_type, event.payload)
        if priority == event.priority:
            return event
        return replace(event, priority=priority)

    def _normalize_fs(self, payload: Dict[str, Any]) -> SystemEvent:
        path = str(payload.get("path", ""))
        actions = self._resolve_fs_actions(payload)
        primary = actions[0] if actions else "modified"
        normalized: Dict[str, Any] = {
            "path": path,
            "action": primary,
            "semantic_actions": actions,
            "raw_flags": payload.get("flags"),
        }
        return SystemEvent(
            event_type="fs.change",
            source=path,
            payload=normalized,
            timestamp=payload.get("ts", time.time()),
            platform_hint=self._manifest.platform_id.value,
        )

    def _normalize_clipboard(self, payload: Dict[str, Any]) -> SystemEvent:
        text = str(payload.get("text", ""))
        return SystemEvent(
            event_type="clipboard.change",
            source="system.clipboard",
            payload={"text": text, "char_count": len(text)},
            timestamp=payload.get("change_ts", payload.get("ts", time.time())),
            platform_hint=self._manifest.platform_id.value,
        )

    def _normalize_focus(self, payload: Dict[str, Any]) -> SystemEvent:
        bundle_id = str(payload.get("bundle_id", ""))
        app_name = str(payload.get("app_name", bundle_id))
        normalized: Dict[str, Any] = {"bundle_id": bundle_id, "app_name": app_name}
        window_title = payload.get("window_title")
        if window_title:
            normalized["window_title"] = str(window_title)
        return SystemEvent(
            event_type="app.focus_change",
            source=bundle_id,
            payload=normalized,
            timestamp=payload.get("ts", time.time()),
            platform_hint=self._manifest.platform_id.value,
        )

    def _normalize_intent(self, payload: Dict[str, Any]) -> SystemEvent:
        return SystemEvent(
            event_type="intent.signal",
            source=str(payload.get("app_bundle_id", "unknown")),
            payload=payload,
            timestamp=payload.get("ts", time.time()),
            platform_hint=self._manifest.platform_id.value,
        )

    def _normalize_ui_action(self, payload: Dict[str, Any]) -> SystemEvent:
        """Normalize raw ui_action events into canonical ui.action SystemEvents."""
        action = str(payload.get("action", "unknown"))
        app_bundle_id = str(payload.get("app_bundle_id", ""))
        ts = payload.get("timestamp", time.time())

        # Drag complete → ui.action with sub_type="drag"
        if action == "drag_complete":
            return self._normalize_drag(payload, app_bundle_id, ts)

        element_role = str(payload.get("element_role", ""))
        element_label = str(payload.get("element_label", ""))
        element_id = str(payload.get("element_id", ""))

        normalized: Dict[str, Any] = {
            "sub_type": action,
            "role": element_role,
            "label": element_label,
            "node_id": element_id,
            "app_bundle_id": app_bundle_id,
        }

        if action == "click":
            normalized["mouse_x"] = payload.get("mouse_x", 0)
            normalized["mouse_y"] = payload.get("mouse_y", 0)
            normalized = self._enhance_ax_semantics(app_bundle_id, normalized)
        elif action in ("type", "shortcut"):
            normalized["key_code"] = payload.get("key_code", 0)
            normalized["modifiers"] = payload.get("modifiers", [])
            char = payload.get("char")
            if char:
                normalized["char"] = str(char)
        elif action == "scroll":
            normalized["delta_x"] = payload.get("delta_x", 0)
            normalized["delta_y"] = payload.get("delta_y", 0)
            normalized["mouse_x"] = payload.get("mouse_x", 0)
            normalized["mouse_y"] = payload.get("mouse_y", 0)

        return SystemEvent(
            event_type="ui.action",
            source=app_bundle_id or "unknown",
            payload=normalized,
            timestamp=ts,
            platform_hint=self._manifest.platform_id.value,
        )

    def _normalize_drag(
        self, payload: Dict[str, Any], app_bundle_id: str, ts: float,
    ) -> SystemEvent:
        """Normalize drag_complete into ui.action with sub_type='drag'."""
        normalized: Dict[str, Any] = {
            "sub_type": "drag",
            "app_bundle_id": app_bundle_id,
            "start_x": payload.get("start_x", 0),
            "start_y": payload.get("start_y", 0),
            "end_x": payload.get("end_x", 0),
            "end_y": payload.get("end_y", 0),
            "role": str(payload.get("start_element_role", "")),
            "label": str(payload.get("start_element_label", "")),
            "node_id": str(payload.get("start_element_id", "")),
            "end_role": str(payload.get("end_element_role", "")),
            "end_label": str(payload.get("end_element_label", "")),
            "end_node_id": str(payload.get("end_element_id", "")),
            "start_app": str(payload.get("start_app", "")),
            "end_app": str(payload.get("end_app", "")),
            "cross_app": bool(payload.get("cross_app", False)),
        }
        return SystemEvent(
            event_type="ui.action",
            source=app_bundle_id or "unknown",
            payload=normalized,
            timestamp=ts,
            platform_hint=self._manifest.platform_id.value,
        )

    def _normalize_context(self, payload: Dict[str, Any]) -> SystemEvent:
        """Normalize context_change events (window title changes, etc.)."""
        bundle_id = str(payload.get("bundle_id", ""))
        normalized: Dict[str, Any] = {
            "bundle_id": bundle_id,
            "window_title": str(payload.get("window_title", "")),
        }
        url = payload.get("url")
        if url:
            normalized["url"] = str(url)
        return SystemEvent(
            event_type="context.change",
            source=bundle_id,
            payload=normalized,
            timestamp=payload.get("ts", time.time()),
            platform_hint=self._manifest.platform_id.value,
        )

    def _enhance_ax_semantics(
        self, bundle_id: str, normalized: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply app-specific AX semantic enhancement (best-effort)."""
        enhancer = self._ax_enhancer or AppSemanticEnhancer.instance()
        return enhancer.enhance(bundle_id, normalized)

    def _resolve_fs_actions(self, payload: Dict[str, Any]) -> List[str]:
        """Return the ordered list of semantic FS actions for a raw payload.

        Preference order:
        1. Swift-side ``semantic_actions`` list (already decoded).
        2. Single ``semantic_action`` string (legacy single-value API).
        3. macOS ``flags`` bitmap decoded via :func:`_decode_fs_flags`.
        4. Linux ``mask`` (inotify) translated to canonical actions.
        """
        forwarded = payload.get("semantic_actions")
        if isinstance(forwarded, (list, tuple)) and forwarded:
            return [str(a) for a in forwarded]

        single = payload.get("semantic_action")
        if isinstance(single, str) and single:
            return [single]

        flags = payload.get("flags", 0)
        if isinstance(flags, int) and flags:
            decoded = _decode_fs_flags(flags)
            if decoded:
                return decoded

        mask = payload.get("mask", 0)
        if isinstance(mask, int) and mask:
            inotify_actions: List[str] = []
            if mask & _INOTIFY_CREATE:
                inotify_actions.append("created")
            if mask & _INOTIFY_DELETE:
                inotify_actions.append("deleted")
            if mask & _INOTIFY_MOVED_TO:
                inotify_actions.append("renamed")
            if inotify_actions:
                return inotify_actions

        return ["modified"]

    def _infer_fs_action(self, payload: Dict[str, Any]) -> str:
        """Backward-compatible single-action inference (delegates to the list API)."""
        actions = self._resolve_fs_actions(payload)
        return actions[0] if actions else "modified"


# ══════════════════════════════════════════════════════════════════════
# App-specific AX semantic enhancer
# ══════════════════════════════════════════════════════════════════════

EnhancerFn = Callable[[Dict[str, Any]], Dict[str, Any]]


class AppSemanticEnhancer:
    """Best-effort AX semantic enrichment for apps with poor AX quality.

    Uses bundle_id glob patterns to dispatch to app-specific enhancers.
    Enhancers MUST NOT remove or overwrite existing fields — only add new ones.
    If no enhancer matches, the input is returned unchanged.
    """

    _instance: "AppSemanticEnhancer | None" = None

    def __init__(self) -> None:
        self._enhancers: List[Tuple[str, EnhancerFn]] = []
        self._register_defaults()

    @classmethod
    def instance(cls) -> "AppSemanticEnhancer":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(self, bundle_pattern: str, enhancer: EnhancerFn) -> None:
        """Register an enhancer for a bundle_id glob pattern."""
        self._enhancers.append((bundle_pattern, enhancer))

    def enhance(self, bundle_id: str, info: Dict[str, Any]) -> Dict[str, Any]:
        """Apply matching enhancers to the AX info dict."""
        if not bundle_id:
            return info
        for pattern, enhancer in self._enhancers:
            if fnmatch(bundle_id, pattern):
                info = enhancer(info)
                break
        return info

    def _register_defaults(self) -> None:
        self.register("com.google.Chrome*", _enhance_chrome)
        self.register("com.apple.Safari*", _enhance_browser)
        self.register("com.microsoft.VSCode*", _enhance_vscode)
        self.register("com.alibaba-inc.DingTalk*", _enhance_electron)
        self.register("com.tinyspeck.slackmacgap*", _enhance_electron)


def _enhance_chrome(info: Dict[str, Any]) -> Dict[str, Any]:
    role = info.get("role", "")
    if role == "AXWebArea":
        info["semantic_context"] = "web_content"
    elif role == "AXTextField":
        identifier = info.get("node_id", "")
        if "address" in identifier or "url" in identifier:
            info["semantic_role"] = "url_bar"
    return info


def _enhance_browser(info: Dict[str, Any]) -> Dict[str, Any]:
    role = info.get("role", "")
    if role == "AXWebArea":
        info["semantic_context"] = "web_content"
    return info


def _enhance_vscode(info: Dict[str, Any]) -> Dict[str, Any]:
    role = info.get("role", "")
    if role == "AXTab":
        info["semantic_role"] = "editor_tab"
    elif role == "AXTextArea":
        info["semantic_role"] = "code_editor"
    return info


def _enhance_electron(info: Dict[str, Any]) -> Dict[str, Any]:
    role = info.get("role", "")
    label = info.get("label", "")
    if role in ("AXGroup", "AXWebArea") and not label:
        info["semantic_context"] = "electron_unlabeled"
    return info
