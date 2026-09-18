# Copyright (c) Alibaba, Inc. and its affiliates.
"""LeapSpaceHostRpc — serve LeapFlow's HostRpc from leapspace ground truth.

EVO-02 LS-3. The missing adapter that lets the *shipped* LeapFlow perception
(``StateSnapshotService``) observe a leapspace environment directly, instead of
LeapFlow's snapshots being disconnected from the environment the apps actually
render. It reads the ground truth ``BaseLeapApp`` already writes -- ``state.json``
per app under the shared state root -- and answers the ``HostRpc`` method set
LeapFlow calls (``ax.tree``, ``window.active``, ``clipboard.get``, ``app.list``).

Design constraints (executable, not aspirational):

* **Stdlib only, no host SDK.** It reads JSON files; it never imports
  ``cua_sandbox``. So it runs headless -- an offscreen Qt app (or any producer of
  the real envelope) writes ``state.json`` and LeapFlow observes it, on any host,
  with no hypervisor.
* **No LeapFlow import.** It satisfies the ``HostRpc`` Protocol structurally
  (one ``async call`` method), keeping the dependency direction one-way: leapspace
  never imports engine/registry/perception. The connection is a discoverable
  adapter, which is what "Everything Is a Plugin" requires here.
* **Never raises.** An unknown method or a missing/corrupt ``state.json`` returns
  a structured empty result, matching how ``StateSnapshotService`` already treats
  RPC facets that degrade -- an absent environment lowers fidelity, it does not
  fail a turn.

Pixels are deliberately absent (``screen.capture_frame`` returns ``{}``): a
headless run has no framebuffer. The real-AX/pixel tier belongs to the VM lane
where the CUA driver is present; this adapter delivers the structural + event
signal that lane and this one share.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class LeapSpaceHostRpc:
    """A ``HostRpc`` backed by the app state root a leapspace run writes to.

    ``state_root`` is the directory holding one ``<app_id>/`` subdir per running
    app (the ``get_sandbox_state_dir`` convention); each subdir carries the app's
    atomically written ``state.json`` and ``events.jsonl``.
    """

    def __init__(self, state_root: Path | str) -> None:
        self._root = Path(state_root)

    async def call(self, method: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Answer one HostRpc method from the on-disk ground truth."""
        params = dict(params or {})
        if method == "ax.tree":
            return self._ax_tree(str(params.get("app_id") or ""))
        if method == "window.active":
            return self._window_active()
        if method == "clipboard.get":
            # A leapspace app models no system clipboard; report empty rather
            # than fabricate, so the snapshot's clipboard facet is honestly blank.
            return {"text": ""}
        if method == "app.list":
            return {"apps": self._app_ids()}
        if method == "screen.capture_frame":
            # No framebuffer headless; the phash facet degrades to empty.
            return {}
        # Unknown method: structured, non-raising, matching the graceful-degrade
        # contract every HostRpc caller in LeapFlow already assumes.
        return {"error": "unsupported_method", "method": str(method)}

    # ── ground-truth reads ────────────────────────────────────────────────

    def _ax_tree(self, app_id: str) -> Dict[str, Any]:
        """Structural element list for one app, in the shape ax.tree consumers read.

        Prefers an enriched ``elements`` channel (role + name), which a
        role-aware ``BaseLeapApp`` can emit; falls back to the always-present
        ``interface`` names so the digest is meaningful even before that
        enrichment lands. Either way a rename/removal changes the element set,
        which is exactly the structural signal a hash-only ``ax_digest`` needs.
        """
        envelope = self._read_state(app_id) if app_id else self._read_active_state()
        if not envelope:
            return {"elements": []}
        elements = _elements_from_envelope(envelope)
        return {
            "app_id": str(envelope.get("app_id") or ""),
            "app_title": str(envelope.get("app_title") or ""),
            "version": str(envelope.get("version") or ""),
            "elements": elements,
        }

    def _window_active(self) -> Dict[str, Any]:
        envelope = self._read_active_state()
        if not envelope:
            return {"app_id": "", "title": ""}
        return {
            "app_id": str(envelope.get("app_id") or ""),
            "title": str(envelope.get("app_title") or ""),
        }

    def _app_ids(self) -> list[str]:
        if not self._root.exists():
            return []
        return sorted(
            child.name
            for child in self._root.iterdir()
            if child.is_dir()
            and not child.name.startswith(".")
            and (child / "state.json").exists()
        )

    def _read_state(self, app_id: str) -> Dict[str, Any]:
        path = self._root / app_id / "state.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return {}
        return data if isinstance(data, dict) else {}

    def _read_active_state(self) -> Dict[str, Any]:
        """The most recently modified app state -- the closest thing to focus.

        A single-app run has exactly one; a multi-app run reports the one whose
        ground truth changed last, which is the app the last action touched.
        """
        latest: tuple[float, Dict[str, Any]] | None = None
        for app_id in self._app_ids():
            path = self._root / app_id / "state.json"
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            envelope = self._read_state(app_id)
            if envelope and (latest is None or mtime > latest[0]):
                latest = (mtime, envelope)
        return latest[1] if latest is not None else {}


def _elements_from_envelope(envelope: Dict[str, Any]) -> list[Dict[str, str]]:
    """Project a state envelope into ``[{role, label}]`` for ax.tree consumers."""
    raw = envelope.get("elements")
    if isinstance(raw, list) and raw:
        rendered: list[Dict[str, str]] = []
        for item in raw:
            if isinstance(item, dict):
                label = str(item.get("name") or item.get("label") or "")
                if label:
                    rendered.append({"role": str(item.get("role") or "widget"), "label": label})
        if rendered:
            return rendered
    # Fallback: the always-present interface names. Role is unknown headless, so
    # it is reported uniformly; the label carries the identity a delta keys on.
    interface = envelope.get("interface")
    if isinstance(interface, list):
        return [{"role": "widget", "label": str(name)} for name in interface if str(name)]
    return []


__all__ = ["LeapSpaceHostRpc"]
