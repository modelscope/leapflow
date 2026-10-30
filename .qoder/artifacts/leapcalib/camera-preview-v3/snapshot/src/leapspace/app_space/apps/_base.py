# Copyright (c) Alibaba, Inc. and its affiliates.
"""BaseLeapApp — abstract base for all LeapSpace scenario apps.

Contract:

- Subclasses implement exactly three methods: ``build_ui()`` (widget tree +
  ``bind`` calls), ``reset(data)`` (apply seed/precondition data), and
  ``state()`` (return the current snapshot as a JSON-serializable dict).
- Subclasses never perform I/O. The base owns all persistence: after every
  ``emit`` it atomically writes ``state.json`` (envelope with app_id,
  version, interface, a11y_violations, data) and ``events.jsonl`` (one
  JSON object per line, seq/ts stamped by the base) into the state dir.
- Subclasses must not override ``__init__`` (enforced). The base template
  runs: build_ui -> title -> self-check -> hook registration ->
  before_launch hooks -> initial persist.
- Task hooks are functions in the task's action.py, wired to hook points by
  the task config (``hooks: {<app>: {<point>: <function>}}``); the harness
  drops ``hooks.json`` (the mapping) and ``hooks.py`` (the action.py copy)
  into the state dir before launch. The base fires ``before_launch``; apps
  fire their own declared points (e.g. chat's ``after_message_sent``).
  ``reset()`` doubles as the task-facing seeding API — a ``before_launch``
  hook calling ``app.reset(data)`` is how a task installs preconditions.
- ``bind`` is a dual contract: an a11y discipline (every interactive widget
  gets an accessible name) and the semantic API that task action.py scripts
  address. Layout may evolve; bound names must not be renamed.
- Self-check violations (unnamed interactive widgets, duplicate bound
  names, unserializable state, hook/reset failures) are recorded into
  ``a11y_violations`` and stderr, never raised: a broken environment must
  stay observable — legality is action.py's verdict, not the app's.

State dir: ``get_sandbox_state_dir``'s fixed per-OS root + ``<app_id>``
(``/tmp/leapspace/<app_id>/`` on Linux) — the same convention the harness
computes from its image preset, so both sides agree without configuration.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import time
import traceback
from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

from PyQt6.QtWidgets import (
    QAbstractButton,
    QAbstractItemView,
    QAbstractSlider,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QLineEdit,
    QMainWindow,
    QMenuBar,
    QPlainTextEdit,
    QScrollBar,
    QTabBar,
    QTextEdit,
    QWidget,
)

from leapspace.app_space.state import get_sandbox_state_dir, write_atomic

# Widget types an agent may act on; self-check requires each to be named.
INTERACTIVE_TYPES: tuple[type[QWidget], ...] = (
    QAbstractButton,
    QAbstractItemView,
    QAbstractSlider,
    QAbstractSpinBox,
    QComboBox,
    QLineEdit,
    QMenuBar,
    QPlainTextEdit,
    QTabBar,
    QTextEdit,
)


# QMainWindow's sip metaclass and ABCMeta conflict at class-creation time;
# combining them is the minimal way to keep @abstractmethod enforcement.
class _LeapAppMeta(type(QMainWindow), ABCMeta):
    pass


class BaseLeapApp(QMainWindow, ABC, metaclass=_LeapAppMeta):
    """Abstract base for scenario apps. See module docstring for the contract."""

    app_id: ClassVar[str]
    app_title: ClassVar[str]
    version: ClassVar[str]

    # App-level affordances this version of the surface offers, e.g.
    # ``("app.chat.v2",)``. Declared, never inferred: a capability provider states
    # which affordances it needs (``requires_environment_affordances``) and the
    # resolver excludes one the environment does not offer, so this is the channel
    # that lets an app version express "the way to do this changed". Deliberately
    # separate from *host* platform capabilities -- an app affordance is a property
    # of the surface, not of the machine. Empty by default: an app that models no
    # affordance simply says nothing.
    affordances: ClassVar[tuple[str, ...]] = ()

    # Hook points this app opens to task hooks; apps extend with their own
    # before_xxx/after_xxx points (a superset tuple, always keeping
    # "before_launch" — the base fires it in the template below).
    supported_hooks: ClassVar[tuple[str, ...]] = ("before_launch",)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if "__init__" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} must not override __init__; "
                "put setup in build_ui() and reset()"
            )

    def __init__(self) -> None:
        super().__init__()
        # Class-level identity is an authoring contract; fail fast if absent.
        self._validate_class_contract()

        # Ground-truth location: fixed per-OS convention shared with the harness.
        self._state_dir = Path(get_sandbox_state_dir(in_sandbox=True)) / self.app_id
        self._state_dir.mkdir(parents=True, exist_ok=True)

        self._events: list[dict[str, Any]] = []
        self._interface: dict[str, QWidget] = {}
        self._violations: list[str] = []
        self._hooks: dict[str, Callable[[BaseLeapApp], Any]] = {}
        self._title_status = ""
        # Probed once so notify() never re-pays a failing subprocess attempt.
        self._notify_available = shutil.which("notify-send") is not None

        # Template order: widgets must exist before the check walks them;
        # before_launch hooks (task seeding) run before the first persist so
        # the initial state.json already reflects the task's preconditions.
        self.build_ui()
        self._update_title()
        self._self_check()
        self._register_hooks()
        self._execute_hooks("before_launch")
        self._persist()

    # ------------------------------------------------------------------
    # Subclass contract
    # ------------------------------------------------------------------

    @abstractmethod
    def build_ui(self) -> None:
        """Create the whole widget tree, wiring interactive widgets via bind()."""

    @abstractmethod
    def reset(self, data: dict[str, Any]) -> None:
        """Apply seed/precondition data to reach a given state (task-facing)."""

    @abstractmethod
    def state(self) -> dict[str, Any]:
        """Return the current snapshot as a JSON-serializable dict."""

    # ------------------------------------------------------------------
    # Base-provided API
    # ------------------------------------------------------------------

    def bind(self, widget: QWidget, name: str, description: str = "") -> QWidget:
        """Register a widget in the semantic interface and set its a11y name."""
        # Duplicate names make semantic addressing ambiguous; record but do
        # not crash — the violation lands in state.json for action.py to judge.
        if name in self._interface:
            self._violations.append(f"duplicate interface name: {name!r}")
        widget.setObjectName(name)
        widget.setAccessibleName(name)
        if description:
            widget.setAccessibleDescription(description)
        self._interface[name] = widget
        return widget

    def emit(self, kind: str, **payload: Any) -> None:
        """Append a stamped event and persist both ground-truth files."""
        # seq/ts are base-stamped so cross-app sequence assertions can rely
        # on one uniform format; subclasses only supply kind + payload.
        self._events.append(
            {"seq": len(self._events), "ts": time.time(), "kind": kind, **payload}
        )
        self._persist()

    def notify(self, title: str, body: str) -> None:
        """Fire a desktop notification (transient signal for the observer).

        The event is recorded either way, with ``delivered`` distinguishing a
        notification that reached the desktop from one that could not (no
        notify-send on the system).
        """
        delivered = False
        if self._notify_available:
            try:
                subprocess.Popen(
                    ["notify-send", "-a", self.app_title, title, body],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                delivered = True
            except FileNotFoundError:
                # Binary vanished between probe and use; stop retrying.
                self._notify_available = False
        self.emit("notification", title=title, body=body, delivered=delivered)

    def set_title_status(self, text: str) -> None:
        """Show extra text in the window title (persistent signal).

        Generic by design: an unread count for chat, "modified" for an
        editor, "5 open" for tickets. Empty text clears it.
        """
        if text == self._title_status:
            return
        self._title_status = text
        self._update_title()
        self.emit("title_status", text=text)

    @property
    def state_dir(self) -> Path:
        return self._state_dir

    @property
    def interface(self) -> tuple[str, ...]:
        return tuple(sorted(self._interface))

    @classmethod
    def launch(cls) -> int:
        """Uniform entry point: ``python3 <app>.py`` with no arguments."""
        app = QApplication(sys.argv)
        app.setApplicationName(cls.app_title)
        window = cls()
        window.show()
        return app.exec()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _validate_class_contract(self) -> None:
        for attr in ("app_id", "app_title", "version"):
            value = getattr(type(self), attr, None)
            if not isinstance(value, str) or not value:
                raise TypeError(
                    f"{type(self).__name__} must define a non-empty str {attr}"
                )
        points = getattr(type(self), "supported_hooks", None)
        if (
            not isinstance(points, tuple)
            or any(not isinstance(point, str) or not point for point in points)
            or "before_launch" not in points
        ):
            raise TypeError(
                f"{type(self).__name__} must define supported_hooks as a tuple "
                "of non-empty str containing 'before_launch'"
            )

    def _update_title(self) -> None:
        # Title is the app's own name (apps are named LeapXxx by convention);
        # the status suffix is the list_windows-visible signal.
        title = self.app_title
        if self._title_status:
            title += f" ({self._title_status})"
        self.setWindowTitle(title)

    def _self_check(self) -> None:
        # Interactive widgets without an accessible name collapse into
        # unaddressable tree nodes — the D1 failure mode, caught statically.
        for widget in self.findChildren(QWidget):
            # QScrollBar is auto-created by scroll areas (not author code)
            # and scrolling is coordinate-based, so naming it is noise.
            if isinstance(widget, QScrollBar):
                continue
            if isinstance(widget, INTERACTIVE_TYPES) and not widget.accessibleName():
                self._violations.append(
                    f"{type(widget).__name__} without accessibleName "
                    f"(objectName={widget.objectName()!r})"
                )
        # Also forces one state() round-trip so serialization violations
        # surface at startup instead of on the first emit.
        self._snapshot_data()
        for violation in self._violations:
            print(f"[{self.app_id}] self-check: {violation}", file=sys.stderr)

    def _register_hooks(self) -> None:
        """Register task hooks declared by the task config.

        The harness drops two files into the state dir before launch:
        ``hooks.json`` (the config's ``hooks`` mapping for this app:
        point -> function name) and ``hooks.py`` (the task's action.py,
        the module the named functions live in). Registration failures —
        bad mapping, undeclared point, missing function, broken import —
        are config/task-authoring defects: recorded, never raised, so the
        environment still boots observable.
        """
        spec_path = self._state_dir / "hooks.json"
        if not spec_path.exists():
            return
        try:
            spec = json.loads(spec_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            self._violations.append(f"invalid hooks.json: {exc}")
            return
        if not isinstance(spec, dict):
            self._violations.append("hooks.json must map hook points to function names")
            return

        hooks_path = self._state_dir / "hooks.py"
        if not hooks_path.exists():
            self._violations.append("hooks.json present but hooks.py missing")
            return
        try:
            import_spec = importlib.util.spec_from_file_location(
                "leapspace_task_hooks", hooks_path
            )
            module = importlib.util.module_from_spec(import_spec)
            import_spec.loader.exec_module(module)
        except Exception as exc:
            self._violations.append(f"hooks.py failed to import: {exc}")
            traceback.print_exc()
            return

        for point, fn_name in spec.items():
            if point not in self.supported_hooks:
                self._violations.append(
                    f"hook point {point!r} not declared in supported_hooks"
                )
                continue
            fn = getattr(module, fn_name, None)
            if not callable(fn):
                self._violations.append(
                    f"hook function {fn_name!r} not found in hooks.py"
                )
                continue
            self._hooks[point] = fn

    def _execute_hooks(self, point: str) -> None:
        """Run the registered task hook at a declared point (app-internal).

        Only the app itself calls this — the base fires ``before_launch``
        in the template, apps fire their own declared points. An undeclared
        point is an app-authoring bug and fails fast; a failing hook is a
        broken task environment and is recorded, never raised.
        """
        if point not in self.supported_hooks:
            raise ValueError(
                f"{type(self).__name__} does not declare hook point {point!r}"
            )
        fn = self._hooks.get(point)
        if fn is None:
            return
        try:
            fn(self)
        except Exception as exc:
            self._violations.append(f"hook {point!r} failed: {exc}")
            traceback.print_exc()

    def _snapshot_data(self) -> dict[str, Any]:
        # state() is subclass code and may break at runtime (especially under
        # agent-driven evolution); degrade to an error record, never crash.
        try:
            data = self.state()
            json.dumps(data)
            return data
        except Exception as exc:
            message = f"state() failed: {exc}"
            if message not in self._violations:
                self._violations.append(message)
            return {"error": message}

    def _persist(self) -> None:
        envelope = {
            "app_id": self.app_id,
            "app_title": self.app_title,
            "version": self.version,
            "interface": sorted(self._interface),
            # App-level affordances this version declares (see the ClassVar). Part of
            # the ground truth because an environment probe compares them across
            # versions to tell an affordance migration from a pure rename.
            "affordances": list(type(self).affordances),
            # Role-aware structural view of the bound interface, derived from the
            # live widget tree (not the declared name list). This is the AX-like
            # signal LeapFlow's perception reads to tell one control from another
            # and to name *which* element changed -- a renamed or removed control
            # shows up here as a changed role/name set, which a bare name list can
            # show only as "the set changed". Additive: consumers that read only
            # ``interface`` are unaffected.
            "elements": self._structure(),
            "a11y_violations": list(self._violations),
            "data": self._snapshot_data(),
        }
        write_atomic(
            self._state_dir / "state.json", json.dumps(envelope, indent=2) + "\n"
        )
        # Full rewrite is fine at benchmark event volumes and keeps the file
        # atomic per write (no torn lines for the assertion side).
        write_atomic(
            self._state_dir / "events.jsonl",
            "".join(json.dumps(event) + "\n" for event in self._events),
        )

    def _structure(self) -> list[dict[str, Any]]:
        """Role-aware view of the bound interface: ``[{name, role, enabled}]``.

        ``role`` is the Qt widget class (``QPushButton``, ``QLineEdit``, ...), the
        closest headless analogue of an accessibility role. Derived from the live
        widget objects ``bind`` recorded, so it tracks the real UI rather than a
        declaration. Each widget read is guarded: a persist during teardown (the
        C++ object already gone) must degrade to a stable placeholder, never crash
        the ground-truth write.
        """
        structure: list[dict[str, Any]] = []
        for name in sorted(self._interface):
            widget = self._interface[name]
            try:
                role = type(widget).__name__
                enabled = bool(widget.isEnabled())
            except RuntimeError:
                role, enabled = "QWidget", True
            structure.append({"name": name, "role": role, "enabled": enabled})
        return structure
