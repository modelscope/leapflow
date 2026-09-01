"""Hermetic unit tests for BaseLeapApp (offscreen Qt, tmp state dirs)."""

import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PyQt6")  # leapspace dependency group only

from PyQt6.QtWidgets import QApplication, QLabel, QLineEdit, QPushButton, QWidget  # noqa: E402

from leapspace.app_space.apps import BaseLeapApp  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class MiniApp(BaseLeapApp):
    app_id = "mini"
    app_title = "Mini"
    version = "1"

    def build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        self.input = self.bind(QLineEdit(), "message_input")
        self.send = self.bind(QPushButton("Send"), "send_button")
        self.status = QLabel("ready")
        self.sent: list[str] = []
        self.loaded_data: dict | None = None

    def reset(self, data: dict) -> None:
        self.loaded_data = data

    def state(self) -> dict:
        return {"sent": list(self.sent)}


@pytest.fixture()
def app(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("LEAPSPACE_STATE_DIR", str(tmp_path))
    window = MiniApp()
    yield window, tmp_path
    window.deleteLater()


def read_state(state_dir):
    return json.loads((state_dir / "state.json").read_text())


def read_events(state_dir):
    text = (state_dir / "events.jsonl").read_text()
    return [json.loads(line) for line in text.splitlines() if line]


def test_abstract_methods_enforced(qapp):
    class Incomplete(BaseLeapApp):
        app_id = "bad"
        app_title = "Bad"
        version = "1"

    with pytest.raises(TypeError):
        Incomplete()


def test_init_override_forbidden():
    with pytest.raises(TypeError, match="must not override __init__"):

        class Naughty(BaseLeapApp):
            app_id = "naughty"
            app_title = "Naughty"
            version = "1"

            def __init__(self):
                pass

            def build_ui(self):
                pass

            def reset(self, fixture):
                pass

            def state(self):
                return {}


def test_missing_classvars_rejected(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("LEAPSPACE_STATE_DIR", str(tmp_path))

    class NoId(MiniApp):
        app_id = ""

    with pytest.raises(TypeError, match="app_id"):
        NoId()


def test_initial_persist_and_title(app):
    window, state_dir = app
    envelope = read_state(state_dir)
    assert envelope["app_id"] == "mini"
    assert envelope["version"] == "1"
    assert envelope["interface"] == ["message_input", "send_button"]
    assert envelope["a11y_violations"] == []
    assert envelope["data"] == {"sent": []}
    assert read_events(state_dir) == []
    assert window.windowTitle() == "Mini"


def test_unnamed_interactive_widget_recorded_not_fatal(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("LEAPSPACE_STATE_DIR", str(tmp_path))

    class Sloppy(MiniApp):
        def build_ui(self):
            super().build_ui()
            QPushButton("orphan", self)

    window = Sloppy()
    violations = read_state(tmp_path)["a11y_violations"]
    assert any("without accessibleName" in v for v in violations)
    window.deleteLater()


def test_duplicate_bind_name_recorded(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("LEAPSPACE_STATE_DIR", str(tmp_path))

    class Dupe(MiniApp):
        def build_ui(self):
            super().build_ui()
            self.bind(QPushButton("again"), "send_button")

    window = Dupe()
    violations = read_state(tmp_path)["a11y_violations"]
    assert any("duplicate interface name" in v for v in violations)
    window.deleteLater()


def test_emit_stamps_and_persists(app):
    window, state_dir = app
    window.emit("message_sent", text="hi")
    window.emit("message_sent", text="again")
    events = read_events(state_dir)
    assert [e["seq"] for e in events] == [0, 1]
    assert all(e["kind"] == "message_sent" and e["ts"] > 0 for e in events)
    assert events[0]["text"] == "hi"


def test_state_snapshot_updates_on_emit(app):
    window, state_dir = app
    window.sent.append("hi")
    window.emit("message_sent")
    assert read_state(state_dir)["data"] == {"sent": ["hi"]}


def test_unserializable_state_recorded(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("LEAPSPACE_STATE_DIR", str(tmp_path))

    class BadState(MiniApp):
        def state(self):
            return {"obj": object()}

    window = BadState()
    envelope = read_state(tmp_path)
    assert any("state() failed" in v for v in envelope["a11y_violations"])
    assert "error" in envelope["data"]
    window.deleteLater()


def write_hooks(state_dir, mapping, source=None):
    (state_dir / "hooks.json").write_text(json.dumps(mapping))
    if source is not None:
        (state_dir / "hooks.py").write_text(source)


def test_before_launch_hook_seeds_via_reset(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("LEAPSPACE_STATE_DIR", str(tmp_path))
    write_hooks(
        tmp_path,
        {"before_launch": "seed"},
        "def seed(app):\n    app.reset({'unread': 3})\n",
    )
    window = MiniApp()
    assert window.loaded_data == {"unread": 3}
    assert read_state(tmp_path)["a11y_violations"] == []
    window.deleteLater()


def test_hook_unknown_point_recorded(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("LEAPSPACE_STATE_DIR", str(tmp_path))
    write_hooks(
        tmp_path, {"after_whatever": "seed"}, "def seed(app):\n    pass\n"
    )
    window = MiniApp()
    violations = read_state(tmp_path)["a11y_violations"]
    assert any("not declared in supported_hooks" in v for v in violations)
    window.deleteLater()


def test_hook_missing_function_recorded(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("LEAPSPACE_STATE_DIR", str(tmp_path))
    write_hooks(tmp_path, {"before_launch": "nope"}, "def seed(app):\n    pass\n")
    window = MiniApp()
    violations = read_state(tmp_path)["a11y_violations"]
    assert any("not found in hooks.py" in v for v in violations)
    window.deleteLater()


def test_hook_exception_recorded_not_fatal(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("LEAPSPACE_STATE_DIR", str(tmp_path))
    write_hooks(
        tmp_path,
        {"before_launch": "seed"},
        "def seed(app):\n    raise RuntimeError('boom')\n",
    )
    window = MiniApp()
    violations = read_state(tmp_path)["a11y_violations"]
    assert any("hook 'before_launch' failed: boom" in v for v in violations)
    window.deleteLater()


def test_hooks_py_missing_recorded(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("LEAPSPACE_STATE_DIR", str(tmp_path))
    write_hooks(tmp_path, {"before_launch": "seed"})
    window = MiniApp()
    violations = read_state(tmp_path)["a11y_violations"]
    assert any("hooks.py missing" in v for v in violations)
    window.deleteLater()


def test_execute_undeclared_point_raises(app):
    window, _ = app
    with pytest.raises(ValueError, match="does not declare hook point"):
        window._execute_hooks("after_whatever")


def test_app_declared_point_fires(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("LEAPSPACE_STATE_DIR", str(tmp_path))
    write_hooks(
        tmp_path,
        {"after_ping": "on_ping"},
        "def on_ping(app):\n    app.sent.append('pong')\n",
    )

    class PingApp(MiniApp):
        supported_hooks = ("before_launch", "after_ping")

        def ping(self):
            self._execute_hooks("after_ping")

    window = PingApp()
    window.ping()
    assert window.sent == ["pong"]
    window.deleteLater()


def test_supported_hooks_must_keep_before_launch(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv("LEAPSPACE_STATE_DIR", str(tmp_path))

    class NoLaunch(MiniApp):
        supported_hooks = ()

    with pytest.raises(TypeError, match="supported_hooks"):
        NoLaunch()


def test_set_title_status_updates_title_and_emits(app):
    window, state_dir = app
    window.set_title_status("3")
    assert window.windowTitle() == "Mini (3)"
    window.set_title_status("")
    assert window.windowTitle() == "Mini"
    events = read_events(state_dir)
    assert [(e["kind"], e["text"]) for e in events] == [
        ("title_status", "3"),
        ("title_status", ""),
    ]


def test_notify_delivered_when_binary_available(app, monkeypatch):
    window, state_dir = app
    calls = []

    class FakeProc:
        pass

    monkeypatch.setattr(
        "leapspace.app_space.apps._base.subprocess.Popen", lambda *a, **k: calls.append(a) or FakeProc()
    )
    window._notify_available = True
    window.notify("hello", "world")
    assert calls, "notify-send should be invoked when available"
    event = read_events(state_dir)[-1]
    assert event["kind"] == "notification" and event["delivered"] is True


def test_notify_undelivered_when_binary_missing(app, monkeypatch):
    window, state_dir = app
    monkeypatch.setattr(
        "leapspace.app_space.apps._base.subprocess.Popen",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not be called")),
    )
    window._notify_available = False
    window.notify("hello", "world")
    event = read_events(state_dir)[-1]
    assert event["kind"] == "notification" and event["delivered"] is False


def test_bind_sets_names_and_returns_widget(app):
    window, _ = app
    assert window.send.accessibleName() == "send_button"
    assert window.send.objectName() == "send_button"
    assert window.interface == ("message_input", "send_button")
