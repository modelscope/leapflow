from __future__ import annotations

import argparse
import logging
from dataclasses import replace
from pathlib import Path

import pytest

from conftest import make_settings


def test_context_constructs_approval_gate_before_initialize(tmp_path) -> None:
    from leapflow.cli.context import Context

    ctx = Context(make_settings(str(tmp_path)), mock_host=True)

    assert hasattr(ctx, "_approval_gate")
    assert hasattr(ctx, "_tui_approval")
    assert not hasattr(ctx, "shortcuts")


def test_shortcut_commands_are_not_registered() -> None:
    from leapflow.cli.commands.registry import commands_by_category, resolve_command

    assert resolve_command("shortcut") is None
    assert resolve_command("shortcut add hello = hi") is None
    assert "Shortcuts" not in commands_by_category()


@pytest.mark.asyncio
async def test_context_initialize_wires_gateway_approval_gate(tmp_path) -> None:
    from leapflow.cli.context import Context
    import leapflow.tools.gateway_tool as gateway_tool

    ctx = Context(make_settings(str(tmp_path)), mock_host=True)
    await ctx.initialize()
    try:
        assert gateway_tool._approval_gate is ctx._approval_orchestrator
        assert gateway_tool._approval_gate.grants is ctx._approval_orchestrator.grants
    finally:
        await ctx.cleanup()


@pytest.mark.asyncio
async def test_context_initialize_degrades_when_primary_db_is_locked(
    tmp_path,
    monkeypatch,
) -> None:
    from pathlib import Path

    from leapflow.cli.context import Context
    from leapflow.storage.duckdb_connect import DatabaseLockedError
    import leapflow.storage.connection as connection_module

    settings = make_settings(str(tmp_path))
    original_connect = connection_module._lock_aware_connect

    def locked_primary_connect(db_path: Path):
        if Path(db_path) == settings.duckdb_path:
            raise DatabaseLockedError(settings.duckdb_path, RuntimeError("locked"))
        return original_connect(db_path)

    monkeypatch.setattr(
        connection_module,
        "_lock_aware_connect",
        locked_primary_connect,
    )

    ctx = Context(settings, mock_host=True)
    await ctx.initialize()
    try:
        assert ctx.storage_volatile is True
        assert ctx.engine is not None
        assert ctx._db_holder.db_path != settings.duckdb_path
    finally:
        await ctx.cleanup()


def test_visual_track_defaults_off_without_env(monkeypatch, tmp_path) -> None:
    from leapflow.config import DEFAULT_LLM_CONTEXT_LENGTH, _build_settings_from_env

    monkeypatch.delenv("LEAPFLOW_VISUAL_TRACK_ENABLED", raising=False)
    monkeypatch.delenv("LEAPFLOW_LLM_API_KEY", raising=False)
    monkeypatch.delenv("LEAPFLOW_VLM_API_KEY", raising=False)
    monkeypatch.delenv("LEAPFLOW_LLM_CONTEXT_LENGTH", raising=False)
    monkeypatch.setenv("LEAPFLOW_DATA_DIR", str(tmp_path))

    settings = _build_settings_from_env()

    assert settings.visual_track_enabled is False
    assert settings.has_vlm_credentials is False
    assert settings.llm_context_length == DEFAULT_LLM_CONTEXT_LENGTH


def test_profile_name_rejects_path_traversal(monkeypatch, tmp_path) -> None:
    from leapflow.config import _build_settings_from_env

    monkeypatch.setenv("LEAPFLOW_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("LEAPFLOW_PROFILE", "../../escape")

    with pytest.raises(ValueError, match="Invalid LEAPFLOW_PROFILE"):
        _build_settings_from_env()

    assert not (tmp_path.parent / "escape").exists()


def test_context_length_is_exposed_in_env_templates() -> None:
    from leapflow.config import DEFAULT_LLM_CONTEXT_LENGTH
    from leapflow._env_template import ENV_TEMPLATE

    expected = f"LEAPFLOW_LLM_CONTEXT_LENGTH={DEFAULT_LLM_CONTEXT_LENGTH}"
    example = (Path(__file__).parents[1] / ".env.example").read_text(encoding="utf-8")

    assert expected in ENV_TEMPLATE
    assert expected in example
    assert "Runtime context budget" in ENV_TEMPLATE


def test_build_visual_components_degrades_without_credentials(
    caplog,
    tmp_path,
) -> None:
    from leapflow.cli.context import _build_visual_components

    settings = replace(
        make_settings(str(tmp_path)),
        llm_api_key="",
        vlm_api_key="",
        visual_track_enabled=True,
    )

    with caplog.at_level(logging.WARNING, logger="leapflow.cli.context"):
        perception_session = _build_visual_components(settings, rpc=object())

    assert perception_session is None
    assert any("Visual perception disabled" in record.message for record in caplog.records)


def test_build_visual_components_accepts_vlm_only_credentials(
    monkeypatch,
    tmp_path,
) -> None:
    import leapflow.cli.context as context_module

    captured = {}

    class FakeOpenAIChat:
        def __init__(self, *, api_key: str, base_url: str, model: str) -> None:
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            captured["model"] = model

    monkeypatch.setattr(context_module, "OpenAIChat", FakeOpenAIChat)
    settings = replace(
        make_settings(str(tmp_path)),
        llm_api_key="",
        vlm_api_key="vlm-test-key",
        vlm_base_url="https://vlm.example.invalid/v1",
        vlm_model="vlm-test-model",
        visual_track_enabled=True,
    )

    perception_session = context_module._build_visual_components(settings, rpc=object())

    assert perception_session is not None
    assert captured == {
        "api_key": "vlm-test-key",
        "base_url": "https://vlm.example.invalid/v1",
        "model": "vlm-test-model",
    }


@pytest.mark.asyncio
async def test_context_initialize_degrades_visual_track_without_credentials(
    caplog,
    tmp_path,
) -> None:
    from leapflow.cli.context import Context

    settings = replace(
        make_settings(str(tmp_path)),
        llm_api_key="",
        vlm_api_key="",
        visual_track_enabled=True,
    )

    ctx = Context(settings, mock_host=True)
    with caplog.at_level(logging.WARNING, logger="leapflow.cli.context"):
        await ctx.initialize()
    try:
        assert ctx.perception_session is None
        assert ctx.engine is not None
        assert any("Visual perception disabled" in record.message for record in caplog.records)
    finally:
        await ctx.cleanup()


@pytest.mark.asyncio
async def test_context_hot_reloads_llm_credentials_from_env_file(
    monkeypatch,
    tmp_path,
) -> None:
    from leapflow.cli.context import Context

    data_dir = tmp_path / "leap-home"
    data_dir.mkdir()
    env_path = data_dir / ".env"
    env_path.write_text(
        "LEAPFLOW_LLM_API_KEY=\n"
        "LEAPFLOW_LLM_BASE_URL=https://old.example.invalid/v1\n"
        "LEAPFLOW_LLM_MODEL=old-model\n"
        "LEAPFLOW_LLM_CONTEXT_LENGTH=128000\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LEAPFLOW_LLM_API_KEY", raising=False)
    monkeypatch.delenv("LEAPFLOW_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LEAPFLOW_LLM_MODEL", raising=False)
    monkeypatch.delenv("LEAPFLOW_LLM_CONTEXT_LENGTH", raising=False)

    settings = replace(
        make_settings(str(tmp_path)),
        data_dir=data_dir,
        llm_api_key="",
        llm_base_url="https://old.example.invalid/v1",
        llm_model="old-model",
        llm_context_length=128_000,
        vlm_api_key="",
        visual_track_enabled=False,
    )

    ctx = Context(settings, mock_host=True)
    await ctx.initialize()
    try:
        assert ctx.settings.has_llm_credentials is False
        assert ctx.engine is not None
        assert ctx.engine._settings.has_llm_credentials is False

        env_path.write_text(
            "LEAPFLOW_LLM_API_KEY=sk-hot-reload\n"
            "LEAPFLOW_LLM_BASE_URL=https://new.example.invalid/v1\n"
            "LEAPFLOW_LLM_MODEL=new-model\n"
            "LEAPFLOW_LLM_CONTEXT_LENGTH=512000\n",
            encoding="utf-8",
        )

        assert ctx.reload_runtime_config_if_changed() is True
        assert ctx.settings.llm_api_key == "sk-hot-reload"
        assert ctx.settings.llm_base_url == "https://new.example.invalid/v1"
        assert ctx.settings.llm_model == "new-model"
        assert ctx.settings.llm_context_length == 512_000
        assert ctx.engine._settings.llm_api_key == "sk-hot-reload"
        assert ctx.engine._settings.llm_context_length == 512_000
        assert ctx.engine.model_capabilities.resolve("new-model").context_length == 512_000
        assert ctx.engine._settings.has_llm_credentials is True
        assert ctx.intent_classifier.__class__.__name__ == "LLMIntentClassifier"
    finally:
        await ctx.cleanup()


@pytest.mark.asyncio
async def test_context_hot_reloads_llm_credentials_from_config_yaml(
    monkeypatch,
    tmp_path,
) -> None:
    from leapflow.cli.context import Context

    data_dir = tmp_path / "leap-home"
    data_dir.mkdir()
    (data_dir / ".env").write_text("LEAPFLOW_LLM_API_KEY=\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LEAPFLOW_LLM_API_KEY", raising=False)
    monkeypatch.delenv("LEAPFLOW_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("LEAPFLOW_LLM_MODEL", raising=False)
    monkeypatch.delenv("LEAPFLOW_LLM_CONTEXT_LENGTH", raising=False)

    settings = replace(
        make_settings(str(tmp_path)),
        data_dir=data_dir,
        llm_api_key="",
        llm_base_url="https://old.example.invalid/v1",
        llm_model="old-model",
        llm_context_length=128_000,
        vlm_api_key="",
        visual_track_enabled=False,
    )

    ctx = Context(settings, mock_host=True)
    await ctx.initialize()
    try:
        assert ctx.settings.has_llm_credentials is False
        (data_dir / "config.yaml").write_text(
            "llm:\n"
            "  api_key: sk-yaml-hot-reload\n"
            "  base_url: https://yaml.example.invalid/v1\n"
            "  model: yaml-model\n"
            "  context_length: 640000\n",
            encoding="utf-8",
        )

        assert ctx.reload_runtime_config_if_changed() is True
        assert ctx.settings.llm_api_key == "sk-yaml-hot-reload"
        assert ctx.settings.llm_base_url == "https://yaml.example.invalid/v1"
        assert ctx.settings.llm_model == "yaml-model"
        assert ctx.settings.llm_context_length == 640_000
        assert ctx.engine._settings.llm_api_key == "sk-yaml-hot-reload"
        assert ctx.engine.model_capabilities.resolve("yaml-model").context_length == 640_000
    finally:
        await ctx.cleanup()


@pytest.mark.asyncio
async def test_known_model_table_does_not_override_explicit_context_length(
    tmp_path,
) -> None:
    from leapflow.cli.context import Context

    settings = replace(
        make_settings(str(tmp_path)),
        llm_model="qwen3.7-plus",
        llm_context_length=300_000,
        visual_track_enabled=False,
    )

    ctx = Context(settings, mock_host=True)
    await ctx.initialize()
    try:
        assert ctx.engine is not None
        caps = ctx.engine.model_capabilities.resolve("qwen3.7-plus")
        assert caps.context_length == 300_000
        assert caps.supports_thinking is True
    finally:
        await ctx.cleanup()


@pytest.mark.asyncio
async def test_daemon_fallback_initializes_local_interactive_with_real_config(
    monkeypatch,
    tmp_path,
) -> None:
    from leapflow.cli import cli
    from leapflow.cli.helpers import require_initialized
    import leapflow.cli.commands.interactive as interactive_module
    import leapflow.daemon.client as daemon_client

    data_dir = tmp_path / "leap-home"
    events: list[str] = []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LEAPFLOW_DATA_DIR", str(data_dir))
    monkeypatch.delenv("LEAPFLOW_LLM_API_KEY", raising=False)
    monkeypatch.delenv("LEAPFLOW_VLM_API_KEY", raising=False)
    monkeypatch.delenv("LEAPFLOW_VISUAL_TRACK_ENABLED", raising=False)
    monkeypatch.delenv("LEAPFLOW_LLM_CONTEXT_LENGTH", raising=False)

    async def fake_ensure_daemon_client(*args, **kwargs):
        raise daemon_client.DaemonUnavailableError("daemon unavailable")

    async def fake_interactive(ctx, *, resume_id=None) -> int:
        require_initialized(ctx)
        assert resume_id is None
        assert ctx.settings.llm_api_key == ""
        assert ctx.settings.visual_track_enabled is False
        assert ctx.perception_session is None
        events.append("interactive")
        return 0

    monkeypatch.setattr(daemon_client, "ensure_daemon_client", fake_ensure_daemon_client)
    monkeypatch.setattr(interactive_module, "cmd_interactive", fake_interactive)

    result = await cli._async_daemon_main(
        argparse.Namespace(command="interactive", mock_host=True, resume=None)
    )

    assert result == 0
    assert events == ["interactive"]
    assert (data_dir / ".env").exists()


@pytest.mark.asyncio
async def test_daemon_runtime_bridge_recovers_and_resumes_session() -> None:
    from leapflow.cli.commands.interactive import _DaemonRuntimeBridge
    from leapflow.daemon.client import DaemonUnavailableError

    class Console:
        def __init__(self) -> None:
            self.warnings: list[str] = []
            self.systems: list[str] = []
            self.successes: list[str] = []

        def warning(self, message: str) -> None:
            self.warnings.append(message)

        def system(self, message: str) -> None:
            self.systems.append(message)

        def success(self, message: str) -> None:
            self.successes.append(message)

    class BrokenClient:
        async def status(self):
            raise DaemonUnavailableError("socket disappeared")

    class RecoveredClient:
        def __init__(self) -> None:
            self.resumed: list[str] = []

        async def status(self):
            return {"pid": 99, "session_id": "sess-1"}

        async def session_resume(self, session_id: str):
            self.resumed.append(session_id)
            return {"found": True, "session_id": session_id}

    class Settings:
        pass

    active_session_id = "sess-1"
    metadata: list[dict] = []
    recovered = RecoveredClient()

    async def factory(settings, *, mock_host: bool = False, status_callback=None):
        if status_callback is not None:
            status_callback("Connected to recovered leapd.")
        return recovered

    bridge = _DaemonRuntimeBridge(
        BrokenClient(),
        Settings(),
        Console(),
        session_id_getter=lambda: active_session_id,
        session_id_setter=lambda value: None,
        metadata_applier=metadata.append,
        client_factory=factory,
    )

    result = await bridge.call(lambda current_client: current_client.status(), description="status")

    assert result == {"pid": 99, "session_id": "sess-1"}
    assert recovered.resumed == ["sess-1"]
    assert metadata == [{"pid": 99, "session_id": "sess-1"}]


def test_leap_default_command_uses_daemon_client(monkeypatch) -> None:
    from leapflow.cli import cli

    captured = {}

    async def fake_daemon_main(args):
        captured["command"] = args.command
        captured["no_daemon"] = args.no_daemon
        return 0

    monkeypatch.setattr(cli, "_async_daemon_main", fake_daemon_main)

    assert cli.main([]) == 0
    assert captured == {"command": "interactive", "no_daemon": False}


def test_leap_no_daemon_initializes_and_runs_interactive(monkeypatch, tmp_path) -> None:
    from leapflow.cli import cli
    from leapflow.cli.helpers import require_initialized
    import leapflow.cli.commands.interactive as interactive_module

    data_dir = tmp_path / "leap-home"
    events: list[str] = []

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LEAPFLOW_DATA_DIR", str(data_dir))
    monkeypatch.delenv("LEAPFLOW_LLM_API_KEY", raising=False)
    monkeypatch.delenv("LEAPFLOW_VLM_API_KEY", raising=False)
    monkeypatch.delenv("LEAPFLOW_VISUAL_TRACK_ENABLED", raising=False)
    monkeypatch.delenv("LEAPFLOW_LLM_CONTEXT_LENGTH", raising=False)

    async def fake_interactive(ctx, *, resume_id=None) -> int:
        require_initialized(ctx)
        assert resume_id is None
        assert ctx.settings.llm_api_key == ""
        assert ctx.settings.visual_track_enabled is False
        assert ctx.settings.has_vlm_credentials is False
        assert ctx.perception_session is None
        assert ctx.engine is not None
        events.append("interactive")
        return 0

    monkeypatch.setattr(interactive_module, "cmd_interactive", fake_interactive)

    assert cli.main(["--no-daemon", "--mock-host"]) == 0
    assert events == ["interactive"]
    assert (data_dir / ".env").exists()


def test_leap_daemon_restart_routes_to_daemon_command(monkeypatch) -> None:
    from leapflow.cli import cli
    import leapflow.cli.commands.daemon as daemon_module

    captured = {}

    def fake_cmd_daemon(args):
        captured["action"] = args.daemon_action
        return 0

    monkeypatch.setattr(daemon_module, "cmd_daemon", fake_cmd_daemon)

    assert cli.main(["daemon", "restart"]) == 0
    assert captured == {"action": "restart"}


def test_stdin_echo_guard_restores_and_flushes_tty(monkeypatch) -> None:
    from leapflow.cli import cli

    calls = []

    class FakeStdin:
        def isatty(self) -> bool:
            return True

        def fileno(self) -> int:
            return 7

    class FakeTermios:
        ECHO = 8
        TCSADRAIN = 1
        TCIFLUSH = 2
        error = OSError

        @staticmethod
        def tcgetattr(fd):
            calls.append(("get", fd))
            return [0, 0, 0, 15]

        @staticmethod
        def tcsetattr(fd, when, attrs):
            calls.append(("set", fd, when, attrs[3]))

        @staticmethod
        def tcflush(fd, queue):
            calls.append(("flush", fd, queue))

    monkeypatch.setattr(cli.sys, "stdin", FakeStdin())
    monkeypatch.setattr(cli, "termios", FakeTermios)

    with cli._StdinEchoGuard():
        pass

    assert calls == [
        ("get", 7),
        ("set", 7, FakeTermios.TCSADRAIN, 7),
        ("set", 7, FakeTermios.TCSADRAIN, 15),
        ("flush", 7, FakeTermios.TCIFLUSH),
    ]


def test_context_uses_mock_bridge_when_cua_driver_disabled(tmp_path) -> None:
    from leapflow.cli.context import Context
    from leapflow.platform.mock import MockBridge

    settings = replace(make_settings(str(tmp_path)), mock_host=False, use_cua_driver=False)
    ctx = Context(settings, mock_host=False)

    assert isinstance(ctx.rpc, MockBridge)


@pytest.mark.asyncio
async def test_context_initialize_replaces_failed_cua_driver_with_mock(monkeypatch, tmp_path) -> None:
    import leapflow.cli.context as context_module
    from leapflow.platform.mock import MockBridge

    class FailingCuaDriverClient:
        def start(self) -> None:
            raise RuntimeError("cua-driver unavailable")

        def stop(self) -> None:
            raise AssertionError("failed driver should be replaced")

    settings = replace(make_settings(str(tmp_path)), mock_host=False, use_cua_driver=True)
    monkeypatch.setattr(context_module, "CuaDriverClient", FailingCuaDriverClient)

    ctx = context_module.Context(settings, mock_host=False)
    await ctx.initialize()
    try:
        assert isinstance(ctx.rpc, MockBridge)
        assert ctx.engine is not None
    finally:
        await ctx.cleanup()


@pytest.mark.asyncio
async def test_context_cleanup_continues_when_cua_driver_stop_fails(tmp_path) -> None:
    from leapflow.cli.context import Context
    from leapflow.platform.cua_client import CuaDriverClient

    class FailingCuaDriverClient(CuaDriverClient):
        def __init__(self) -> None:
            pass

        def stop(self) -> None:
            raise RuntimeError("stop failed")

    class CloseTracker:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    ctx = Context(make_settings(str(tmp_path)), mock_host=True)
    await ctx.initialize()
    tracker = CloseTracker()
    ctx.rpc = FailingCuaDriverClient()
    ctx.skill_lib = tracker

    await ctx.cleanup()

    assert tracker.closed is True


@pytest.mark.asyncio
async def test_host_doctor_stops_client_when_probe_fails(monkeypatch) -> None:
    from leapflow.cli.commands import host as host_module
    import leapflow.platform.cua_client as cua_module

    calls: list[str] = []

    class FakeSession:
        available_tools = {"list_apps": set()}
        capability_version = "test-cap"

        def call_tool_sync(self, name, args, timeout=5.0):
            calls.append(f"probe:{name}")
            raise RuntimeError("probe failed")

    class FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            self._session = FakeSession()

        def start(self) -> None:
            calls.append("start")

        def stop(self) -> None:
            calls.append("stop")

    monkeypatch.setattr(host_module, "_cua_driver_installed", lambda: True)
    monkeypatch.setattr(host_module, "_cua_driver_version", lambda: "test-version")
    monkeypatch.setattr(host_module.shutil, "which", lambda command: "/tmp/cua-driver")
    monkeypatch.setattr(cua_module, "CuaDriverClient", FakeClient)

    result = await host_module._cmd_doctor()

    assert result == 1
    assert calls == ["start", "probe:list_apps", "stop"]


@pytest.mark.asyncio
async def test_host_status_reports_daemon_host_backend(monkeypatch, tmp_path, capsys) -> None:
    from conftest import make_settings
    from leapflow.cli.commands import host as host_module

    class Info:
        pid = 123
        is_healthy = True
        is_running = True
        sock_path = tmp_path / "leapd.sock"

    settings = replace(make_settings(str(tmp_path)), use_cua_driver=True)

    async def fake_fetch(settings_obj):
        return Info(), {
            "host_backend": {
                "backend": "cua-driver",
                "started": True,
                "pid": None,
                "pid_source": "unavailable",
                "command": "/tmp/cua-driver",
                "args": ["mcp"],
                "tools_count": 3,
                "restart_count": 1,
            }
        }, ""

    monkeypatch.setattr(host_module, "load_config", lambda: settings)
    monkeypatch.setattr(host_module, "_fetch_leapd_status", fake_fetch)
    monkeypatch.setattr(host_module, "_read_pid_file", lambda: None)
    monkeypatch.setattr(host_module, "_cua_driver_installed", lambda: True)
    monkeypatch.setattr(host_module, "_cua_driver_version", lambda: "test-version")
    monkeypatch.setattr(host_module.shutil, "which", lambda command: "/tmp/cua-driver")

    assert await host_module._cmd_status() == 0

    output = capsys.readouterr().out
    assert "leapd healthy" in output
    assert "Backend: cua-driver started=True" in output
    assert "Tools: 3 restarts=1" in output


@pytest.mark.asyncio
async def test_daemon_tui_exit_prompt_stops_by_default(monkeypatch, tmp_path) -> None:
    from leapflow.cli.commands import interactive as interactive_module
    import leapflow.daemon.lifecycle as lifecycle_module

    class Client:
        async def status(self):
            return {"pid": 1234}

        async def shutdown(self):
            calls.append("shutdown")

    class Console:
        def __init__(self) -> None:
            self.systems: list[str] = []
            self.warnings: list[str] = []

        def system(self, message: str) -> None:
            self.systems.append(message)

        def warning(self, message: str) -> None:
            self.warnings.append(message)

    class Settings:
        profile_dir = tmp_path

    async def yes(prompt: str) -> bool:
        return True

    def record_stop(run_dir, **kwargs):
        calls.append((run_dir, kwargs))
        return lifecycle_module.StopDaemonResult(pid=1234, stopped=True)

    calls = []
    monkeypatch.setattr(interactive_module, "_ask_yes_no_default_yes", yes)
    monkeypatch.setattr(lifecycle_module, "stop_daemon", record_stop)
    console = Console()

    await interactive_module._prompt_stop_daemon_on_exit(Client(), Settings(), console)

    assert "shutdown" in calls
    assert any(isinstance(call, tuple) and call[0] == tmp_path / "run" for call in calls)
    assert "leapd stopped" in console.systems[-1]


@pytest.mark.asyncio
async def test_daemon_tui_exit_prompt_can_keep_daemon(monkeypatch, tmp_path) -> None:
    from leapflow.cli.commands import interactive as interactive_module
    import leapflow.daemon.lifecycle as lifecycle_module

    class Client:
        async def status(self):
            return {"pid": 1234}

    class Console:
        def __init__(self) -> None:
            self.systems: list[str] = []
            self.warnings: list[str] = []

        def system(self, message: str) -> None:
            self.systems.append(message)

        def warning(self, message: str) -> None:
            self.warnings.append(message)

    class Settings:
        profile_dir = tmp_path

    def fail_stop(*args, **kwargs):
        raise AssertionError("daemon should be kept running")

    async def no(prompt: str) -> bool:
        return False

    monkeypatch.setattr(interactive_module, "_ask_yes_no_default_yes", no)
    monkeypatch.setattr(lifecycle_module, "stop_daemon", fail_stop)
    console = Console()

    await interactive_module._prompt_stop_daemon_on_exit(Client(), Settings(), console)

    assert any("kept running" in message for message in console.systems)


@pytest.mark.asyncio
async def test_daemon_tui_exit_prompt_keeps_daemon_by_default_for_other_clients(
    monkeypatch,
    tmp_path,
) -> None:
    from leapflow.cli.commands import interactive as interactive_module
    import leapflow.daemon.lifecycle as lifecycle_module

    class Client:
        async def status(self):
            return {"pid": 1234, "connected_clients": 2}

    class Console:
        def __init__(self) -> None:
            self.systems: list[str] = []
            self.warnings: list[str] = []

        def system(self, message: str) -> None:
            self.systems.append(message)

        def warning(self, message: str) -> None:
            self.warnings.append(message)

    class Settings:
        profile_dir = tmp_path

    prompts: list[str] = []

    async def default_no(prompt: str) -> bool:
        prompts.append(prompt)
        return False

    def fail_stop(*args, **kwargs):
        raise AssertionError("daemon should be kept running while other clients exist")

    monkeypatch.setattr(interactive_module, "_ask_yes_no_default_no", default_no)
    monkeypatch.setattr(lifecycle_module, "stop_daemon", fail_stop)
    console = Console()

    await interactive_module._prompt_stop_daemon_on_exit(Client(), Settings(), console)

    assert prompts == ["Stop leapd anyway (pid=1234)? [y/N]: "]
    assert any("other Leap client" in message for message in console.systems)
    assert any("kept running" in message for message in console.systems)


def test_leap_prompt_uses_daemon_chat_route(monkeypatch) -> None:
    from leapflow.cli import cli

    captured = {}

    async def fake_daemon_main(args):
        captured["command"] = args.command
        captured["prompt"] = args.prompt
        return 0

    monkeypatch.setattr(cli, "_async_daemon_main", fake_daemon_main)

    assert cli.main(["hello", "world"]) == 0
    assert captured == {"command": "chat", "prompt": "hello world"}


@pytest.mark.asyncio
async def test_teach_start_without_session_returns_structured_error() -> None:
    from types import SimpleNamespace

    from leapflow.cli.commands.slash_handlers import command_execute

    result = await command_execute(SimpleNamespace(session=None), "teach start", "")

    assert result == {"ok": False, "message": "No active session.", "session_mode": "idle"}


@pytest.mark.asyncio
async def test_hub_fallback_message_formats_command_without_literal_strip(monkeypatch) -> None:
    from types import SimpleNamespace

    import leapflow.cli.commands.slash_handlers as slash_handlers

    monkeypatch.setitem(__import__("sys").modules, "leapflow.cli.commands.hub", None)
    result = await slash_handlers._execute_hub(SimpleNamespace(), "hub", "search demo")

    assert result["ok"] is False
    assert "'.strip()'" not in result["message"]
    assert "Hub command '/hub search demo'" in result["message"]
