from __future__ import annotations

from leapflow.cli.commands.router import CommandRouter
from leapflow.cli.commands.interactive import _is_app_command


def test_command_router_parses_command_args_and_runtime_support() -> None:
    router = CommandRouter("daemon")

    invocation = router.parse("/host restart")

    assert invocation is not None
    assert invocation.command.name == "host"
    assert invocation.args == "restart"
    assert invocation.command.supports_runtime("daemon") is True
    assert router.unsupported_result(invocation) is None


def test_command_router_all_commands_supported_in_daemon() -> None:
    """All commands are now supported in both runtimes."""
    daemon_router = CommandRouter("daemon")

    for cmd_text in (
        "/app status feishu",
        "/app connect feishu",
        "/teach start",
        "/skill show demo",
        "/hub search test",
        "/gateway",
        "/arm test_skill 0 * * * *",
        "/task",
    ):
        inv = daemon_router.parse(cmd_text)
        assert inv is not None, f"parse failed for {cmd_text}"
        assert daemon_router.unsupported_result(inv) is None, (
            f"{cmd_text} should be supported in daemon mode"
        )


def test_command_router_client_local_commands() -> None:
    """Client-local commands are marked for direct TUI handling."""
    router = CommandRouter("daemon")

    # Client-local commands
    for cmd_text in ("/exit", "/clear", "/help", "/cancel", "/pause", "/resume", "/queue"):
        inv = router.parse(cmd_text)
        assert inv is not None, f"parse failed for {cmd_text}"
        assert inv.command.client_local is True, f"{cmd_text} should be client_local"

    # Engine-routed commands
    for cmd_text in ("/teach start", "/skill", "/gateway", "/tool", "/model"):
        inv = router.parse(cmd_text)
        assert inv is not None, f"parse failed for {cmd_text}"
        assert inv.command.client_local is False, f"{cmd_text} should NOT be client_local"

def test_plural_skill_tool_task_commands_are_not_registered() -> None:
    router = CommandRouter("daemon")

    for cmd_text in ("/skills", "/skills show demo", "/tools", "/tasks"):
        assert router.parse(cmd_text) is None


def test_app_commands_are_registered_for_completion() -> None:
    from leapflow.cli.commands.registry import completion_entries

    entries = dict(completion_entries())

    assert entries["app"] == "List supported external apps or open an app setup guide"
    assert entries["app list"] == "List supported external apps"
    assert entries["app status"] == "Show App Connector status"
    assert entries["app connect"] == "Connect a supported external app"
    assert entries["app actions"] == "List App Connector action domains"


def test_interactive_app_command_boundary_rejects_prefix_collisions() -> None:
    assert _is_app_command("app") is True
    assert _is_app_command("app status feishu") is True
    assert _is_app_command("apple") is False
    assert _is_app_command("application status") is False


def test_command_router_unsupported_always_returns_none() -> None:
    """unsupported_result always returns None — all commands are supported."""
    router = CommandRouter("daemon")

    invocation = router.parse("/skill show demo")

    assert invocation is not None
    assert invocation.command.name == "skill show"
    assert router.unsupported_result(invocation) is None
