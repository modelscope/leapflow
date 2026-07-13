from __future__ import annotations

from leapflow.cli.commands.router import CommandRouter


def test_command_router_parses_command_args_and_runtime_support() -> None:
    router = CommandRouter("daemon")

    invocation = router.parse("/host restart")

    assert invocation is not None
    assert invocation.command.name == "host"
    assert invocation.args == "restart"
    assert invocation.command.supports_runtime("daemon") is True
    assert router.unsupported_result(invocation) is None


def test_command_router_parses_app_subcommands_without_daemon_parity() -> None:
    in_process_router = CommandRouter("in_process")
    daemon_router = CommandRouter("daemon")

    invocation = in_process_router.parse("/app status feishu")
    daemon_invocation = daemon_router.parse("/app connect feishu")

    assert invocation is not None
    assert invocation.command.name == "app status"
    assert invocation.args == "feishu"
    assert in_process_router.unsupported_result(invocation) is None
    assert daemon_invocation is not None
    assert daemon_invocation.command.name == "app connect"
    unsupported = daemon_router.unsupported_result(daemon_invocation)
    assert unsupported is not None
    assert "/app connect" in unsupported.title


def test_app_commands_daemon_parity_boundary() -> None:
    """Read-only /app commands are available in daemon; write commands are not."""
    daemon_router = CommandRouter("daemon")

    # Commands that MUST be supported in daemon mode
    for cmd_text in ("/app", "/app list", "/app status feishu", "/app actions feishu"):
        inv = daemon_router.parse(cmd_text)
        assert inv is not None, f"parse failed for {cmd_text}"
        assert daemon_router.unsupported_result(inv) is None, (
            f"{cmd_text} should be supported in daemon mode"
        )

    # Commands that must NOT be supported in daemon mode
    for cmd_text in ("/app connect feishu", "/app disconnect feishu", "/app remove feishu"):
        inv = daemon_router.parse(cmd_text)
        assert inv is not None, f"parse failed for {cmd_text}"
        unsupported = daemon_router.unsupported_result(inv)
        assert unsupported is not None, f"{cmd_text} should be blocked in daemon mode"
        assert unsupported.ok is False


def test_app_commands_are_registered_for_completion() -> None:
    from leapflow.cli.commands.registry import completion_entries

    entries = dict(completion_entries())

    assert entries["app"] == "List supported external apps or open an app setup guide"
    assert entries["app list"] == "List supported external apps"
    assert entries["app status"] == "Show App Connector status"
    assert entries["app connect"] == "Connect a supported external app"
    assert entries["app actions"] == "List App Connector action domains"


def test_command_router_returns_standard_unsupported_result() -> None:
    router = CommandRouter("daemon")

    invocation = router.parse("/skills show demo")

    assert invocation is not None
    assert invocation.command.name == "skills show"
    result = router.unsupported_result(invocation)
    assert result is not None
    assert result.ok is False
    assert "daemon" in result.title
