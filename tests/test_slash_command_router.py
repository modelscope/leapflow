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


def test_command_router_returns_standard_unsupported_result() -> None:
    router = CommandRouter("daemon")

    invocation = router.parse("/skills show demo")

    assert invocation is not None
    assert invocation.command.name == "skills show"
    result = router.unsupported_result(invocation)
    assert result is not None
    assert result.ok is False
    assert "daemon" in result.title
