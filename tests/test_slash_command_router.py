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

def test_board_commands_resolve_as_engine_routed() -> None:
    """Board commands must resolve as non-client-local so the in-process REPL
    routes them through command_execute instead of leaking to the LLM chat."""
    router = CommandRouter("daemon")

    for cmd_text in ("/board", "/board finance", "/board templates", "/board refresh", "/board status"):
        inv = router.parse(cmd_text)
        assert inv is not None, f"parse failed for {cmd_text}"
        assert inv.command.name.startswith("board"), cmd_text
        assert inv.command.client_local is False, f"{cmd_text} must be engine-routed"

    # A bare template name resolves to the base `board` command (template = arg).
    finance = router.parse("/board finance")
    assert finance is not None and finance.command.name == "board" and finance.args == "finance"
    # Reserved verbs resolve to their dedicated command.
    assert router.parse("/board templates").command.name == "board templates"


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


def test_orient_command_is_registered_and_read_only() -> None:
    """E-1: /orient is a registered, read-only, engine-routed observability command."""
    from leapflow.cli.commands.registry import CommandEffect

    router = CommandRouter("daemon")
    invocation = router.parse("/orient")
    assert invocation is not None
    assert invocation.command.name == "orient"
    assert invocation.command.effect == CommandEffect.READ_ONLY
    assert invocation.command.client_local is False   # routed through command_execute in both modes


def test_build_orient_payload_renders_layers_and_guards_missing_engine() -> None:
    from types import SimpleNamespace

    from leapflow.cli.commands.slash_handlers import build_orient_payload
    from leapflow.world_model.orientation import aggregate_orientation

    # Graceful when no engine yet.
    assert build_orient_payload(SimpleNamespace(engine=None))["ok"] is False

    fake_engine = SimpleNamespace(
        orientation_view=lambda: aggregate_orientation(
            working=["finding A", "[open] does B cache?"], now=0.0,
        ),
        focus_view=lambda: {
            "active_focus": {"canonical_name": "MiniCPM-O 4.5 Technical Report", "kind": "paper"},
            "recent_control_events": [
                {"user_visible_summary": "llm.model -> qwen3.8-max"},
            ],
        },
    )
    payload = build_orient_payload(SimpleNamespace(engine=fake_engine, _reentry_store=None))
    assert payload["ok"] is True
    assert "finding A" in payload["message"]
    assert "Active focus: MiniCPM-O 4.5 Technical Report (paper)" in payload["message"]
    assert "llm.model -> qwen3.8-max" in payload["message"]
    assert payload["orientation"]["total"] == 2
    assert payload["focus"]["active_focus"]["canonical_name"] == "MiniCPM-O 4.5 Technical Report"


def test_tools_payload_groups_desktop_tools_when_perception_online() -> None:
    """/tools must reflect the live catalog, not just the static registry."""
    from types import SimpleNamespace

    from leapflow.cli.commands.slash_handlers import build_tool_payload
    from leapflow.skills.semantic_schema import semantic_tool_to_openai
    from leapflow.skills.tool_executor import ToolDefinition
    from leapflow.plugins import get_registry
    _tool_reg = get_registry()

    ctx = SimpleNamespace(rpc=SimpleNamespace(connected=False), platform_tools=[])

    # An earlier test in this worker may have constructed an AgentEngine, which
    # installs a catalog provider globally; the offline baseline needs a clean
    # slate.
    _tool_reg.set_capability_catalog_provider(None)
    offline = build_tool_payload(ctx)
    assert "desktop" not in offline["groups"]
    offline_total = offline["total"]

    desktop_defs = [
        semantic_tool_to_openai(ToolDefinition(name=name, description="d", parameters={}))
        for name in ("click", "observe_ui", "list_apps")
    ]
    _tool_reg.set_capability_catalog_provider(lambda: list(_tool_reg.tool_definitions) + desktop_defs)
    try:
        online = build_tool_payload(ctx)
        assert set(online["groups"]["desktop"]) == {"click", "list_apps", "observe_ui"}
        assert online["total"] == offline_total + 3
        # Existing display categories stay untouched.
        assert "shell_run" in online["groups"]["shell"]
        assert "file_read" in online["groups"]["file"]
    finally:
        _tool_reg.set_capability_catalog_provider(None)


# ════════════════════════════════════════════════════════════════
# /board completes its own second token
# ════════════════════════════════════════════════════════════════


def _board_completer(templates: tuple[str, ...] = ("capability", "hardware", "signals")):
    from leapflow.cli.tui_app.input import SlashCommandCompleter

    return SlashCommandCompleter(
        [("board", "Board"), ("config", "Config")], board_templates=templates
    )


def _offered(completer, text: str) -> list[str]:
    from prompt_toolkit.document import Document

    return [item.text for item in completer.get_completions(Document(text, len(text)), None)]


def test_board_offers_both_verbs_and_lenses() -> None:
    """The dispatcher accepts either in the same position, so both must be offered.

    ``/board`` took a reserved verb *or* a template name as its second token and the
    completer offered neither: every lens and every control verb was undiscoverable,
    reachable only by reading the source or the args hint.
    """
    offered = _offered(_board_completer(), "/board ")
    assert {"templates", "status", "refresh", "pause", "resume", "stop"} <= set(offered)
    assert {"capability", "hardware", "signals"} <= set(offered)


def test_board_narrows_on_the_typed_prefix() -> None:
    """One prefix can match a verb and a lens at once; both must survive."""
    completer = _board_completer(("sentiment", "signals"))
    assert _offered(completer, "/board h") == []
    assert set(_offered(completer, "/board s")) == {
        "status", "stop", "sentiment", "signals",
    }, "a prefix shared by verbs and lenses must not drop either kind"
    assert _offered(completer, "/board cap") == []


def test_board_offers_nothing_for_a_watch_id() -> None:
    """Only the running daemon knows watch ids, so inventing one would mislead."""
    completer = _board_completer()
    assert _offered(completer, "/board stop ") == []
    assert _offered(completer, "/board stop abc") == []


def test_board_lenses_come_from_the_installed_templates() -> None:
    """A lens is a YAML file an operator can add; a hardcoded list would omit theirs."""
    completer = _board_completer(("my_bench",))
    assert "my_bench" in _offered(completer, "/board ")
    assert "capability" not in _offered(completer, "/board "), (
        "the lens list must be the one supplied, not a built-in default"
    )


def test_the_completer_and_the_dispatcher_agree_on_the_reserved_verbs() -> None:
    """Two literal lists drift, so the agreement is asserted rather than assumed.

    A verb the dispatcher accepts but never offers is undiscoverable; one offered but
    rejected is worse than no completion at all, because it teaches the user a command
    that does not exist.
    """
    from leapflow.cli.commands.slash_handlers import _BOARD_VERBS as dispatcher_verbs
    from leapflow.cli.tui_app.input import _BOARD_VERBS as completer_verbs

    assert {verb for verb, _ in completer_verbs} == set(dispatcher_verbs)
