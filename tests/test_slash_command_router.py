from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from leapflow.cli.commands.router import CommandRouter
from leapflow.cli.commands.interactive import _is_app_command
from leapflow.hardware.context import (
    Channel,
    ContextProvenance,
    ContextSource,
    Direction,
    Envelope,
    HardwareContext,
    HardwareEffect,
    PrivacyTier,
    Representation,
    TransportRef,
)
from leapflow.hardware.registry import HardwareRegistry, HardwareSettings
from leapflow.hardware.transport import (
    SIDE_EFFECT_NONE,
    FrameReading,
    Reading,
    TransportStatus,
    WriteOutcome,
)


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


# ── /board peripherals ──────────────────────────────────────────────────────
#
# The device verbs, and in particular why ``preview`` is not just a deep link. The
# browser cannot obtain consent for a camera: ``ApprovalCoordinator.request_approval``
# denies when no approval route is installed, and the daemon installs one for
# ``command.execute`` but not for an ordinary RPC. So the slash command is the surface
# where the prompt can appear, and the grant it mints is what the board's stream is
# covered by. These tests pin that, and pin that a refusal opens nothing.

_JPEG = (
    b"\xff\xd8"
    b"\xff\xc0\x00\x11\x08\x00\x02\x00\x04\x03\x01\x22\x00\x02\x11\x01\x03\x11\x01"
    b"\xff\xd9"
)


class _FrameSource:
    """A frame-capable transport that counts how often the device was actually read."""

    kind = "fake_media"

    def __init__(self) -> None:
        self.grabs = 0

    async def open(self, context: HardwareContext) -> TransportStatus:
        return TransportStatus(connected=True)

    async def close(self) -> TransportStatus:
        return TransportStatus(connected=False)

    async def probe(self) -> TransportStatus:
        return TransportStatus(connected=True)

    async def halt(self) -> TransportStatus:
        return TransportStatus(connected=True, halt_supported=False)

    async def read(self, channel_id: str) -> Reading:
        return Reading(device_id="cam", channel_id=channel_id, value="frame")

    async def write(self, channel_id: str, value: object) -> WriteOutcome:
        return WriteOutcome(ok=False, side_effect_state=SIDE_EFFECT_NONE, error="read-only")

    async def read_frame(
        self, channel_id: str, *, max_width: int = 0, quality: int = 0
    ) -> FrameReading:
        self.grabs += 1
        return FrameReading(
            device_id="cam", channel_id=channel_id, data=_JPEG,
            width=320, height=180, sequence=self.grabs,
        )


def _camera(device_id: str = "cam") -> HardwareContext:
    return HardwareContext(
        device_id=device_id,
        display_name="Fake camera",
        device_class="camera",
        # A registered kind, because admission rule V4 rejects one it cannot build; the
        # transport itself is replaced below.
        transport=TransportRef(kind="mock"),
        channels=(
            Channel(
                channel_id="frame",
                direction=Direction.READ.value,
                quantity="image_frame",
                effect=HardwareEffect.READ.value,
                sample_rate_hz=2.0,
                representation=Representation.FRAME.value,
                media_type="image/jpeg",
                privacy=PrivacyTier.ENVIRONMENT.value,
            ),
        ),
        provenance=ContextProvenance(source=ContextSource.DISCOVERED.value),
    )


def _thermometer() -> HardwareContext:
    return HardwareContext(
        device_id="rig",
        display_name="Bench rig",
        device_class="bench",
        transport=TransportRef(kind="mock"),
        channels=(
            Channel(
                channel_id="temperature",
                quantity="temperature",
                unit="degC",
                envelope=Envelope(declared=True, min_value=0.0, max_value=100.0),
            ),
        ),
        provenance=ContextProvenance(verified_by="tester"),
    )


class _Allow:
    def __init__(self) -> None:
        self.seen: list[object] = []

    async def evaluate(self, descriptor: object) -> object:
        self.seen.append(descriptor)

        class _Result:
            approved = True

        return _Result()


class _Deny:
    async def evaluate(self, descriptor: object) -> object:
        class _Result:
            approved = False
            denial_message = "You declined to share the camera."

        return _Result()


def _board_ctx(*contexts: HardwareContext, gate: object = None, source: object = None):
    """Return (ctx, source) with a loaded registry whose transport is the fake source."""

    class _Provider:
        kind = "fake"

        def discover(self) -> tuple[HardwareContext, ...]:
            return contexts

    registry = HardwareRegistry(HardwareSettings(enabled=True), providers=(_Provider(),))
    registry.load()
    transport = source or _FrameSource()

    async def _transport(device_id: str) -> object:
        return transport

    registry.transport = _transport  # type: ignore[method-assign]
    ctx = SimpleNamespace(
        _hardware_registry=registry, _approval_orchestrator=gate, monitors=None, settings=None
    )
    return ctx, transport, registry


def _board(ctx, command: str) -> dict:
    from leapflow.cli.commands.slash_handlers import _execute_dashboard

    return asyncio.run(_execute_dashboard(ctx, f"board {command}"))


def test_board_devices_groups_peripherals_and_points_at_the_next_command() -> None:
    ctx, _source, _registry = _board_ctx(_camera(), _thermometer())

    result = _board(ctx, "devices")

    assert result["ok"] is True
    assert result["mode"] == "devices", "must not be 'open': this prints, it does not launch a browser"
    assert {group["device_class"] for group in result["groups"]} == {"camera", "bench"}
    assert result["counts"]["previewable"] == 1
    assert "/board device" in result["hint"]


def test_board_devices_says_how_to_enable_hardware_when_it_is_off() -> None:
    """Off is the default, so "nothing here" is the expected state, not a failure."""
    ctx = SimpleNamespace(_hardware_registry=None, monitors=None, settings=None)

    result = _board(ctx, "devices")

    assert result["ok"] is False
    assert "hardware.enabled" in result["message"]
    assert "restart" in result["message"].lower()


def test_a_device_id_resolves_from_a_unique_prefix() -> None:
    """Discovered ids are long, so a prefix is what makes them typeable.

    Matches how ``/board stop`` already resolves a watch id.
    """
    ctx, _source, _registry = _board_ctx(_camera("camera_0_macbook_pro"))

    result = _board(ctx, "device camera_0")

    assert result["ok"] is True
    assert result["mode"] == "open"
    assert result["template"] == "hardware", (
        "one lens with a target, not a second lens: naming a device is what switches the "
        "hardware template from the fleet to that device"
    )
    assert result["device"] == "camera_0_macbook_pro"


def test_an_ambiguous_prefix_names_the_candidates() -> None:
    """"Not found" for a prefix that matched too much sends someone hunting a typo."""
    ctx, _source, _registry = _board_ctx(_camera("camera_0_left"), _camera("camera_1_right"))

    result = _board(ctx, "device camera_")

    assert result["ok"] is False
    assert "camera_0_left" in result["message"]
    assert "camera_1_right" in result["message"]


def test_board_device_without_a_target_asks_for_one() -> None:
    ctx, _source, _registry = _board_ctx(_camera())

    result = _board(ctx, "device")

    assert result["ok"] is False
    assert "/board devices" in result["message"]


def test_board_preview_establishes_consent_then_opens_the_device() -> None:
    """The one frame is the mechanism, not a side effect.

    It goes through the ordinary approval chain from a surface that *has* a route, so the
    prompt appears in the TUI and the grant it produces is what the browser's stream is
    covered by. Requesting zero frames would open a page whose Start button is refused
    with no way to answer.
    """
    gate = _Allow()
    ctx, source, registry = _board_ctx(_camera(), gate=gate)

    result = _board(ctx, "preview cam")

    assert result["ok"] is True
    assert result["template"] == "hardware"
    assert result["device"] == "cam"
    assert result["channel"] == "frame", "the only frame channel must be resolved without naming it"
    assert source.grabs == 1, "consent was not exercised against the device"
    assert gate.seen, "the approval chain was bypassed"
    assert any("released automatically" in note for note in result["notes"]), (
        "the operator is not told the device stops on its own"
    )
    asyncio.run(registry.preview_broker.close())


@pytest.mark.parametrize("gate,label", [(_Deny(), "denied"), (None, "no gate installed")])
def test_a_refused_preview_opens_nothing_and_reads_nothing(gate, label) -> None:
    """Opening a board whose preview cannot work is a dead end, so it is not opened."""
    ctx, source, _registry = _board_ctx(_camera(), gate=gate)

    result = _board(ctx, "preview cam")

    assert result["ok"] is False, label
    assert result.get("mode") != "open", f"{label}: the board was opened despite the refusal"
    assert result["code"] == "consent_required", label
    assert source.grabs == 0, f"{label}: the device was read despite the refusal"


def test_previewing_a_device_with_no_frame_channel_says_so() -> None:
    """A thermometer has no preview, and the refusal points at what it does have."""
    ctx, _source, _registry = _board_ctx(_thermometer(), gate=_Allow())

    result = _board(ctx, "preview rig")

    assert result["ok"] is False
    assert "no previewable channel" in result["message"]
    assert "/board device rig" in result["message"]


def test_board_rescan_reports_what_converged() -> None:
    ctx, _source, _registry = _board_ctx(_camera(), _thermometer())

    result = _board(ctx, "rescan")

    assert result["ok"] is True
    assert result["mode"] == "rescan"
    assert set(result["admitted"]) == {"cam", "rig"}
    assert "/board devices" in result["message"]


def test_the_unknown_verb_message_mentions_the_device_verbs() -> None:
    """An unknown command must list what does exist, including the new vocabulary."""
    ctx, _source, _registry = _board_ctx(_camera())

    result = _board(ctx, "bogus")

    assert result["ok"] is False
    for verb in ("devices", "device", "preview", "rescan"):
        assert verb in result["message"]
