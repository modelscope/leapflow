"""Tests for CLI help-based command discovery."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from leapflow.gateway.connectors.cli_discovery import (
    CliDiscovery,
    DiscoveredCommand,
    HelpArgument,
    HelpParser,
    _action_name,
    _build_argv_template,
    _build_schema,
    _infer_effect,
)
from leapflow.gateway.connectors.action_registry import ActionRegistry
from leapflow.gateway.connectors.protocol import ActionSpec, BackendKind


# ── Realistic help text fixtures ─────────────────────────────────────

CLICK_TOP_LEVEL_HELP = """\
Usage: lark-cli [OPTIONS] COMMAND [ARGS]...

  Official Feishu/Lark CLI tool.

Options:
  --profile TEXT  Profile name
  --as TEXT       Identity (bot or user)
  --help          Show this message and exit.

Commands:
  auth       Manage authentication
  im         Instant messaging operations
  docs       Document management
  calendar   Calendar event management
  drive      Cloud drive operations
  task       Task management
  mail       Mail operations
"""

CLICK_GROUP_HELP = """\
Usage: lark-cli im [OPTIONS] COMMAND [ARGS]...

  Instant messaging operations for Feishu/Lark.

Commands:
  +messages-send    Send a message to a chat
  +messages-list    List messages in a chat
  +chat-list        List chats the bot is in
  +chat-search      Search chats by keyword
  +chat-create      Create a new group chat
"""

CLICK_COMMAND_HELP = """\
Usage: lark-cli im +messages-send [OPTIONS]

  Send a text message to a Feishu/Lark chat.

Options:
  --chat-id TEXT     Target chat ID (required)
  --text TEXT        Message content (required)
  --thread-id TEXT   Optional thread for replies
  --dry-run          Preview without sending
  --help             Show this message and exit.
"""

COBRA_HELP = """\
Manage Kubernetes resources

Usage:
  kubectl get [resource] [flags]

Flags:
  -o, --output string     Output format (json|yaml|wide)
  -n, --namespace string  Namespace scope (required)
  -l, --selector string   Label selector
      --all-namespaces     List across all namespaces
  -h, --help              help for get
"""

ARGPARSE_HELP = """\
usage: mytool [-h] {send,list,delete} ...

My CLI tool for managing resources.

positional arguments:
  {send,list,delete}
    send              Send a notification
    list              List all items
    delete            Delete an item

options:
  -h, --help          show this help message and exit
"""

MINIMAL_HELP = """\
Usage: simple-tool [command]

Commands:
  run     Execute the main task
  check   Verify configuration
"""


# ── HelpParser tests ─────────────────────────────────────────────────

class TestHelpParser:

    def test_parse_click_top_level(self) -> None:
        parser = HelpParser()
        result = parser.parse(CLICK_TOP_LEVEL_HELP, binary="lark-cli")

        assert result.binary == "lark-cli"
        assert len(result.subcommands) == 7
        names = [cmd.name for cmd in result.subcommands]
        assert "im" in names
        assert "docs" in names
        assert "calendar" in names
        assert result.subcommands[1].name == "im"
        assert "messaging" in result.subcommands[1].description.lower()

    def test_parse_click_group(self) -> None:
        parser = HelpParser()
        result = parser.parse(CLICK_GROUP_HELP, binary="lark-cli", prefix=["im"])

        assert result.prefix == ("im",)
        assert len(result.subcommands) == 5
        names = [cmd.name for cmd in result.subcommands]
        assert "+messages-send" in names
        assert "+chat-search" in names

    def test_parse_click_command_arguments(self) -> None:
        parser = HelpParser()
        result = parser.parse(
            CLICK_COMMAND_HELP,
            binary="lark-cli",
            prefix=["im", "+messages-send"],
        )

        assert len(result.arguments) >= 3
        arg_names = [a.name for a in result.arguments]
        assert "chat_id" in arg_names
        assert "text" in arg_names
        assert "thread_id" in arg_names

        chat_id = next(a for a in result.arguments if a.name == "chat_id")
        assert chat_id.flag == "--chat-id"
        assert chat_id.required is True

    def test_parse_cobra_flags(self) -> None:
        parser = HelpParser()
        result = parser.parse(COBRA_HELP, binary="kubectl", prefix=["get"])

        assert len(result.arguments) >= 3
        ns = next((a for a in result.arguments if a.name == "namespace"), None)
        assert ns is not None
        assert ns.flag == "--namespace"
        assert ns.required is True

    def test_parse_argparse_subcommands(self) -> None:
        parser = HelpParser()
        result = parser.parse(ARGPARSE_HELP, binary="mytool")

        assert len(result.subcommands) == 3
        names = [cmd.name for cmd in result.subcommands]
        assert "send" in names
        assert "delete" in names

    def test_parse_minimal_help(self) -> None:
        parser = HelpParser()
        result = parser.parse(MINIMAL_HELP, binary="simple-tool")

        assert len(result.subcommands) == 2
        assert result.subcommands[0].name == "run"

    def test_parse_empty_help(self) -> None:
        parser = HelpParser()
        result = parser.parse("", binary="empty")

        assert result.subcommands == ()
        assert result.arguments == ()

    def test_description_extraction(self) -> None:
        parser = HelpParser()
        result = parser.parse(CLICK_TOP_LEVEL_HELP, binary="lark-cli")

        assert "official" in result.description.lower() or "feishu" in result.description.lower()

    def test_help_and_version_flags_excluded(self) -> None:
        parser = HelpParser()
        result = parser.parse(CLICK_COMMAND_HELP, binary="lark-cli")

        flag_names = [a.flag for a in result.arguments]
        assert "--help" not in flag_names
        assert "--version" not in flag_names


# ── DiscoveredCommand → ActionSpec builders ──────────────────────────

class TestSpecBuilders:

    def test_action_name_two_level(self) -> None:
        cmd = DiscoveredCommand(
            binary="lark-cli",
            argv_prefix=("im", "+messages-send"),
            group="im",
        )
        assert _action_name(cmd) == "im.messages_send"

    def test_action_name_with_domain(self) -> None:
        cmd = DiscoveredCommand(
            binary="lark-cli",
            argv_prefix=("im", "+messages-send"),
        )
        assert _action_name(cmd, domain="messaging") == "messaging.messages_send"

    def test_action_name_single_level(self) -> None:
        cmd = DiscoveredCommand(
            binary="tool",
            argv_prefix=("status",),
        )
        assert _action_name(cmd) == "status.status"

    def test_build_argv_template(self) -> None:
        cmd = DiscoveredCommand(
            binary="lark-cli",
            argv_prefix=("im", "+messages-send"),
            arguments=(
                HelpArgument(name="chat_id", flag="--chat-id", required=True),
                HelpArgument(name="text", flag="--text", required=True),
                HelpArgument(name="thread_id", flag="--thread-id", required=False),
            ),
        )
        template = _build_argv_template(cmd)
        assert template == ["im", "+messages-send", "--chat-id", "{chat_id}", "--text", "{text}"]

    def test_build_schema(self) -> None:
        cmd = DiscoveredCommand(
            binary="lark-cli",
            argv_prefix=("im", "+send"),
            arguments=(
                HelpArgument(name="chat_id", flag="--chat-id", required=True, description="Target chat"),
                HelpArgument(name="limit", flag="--limit", type_hint="integer"),
            ),
        )
        schema = _build_schema(cmd)
        assert schema["required"] == ["chat_id"]
        assert "chat_id" in schema["properties"]
        assert schema["properties"]["limit"]["type"] == "integer"

    def test_infer_effect_from_command_name(self) -> None:
        assert _infer_effect(DiscoveredCommand(binary="t", argv_prefix=("list",))) == "read"
        assert _infer_effect(DiscoveredCommand(binary="t", argv_prefix=("send",))) == "send"
        assert _infer_effect(DiscoveredCommand(binary="t", argv_prefix=("create",))) == "write"
        assert _infer_effect(DiscoveredCommand(binary="t", argv_prefix=("delete",))) == "execute"

    def test_infer_effect_word_boundary_avoids_false_positives(self) -> None:
        assert _infer_effect(DiscoveredCommand(binary="t", argv_prefix=("reset",))) == "execute"
        assert _infer_effect(DiscoveredCommand(binary="t", argv_prefix=("unset",))) == "execute"
        assert _infer_effect(DiscoveredCommand(binary="t", argv_prefix=("forget",))) == "execute"
        assert _infer_effect(DiscoveredCommand(binary="t", argv_prefix=("offset",))) == "execute"

    def test_infer_effect_matches_hyphenated_commands(self) -> None:
        assert _infer_effect(DiscoveredCommand(binary="t", argv_prefix=("im", "+messages-send"))) == "send"
        assert _infer_effect(DiscoveredCommand(binary="t", argv_prefix=("im", "+chat-list"))) == "read"
        assert _infer_effect(DiscoveredCommand(binary="t", argv_prefix=("admin", "force-delete"))) == "execute"

    def test_infer_effect_from_description(self) -> None:
        cmd = DiscoveredCommand(
            binary="t",
            argv_prefix=("do-thing",),
            description="Search for items in the database",
        )
        assert _infer_effect(cmd) == "read"


# ── CliDiscovery async tests ─────────────────────────────────────────

class TestCliDiscovery:

    @pytest.mark.asyncio
    async def test_discover_parses_help_output(self) -> None:
        discovery = CliDiscovery(binary="lark-cli")
        with patch.object(discovery, "_run_help", return_value=CLICK_TOP_LEVEL_HELP):
            result = await discovery.discover()

        assert result.binary == "lark-cli"
        assert len(result.subcommands) == 7

    @pytest.mark.asyncio
    async def test_discover_group_builds_commands(self) -> None:
        discovery = CliDiscovery(binary="lark-cli")

        async def fake_run_help(argv):
            if argv[-2:] == ["im", "--help"] or "im" in argv and "--help" in argv and "+messages" not in " ".join(argv):
                return CLICK_GROUP_HELP
            return CLICK_COMMAND_HELP

        with patch.object(discovery, "_run_help", side_effect=fake_run_help):
            commands = await discovery.discover_group("im")

        assert len(commands) == 5
        send_cmd = next(c for c in commands if "+messages-send" in c.argv_prefix)
        assert send_cmd.group == "im"
        assert send_cmd.depth == 2
        assert len(send_cmd.arguments) >= 3

    @pytest.mark.asyncio
    async def test_to_action_spec_defaults_high_risk(self) -> None:
        discovery = CliDiscovery(binary="lark-cli")
        cmd = DiscoveredCommand(
            binary="lark-cli",
            argv_prefix=("im", "+messages-send"),
            description="Send a message to a chat",
            arguments=(
                HelpArgument(name="chat_id", flag="--chat-id", required=True),
                HelpArgument(name="text", flag="--text", required=True),
            ),
            group="im",
        )

        spec = discovery.to_action_spec(cmd)

        assert spec.backend_kind == "cli"
        assert spec.risk_level == "high"
        assert spec.effect == "send"
        assert spec.backend_config["discovered"] is True
        assert "chat_id" in spec.schema.get("required", [])
        argv = spec.backend_config["argv"]
        assert list(argv) == ["im", "+messages-send", "--chat-id", "{chat_id}", "--text", "{text}"]

    @pytest.mark.asyncio
    async def test_to_action_spec_dangerous_command_is_critical(self) -> None:
        discovery = CliDiscovery(binary="tool")
        cmd = DiscoveredCommand(
            binary="tool",
            argv_prefix=("admin", "delete-all"),
            description="Delete all resources permanently",
        )

        spec = discovery.to_action_spec(cmd)

        assert spec.risk_level == "critical"

    @pytest.mark.asyncio
    async def test_cache_avoids_repeated_help_calls(self) -> None:
        discovery = CliDiscovery(binary="lark-cli", cache_ttl_s=60.0)
        call_count = 0

        async def counting_run_help(argv):
            nonlocal call_count
            call_count += 1
            return CLICK_TOP_LEVEL_HELP

        with patch.object(discovery, "_run_help", side_effect=counting_run_help):
            await discovery.discover()
            await discovery.discover()
            await discovery.discover()

        assert call_count == 1

    @pytest.mark.asyncio
    async def test_invalidate_clears_cache(self) -> None:
        discovery = CliDiscovery(binary="lark-cli", cache_ttl_s=60.0)
        call_count = 0

        async def counting_run_help(argv):
            nonlocal call_count
            call_count += 1
            return CLICK_TOP_LEVEL_HELP

        with patch.object(discovery, "_run_help", side_effect=counting_run_help):
            await discovery.discover()
            discovery.invalidate()
            await discovery.discover()

        assert call_count == 2


# ── ActionRegistry + discovery integration ───────────────────────────

class TestRegistryDiscovery:

    def test_merge_discovered_does_not_overwrite_static(self) -> None:
        static_spec = ActionSpec(
            name="im.send_message",
            backend_kind="cli",
            description="Static verified spec",
            risk_level="high",
        )
        registry = ActionRegistry({"im.send_message": static_spec})

        discovered_spec = ActionSpec(
            name="im.send_message",
            backend_kind="cli",
            description="Discovered draft spec",
            risk_level="high",
        )
        added = registry.merge_discovered([discovered_spec])

        assert added == 0
        assert registry.get("im.send_message") is static_spec
        assert not registry.is_discovered("im.send_message")

    def test_merge_discovered_adds_new_specs(self) -> None:
        static_spec = ActionSpec(
            name="im.send_message",
            backend_kind="cli",
        )
        registry = ActionRegistry({"im.send_message": static_spec})

        new_spec = ActionSpec(
            name="calendar.create_event",
            backend_kind="cli",
            description="Discovered via --help",
        )
        added = registry.merge_discovered([new_spec])

        assert added == 1
        assert registry.get("calendar.create_event") is new_spec
        assert registry.is_discovered("calendar.create_event")

    def test_all_returns_merged_specs(self) -> None:
        static = ActionSpec(name="im.send", backend_kind="cli")
        discovered = ActionSpec(name="docs.create", backend_kind="cli")
        registry = ActionRegistry({"im.send": static})
        registry.merge_discovered([discovered])

        all_specs = registry.all()

        assert "im.send" in all_specs
        assert "docs.create" in all_specs
        assert all_specs["im.send"] is static

    def test_static_and_discovered_specs_separated(self) -> None:
        static = ActionSpec(name="im.send", backend_kind="cli")
        discovered = ActionSpec(name="docs.create", backend_kind="cli")
        registry = ActionRegistry({"im.send": static})
        registry.merge_discovered([discovered])

        assert "im.send" in registry.static_specs()
        assert "im.send" not in registry.discovered_specs()
        assert "docs.create" in registry.discovered_specs()
        assert "docs.create" not in registry.static_specs()

    @pytest.mark.asyncio
    async def test_refresh_discovery_uses_attached_source(self) -> None:
        discovered_spec = ActionSpec(
            name="task.list",
            backend_kind="cli",
            description="List tasks",
        )

        class FakeDiscovery:
            async def discover_actions(self, *, groups=()):
                return [discovered_spec]

        registry = ActionRegistry({}, discovery=FakeDiscovery())
        added = await registry.refresh_discovery()

        assert added == 1
        assert registry.get("task.list") is discovered_spec

    @pytest.mark.asyncio
    async def test_refresh_discovery_without_source_returns_zero(self) -> None:
        registry = ActionRegistry({})
        added = await registry.refresh_discovery()
        assert added == 0

    @pytest.mark.asyncio
    async def test_refresh_discovery_error_is_swallowed(self) -> None:
        class FailingDiscovery:
            async def discover_actions(self, *, groups=()):
                raise RuntimeError("network error")

        registry = ActionRegistry({}, discovery=FailingDiscovery())
        added = await registry.refresh_discovery()
        assert added == 0
