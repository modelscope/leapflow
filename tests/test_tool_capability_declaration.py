# Copyright (c) Alibaba, Inc. and its affiliates.
"""Unit tests for the declarative capability metadata on ToolMetadata.

The resolver (built in a follow-up P1) needs two facts about a tool that today
have to be guessed: what capability it offers (``provides_capabilities``) and
what platform features it needs (``requires_platform_capabilities``). Putting
them on ToolMetadata keeps the SSOT rule intact — one place, per tool — and
makes them visible to schema-only consumers through ``to_openai_schema``.
"""

from __future__ import annotations

from typing import Any

import pytest

from leapflow.domain.platform import Capability
from leapflow.plugins.protocol import ToolMetadata


async def _handler(**kwargs: Any) -> dict[str, Any]:
    return {"ok": True}


def _tool(**overrides: Any) -> ToolMetadata:
    base = dict(
        name="probe",
        description="probe tool",
        parameters_schema={"type": "object", "properties": {}},
        handler=_handler,
    )
    base.update(overrides)
    return ToolMetadata(**base)


def test_capability_fields_default_to_empty_tuples() -> None:
    """Tools that declare nothing pay no schema or metadata cost."""
    tool = _tool()

    assert tool.provides_capabilities == ()
    assert tool.requires_capabilities == ()
    assert tool.requires_platform_capabilities == ()

    schema = tool.to_openai_schema()
    x_leapflow = schema["function"].get("x_leapflow", {})
    assert "provides_capabilities" not in x_leapflow
    assert "requires_capabilities" not in x_leapflow
    assert "requires_platform_capabilities" not in x_leapflow


def test_capability_fields_are_immutable() -> None:
    """ToolMetadata is a frozen dataclass; capability declarations cannot drift."""
    tool = _tool(provides_capabilities=("json.pretty",))

    with pytest.raises(Exception):  # FrozenInstanceError, but we don't couple to the name
        tool.provides_capabilities = ("mutated",)  # type: ignore[misc]


def test_capability_fields_fold_into_openai_schema() -> None:
    """Declared capabilities travel with the schema for schema-only consumers."""
    tool = _tool(
        provides_capabilities=("json.pretty", "json.validate"),
        requires_capabilities=("file.read",),
        requires_platform_capabilities=("shell.exec",),
    )

    x_leapflow = tool.to_openai_schema()["function"]["x_leapflow"]
    assert x_leapflow["provides_capabilities"] == ["json.pretty", "json.validate"]
    assert x_leapflow["requires_capabilities"] == ["file.read"]
    assert x_leapflow["requires_platform_capabilities"] == ["shell.exec"]


def test_x_leapflow_hint_wins_over_field_default() -> None:
    """A tool can override the folded value through x_leapflow when needed.

    The fold uses ``setdefault``, so an explicit ``x_leapflow`` entry keeps its
    shape and stays authoritative -- no accidental data loss when the same key
    appears in both places.
    """
    tool = _tool(
        provides_capabilities=("field.value",),
        x_leapflow={"provides_capabilities": ["hint.override"]},
    )

    x_leapflow = tool.to_openai_schema()["function"]["x_leapflow"]
    assert x_leapflow["provides_capabilities"] == ["hint.override"]


def test_platform_capability_string_matches_enum_value() -> None:
    """Platform requirements are grounded in the Capability enum vocabulary.

    Strings are stored (not the enum) so declarations remain JSON-serializable
    and travel through ``x_leapflow`` cleanly; but the vocabulary must line up
    with what ``PlatformManifest`` actually reports, or environment-fit scoring
    can never resolve a match.
    """
    tool = _tool(requires_platform_capabilities=(Capability.SHELL_EXEC.value,))

    assert tool.requires_platform_capabilities == ("shell.exec",)
    # Round-trip: the string a tool declares can be looked up as a Capability.
    resolved = Capability(tool.requires_platform_capabilities[0])
    assert resolved is Capability.SHELL_EXEC


def test_shell_run_declares_platform_requirement() -> None:
    """The built-in ``shell_run`` tool anchors the annotation in a real plugin.

    Anchoring one real tool proves the field is not a decorative type addition:
    a resolver walking ``registry.all_metadata`` can now exclude ``shell_run``
    on a host that does not report ``shell.exec``.
    """
    from leapflow.plugins.tool_plugins.shell_terminal import ShellTerminalPlugin

    plugin = ShellTerminalPlugin()
    shell_run = next(t for t in plugin.tools if t.name == "shell_run")

    assert Capability.SHELL_EXEC.value in shell_run.requires_platform_capabilities


def _is_builtin_plugin(plugin: Any) -> bool:
    """Return whether a plugin comes from the built-in tool plugin package."""
    if getattr(plugin, "__leapflow_plugin_path__", ""):
        return False
    module_name = str(getattr(plugin.__class__, "__module__", ""))
    return module_name.startswith("leapflow.plugins.tool_plugins.")


def test_builtin_tools_declare_provided_capabilities() -> None:
    """Built-in tools expose declarative capability tags for adaptive selection."""
    from leapflow.plugins.tool_plugins import get_all_plugins

    missing = [
        (plugin.plugin_id, tool.name)
        for plugin in get_all_plugins()
        if _is_builtin_plugin(plugin)
        for tool in plugin.tools
        if not tool.provides_capabilities
    ]

    assert missing == []


def test_builtin_mutating_tools_declare_platform_requirements() -> None:
    """Mutating built-in tools expose host requirements for environment scoring."""
    from leapflow.plugins.tool_plugins import get_all_plugins

    missing = [
        (plugin.plugin_id, tool.name)
        for plugin in get_all_plugins()
        if _is_builtin_plugin(plugin)
        for tool in plugin.tools
        if tool.mutates_state and not tool.requires_platform_capabilities
    ]

    assert missing == []
