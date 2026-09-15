# Copyright (c) Alibaba, Inc. and its affiliates.
"""R7 — adaptive plugin closed loop through a real daemon.

Phases: missing capability evidence is observed, a fixture plugin is installed
through the real self-management tool and approval path, the new tool is usable,
then disable/remove mutate the live registry and the command surface reflects the
change.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests._harness.cassette_proxy import answer, scripted, tool_call
from tests._harness.journey import JourneyFactory
from tests._harness.leapd import await_for

SUBJECT_PATHS = (
    "src/leapflow/plugins/",
    "src/leapflow/learning/capability_observation.py",
    "src/leapflow/storage/capability_plan_store.py",
    "src/leapflow/daemon/",
    "src/leapflow/cli/commands/slash_handlers.py",
)

# The journey exercises daemon/plugin mutation wiring rather than model quality.
LIVE_SIGNAL = False

SESSION = "r7-adaptive-loop"
PLUGIN_ID = "json_pretty_loop_e2e"
TOOL_NAME = "json_pretty_loop_e2e"

PLUGIN_CODE = """from __future__ import annotations

import json
from typing import Any

from leapflow.plugins.protocol import ToolMetadata


async def json_pretty_loop_e2e(text: str = "", **kwargs: Any) -> dict[str, Any]:
    payload = text or kwargs.get("payload") or "{}"
    try:
        parsed = json.loads(str(payload))
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": str(exc)}
    return {"ok": True, "content": json.dumps(parsed, ensure_ascii=False, indent=2, sort_keys=True)}


class JsonPrettyLoopE2EPlugin:
    @property
    def plugin_id(self) -> str:
        return "json_pretty_loop_e2e"

    @property
    def category(self) -> str:
        return "formatting"

    @property
    def dependencies(self) -> list[str]:
        return []

    @property
    def tools(self) -> list[ToolMetadata]:
        return [
            ToolMetadata(
                name="json_pretty_loop_e2e",
                description="Pretty-print JSON for the adaptive closed-loop journey.",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "JSON text to format"}
                    },
                },
                handler=json_pretty_loop_e2e,
                x_leapflow={
                    "category": "formatting",
                    "risk_level": "read_only",
                    "schema_cost": "low",
                    "requires_approval": False,
                },
                provides_capabilities=("json.pretty",),
                requires_platform_capabilities=("file.ops",),
            )
        ]

    def bind_runtime(self, **deps: Any) -> None:
        return None


plugin = JsonPrettyLoopE2EPlugin()
"""


async def _drive_with_auto_approval(
    client: Any,
    message: str,
    *,
    session_id: str,
    workspace: str,
) -> list[Any]:
    events: list[Any] = []
    async for event in client.engine_chat(message, session_id=session_id, workspace_root=workspace):
        events.append(event)
        if event.type == "approval_request":
            approval = (event.metadata or {}).get("approval") or {}
            pending_id = str(approval.get("pending_id") or "")
            assert pending_id, f"approval event lacked pending_id: {event.metadata}"
            await client.approval_resolve(pending_id, "allow_once", reason="r7 adaptive loop")
    return events


def _completed(events: list[Any], tool_name: str) -> bool:
    return any(event.type == "tool_complete" and event.content == tool_name for event in events)


async def _plugin_ids(client: Any) -> set[str]:
    payload = await client.command_execute("plugin list", session_id=SESSION)
    assert payload.get("ok") is True, f"/plugin list failed: {payload}"
    return {str(item.get("plugin_id")) for item in payload.get("plugins") or []}


async def _latest_plan(client: Any) -> dict[str, Any] | None:
    payload = await client.command_execute("plugin plan", "--latest", session_id=SESSION)
    assert payload.get("ok") is True, f"/plugin plan failed: {payload}"
    return payload.get("latest")


@pytest.mark.asyncio
async def test_r7_adaptive_plugin_closed_loop(journeys: JourneyFactory) -> None:
    journey = journeys(
        "r7_adaptive_plugin_loop",
        script=scripted(
            tool_call("missing_json_pretty_e2e", text='{"b":2,"a":1}'),
            answer("Observed the missing JSON pretty tool."),
            tool_call("plugin_install", plugin_id=PLUGIN_ID, code=PLUGIN_CODE, version_label="r7"),
            answer("Installed json pretty plugin."),
            tool_call(TOOL_NAME, text='{"b":2,"a":1}'),
            answer("Formatted JSON with the new plugin."),
            tool_call("plugin_remove", plugin_id=PLUGIN_ID, delete_source=True),
            answer("Removed json pretty plugin."),
        ),
        deadline_s=120.0,
        max_llm_calls=12,
        max_llm_tokens=220_000,
    )
    workspace = journey.workspace("adaptive")
    client = journey.client(timeout_s=180.0)

    with journey.phase("baseline: plugin is absent"):
        assert PLUGIN_ID not in await _plugin_ids(client)

    with journey.phase("observe: unknown tool writes an adaptive plan record"):
        events = await _drive_with_auto_approval(
            client,
            "Try the missing_json_pretty_e2e tool so LeapFlow records a capability gap.",
            session_id=SESSION,
            workspace=str(workspace),
        )
        assert _completed(events, "missing_json_pretty_e2e"), [event.type for event in events]
        latest = await await_for(
            lambda: _latest_plan(client), timeout_s=10.0, what="observed capability plan"
        )
        assert latest.get("source") == "engine_observe", latest
        assert latest.get("phase") == "observation", latest

    with journey.phase("install: approval-gated tool mutates live registry"):
        events = await _drive_with_auto_approval(
            client,
            "Install the prepared adaptive JSON pretty plugin.",
            session_id=SESSION,
            workspace=str(workspace),
        )
        assert _completed(events, "plugin_install"), [event.type for event in events]
        assert PLUGIN_ID in await await_for(
            lambda: _plugin_ids(client), timeout_s=10.0, what="installed plugin visible in registry"
        )

    with journey.phase("use: newly installed tool is executable"):
        events = await _drive_with_auto_approval(
            client,
            "Use the json_pretty_loop_e2e tool to format the sample JSON.",
            session_id=SESSION,
            workspace=str(workspace),
        )
        assert _completed(events, TOOL_NAME), [event.type for event in events]

    with journey.phase("disable: registry strategy removes the plugin from selection"):
        disabled = await client.command_execute(
            "plugin disable",
            PLUGIN_ID,
            session_id=SESSION,
            on_stream_event=lambda event: _approve_event(client, event),
        )
        assert disabled.get("ok") is True, f"disable failed: {disabled}"
        assert PLUGIN_ID not in await _plugin_ids(client)

    with journey.phase("remove: terminal cleanup clears profile source"):
        events = await _drive_with_auto_approval(
            client,
            "Remove the adaptive JSON pretty plugin completely.",
            session_id=SESSION,
            workspace=str(workspace),
        )
        assert _completed(events, "plugin_remove"), [event.type for event in events]
        assert PLUGIN_ID not in await _plugin_ids(client)

    journey.finish()


async def _approve_event(client: Any, event: Any) -> None:
    if event.type != "approval_request":
        return
    approval = (event.metadata or {}).get("approval") or {}
    pending_id = str(approval.get("pending_id") or "")
    assert pending_id, f"approval event lacked pending_id: {event.metadata}"
    await client.approval_resolve(pending_id, "allow_once", reason="r7 adaptive loop")
