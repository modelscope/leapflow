# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tool definitions for the node-skill tools ``SkillService`` serves (see ``skill_tools.py``)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from leapflow.hardware.skills.episodes import MAX_INTENT_CHARS, MAX_LABEL_NOTE_CHARS
from leapflow.plugins.protocol import ToolMetadata

if TYPE_CHECKING:
    from leapflow.hardware.skill_tools import SkillService

DEFAULT_RUN_TIMEOUT_S = 120.0
MAX_RUN_TIMEOUT_S = 600.0

_WRITE_METADATA: dict[str, Any] = {
    "category": "hardware",
    "risk_level": "external",
    "requires_approval": True,
    "effect_scope": "external",
    "idempotency_scope": "session",
    "schema_cost": "low",
}
_READ_METADATA: dict[str, Any] = {
    "category": "hardware",
    "risk_level": "read_only",
    "requires_approval": False,
    "schema_cost": "low",
}
_RUN_ID = {"type": "string", "description": "Run id from hw_skill_run (omit for all runs)"}
_RUN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "workcell": {"type": "string", "description": "<node>/<workcell>"},
        "skill": {"type": "string", "description": "Skill id from hw_skill_list"},
        "args": {"type": "object", "description": "Arguments per args_schema"},
        "dry_run": {"type": "boolean", "description": "Plan only; nothing moves"},
        "intent": {
            "type": "string",
            "maxLength": MAX_INTENT_CHARS,
            "description": (
                "The sub-goal this run serves, in one sentence (e.g. 'put the red "
                "cup on the tray'); recorded with the run's episode as its language "
                "label. Changes nothing the run does."
            ),
        },
        "wait": {"type": "boolean", "description": "Wait for the outcome (default true)"},
        "timeout_s": {
            "type": "number",
            "description": (
                f"Longest wait before stopping the run (default "
                f"{DEFAULT_RUN_TIMEOUT_S:g}, max {MAX_RUN_TIMEOUT_S:g})"
            ),
        },
    },
    "required": ["workcell", "skill"],
}


def build_skill_tools(service: "SkillService") -> list[ToolMetadata]:
    """The node-skill tools bound to *service*."""
    return [
        ToolMetadata(
            name="hw_skill_list",
            description=(
                "List robot workcells on the hardware nodes and the skills each offers: skill "
                "cards, argument schemas, availability, calibration, and limits. Start here "
                "before hw_skill_run; workcell ids are qualified as <node>/<workcell>."
            ),
            parameters_schema={
                "type": "object",
                "properties": {"node": {"type": "string", "description": "Limit to one node"}},
            },
            handler=service.hw_skill_list,
            x_leapflow=dict(_READ_METADATA),
            provides_capabilities=("hw.skill_list",),
        ),
        ToolMetadata(
            name="hw_skill_run",
            description=(
                "Run a node skill on a workcell (a perception query, a motion program, a "
                "policy). The node plans the run; an actuating plan is shown to the user for "
                "approval, then the node executes it while the host keeps its lease alive. Use "
                "dry_run to see the plan without moving. A failed run whose side_effect_state "
                "is not 'none' must never be repeated: the next attempt is a new plan."
            ),
            parameters_schema=dict(_RUN_SCHEMA),
            handler=service.hw_skill_run,
            x_leapflow={
                **_WRITE_METADATA,
                "mutates_state": True,
                "execution_policy": "external_side_effect",
            },
            mutates_state=True,
            execution_policy="external_side_effect",
            provides_capabilities=("hw.skill_run",),
        ),
        ToolMetadata(
            name="hw_skill_status",
            description=(
                "Report skill runs: still running, or their outcome with failure_code, "
                "side_effect_state, and evidence. Has no physical effect."
            ),
            parameters_schema={"type": "object", "properties": {"run_id": dict(_RUN_ID)}},
            handler=service.hw_skill_status,
            x_leapflow=dict(_READ_METADATA),
            provides_capabilities=("hw.skill_status",),
        ),
        ToolMetadata(
            name="hw_skill_stop",
            description=(
                "Stop a skill run (or every run) and halt what it moved. Never requires "
                "approval; use it whenever a run behaves unexpectedly."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "run_id": dict(_RUN_ID),
                    "reason": {"type": "string", "description": "Why, for the audit log"},
                },
            },
            handler=service.hw_skill_stop,
            x_leapflow={**_WRITE_METADATA, "requires_approval": False, "idempotency_scope": "turn"},
            provides_capabilities=("hw.skill_stop",),
        ),
        ToolMetadata(
            name="hw_skill_label",
            description=(
                "Label a finished run with whether its goal was actually reached, after you "
                "checked (e.g. located the object where it should be). The label is what the "
                "run's recorded episode is trained on; a run's own 'succeeded' only means it "
                "completed its plan. Has no physical effect."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "run_id": {"type": "string", "description": "Run id from hw_skill_run"},
                    "success": {"type": "boolean", "description": "Whether the goal was met"},
                    "note": {
                        "type": "string",
                        "maxLength": MAX_LABEL_NOTE_CHARS,
                        "description": "What you observed, briefly",
                    },
                },
                "required": ["run_id", "success"],
            },
            handler=service.hw_skill_label,
            x_leapflow={**_READ_METADATA, "idempotency_scope": "turn"},
            provides_capabilities=("hw.skill_label",),
        ),
    ]


__all__ = ["DEFAULT_RUN_TIMEOUT_S", "MAX_RUN_TIMEOUT_S", "build_skill_tools"]
