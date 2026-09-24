# Copyright (c) Alibaba, Inc. and its affiliates.
"""Module-level helpers for :mod:`physical_skill`.

Split from the main plugin module so the plugin file stays under the
engine sub-package's 800-line ceiling.  Nothing here holds state -- the
tool metadata factory, compute-budget coercion, action-to-command
mapping, and HuggingFace-cache discovery are all pure functions that
take the plugin (or plain arguments) and return derived data.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from leapflow.plugins.protocol import ToolMetadata
from leapflow.robot.inference.strategy import ComputeBudget

if TYPE_CHECKING:
    from leapflow.plugins.tool_plugins.physical_skill import PhysicalSkillPlugin

logger = logging.getLogger(__name__)

__all__ = [
    "build_physical_skill_tools",
    "coerce_compute_budget",
    "action_to_commands",
    "discover_local_policies",
]


def build_physical_skill_tools(
    plugin: PhysicalSkillPlugin,
) -> list[ToolMetadata]:
    """Build the ``ToolMetadata`` list exposed by the physical skill plugin.

    Kept out of the plugin class so the class stays focused on the OODA+V
    execution loop; adding a new tool means editing this factory, not the
    hot path.
    """
    return [
        ToolMetadata(
            name="hw_policy_infer",
            description=(
                "Execute VLA policy inference on a robot device. "
                "Reads observation, runs policy, optionally writes action "
                "to actuators. Supports local (HuggingFace) and remote "
                "(PolicyServer) policies."
            ),
            handler=plugin.policy_infer,
            parameters_schema={
                "type": "object",
                "properties": {
                    "device_id": {"type": "string", "description": "Target robot device"},
                    "policy": {"type": "string", "description": "Policy path or 'remote:host:port'"},
                    "task": {"type": "string", "description": "Natural language task description"},
                    "execute": {"type": "boolean", "description": "Write action to device (default true)"},
                    "chunk_size": {"type": "integer", "description": "Action steps to predict (default 1)"},
                    "verify": {"type": "boolean", "description": "Run post-execution verification"},
                    "compute_budget": {
                        "type": "object",
                        "description": (
                            "Test-time compute budget: trades latency for quality "
                            "by controlling ensemble size, chunk size, and max "
                            "latency. Strategies ignore knobs they do not support."
                        ),
                        "properties": {
                            "max_latency_ms": {
                                "type": "number",
                                "description": "Maximum acceptable inference latency in milliseconds",
                            },
                            "ensemble_size": {
                                "type": "integer",
                                "description": "Number of ensemble samples to draw and vote",
                            },
                            "chunk_size": {
                                "type": "integer",
                                "description": (
                                    "Future action steps to predict in one call "
                                    "(overrides top-level chunk_size)"
                                ),
                            },
                            "max_refinement_steps": {
                                "type": "integer",
                                "description": "Iterative refinement steps to run",
                            },
                            "confidence_threshold": {
                                "type": "number",
                                "description": "Confidence below which more compute should be spent",
                            },
                        },
                    },
                },
                "required": ["device_id", "policy"],
            },
            x_leapflow={
                "category": "hardware",
                "risk_level": "high",
                "mutates_state": True,
                "execution_policy": "serial",
                "requires_capabilities": ["hardware_control"],
                "provides_capabilities": ["physical_skill_inference"],
            },
            mutates_state=True,
            execution_policy="serial",
            requires_capabilities=("hardware_control",),
            provides_capabilities=("physical_skill_inference",),
        ),
        ToolMetadata(
            name="hw_policy_episode",
            description=(
                "Manage policy inference episode lifecycle. Start an episode "
                "to begin a task, stop to finalise and get stats."
            ),
            handler=plugin.policy_episode,
            parameters_schema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["start", "stop", "status"]},
                    "device_id": {"type": "string", "description": "Robot device (for start)"},
                    "policy": {"type": "string", "description": "Policy path (for start)"},
                    "task": {"type": "string", "description": "Task description"},
                },
                "required": ["action"],
            },
            x_leapflow={
                "category": "hardware",
                "risk_level": "medium",
                "mutates_state": False,
            },
        ),
        ToolMetadata(
            name="hw_policy_list",
            description="List available VLA policies and their capabilities.",
            handler=plugin.policy_list,
            parameters_schema={
                "type": "object",
                "properties": {
                    "source": {"type": "string", "enum": ["local", "remote", "all"]},
                },
            },
            x_leapflow={
                "category": "hardware",
                "risk_level": "low",
                "mutates_state": False,
            },
        ),
    ]


def coerce_compute_budget(
    raw: Any,
    fallback_chunk_size: int,
) -> ComputeBudget | None:
    """Build a :class:`ComputeBudget` from raw tool parameters.

    Returns ``None`` when the caller supplied neither a ``compute_budget``
    dict nor a chunk override, so a strategy that does not care about
    budgets is not forced to interpret defaults.  Non-numeric or missing
    knobs fall back to the dataclass defaults; ``chunk_size`` inherits
    from the top-level ``chunk_size`` parameter when the budget dict
    omits it, keeping the tool's call-site behaviour compatible with the
    pre-registry contract.
    """
    if raw is None and fallback_chunk_size <= 1:
        return None
    knobs: dict[str, Any] = {}
    if isinstance(raw, dict):
        for key in (
            "max_latency_ms",
            "max_refinement_steps",
            "ensemble_size",
            "confidence_threshold",
            "chunk_size",
        ):
            if key in raw and raw[key] is not None:
                knobs[key] = raw[key]
    knobs.setdefault("chunk_size", max(1, int(fallback_chunk_size or 1)))
    try:
        return ComputeBudget(**knobs)
    except (TypeError, ValueError):
        logger.warning("Invalid compute_budget %r; ignoring", raw)
        return ComputeBudget(chunk_size=max(1, int(fallback_chunk_size or 1)))


def action_to_commands(
    action: Any,
    device: Any,
) -> list[tuple[str, Any]]:
    """Convert an action vector to ``(channel_id, value)`` command pairs.

    A dict action is trusted verbatim (its keys are the channel ids).  A
    sequence is mapped positionally to the device's writable, non-frame
    channels; extra values without a matching channel are dropped with a
    debug log rather than silently applied to the wrong actuator.
    """
    if isinstance(action, dict):
        return [(str(k), v) for k, v in action.items()]

    # Resolve to a plain list.
    values: list[Any]
    if hasattr(action, "tolist"):
        values = action.tolist()
    elif hasattr(action, "cpu"):
        values = action.cpu().tolist()
    elif isinstance(action, (list, tuple)):
        values = list(action)
    else:
        values = [action]

    # Flatten nested lists (e.g. [[v1, v2, ...]] from chunked inference).
    if len(values) == 1 and isinstance(values[0], list):
        values = values[0]

    # Map values to writable channels from the device context.
    context = getattr(device, "context", None)
    writable_channels: list[str] = []
    if context is not None:
        for ch in context.channels:
            if getattr(ch, "is_writable", False):
                # Skip frame channels.
                if getattr(ch, "representation", "") == "frame":
                    continue
                writable_channels.append(ch.channel_id)

    commands: list[tuple[str, Any]] = []
    for i, val in enumerate(values):
        if i < len(writable_channels):
            commands.append((writable_channels[i], val))
        else:
            logger.debug(
                "Action value at index %d has no mapped writable channel; dropped",
                i,
            )
    return commands


def discover_local_policies() -> list[dict[str, Any]]:
    """Best-effort discovery of locally cached HuggingFace policies.

    Empty when ``huggingface_hub`` is not installed or the scan raises;
    the caller merges the result with already-loaded strategies, so a
    discovery failure only means an incomplete list, never a wrong one.
    """
    try:
        from huggingface_hub import scan_cache_dir  # type: ignore[import-untyped]
    except ImportError:
        return []

    policies: list[dict[str, Any]] = []
    try:
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            # Heuristic: policy repos typically contain "policy" or "act" in the name.
            repo_id = str(repo.repo_id)
            if any(kw in repo_id.lower() for kw in ("policy", "act", "vla", "diffusion")):
                policies.append({
                    "name": repo_id,
                    "type": "local",
                    "loaded": False,
                    "source": "huggingface_cache",
                })
    except Exception:  # noqa: BLE001 - discovery must not crash the plugin
        logger.debug("Local policy discovery failed", exc_info=True)
    return policies
