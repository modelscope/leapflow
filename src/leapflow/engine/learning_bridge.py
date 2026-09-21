# Copyright (c) Alibaba, Inc. and its affiliates.
"""Learning/observation bridge helpers for :class:`AgentEngine`.

Extracted from ``engine.py`` (Phase 3 refactor). This component owns the
post-turn review, episode persistence, world-model / experience-store bridging,
learning event emission (chat interactions, episodes, execution traces),
semantic tool-focus recording, and the observe-only capability / co-evolution
outcome recording. It holds a back-reference to the owning engine so every
access reads the engine's *live* mutable state (stores injected at runtime via
``set_*`` methods), preserving exact runtime semantics.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List

from leapflow.engine.session.session import SessionMode
from leapflow.engine.context.context_focus import ContextPlane
from leapflow.engine.tools.execution_trace import ExecutionMode, ExecutionTrace
from leapflow.engine.turn_usage import build_adaptive_learning_signal
from leapflow.engine._tool_helpers import _default_tool_registry

if TYPE_CHECKING:  # pragma: no cover - typing only
    from leapflow.engine.engine import AgentEngine

logger = logging.getLogger(__name__)


class LearningBridge:
    """Learning-signal emission and observation bridge, held by composition."""

    # Deprecated fallback: name-based context_plane inference.
    # Tools should declare context_plane via x_leapflow metadata in their spec.
    _EVIDENCE_TOOL_NAMES: frozenset[str] = frozenset(
        {"file_read", "web_fetch", "code_search", "text_search", "memory_search"}
    )

    def __init__(self, engine: "AgentEngine") -> None:
        self._engine = engine

    def _emit_chat_event(self, sub_action: str, payload: Dict[str, Any]) -> None:
        """Emit a chat interaction event for trajectory recording during LEARNING.

        Only fires when the session is in LEARNING mode and an EventBus is available.
        The recorder's state machine ensures these events are only persisted as
        trajectory steps when recording is active.
        """
        if self._engine._event_bus is None:
            return
        if self._engine._session is None or self._engine._session.mode != SessionMode.LEARNING:
            return
        from leapflow.domain.events import SystemEvent

        event = SystemEvent(
            event_type="chat.interaction",
            source="leapflow.engine",
            payload={"action": sub_action, **payload},
            timestamp=time.time(),
        )
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                self._engine._event_bus.handle_event(
                    event.event_type,
                    event.payload,
                )
            )
        except RuntimeError:
            pass

    def _record_tool_focus(
        self,
        tool_name: str,
        arguments: Dict[str, Any] | None,
        result: Any,
    ) -> None:
        """Record semantic focus/control-plane state from a completed tool."""
        try:
            self._engine._focus_state.record_tool_result(
                tool_name,
                arguments or {},
                result,
                turn_id=self._engine._focus_turn_id(),
            )
        except (TypeError, ValueError, RuntimeError):
            logger.debug("semantic focus update failed for tool %s", tool_name, exc_info=True)

    def _tool_focus_metadata(
        self,
        tool_name: str,
        arguments: Dict[str, Any] | None,
        result: Any,
    ) -> Dict[str, Any]:
        """Return compact metadata describing a tool result's context plane."""
        name = str(tool_name or "").removeprefix("gp_")

        # Primary path: check tool manifest metadata (declarative)
        spec = _default_tool_registry().specs.get(name)
        if spec is not None:
            declared_plane = getattr(spec, "context_plane", None)
            if declared_plane:
                return {"context_plane": declared_plane}

        # Deprecated fallback: name-based inference (to be removed once all tools declare metadata)
        if name.startswith("config_"):
            logger.debug(
                "context_plane inferred from prefix for %s "
                "(deprecated; declare x_leapflow.context_plane)",
                name,
            )
            metadata: Dict[str, Any] = {"context_plane": ContextPlane.CONTROL_PLANE.value}
            if isinstance(result, dict):
                key = str(result.get("key") or (arguments or {}).get("key") or "")
                if key:
                    metadata["control_event_key"] = key
            return metadata
        if name in self._EVIDENCE_TOOL_NAMES:
            logger.debug(
                "context_plane inferred from name set for %s "
                "(deprecated; declare x_leapflow.context_plane)",
                name,
            )
            return {"context_plane": ContextPlane.TOOL_EVIDENCE.value}
        return {}

    async def _post_turn_review(self, messages: List[Dict[str, Any]], final_content: str) -> None:
        """Background post-turn review: detect memorable patterns and persist episodes.

        Scans the turn's tool calls for interesting patterns (successes, failures)
        and records them as skill episodes for evolution learning. Delegates
        persistence, world-model bridging, and event emission to focused helpers.
        """
        try:
            tool_actions: List[Dict[str, Any]] = []
            for msg in messages:
                if msg.get("role") == "assistant":
                    for tc in msg.get("tool_calls") or []:
                        fn = tc.get("function", {})
                        tool_actions.append(
                            {
                                "tool": fn.get("name", ""),
                                "args_preview": fn.get("arguments", "")[:100],
                            }
                        )

            if not tool_actions:
                return

            has_success = any(
                '"ok": true' in m.get("content", "") or '"ok":true' in m.get("content", "")
                for m in messages
                if m.get("role") in ("tool", "user")
            )
            has_failure = any(
                '"ok": false' in m.get("content", "") or '"ok":false' in m.get("content", "")
                for m in messages
                if m.get("role") in ("tool", "user")
            )

            reward = 0.5
            if has_success and not has_failure:
                reward = 1.0
            elif has_failure and not has_success:
                reward = -0.5

            skill_name = tool_actions[0]["tool"] if tool_actions else "unknown"
            episode_context = {"final_content_preview": final_content[:200]}
            episode_context.update(self._engine._usage_tracker.to_learning_signal())
            episode_context.update(
                build_adaptive_learning_signal(self._engine._last_context_snapshot or {})
            )
            episode = self._engine._evolution.record_episode(
                skill_name=f"turn_{skill_name}",
                actions=tool_actions[:10],
                outcome="completed" if has_success else "mixed",
                reward=reward,
                context=episode_context,
            )

            self._persist_episode(episode)
            self._bridge_to_experience_store(
                episode, tool_actions, reward, has_success, has_failure
            )
            self._emit_episode_event(episode, reward)
        except Exception:
            logger.debug("post_turn_review failed", exc_info=True)

    def _persist_episode(self, episode: Any) -> None:
        """Incremental persistence: write episode to DuckDB immediately."""
        if self._engine._evolution_store is None or episode is None:
            return
        try:
            self._engine._evolution_store.save_episode(
                episode_id=episode.episode_id,
                skill_name=episode.skill_name,
                actions=episode.actions,
                outcome=episode.outcome,
                reward=episode.reward,
                context=episode.context,
                timestamp=episode.timestamp,
            )
        except Exception:
            logger.debug("evolution_store.save_episode failed", exc_info=True)

    def _bridge_to_experience_store(
        self,
        episode: Any,
        tool_actions: List[Dict[str, Any]],
        reward: float,
        has_success: bool,
        has_failure: bool,
    ) -> None:
        """Bridge tool-loop outcomes to ExperienceStore for world-model trajectory."""
        if self._engine._experience_store is None or episode is None:
            return
        try:
            tool_names = ",".join(a.get("tool", "") for a in tool_actions[:3])
            self._engine._experience_store.store(
                action_description=f"chat_tools:{tool_names}",
                app_context="",
                predicted_effect="",
                actual_effect=episode.outcome,
                delta=abs(reward),
                grade_label="helpful" if has_success and not has_failure else "mixed",
            )
        except Exception:
            logger.debug("experience_store.store failed", exc_info=True)

    def _emit_episode_event(self, episode: Any, reward: float) -> None:
        """Emit high-value episodes to EventBus for active learning consumption."""
        if episode is None or self._engine._event_bus is None:
            return
        threshold = getattr(self._engine._settings, "episode_emit_reward_threshold", 0.8)
        if abs(reward) < threshold:
            return
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                self._engine._event_bus.handle_event(
                    "learning.episode_recorded",
                    {
                        "skill_name": episode.skill_name,
                        "reward": episode.reward,
                        "actions": [a.get("tool", "") for a in episode.actions[:5]],
                        "outcome": episode.outcome,
                    },
                )
            )
        except RuntimeError:
            pass

    def _observe_capability_results(self, results: List[Dict[str, Any]]) -> None:
        """Observe structured tool results without mutating runtime state."""
        for item in results:
            result = item.get("result") if isinstance(item, dict) else None
            self._observe_capability_result(result)
            self._record_coevolution_outcome(
                item, str(getattr(self._engine._settings, "workspace_root", "") or "")
            )

    @staticmethod
    def _record_coevolution_outcome(item: Any, workspace: str = "") -> None:
        """Pair a tool outcome with the requirement its plugin was selected to serve.

        Recorded here rather than at the usage sink because this is the only place that
        sees the *full result payload*, and the payload is where a tool reports what it
        observably did. Without that, a successful call can only be graded
        ``unverifiable`` -- so verification could refute an acquisition but never
        confirm one.

        A no-op for every plugin the system did not acquire, which is almost all of
        them. Bookkeeping only: never raises.
        """
        if not isinstance(item, dict):
            return
        try:
            from leapflow.evolution.observations import record_tool_outcome
            from leapflow.learning.capability_effect_verifier import (
                observed_effect_from_result,
            )
            from leapflow.plugins import get_registry

            tool_name = str(item.get("name") or "")
            if not tool_name:
                return
            plugin_id = str((get_registry().tool_owners or {}).get(tool_name) or "")
            if not plugin_id:
                return
            result = item.get("result")
            ok = True
            if isinstance(result, dict):
                ok = bool(result.get("ok", True)) and not result.get("error")
            record_tool_outcome(
                plugin_id,
                tool_name,
                ok,
                observed_effect=observed_effect_from_result(result),
                workspace=workspace,
            )
        except Exception:  # noqa: BLE001 - observation must never affect execution
            logger.debug("co-evolution outcome not recorded", exc_info=True)

    def _observe_capability_result(self, result: Any) -> None:
        """Persist an observe-only adaptive capability plan from structured gaps.

        This hook intentionally performs no install, disable, remove, retry, or
        natural-language classification. It only reflects structured tool-result
        evidence into the capability plan store so the next disclosure/planning
        step can see an explicit, reviewable requirement.
        """
        if not isinstance(result, dict):
            return
        try:
            buffer = getattr(self._engine, "_capability_observation_buffer", None)
            if buffer is None:
                from leapflow.learning.capability_observation import (
                    CapabilityEvidenceClassifier,
                    CapabilityObservationBuffer,
                )

                # The buffer gate runs first, so it must honour the same accepted
                # set as the durable service; otherwise a configured evidence kind
                # would be dropped here and the setting would have no effect.
                buffer = CapabilityObservationBuffer(
                    classifier=CapabilityEvidenceClassifier.from_settings(self._engine._settings)
                )
                self._engine._capability_observation_buffer = buffer
            if not buffer.add_result(result):
                return

            profile_layout = getattr(self._engine._settings, "profile_layout", None)
            if profile_layout is None:
                return

            from leapflow.domain.environment_fingerprint import EnvironmentFingerprint
            from leapflow.domain.platform import PlatformManifest
            from leapflow.learning.capability_observation import (
                CapabilityEvidenceClassifier,
                CapabilityObservationService,
            )
            from leapflow.plugins import get_registry
            from leapflow.plugins.adaptive_loop import (
                AdaptiveLoopRequest,
                AdaptivePluginLoop,
                live_learning_signals,
            )
            from leapflow.storage.capability_observation_store import JsonCapabilityObservationStore
            from leapflow.storage.capability_plan_store import JsonCapabilityPlanStore

            registry = get_registry()
            environment = EnvironmentFingerprint.from_platform_manifest(
                PlatformManifest.default_darwin(),
                workspace_root=getattr(self._engine._settings, "workspace_root", ""),
            )
            observation_store = JsonCapabilityObservationStore(
                profile_layout.capability_observations_path
            )
            observation_service = CapabilityObservationService(
                observation_store,
                classifier=CapabilityEvidenceClassifier.from_settings(self._engine._settings),
            )
            observation_record = observation_service.observe_result(
                result,
                environment=environment,
                source="engine_observe",
                session_id=str(getattr(self._engine, "_current_session_id", "") or ""),
                turn_id=str(getattr(self._engine, "_current_turn_id", "") or ""),
                workspace_root=str(getattr(self._engine._settings, "workspace_root", "") or ""),
            )
            requirements = observation_service.requirements(min_count=1)
            if not requirements:
                return
            loop_id = "observe-{}-{}".format(
                str(
                    getattr(self._engine, "_current_turn_id", "")
                    or getattr(self._engine, "_current_session_id", "")
                    or "turn"
                ),
                len(buffer.observations()),
            )
            store = JsonCapabilityPlanStore(profile_layout.capability_plans_path)
            trust_ledger, usage_tracker = live_learning_signals()
            loop = AdaptivePluginLoop(
                registry=registry,
                plan_store=store,
                # Without these two, ``TrustScorer`` and ``ReliabilityScorer`` report
                # "unavailable" and score 0 for every candidate, so the two adaptive
                # signals contribute nothing and an alphabetical tie-break decides.
                trust_ledger=trust_ledger,
                usage_tracker=usage_tracker,
                # The live settings, not ``get_settings()``: that singleton is a boot
                # snapshot with no refresh path, while ``_settings`` is what
                # ``reconfigure_runtime`` replaces. Pushing it is what makes
                # ``selection.policy`` genuinely hot-reloadable.
                settings=self._engine._settings,
                # Channel C2: the teacher's rebind recommendation becomes a *preference*
                # in scoring. Resolved through the engine's own store so it follows the
                # same expiry and retraction as the knowledge it came from.
                distilled_preferences=self._engine._prompt_assembler._rebind_preferences,
            )
            decision = loop.resolve_once(
                AdaptiveLoopRequest(
                    environment=environment,
                    requirements=requirements,
                    source="engine_observe",
                    loop_id=loop_id,
                ),
                phase="observation",
                registry_version_before=registry.version,
                registry_version_after=registry.version,
                mutation={
                    "action": "observe",
                    # The real evidence kind, not a hardcoded literal. Stamping every
                    # observation as "unknown_tool" made the causal ledger classify a
                    # world-model or environment-driven episode as an unknown-tool one,
                    # so the driver attribution on the board was wrong for exactly the
                    # episodes self-evolution cares about.
                    "error_type": str(result.get("error_type") or "unknown_tool"),
                    "observation_id": (observation_record or {}).get("observation_id", ""),
                },
            )
            self._engine._active_capability_plan = decision.plan.to_dict()
            # Retire evidence whose gap this resolution closed. Without it the
            # observation backlog only ever grows and keeps reporting capabilities
            # the system already has.
            for resolution in getattr(decision, "resolutions", ()):
                self._record_coevolution_resolution(resolution)
                if getattr(resolution, "unmet", True):
                    continue
                capability = getattr(getattr(resolution, "requirement", None), "capability", "")
                if capability:
                    observation_service.resolve_capability(
                        capability, reason=f"resolved in {loop_id}"
                    )
        except (ImportError, AttributeError, RuntimeError, OSError, TypeError, ValueError) as exc:
            logger.debug("capability observation skipped: %s", exc, exc_info=True)

    @staticmethod
    def _record_coevolution_resolution(resolution: Any) -> None:
        """Report one resolution to the co-evolution buffer for the cold-path sweep.

        Exclusions are recorded as the excluded component's **scorer name**
        (``risk_cost``, ``environment_affordance``, ...) rather than its prose. The
        reaper needs to tell a durable exclusion from an environment one, and keying
        that off a human-readable reason would stop working the moment the resolver
        rewords it.

        Bookkeeping only: never raises, so a buffer problem cannot disturb the turn
        that produced the resolution.
        """
        try:
            from leapflow.evolution.observations import record_resolution

            selected = getattr(resolution, "selected", None)
            selected_id = ""
            if selected is not None:
                selected_id = str(getattr(getattr(selected, "candidate", None), "plugin_id", ""))
            exclusions: dict[str, list[str]] = {}
            for score in getattr(resolution, "candidates", ()) or ():
                plugin_id = str(getattr(getattr(score, "candidate", None), "plugin_id", ""))
                if not plugin_id or getattr(score, "eligible", False):
                    continue
                exclusions[plugin_id] = [
                    str(getattr(component, "scorer", ""))
                    for component in getattr(score, "components", ()) or ()
                    if getattr(component, "excluded", False)
                ]
            record_resolution(
                requirement=getattr(resolution, "requirement", None),
                selected_plugin=selected_id,
                exclusions=exclusions,
            )
        except Exception:  # noqa: BLE001 - observation must never affect execution
            logger.debug("co-evolution resolution not recorded", exc_info=True)

    async def _emit_execution_trace(self, trace: ExecutionTrace) -> None:
        """Fire-and-forget: emit trace as learning signal for the evolution ring."""
        try:
            logger.debug("emit_trace steps=%d tokens=%d", trace.step_count, trace.total_tokens)
            # Write episode to evolution memory if available
            if self._engine._evolution and self._engine._settings.memory_integration_enabled:
                actions = [
                    {"state": e.state.value, **(e.action or {})}
                    for e in trace.entries
                    if e.state == ExecutionMode.ACTING and e.action
                ]
                outcome = "success" if trace.success else "failure"
                reward = 1.0 if trace.success else -0.5
                self._engine._evolution.record_episode(
                    skill_name="react_loop",
                    actions=actions,
                    outcome=outcome,
                    reward=reward,
                    context={
                        "steps": trace.step_count,
                        "tokens": trace.total_tokens,
                        **build_adaptive_learning_signal(self._engine._last_context_snapshot or {}),
                    },
                )
                logger.debug(
                    "evolution.record_episode outcome=%s actions=%d", outcome, len(actions)
                )
        except Exception:
            pass  # never fail the main loop
