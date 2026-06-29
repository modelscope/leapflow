"""Prediction Loop — the core Predict → Execute → Compare → Learn cycle.

Implements on-policy predictive coding: before each action execution,
the world model predicts the expected effect; after execution, it compares
the actual outcome against the prediction to compute a prediction error δ.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Optional, Tuple

if TYPE_CHECKING:
    from leapflow.world_model.budget import LearningBudgetController
    from leapflow.world_model.experience_store import ExperienceStore
    from leapflow.perception.state_snapshot import SnapshotFidelity, StateSnapshot, StateSnapshotService

from leapflow.llm.base import LLMProvider
from leapflow.llm.message_builder import build_system_message, build_user_message_text
from leapflow.world_model._json_utils import extract_json_object

logger = logging.getLogger(__name__)

_PREDICT_PROMPT = """\
Given the current state:
- App: {app_bundle_id} | Window: {window_title}
- Recent: {recent_events_summary}
{past_experience_section}
Action: {action_description}

Predict in one sentence: what state change will occur?
Rate your prediction confidence (0.0-1.0).
Output JSON: {{"effect": "...", "confidence": 0.8}}"""

_COMPARE_PROMPT = """\
Predicted effect: "{predicted}"
Actual changes: App={app}, AX structure changed={ax_changed}, Clipboard changed={clip_changed}
Observed changes: {recent_events_after}

Rate how different the actual outcome is from the prediction (0.0=identical, 1.0=completely different).
Output JSON: {{"distance": 0.3, "actual_effect": "one sentence"}}"""


@dataclass(frozen=True)
class Prediction:
    """A world-model prediction about an action's expected outcome."""

    action_description: str
    expected_effect: str
    confidence: float
    reasoning: str = ""


@dataclass(frozen=True)
class PredictionOutcome:
    """The result of comparing a prediction against observed reality."""

    prediction: Prediction
    pre_snapshot: Any  # StateSnapshot
    post_snapshot: Any  # StateSnapshot
    actual_effect: str
    delta: float
    delta_source: str  # "structural" | "semantic" | "blended"
    timestamp: float
    experience_id: str = ""


class PredictionLoop:
    """Core on-policy learning engine: Predict → Execute → Compare → Learn.

    Wraps action execution to transparently inject prediction and comparison.
    When disabled or budget-exhausted, passes through execution unchanged.

    Maintains a trajectory buffer for OPD trajectory-level teacher grading.
    """

    def __init__(
        self,
        llm: LLMProvider,
        snapshot_service: "StateSnapshotService",
        experience_store: "ExperienceStore",
        budget: "LearningBudgetController",
        *,
        enabled: bool = True,
        delta_threshold: float = 0.3,
        structural_blend_weight: float = 0.4,
        semantic_blend_weight: float = 0.6,
        semantic_compare_threshold: float = 0.1,
        rag_advantage_floor: float = -0.3,
        failure_advantage: float = -0.5,
        on_prediction_outcome: Optional[Callable[[PredictionOutcome], None]] = None,
    ) -> None:
        self._llm = llm
        self._snapshot = snapshot_service
        self._store = experience_store
        self._budget = budget
        self._enabled = enabled
        self._delta_threshold = delta_threshold
        self._structural_blend = structural_blend_weight
        self._semantic_blend = semantic_blend_weight
        self._semantic_threshold = semantic_compare_threshold
        self._rag_advantage_floor = rag_advantage_floor
        self._failure_advantage = failure_advantage
        self._on_outcome = on_prediction_outcome
        self._trajectory_buffer: list[dict] = []
        self._last_goal: str = ""
        self._pending_pre_snapshot: Any = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enabled = value

    @property
    def trajectory_buffer(self) -> list[dict]:
        """Read-only access to the current trajectory buffer."""
        return list(self._trajectory_buffer)

    def flush_trajectory(self) -> tuple[list[dict], str]:
        """Return and clear the trajectory buffer (for end-of-session grading).

        Returns (trajectory_steps, last_goal).
        """
        buf = list(self._trajectory_buffer)
        goal = self._last_goal
        self._trajectory_buffer.clear()
        self._last_goal = ""
        return buf, goal

    def set_goal(self, goal: str) -> None:
        """Record the current user goal for trajectory-level teacher grading."""
        self._last_goal = goal

    def record_failure(self, action_desc: str, error: str) -> None:
        """Record a failed action as a negative experience for future avoidance."""
        try:
            self._store.store(
                action_description=action_desc,
                app_context="",
                predicted_effect=f"execute successfully",
                actual_effect=f"FAILED: {error}",
                delta=1.0,
                advantage=self._failure_advantage,
                grade_label="harmful",
            )
        except Exception:
            logger.debug("record_failure failed", exc_info=True)

    async def wrap_execution(
        self,
        action_desc: str,
        execute_fn: Callable[..., Awaitable[Any]],
        *args: Any,
        fidelity: Optional["SnapshotFidelity"] = None,
        **kwargs: Any,
    ) -> Tuple[Any, Optional[PredictionOutcome]]:
        """Wrap an action execution with the prediction-comparison loop.

        Returns (execution_result, prediction_outcome_or_none).
        """
        if not self._enabled or not self._budget.has_tokens("prediction"):
            result = await execute_fn(*args, **kwargs)
            return result, None

        try:
            return await self._run_loop(action_desc, execute_fn, args, kwargs, fidelity)
        except Exception:
            logger.debug("prediction_loop.wrap_execution failed; executing raw", exc_info=True)
            result = await execute_fn(*args, **kwargs)
            return result, None

    async def _run_loop(
        self,
        action_desc: str,
        execute_fn: Callable[..., Awaitable[Any]],
        args: tuple,
        kwargs: dict,
        fidelity: Optional["SnapshotFidelity"],
    ) -> Tuple[Any, Optional[PredictionOutcome]]:
        from leapflow.perception.state_snapshot import SnapshotFidelity

        fid = fidelity or SnapshotFidelity.LIGHT

        # Phase 1: Capture pre-state
        pre = await self._snapshot.capture(fid)

        # Phase 2: Predict
        prediction = await self._predict(action_desc, pre)

        # Phase 3: Execute
        result = await execute_fn(*args, **kwargs)

        # Phase 4: Capture post-state
        post = await self._snapshot.capture(fid)

        # Phase 5: Compare
        outcome = await self._compare(prediction, pre, post)

        # Phase 6: Store experience
        exp_id = self._store.store(
            action_description=action_desc,
            app_context=pre.app_bundle_id,
            predicted_effect=prediction.expected_effect,
            actual_effect=outcome.actual_effect,
            delta=outcome.delta,
            pre_state_summary=pre.to_prompt_context(budget_tokens=100),
            post_state_summary=post.to_prompt_context(budget_tokens=100),
        )

        outcome = replace(outcome, experience_id=exp_id)

        # Phase 7: Accumulate trajectory for OPD grading
        self._trajectory_buffer.append({
            "experience_id": exp_id,
            "action_description": action_desc,
            "app_context": pre.app_bundle_id,
            "predicted_effect": prediction.expected_effect,
            "actual_effect": outcome.actual_effect,
            "delta": outcome.delta,
        })

        self._budget.spend("prediction")

        if self._on_outcome is not None:
            try:
                self._on_outcome(outcome)
            except Exception:
                logger.debug("on_prediction_outcome callback failed", exc_info=True)

        return result, outcome

    async def _predict(self, action_desc: str, pre: "StateSnapshot") -> Prediction:
        """Generate a prediction using LLM with retrieval-augmented context."""
        similar = self._store.retrieve_similar(
            action_desc, pre.app_bundle_id, limit=3,
            advantage_floor=self._rag_advantage_floor,
        )
        experience_section = ""
        if similar:
            lines = [
                f"- Past: {e.action_description} → {e.actual_effect} (δ={e.delta:.2f})"
                for e in similar
            ]
            experience_section = "Past experiences:\n" + "\n".join(lines)

        prompt = _PREDICT_PROMPT.format(
            app_bundle_id=pre.app_bundle_id,
            window_title=pre.window_title,
            recent_events_summary="; ".join(pre.recent_events[:3]) if pre.recent_events else "none",
            past_experience_section=experience_section,
            action_description=action_desc,
        )

        try:
            resp = await self._llm.achat(
                [build_system_message("You are a predictive world model."),
                 build_user_message_text(prompt)],
                stream=False, enable_thinking=False,
            )
            parsed = extract_json_object(resp.content or "")
            return Prediction(
                action_description=action_desc,
                expected_effect=str(parsed.get("effect", "")),
                confidence=float(parsed.get("confidence", 0.5)),
            )
        except Exception:
            logger.debug("prediction.predict failed; using null prediction", exc_info=True)
            return Prediction(
                action_description=action_desc,
                expected_effect="unknown",
                confidence=0.0,
            )

    async def _compare(
        self, prediction: Prediction, pre: "StateSnapshot", post: "StateSnapshot",
    ) -> PredictionOutcome:
        """Compare prediction against observed state change."""
        structural_delta = pre.semantic_distance(post)

        if structural_delta > self._semantic_threshold and self._budget.has_tokens("comparison"):
            semantic_delta = await self._semantic_compare(prediction, pre, post)
            delta = self._structural_blend * structural_delta + self._semantic_blend * semantic_delta
            source = "blended"
            self._budget.spend("comparison")
        else:
            delta = structural_delta
            source = "structural"

        actual_effect = self._describe_change(pre, post)

        return PredictionOutcome(
            prediction=prediction,
            pre_snapshot=pre,
            post_snapshot=post,
            actual_effect=actual_effect,
            delta=delta,
            delta_source=source,
            timestamp=time.time(),
        )

    async def _semantic_compare(
        self, prediction: Prediction, pre: "StateSnapshot", post: "StateSnapshot",
    ) -> float:
        """Use LLM to semantically compare predicted vs actual outcome."""
        prompt = _COMPARE_PROMPT.format(
            predicted=prediction.expected_effect,
            app=post.app_bundle_id,
            ax_changed=pre.ax_digest != post.ax_digest,
            clip_changed=pre.clipboard_text != post.clipboard_text,
            recent_events_after="; ".join(post.recent_events[:3]) if post.recent_events else "none",
        )
        try:
            resp = await self._llm.achat(
                [build_system_message("You are a precise outcome comparator."),
                 build_user_message_text(prompt)],
                stream=False, enable_thinking=False,
            )
            parsed = extract_json_object(resp.content or "")
            return float(parsed.get("distance", 0.5))
        except Exception:
            return 0.5

    @staticmethod
    def _describe_change(pre: "StateSnapshot", post: "StateSnapshot") -> str:
        """Build a structural description of observed changes."""
        changes: list[str] = []
        if pre.app_bundle_id != post.app_bundle_id:
            changes.append(f"app changed to {post.app_bundle_id}")
        if pre.window_title != post.window_title:
            changes.append(f"window changed to '{post.window_title}'")
        if pre.ax_digest != post.ax_digest:
            changes.append("UI structure changed")
        if pre.clipboard_text != post.clipboard_text:
            changes.append("clipboard changed")
        return "; ".join(changes) if changes else "no observable change"

    def create_from_react_prediction(
        self, action_desc: str, predicted_effect: str, confidence: float = 0.5,
    ) -> Prediction:
        """Create a Prediction from ReAct loop's inline predicted_effect field."""
        return Prediction(
            action_description=action_desc,
            expected_effect=predicted_effect,
            confidence=confidence,
        )

    async def verify_prediction(
        self,
        prediction: Prediction,
        *,
        fidelity: Optional["SnapshotFidelity"] = None,
    ) -> Optional[PredictionOutcome]:
        """Verify a pre-existing prediction against the current environment state.

        Use after an action has already been executed (e.g. ReAct bridge calls)
        to compare the predicted effect against reality and store the experience.
        Requires a prior ``capture_pre_snapshot()`` call to have a baseline.
        """
        pre = self._pending_pre_snapshot
        if pre is None or not self._enabled:
            return None
        if not self._budget.has_tokens("prediction"):
            return None

        from leapflow.perception.state_snapshot import SnapshotFidelity as Fid
        fid = fidelity or Fid.LIGHT

        try:
            post = await self._snapshot.capture(fid)
            outcome = await self._compare(prediction, pre, post)
            exp_id = self._store.store(
                action_description=prediction.action_description,
                app_context=pre.app_bundle_id,
                predicted_effect=prediction.expected_effect,
                actual_effect=outcome.actual_effect,
                delta=outcome.delta,
                pre_state_summary=pre.to_prompt_context(budget_tokens=100),
                post_state_summary=post.to_prompt_context(budget_tokens=100),
            )
            outcome = replace(outcome, experience_id=exp_id)
            self._trajectory_buffer.append({
                "experience_id": exp_id,
                "action_description": prediction.action_description,
                "app_context": pre.app_bundle_id,
                "predicted_effect": prediction.expected_effect,
                "actual_effect": outcome.actual_effect,
                "delta": outcome.delta,
            })
            self._budget.spend("prediction")
            if self._on_outcome is not None:
                try:
                    self._on_outcome(outcome)
                except Exception:
                    logger.debug("on_prediction_outcome callback failed", exc_info=True)
            return outcome
        except Exception:
            logger.debug("verify_prediction failed", exc_info=True)
            return None
        finally:
            self._pending_pre_snapshot = None

    async def capture_pre_snapshot(
        self, fidelity: Optional["SnapshotFidelity"] = None,
    ) -> None:
        """Capture a pre-execution snapshot for later ``verify_prediction()``."""
        from leapflow.perception.state_snapshot import SnapshotFidelity as Fid
        fid = fidelity or Fid.LIGHT
        try:
            self._pending_pre_snapshot = await self._snapshot.capture(fid)
        except Exception:
            logger.debug("capture_pre_snapshot failed", exc_info=True)
            self._pending_pre_snapshot = None
