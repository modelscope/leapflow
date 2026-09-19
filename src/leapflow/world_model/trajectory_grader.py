# Copyright (c) Alibaba, Inc. and its affiliates.
"""Trajectory Grader — OPD teacher role for train-free agentic learning.

Implements the "teacher-as-reward-model" pattern from On-Policy Distillation:
the LLM, given *full hindsight context* (goal + trajectory + outcomes), grades
each action step with an advantage signal and identifies forking actions where
the chosen path diverges materially from the optimal one.

Teacher/student asymmetry comes from *information context*, not model capability:
  - Teacher sees the complete trajectory including final outcomes.
  - Student (prediction loop) acts with only the current state visible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Mapping, Sequence

if TYPE_CHECKING:
    from leapflow.world_model.budget import LearningBudgetController
    from leapflow.world_model.experience_store import ExperienceStore

from leapflow.domain.adaptation_verdict import AdaptationVerdict
from leapflow.domain.evolution_intent import EvolutionIntent, is_capability_name
from leapflow.llm.base import LLMProvider
from leapflow.llm.message_builder import build_system_message, build_user_message_text
from leapflow.world_model._json_utils import extract_json_object

logger = logging.getLogger(__name__)

DEFAULT_GRADE_LABELS: tuple[str, ...] = (
    "optimal", "acceptable", "suboptimal", "harmful",
)

_GRADE_PROMPT = """\
You are evaluating an agent's action trajectory with FULL hindsight.

Goal: {goal}

Trajectory (chronological order):
{trajectory_text}

For each step, provide:
- advantage: float in [-1, 1]. Positive = better than average, negative = harmful.
- is_forking: true if a meaningfully different action here would have led to a \
very different outcome. These are critical decision points.
- grade_label: one of {grade_labels}.

Output JSON:
{{"grades": [{{"step": 1, "advantage": 0.3, "is_forking": false, \
"grade_label": "{example_label}"}}, ...]}}"""

# Appended when the teacher is also asked to propose capability gaps. Kept in the
# *same* call as grading so a proposal costs no additional budget token: the
# hindsight context needed to grade is exactly the context needed to notice that a
# capability is missing.
#
# The teacher is deliberately not asked for a risk level. An intent is a
# hypothesis, not an authorisation; the risk ceiling is imposed by the trusted
# caller (see ``EvolutionIntent`` / ``MODEL_AUTHORED_RISK_CEILING``).

#: How many declared capability names to show the teacher. Bounded so a large
#: registry cannot crowd out the trajectory it is supposed to be grading.
_MAX_DECLARED_SHOWN = 60


def _declared_capability_section() -> str:
    """The capability names tools already declare, for the teacher to reuse.

    Shown, never enforced. A teacher constrained to this list could no longer report a
    genuinely *missing* capability, which is the main thing it is asked for. Shown so
    that when the capability does exist under a name the teacher would not have
    guessed, it names the existing one -- otherwise the same ability accumulates a
    second name, and a capability with two names has one provider each instead of two
    competing providers for one name.

    Read live from the registry rather than from a table: a third-party or generated
    plugin's declarations must appear too, and a hardcoded list would go stale the
    moment the tool set changed.
    """
    try:
        from leapflow.plugins import get_registry

        registry = get_registry()
        names = sorted(
            {
                capability
                for plugin in registry.plugins.values()
                for tool in plugin.tools
                for capability in (tool.provides_capabilities or ())
                if capability
            }
        )
    except Exception:  # noqa: BLE001 - no catalog degrades the hint, not the grading
        logger.debug("teacher: declared capability catalog unavailable", exc_info=True)
        return ""
    if not names:
        return ""
    shown = names[:_MAX_DECLARED_SHOWN]
    more = f" (and {len(names) - len(shown)} more)" if len(names) > len(shown) else ""
    return (
        "\nCapability names already declared by existing tools"
        + more
        + ". If the ability you\nare reporting is one of these, use that exact name; only invent a new name when\nnone of these is the ability in question:\n"
        + ", ".join(shown)
        + "\n"
        # Naming the list is not the same as saying any of it fits. Measured: asked about
        # a chat app failure while shown this catalogue, a real model answered `rebind`
        # on 3 of 3 trials -- pointing at a neighbour that shares no environment with the
        # failure, on a unit whose candidate set had exactly one entry. A `rebind` whose
        # target cannot serve the environment is worse than no recommendation: it leaves
        # the failure in place and puts a misleading "Prefer X" in front of every later
        # turn. So the list is scoped to what it is for.
        + "This list is for *naming*. It does not mean any of these can serve the failing\n"
        "capability -- most require an environment that is not present. Only answer\n"
        "`rebind` when a capability here is genuinely a provider for the same ability in\n"
        "the same environment, and name it in `target`. If you cannot point to one,\n"
        "`rebind` is the wrong action.\n"
    )


def _offered_providers(
    degraded: Sequence[Mapping[str, Any]] = (),
) -> frozenset[str]:
    """Every provider named to the teacher as an alternative.

    A target it picked from a list we supplied has to be acceptable, whatever the registry
    currently holds. The registry check stays as a second route -- it catches a target the
    teacher invented rather than selected.
    """
    names: set[str] = set()
    for fact in degraded or ():
        for row in fact.get("alternatives") or ():
            for key in ("tool_name", "plugin_id"):
                value = str(row.get(key) or "").strip()
                if value:
                    names.add(value)
    return frozenset(names)


def _alternatives_line(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render the other providers of a capability, and whether each can run here.

    Stated as an absence when there are none, because "no alternative exists" is the
    positive evidence for ``acquire`` -- and a silent omission would read as "not
    checked", which is exactly the ambiguity that produced a guess.
    """
    if not rows:
        return "\n  no other installed provider offers this capability"
    usable = [r for r in rows if r.get("fits_here")]
    parts = [
        f"{r.get('tool_name') or r.get('plugin_id')}"
        + ("" if r.get("fits_here") else f" (needs {', '.join(r.get('requires') or ()) or 'unmet affordances'})")
        for r in rows
    ]
    verdict = (
        "one of these could take over"
        if usable
        else "none of these can run in this environment"
    )
    return f"\n  other providers: {'; '.join(parts)} -- {verdict}"


def _degraded_capability_section(degraded: Sequence[Mapping[str, Any]]) -> str:
    """Capabilities whose current provider has been failing, for the teacher to judge.

    Facts only, and deliberately without a verdict attached. A consecutive-failure
    count cannot distinguish a badly written implementation from an environment that
    moved underneath a correct one -- both produce the same streak and they want
    opposite actions, rebuild versus rebind. The teacher has the trajectory and
    hindsight, so it is the component that can tell them apart; passing it a threshold
    decision would replace that judgement with a counter.

    Empty when nothing is degraded, so the prompt gains nothing on a healthy session.
    """
    rows = [
        (
            str(item.get("capability") or "").strip(),
            str(item.get("plugin_id") or "").strip(),
            int(item.get("failure_streak") or 0),
            str(item.get("failure_class") or "").strip(),
        )
        for item in degraded or ()
    ]
    rows = [row for row in rows if row[0]]
    if not rows:
        return ""
    alternatives = {
        str(item.get("capability") or ""): tuple(item.get("alternatives") or ())
        for item in degraded or ()
    }
    prior = {
        str(item.get("capability") or ""): (
            str(item.get("prior_action") or ""),
            str(item.get("prior_knowledge") or ""),
        )
        for item in degraded or ()
        if item.get("prior_knowledge")
    }
    lines = "\n".join(
        f"- {capability}: current provider {plugin or '(unknown)'} has "
        f"{streak} consecutive failure(s) and is still serving"
        + (f", failing as {failure_class}" if failure_class else "")
        # What was concluded last time. Stated as history rather than as a verdict on
        # the verdict: knowledge outliving the failure it describes is evidence the
        # previous adaptation did not resolve it, not proof the judgement was wrong.
        + (
            f"\n  (last time you answered '{prior[capability][0]}' and recorded: "
            f"{prior[capability][1]})"
            if capability in prior
            else ""
        )
        # The fact `rebind` and `acquire` are *defined* by. Without it the choice between
        # them is a guess, which is what a real model was measured doing.
        + _alternatives_line(alternatives.get(capability, ()))
        for capability, plugin, streak, failure_class in sorted(rows)
    )
    # Named when every degradation shares one environment, because then the right
    # answer is usually one rebind rather than one rebuild per capability.
    shared = {
        str((item.get("environment") or {}).get("fingerprint_id") or "")
        for item in degraded or ()
    }
    # Only meaningful for two or more: telling the teacher that one degradation "may be
    # one change rather than several" is noise that reads as a hint it must reconcile.
    common = (
        "\nAll of these were seen in the same environment, so they may be one change "
        "rather than several.\n"
        if len(rows) > 1 and len(shared) == 1 and next(iter(shared))
        else ""
    )
    return (
        "\nCapabilities whose existing provider has been failing while still in\n"
        "service. A capability appearing here already exists, so the question is not\n"
        "whether the ability is absent -- it is which of the four actions the evidence\n"
        "supports. Prefer the cheapest that fits: absorb costs nothing, rebind reuses\n"
        "what is installed, and acquire replaces a working-but-wrong implementation at\n"
        "the price of new code.\n" + common + lines + "\n"
    )


_GAP_PROMPT_SECTION = """

Additionally, judge what this episode's evidence says the system should *do* about
the environment it ran in. You are being asked for an action, not for blame: when an
application upgrades, the existing implementation was not written wrongly -- it was
right for the old version -- and yet a new adapter may still be the only way forward.
"Whose fault is it" and "what should be done" are different questions.

Choose one action per capability, from these four only, cheapest first:
- absorb:   the retry or semantic-addressing layer already handles this. A label moved,
            an element was renamed, a call timed out. The capability set does not change.
            This is the correct answer most of the time.
- rebind:   another installed capability already covers the new environment. Name it in
            `target`. Prefer this over acquire whenever anything already declared fits.
- acquire:  nothing installed covers this, so a new implementation is warranted. This is
            the ONLY action that causes code to be written, so use it last.
- escalate: this needs a person -- a missing permission or credential, or a decision the
            agent must not make for itself. Put what the human has to do in `target`.

Every verdict MUST carry `knowledge`: one or two sentences stating what is now true
about the environment, written for the agent that will act next. It is read as ordinary
context, so write a statement about the world ("the send control is now labelled
Dispatch and lives in the toolbar"), never an instruction to the framework ("regenerate
the plugin"). This field is the point of the exercise: three of the four actions change
nothing except what the acting agent knows.

Two rules that override everything above:
- Do NOT restate the task, the goal, or the episode name as a capability. A capability
  is a reusable ability such as "chat.reply", never a description of this attempt.
- Report nothing at all if the episode's failure was simply a wrong choice among
  actions that were available and working. An invented verdict is worse than a missed
  one, because acquire builds code and rebind redirects traffic.

For each verdict provide:
- action: one of absorb | rebind | acquire | escalate.
- capability: a stable dotted capability name (e.g. "chat.reply").
- knowledge: what is now true about the environment. Required.
- rationale: why this action rather than a cheaper one, in one sentence.
- confidence: float in [0, 1].
- target: for rebind, the capability or tool to use instead; for escalate, what the
  human must do; omit otherwise.
- target_affordance: for acquire, the environment affordance a new adapter should
  target, if visible.
- expected_effect: for acquire, what should observably happen once it exists.
{declared_section}{degraded_section}
Add to the JSON:
{{"adaptation_verdicts": [{{"action": "rebind", "capability": "...", \
"knowledge": "...", "rationale": "...", "confidence": 0.7, "target": "...", \
"target_affordance": "...", "expected_effect": "..."}}, ...]}}
Use an empty list when the episode warrants no adaptation."""

def _is_declared_capability(name: str) -> bool:
    """Whether some live tool declares this capability or answers to this tool name.

    Reads the registry, not a list, so a generated or third-party plugin counts. Accepts a
    tool name as well as a capability name because a teacher naming a concrete provider is
    being *more* specific than asked, and rejecting that would push it toward the vaguer
    answer.

    An unavailable registry returns ``True``: the guard exists to catch a target that is
    demonstrably absent, and failing closed here would silently discard every rebind in
    any process that composes no registry.
    """
    candidate = str(name or "").strip()
    if not candidate:
        return False
    try:
        from leapflow.plugins import get_registry

        registry = get_registry()
        for plugin in registry.plugins.values():
            for tool in plugin.tools:
                if tool.name == candidate:
                    return True
                if candidate in (tool.provides_capabilities or ()):
                    return True
    except Exception:  # noqa: BLE001 - no registry cannot mean no valid rebind
        logger.debug("teacher: cannot verify rebind target", exc_info=True)
        return True
    return False


def _echoes_goal(capability: str, goal: str) -> bool:
    """Whether the capability is just the goal (or episode name) restated.

    A model handed a goal string will sometimes hand it straight back as the capability.
    Compared on alphanumerics only, so separator and case differences do not let an echo
    through.
    """
    def norm(value: str) -> str:
        return "".join(ch for ch in str(value).lower() if ch.isalnum())

    normalised_goal = norm(goal)
    return bool(normalised_goal) and norm(capability) == normalised_goal


@dataclass(frozen=True)
class ActionGrade:
    """Teacher's grading of a single action step."""

    experience_id: str
    advantage: float
    is_forking: bool
    grade_label: str


@dataclass(frozen=True)
class TeacherVerdict:
    """Everything one hindsight evaluation produced.

    ``grades`` distil into the experience store as advantage signal; ``verdicts`` say
    what the episode's evidence warrants doing about the environment. Both come from a
    single LLM call, so a verdict costs one ``grading`` budget token.

    ``intents`` is *derived* from the ``acquire`` verdicts rather than parsed
    separately. One source of truth: an ``EvolutionIntent`` can only exist because a
    verdict asked for code to be written, so a recommendation to *rebind* can never
    silently queue an acquisition.
    """

    grades: tuple[ActionGrade, ...] = ()
    verdicts: tuple[AdaptationVerdict, ...] = ()
    raw_payload: Mapping[str, Any] = field(default_factory=dict)

    @property
    def intents(self) -> tuple[EvolutionIntent, ...]:
        """The acquisition intents, one per ``acquire`` verdict."""
        derived = (verdict.to_intent() for verdict in self.verdicts)
        return tuple(intent for intent in derived if intent is not None)

    def by_action(self, action: str) -> tuple[AdaptationVerdict, ...]:
        """Verdicts asking for one particular action."""
        return tuple(v for v in self.verdicts if v.action == action)


class TrajectoryGrader:
    """Grades completed trajectories from a teacher perspective (full hindsight).

    After a session's prediction loop accumulates a trajectory buffer, this
    component performs a single LLM call to grade all steps, then writes the
    advantage/forking signals back into the ExperienceStore.
    """

    def __init__(
        self,
        llm: LLMProvider,
        experience_store: "ExperienceStore",
        budget: "LearningBudgetController",
        *,
        min_trajectory_length: int = 3,
        grade_labels: tuple[str, ...] = DEFAULT_GRADE_LABELS,
    ) -> None:
        self._llm = llm
        self._store = experience_store
        self._budget = budget
        self._min_len = min_trajectory_length
        self._grade_labels = grade_labels

    async def grade_trajectory(
        self,
        trajectory: List[dict],
        goal: str = "",
    ) -> List[ActionGrade]:
        """Grade a completed trajectory and persist advantage signals.

        Each element of *trajectory* must contain at minimum:
            experience_id, action_description, predicted_effect,
            actual_effect, delta
        """
        if len(trajectory) < self._min_len:
            return []
        if not self._budget.has_tokens("grading"):
            return []

        traj_text = self._format_trajectory(trajectory)
        payload = await self._call_teacher_raw(traj_text, goal, propose_gaps=False)
        self._budget.spend("grading")

        grades = self._persist_grades(trajectory, self._parse_grades(payload))
        return grades

    async def grade_and_propose(
        self,
        trajectory: List[dict],
        goal: str = "",
        *,
        degraded_capabilities: Sequence[Mapping[str, Any]] = (),
        raise_on_error: bool = False,
    ) -> "TeacherVerdict":
        """Grade the trajectory *and* propose capability gaps, in one LLM call.

        This is the teacher's full verdict: the advantage signal that distils into
        experience, plus any :class:`EvolutionIntent` describing a capability the
        agent lacked. Both come from the same hindsight context and the same
        single ``grading`` budget token, so proposing costs nothing beyond grading.

        The returned intents are hypotheses. They carry no authorisation and must
        still traverse the deterministic chain (declared-fitness resolution, risk
        classification, approval, artifact validation, trust) before anything is
        acquired.
        """
        if len(trajectory) < self._min_len:
            return TeacherVerdict((), ())
        if not self._budget.has_tokens("grading"):
            return TeacherVerdict((), ())

        traj_text = self._format_trajectory(trajectory)
        payload = await self._call_teacher_raw(
            traj_text,
            goal,
            propose_gaps=True,
            degraded_capabilities=degraded_capabilities,
            raise_on_error=raise_on_error,
        )
        self._budget.spend("grading")

        grades = self._persist_grades(trajectory, self._parse_grades(payload))
        return TeacherVerdict(
            tuple(grades),
            self._parse_verdicts(payload, goal, degraded_capabilities),
            dict(payload),
        )

    async def _call_teacher_raw(
        self,
        trajectory_text: str,
        goal: str,
        *,
        propose_gaps: bool = False,
        degraded_capabilities: Sequence[Mapping[str, Any]] = (),
        raise_on_error: bool = False,
    ) -> dict:
        """Single LLM call: teacher evaluates with full hindsight.

        Returns the parsed JSON object so grades and capability gaps can both be
        derived from one response. A failed or unparseable call yields ``{}`` --
        the teacher is advisory, so it must never fail the caller.
        """
        labels_str = ", ".join(f'"{label}"' for label in self._grade_labels)
        prompt = _GRADE_PROMPT.format(
            goal=goal or "(not specified)",
            trajectory_text=trajectory_text,
            grade_labels=labels_str,
            example_label=self._grade_labels[1] if len(self._grade_labels) > 1 else self._grade_labels[0],
        )
        if propose_gaps:
            prompt += _GAP_PROMPT_SECTION.format(
                declared_section=_declared_capability_section(),
                degraded_section=_degraded_capability_section(degraded_capabilities),
            )
        try:
            resp = await self._llm.achat(
                [build_system_message(
                    "You are a trajectory evaluator with perfect hindsight.",
                ),
                 build_user_message_text(prompt)],
                stream=False, enable_thinking=False,
            )
            return extract_json_object(resp.content or "") or {}
        except Exception:
            if raise_on_error:
                raise
            logger.debug("trajectory_grader.call_teacher failed", exc_info=True)
            return {}

    def _format_trajectory(self, trajectory: List[dict]) -> str:
        """Render trajectory steps into a numbered text block."""
        lines: list[str] = []
        for i, step in enumerate(trajectory, 1):
            lines.append(
                f"Step {i}: action={step.get('action_description', '?')}\n"
                f"  predicted: {step.get('predicted_effect', '?')}\n"
                f"  actual: {step.get('actual_effect', '?')}\n"
                f"  delta: {step.get('delta', '?')}"
            )
        return "\n".join(lines)

    def _parse_grades(self, payload: dict) -> List[ActionGrade]:
        """Parse the teacher payload into ActionGrade objects."""
        raw_grades = payload.get("grades", []) if isinstance(payload, dict) else []
        results: List[ActionGrade] = []
        for raw in raw_grades:
            if not isinstance(raw, dict):
                continue
            try:
                advantage = float(raw.get("advantage", 0))
            except (TypeError, ValueError):
                advantage = 0.0
            results.append(ActionGrade(
                experience_id="",
                advantage=max(-1.0, min(1.0, advantage)),
                is_forking=bool(raw.get("is_forking", False)),
                grade_label=str(raw.get("grade_label", "acceptable")),
            ))
        return results

    def _parse_verdicts(
        self,
        payload: dict,
        goal: str = "",
        degraded_capabilities: Sequence[Mapping[str, Any]] = (),
    ) -> tuple[AdaptationVerdict, ...]:
        """Parse declared adaptation verdicts, skipping the ones that cannot be acted on.

        A verdict without a declared capability, a recognised action, or knowledge is
        discarded: the capability name must be declared, never inferred from prose, and a
        verdict that teaches the acting agent nothing leaves the teacher with no effect
        even when its judgement was right.

        Two further rejections exist because a live model was measured doing exactly
        this. Asked to diagnose an episode that failed for a *non-capability* reason,
        ``qwen3.7-plus`` returned the episode's own name as the capability on 3 of 3
        trials. Nothing downstream would have caught it -- the name is well-formed, so
        it would have become a requirement and the governed pipeline would have
        faithfully tried to build ``chat.cosmetic.example``.

        So a capability must *look* like a capability, and must not be a restatement of
        the goal. Neither check can catch a plausible-but-wrong verdict; that is what
        validation, effect verification and quarantine are for. These catch the
        degenerate case, which is the one that produces pure noise.
        """
        raw_verdicts = (
            payload.get("adaptation_verdicts", []) if isinstance(payload, dict) else []
        )
        # Providers we ourselves named as alternatives. Rejecting a target we offered would
        # be incoherent, and it silently was: the guard consulted only the live registry,
        # so in any process whose alternatives come from somewhere else -- a replay, a
        # study, a registry that has not caught up -- every legitimate rebind was
        # discarded. Measured as 3/3 silence on a rebind unit across two unrelated corpus
        # designs, which read as "the model will not answer rebind" and was in fact "we
        # threw the answer away".
        offered = _offered_providers(degraded_capabilities)
        verdicts: List[AdaptationVerdict] = []
        for raw in raw_verdicts:
            if not isinstance(raw, dict):
                continue
            capability = str(raw.get("capability") or "").strip()
            if not is_capability_name(capability):
                logger.debug(
                    "trajectory_grader: rejected non-capability name %r", capability
                )
                continue
            if _echoes_goal(capability, goal):
                logger.debug(
                    "trajectory_grader: rejected goal restatement %r", capability
                )
                continue
            try:
                confidence = float(raw.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            action = str(raw.get("action") or "").strip().lower()
            target = str(raw.get("target") or "").strip()
            if action == "rebind" and not (
                target in offered or _is_declared_capability(target)
            ):
                # A rebind that cannot point at something real is not a cheaper answer,
                # it is a dead end wearing one: the failure stays and the student is told
                # to "Prefer" a provider that does not exist. Measured on a real model at
                # 3 of 3 trials, pointing at a neighbour from the naming catalogue.
                logger.debug(
                    "trajectory_grader: rejected rebind to unknown target %r", target
                )
                continue
            try:
                verdicts.append(
                    AdaptationVerdict.create(
                        str(raw.get("action") or ""),
                        capability,
                        str(raw.get("knowledge") or ""),
                        rationale=str(raw.get("rationale") or ""),
                        confidence=confidence,
                        target=str(raw.get("target") or ""),
                        target_affordance=str(raw.get("target_affordance") or ""),
                        expected_effect=str(raw.get("expected_effect") or ""),
                    )
                )
            except ValueError:
                logger.debug(
                    "trajectory_grader: discarded unusable adaptation verdict %r", raw
                )
        return tuple(verdicts)

    def _persist_grades(
        self,
        trajectory: List[dict],
        grades: List[ActionGrade],
    ) -> List[ActionGrade]:
        """Write grading results back to ExperienceStore.

        Returns grades with populated experience_id fields.
        """
        if len(grades) != len(trajectory):
            logger.debug(
                "trajectory_grader: grade count (%d) != trajectory length (%d)",
                len(grades), len(trajectory),
            )

        populated: List[ActionGrade] = []
        for i, grade in enumerate(grades):
            if i >= len(trajectory):
                break
            exp_id = trajectory[i].get("experience_id", "")
            bound = ActionGrade(
                experience_id=exp_id,
                advantage=grade.advantage,
                is_forking=grade.is_forking,
                grade_label=grade.grade_label,
            )
            populated.append(bound)
            if not exp_id:
                continue
            self._store.update_advantage(
                experience_id=exp_id,
                advantage=grade.advantage,
                is_forking=grade.is_forking,
                grade_label=grade.grade_label,
            )
        return populated

    @staticmethod
    def trajectory_from_outcomes(
        outcomes: list,
    ) -> List[dict]:
        """Convert a list of PredictionOutcome objects to trajectory dicts."""
        result: List[dict] = []
        for o in outcomes:
            result.append({
                "experience_id": getattr(o, "experience_id", ""),
                "action_description": o.prediction.action_description,
                "predicted_effect": o.prediction.expected_effect,
                "actual_effect": o.actual_effect,
                "delta": o.delta,
            })
        return result
