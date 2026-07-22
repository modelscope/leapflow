# Adaptive-Depth Execution: Toward an Infinite OODA Loop

> A design-methodology document. It explains the *why*, the *how*, and the *when* of
> LeapFlow's adaptive-depth execution architecture. It is deliberately light on code:
> the goal is to convey the reasoning, the staged construction, and the usage model —
> not a line-by-line map.

## 1. Problem Statement

A capable agent must handle tasks whose intrinsic difficulty varies by orders of
magnitude — from a one-shot factual answer to a multi-hour investigation that spans
many tool calls, dead ends, and revisions. Yet most agent loops are built around a
*fixed* control budget: a constant iteration cap, a static context-disclosure policy,
and a single start→finish horizon. This mismatch produces two failure modes.

**Under-provisioning.** A hard task is cut off prematurely because the loop exhausts a
budget calibrated for the average case. The agent "gives up" while still making progress.

**Over-provisioning.** A trivial task carries the full apparatus — maximal tool
disclosure, aggressive context retention, many speculative iterations — inflating cost
and latency for no benefit.

A deeper limitation is *temporal*: the classical loop ends when the turn ends. It cannot
maintain orientation across sessions, resume when the environment changes, or act
proactively under governance. Real work is rarely a single bounded turn; it is an
ongoing engagement with a changing world.

The question this architecture answers is: **how can a single agent loop adapt its depth
to each task's difficulty, persist and refine its orientation over time, and — under
strict governance — extend into continuous, proactive operation, without ever becoming
unbounded or unsafe?**

## 2. Methodology

Five principles organize the design.

**Signal-driven, not rule-driven.** Depth, posture, and commitment are derived from
*observed signals* (difficulty estimates, effective token cost, tool-evidence
saturation), not from hardcoded keyword rules. Behavior that cannot be grounded in a
signal is out of scope.

**Boundedness by composition.** "Infinite" capability is achieved by *composing bounded
frames*, never by removing bounds from a single frame. Every unit of work — a turn, a
recursive subagent, a re-entry — is a frame with its own budget, deadline, and cost
ceiling. Unboundedness is an emergent property of chaining and nesting bounded frames,
which keeps every point in the system individually accountable.

**Progressive trust.** Autonomy is *earned*, never assumed. A proactive action is
auto-approved only after repeated human approvals of similar actions have raised the
relevant scope's trust; otherwise it falls back to explicit approval. Destructive or
first-time actions are never implicit.

**OODA as the organizing lens.** The loop is read as Observe → Orient → Decide → Act.
The design consistently invests in **Orient** — persistent findings, layered
orientation, learned calibration — because in OODA the quality of orientation cascades
into every downstream decision (see `ooda_framework.md`).

**Default-off, zero-regression.** Every new capability is gated behind configuration and
defaults to off, byte-equivalent to prior behavior. Adoption is a deliberate, reversible
choice, and each increment is independently verifiable.

These principles are realized as a **staged evolution S0 → S4**, where each stage adds a
capability while preserving the invariants of the ones below it.

## 3. S0 — The Adaptive-Depth Frame

The base stage makes a *single turn* elastic. Four coupled mechanisms:

- **Difficulty as a first-class signal.** Each turn continuously estimates task
  difficulty from context-governance evidence (tool-call breadth, evidence sources,
  convergence). Difficulty drives an **elastic iteration budget** whose cap widens from a
  safe floor toward a ceiling in proportion to observed hardness — a hard task earns a
  wider horizon; a simple one stays near the floor and self-stops.

- **Posture.** The turn adopts a research / expanding / finalizing posture, adjusting how
  much context and tooling it discloses. Posture is a signal-driven gradient, not a
  one-way ratchet.

- **Prefix commitment and cacheable stability.** Once a turn commits to a stable working
  prefix, that prefix (system instructions, tool schema, task contract) is held
  byte-stable across rounds so that provider prefix-caches are reused; volatile content
  (live signals, the research ledger) is appended at the tail, never woven into the
  cacheable prefix. Context compression is *append-only*: each historical window is
  summarized once and then frozen, which both preserves long-task state (signal-to-noise
  first) and keeps the frozen region cache-stable.

- **The research ledger.** A compact, durable record of findings, open questions,
  decisions, and the next step accompanies the turn. It is the turn's working memory of
  *intent and progress*, resistant to compression drift, and it supplies a reliable
  sufficiency signal: a task with tracked open questions is never cut short by premature
  convergence.

Finally, S0 makes the loop **recursive**: a subagent runs the *same* adaptive loop on an
isolated child frame with its own fresh budget and subsystems. Recursion is depth-gated
and state-isolated, so a subagent can decompose a hard problem without contaminating the
parent's orientation.

## 4. S1–S2 — Persistent Orientation and Event-Driven Re-entry

S1 lifts orientation beyond a single turn: the research ledger is persisted across
sessions, so a long-running task's accumulated understanding survives restarts and
resumes where it left off.

S2 breaks the start→finish horizon. A turn may register a **re-entry trigger** — a saved
orientation snapshot plus a firing condition (a delay, or an inbound environment event).
Later, that trigger fires *at most once* and seeds a fresh, isolated run from the saved
orientation. Crucially, re-entry is not a suspended coroutine held in memory; it is a
*finalize-then-reseed* pattern, which keeps the mechanism robust and the running system
uncontaminated. Inbound platform events enter as structured signals, are filtered and
classified, and only then may drive a governed re-entry — extending the agent's Observe
boundary into the collaboration environment.

## 5. S3 — Learning Closure

Orientation should improve with use. S3 closes a learning loop over the difficulty and
threshold machinery: each turn's *predicted* difficulty and posture are recorded
alongside its *actual* effort and outcome; offline analysis relates the two and proposes
a bounded adjustment to the difficulty-sensitivity weight and the finalize threshold;
and — when explicitly enabled — that adjustment is applied online, always derived from
the configured baseline (so it never compounds or drifts) and always reversible. The
difficulty signal thus migrates from hand-tuned toward learned, and the quality of
orientation rises monotonically with experience.

## 6. Governed Proactive Action

The most delicate capability is *acting outward* on the agent's own initiative — for
example, replying to the chat that originated a task once a background re-entry has
produced a result. The design refuses ungoverned autonomy. A proactive send passes a
pure decision kernel that combines a **send-scope trust ledger** (the progressive-trust
gradient), rate limits, idempotency, and a global budget. The verdict is one of:
auto-allow (only for non-destructive actions in a scope that has earned trust), queue for
asynchronous human approval (which, on approval, also accrues trust), or deny. Absent a
reachable approver, the default is to *not act*. External side effects are never silent.

## 7. D1 → S4 — Layered Orientation and the Infinite Loop

S4 is the north star: a **resident** agent that runs a continuous, resource-governed OODA
*tempo* — observing signals, maintaining a layered orientation, expanding bounded
subframes on demand, deciding through the trust-gated guidance described above, and
learning continuously — **with no hard horizon, only a governed cadence**.

The first, observe-only step of S4 (D1) is a **multi-layer orientation** query that
unifies three layers with time decay: *immediate* (live signals), *working* (the current
task ledger), and *long-term* (durable cross-session findings and retrieved memory).
Recent salience dominates while durable facts persist quietly. This makes Orient a
first-class, inspectable object — a prerequisite for any autonomous decision, and useful
on its own for diagnosis.

The remaining synthesis (a general implicit-guidance gate, a tempo governor with
backpressure, and the resident loop itself) is *designed but intentionally not enabled*:
a continuously autonomous, outward-acting loop is the highest-risk capability in the
system and is gated behind explicit authorization and review.

## 8. Safety and Governance

The architecture's safety rests on a small set of invariants that hold at every scale:

- **Bounded frames everywhere.** Cost ceilings, budgets, and deadlines apply to each
  turn, each recursive subframe, and each re-entry point.
- **Default-off and reversible.** Autonomy, re-entry, outbound delivery, online
  calibration, and full-loop subagents each require an explicit opt-in; disabled, the
  system is byte-equivalent to its prior behavior.
- **Approval, redaction, and audit on every act.** Outbound and other side-effecting
  actions flow through the existing approval, redaction, and audit paths; proactive
  action additionally requires progressive trust.
- **Isolation.** Recursive subagents run on fresh state with their own session, and never
  pollute the parent's learning or conversation.

## 9. Usage and Scenarios

By default the agent already benefits from S0: hard tasks transparently earn more depth
and a research posture; simple tasks stay short. No configuration is required, and the
current orientation can be inspected at any time through a read-only orientation view.

The remaining capabilities are opt-in and best adopted one at a time:

- **Long, multi-session investigations** benefit from persistent orientation and, when a
  follow-up is warranted, event-driven re-entry.
- **Decomposable problems** benefit from full-loop recursive subagents, which give each
  sub-task the full adaptive apparatus under isolation.
- **Environments with accumulating outcome data** benefit from online calibration, which
  tunes the difficulty and finalize thresholds to the observed workload.
- **Collaboration settings** may, under progressive trust and human approval, let a
  completed background task deliver its result back to its originating conversation.

Because every capability is bounded and reversible, the recommended path is to enable a
single feature, observe its behavior on a representative task, and expand adoption only as
confidence grows.

## 10. Conclusion

The design treats "depth" and "autonomy" not as switches but as *governed gradients*
driven by signals and earned through trust. By composing bounded OODA frames — adaptive
in depth, persistent in orientation, self-calibrating, and gated in action — the system
approaches the ideal of a continuous, infinite OODA loop while keeping every constituent
step accountable, inspectable, and safe. The infinite loop, in this view, is not the
removal of limits but their disciplined composition.
