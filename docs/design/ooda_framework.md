# The Adaptive Learning Loop: An OODA Interpretation of LEAP

> A design philosophy document. Not a code architecture blueprint; not a line-by-line mapping.

## 1. Problem Statement

LeapFlow addresses a specific and challenging class of problems: **learning reusable automation skills from noisy, unstructured human demonstrations in open-ended desktop environments**.

This differs from classical Learning from Demonstration (LfD) in three fundamental ways:

**Open-world state space.** A robotic arm operates in a constrained state space (joint angles, end-effector position). A desktop environment has no closed state description — the user may interact with any combination of applications, files, and UI elements, in any order, across an unbounded variety of tasks. The agent cannot enumerate possible states in advance; it must generalize from sparse examples.

**The intent-action gap.** What a user *does* is not what they *mean*. A 50-step raw recording typically contains 40-60% noise: error corrections (undo/redo), exploratory detours, redundant operations, environmental interruptions, and suboptimal paths. The essential intent — the transferable skill — is buried within. Any system that cannot reliably extract intent from action will produce brittle, noisy skills.

**Situated learning.** The agent learns in the same environment where it will act. Unlike offline training paradigms (learn from dataset, then deploy), LEAP's learning and execution share the same event bus, the same perception layer, and the same execution ports. This co-location is both an opportunity (the agent can validate its understanding through execution) and a constraint (learning must not block normal use).

The question is: **what architectural pattern enables an agent to continuously observe, learn, and act in such an environment — and to improve with each cycle?**

---

## 2. Why OODA

We adopt the **OODA framework** (Observe, Orient, Decide, Act) as a conceptual lens for understanding LEAP's architecture. OODA was originally formulated to describe how adaptive systems maintain coherent behavior in rapidly changing, uncertain environments — which is precisely the challenge facing a desktop automation agent.

Three properties of the OODA framework make it particularly apt:

### 2.1 Orient as the Center of Gravity

In the OODA framework, Observe collects raw data, but **Orient** is where understanding is constructed. Orient synthesizes new observations with prior experience, cultural knowledge, and analytical models to produce a coherent situational picture. It is the heaviest, most consequential phase — the quality of orientation determines the quality of everything downstream.

```
                     Implicit Guidance
               ┌──────────────────────────┐
               │                          │
    Observe ───┤──→ Orient ──→ Decide ────┤──→ Act
               │       │                  │
               │       └──────────────────┘
               │   Orient feeds back to Observe
               │   and forward to Act (bypassing Decide)
               └───────────────────────────────────────→
                              environment feedback
```

In LEAP, the imitation learning pipeline — segmentation, multi-layer denoising, semantic abstraction, cross-modal verification, causal chain analysis, skill distillation — *is* the Orient phase. This is not a coincidence. The central challenge of the system (extracting intent from noisy action) maps directly to the central challenge of OODA (building accurate orientation from noisy observation). The pipeline's six-layer noise architecture is an engineering realization of the principle that **good orientation requires progressive refinement, not a single-pass filter**.

### 2.2 Implicit Guidance and the Trust Gradient

A distinctive feature of the OODA framework is that Orient does not merely feed into Decide. In mature systems, Orient directly shapes Observe (determining what to attend to) and directly triggers Act (bypassing deliberation entirely). This "implicit guidance" is what separates novice behavior (every action requires conscious decision) from expert behavior (appropriate action flows from situational awareness without deliberation).

LEAP implements this through **graduated confirmation**:

| Maturity | Behavior | OODA Interpretation |
|----------|----------|---------------------|
| New skill (v1, low confidence) | STEP — each action requires explicit approval | Full O→O→D→A cycle |
| Developing skill (v2) | CONFIRM / NOTIFY — approve the plan, not each step | Compressed Decide |
| Mature skill (v3+, high confidence) | AUTO — execute silently, log only | Orient → Act (implicit guidance) |

This graduation is not merely a UX convenience. It is a **trust gradient** — a formal mechanism by which the system earns operational autonomy through demonstrated competence. The trajectory from STEP to AUTO mirrors how human organizations delegate: a new employee's work is reviewed step-by-step; a trusted expert is given objectives and left to execute.

The trust gradient also reverses. When the evolution loop detects a regression (execution diverges negatively from the stored skill), it **downgrades** the skill's confirmation level — revoking autonomy until competence is re-established. This bidirectional trust adjustment is critical for safety in open-ended environments where the cost of a wrong action (deleting the wrong file, sending to the wrong recipient) can be high.

### 2.3 Multi-Scale Temporal Nesting

OODA is not a single loop. It operates at multiple timescales simultaneously, with faster inner loops refining the context that slower outer loops depend on. This multi-scale property maps naturally to LEAP's architecture, which operates three distinct adaptive loops.

---

## 3. Three Nested Loops

### Loop &alpha; — Skill Acquisition

**Timescale**: minutes to days. **Trigger**: a learning session.

This is the agent's primary learning mechanism: observe a demonstration, build understanding, and crystallize that understanding into a reusable, executable skill.

| Phase | Function | Key Architectural Principle |
|-------|----------|---------------------------|
| **Observe** | Capture raw trajectory via zero-invasion observation. Record every event; annotate suspected noise but discard nothing. | **Data as non-renewable resource.** Premature filtering destroys information that downstream layers may need. The recording layer preserves full fidelity, annotating rather than discarding suspected noise. |
| **Orient** | Transform noisy trajectory into clean understanding through six progressive layers of refinement. | **Layered refinement.** Each noise type has an optimal detection layer. Structural noise (undo/redo) is best caught by deterministic rules; granularity noise (scattered events) by semantic grouping; path noise (suboptimal sequences) by causal analysis; individual noise (per-session artifacts) by cross-trajectory consensus. No single layer attempts to solve all noise types — residual noise is explicitly passed to the next. |
| **Decide** | Route the distilled candidate: new skill, known skill, or ambiguous. | **Three-way routing.** High similarity to an existing skill triggers feedback evaluation; low similarity triggers direct registration; the ambiguous middle zone triggers LLM refinement or multi-trajectory consensus — the most expensive but most accurate path. |
| **Act** | Generate code, validate safety, bind to system ports, register. | **The agent's repertoire changes.** This is the consequential output — the skill library is the accumulated "experience" that future Orient phases draw on. |

**Insight: The Compression Ratio as Quality Metric.** A well-functioning Orient phase exhibits high semantic compression: 50 raw steps → 6 semantic actions → 4 causal steps. This ~12:1 ratio is not merely noise removal — it is **abstraction**, the process of identifying what is essential and discarding what is incidental. A low compression ratio (e.g., 50 → 40) suggests the pipeline is failing to separate signal from noise. A ratio that is too high (e.g., 50 → 1) suggests over-compression — loss of essential steps. Monitoring this ratio provides a quantitative signal about Orient quality.

### Loop &beta; — Skill Execution

**Timescale**: seconds to minutes. **Trigger**: a user request or pattern match.

The fastest loop. The agent perceives a request, assesses the situation, and acts.

| Phase | Function | Key Architectural Principle |
|-------|----------|---------------------------|
| **Observe** | Classify user intent; match against known skill triggers. | **Dual-pathway recognition.** Keyword matching provides zero-latency triggering for known phrases; LLM classification handles novel formulations. The system doesn't choose one path — it runs both, preferring the faster path when it produces a confident match. |
| **Orient** | Assess skill maturity, execution history, destructive potential, environmental context. | **Risk-proportional deliberation.** The system invests more deliberation in higher-risk situations (new skill, destructive operations, recent regressions) and less in low-risk ones (mature skill, read-only operations, clean history). This is not a fixed policy but a continuous function of context. |
| **Decide** | Human-in-the-loop confirmation — or implicit bypass for mature skills. | **The trust gradient** (see Section 2.2). |
| **Act** | Execute through bound perception and execution ports. | **Port abstraction.** The skill doesn't know (or care) whether it's running on macOS, a mock host, or a future Linux host. The Virtual System Interface ensures that Act is platform-independent. |

**Insight: Tempo Asymmetry.** Loop &alpha; is slow and deliberate; Loop &beta; is fast and (ideally) intuitive. This temporal asymmetry is a feature, not a limitation. The slow, thorough Orient of Loop &alpha; produces a rich skill representation that enables the fast, efficient Orient of Loop &beta;. The investment in careful learning pays dividends in rapid execution — just as a musician's slow, deliberate practice enables fast, fluid performance.

### Loop &gamma; — Skill Evolution

**Timescale**: continuous, post-execution. **Trigger**: every execution (automatic).

The silent loop. Every skill execution is recorded, evaluated, and used to refine the skill — without explicit user involvement.

| Phase | Function | Key Architectural Principle |
|-------|----------|---------------------------|
| **Observe** | Capture execution trajectory through the same always-on recorder used in LEARN mode. | **Unified observation.** The same event bus and the same recorder serve both demonstration capture (Loop &alpha;) and execution capture (Loop &gamma;). There is no separate instrumentation for feedback — recording is a background constant. |
| **Orient** | Compute structural diff between execution trajectory and stored skill using LCS-aligned comparison. | **Same-environment validation.** Because the agent learns and acts in the same environment, execution trajectories are directly comparable to demonstration trajectories. This is the "situated learning" advantage — no sim-to-real transfer gap. |
| **Decide** | Classify the outcome: improved (merge), unchanged (confidence++), regressed (downgrade). | **Conservative evolution.** Improvements must clear an evidence threshold before being auto-applied. Regressions trigger immediate protective action (confidence reduction, confirmation level increase). The system is biased toward stability. |
| **Act** | Update the skill version, adjust confidence, modify confirmation level. | **Feedback as environment change.** The Act of Loop &gamma; changes the state that Loop &beta; will observe in its next cycle — modified skill, different confidence, potentially different confirmation level. This is the cross-loop coupling that enables self-improvement. |

**Insight: The Dual Role of Execution.** Each skill execution serves two purposes simultaneously: (1) it accomplishes the user's task, and (2) it generates a clean reference trajectory that the system can compare against the (possibly noisy) original demonstration. Over multiple executions, the accumulated clean trajectories enable multi-trajectory consensus distillation — extracting what is common across all runs and discarding what is incidental. The agent doesn't need the user to "demonstrate again"; normal use provides the data for continuous refinement.

---

## 4. Cross-Loop Dynamics

The three loops are not independent. They form a reinforcing system whose collective behavior is richer than any single loop:

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│   Loop α (Acquisition)       Loop β (Execution)       Loop γ        │
│   ┌──────────────────┐      ┌────────────────────┐   (Evolution)    │
│   │ O → O → D → A    │─────→│ O → O → [D] → A   │──→┌──────────┐  │
│   │ slow, deliberate  │      │ fast, intuitive    │   │ O→O→D→A  │  │
│   └──────────────────┘      └────────────────────┘   │ silent    │  │
│          ▲                          ▲                 └──────────┘  │
│          │                          │                      │         │
│          │     confidence, version, confirmation level     │         │
│          └──────────────────────────┴──────────────────────┘         │
│                                                                      │
│          ▲                                                           │
│          │     repeated-pattern detection (no matching skill)        │
│          └───────────────────────────────────────────────────────────│
│                    Orient shapes what to Observe next                 │
└──────────────────────────────────────────────────────────────────────┘
```

Four critical cross-loop interactions:

**1. Skill transfer (&alpha; → &beta;).** Skills produced by the acquisition loop become the operational repertoire of the execution loop. A richer skill library means more user requests can be handled through proven, tested automation rather than general-purpose reasoning.

**2. Silent feedback (&beta; → &gamma;).** Every execution generates evaluation data without any additional user burden. This is the "free lunch" of situated learning — the agent extracts learning signal from its own operation.

**3. Trust modulation (&gamma; → &beta;).** Evolution loop outcomes directly modify execution loop behavior. A regressed skill demands more oversight; a proven skill earns more autonomy. This coupling ensures that the trust gradient responds to actual performance, not just to time or version count.

**4. Attention steering (&beta; → &alpha;, implicit).** When the system detects repeated application sequences that don't match any existing skill, it signals an acquisition opportunity — suggesting that the user enter LEARN mode. This is Orient shaping Observe: the agent's accumulated understanding of its own gaps directs where new learning should focus. This closes the outermost feedback loop, ensuring that the system's learning priority is driven by actual usage patterns rather than predefined curricula.

---

## 5. Insights for Complex, Real-World Tasks

The OODA interpretation reveals several non-obvious properties of the architecture that are particularly relevant when operating on diverse, real-world tasks:

### 5.1 Generalization Through Abstraction, Not Through Volume

Classical machine learning often achieves generalization through large training datasets. LEAP takes a different approach: **generalization through progressive abstraction**. A single demonstration, processed through six layers of refinement, can yield a skill that generalizes to variations the user never explicitly showed — because the abstraction process strips away incidental details (specific file names, window positions, timing) while preserving structural intent (scan, classify, move).

This matters for real-world deployment where demonstrations are expensive. A user should not need to show 100 examples of "organize my downloads" — one or two demonstrations, thoroughly abstracted, should suffice. The multi-layer Orient pipeline is what makes this possible.

### 5.2 Graceful Degradation Under Resource Constraints

The dual-pathway design (heuristic vs. LLM) throughout the pipeline provides natural degradation boundaries. When LLM calls are unavailable (network failure, quota exhaustion, latency constraints), the system falls back to heuristic paths — producing lower-confidence but still functional skills. This is not an error path; it is a designed operating mode.

In OODA terms: the Orient phase maintains multiple fidelity levels. The highest fidelity (LLM-powered intent inference, cross-modal VLM verification) produces the richest understanding. But even without these, the deterministic layers (structural denoising, pattern matching, causal chain analysis) still produce usable output. The system degrades gracefully rather than failing catastrophically.

### 5.3 The Cold Start Problem and Its Resolution

A newly deployed LEAP agent has no skills, no execution history, and no patterns beyond the bundled library. Its Loop &beta; can match nothing; its Loop &gamma; has nothing to evaluate. All intelligence must flow through Loop &alpha;.

But this cold start resolves rapidly, because:

- Each acquisition loop cycle produces at least one skill, immediately expanding Loop &beta;'s repertoire.
- Each execution of a new skill generates feedback for Loop &gamma;, even before the skill has matured.
- The pattern detection mechanism in Loop &beta; identifies acquisition opportunities, so the system's suggestions become more targeted as it accumulates more context about user behavior.

The nested loop structure means that **the first few cycles have disproportionate impact** — each new skill unlocks feedback opportunities that wouldn't otherwise exist. This is a positive bootstrapping dynamic, in contrast to systems that require large upfront investment before producing value.

### 5.4 Robustness to Environmental Diversity

Desktop environments are extraordinarily diverse. The same conceptual task ("organize files") manifests differently across operating systems, applications, file managers, and user preferences. The VSI port abstraction insulates skills from platform specifics, but more fundamentally, **the abstraction pipeline is what provides cross-environment generalization**.

A skill distilled at the semantic action level (`scan_directory`, `classify_by_content`, `batch_move`) is inherently more portable than one recorded at the raw event level (`click Finder toolbar button at (342, 78)`, `drag from row 3 to sidebar item 7`). The Orient phase's job is precisely this: to lift the representation from the incidental (pixel coordinates, specific UI elements) to the essential (intent, data flow, causal structure).

### 5.5 The Role of Cross-Modal Verification

In complex tasks, structural data (accessibility trees, file system events) and visual data (screen captures) provide complementary and sometimes contradictory signals. A click event might report a UI element ID that has changed since the demonstration; a screen capture might reveal a modal dialog that the accessibility API doesn't fully describe.

Cross-modal verification serves as an **internal consistency check** within Orient. When structural and visual modalities agree, confidence increases. When they diverge (temporal drift > threshold), confidence is penalized — the system flags that its understanding may be incomplete. This is a form of **epistemic humility**: the Orient phase not only produces situational understanding but also quantifies its own uncertainty about that understanding.

### 5.6 Why Multi-Trajectory Consensus Works

When the same task is demonstrated multiple times — whether intentionally or simply through natural repeated use — the resulting trajectories will share essential steps but differ in noise. The Longest Common Subsequence (LCS) across trajectories naturally extracts the intersection: steps that appear in every demonstration.

This works because **noise is stochastic but intent is deterministic**. The user's errors, detours, and interruptions differ between sessions, but the core task steps are consistent. Multi-trajectory consensus is, in effect, a statistical filter: it requires no noise model, no threshold tuning, and no LLM — the structure of the problem guarantees convergence.

In OODA terms, multi-trajectory consensus enriches Orient by synthesizing multiple observations of the same phenomenon, analogous to how repeated exposure to a situation deepens situational awareness.

---

## 6. Applying the Framework

### For Design Decisions

Ask: **Which loop does this change primarily affect, and what is its cross-loop impact?**

Changes that improve Orient in any loop tend to cascade positively. A better denoising strategy (Loop &alpha; Orient) produces cleaner skills, which produce cleaner execution trajectories (Loop &gamma; Observe), which produce more accurate feedback verdicts (Loop &gamma; Orient), which produce better trust calibration (Loop &beta; Orient). Conversely, changes that optimize only one phase of one loop have bounded impact.

### For Diagnosing System Behavior

| Symptom | Likely Loop | Phase to Inspect |
|---------|-------------|-----------------|
| Skill quality poor despite clean demonstrations | &alpha; | Orient — is the abstraction pipeline compressing appropriately? |
| Skill quality poor with noisy demonstrations | &alpha; | Orient — is DenoisePass engaged? Are noise annotations present? |
| Correct skill not triggered | &beta; | Observe — are trigger phrases adequate? Is intent classification accurate? |
| Skill not graduating despite successful executions | &gamma; | Decide — are confidence thresholds or regression checks too conservative? |
| Skill regressed but still running at AUTO | &gamma; | Decide → Act — is the regression-to-downgrade path functioning? |
| System not suggesting LEARN for repeated manual tasks | &beta; → &alpha; | Implicit feedback — is pattern detection active? Is the threshold too high? |

### For Feature Prioritization

**Orient-centric prioritization**: features that enrich the Orient phase of any loop pay compound dividends across the entire system. A rough priority heuristic:

1. Features that improve Orient quality (noise handling, abstraction, cross-modal verification) — highest leverage.
2. Features that strengthen cross-loop coupling (better feedback verdicts, attention steering, consensus triggers) — multiplier effects.
3. Features that optimize single-phase, single-loop behavior (faster trigger matching, smoother UI) — bounded but visible impact.

---

## 7. Boundaries of the Framework

OODA captures the **adaptive** character of LEAP — how it observes, learns, decides, and acts in a continuous improvement cycle. It is the right lens for understanding why the system gets better with use.

It is not the only lens needed:

- **Compositional structure** (DAG planning, skill chaining, task graphs) is better described by dataflow models.
- **Platform abstraction** (VSI, port protocols, capability handshake) is better described by layered architecture patterns.
- **Memory management** (immediate → working → long-term, decay functions) is better described by cognitive architecture models.

The purpose of this document is not to compress the entire system into one framework, but to offer a vantage point that illuminates the aspect most central to LEAP's value proposition: an agent that **genuinely improves through use** — not through retraining, not through manual rule updates, but through the natural cycle of observation, understanding, action, and reflection.

---

## 8. Context Learning Attention Mechanism

### 8.1 The Problem: Signal-to-Noise in Open Environments

Section 1 identified the open-world state space as a fundamental challenge. A corollary — less obvious but equally consequential — is that **the observation stream itself has a signal-to-noise problem**.

When LEAP records a demonstration, the event bus captures *everything*: file system changes from background sync daemons, focus flickers from notification pop-ups, clipboard updates from password managers, and process activity from dozens of applications the user never intentionally touched. In a typical macOS session, fewer than 30% of captured events are relevant to the user's demonstrative intent.

This is not a denoising problem in the traditional sense. The existing six-layer Orient pipeline (Section 3, Loop α) handles *action-level* noise — undo/redo sequences, idempotent repetitions, distraction switches. What it cannot handle is *observation-level* noise: events that should never have entered the pipeline at all, because they originate from sources entirely unrelated to the demonstration.

The distinction matters architecturally. Action-level denoising operates on a trajectory that already represents the user's activity. Observation-level noise pollutes the trajectory with events from *other agents and processes* — making subsequent segmentation, abstraction, and distillation operate on a fundamentally corrupted input.

### 8.2 Attention as Orient Shaping Observe

The OODA framework provides the conceptual foundation for the solution. In Section 2.2, we described how mature systems exhibit **implicit guidance** — Orient directly shapes Observe, determining what to attend to. The Context Learning Attention Mechanism is the engineering realization of this principle for the recording layer.

```
                 ┌────────────────────────────────────────┐
                 │          Implicit Guidance              │
                 │                                        │
    Observe ─────┤──→ Orient ──→ Decide ──→ Act          │
       ▲         │       │                               │
       │         │       │                               │
       └─────────┴───────┘                               │
     Orient feeds back to                                 │
     shape WHAT is observed                               │
                                                          │
     Context Learning Attention                           │
     = this feedback arrow                                │
     made concrete                                        │
└────────────────────────────────────────────────────────┘
```

The mechanism does not discard events outright. It **annotates** suspected noise, preserving full fidelity while providing a signal for downstream layers to preferentially weight relevant observations. This honors the "Data as non-renewable resource" principle (Section 3, Loop α) while still solving the signal-to-noise problem.

### 8.3 Four Layers of Attention

The attention mechanism operates at four priority levels, each addressing a distinct noise source. Like the six-layer Orient pipeline, these are ordered by specificity — broader filters first, finer-grained filters later — so each layer operates on a progressively cleaner signal.

| Priority | Name | Scope | Principle |
|----------|------|-------|-----------|
| **P0** | Foreground Gate | Real-time | Only apps the user has actively focused during this session carry demonstrative intent |
| **P1** | Goal-Directed Relevance | Post-hoc | The user's stated goal defines a relevance field; events outside it are noise |
| **P2** | Noise Source Rules | Real-time | Configurable patterns identify known noise sources (system paths, background daemons) |
| **P3** | Working Directory Inference | Real-time | The user's working context is inferred from early file activity; events outside it are noise |

**P0: Foreground Gate** — The simplest and most powerful filter. A user demonstrates a task by *interacting with applications*. If an application has never received focus during a recording session, events attributed to it are almost certainly background noise. The filter dynamically builds a set of "attended apps" from `app.focus_change` events and gates all other events through membership. This is attention in its most literal sense: the system attends to what the user attends to.

**P1: Goal-Directed Relevance** — When the user provides a goal ("Organize PDF files"), that goal defines a semantic field. Steps whose targets, labels, and app names share no lexical overlap with the goal are unlikely to be part of the demonstration. This filter operates post-hoc (after recording, before segmentation) because it requires the complete trajectory to score. It is the mechanism by which *Orient shapes future Observe* — the user's stated intent (an Orient-phase artifact) retroactively filters what is considered signal in the Observe phase.

**P2: Noise Source Rules** — Configurable regex patterns that identify known noise generators. Unlike hardcoded app blacklists, these are user- and environment-specific, loaded from configuration. Examples: `/Library/Caches/`, `com.apple.CloudDocs`, `.DS_Store`. The generality matters — the same mechanism works for any platform, any noise source, without code changes.

**P3: Working Directory Inference** — File system events are the noisiest channel in a desktop environment. The filter infers the user's working directory from the common path prefix of early file events, then annotates subsequent file events outside that scope. This is a form of *spatial attention* — the system learns *where* the user is working and focuses observation there.

### 8.4 Design Philosophy: Annotate, Don't Discard

A critical design decision: the attention filters **annotate** rather than **discard**. Events flagged as noise are still recorded, but with a structured `_noise` annotation that downstream layers can use to adjust their behavior. This preserves the "data as non-renewable resource" principle while still improving signal-to-noise ratio for the Orient pipeline.

This choice has practical consequences:

- **Reversibility.** A mis-classified event can be recovered. As the system's understanding improves (through multi-trajectory consensus, user correction, or context refinement), previously-annotated events may become relevant.
- **Transparency.** The user can inspect *why* the system considered an event noise (the annotation carries a `reason` field). This supports trust calibration — the user can verify that the system's attention aligns with their intent.
- **Composability.** Multiple filters can annotate the same event independently. The downstream pipeline can apply different confidence thresholds depending on context — strict for code generation, lenient for trajectory visualization.

### 8.5 Relationship to the Layered Refinement Architecture

The attention mechanism is not a replacement for the existing six-layer Orient pipeline (DenoisePass, GroupingPass, PatternPass, etc.). It is a **zeroth layer** — a pre-filter that improves the quality of input to the existing pipeline.

```
Event Bus
    │
    ▼
┌──────────────────────────────────────────┐
│  Context Learning Attention (Layer 0)     │
│  P0: Foreground Gate                      │
│  P2: Noise Source Rules                   │
│  P3: Working Directory Inference          │
│  ─────────────────────────────────────── │
│  Annotate observation-level noise         │
└──────────────────────────────────────────┘
    │
    ▼  (trajectory with noise annotations)
┌──────────────────────────────────────────┐
│  Goal Relevance Filter (P1)               │
│  Post-hoc semantic relevance scoring      │
└──────────────────────────────────────────┘
    │
    ▼  (filtered trajectory)
┌──────────────────────────────────────────┐
│  Orient Pipeline (Layers 1-6)             │
│  L1: DenoisePass (undo, idempotent, ...)  │
│  L2: GroupingPass (merge consecutive)     │
│  L3: PatternPass (recognize sequences)    │
│  L4: Causal Analysis                      │
│  L5: Cross-modal Verification             │
│  L6: Consensus Distillation               │
└──────────────────────────────────────────┘
    │
    ▼  (clean skill representation)
```

The separation is intentional. Observation-level noise (wrong source) is fundamentally different from action-level noise (wrong execution) — they require different detection strategies, operate at different temporal granularity, and have different confidence profiles. Conflating them into a single denoising layer would violate the Single Responsibility Principle and produce a monolithic, hard-to-tune filter.

### 8.6 Insight: Attention as Acquired Capability

In the initial cold start (Section 5.3), the attention mechanism has minimal context: no focused apps (P0 passes everything), no noise patterns (P2 inactive), no working directory (P3 passes everything). Only P1 operates from the beginning, because the user provides the goal before recording starts.

As the recording session progresses, the attention mechanism **learns its own context**:

- The first `app.focus_change` event populates P0's attended-app set.
- The first few `fs.change` events allow P3 to infer the working directory.
- User-configured noise patterns (P2) represent accumulated environmental knowledge.

This mirrors the OODA dynamic described in Section 3: the earliest observations shape subsequent observation quality. The attention mechanism bootstraps itself — each event it observes refines the criteria by which future events are evaluated. This is a miniature OODA loop operating *within* the Observe phase itself, demonstrating the fractal nature of the framework: adaptive loops nested within adaptive loops, each improving the fidelity of the one above it.
