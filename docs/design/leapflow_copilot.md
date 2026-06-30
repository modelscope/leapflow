# Workflow Copilot

## Vision

What GitHub Copilot did for code, LeapFlow Copilot does for workflows.

Code completion predicts the next line of code from cursor context. Workflow completion predicts the next operation from the user's current application state and recent action history. The trigger is not a keystroke, but a natural pause between operations. The suggestion is not a text insertion, but a ghost hint for the next step — accepted with a tap, dismissed by continuing to work.

Core value: **transform repetitive workflow decisions from "user recall" into "system proactive suggestion"**, reducing operational decision latency to near-zero for learned patterns.

## Key Insight

Workflow completion holds structural advantages over code completion:

| Dimension | Code Completion | Workflow Completion | Implication |
|:---|:---|:---|:---|
| Time budget | ~50ms between keystrokes | 300–3000ms between operations | **10× more compute time** before the user expects a response |
| Network dependency | Cloud model required | L0–L2 run entirely local | Works offline, no latency penalty |
| Personalization | Global model, slow RLHF cycle | Local model, real-time adaptation | Each user's model evolves independently |
| Cold start | Needs index upload | Local index + in-memory state | First response is instant |
| Evolution | Centralized model updates | **Decentralized, per-user evolution** | Gets better with every interaction |

The fundamental insight: because operation intervals are orders of magnitude longer than keystroke intervals, LeapFlow can run multi-layer prediction pipelines *during the operation itself*, making results available *before* the user pauses. This turns the prediction problem from "race against a deadline" into "fill idle time with useful speculation."

## How It Works

The end-to-end flow follows a six-stage loop:

```
Signal → Context → Predict → Suggest → Feedback → Evolve
```

1. **Signal**: Raw events arrive from the perception layer — app focus changes, file operations, clipboard updates, UI interactions, keyboard/mouse activity.

2. **Context**: An incremental encoder maintains a rolling context state. Only changed fields are updated per event (O(1) cost), producing a compact representation of "where the user is and what they just did."

3. **Predict**: Multiple prediction layers run in parallel, from fast exact-match lookups to deep LLM reasoning. Results are cached in a tiered structure as they complete.

4. **Suggest**: When a pause is detected, the best available prediction is rendered as a non-intrusive ghost hint — only if confidence exceeds threshold and the result arrived within the time window.

5. **Feedback**: The user's response (accept / ignore / correct) is captured as a structured signal.

6. **Evolve**: Feedback updates prediction confidence, adjusts layer weights, and refines the underlying models. This is Loop γ — "suggestion as learning."

## Prediction Architecture

Predictions are organized as a cascade of layers, each trading latency for reasoning depth:

| Layer | Strategy | Latency | When It Shines |
|:---:|:---|:---:|:---|
| **L0** | Context-hash exact match | <5ms | Daily routines, highly repetitive patterns |
| **L1** | N-gram / Markov transition | <10ms | Sequential patterns within a single app |
| **L2** | Embedding nearest-neighbor | <100ms | Semantically similar but not identical contexts |
| **L3** | LLM + RAG reasoning | <3000ms | Novel cross-application scenarios |

**Design philosophy**: Lower layers are always attempted first. If L0 hits with high confidence, higher layers are never invoked — the fast path dominates for learned patterns. When multiple layers agree on the same prediction, their confidences fuse:

```
P(combined) = 1 − ∏(1 − Pᵢ)
```

This creates a natural "consensus boost" — if both L0 and L1 predict the same action, the combined confidence rises sharply, enabling faster display.

Over time, the system self-optimizes: frequent patterns migrate from expensive L3 reasoning down to cheap L0 exact-match. The user experiences this as the system "getting faster" — because it literally is.

## Speculative Pre-computation

The core principle: **predict during the operation, display on the pause.**

This is analogous to CPU branch prediction. A modern CPU doesn't wait for a branch instruction to resolve before fetching the next instruction — it speculates. Similarly, LeapFlow doesn't wait for the user to pause before computing predictions — it speculates the moment an action occurs.

```
Timeline:
│── User Action A ──│── Pause 300ms ──│── User Action B ──│
│                   │                  │                   │
│  ↓ event fires    │  ↓ show hint     │                   │
│  L0+L1: instant   │  read from cache │                   │
│  L2: async fill   │                  │                   │
│  L3: conditional  │                  │                   │
```

**Three-tier cache architecture**:

| Tier | Source | Fill strategy | Access time |
|:---:|:---|:---|:---:|
| Instant | L0 + L1 | Synchronous with event | <1ms |
| Warm | L2 | Async task, callback to cache | <1ms (read) |
| Deep | L3 | Conditional async, high-value contexts only | <1ms (read) |

When a pause is detected, the system reads the best available result from cache — not computes it. Display latency drops from "prediction time" to "memory read time." If no result is ready, no suggestion is shown. The principle is strict: **better to show nothing than to show late.**

## Application Scenarios

**Scenario 1 — File Organization Completion**

The user drags `report_q1.pdf` from Downloads to `Documents/Reports/2025/`. The system detects the pattern and suggests: *"report_q2.pdf and report_q3.pdf are also in Downloads — move them too?"* The prediction comes from L1 recognizing the file-batch pattern and L0 matching the specific directory context.

**Scenario 2 — Application Switch Completion**

The user opens Zoom, then checks the calendar for the next meeting. The system suggests: *"Open the related meeting document and last week's notes?"* This fires because historical data shows 80% of Zoom+Calendar sequences are followed by document access.

**Scenario 3 — Terminal Workflow Completion**

The user runs `cd ~/project && git pull`. The system suggests: *"Last time you ran npm install && npm run dev after pulling — continue?"* L0 exact-matches the context hash (same repo, same command sequence).

**Scenario 4 — Form Auto-fill Completion**

The user opens an expense report form and fills in the date and amount. The system suggests: *"Category: Travel, Cost Center: R&D — auto-fill based on your last 5 reports?"* L2 retrieves semantically similar past form-filling sessions.

**Scenario 5 — Cross-Application Workflow Completion**

The user copies a requirement description from Slack. The system suggests: *"Create a Jira ticket with this description?"* L1's Markov model knows that Slack→Copy transitions to Jira→Create 70% of the time for this user.

## Self-Evolution Loop

The system improves with every interaction through a tight feedback cycle:

```
Predict → Display → User Response → Learn → (better) Predict
```

User responses carry different signal strengths:

| Response | Signal | Effect |
|:---|:---:|:---|
| **Accept** (Tab) | +1.0 | Strong positive reinforcement; pattern is confirmed |
| **Ignore** (continue working) | −0.1 | Weak negative; pattern confidence decays slowly |
| **Correct** (do something else) | −0.5 | Moderate negative; correct action is stored as ground truth |
| **Reject** (Esc) | −1.0 | Strong negative; pattern is actively suppressed |

Confidence updates use exponential moving average: `new = α·reward + (1−α)·old`. This ensures recent feedback matters most while preserving long-term trends.

The critical property: **the system's prediction accuracy is monotonically non-decreasing over usage time.** Patterns that work get reinforced; patterns that fail get suppressed; new patterns emerge from corrective signals.

## Design Principles

**1. Ghost Hints — Non-Intrusive by Default**

Suggestions appear as faint overlays (analogous to Copilot's gray text). They never block the user's current operation, never steal focus, never require dismissal. The user can simply continue working — the hint fades on its own.

**2. Never Late, Rather Absent**

If prediction hasn't completed by the time display is needed, the opportunity is forfeit. A delayed suggestion is worse than no suggestion — it interrupts flow rather than assisting it. Results are cached for the next similar context.

**3. Confidence-Gated Display**

| Confidence | Behavior |
|:---:|:---|
| < 0.5 | Silent — logged but never shown |
| 0.5 – 0.8 | Faint ghost hint — low visual weight |
| 0.8 – 0.95 | Clear suggestion with accept shortcut |
| > 0.95 + non-destructive + historical 100% accept | **Auto-execute** — zero-click completion |

This is the **trust gradient** applied to prediction: earned trust unlocks progressively autonomous behavior.

**4. Graceful Degradation**

Under resource pressure, the system sheds layers top-down (L3 → L2 → L1 → L0 → disabled). In the worst case, Copilot silently pauses — foreground operations are never impacted. Recovery is automatic when resources free up.

**5. Safety-First for Destructive Actions**

Predictions involving irreversible operations (file deletion, message sending, data modification) are never auto-executed regardless of confidence. They require explicit user confirmation even at the highest trust level.

## Integration with LeapFlow

Workflow Copilot is not a standalone system — it is a natural composition of LeapFlow's existing infrastructure:

| Copilot Component | LeapFlow Subsystem | Role |
|:---|:---|:---|
| Event sensing | Perception Layer + EventBus | Provides raw signal stream |
| Context state | WorkingMemory + RecordingContext | Maintains operational context |
| L0 exact-match | SkillLibrary (DuckDB) | Hash-indexed pattern lookup |
| L1 sequence prediction | Causal Pipeline | Markov transition statistics |
| L2 embedding retrieval | ExperienceStore + SemanticMemory | Vector similarity search |
| L3 reasoning | LLM Provider + WorldModel | Deep inference for novel scenarios |
| Feedback learning | Loop γ + FeedbackEvaluator | Closed-loop model updates |

**~70% of the required infrastructure already exists.** The primary gaps are three connecting layers: idle detection (recognizing pause windows), speculative pre-computation (filling cache during operations), and overlay rendering (showing ghost hints). The core perception, memory, and reasoning capabilities are already in place.

This means Workflow Copilot can be incrementally activated — starting with L0 exact-match over existing skill trajectories, progressively enabling deeper layers as confidence data accumulates.
