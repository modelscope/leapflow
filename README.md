# LeapFlow

**Learning and Evolving from Actual Practice**

LeapFlow is a self-evolving desktop automation framework built on this premise. Instead of hand-coding every automation rule, LeapFlow observes how you interact with your computer — clicks, keystrokes, file operations, app switches — and distills those interactions into reusable, executable skills. The more you use it, the smarter it gets.

## Why LeapFlow

Traditional automation tools require explicit programming: write a script, define triggers, maintain brittle code. AI agents powered by LLMs can reason about tasks, but they start from scratch every time — no memory of what worked before, no ability to improve.

LeapFlow bridges this gap. It combines real-time system perception with a recording → analysis → learning pipeline that transforms your natural workflows into parameterized, executable skills — automatically. Skills evolve through a feedback loop: each re-execution is compared against the stored version, and improvements are auto-applied or surfaced for review.

**Core thesis**: The best automation comes from actual practice, not from prompt engineering.

## How It Works

### The Experience Hierarchy

LeapFlow organizes what it observes into a five-level hierarchy, each level more abstract and reusable than the last:

```
Trajectory    Raw recording of a work session (timestamped events)
    ↓ segmentation (time gaps, app switches, semantic shifts)
Episode       A coherent unit of work (e.g., "organized downloads folder")
    ↓ multi-level abstraction (L0→L1→L2→L3)
Pattern       Recognized action sequences (e.g., copy → switch app → paste = "transfer_data")
    ↓ distillation (heuristic + LLM)
Skill         Parameterized, executable automation (code-generated, AST-validated)
    ↓ composition
Workflow      Chained skills solving complex, multi-step goals
```

### The Learning Loop

```
                    ┌─────────────────────────────────────┐
                    │         You work normally            │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │   DemonstrationRecorder              │
                    │   (subscribes to EventBus,           │
                    │    zero-invasion observer)            │
                    └──────────────┬──────────────────────┘
                                   │ Trajectory
                    ┌──────────────▼──────────────────────┐
                    │   SegmentDetector                    │
                    │   (time gaps + app switches +        │
                    │    semantic boundary detection)       │
                    └──────────────┬──────────────────────┘
                                   │ Episodes
                    ┌──────────────▼──────────────────────┐
                    │   ActionAbstractor                   │
                    │   DenoisePass → GroupingPass →        │
                    │   PatternPass (YAML-driven)           │
                    └──────────────┬──────────────────────┘
                                   │ Semantic Actions
                    ┌──────────────▼──────────────────────┐
                    │   SkillDistiller                     │
                    │   (heuristic + LLM dual pathway)     │
                    └──────────────┬──────────────────────┘
                                   │ DistillationCandidate
                    ┌──────────────▼──────────────────────┐
                    │   ActiveLearningObserver             │
                    │   • New skill? → save + codegen      │
                    │   • Similar to existing? → merge     │
                    │   • Same skill, better? → evolve     │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │   SkillActivator                     │
                    │   compile → bind ports → register    │
                    │   → ready to execute on next match   │
                    └─────────────────────────────────────┘
```

### Self-Evolution

Skills aren't static. When LeapFlow detects you performing a task similar to an existing skill but with variations, it enters the **feedback loop**:

1. **Structural diff** — compares the new execution against the stored skill (action-level + step-level, LCS-aligned)
2. **Heuristic verdict** — classifies the change: `improved` (additive/efficiency/structural), `unchanged`, or `regressed`
3. **LLM refinement** (optional) — when heuristic confidence is low, an LLM evaluates whether the change is genuinely better
4. **Auto-apply or suggest** — high-confidence improvements are merged automatically (version bumped, triggers expanded, steps aligned); ambiguous cases are surfaced as `SkillUpdateSuggestion` for user review

Over time, skills accumulate knowledge across multiple executions, becoming more robust and covering more edge cases — without manual intervention.

## Cross-Modal & Multi-Scale Imitation Learning

LeapFlow's core innovation is a **cross-modal & multi-scale imitation learning** pipeline. It fuses multiple observation modalities — structural (AXTree events, file system changes), visual (screen capture frames), and semantic (LLM-inferred intent) — and operates across multiple temporal scales: from raw input events (milliseconds) through grouped actions (seconds) and coherent episodes (minutes) up to reusable skills and composed workflows. This cross-modal, multi-scale fusion transforms noisy human demonstrations into clean, executable skills through a multi-stage process spanning three operating modes: LEARN, DISTILL, and EXECUTE.

### Design Philosophy: Context Attention

Beneath the three modes lies a deeper challenge: **the OS is an overwhelmingly complex environment**. At any moment, hundreds of applications compete for attention, thousands of UI elements populate the accessibility tree, and continuous streams of events — file changes, clipboard updates, app switches, notifications — flood in at rates far exceeding what any agent can process. A naive observer that records everything drowns in noise; one that filters too aggressively misses the signal.

LeapFlow's answer is **Context Attention** — a mechanism that dynamically focuses the agent's perception, learning, and execution on the subset of OS context that matters for the current task. Like biological attention, it operates at multiple timescales: immediate (which events to record now), tactical (which patterns to extract from a recording), and strategic (which skills to refine over time). Context Attention is not a single module but an architectural principle woven through every layer — from `GoalRelevanceFilter` and `AttentionContext` in recording, through `CausalChainAnalyzer` in analysis, to the graduated confirmation system in execution.

We use the [OODA framework](docs/design/ooda_framework.md) (Observe, Orient, Decide, Act) to map how Context Attention manifests as **three nested adaptive loops operating at different timescales**:

```
┌─────────────────────────────────────────────────────────────────────┐
│  Loop α — Skill Acquisition (hours~days)                            │
│  Observe → Orient → Decide → Act                                    │
│  Watch demos   Build understanding   Route candidate   Register skill│
│                (6-layer pipeline)    (new/merge/evolve) (codegen)    │
├─────────────────────────────────────────────────────────────────────┤
│  Loop β — Skill Execution (seconds~minutes)                         │
│  Observe → Orient → Decide → Act                                    │
│  Classify intent  Assess risk    Human confirm   Execute via ports   │
│                   & maturity     (or AUTO bypass)                    │
├─────────────────────────────────────────────────────────────────────┤
│  Loop γ — Skill Evolution (post-execution, continuous)              │
│  Observe → Orient → Decide → Act                                    │
│  Capture exec    Structural diff  Verdict routing  Merge/regress    │
│  trajectory      (LCS alignment)  (improve/regress) (version++)     │
└─────────────────────────────────────────────────────────────────────┘
        γ.Act feeds back to β.Orient and α.Orient
        β.Act feeds forward to γ.Observe
        α.Act feeds forward to β.Observe
```

The key insight: as skills mature through repeated execution, they **graduate from explicit decision-making to implicit guidance** — a new skill requires step-by-step human confirmation (STEP), while a proven skill executes autonomously (AUTO). This progression from deliberation to intuition is what makes LeapFlows genuinely improve with use.

For the full analysis — including cross-loop interactions, how to use this framework for design reviews and debugging, and its limitations — see **[The Adaptive Learning Loop: An OODA Interpretation of LeapFlow](docs/design/ooda_framework.md)**.

### Operating Modes

```
┌─────────────────────────────────────────────────────────────┐
│                       LeapFlow Modes                       │
│                                                              │
│  ┌──────────┐    ┌──────────┐    ┌──────────────────────┐   │
│  │  LEARN   │───→│ DISTILL  │───→│      EXECUTE         │   │
│  │ Observe  │    │ Extract  │    │ Run (human-in-loop)  │   │
│  └──────────┘    └──────────┘    └──────────────────────┘   │
│       ▲                                     │               │
│       └─────────── feedback loop ───────────┘               │
└─────────────────────────────────────────────────────────────┘
```

The three modes are not strictly sequential — distillation runs asynchronously in the background, and learning and execution can alternate freely. A `SessionMode` state machine manages transitions:

```
                 learn start
    IDLE ─────────────────────→ LEARNING
     ▲                            │
     │                            │ learn stop
     │                            ▼
     │                        DISTILLING ─── (background, auto-returns to IDLE)
     │                            │
     │                            │ distillation complete
     │◄───────────────────────────┘
     │
     │         execute / trigger
     └─────────────────────────→ EXECUTING
                                   │
                                   │ complete / abort
     IDLE ◄────────────────────────┘
```

All mode transitions are protected by an `asyncio.Lock` for concurrency safety. The DISTILLING state is transparent to the user — they can continue using all IDLE features while distillation runs in the background.

### LEARN Mode — Recording Demonstrations

LEARN mode uses zero-invasion observation: `DemonstrationRecorder` subscribes to the `EventBus` as a passive listener, recording everything without interfering with normal operation.

**Entering and exiting**:

```bash
# CLI
leap learn --prompt "Organize my Downloads folder"

# Interactive
> learn start organize downloads
> learn stop

# Programmatic
await session.enter_learning(goal="organize downloads")
await session.exit_learning()
```

**During learning**, users work normally and can:

| Command | Effect |
|---------|--------|
| `annotate <text>` | Add annotation to current step, boosting distillation weight |
| `learn pause` / `learn resume` | Pause/resume recording (e.g., taking a call) |
| `skip [n]` | Mark last n steps as noise (user-initiated skip signal) |
| `learn stop` | End learning, trigger async distillation |

**Dual-track recording**:

- **Structural track** — AXTree snapshots + EventBus events (clicks, keystrokes, file ops, app switches), captured at three fidelity levels (FULL / LIGHT / MINIMAL) based on event importance
- **Visual track** — strategic screen capture at key moments (app focus change, click, post-typing), stored as compressed frames with references in `StateSnapshot.visual_frame_ref`

Recording is fault-tolerant: DuckDB write failures are buffered in memory (up to 500 entries) and retried on the next successful write.

### DISTILL Mode — From Noisy Traces to Clean Skills

When learning ends, the distillation pipeline runs asynchronously through six stages:

```
Raw Trajectory (noisy recording)
    │
    │ ① Segmentation
    │   TimeGapDetector + AppSwitchDetector + SemanticBoundaryDetector
    ▼
Episodes[] (semantically cohesive segments)
    │
    │ ② Multi-level Abstraction
    │   DenoisePass → GroupingPass → PatternPass [→ VisualPass]
    ▼
SemanticActions[] (clean, abstract action sequences)
    │
    │ ③ Intent Inference (optional, LLM-powered)
    │   Infer user intent for each episode
    ▼
Episodes with inferred_goal
    │
    │ ④ Skill Distillation
    │   Heuristic path (zero LLM cost) or LLM path (higher precision)
    ▼
DistillationCandidate[]
    │
    │ ⑤ Active Learning Decision
    │   score < 0.3   → new skill, save directly
    │   score > 0.85  → existing skill, feedback evaluation
    │   0.3 ~ 0.85   → ambiguous, LLM refinement or multi-trajectory consensus
    ▼
Skill registration (codegen → AST validation → port binding → registry)
```

**Noise robustness** is achieved through a multi-layer defense architecture — the core design principle is that each type of noise has an optimal detection layer, and residual noise is passed to the next:

| Layer | Name | What it removes | Mechanism |
|-------|------|----------------|-----------|
| 0 | Recording | (nothing removed) | Full capture + noise annotation (`_noise` field on each step) |
| 1 | DenoisePass | Structural noise | `UndoCollapseStrategy` (error corrections), `IdempotentMergeStrategy` (redundant saves/scrolls), `DistractionFilterStrategy` (notification glances) |
| 2 | Abstraction | Granularity noise | `GroupingPass` (merge consecutive same-type), `PatternPass` (YAML pattern recognition) |
| 3 | Distillation | Path noise | `CausalChainAnalyzer` (keep only causally relevant steps), LLM path optimization |
| 4 | Consensus | Individual noise | `MultiTrajectoryDistiller` (LCS across multiple demonstrations of the same task) |
| 5 | Execution feedback | Residual noise | `FeedbackEvaluator` (runtime execution results as the final arbiter) |

A typical 50-step noisy demonstration is reduced to a 4-5 step clean skill — ~92% noise elimination.

**Cross-modal verification**: `VisualAbstractionPass` pairs each semantic action with the nearest visual frame by timestamp, using a VLM to verify that structural and visual information agree. Drift > 5s between modalities triggers a confidence penalty, catching cases where AXTree data alone might be misleading.

**Progress feedback**: The pipeline fires `ProgressCallback(stage, current, total)` at each stage, allowing the UI to display real-time learning progress.

### EXECUTE Mode — Human-in-the-Loop Skill Execution

Skills are triggered by natural language matching (`SkillRegistry.find_by_trigger()`), explicit commands, or as nodes in a DAG plan (`GraphPlanner`).

**Graduated confirmation** — skills "graduate" from strict to autonomous as they prove reliable:

```
STEP (new/low-confidence)  →  CONFIRM  →  NOTIFY  →  AUTO (mature)
   v1, conf < 0.6               default       v2+       v3+, conf ≥ 0.85
```

| Level | Behavior | When |
|-------|----------|------|
| `STEP` | Execute one step at a time, confirm each | New skill (v1) or confidence < 0.6 |
| `CONFIRM` | Show full plan, wait for yes/no/step | Default, or recent regression detected |
| `NOTIFY` | Execute immediately, notify result | Version 2+, no regressions |
| `AUTO` | Silent execution, audit log only | Version 3+, confidence ≥ 0.85, no regressions |

Regression detection: `ConfirmationHandler` checks the last 5 executions via `SkillLibraryStore` — any "regressed" verdict downgrades the skill back to CONFIRM.

**Step-through execution** uses a real `StepExecutor` callback that actually invokes each step via the skill's `run` function and bound ports. Failed steps are reported immediately, and the user can choose to continue or abort.

**Skill composition**: `execute_skill_sequence()` chains multiple skills in sequence, stopping on first failure — enabling complex multi-step workflows from simpler building blocks.

**Undo support**: `ExecutionPort.undo_last()` provides best-effort rollback for the last operation (reverses file moves/copies; returns `not_reversible` for destructive ops).

### End-to-End Example

```
User:   learn start organize PDF files in Downloads

        [User works normally — opens Finder, sorts by kind,
         drags PDFs to categorized folders, occasionally
         checks WeChat notifications, makes a wrong move
         and undoes it...]

Agent:  Learning complete. Recorded 35 steps in 2m34s. Analyzing...

        [Background: DenoisePass removes 12 noise steps,
         GroupingPass + PatternPass compress to 6 semantic actions,
         CausalChainAnalyzer extracts 4-step causal chain,
         SkillDistiller produces DistillationCandidate]

Agent:  New skill "Categorize PDFs by Content" ready (confidence: 0.72).
        Say "organize my PDF downloads" to trigger.

--- later ---

User:   organize my PDF downloads

Agent:  Ready to execute "Categorize PDFs by Content":
          1. Scan ~/Downloads/ for PDF files
          2. Classify by content
          3. Move to categorized directories
        Confirm? (yes/no/step)

User:   yes

Agent:  Done — 8 files organized into 3 categories.

        [Background: execution trajectory compared to v1 skill,
         verdict: unchanged, confidence nudged to 0.74]
```

After 3 successful executions, the skill graduates to AUTO — next time it runs without asking.

## Architecture

LeapFlow follows a dual-process design: a **Python Brain** (reasoning, memory, learning) communicating with a platform-specific **Host** (perception, execution) over MsgPack-RPC.

```
┌─────────────────────────────────────────────────────────────────┐
│                         Python Brain                             │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌───────────────┐  │
│  │ Intent   │→ │ ReAct    │→ │ LLM       │→ │ Memory        │  │
│  │Classifier│  │ Engine   │  │ Provider  │  │ Imm → WM → LT │  │
│  └──────────┘  └──────────┘  └───────────┘  └───────────────┘  │
│       │              │              │                ▲           │
│       └──────────────┴──────────────┴────────────────┘          │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              engine/ — Session Controller                  │    │
│  │  SessionMode state machine (IDLE↔LEARN↔DISTILL↔EXECUTE)   │    │
│  │  asyncio.Lock concurrency · AuditLogger (JSONL)          │    │
│  │  ConfirmationHandler (graduated confirmation)             │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  recording/ + analysis/ — Imitation Learning Pipeline     │    │
│  │  Recorder → Segmenter → Abstractor → Distiller            │    │
│  │  DenoisePass · VisualPass · CausalChainAnalyzer           │    │
│  │  MultiTrajectoryDistiller (cross-trajectory consensus)    │    │
│  │       ↕ EventBus              ↕ SkillLibrary (DuckDB)     │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  skills/ + learning/ — Skill System                       │    │
│  │  ActiveLearningObserver ← FeedbackEvaluator               │    │
│  │  SkillActivator → SkillRegistry (OCP)                     │    │
│  │  CodeGenerator (LLM + Template, AST-validated)            │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  platform/ — Platform Abstraction Layer                   │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐   │    │
│  │  │ Perception   │  │ Execution    │  │ Event         │   │    │
│  │  │ Port         │  │ Port (+undo) │  │ Normalizer    │   │    │
│  │  └──────────────┘  └──────────────┘  └───────────────┘   │    │
│  │  Adapters: Darwin / Mock / (Linux, planned)               │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└───────────────────────────────┬──────────────────────────────────┘
                                │ MsgPack-RPC (Unix Domain Socket)
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Platform Host (e.g., Swift on macOS)           │
│  Perception: AXTree, FSEvents, Clipboard, ScreenCapture, UIAction│
│  Execution:  File ops, App control, Shell, Notifications         │
│  Security:   PermissionGuard + AuditLog (JSONL)                  │
└─────────────────────────────────────────────────────────────────┘
```

### Event-Driven Memory

LeapFlow's memory is **event-driven**, not request-driven. The agent processes information the moment it arrives — not when you ask about it.

```
System Events (file changes, clipboard, app focus, UI actions)
    │ push (Host → Brain via MsgPack event frame)
    ▼
EventBus → EventNormalizer (platform-specific → SystemEvent)
    │
    ▼
Immediate Memory ──────────────────────── DemonstrationRecorder
    • TTL = 5 min, GC every 30s                (parallel observer,
    • Max 200 fragments (ring eviction)         zero-invasion hookup
    • touch() triggers promotion                via EventBus.subscribe)
    ▼
Working Memory (ring buffer, ~8K tokens, LRU eviction)
    ▼
Long-Term Memory (DuckDB)
    • Decay: W = S · e^(-λt) · (1 + ln F)
    • ILIKE keyword + time-range retrieval
```

The DemonstrationRecorder subscribes to the same EventBus as the memory system — no separate instrumentation, no extra overhead. Recording is just another subscriber.

### Multi-Platform via Platform Layer

The **Platform layer** (`platform/`) decouples the Brain from any specific OS. At startup, a capability handshake (`system.manifest` RPC) discovers what the host can do:

| Platform | Status | Capabilities |
|----------|--------|-------------|
| macOS 14–15 (Legacy) | Implemented | AXTree, FSEvents, Clipboard, File ops, App control |
| macOS 26+ (Tahoe) | Planned | App Intents, SemanticIndex, GPU ScreenCapture |
| Linux GNOME | Planned | AT-SPI, inotify, D-Bus, PipeWire |
| Linux KDE | Planned | AT-SPI, inotify, D-Bus |

The Brain code — including the entire recording and analysis pipeline — is platform-agnostic. To support a new platform, implement a Host process and the corresponding platform adapter. The `MockPerceptionAdapter` / `MockExecutionAdapter` allow full Brain development and testing without any Host at all.

### Action Abstraction Levels

Raw events are noisy. LeapFlow's `ActionAbstractor` runs a composable pipeline of passes to extract meaning:

| Level | Name | Example | Mechanism |
|-------|------|---------|-----------|
| L0 | Raw | `keyDown(code=36)` | Direct from EventBus |
| L0' | Denoised | (undo/redo collapsed, distractions removed) | `DenoisePass` — structural noise elimination |
| L1 | Grouped | `type_text × 15` | `GroupingPass` — merge consecutive same-type actions |
| L2 | Pattern | `transfer_data(src→dst)` | `PatternPass` — YAML-driven pattern library with wildcards |
| L3 | Verified | Pattern + visual confirmation | `VisualPass` — VLM cross-modal verification (optional) |

Each pass is an `AbstractionPass` — add new passes without modifying existing ones.

## Installation

### Prerequisites

| Component | Version | Required | Purpose |
|-----------|---------|----------|---------|
| Python | 3.11+ | Yes | Brain runtime |
| [uv](https://github.com/astral-sh/uv) | latest | Yes | Package & venv manager |
| macOS | 14+ | No | Native Host platform |
| Xcode CLI Tools | latest | No | Build Swift Host |

> macOS and Xcode are only needed for **full mode** (native Host with real system perception). The Python Brain runs standalone with `--mock-host` on any platform.

### One-Line Setup

```bash
git clone https://github.com/modelscope/leapflow.git
cd leapflow
chmod +x scripts/setup.sh && ./scripts/setup.sh
```

Or equivalently: `make setup`.

The script performs:
1. Creates a Python virtual environment via `uv`
2. Installs all runtime + dev dependencies (`uv sync --all-extras`)
3. Copies `.env.example` → `.env` (if not already present)
4. Prints next-step instructions

### Manual Setup (Python Brain)

If you prefer manual setup over the script:

```bash
# Option A — uv (recommended)
uv sync --all-extras

# Option B — pip
pip install -e ".[dev]"
```

Both methods install all dependencies and register the `leap` CLI entry point:

```bash
# With uv (no venv activation needed)
uv run leap chat --mock-host --prompt "hello"

# With pip (after activating the venv)
source .venv/bin/activate
leap chat --mock-host --prompt "hello"
```

### Building the Swift Host (macOS)

The native Host provides real-time system perception (FSEvents, AXTree, Clipboard, ScreenCapture) and execution (file ops, app control, shell). Skip this if using `--mock-host`.

```bash
# Build
cd os_host 

# Build and Run (stays in foreground, listens on Unix socket)
swift build -c debug && swift run -c debug OSHost

```

Or from the project root: `make swift-build` (build only) / `make host` (build + run).

The Host communicates with the Brain over a Unix domain socket (`/tmp/leapflow.sock` by default). Both sides must share the same `LEAPFLOW_BRIDGE_SOCKET` path. No external Swift dependencies — only Apple system frameworks.

### macOS Permissions

On first run with the native Host, grant these in **System Settings → Privacy & Security**:

| Permission | Purpose | When needed |
|------------|---------|-------------|
| **Accessibility** | AXTree reading, UI actions | Always (full mode) |
| **Full Disk Access** or **Files and Folders** | File system monitoring (FSEvents) | Always (full mode) |
| **Screen Recording** | Screen capture for visual track | When visual track is enabled |

## OS Host Service

The **OS Host** is a standalone macOS background service that provides system perception (UI events, screen capture, file system monitoring, app state) for the LeapFlow Brain. It runs as a self-contained `.app` bundle so privacy permissions bind to **LeapHost.app** itself — independent of Terminal, your IDE, or whichever shell launched the Brain. The Brain (Python) talks to it over a Unix Domain Socket.

### Quick Deploy

```bash
# One-shot: build, install as .app bundle, register login auto-start
leap host setup

# Or run the steps individually
make host-build          # compile release binary
make host-install        # package .app bundle into ~/.leapflow/host/
leap host start          # start the service
```

### Lifecycle

```bash
leap host start          # start
leap host stop           # graceful stop
leap host restart        # restart
leap host status         # PID, uptime, permission state
leap host logs           # tail recent logs
leap host logs --follow  # stream logs live
```

### Permissions

The Host needs two macOS privacy grants. Because it runs as its own `.app`, both appear under **LeapHost** in System Settings (not Terminal):

| Permission | Purpose |
|------------|---------|
| **Accessibility** | Read AXTree, dispatch UI actions |
| **Screen Recording** | Capture frames for the visual track |

`leap host setup` prompts and opens the relevant System Settings panes automatically the first time each permission is required.

### Auto-Start at Login

`leap host setup` registers a macOS **LaunchAgent** (`~/Library/LaunchAgents/com.leapflow.host.plist`) so the Host launches automatically when you log in. To remove it:

```bash
leap host uninstall      # unregister LaunchAgent and remove the .app bundle
```

### Development Mode

For active OS Host development with auto-rebuild on file changes:

```bash
leap host dev
# or
make host-dev
```

This watches `os_host/Sources/` for Swift file changes, automatically rebuilds (debug),
and restarts the host process. Press Ctrl+C to stop.

For one-off manual runs without packaging:

```bash
make host          # swift run (no .app packaging)
```

Or develop the Brain alone with no Host at all:

```bash
leap chat --mock-host --interactive   # pure-Python, in-memory mock
```

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `LEAPFLOW_DATA_DIR` | `~/.leapflow` | Root directory for Host bundle, sockets, logs |
| `LEAPFLOW_BRIDGE_SOCKET` | `~/.leapflow/var/host.sock` | Unix Domain Socket path |
| `LEAPFLOW_HOST_AUTO_START` | `1` | Whether the CLI auto-starts the Host on demand |

### Troubleshooting

| Symptom | Check |
|---------|-------|
| Host not running | `leap host status` → if stopped, `leap host start` |
| Permission denied (AX/Screen) | `leap host status` shows permission state → grant in System Settings |
| Socket connection failed | Verify `$LEAPFLOW_BRIDGE_SOCKET` exists and the Host PID is alive |
| Auto-start not firing | Confirm `~/Library/LaunchAgents/com.leapflow.host.plist` exists; reload via `leap host setup` |

## Configuration

Copy `.env.example` to `.env`. Only `LEAPFLOW_LLM_API_KEY` is required — all other values have sensible defaults.

```bash
cp .env.example .env
# Edit .env and set your API key
```

### LLM Provider

| Variable | Description | Default |
|----------|-------------|---------|
| `LEAPFLOW_LLM_API_KEY` | API key for your LLM provider | *(required)* |
| `LEAPFLOW_LLM_BASE_URL` | OpenAI-compatible base URL | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `LEAPFLOW_LLM_MODEL` | Model name | `qwen-plus` |
| `LEAPFLOW_LLM_MAX_RETRIES` | Max retries on LLM call failure | `3` |

The provider is **auto-detected** from `base_url`:

| URL contains | Detected provider |
|-------------|-------------------|
| `dashscope` | DashScope (Qwen) |
| `api.openai.com` | OpenAI |
| `api.deepseek.com` | DeepSeek |
| `api.groq.com` | Groq |
| `openai.azure.com` | Azure OpenAI |

Any OpenAI-compatible endpoint works — just set the appropriate `base_url`.

### Bridge & Host

| Variable | Description | Default |
|----------|-------------|---------|
| `LEAPFLOW_BRIDGE_SOCKET` | Unix socket path for Brain ↔ Host IPC | `/tmp/leapflow.sock` |
| `LEAPFLOW_MOCK_HOST` | `1` to use in-memory mock host (no Swift) | `0` |

### Storage & Logging

| Variable | Description | Default |
|----------|-------------|---------|
| `LEAPFLOW_DUCKDB_PATH` | DuckDB storage path | `~/.leapflow/memory.duckdb` |
| `LEAPFLOW_AUDIT_LOG_PATH` | Structured JSONL audit trail | `~/.leapflow/audit.jsonl` |
| `LEAPFLOW_LOG_LEVEL` | Log level (DEBUG, INFO, WARNING, ERROR) | `INFO` |

### Learning & Execution

| Variable | Description | Default |
|----------|-------------|---------|
| `LEAPFLOW_LEARN_IDLE_TIMEOUT` | Auto-stop learning after N seconds idle (0 = disabled) | `300` |
| `LEAPFLOW_LEARN_AUTO_DISTILL` | Auto-trigger distillation when learning stops | `true` |
| `LEAPFLOW_CONFIRM_DEFAULT_LEVEL` | Default confirmation level: `step` / `confirm` / `notify` / `auto` | `confirm` |

### Recording & Analysis Pipeline

| Variable | Description | Default |
|----------|-------------|---------|
| `LEAPFLOW_PATTERN_LIBRARY_PATH` | Custom patterns.yaml path (empty = bundled) | *(empty)* |
| `LEAPFLOW_SNAPSHOT_LEVEL_DEFAULT` | Recording fidelity: `full` / `light` / `minimal` | `light` |
| `LEAPFLOW_INTENT_INFERENCE_ENABLED` | Enable LLM-powered intent inference | `true` |
| `LEAPFLOW_INTENT_INFERENCE_LANGUAGE` | Language for intent prompts: `zh` / `en` | `zh` |

### Code Generation

| Variable | Description | Default |
|----------|-------------|---------|
| `LEAPFLOW_CODEGEN_SANDBOX` | AST safety validation (block `os.system`, etc.) | `true` |
| `LEAPFLOW_CODEGEN_MAX_RETRIES` | Max LLM retries for code generation | `2` |

### DAG Task Planning

| Variable | Description | Default |
|----------|-------------|---------|
| `LEAPFLOW_MAX_DAG_CONCURRENCY` | Max parallel nodes in DAG execution | `3` |
| `LEAPFLOW_DAG_NODE_TIMEOUT` | Per-node timeout in seconds | `300` |

### Visual Track

| Variable | Description | Default                      |
|----------|-------------|------------------------------|
| `LEAPFLOW_VISUAL_TRACK_ENABLED` | Enable visual (screen capture) track | `true`                       |
| `LEAPFLOW_VISUAL_FRAME_CACHE_DIR` | Directory for cached visual frames | `~/.leapflow/cache/frames` |
| `LEAPFLOW_VISUAL_SAMPLE_STRATEGY` | Sampling: `keyframe` / `periodic` / `all` | `keyframe`                   |
| `LEAPFLOW_VLM_MODEL` | VLM model for cross-modal verification | `qwen-vl-plus`               |
| `LEAPFLOW_VLM_API_KEY` | VLM API key (empty = reuse `LEAPFLOW_LLM_API_KEY`) | *(empty)*                    |
| `LEAPFLOW_PRIVACY_SENSITIVE_APPS` | Skip capture for these apps (comma-separated bundle IDs) | `com.apple.keychains`        |

### VLM Cost Optimization (P1–P4)

**P1 — Prefiltering** (skip VLM for high-confidence structural actions):

| Variable | Description | Default |
|----------|-------------|---------|
| `LEAPFLOW_VLM_PREFILTER_ENABLED` | Enable prefiltering | `true` |
| `LEAPFLOW_VLM_PREFILTER_SKIP_ACTIONS` | Action types to skip | `file.create,file.delete,...` |
| `LEAPFLOW_VLM_PREFILTER_CONFIDENCE_THRESHOLD` | Skip VLM above this confidence | `0.85` |

**P2 — Frame Result Cache** (avoid re-verifying identical frames):

| Variable | Description | Default |
|----------|-------------|---------|
| `LEAPFLOW_VLM_CACHE_ENABLED` | Enable result caching | `true` |
| `LEAPFLOW_VLM_CACHE_TTL` | Cache TTL in seconds | `300` |
| `LEAPFLOW_VLM_CACHE_MAX_SIZE` | Max cached entries | `1000` |

**P3 — Image Compression** (reduce latency and cost):

| Variable | Description | Default |
|----------|-------------|---------|
| `LEAPFLOW_VLM_COMPRESSION_ENABLED` | Enable compression | `true` |
| `LEAPFLOW_VLM_COMPRESSION_MAX_RESOLUTION` | Max long-edge in pixels | `1024` |
| `LEAPFLOW_VLM_COMPRESSION_QUALITY` | JPEG quality (1–100) | `75` |
| `LEAPFLOW_VLM_COMPRESSION_ADAPTIVE` | Adaptive quality by action type | `true` |

**P4 — Frame Tiling** (batch multiple frames in one VLM call):

| Variable | Description | Default |
|----------|-------------|---------|
| `LEAPFLOW_VLM_TILING_ENABLED` | Enable tiling | `false` |
| `LEAPFLOW_VLM_TILING_MAX_FRAMES` | Max frames per tiled image | `4` |
| `LEAPFLOW_VLM_TILING_TILE_SIZE` | Per-tile max long-edge in pixels | `384` |
| `LEAPFLOW_VLM_TILING_GAP` | Grid gap in pixels | `4` |

### Signal Fusion (Vision-Only Mode)

In `vision_only` mode, the VLM extraction pipeline relies solely on screen captures to reconstruct user actions. However, 2-second polling intervals miss intermediate interactions (e.g., right-click → copy → app switch appears as a single "screen changed" frame pair). **Signal Fusion** injects lightweight OS interaction signals (click coordinates, app switches, clipboard events, keyboard shortcuts) as temporal anchors into VLM prompts, dramatically improving action extraction accuracy.

| Variable | Description | Default |
|----------|-------------|---------|
| `LEAPFLOW_SIGNAL_CHANNELS` | Comma-separated signal channels to enable (see below) | *(empty = disabled)* |
| `LEAPFLOW_SIGNAL_REACTIVE_CAPTURE` | Trigger immediate frame capture on high-value signals | `false` |

**Available channels**: `click`, `app_switch`, `clipboard`, `clipboard_content`, `keyboard`, `scroll`, `drag`, or `all`.

- `clipboard` records only the event direction (copy/paste); `clipboard_content` additionally captures the text content (up to 200 chars)
- When `LEAPFLOW_SIGNAL_REACTIVE_CAPTURE=true`, a frame is captured immediately on click/app_switch/clipboard/drag signals rather than waiting for the next polling cycle

**Ablation Experiment Matrix** — incremental signal addition to quantify each channel's contribution:

| Experiment | `LEAPFLOW_SIGNAL_CHANNELS` | `LEAPFLOW_SIGNAL_REACTIVE_CAPTURE` | Description |
|------------|------------------------|-------------------------------|-------------|
| V0 | *(empty)* | `false` | Pure vision baseline (current `vision_only` behavior) |
| V1 | `click` | `false` | + click coordinates |
| V2 | `click` | `true` | + reactive capture on clicks |
| V3 | `click,app_switch` | `true` | + app transition signals |
| V4 | `click,app_switch,clipboard` | `true` | + clipboard direction |
| V5 | `click,app_switch,clipboard_content` | `true` | + clipboard text content |
| V6 | `all` | `true` | All signals + reactive capture |
| V7 | `all` | `false` | All signals, no reactive capture (isolates reactive contribution) |

**Experiment design rationale**:
- V0→V1 measures click coordinate contribution (VLM knows WHERE the user clicked)
- V1→V2 measures reactive capture value (frame captured AT the moment of interaction vs next poll)
- V2→V3 measures app switch signals (VLM knows WHEN app transitions occurred)
- V3→V4→V5 measures clipboard intelligence (direction only vs full content)
- V6 vs V7 isolates the reactive capture contribution with all signals active
- Each step adds exactly one variable, enabling clean attribution of accuracy gains

**Zero impact on default mode** — signal fusion only activates when `LEAPFLOW_RECORDING_MODE=vision_only` AND `LEAPFLOW_SIGNAL_CHANNELS` is set. The `default` mode recording pipeline is completely unaffected.

### Recommended Configurations

**Minimal** — just get started:

```bash
LEAPFLOW_LLM_API_KEY=sk-your-key-here
```

**Development** — mock host + verbose logging:

```bash
LEAPFLOW_LLM_API_KEY=sk-your-key-here
LEAPFLOW_MOCK_HOST=1
LEAPFLOW_LOG_LEVEL=DEBUG
```

**Production** — full mode with visual track and VLM:

```bash
LEAPFLOW_LLM_API_KEY=sk-your-key-here
LEAPFLOW_MOCK_HOST=0
LEAPFLOW_VISUAL_TRACK_ENABLED=true
LEAPFLOW_VLM_MODEL=qwen-vl-plus
LEAPFLOW_VLM_PREFILTER_ENABLED=true
LEAPFLOW_VLM_CACHE_ENABLED=true
LEAPFLOW_VLM_COMPRESSION_ENABLED=true
LEAPFLOW_LOG_LEVEL=INFO
```

### Verify Installation

```bash
# Run tests
uv run pytest tests/ -q

# Import check
uv run python -c "from leapflow.config import load_config; print('OK')"

# Smoke test — mock host (no Swift needed)
uv run python -m leapflow --mock-host --prompt "hello"

# Smoke test — full mode (requires Host running in another terminal)
./scripts/run.sh --prompt "hello"
```

## Quick Start

### Try It in 30 Seconds

No Swift build needed. Works on any platform with Python 3.11+:

```bash
# Mock mode (any platform)
uv run leap chat --mock-host --prompt "hello"
```

### Interactive REPL

```bash
# Mock mode (any platform)
uv run leap chat --mock-host --interactive

# Full mode (requires Host running)
uv run leap chat --interactive
```

In the REPL you can chat, learn, run skills, and manage skills — all in one session.

### Learn → Distill → Execute (End-to-End)

The core LeapFlow workflow in one continuous flow:

```bash
# Step 1: Start learning — tell the agent what you're about to demonstrate
leap learn --prompt "Organize PDF files"

# [Work normally — the agent observes silently]
# Available commands during learning:
#   annotate <text>  — highlight important steps
#   skip [n]         — mark last n steps as noise
#   learn pause      — pause recording
#   learn resume     — resume recording

# Step 2: Stop learning (triggers distillation automatically)
learn stop

# Output:
# [ LEARNING STOPPED ]
#   Trajectory: traj-abc123
#   Steps:      28
#   Duration:   135.2s
#   Events:     click=12, type=8, file_change=2
#
# [ LEARNING ... ]
#
# [ NEW SKILLS (1) ]
#   * Organize PDF Files
#       confidence: 72%
#       triggers:   organize pdf files, sort pdf
#       steps:      4
#
# [ STORED AT ]
#   Skills:  ~/.leapflow/skill_library.duckdb
#   Audit:   ~/.leapflow/audit.jsonl
#
# [ TRY IT ]
#   leap run "organize my PDF files"

# Step 3: Execute the learned skill
leap run "organize my PDF files"
# or explicitly by skill name:
leap run --skill "organize_pdf_files"
# or step-by-step for review:
leap run --skill "organize_pdf_files" --step
```

### Full Mode (macOS, Dual Terminal)

```bash
# Terminal 1 — start the Host
make host

# Terminal 2 — start the Brain (interactive)
make brain ARGS='--interactive'
```

The Host stays in the foreground, providing real-time perception. The Brain connects automatically via the Unix socket.

### Skill Management

```bash
leap skills list
leap skills show "categorize_pdfs"
leap skills export "categorize_pdfs" -o skill.json
leap skills import skill.json
leap skills disable "categorize_pdfs"
leap skills delete "categorize_pdfs"
```

### Mode Comparison

| Feature | Mock Mode | Full Mode |
|---------|-----------|-----------|
| Platform | Any (Python 3.11+) | macOS 14+ |
| Swift Host | Not needed | Required |
| System perception | Simulated | Real (AXTree, FSEvents, Clipboard) |
| Screen capture | Not available | Available (with permission) |
| File operations | In-memory mock | Real filesystem |
| Best for | Development, testing, learning | Production, real automation |
| Start command | `leap chat --mock-host` | `make host` + `make brain` |

## CLI

> See [Quick Start](#quick-start) for usage scenarios. Below is the command reference.
>
> Run via `uv run leap ...`, or activate the venv (`source .venv/bin/activate`) to use `leap` directly.

### Commands

```bash
# Chat (single-turn or interactive)
leap chat --prompt "..."              # single-turn
leap chat --interactive               # REPL
leap chat --mock-host --prompt "..."  # mock mode
leap chat --thinking --prompt "..."   # enable reasoning

# Learn
leap learn --prompt "goal description" [--timeout 600]

# Run (execute skills)
leap run "natural language trigger"
leap run --skill "skill_name" [--step]

# Skills management
leap skills list | show | export | import | disable | delete | audit
```

**Output when a skill matches** (transparent attribution):

```
$ leap run "organize my PDF downloads"

[ MATCHED SKILL ]
  Name:        categorize_pdfs (v2)
  Confidence:  85%  (matched trigger: "organize my PDF downloads")
  Description: Sort and categorize PDF files by content type
  Source:      learned
  Preconds:    ~/Downloads readable

Skill 'categorize_pdfs' executed successfully.
```

When no skill matches, LeapFlow falls back to the LLM agent (intent classified by `IntentClassifier`; no hard-coded keyword rules).

### Flags

| Flag | Description |
|------|-------------|
| `--prompt TEXT` | User instruction |
| `--mock-host` | Use in-memory mock host (overrides `LEAPFLOW_MOCK_HOST`) |
| `--thinking` | Enable LLM thinking/reasoning mode |
| `--interactive` | Persistent REPL session |
| `--timeout N` | Learning idle timeout in seconds (default: 300) |
| `--step` | Step-through execution (confirm each step) |
| `--skill NAME` | Run a specific skill by name |

### Interactive Mode Commands

```
learn start [goal]     — Start learning mode
learn stop             — Stop learning and distill
learn pause / resume   — Pause/resume recording
annotate <text>        — Add annotation during learning
skip [n]               — Mark last n steps as noise

run <trigger>          — Execute a skill by trigger phrase
skills list            — List registered skills
skills show <name>     — Show skill details
skills disable <name>  — Disable a learned skill
skills delete <name>   — Permanently delete a learned skill
skills audit [name] [--limit N] — View skill execution history (timeline of confirmations, executions, failures)

help                   — Show all commands
exit / quit            — End session
```

**Realtime feedback during learning & execution:**

- Distillation progress streamed inline: `[segmentation] 15/50`, `[abstraction] 30/50`, ...
- Learning completion summary: `[LeapFlow] Learning complete — 2 new skill(s)`
- Step-by-step execution progress: `  [1/4] Open ~/Downloads folder`

## Project Structure

```
leapflow/
├── src/leapflow/              # Python Brain (layered architecture)
│   ├── domain/                  # L0 — Shared types (zero internal deps)
│   │   ├── trajectory.py        #   Trajectory, Episode, SemanticAction, RawAction, etc.
│   │   ├── events.py            #   SystemEvent, UINode, PerceptionPort, ExecutionPort
│   │   ├── platform.py          #   PlatformID, Capability, PlatformManifest
│   │   └── skill_types.py       #   DistillationCandidate, SkillParameter, SkillMetadata
│   ├── storage/                 # L1 — Persistence
│   │   ├── trajectory_store.py  #   DuckDB trajectory store (fault-tolerant write buffering)
│   │   ├── skill_library.py     #   DuckDB skill persistence + update suggestions
│   │   ├── skill_docs.py        #   Filesystem SKILL.md store + LLM-backed execution
│   │   └── session_store.py     #   Learning session state persistence
│   ├── platform/                # L2 — RPC + platform adaptation
│   │   ├── client.py            #   BridgeClient (Unix Socket, auto-reconnect)
│   │   ├── protocol.py          #   MsgPack-RPC framing + method constants
│   │   ├── event_bus.py         #   Event routing → Normalizer → Memory
│   │   ├── mock.py              #   In-memory mock bridge
│   │   ├── facade.py            #   VirtualSystemInterface — platform handshake
│   │   ├── normalizer.py        #   Platform events → SystemEvent
│   │   ├── relevance.py         #   Platform-aware relevance scoring
│   │   └── adapters/            #   Darwin / Mock adapters
│   ├── recording/               # L3 — Real-time recording + attention
│   │   ├── recorder.py          #   DemonstrationRecorder (EventBus subscriber)
│   │   └── attention.py         #   Attention filters (whitelist, noise, working-dir)
│   ├── perception/              # L3.5 — Visual perception (video-first)
│   │   ├── session.py           #   PerceptionSession lifecycle management
│   │   ├── config.py            #   Perception configuration
│   │   ├── types.py             #   VideoSegment, TimelineMarker, VideoAction, etc.
│   │   └── video/               #   Video-first recording + multi-scale VLM analysis
│   │       ├── recorder.py      #     VideoRecorder — Host RPC video capture lifecycle
│   │       ├── timeline.py      #     SignalTimeline — event markers for VLM context
│   │       ├── segmenter.py     #     VideoSegmenter — semantic video splitting
│   │       └── analyzer.py      #     VideoAnalyzer — L1/L2/L3 progressive VLM analysis
│   ├── analysis/                # L4 — Offline analysis pipeline
│   │   ├── pipeline.py          #   Orchestrator: record → segment → abstract → distill
│   │   ├── synthesis.py         #   PlatformSynthesisPass — rename pairs, batch ops, noise
│   │   ├── abstractor.py        #   Multi-level action abstraction (L0→L1→L2)
│   │   ├── segmenter.py         #   Episode segmentation (time/app/semantic boundaries)
│   │   ├── denoise.py           #   DenoisePass — undo collapse, idempotent merge
│   │   ├── intent_inferrer.py   #   LLM-powered intent inference for episodes
│   │   ├── consensus.py         #   MultiTrajectoryDistiller — cross-trajectory LCS
│   │   ├── causal.py            #   CausalChainAnalyzer — extract causally relevant steps
│   │   ├── patterns.py          #   YAML-driven pattern matching engine
│   │   └── patterns.yaml        #   Extensible action pattern library
│   ├── learning/                # L5 — Skill learning chain
│   │   ├── distiller.py         #   Heuristic + LLM skill extraction
│   │   ├── active_learning.py   #   Similarity detection + merge/evolve decisions
│   │   ├── codegen.py           #   LLM + template code generation (AST-validated)
│   │   ├── doc_generator.py     #   SKILL.md document generation
│   │   ├── document.py          #   Skill document parser/renderer
│   │   ├── feedback.py          #   Structural diff + auto-improvement loop
│   │   └── similarity.py        #   Heuristic + LLM similarity scoring
│   ├── skills/                  # L6 — Skill runtime
│   │   ├── registry.py          #   SkillRegistry (OCP, runtime registration)
│   │   ├── activator.py         #   Code compilation + port binding + registry
│   │   ├── tool_executor.py     #   ReAct tool-use executor for SKILL.md
│   │   └── builtin/             #   Built-in skills (file organizer, clipboard, app launcher)
│   ├── engine/                  # L8 — Orchestration + scheduling
│   │   ├── session.py           #   SessionController — LEARN↔DISTILL↔EXECUTE state machine
│   │   ├── engine.py            #   AgentEngine — ReAct loop + skill registry assembly
│   │   ├── confirmation.py      #   ConfirmationHandler — graduated human-in-the-loop
│   │   ├── intent_classifier.py #   Two-stage routing (keyword + LLM)
│   │   ├── graph_planner.py     #   LLM-driven DAG task planning
│   │   ├── scheduler.py         #   DAG concurrent execution
│   │   ├── audit.py             #   AuditLogger — structured JSONL event logging
│   │   └── terminal_io.py       #   Terminal I/O provider
│   ├── cli/                     # L10 — CLI dispatch
│   │   └── app.py               #   Full CLI implementation (learn, run, skills, chat)
│   ├── memory/                  # Cross-cutting — Three-tier event-driven memory
│   │   ├── immediate.py         #   TTL fragments + GC + promotion
│   │   ├── working_memory.py    #   Ring buffer (LLM context window)
│   │   ├── long_term.py         #   DuckDB persistent store
│   │   └── decay.py             #   W = S·e^(-λt)·(1+ln F)
│   ├── llm/                     # Cross-cutting — LLM abstraction
│   │   ├── base.py              #   LLMProvider ABC
│   │   ├── openai_provider.py   #   Multi-provider auto-detection
│   │   └── message_builder.py   #   Multimodal message construction
│   ├── config.py                # Settings from .env
│   └── __main__.py              # Entry point → cli.app.main()
├── os_host/                     # Swift Host (macOS)
│   ├── Package.swift
│   └── Sources/OSHost/
│       ├── main.swift           #   Entry point
│       ├── Bridge/              #   SocketServer + MessageCodec (MsgPack)
│       ├── Perception/          #   AXTree, Clipboard, FSEvents, UIAction observers
│       ├── Execution/           #   File ops, App control, Shell
│       ├── Platform/            #   Legacy/Tahoe provider factory + capability detection
│       └── Security/            #   PermissionGuard + AuditLog
├── tests/                       # Pytest suite (610+ tests)
├── scripts/
│   ├── setup.sh                 #   One-line install
│   └── run.sh                   #   Quick-start wrapper
├── pyproject.toml               #   Project config (hatchling)
└── Makefile                     #   Build shortcuts
```

### Layer Dependency Rules

Dependencies flow strictly downward. Each layer may only import from layers below it:

```
cli (L10) → engine (L8) → skills (L6) → learning (L5) → analysis (L4) → recording (L3) → platform (L2) → storage (L1) → domain (L0)
```

Cross-cutting modules (`memory/`, `llm/`, `prompts/`) are available to all layers.

## Development

After [Installation](#installation), use the Makefile for common workflows:

```bash
make test     # uv run pytest tests/ -q
make lint     # uv run ruff check src/leapflow/ tests/
make host     # build + run Swift Host
make brain ARGS='--prompt "hello"'  # run Brain
```

## Production Readiness

### Audit Logging

All mode transitions, skill executions, and learning results are written to a structured JSONL audit log (`~/.leapflow/audit.jsonl`, configurable via `LEAPFLOW_AUDIT_LOG_PATH`):

```jsonl
{"ts":1747468200.0,"event":"mode.learning.start","session":"a1b2c3","goal":"organize downloads"}
{"ts":1747468320.0,"event":"mode.learning.stop","session":"a1b2c3","steps":23,"duration":120.0}
{"ts":1747468321.0,"event":"mode.learning.done","session":"a1b2c3","candidates":2,"new_skills":1}
{"ts":1747468500.0,"event":"skill.execute","session":"","skill":"classify_pdf","level":"confirm"}
```

### Fault Tolerance

| Scenario | Strategy |
|----------|----------|
| Learning: host disconnects | Auto-pause recording, attempt reconnect, save partial on timeout |
| Learning: LLM call fails | Degrade to heuristic path (lower confidence, still functional) |
| Learning: background exception | `finally` clause guarantees return to IDLE mode |
| Execution: step fails | Report error, user chooses continue/abort; `undo_last()` for rollback |
| DuckDB: write failure | In-memory buffer (cap 500), retry on next successful write |
| Concurrent mode transitions | `asyncio.Lock` prevents race conditions |

### Skill Version Management

Each `StoredSkill` tracks: `version` (incremented on merge/improve), `confidence` (weighted sliding average), `status` (active / deprecated / disabled / deleted), and execution history. Skills graduate from STEP confirmation to AUTO as they prove reliable — and regress back to CONFIRM if recent executions show regressions.

## Roadmap

- **M1 (Complete)**: Structural recording — AXTree + EventBus trajectory capture, episode segmentation, multi-level action abstraction, heuristic skill distillation, session orchestration, human-in-the-loop confirmation, audit logging
- **M2 (Complete)**: Analysis & distillation — LLM-powered intent inference, code generation for executable skills, active learning with feedback loops, multi-trajectory consensus distillation
- **M3 (Complete)**: Cross-modal visual track — ScreenCapture + VLM verification, dual-track recording (structural + visual), cross-modal timestamp alignment, VLM cost optimization P1–P4 (prefiltering, caching, compression, tiling)
- **Beyond**: Linux host implementation, workflow composition, skill marketplace, distributed skill sharing

## License

Apache 2.0 — see [LICENSE](LICENSE).
