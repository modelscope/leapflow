# LeapFlow

**A signal-driven, self-evolving agent framework that learns autonomously from the real world.**

---

## What is LeapFlow?

LeapFlow is a **general-purpose intelligent agent framework** designed around a single conviction: agents should learn the way humans do — by observing the world, forming causal understanding, and continuously refining their skills through practice.

Unlike instruction-driven agents (Computer-Use, RPA) that reason from scratch on every request, LeapFlow **accumulates knowledge across episodes**. It perceives multi-modal signals from the operating environment, distills reusable skills from demonstrations, and self-improves every time those skills are executed. The result is an agent that gets smarter the more you use it.

**LeapFlow is not another desktop automation tool.** Where RPA replays brittle scripts and Computer-Use agents burn tokens re-deriving every action, LeapFlow builds a persistent, evolving cognitive model — fusing perception, causal reasoning, world modeling, and skill synthesis into a self-reinforcing learning loop.

## Core Philosophy

- **Evolution over Instruction** — Learning is not a one-shot prompt; it is a continuous loop of observation, hypothesis, verification, and refinement across episodes.
- **Signals as First-Class Citizens** — Multi-modal signals (visual, accessibility tree, file system, clipboard, keyboard, etc.) are fused into a unified causal timeline, not treated as isolated events.
- **Persistent Knowledge** — Skills, world-model experiences, and causal patterns are durably stored and compound over time. Nothing learned is ever lost to a session boundary.
- **Trust Gradient** — New skills start under full human supervision (`STEP`) and progressively earn autonomy (`CONFIRM → NOTIFY → AUTO`) by proving competence through successful executions.
- **Prediction-Error-Driven Learning** — The world model predicts outcomes before execution and learns from the delta between prediction and reality — mirroring predictive coding in cognitive neuroscience.
- **Safety as Architecture** — Tiered autonomy, sandbox verification, and reversibility checks are structural guarantees, not bolt-on constraints.

## Architecture Overview

LeapFlow implements a layered cognitive pipeline:

```
┌───────────────────────────────────────────────────────────┐
│  Copilot           Workflow-level next-step prediction    │
├───────────────────────────────────────────────────────────┤
│  World Model       State prediction · Experience replay   │
├───────────────────────────────────────────────────────────┤
│  Skill Synthesis   Hypothesis → Draft → Verified → Prod  │
├───────────────────────────────────────────────────────────┤
│  Causal Engine     Rule · Heuristic · VLM verification    │
├───────────────────────────────────────────────────────────┤
│  Perception        Multi-channel signal fusion (7 ch)     │
└───────────────────────────────────────────────────────────┘
```

**Perception** fuses raw signals into a causal timeline. The **Causal Engine** infers why things happened, not just what. The **World Model** builds an internal representation of the environment and learns from prediction errors. **Skill Synthesis** distills observations into parameterized, reusable skills with maturity tracking. The **Copilot** predicts your next workflow step and offers proactive suggestions — like GitHub Copilot, but for everything you do on your computer.

---

## Prerequisites

| Component | Version | Purpose |
|-----------|---------|---------|
| Python | ≥ 3.11 | Runtime (3.11–3.14 supported) |
| [uv](https://github.com/astral-sh/uv) | latest | Fast package manager & virtualenv |
| macOS | 14.0+ (Sonoma) | Required for native OS Host perception |
| Xcode Command Line Tools | latest | Swift compiler for OS Host build |
| LLM API Key | — | DashScope, OpenAI, DeepSeek, or any OpenAI-compatible provider |

> **Note:** You can run LeapFlow on any platform with `--mock-host` (no native perception), but full signal capture requires macOS with Accessibility permissions.

## Installation

### 1. Clone & Setup

```bash
git clone https://github.com/modelscope/leapflow.git
cd leapflow
make setup
```

`make setup` handles everything: creates a virtualenv via `uv`, installs all dependencies, and generates a `.env` file from the template.

<details>
<summary>Manual steps (if you prefer)</summary>

```bash
uv sync --all-extras          # Install Python deps
cp .env.example .env          # Create config file
```
</details>

### 2. Configure Your LLM

Edit `.env` — only one field is required:

```bash
LEAPFLOW_LLM_API_KEY=sk-your-key-here
```

Defaults point to DashScope (Qwen). To use a different provider:

```bash
LEAPFLOW_LLM_BASE_URL=https://api.openai.com/v1
LEAPFLOW_LLM_MODEL=gpt-4o
```

### 3. Build OS Host (Optional — macOS only)

The native OS Host captures screen recordings, accessibility trees, and input events. Skip this step if you just want to explore with `--mock-host`.

```bash
make swift-build              # Debug build
```

This compiles the Swift host binary to `os_host/darwin/.build/debug/OSHost`.

For production deployment:

```bash
make host-install             # Release build + .app bundle → ~/.leapflow/host/
```

### 4. Verify Installation

```bash
uv run leap --mock-host "hello, are you ready?"
```

> Expected: LeapFlow responds with a greeting confirming it's operational.

---

## Configuration Reference

The `.env` file lives in your project root (or `~/.leapflow/.env` for global defaults). Key variables:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LEAPFLOW_LLM_API_KEY` | **Yes** | — | Your LLM provider API key |
| `LEAPFLOW_LLM_BASE_URL` | No | DashScope endpoint | OpenAI-compatible base URL |
| `LEAPFLOW_LLM_MODEL` | No | `qwen3.7-plus` | Model identifier |
| `LEAPFLOW_MOCK_HOST` | No | `0` | Set `1` to skip native Host |
| `LEAPFLOW_RECORDING_MODE` | No | `video` | `video` / `default` / `vision_only` |
| `LEAPFLOW_LOG_LEVEL` | No | `INFO` | `DEBUG` / `INFO` / `WARNING` |
| `LEAPFLOW_DUCKDB_PATH` | No | `~/.leapflow/memory.duckdb` | Persistent storage location |
| `LEAPFLOW_DATA_DIR` | No | `~/.leapflow` | Root data directory |

See [`.env.example`](.env.example) for the complete reference (50+ tunable parameters).

---

## Quick Start — First Experience

### Step 1: Launch Interactive Mode

```bash
uv run leap --mock-host
```

> You'll see the LeapFlow banner and a `>` prompt — you're in the interactive REPL.

### Step 2: Have a Conversation

```
> What can you help me with?
```

> LeapFlow explains its capabilities: task execution, skill learning, workflow automation.

### Step 3: Teach a Skill (Learn Mode)

Open a new terminal and start a learning session:

```bash
uv run leap learn "organize screenshots by date"
```

> LeapFlow begins observing your actions (screen recording + event capture). Work normally — move files, rename, create folders. When done:

```
> stop
```

> LeapFlow distills your demonstration into a reusable skill with confidence scoring.

### Step 4: Execute a Learned Skill

```bash
uv run leap run "organize my screenshots"
```

> LeapFlow matches your request to the learned skill and executes it. Each successful run increases skill confidence.

### Step 5: Manage Your Skills

```bash
uv run leap skills list              # View all learned skills
uv run leap skills show "skill-name" # Inspect a specific skill
```

---

## Usage Patterns

### Interactive Mode — Conversational Agent

```bash
uv run leap                          # Enter REPL
uv run leap "summarize this PDF"     # Single-turn (answer + exit)
```

The REPL supports multi-turn conversation with tool use, memory, and real-time streaming.

### Teach Mode — Learn from Demonstration

```bash
uv run leap learn "describe what you'll demonstrate"
```

LeapFlow records your actions as a trajectory, then distills them into a parameterized skill. The skill progresses through maturity tiers: `DRAFT → VERIFIED → PRODUCTION`.

Options:
- `--timeout 600` — Custom idle timeout (seconds) before auto-stopping
- `--field "Safari:browsing:full"` — Per-app perception rules

### Autonomous Mode — Execute Learned Skills

```bash
uv run leap run "trigger phrase"         # Match by natural language
uv run leap run --skill "exact-name"     # Match by skill name
uv run leap run --step "careful task"    # Step-through with confirmation
uv run leap run --auto "routine task"    # Skip confirmations
```

Skills start at `STEP` tier (human confirms each action) and graduate to `AUTO` as confidence grows.

---

## OS Host Management

For full perception (screen capture, accessibility tree, input events), you need the native OS Host running:

```bash
# Development (foreground, debug build)
make host                        # Terminal 1: builds + runs OS Host
uv run leap                      # Terminal 2: interactive REPL

# Production (daemon mode)
uv run leap host setup           # Build, install, register as LaunchAgent
uv run leap host start           # Start the daemon
uv run leap host status          # Check if running
uv run leap host stop            # Stop gracefully
```

> **Important:** macOS will prompt for Accessibility and Screen Recording permissions on first launch. Grant both in System Settings → Privacy & Security.

---

## Development

```bash
make setup            # Initialize environment
make test             # Run tests (pytest)
make lint             # Lint (ruff)
make swift-build      # Build Swift Host (debug)
make host-build       # Build Swift Host (release)
make host-dev         # Auto-rebuild on source changes
```

## Key Modules

| Module | Role |
|--------|------|
| `perception/` | Multi-channel signal capture and fusion (video, AX tree, clipboard, keyboard, file system, etc.) |
| `signal_fusion/` | Cross-modal temporal alignment and surprise detection |
| `causal/` | Three-tier causal inference engine (rule → heuristic → VLM) |
| `world_model/` | Predictive coding loop, experience store, curiosity-driven learning |
| `learning/` | Skill distillation, parameterization, and maturity lifecycle |
| `skills/` | Skill library, runtime execution, and self-evolution (Loop γ) |
| `copilot/` | Workflow-level next-step prediction and proactive suggestion |
| `analysis/` | Six-layer denoising pipeline for trajectory refinement |
| `engine/` | Session orchestration and ReAct execution loop |
| `memory/` | Three-tier event-driven memory (working → episodic → long-term) |
| `platform/` | Platform adaptation layer and RPC bridge |
| `os_host/` | Native host service — macOS (Swift), Linux & Windows (planned) |

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `OS Host connection failed` | Host not running or socket mismatch | Run `make host` in another terminal, or use `--mock-host` |
| `LEAPFLOW_LLM_API_KEY is empty` | Missing API key | Set `LEAPFLOW_LLM_API_KEY` in `.env` |
| `Accessibility permission denied` | macOS privacy gate | System Settings → Privacy & Security → Accessibility → enable LeapHost |
| `Screen Recording blocked` | macOS privacy gate | System Settings → Privacy & Security → Screen Recording → enable LeapHost |
| `swiftc: command not found` | Xcode CLT missing | `xcode-select --install` |
| Host builds but crashes | SDK version mismatch | Ensure macOS 14+ and latest Xcode CLT |

---

## License

Apache 2.0 — see [LICENSE](LICENSE).
