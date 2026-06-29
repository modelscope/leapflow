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

## Quick Start

### Prerequisites

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.11+ | Required |
| [uv](https://github.com/astral-sh/uv) | latest | Package manager |
| macOS | 14+ | Native Host (optional — `--mock-host` bypasses this) |

### Install

```bash
git clone https://github.com/modelscope/leapflow.git
cd leapflow
make setup        # Creates venv, installs deps, generates .env
```

### Configure

Edit `.env` — the only required field is your LLM API key:

```bash
LEAPFLOW_LLM_API_KEY=sk-your-key-here
# Defaults to DashScope (qwen3.7-plus); any OpenAI-compatible endpoint works.
# LEAPFLOW_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
# LEAPFLOW_LLM_MODEL=qwen3.7-plus
```

See [`.env.example`](.env.example) for the full configuration reference.

### Run

```bash
# Mock mode — works on any platform, no native Host required
uv run leap --mock-host

# Full mode (macOS) — start the native Host first
make host          # Terminal 1: build & launch OS Host
uv run leap        # Terminal 2: interactive REPL
```

### A Taste of the Loop: Record → Distill → Execute → Evolve

```bash
> learn start organize PDF files in Downloads
# [Work normally — LeapFlow observes in the background]

> learn stop
# → Skill "Organize PDF Files" distilled (confidence: 72%, tier: DRAFT)

> run organize my PDFs
# → Executing… done. Skill confidence rises to 81%.

# Each execution feeds back into the skill — it literally gets better every time.
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

## Development

```bash
make setup            # Initialize environment
make test             # Run tests (pytest)
make lint             # Lint (ruff)
make host             # Build & run Swift Host (debug)
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
