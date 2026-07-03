# LeapFlow — Development Guide

Instructions for AI coding assistants working on the LeapFlow codebase.

## Project Overview

LeapFlow is a **signal-driven, self-evolving agent framework** that learns autonomously from real-world observation. Unlike instruction-driven agents that reason from scratch every time, LeapFlow accumulates knowledge across episodes through a continuous perception → reasoning → skill-synthesis loop.

### Design Philosophy

1. **Signal-Driven** — All intelligence derives from real-world signal observation, not hardcoded rules
2. **Context Pipeline** — Signal → Filter → Compress → Store → Retrieve → Decide
3. **Progressive Trust** — Skills earn autonomy: `DRAFT → CANDIDATE → VERIFIED → PRODUCTION → AUTO`
4. **LLM-Native** — Protocols and abstractions designed for LLM-first reasoning
5. **Dual-Tier Execution** — Local scheduler + Cloud studio for long-horizon tasks

## Architecture

```
Copilot          Workflow-level next-step prediction (L0–L3 cascade)
World Model      State prediction · Experience replay · Curiosity
Skill Synthesis  Observation → Distillation → Verification → Production
Causal Engine    Rule → Heuristic → VLM (three-tier inference)
Perception       Multi-channel signal fusion (7 channels)
```

All layers communicate through an **EventBus** with typed domain events. The system uses **Protocol-based DI** — core logic depends on protocols, never concrete implementations.

## Key Modules

| Module | Path | Responsibility |
|--------|------|----------------|
| `perception` | `src/leapflow/perception/` | Multi-channel signal capture (video, AX tree, clipboard, keyboard, file system) |
| `signal_fusion` | `src/leapflow/signal_fusion/` | Cross-modal temporal alignment and surprise detection |
| `causal` | `src/leapflow/causal/` | Three-tier causal inference (rule → heuristic → VLM) |
| `world_model` | `src/leapflow/world_model/` | Predictive coding loop, experience store, curiosity-driven learning |
| `learning` | `src/leapflow/learning/` | Skill distillation, parameterization, learnability assessment |
| `skills` | `src/leapflow/skills/` | Skill library, execution, maturity lifecycle |
| `copilot` | `src/leapflow/copilot/` | Speculative L0–L3 prediction cascade, idle detection, feedback loop |
| `engine` | `src/leapflow/engine/` | Session orchestration and ReAct execution loop |
| `memory` | `src/leapflow/memory/` | Three-tier memory: working → episodic → long-term |
| `platform` | `src/leapflow/platform/` | RPC bridge (msgpack over Unix socket), platform abstraction |
| `recording` | `src/leapflow/recording/` | Video recording orchestration and segmentation |
| `analysis` | `src/leapflow/analysis/` | Six-layer trajectory denoising pipeline |
| `storage` | `src/leapflow/storage/` | DuckDB-backed persistence for skills, trajectories, audit |
| `domain` | `src/leapflow/domain/` | Shared types, events, perception models |
| `hub` | `src/leapflow/hub/` | Skill marketplace with multi-backend (ModelScope, GitHub, local) |
| `os_host` | `os_host/darwin/` | Native macOS host (Swift): screen capture, AX tree, input events |

## Code Quality Requirements

- Code must follow SOLID principles
- Apply Occam's Razor — maximize elegance and efficiency
- Design for generalization and universality
- Easy to extend, avoid hardcoding and hard rules
- Industrial-grade robustness

## Development Conventions

| Aspect | Standard |
|--------|----------|
| Language | Python 3.11+ |
| Package manager | `uv` |
| Native host | Swift (macOS), planned: Rust (Linux/Windows) |
| Persistence | DuckDB |
| Concurrency | `asyncio` |
| DI pattern | Protocol-based (no concrete deps in core) |
| Comments & docstrings | English only |
| Linting | `ruff` |
| Testing | `pytest` |
| Project layout | `src/` layout (`src/leapflow/`) |
| Env config | `.env` file (secrets + runtime tuning) |

### Key Conventions

- **Protocols over inheritance** — Define behavior contracts via `typing.Protocol`, not ABC hierarchies
- **Event-driven communication** — Modules interact through typed events on EventBus, not direct imports
- **Immutable domain types** — Use `@dataclass(frozen=True)` or `NamedTuple` for domain objects
- **No placeholder comments** — Never leave `# TODO: implement` stubs; implement or don't add
- **English comments only** — All code comments, docstrings, and log messages in English
- **ANSI output must check TTY** — Use `sys.stdout.isatty()` before emitting escape codes

## CLI Commands

```bash
uv run leap                          # Interactive REPL
uv run leap "question"               # Single-turn chat
uv run leap teach "goal"             # Record demonstration → distill skill
uv run leap run "trigger"            # Execute a matched skill
uv run leap skills list              # Manage skill library
uv run leap host start|stop|status   # OS Host lifecycle
```

Global flags: `--mock-host` (skip native perception), `--thinking` (extended reasoning).

## Testing

```bash
make test                                        # Full suite
uv run pytest tests/ -q                          # Direct pytest
uv run pytest tests/test_pure_algorithms.py -q   # Single file
uv run pytest -k "test_world_model" -q           # By keyword
```

Tests must not require network access or native host. Use `--mock-host` patterns for integration tests.

## Build Commands

```bash
make setup          # Initialize environment (uv sync + .env)
make test           # Run tests
make lint           # Lint (ruff)
make swift-build    # Build Swift Host (debug)
make host-build     # Build Swift Host (release)
make host-install   # Release build + .app bundle → ~/.leapflow/host/
make host-dev       # Auto-rebuild on source changes
```

## Project Structure

```
leapflow/
├── src/leapflow/          # Python brain (src layout)
│   ├── cli/               # CLI entry + subcommands
│   ├── copilot/           # Workflow Copilot (L0–L3 predictors)
│   ├── engine/            # Session + ReAct execution loop
│   ├── perception/        # Signal channels + fusion
│   ├── signal_fusion/     # Cross-modal temporal fusion
│   ├── causal/            # Causal inference pipeline
│   ├── world_model/       # Predictive coding + experience store
│   ├── learning/          # Skill distillation + assessment
│   ├── skills/            # Skill library + execution
│   ├── memory/            # Three-tier memory system
│   ├── recording/         # Video recording orchestration
│   ├── platform/          # RPC bridge + platform layer
│   ├── storage/           # DuckDB persistence
│   └── domain/            # Shared types & events
├── os_host/darwin/        # Native macOS Host (Swift)
├── tests/                 # Pytest suite
├── docs/design/           # Design documents
└── scripts/               # Setup & run scripts
```
