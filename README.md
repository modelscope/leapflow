# LeapFlow

**A desktop automation framework that learns and evolves from real user operations**

LeapFlow observes your daily interactions with the computer and automatically distills operation demonstrations into reusable, parameterized skills. Skills continuously evolve with each execution — the more you use them, the smarter they get.

## Core Features

- **Zero-Intrusion Recording** — Observes in the background without disrupting normal operations
- **Six-Layer Noise Filtering** — Distills 4-5 clean skill steps from 50 noisy recorded steps
- **Progressive Trust** — New skills require step-by-step confirmation; mature skills execute autonomously (STEP → CONFIRM → AUTO)
- **Video-First Perception** — Continuous video recording + VLM multi-scale analysis for precise intent reconstruction
- **Workflow Copilot** — Next-step prediction and proactive suggestions powered by a world model

## Quick Start

### Requirements

| Component | Version | Notes |
|------|------|------|
| Python | 3.11+ | Required |
| [uv](https://github.com/astral-sh/uv) | latest | Package manager |
| macOS | 14+ | Native Host (optional, `--mock-host` bypasses this) |

### Installation

```bash
git clone https://github.com/modelscope/leapflow.git
cd leapflow
make setup
```

`make setup` automatically: creates a virtual environment, installs dependencies, and generates the `.env` configuration file.

### Configuration

Edit `.env` to set your LLM API Key (the only required field):

```bash
LEAPFLOW_LLM_API_KEY=sk-your-key-here
# Defaults to DashScope (qwen3.7-plus), supports any OpenAI-compatible endpoint
# LEAPFLOW_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
# LEAPFLOW_LLM_MODEL=qwen3.7-plus
```

### Launch

```bash
# Mock mode (any platform, no Swift Host required)
uv run leap --mock-host

# Full mode (macOS, Host must be started first)
make host          # Terminal 1: Start the native Host
uv run leap        # Terminal 2: Enter the interactive REPL
```

## Usage

### Command Overview

```bash
leap                    # Interactive REPL (default)
leap "your question"    # Single-turn conversation
leap learn              # Record an operation demonstration
leap run "organize PDF" # Trigger skill execution
leap skills list        # List learned skills
leap host start/stop    # Manage the OS Host service
```

### Interactive Mode

The following commands are available in the REPL:

```
learn start [goal]    — Start recording
learn stop           — Stop recording and distill
annotate <text>      — Annotate the current step
skip [n]             — Mark noisy steps
run <trigger>        — Execute a skill
skills list/show     — Manage skills
help                 — Show all commands
exit                 — Exit
```

### Typical Workflow: Record → Distill → Execute

```bash
# 1. Record a demonstration
> learn start organize PDF files in Downloads

# [Perform operations normally, LeapFlow observes in the background...]

# 2. Stop recording (auto-distills)
> learn stop
# → New skill "Organize PDF Files" is ready (confidence: 72%)

# 3. Trigger directly next time
> run organize my PDFs
# → Executing "Organize PDF Files", done.
```

## OS Host Service (macOS)

The OS Host provides native system perception capabilities (AXTree, screen recording, file monitoring):

```bash
leap host setup      # Build + install + register for auto-start
leap host start      # Start
leap host stop       # Stop
leap host status     # Check status
```

First-time use requires granting Accessibility and Screen Recording permissions in **System Settings → Privacy & Security**.

## Project Structure

```
leapflow/
├── src/leapflow/           # Python Brain
│   ├── cli/                  # CLI entry point (leap command)
│   ├── copilot/              # Workflow Copilot (next-step prediction)
│   ├── domain/               # Domain type definitions
│   ├── engine/               # Session orchestration & ReAct engine
│   ├── recording/            # Real-time recording
│   ├── perception/           # Video perception & VLM analysis
│   ├── analysis/             # Offline analysis pipeline
│   ├── learning/             # Skill distillation & code generation
│   ├── skills/               # Skill runtime & registry
│   ├── platform/             # Platform adaptation layer (RPC Bridge)
│   ├── memory/               # Three-tier event-driven memory
│   ├── world_model/          # World model & predictive coding
│   ├── causal/               # Causal reasoning engine
│   ├── signal_fusion/        # Multi-modal signal fusion
│   └── llm/                  # LLM Provider abstraction
├── os_host/                # Native Host (cross-platform)
│   ├── darwin/               # macOS implementation (Swift)
│   ├── linux/                # Linux (planned)
│   └── windows/              # Windows (planned)
├── tests/                  # Test suite
├── Makefile                # Build shortcuts
└── pyproject.toml          # Project configuration
```

## Development

```bash
make setup            # Initialize environment
make test             # Run tests (pytest)
make lint             # Code linting (ruff)
make host             # Build and run Swift Host (debug)
make brain ARGS='--mock-host --prompt "hello"'  # Run Brain
```

### Common Environment Variables

| Variable | Default | Description |
|------|--------|------|
| `LEAPFLOW_LLM_API_KEY` | — | LLM API Key (required) |
| `LEAPFLOW_LLM_BASE_URL` | DashScope | OpenAI-compatible endpoint |
| `LEAPFLOW_LLM_MODEL` | `qwen3.7-plus` | Model name |
| `LEAPFLOW_MOCK_HOST` | `0` | `1` to enable Mock mode |
| `LEAPFLOW_RECORDING_MODE` | `video` | Recording mode: video / default / vision_only |
| `LEAPFLOW_LOG_LEVEL` | `INFO` | Log level |

See [`.env.example`](.env.example) for the full configuration reference.

## License

Apache 2.0 — see [LICENSE](LICENSE).
