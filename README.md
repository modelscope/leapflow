# LeapFlow

**A signal-driven, self-evolving agent framework that learns autonomously from the real world.**


### News

- **2025-06-30**: LeapFlow Preview released — initial public release with record & replay, multi-modal signal fusion, and Workflow Copilot.

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

LeapFlow implements a **three-layer hybrid architecture** — a Python intelligence core, a protocol-driven platform adaptation layer, and pluggable execution backends — communicating via the MCP (Model Control Protocol) standard:

```
┌─────────────────────────────────────────────────────────┐
│  Intelligence Core (Python)                             │
│  ├── Engine / OODA Loop + Learning + Copilot           │
│  ├── Signal Fusion → Causal Engine → World Model       │
│  └── Skill Synthesis + Memory System                   │
├─────────────────────────────────────────────────────────┤
│  Platform Adaptation Layer                              │
│  ├── Protocol Client (MCP stdio / WebSocket / gRPC)    │
│  ├── Event Normalizer + Reorder Buffer                 │
│  └── Capability Negotiation                            │
├─────────────────────────────────────────────────────────┤
│  Execution Layer (pluggable backends)                   │
│  ├── cua-driver (macOS native — default)               │
│  ├── Mock Host (in-process, for testing)               │
│  └── (future: remote VM, cloud sandbox, ...)           │
└─────────────────────────────────────────────────────────┘
```

The cognitive pipeline built on top:

```
┌───────────────────────────────────────────────────────────────┐
│  Copilot           Workflow-level next-step prediction    │
├───────────────────────────────────────────────────────────────┤
│  World Model       State prediction · Experience replay   │
├───────────────────────────────────────────────────────────────┤
│  Skill Synthesis   Hypothesis → Draft → Verified → Prod  │
├───────────────────────────────────────────────────────────────┤
│  Causal Engine     Rule · Heuristic · VLM verification    │
├───────────────────────────────────────────────────────────────┤
│  Perception        Multi-channel signal fusion (7 ch)     │
├───────────────────────────────────────────────────────────────┤
│  Execution Layer   OS interaction (screen, input, AX)     │
└───────────────────────────────────────────────────────────────┘
```

The **Execution Layer** provides native OS interactions — screen capture, accessibility tree queries, and input injection. The default backend is `cua-driver` (macOS, MCP stdio transport), but the architecture is backend-agnostic via the Platform Adaptation Layer. **Perception** fuses raw signals into a causal timeline. The **Causal Engine** infers why things happened, not just what. The **World Model** builds an internal representation of the environment and learns from prediction errors. **Skill Synthesis** distills observations into parameterized, reusable skills with maturity tracking. The **Copilot** predicts your next workflow step and offers proactive suggestions — like GitHub Copilot, but for everything you do on your computer.


---

## Prerequisites

| Component | Version | Purpose |
|-----------|---------|----------|
| Python | ≥ 3.11 | Runtime (3.11–3.14 supported) |
| [uv](https://github.com/astral-sh/uv) | latest | Fast package manager & virtualenv |
| macOS | 14.0+ (Sonoma) | Required for native perception (execution backend) |
| [cua-driver](https://github.com/trycua/cua) | latest | Default execution backend — screen capture, input injection, accessibility |
| LLM API Key | — | DashScope, OpenAI, DeepSeek, or any OpenAI-compatible provider |

> **Note:** You can run LeapFlow on any platform with `--mock-host` (no native perception), but full signal capture requires macOS with an execution backend (currently `cua-driver`) installed and Accessibility permissions granted.

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

### 3. Install Execution Backend (macOS only)

The default execution backend is `cua-driver`, which provides screen capture, accessibility tree access, and input injection via the MCP protocol. Skip this step if you just want to explore with `--mock-host`.

```bash
brew install trycua/tap/cua-driver
```

Verify the driver is available:

```bash
uv run leap host doctor       # Checks execution backend status and permissions
```

> **Tip:** macOS will prompt for Accessibility and Screen Recording permissions on first use. Grant both in System Settings → Privacy & Security.

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
| `LEAPFLOW_LLM_CONTEXT_LENGTH` | No | `256000` | Runtime context budget shown in the TUI status bar |
| `LEAPFLOW_MOCK_HOST` | No | `0` | Set `1` to use in-process mock (no execution backend) |
| `LEAPFLOW_RECORDING_MODE` | No | `video` | `video` / `default` / `vision_only` |
| `LEAPFLOW_LOG_LEVEL` | No | `INFO` | `DEBUG` / `INFO` / `WARNING` |
| `LEAPFLOW_DUCKDB_PATH` | No | `~/.leapflow/memory.duckdb` | Persistent storage location |
| `LEAPFLOW_DATA_DIR` | No | `~/.leapflow` | Root data directory |

<details>
<summary>Full Configuration Reference (all LEAPFLOW_* variables)</summary>

| Variable | Default | Description |
|----------|---------|-------------|
| **LLM** | | |
| `LEAPFLOW_LLM_API_KEY` | _(required)_ | API key for LLM provider |
| `LEAPFLOW_LLM_BASE_URL` | DashScope endpoint | OpenAI-compatible base URL |
| `LEAPFLOW_LLM_MODEL` | `qwen3.7-plus` | Model identifier |
| `LEAPFLOW_LLM_MAX_RETRIES` | `3` | Retry attempts on transient LLM errors |
| `LEAPFLOW_LLM_CONTEXT_LENGTH` | `256000` | Runtime context budget in tokens; explicit config wins over static model hints |
| **Platform** | | |
| `LEAPFLOW_MOCK_HOST` | `0` | `1` to use in-process mock (no execution backend) |
| **Storage** | | |
| `LEAPFLOW_DUCKDB_PATH` | `~/.leapflow/memory.duckdb` | Persistent DuckDB path |
| `LEAPFLOW_DATA_DIR` | `~/.leapflow` | Root data directory |
| `LEAPFLOW_AUDIT_LOG_PATH` | `~/.leapflow/audit.jsonl` | JSONL audit log location |
| **Memory** | | |
| `LEAPFLOW_MEMORY_WORKING_MAX_TOKENS` | `8192` | Working memory token budget |
| `LEAPFLOW_MEMORY_EPISODIC_TTL_S` | `300.0` | Episodic memory TTL (seconds) |
| `LEAPFLOW_MEMORY_EPISODIC_MAX_ENTRIES` | `200` | Max episodic memory entries |
| `LEAPFLOW_MEMORY_EVOLUTION_MAX_EPISODES` | `1000` | Max episodes for evolution |
| **Recording** | | |
| `LEAPFLOW_RECORDING_MODE` | `video` | `video` / `default` / `vision_only` |
| `LEAPFLOW_VIDEO_FPS` | `5` | Screen capture frames per second |
| `LEAPFLOW_VIDEO_RESOLUTION_SCALE` | `0.75` | Resolution scale (0.0–1.0) |
| `LEAPFLOW_VIDEO_CODEC` | `h264` | Video codec (h264 is HW-accelerated on macOS) |
| `LEAPFLOW_VIDEO_MAX_SEGMENT_S` | `600` | Max seconds per video segment |
| `LEAPFLOW_VIDEO_CACHE_DIR` | `~/.leapflow/cache/video` | Video segment cache directory |
| **Video Analysis** | | |
| `LEAPFLOW_VIDEO_VLM_URL_SCHEME` | `base64` | VLM URL scheme (`base64` or HTTPS prefix) |
| `LEAPFLOW_VIDEO_L2_ENABLED` | `true` | Enable moment-level detailed VLM analysis |
| `LEAPFLOW_VIDEO_L3_ENABLED` | `true` | Enable frame-level OCR/UI analysis |
| `LEAPFLOW_VIDEO_MAX_L2_REQUESTS` | `10` | Max VLM calls per segment |
| **Learnability Assessment** | | |
| `LEAPFLOW_LEARNABILITY_ENABLED` | `true` | Master switch for learnability filter |
| `LEAPFLOW_LEARNABILITY_MIN_STEPS` | `3` | Min trajectory steps to consider |
| `LEAPFLOW_LEARNABILITY_LEARN_THRESHOLD` | `0.65` | Score above → auto-distill |
| `LEAPFLOW_LEARNABILITY_ASK_THRESHOLD` | `0.40` | Score above → ask user |
| **Learning & Execution** | | |
| `LEAPFLOW_LEARN_IDLE_TIMEOUT` | `300` | Idle timeout (seconds) during learn mode |
| `LEAPFLOW_LEARN_AUTO_DISTILL` | `true` | Auto-distill after recording stops |
| `LEAPFLOW_CONFIRM_DEFAULT_LEVEL` | `confirm` | Default trust tier for new skills |
| **Execution Loop** | | |
| `LEAPFLOW_REACT_MAX_ITERATIONS` | `20` | Hard limit on ReAct iterations |
| `LEAPFLOW_TOOL_MAX_ITERATIONS` | `30` | Hard limit on tool-call iterations |
| `LEAPFLOW_COMPRESS_THRESHOLD` | `16` | Context compression trigger |
| `LEAPFLOW_MAX_TOOL_OUTPUT_CHARS` | `2000` | Truncate tool output beyond this |
| **Interactive UX** | | |
| `LEAPFLOW_STREAM_OUTPUT` | `true` | Stream LLM tokens in real-time |
| `LEAPFLOW_VERBOSE_PROGRESS` | `true` | Show tool execution progress inline |
| `LEAPFLOW_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` |
| **Signal Fusion** | | |
| `LEAPFLOW_SIGNAL_CHANNELS` | `all` | Active signal channels (comma-separated or `all`) |
| `LEAPFLOW_SURPRISE_ENABLED` | `true` | Enable surprise detection annotations |
| **Causal Inference** | | |
| `LEAPFLOW_CAUSAL_REORDER_WINDOW_MS` | `300` | Event reorder window (ms) |
| `LEAPFLOW_CAUSAL_WINDOW_S` | `3.0` | Causal inference time window |
| `LEAPFLOW_CAUSAL_TIER3_ENABLED` | `false` | Enable VLM-backed Tier 3 inference |
| **World Model** | | |
| `LEAPFLOW_PREDICTION_ENABLED` | `true` | Enable predictive coding loop |
| `LEAPFLOW_PREDICTION_DELTA_THRESHOLD` | `0.3` | Prediction error threshold |
| `LEAPFLOW_CURIOSITY_ALPHA` | `0.4` | Curiosity: novelty weight |
| `LEAPFLOW_REPLAY_ON_SESSION_END` | `true` | Run experience replay on session end |
| **Workflow Copilot** | | |
| `LEAPFLOW_COPILOT_ENABLED` | `true` | Enable Copilot prediction engine |
| `LEAPFLOW_COPILOT_MIN_IDLE_MS` | `500` | Min pause to trigger suggestion |
| `LEAPFLOW_COPILOT_MAX_IDLE_MS` | `5000` | Max idle before suppressing |
| `LEAPFLOW_COPILOT_CACHE_TTL_S` | `30.0` | Speculative cache TTL |
| `LEAPFLOW_COPILOT_SPECULATIVE_CACHE_SIZE` | `100` | Cache entry limit |
| `LEAPFLOW_COPILOT_ACTION_RING_SIZE` | `10` | Context action ring buffer size |
| **RPC** | | |
| `LEAPFLOW_RPC_TIMEOUT_DEFAULT` | `30.0` | Default RPC call timeout (seconds) |

</details>

---

## Quick Start — First Experience

### Step 1: Launch the Interactive TUI

```bash
uv run leap
```

> You'll see the LeapFlow banner, session info (model, context budget, platform, cwd), and a `❯` prompt — you're in the rich interactive TUI. If you do not have a native execution backend available yet, use `uv run leap --mock-host` for a safe first run.

### Step 2: Have a Conversation

```
> What can you help me with?
```

> LeapFlow explains its capabilities: task execution, skill learning, workflow automation.

### Step 3: Teach a Skill (Teaching Mode)

Open a new terminal and start a teaching session:

```bash
uv run leap teach "organize screenshots by date"
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

## Recommended Entry Point — Interactive TUI

LeapFlow is designed to be used primarily through the **interactive terminal UI**. Start here for day-to-day work: chat, inspect status, trigger tools, resume sessions, and progressively teach or execute workflows from one conversational surface.

```bash
uv run leap                         # Recommended: launch the interactive TUI
uv run leap --mock-host             # Safe first run without native OS backend
uv run leap "summarize this repo"   # Single-turn prompt, then exit
```

Why we recommend the TUI first:

- **One surface for the whole loop**: conversation, tool execution, skill learning, status, and session continuity.
- **Real-time transparency**: streaming output, tool progress, daemon status, model name, and context budget are visible while work is running.
- **Lower setup friction**: first-run config is generated automatically, and API key/context changes are hot-reloaded in active sessions.
- **Best default mental model**: use `leap` like an always-available agent console; reach for subcommands only when scripting or automating.

### TUI Status Bar

The bottom toolbar shows the active model and context usage, for example:

```text
qwen3.7-plus │ 0/256K │ [░░░░░░░░░░] 0%
```

The max value comes from `LEAPFLOW_LLM_CONTEXT_LENGTH` — LeapFlow's runtime context budget. Configure it in `~/.leapflow/.env`, project `./.env`, `~/.leapflow/config.yaml`, or real environment variables. Explicit config always wins over static model capability hints.

<details>
<summary>More commands — teaching, execution, skills, host, daemon</summary>

### Teach Mode — Learn from Demonstration

```bash
uv run leap teach "describe what you'll demonstrate"
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

### Command Reference

| Command | Syntax | Description |
|---------|--------|-------------|
| _(default)_ | `leap` | Launch the interactive TUI with multi-turn conversation |
| _(prompt)_ | `leap "question"` | Single-turn chat (answer + exit) |
| `teach` | `leap teach [goal] [options]` | Record a demonstration and distill into a skill |
| `run` | `leap run [prompt] [options]` | Execute a matched skill |
| `skills` | `leap skills [action] [name]` | Manage the skill library |
| `relearn` | `leap relearn <trajectory_id>` | Re-run learning pipeline on a saved trajectory |
| `host` | `leap host <action>` | Manage execution backend connection and diagnostics |
| `daemon` | `leap daemon <action>` | Manage the persistent leapd runtime |

**Global Flags:**

| Flag | Effect |
|------|--------|
| `--mock-host` | Use in-process mock host (no native perception) |
| `--thinking` | Enable LLM extended reasoning mode |

**`leap teach` options:**

| Flag | Description |
|------|-------------|
| `goal` | Goal description (positional) |
| `--timeout <sec>` | Custom idle timeout before auto-stop |
| `--resume <id>` | Resume a previous learning session |
| `--field <rule>` | Perceptual field rule: `app:context[:level]` (repeatable) |

**`leap run` options:**

| Flag | Description |
|------|-------------|
| `prompt` | Natural language trigger (positional) |
| `--skill <name>` | Match by exact skill name |
| `--step` | Step-through with confirmation per action |
| `--auto` | Skip confirmations, execute directly |

**`leap skills` actions:**

| Action | Description |
|--------|-------------|
| `list` | List all registered skills (default) |
| `show <name>` | Inspect a specific skill's details |
| `export <name> [-o file]` | Export skill definition to JSON |
| `import <file>` | Import skill from JSON file |
| `disable <name>` | Deactivate a skill without deletion |
| `delete <name>` | Permanently delete a skill |
| `audit [name] [--limit N]` | View execution history |
| `sessions` | List recorded learning sessions |

**`leap host` actions:**

| Action | Description |
|--------|-------------|
| `doctor` | Check execution backend installation, version, and macOS permissions |
| `status` | Show connection status to execution backend |

</details>

---

## Terminal UI

LeapFlow provides a rich interactive terminal experience built on [Rich](https://github.com/Textualize/rich) (output rendering) and [prompt_toolkit](https://github.com/prompt-toolkit/python-prompt-toolkit) (input handling). The TUI activates automatically when you launch `leap` in a terminal.

### Features

| Feature | Description |
|---------|-------------|
| **Markdown rendering** | LLM responses rendered as styled markdown with syntax-highlighted code blocks |
| **Streaming display** | Real-time token streaming with live markdown updates via `rich.Live` |
| **Tool activity** | Tool calls shown with elapsed timers; completed tools persist in scrollback |
| **Thinking display** | LLM reasoning/thinking rendered in a dimmed panel |
| **Persistent history** | Input history saved to `~/.leapflow/history` (Up/Down to navigate) |
| **Command completion** | Tab-completion for all REPL commands |
| **Multiline editing** | Alt+Enter inserts a newline for multi-line prompts |
| **Status bar** | Live bottom toolbar: mode, skills, platform, model, context usage, turn elapsed |
| **Adaptive theming** | Automatic light/dark detection via `COLORFGBG` / `LEAPFLOW_TUI_THEME` |
| **Session info** | Startup display showing model, platform status, cwd, and skill count |
| **Mode indicators** | Prompt character changes with session mode (idle ❯ / recording ● / paused ⏸) |

The context maximum shown in the status bar is the active runtime budget from `LEAPFLOW_LLM_CONTEXT_LENGTH`. In daemon mode, the TUI synchronizes this value from the daemon runtime so multiple terminal clients show the same budget.

### Theme Configuration

The TUI auto-detects your terminal background. Override with:

```bash
LEAPFLOW_TUI_THEME=light   # or: dark (default)
```

### Architecture

```
tui_app/
├── theme.py      # Color palette + light/dark detection
├── console.py    # Rich console wrapper (markdown, panels, tools, errors)
├── input.py      # prompt_toolkit session (history, completion, keybindings)
├── stream.py     # Live streaming renderer (markdown + tool timers)
└── status.py     # Bottom toolbar (mode, context %, model, elapsed)
```

All output flows through `LeapConsole`, ensuring consistent theming. All input flows through `LeapInput`, providing history persistence and command completion. The `StreamRenderer` handles live-updating displays during LLM streaming with zero flicker.

---

## External Platform Integration (Gateway)

LeapFlow can connect to external messaging platforms — **Feishu (Lark)**, **DingTalk**, **Telegram**, and more — turning any IM channel into a natural-language interface to the agent. Platforms are integrated through a declarative **manifest** system and configured via a conversational setup flow, with no source-code changes required.

### Design Highlights

| Aspect | Approach |
|--------|----------|
| **Config-as-Conversation** | Say *"connect to Feishu"* in the REPL. The agent walks you through credential setup in 1–2 turns — no config files to edit by hand. |
| **Declarative Manifests** | Each platform is defined by a YAML manifest (credentials, setup guide, adapter module). Add a new platform by dropping a `.yaml` file. |
| **Credential Security** | Secrets are encrypted at rest (Fernet AES-128-CBC), never appear in LLM context or logs, and can be overridden via environment variables (`LEAPFLOW_<PLATFORM>_<KEY>`). |
| **Lazy Loading** | Platform SDK dependencies are imported only when a platform is first connected, keeping CLI startup instant. |
| **Adapter Protocol** | Platform adapters implement a simple Python `Protocol` — `connect()`, `disconnect()`, `send()`, `on_message` callback — extensible via `PlatformAdapterMixin` for graceful degradation. |
| **Auto-Reconnect** | Previously configured platforms are automatically reconnected on startup. Connection state persists across sessions via `gateway.yaml`. |
| **Bidirectional** | Inbound: platform messages are processed through LLM with safe tool access. Outbound: the agent can proactively send messages via `gateway_send`. |
| **Independent Sessions** | Each external chat gets its own conversation history with a restricted tool set (read-only), isolated from the CLI session. |
| **Event-Driven** | Inbound messages are logged to episodic memory and emitted as typed events (`GatewayMessageReceived`, `GatewaySessionCreated`, `GatewaySessionEnded`) for downstream subscribers. |

### Architecture

```
                       ┌──────────────────┐
                       │  CLI Agent       │
                       │  (AgentEngine)   │
                       │                  │
                       │  gateway_send ──▶│──┐
                       └──────────────────┘  │
                                             │ send_message()
  ┌─────────────┐    ┌──────────────┐    ┌───▼─────────┐
  │  Platform    │───▶│  Gateway     │───▶│  Gateway    │───▶ LLM + safe tools
  │  Adapter     │    │  Server      │    │  Router     │◀─── reply
  │  (Protocol)  │◀───│  (lifecycle) │◀───│  (per-      │
  └─────────────┘    │              │    │   session)  │
    send reply       │  on_event ──▶│    └─────────────┘
                     └──────┬───────┘
                            │
                   ┌────────▼────────┐
                   │ Episodic Memory │
                   │ (event logging) │
                   └─────────────────┘
```

`Context` is the sole integration point — gateway modules have no dependency on engine or CLI.

### Quick Start

```
❯ connect to Telegram
# Agent shows setup steps, asks for Bot Token
❯ <paste your bot token>
# Agent validates, encrypts, connects — done.
```

Or use the `/gateway` slash command to check connection status:

```
❯ /gateway
┌── Gateway ─────────────────────┐
│ Connected                      │
│   ● Telegram (5m 32s)          │
│ Available                      │
│   飞书, DingTalk, Webhook      │
│                                │
│ Say "connect to <platform>"    │
│ to set up a new integration.   │
└────────────────────────────────┘
```

### Adding a Custom Platform

1. Create a YAML manifest in `~/.leapflow/profiles/<profile>/gateway/manifests/`:

```yaml
platform_id: my_platform
display_name: My Platform
category: im

credentials:
  - key: api_key
    label: API Key
    required: true
    secret: true

setup_guide:
  summary_en: "Provide your API key to connect."

adapter:
  module: my_package.adapter
  class: MyAdapter
  dependencies: [my-sdk]
```

2. Implement the adapter:

```python
from leapflow.gateway.protocol import PlatformAdapter

class MyAdapter:
    def __init__(self, api_key: str, **kwargs): ...
    async def connect(self, *, is_reconnect: bool = False) -> None: ...
    async def send(self, target, content) -> SendResult: ...
    async def disconnect(self) -> None: ...
```

3. Say *"connect to my_platform"* in the REPL — the agent handles the rest.

### Environment Variable Overrides

For deployment environments (CI/CD, containers), set credentials as environment variables instead of interactive configuration:

```bash
export LEAPFLOW_TELEGRAM_BOT_TOKEN=your_token_here
export LEAPFLOW_FEISHU_APP_ID=cli_xxx
export LEAPFLOW_FEISHU_APP_SECRET=xxx
```

Environment variables take precedence over values stored in `gateway.yaml`.

---

## Safety & Approval

LeapFlow enforces a **layered safety architecture** that balances autonomy with human oversight. The goal is minimal interruption — the agent asks for permission only when an action carries real consequences.

### Multi-Layer Defense

```
               ┌───────────────────────────────┐
               │    Hardline Block (always)     │  rm -rf /, mkfs, fork bomb
               ├───────────────────────────────┤
               │    Dangerous Detection         │  sudo, chmod, kill -9 ...
               │    → Approval Gate (prompt)    │  [y]es / [a]lways / [n]o
               ├───────────────────────────────┤
               │    Safe Path / Size Bypass     │  .md, .json, < 500 chars
               ├───────────────────────────────┤
               │    Output Redaction            │  Secrets stripped from results
               ├───────────────────────────────┤
               │    Untrusted Result Wrapping   │  MCP/web tool output delimited
               └───────────────────────────────┘
```

### Approval System

| Feature | Behavior |
|---------|----------|
| **Unified Gate** | A single `ApprovalGate` protocol handles shell commands, file writes, and outbound messages — swappable for TUI, Web UI, or CI modes. |
| **Session Memory** | Choose **[a]lways** once and the same category won't prompt again for the rest of the session. |
| **Per-Category Scoping** | Shell commands, file writes, and each gateway platform are tracked independently. |
| **Smart Approval** | When an auxiliary LLM is configured, low-risk commands (risk < 0.3) are auto-approved; medium/high-risk still prompt. |
| **Fail-Closed** | In non-interactive environments (pipes, CI), all dangerous actions are denied by default. |
| **Rich TUI Display** | Approval prompts render as styled panels in the terminal — not raw text — with full action detail. |
| **Gateway Send** | First outbound message to each platform requires explicit approval; subsequent sends are auto-approved for the session. |
| **Audit Trail** | Every approval decision (allow/deny/session) is logged with timestamp and category. |

### What Gets Approved

| Action | Default | Approval Needed? |
|--------|---------|-----------------|
| Safe shell commands (`ls`, `cat`, `git status`) | Auto-execute | No |
| Dangerous shell (`sudo`, `rm -r`, `kill -9`) | Prompt | Yes (once per session) |
| Destructive shell (`rm -rf /`, `mkfs`) | Always blocked | Cannot override |
| File write (`.md`, `.json`, small files) | Auto-approve | No |
| File write (large, non-safe extensions) | Prompt | Yes (once per session) |
| Gateway send (first message to platform) | Prompt | Yes (once per platform per session) |
| Gateway inbound (external messages) | Restricted tools | Only safe read-only tools |

---

## Workflow Copilot (Preview)

LeapFlow includes a **Workflow Copilot** that predicts your next action and offers proactive suggestions — like GitHub Copilot, but for any workflow on your computer.

### How It Works

```
You work normally → LeapFlow observes patterns → Suggests next steps → You accept/ignore
       │                                                                    │
       └──────────────── Gets smarter (Loop γ) ──────────────────┘
```

The Copilot operates on a **multi-tier prediction model**:

| Tier | Method | Latency | Use Case |
|------|--------|---------|----------|
| L0 | Context hash → exact history match | <1ms | Daily routines |
| L1 | N-gram sequence prediction | <5ms | Common action chains |
| L2 | Embedding retrieval from experience store | <50ms | Cross-app patterns |
| L3 | LLM reasoning + RAG | 200–2000ms | Novel situations |

### Real-Time Design

Predictions are **speculative** — computed while you work, not after you pause:

- When you perform action A, the system immediately predicts Top-K next steps
- Results are cached in memory; displayed only when you naturally pause (>300ms)
- If you start your next action before the suggestion appears, it’s silently discarded
- L0–L2 are fully local (no network); L3 runs asynchronously in the background

### Example Scenarios

- **File operations:** Move one PDF → system suggests moving matching PDFs too
- **App switching:** Open Zoom + Calendar → system offers to open meeting docs
- **Terminal:** `cd project && git pull` → system suggests `npm install && npm run dev`
- **Cross-app:** Copy text from Slack → system offers to create a Jira ticket

### Trust Gradient for Suggestions

Suggestions follow the same trust model as skills:

- **Low confidence (<0.5):** Silent — logged but not shown
- **Medium (0.5–0.8):** Ghost hint (dim text, Tab to accept)
- **High (>0.8):** Explicit suggestion with shortcut key
- **Very high (>0.95) + non-destructive + always accepted:** Auto-execute

### Configuration

```bash
# .env
LEAPFLOW_STREAM_OUTPUT=true        # Enable real-time token streaming
LEAPFLOW_VERBOSE_PROGRESS=true     # Show tool execution progress inline
```

> **Status:** The Copilot module is **fully implemented** — L0–L3 predictors, speculative pipeline, idle detection, feedback loop, and graceful degradation are all in place. The infrastructure is active internally (confidence tracking, pattern learning). Rendering of ghost-hint overlays to end-users is the remaining integration step.

<details>
<summary>Copilot — Current Implementation Status</summary>

| Component | Status | Module |
|-----------|--------|--------|
| L0 Hash Predictor | ✅ Implemented | `copilot/predictors/l0_hash.py` |
| L1 Markov Predictor | ✅ Implemented | `copilot/predictors/l1_markov.py` |
| L2 Embedding Predictor | ✅ Implemented | `copilot/predictors/l2_embed.py` |
| L3 LLM Predictor | ✅ Implemented | `copilot/predictors/l3_llm.py` |
| Speculative Pipeline | ✅ Implemented | `copilot/pipeline.py` |
| Idle Detection | ✅ Implemented | `copilot/idle.py` |
| Context Encoder | ✅ Implemented | `copilot/context.py` |
| Feedback Collector | ✅ Implemented | `copilot/feedback.py` |
| Evolution Loop (Loop γ) | ✅ Implemented | `copilot/feedback.py` |
| Graceful Degradation | ✅ Implemented | `copilot/degradation.py` |
| Memory Adapters | ✅ Implemented | `copilot/adapters.py` |
| Display Gate & Renderer | ✅ Implemented | `copilot/renderer.py` |
| CLI Ghost-Hint Overlay | ⏳ Pending | — |

**Capability Boundaries (Preview):**
- Predictions are computed and cached; display-to-user path is log-only (`LogHintRenderer`)
- L0–L2 run entirely locally with zero network dependency
- L3 requires LLM credentials and runs asynchronously
- Auto-execute is disabled for destructive actions regardless of confidence
- The system learns from implicit feedback (accept/ignore/correct) to improve over time

</details>

---

## Host Management (Execution Backend)

For full perception (screen capture, accessibility tree, input events), you need an execution backend installed. The default is `cua-driver`:

```bash
# Check execution backend and permissions
uv run leap host doctor          # Verifies backend binary, version, permissions

# Start with full perception
uv run leap                      # Connects to execution backend via MCP automatically

# Without native perception
uv run leap --mock-host          # Runs with in-process mock (for testing/exploration)
```

> **Important:** macOS will prompt for Accessibility and Screen Recording permissions on first use. Grant both in System Settings → Privacy & Security.

---

## Development

```bash
make setup            # Initialize environment
make test             # Run tests (pytest)
make lint             # Lint (ruff)
```

<details>
<summary>Project Structure & Extension Guide</summary>

### Directory Layout

```
leapflow/
├── src/leapflow/           # Python brain (src layout)
│   ├── cli/                # CLI entry + subcommands
│   ├── copilot/            # Workflow Copilot (L0–L3 predictors)
│   ├── engine/             # Session + ReAct execution loop
│   ├── perception/         # Signal channels + fusion
│   ├── signal_fusion/      # Cross-modal temporal fusion
│   ├── causal/             # Causal inference pipeline
│   ├── world_model/        # Predictive coding + experience store
│   ├── learning/           # Skill distillation + assessment
│   ├── skills/             # Skill library + execution
│   ├── analysis/           # Trajectory denoising
│   ├── memory/             # Three-tier memory system
│   ├── recording/          # Trajectory recording orchestration
│   ├── llm/                # LLM provider abstraction
│   ├── platform/           # Platform adaptation (CuaDriver client, observers, event bus)
│   ├── domain/             # Shared types & events
│   ├── storage/            # DuckDB persistence
│   ├── tools/              # Built-in tool registry
│   ├── prompts/            # LLM prompt templates
│   └── utils/              # Shared utilities
├── tests/                  # Pytest suite
├── docs/design/            # Design documents
└── scripts/                # Setup & run scripts
```

### Adding a New Skill (Plugin)

1. Create a skill JSON (or teach via `leap teach`).
2. Import it: `leap skills import my_skill.json`
3. The skill appears in the registry with `DRAFT` maturity.
4. Each successful execution increases confidence → `VERIFIED` → `PRODUCTION`.

### Adding a New Copilot Predictor

Implement the `PredictorLayer` protocol:

```python
from leapflow.copilot.types import PredictorLayer, ContextState, PredictionCandidate, FeedbackSignal

class MyPredictor:
    @property
    def layer_id(self) -> str: return "L_custom"
    @property
    def priority(self) -> int: return 5  # lower = higher priority
    @property
    def timeout_ms(self) -> int: return 50

    async def predict(self, context: ContextState) -> list[PredictionCandidate]: ...
    async def on_feedback(self, signal: FeedbackSignal) -> None: ...
```

Register it with `PredictionEngine.register_layer(MyPredictor())`.

### Adding a New Signal Channel

Implement the `SignalChannel` protocol:

```python
from leapflow.copilot.types import SignalChannel, Signal

class MyChannel:
    @property
    def channel_id(self) -> str: return "my_sensor"
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    def subscribe(self, handler) -> None: ...
```

### Running Tests

```bash
make test                              # Full suite
uv run pytest tests/test_pure_algorithms.py -q   # Single file
uv run pytest -k "test_world_model" -q            # By keyword
```

</details>

## Key Modules

| Module | Role |
|--------|------|
| `perception/` | Multi-channel signal capture and fusion (trajectory, AX tree, clipboard, keyboard, file system, etc.) |
| `signal_fusion/` | Cross-modal temporal alignment and surprise detection |
| `causal/` | Three-tier causal inference engine (rule → heuristic → VLM) |
| `world_model/` | Predictive coding loop, experience store, curiosity-driven learning |
| `learning/` | Skill distillation, parameterization, and maturity lifecycle |
| `skills/` | Skill library, runtime execution, and self-evolution (Loop γ) |
| `copilot/` | Workflow-level next-step prediction and proactive suggestion |
| `analysis/` | Six-layer denoising pipeline for trajectory refinement |
| `engine/` | Session orchestration and ReAct execution loop |
| `memory/` | Three-tier event-driven memory (working → episodic → long-term) |
| `platform/` | Platform adaptation layer — protocol abstraction for pluggable execution backends |
| `gateway/` | External platform integration — declarative manifests, credential vault, adapter lifecycle |
| `hub/` | Multi-source skill hub (ModelScope, GitHub, local) |

<details>
<summary>Architecture — Detailed Module Map</summary>

| Module | Path | Key Files | Responsibility |
|--------|------|-----------|----------------|
| Perception | `src/leapflow/perception/` | channels, fusion, pipeline | Raw signal capture (7 channels), frame extraction, privacy gating |
| Signal Fusion | `src/leapflow/signal_fusion/` | timeline, surprise, mhms | Temporal alignment, surprise scoring, multi-hypothesis fusion |
| Causal Engine | `src/leapflow/causal/` | rule, heuristic, vlm_tier | Three-tier causal chain construction (rule→heuristic→VLM) |
| World Model | `src/leapflow/world_model/` | predictor, experience_store, curiosity | Predict-then-verify loop, experience replay, curiosity-driven exploration |
| Learning | `src/leapflow/learning/` | distiller, parameterizer, assessor | Trajectory → skill distillation, learnability assessment |
| Skills | `src/leapflow/skills/` | registry, executor, lifecycle | Skill storage (DuckDB), runtime execution, maturity progression |
| Copilot | `src/leapflow/copilot/` | pipeline, predictors/, engine | Speculative L0–L3 prediction cascade, idle detection, feedback loop |
| Analysis | `src/leapflow/analysis/` | denoiser, layers | Six-layer denoising pipeline for raw trajectory refinement |
| Engine | `src/leapflow/engine/` | session, react_loop, tools | Session orchestration, ReAct loop, tool dispatch, context compression |
| Memory | `src/leapflow/memory/` | working, episodic, long_term | Three-tier event-driven memory with promotion/eviction |
| LLM | `src/leapflow/llm/` | provider, message_builder | LLM abstraction (OpenAI-compatible), streaming, retry logic |
| Platform | `src/leapflow/platform/` | cua_client, adapter, observers | Platform adaptation layer — CuaDriver MCP client, event normalization, observation daemon |
| Domain | `src/leapflow/domain/` | events, perception, types | Shared domain types, event definitions, perception models |
| Recording | `src/leapflow/recording/` | recorder, video, segmenter | Trajectory recording orchestration, segmentation, caching |
| Tools | `src/leapflow/tools/` | registry, builtins | Built-in tool definitions for the ReAct loop |
| CLI | `src/leapflow/cli/` | cli, commands/, banner | Argument parsing, subcommand dispatch, interactive REPL |
| Storage | `src/leapflow/storage/` | duckdb, skill_library | DuckDB-backed persistent storage for skills, trajectories, audit |
| Gateway | `src/leapflow/gateway/` | server, manifest, protocol, credential_vault | External platform integration — manifest discovery, adapter lifecycle, credential encryption |
| Execution Backend | external (pluggable) | — | OS interaction: screen capture, AX tree, input injection (default: `cua-driver` via MCP) |

</details>

<details>
<summary>Key Protocols & Interfaces</summary>

| Protocol | Module | Purpose |
|----------|--------|---------|
| `Signal` | `copilot/types.py` | Unified abstraction for any signal source (event_type, timestamp, payload, source) |
| `PredictorLayer` | `copilot/types.py` | Prediction algorithm interface (predict, on_feedback, priority, timeout) |
| `SignalChannel` | `copilot/types.py` | Dynamically registerable signal source (start, stop, subscribe) |
| `HintRenderer` | `copilot/types.py` | Ghost-hint display abstraction (show, dismiss, is_visible) |
| `PlatformAdapter` | `gateway/protocol.py` | External platform adapter contract (connect, disconnect, send, on_message) |

**Core Data Types:**

| Type | Description |
|------|-------------|
| `ContextState` | Incremental operational context snapshot with delta updates and O(1) hash lookup |
| `PredictionCandidate` | Immutable prediction result (action, confidence, source layer, expiry) |
| `FeedbackSignal` | Structured user response (accept/ignore/correct/reject + latency) |
| `FeedbackType` | Enum: `ACCEPT`, `IGNORE`, `CORRECT`, `EXPLICIT_REJECT` |

**MCP Protocol (LeapFlow → Execution Backend):**

Transport: stdio (JSON-RPC over stdin/stdout) by default. The `PlatformClient` in `platform/` manages the connection lifecycle and abstracts the specific backend.

| Method | Direction | Description |
|--------|-----------|-------------|
| `screen.capture` | LeapFlow → Backend | Capture screen frame(s) |
| `accessibility.tree` | LeapFlow → Backend | Query accessibility tree |
| `accessibility.perform` | LeapFlow → Backend | Perform action on UI element |
| `input.type` / `input.shortcut` | LeapFlow → Backend | Inject keyboard input |
| `input.click` / `input.scroll` | LeapFlow → Backend | Inject mouse input |
| `system.info` | LeapFlow → Backend | Query platform capabilities |

</details>

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `cua-driver not found` | Execution backend not installed | `brew install trycua/tap/cua-driver` |
| `MCP connection failed` | Backend process not responding | Run `leap host doctor` to diagnose; ensure backend is on PATH |
| `LEAPFLOW_LLM_API_KEY is empty` | Missing API key | Set `LEAPFLOW_LLM_API_KEY` in `.env` |
| `Accessibility permission denied` | macOS privacy gate | System Settings → Privacy & Security → Accessibility → grant permission |
| `Screen Recording blocked` | macOS privacy gate | System Settings → Privacy & Security → Screen Recording → grant permission |

---

## License

Apache 2.0 — see [LICENSE](LICENSE).


---

<div align="center">

**[ModelScope](https://github.com/modelscope)** · [⭐ Star us](https://github.com/modelscope/leapflow/stargazers) · [🐛 Report a bug](https://github.com/modelscope/leapflow/issues) · [💬 Discussions](https://github.com/modelscope/leapflow/discussions)

*✨ LeapFlow: Learning and Evolving from Actual Practice. *

</div>

<p align="center">
  <em> ❤️ Thanks for Visiting ✨ LeapFlow !</em><br><br>
  <img src="https://visitor-badge.laobi.icu/badge?page_id=modelscope.leapflow&style=for-the-badge&color=00d4ff" alt="Views">
</p>