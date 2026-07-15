# LeapFlow

**A signal-driven, self-evolving agent framework that learns autonomously from the real world.**


### News

- **2026-07-15**: v0.0.2 released — TUI, Gateway/App Connector, Hub sync, Scheduler, Workflow Copilot, and runtime hardening.
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
| [uv](https://github.com/astral-sh/uv) | latest | Source installs and development only |
| macOS | 14.0+ (Sonoma) | Required for native perception (execution backend) |
| [cua-driver](https://github.com/trycua/cua) | latest | Default execution backend — screen capture, input injection, accessibility |
| LLM API Key | — | DashScope, OpenAI, DeepSeek, or any OpenAI-compatible provider |

> **Note:** You can run LeapFlow on any platform with `--mock-host` (no native perception), but full signal capture requires macOS with an execution backend (currently `cua-driver`) installed and Accessibility permissions granted.

## Installation

### 1. Install LeapFlow

```bash
pip install leapflow
```

This installs the `leap` command. The first `leap` run creates the local LeapFlow home under `~/.leapflow` automatically.

<details>
<summary>Install from source</summary>

```bash
git clone https://github.com/modelscope/leapflow.git
cd leapflow
uv sync --all-extras
uv run leap --help
```
</details>

### 2. Configure Your LLM

If you already have an API key, base URL, and model name, save them through the unified config command:

```bash
leap config llm set \
  --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --model qwen3.7-plus \
  --ask-api-key
```

Paste your API key when prompted. LeapFlow stores the key in the local secret vault and writes only a `secret://` reference into durable config.

For scripts or CI, pass the key explicitly through the same config control plane:

```bash
leap config llm set \
  --base-url https://api.openai.com/v1 \
  --model gpt-4o \
  --api-key "$OPENAI_API_KEY"
```

### 3. Install Execution Backend (macOS only)

The native execution backend enables screen capture, accessibility tree access, and input injection. Skip this step if you only want to chat or explore with `--mock-host`.

```bash
brew install trycua/tap/cua-driver
leap host doctor
```

macOS may ask for Accessibility and Screen Recording permissions on first use. Grant both in System Settings → Privacy & Security.

### 4. Verify Installation

```bash
leap --mock-host "hello, are you ready?"
```

Expected: LeapFlow responds with a greeting confirming it's operational.

---

## Configuration Reference

For normal setup, follow the Installation steps above. For day-to-day changes, use `leap config` as the durable configuration control plane:

```bash
leap config list                  # human-readable catalog: key, value, type, scope, reload, description
leap config show llm.model        # detailed metadata and effective value for one key
leap config keys                  # compact key-only output for scripts
leap config get llm.model
leap config set memory.working_max_tokens 12000
leap config set visual.track_enabled true
```

Inside the TUI, the same control plane is available as `/config`. It supports hot reload for the active session when possible and provides inline completion for subcommands, config keys, and simple values:

```text
/config list
/config list llm
/config show llm.model
/config get llm.model
/config set runtime.log_level DEBUG
/config set visual.track_enabled true
```

### Important Config Keys

| Key | Typical value | Description |
|-----|---------------|-------------|
| `llm.api_key` | secret value | Primary LLM API key; stored in the local vault, never as durable plaintext |
| `llm.base_url` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | OpenAI-compatible LLM endpoint |
| `llm.model` | `qwen3.7-plus` | Primary model used for chat, planning, and tool reasoning |
| `llm.context_length` | `1000000` | Runtime context budget shown in the TUI status bar |
| `memory.working_max_tokens` | `12000` | Working-memory budget injected into active reasoning |
| `visual.track_enabled` | `true` / `false` | Enable screenshot-based visual perception |
| `recording.mode` | `video` | Default teaching/observation recording pipeline |
| `runtime.mock_host` | `true` / `false` | Use the in-process mock host when native OS control is unavailable |
| `runtime.log_level` | `DEBUG` / `INFO` | Runtime diagnostic verbosity |
| `scheduler.tick_seconds` | `1.0` | Scheduler polling interval |

`leap config list` is the authoritative source for all writable keys and generated metadata for less common runtime, perception, learning, gateway, hub, safety, and scheduler settings; use `leap config show <key>` or `/config show <key>` when you need one field's details. LeapFlow does not load `.env` files; durable settings belong in the config control plane and secrets belong in the vault.

---

## Quick Start — Use the TUI First

LeapFlow's default experience is the interactive terminal UI. Start here for chat, tool execution, runtime status, session continuity, and progressively learning workflows from one surface.

### Step 1: Launch LeapFlow

```bash
leap
```

If you only want to try the interface without a native execution backend, run:

```bash
leap --mock-host
```

You'll see the banner, active model, context budget, platform status, current directory, and the `❯` prompt.

### Step 2: Check Setup Hints

On first launch, LeapFlow surfaces missing setup directly in the TUI. For example, if no LLM API key is configured, it shows a short action hint instead of failing silently.

The bottom status bar keeps the important runtime state visible:

```text
qwen3.7-plus │ 0/1M │ [░░░░░░░░░░] 0%
```

### Step 3: Ask Naturally

```text
❯ What can you help me with in this repo?
```

LeapFlow streams responses, shows tool activity as it works, and keeps recoverable status visible instead of leaving the terminal idle.

### Step 4: Use TUI Commands When Needed

Inside the TUI, use slash commands for quick inspection and control:

```text
/status   show runtime and backend status
/tools    inspect available tools
/config   view or update saved config; `/config keys` lists writable settings
/model    show the active model; `/model qwen3.7-plus` updates and hot-reloads it
/usage    inspect context and token usage
/clear    clear the visible conversation
```

For example, this changes the model without leaving the TUI:

```text
/config llm set --model qwen3.7-plus
```

Use the TUI for day-to-day work. Reach for standalone CLI subcommands only when scripting, automation, or explicit one-shot operations are more convenient.

<details>
<summary>Extended CLI usage — one-shot chat, teaching, skills, host, daemon</summary>

### One-shot Chat

```bash
leap "summarize this repo"
```

### Teach Mode — Learn from Demonstration

```bash
leap teach "describe what you'll demonstrate"
```

LeapFlow records your actions as a trajectory, then distills them into a parameterized skill. The skill progresses through maturity tiers: `DRAFT → VERIFIED → PRODUCTION`.

Options:
- `--timeout 600` — Custom idle timeout before auto-stopping
- `--field "Safari:browsing:full"` — Per-app perception rules

### Autonomous Mode — Execute Learned Skills

```bash
leap run "trigger phrase"         # Match by natural language
leap run --skill "exact-name"     # Match by skill name
leap run --step "careful task"    # Step-through with confirmation
leap run --auto "routine task"    # Skip confirmations
```

Skills start at `STEP` tier (human confirms each action) and graduate to `AUTO` as confidence grows.

### Command Reference

| Command | Syntax | Description |
|---------|--------|-------------|
| _(default)_ | `leap` | Launch the interactive TUI with multi-turn conversation |
| _(prompt)_ | `leap "question"` | Single-turn chat, then exit |
| `teach` | `leap teach [goal] [options]` | Record a demonstration and distill into a skill |
| `run` | `leap run [prompt] [options]` | Execute a matched skill |
| `skills` | `leap skills [action] [name]` | Manage the skill library |
| `relearn` | `leap relearn <trajectory_id>` | Re-run learning pipeline on a saved trajectory |
| `host` | `leap host <action>` | Manage execution backend diagnostics |
| `daemon` | `leap daemon <action>` | Manage the persistent leapd runtime |

**Global Flags:**

| Flag | Effect |
|------|--------|
| `--mock-host` | Use in-process mock host without native perception |
| `--thinking` | Enable LLM extended reasoning mode |

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

**`leap daemon` actions:**

| Action | Description |
|--------|-------------|
| `status` | Show daemon runtime, config, DB, model, and context status |
| `start` | Start leapd for the active profile |
| `stop` | Stop the running leapd process for the active profile |
| `restart` | Restart leapd after reinstalling, upgrading, or changing runtime code |

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

### Daemon-backed TUI Lifecycle

By default, `leap` uses `leapd`, a per-profile background runtime shared across terminal clients. Exiting the TUI closes the current client; before returning, LeapFlow asks whether to stop `leapd` and defaults to stopping it. Keep it running when you want another terminal to reuse the same runtime.

After reinstalling or upgrading LeapFlow, restart the daemon so the background process loads the new code:

```bash
leap daemon restart
```

Use diagnostics when the TUI appears stale:

```bash
leap daemon status
```

`status` prints the daemon PID, socket, runtime source path, Python executable, model, context usage, config paths, and DB path.

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
| **Credential Security** | Secrets are encrypted at rest (Fernet AES-128-CBC), never appear in LLM context or logs, and are stored through profile-scoped vault references. |
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

### Credential Management

For deployment environments, provision platform credentials through the same gateway setup flow or profile-scoped secret vault used by interactive sessions. `gateway.yaml` stores only non-secret values and `secret://profile/...` references; plaintext credentials should not be committed or placed in ad-hoc environment files.

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
# Temporary process override
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
leap host doctor          # Verifies backend binary, version, permissions

# Start with full perception
leap                      # Connects to execution backend via MCP automatically

# Without native perception
leap --mock-host          # Runs with in-process mock (for testing/exploration)
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
| Gateway | `src/leapflow/gateway/` | server, manifest, protocol, credential_vault | External platform integration — manifest discovery, adapter lifecycle, vault-backed credential refs |
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
| `LLM API key is empty` | Missing API key | Follow **Configure Your LLM** above or run `leap config llm key` |
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