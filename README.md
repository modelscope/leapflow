# LeapFlow

**Everything is a signal**

LeapFlow is a signal-driven, self-evolving agent framework. Virtual signals (IM apps, OS events, web) and physical signals (robot arms, depth cameras, scientific instruments, IoT) flow through one pipeline into observation → learning → decision-control loops. Knowledge compounds across episodes: skills, device trust, and environment understanding accumulate — the agent gets smarter the more you use it.

---

### News

- **v0.6.0** — Benchmark qualification suite (36 scenarios, 12 external integrations), governed hardware degradation coordinator, failure fingerprints, CUA window-target contract, adaptive context retention.
- **v0.5.0** — LeapRobot: governed robot control, local/remote VLA inference, real-time servo, teleoperation + episode export; hc.v1 descriptors, robot-arm & ROS 2 transports.
- **v0.4.0** — Engine decomposition (6 delegates, 5 sub-packages), BM25 progressive tool disclosure, PCD prefix-commitment, cache-hit telemetry, skill curator, scheduler productization.

<details>
<summary>Previous releases</summary>

- **v0.3.0** — Event-sourced self-evolution pipeline, `leap evolve` CLI, LeapBoard evolution & causal views.
- **v0.2.0** — Teacher/student world-model self-evolution, Hardware Context Protocol (hc.v0), 8 generic hardware tools, LeapBoard hardware lens, device preview pipeline.
- **v0.1.1** — DeepSeek Harness plugin compatibility, restricted Node.js bridge, component-aware verdicts.
- **v0.1.0** — First-class plugin subsystem: adaptive evolution, marketplace (Ed25519 signing), sandbox isolation, Gateway adapter registry.
- **v0.0.9** — Real-time signal pipeline, TUI thinking display, cross-session history, long-task convergence hardening.
- **v0.0.8** — Cross-platform Windows support (DaemonTransport), journey test layer with cassette-backed CI.
- **v0.0.7** — 1M-class context windows, self-calibrating token estimator, concurrent-TUI session isolation.

</details>

---

## Quick Start

**1. Install**

```bash
pip install leapflow
```

<details>
<summary>From source</summary>

```bash
git clone https://github.com/modelscope/leapflow.git && cd leapflow
uv sync --all-extras --no-extra leapspace
```

</details>

**2. Configure LLM** — launch the TUI and set your provider (key goes to the local secret vault, never plaintext):

```bash
leap
```

```text
/config llm set --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
  --model qwen3.8-flash --api-key sk-...
```

Any OpenAI-compatible endpoint works — DashScope, OpenAI, DeepSeek, Anthropic, etc.

<details>
<summary>Shell / CI</summary>

```bash
leap config llm set \
  --base-url https://api.openai.com/v1 \
  --model gpt-6.1-sol \
  --ask-api-key
```

</details>

**3. Go**

```bash
leap                          # interactive TUI
leap "summarize this repo"    # one-shot
leap --mock-host              # no native execution backend
```

Have hardware? See [Physical AI](#physical-ai).

---

## Features

### Coding Tools

Locate → read → edit → verify code, out of the box:

| Tool | What it does |
|---|---|
| `repo_map` | Project orientation — languages, structure, entry points |
| `code_search` | Regex search (ripgrep) |
| `edit_file` | Anchored search-replace or unified diff — rejects ambiguous edits |
| `code_intel` | Document symbols (Python AST) |
| `git_query` / `git_write` | Structured git operations |
| `test_run` / `lint_check` | Auto-detect and run test / lint suites |
| `web_fetch` | Fetch pages and JSON APIs — read-only, cached, retried |
| `terminal_session` | Persistent shell sessions (opt-in) |

### Teach & Learn

```bash
leap teach "open Safari and bookmark this page"
```

Record your actions → distill into parameterized skills → earn autonomy through success:
`DRAFT → CANDIDATE → VERIFIED → PRODUCTION`.

### LeapBoard

```text
/board                  # session analysis (default)
/board finance          # finance lens
/board templates add ./custom.yaml
```

Real-time monitoring dashboard with Server-Driven UI. Bring your own YAML template — one file, zero code.

### Gateway

```text
❯ connect to Telegram
```

Turn any IM channel (Feishu, DingTalk, Telegram, Slack, …) into a signal boundary. Inbound events are classified, filtered, and routed; each chat gets an isolated session. Setup is conversational — no config files to edit.

### Physical AI

LeapFlow bridges LLM intelligence and real-world hardware through a **three-layer protocol stack** and a **layered control hierarchy**:

```
  LHP Fabric          cross-node data plane (JSON Lines)
  ────────────────    inventory · reads · writes · halts · permits · fleet
  HCP                 per-device contract
  ────────────────    6-method transport: open / close / read / write / probe / halt
  Southbound          transport plugins
  ────────────────    ROS2 · Piper · Feetech · RealSense · model service · MCP · media …
```

```
  System-2  (LLM)     0.2–1 Hz     task planning → node skill (plan → approve → run)
                          ↓
  System-1.5 (skills) 5–30 Hz      perceive · grasp · policy.run (VLA action chunks)
                          ↓
  System-1  (servo)   100–1000 Hz  PID / impedance — never crosses the fabric
                          ↓
                      actuators     escalation flows back up ↑
```

The host sends **goals**; the node decomposes them locally. A skill is planned on the node, approved once by its plan digest, then executed next to the device with a run-scoped permit — so a 1 kHz servo loop and a 30 Hz policy stream stay safe regardless of network latency.

**Protocol once, devices for free** — new device = one YAML on the node; new protocol = one transport plugin; new node = one fleet entry.

<details>
<summary>Hardware tools</summary>

| Group | Tools | Risk |
|---|---|---|
| Inspect | `hw_list` · `hw_describe` · `hw_read` · `hw_status` | read-only |
| Act | `hw_configure` · `hw_actuate` · `hw_dispense` · `hw_batch_actuate` | approval-gated |
| Safety | `hw_estop` | no approval — node-local |
| Control | `hw_control_start` · `hw_control_stop` · `hw_control_status` | leased deadman |
| Fleet | `hw_fleet_status` · `hw_fleet_discover` · `hw_fleet_halt_all` | multi-node |
| Skills | `hw_skill_list` · `hw_skill_run` · `hw_skill_status` · `hw_skill_stop` | plan-digest approval, node-local |
| Data | `hw_skill_label` | labels a recorded episode's outcome |

</details>

<details>
<summary>Governance</summary>

- **Admission before visibility** — unverified or halt-less devices are demoted to read-only.
- **Signed execution permits** — Ed25519, single-use, nonce-protected, TTL-limited; node verifies and still re-runs local safety.
- **Halt never waits** — node-local, bypasses approval and network.
- **Progressive trust** — autonomy earned per device/action on verified success, lost on failure.

</details>

<details>
<summary>Bundled physical-AI stack</summary>

- **Robot arms** (`leapflow.robot`) — LeRobot-derived: serial bus (Feetech), teleop, camera capture, episode export, policy registry (`vla_local` / `position_control`).
- **Node skills** — calibration (`calib.*`), perception (`perceive.locate`), grasp proposals (`grasp.propose`), model-written motion programs (`motion.sequence`), and learned policies (`policy.run`) run on the node behind the same gate chain; with an episodes directory configured, every actuating run is recorded as a training episode.
- **Models as devices** — a policy or perception server (`model_service` transport, HTTP) appears as an HCP device; inference never moves anything — only a skill's gated motion writes to actuators.
- **Depth cameras** — RealSense integration; frames as governed binary channels.
- **ROS2** — sidecar transport over Unix socket; topics/services/actions map into HCP.

</details>

<details>
<summary>Deployment modes</summary>

| Mode | Description |
|------|-------------|
| **Host only** | Edge node has no LeapFlow runtime. |
| **Host + node** | Host agent + edge `leap hw serve` (recommended). |
| **Node only** | Edge node operates independently. |

**Host + node setup:**

```bash
# Edge node
pip install leapflow
openssl rand -hex 24 > ~/.leapflow/.lhp-token && chmod 600 ~/.leapflow/.lhp-token
LEAPFLOW_LHP_TOKEN="$(cat ~/.leapflow/.lhp-token)" \
  leap hw serve --node-id edge01 --host 0.0.0.0 --port 8767

# Host
export EDGE_LHP_TOKEN='paste-the-token-here'
export LEAPFLOW_FLEET_NODES='[{"node_id":"edge01","endpoint":"tcp://<ip>:8767","token_env":"EDGE_LHP_TOKEN"}]'
leap daemon restart && leap hw list
```

</details>

<details>
<summary>Device onboarding</summary>

```bash
leap hw init my_sensor          # generate hc.v1 template
leap hw validate my_sensor      # parse + check
leap hw probe my_sensor         # connectivity test
leap hw verify my_sensor --as operator   # digest-bound verification
leap hw permit-key              # Ed25519 keypair for permits
```

</details>

---

## Architecture

### Everything is a Signal

```
  Virtual                              Physical
  IM · OS · clipboard · web            arms · cameras · instruments · IoT
          │                                    │
          └──────────────┬─────────────────────┘
                         ▼
              ┌─────────────────────┐
              │  Unified Pipeline   │
              │  normalize → gate → │
              │  dedup → memory →   │
              │  causal fusion      │
              └────────┬────────────┘
                       ▼
              ┌─────────────────────┐
              │  OODA Engine        │
              │  Observe → Orient → │
              │  Decide → Act       │
              └─────────────────────┘
```

Three nested loops on the pipeline:

| Loop | Scope | Example |
|------|-------|---------|
| **Decision** | per turn | Depth adapts to task difficulty; bounded frames compose |
| **Learning** | per outcome | World-model verdicts, skill distillation, trust accrual |
| **Evolution** | per environment delta | Typed capability requirements → resolution before acquisition |

### Everything is a Plugin

Every capability arrives behind a `typing.Protocol` (`@runtime_checkable`). One lifecycle applies to all: reversible registration, fiber state machine, hot-reload, trust ledger.

| Protocol | Role |
|----------|------|
| `ToolPlugin` | Expose tools to the LLM |
| `GatewayAdapterPlugin` | IM platform adapter |
| `LLMProviderPlugin` | LLM inference backend |
| `SignalSource` / `ActiveSignalSource` | Signal sources |
| `HardwareTransport` | Southbound device protocol (entry point) |
| `InferenceStrategy` | VLA / policy inference (entry point) |
| `CVProcessor` | Computer-vision processor |

---

## Reference

### TUI Commands

| Command | Description |
|---------|-------------|
| `/status` | Runtime and backend status |
| `/tool` | Inspect available tools |
| `/config` | View or update settings (tab-completion) |
| `/model` | Show or switch the active model |
| `/usage` | Context and token usage |
| `/board` | Open the monitoring dashboard |
| `/gateway` | Platform connection status |
| `/orient` | Layered orientation and re-entries |

### Safety

| Action | Behavior |
|--------|----------|
| Safe shell (`ls`, `git status`) | Auto-execute |
| Dangerous shell (`sudo`, `rm -r`) | Approval prompt — **[a]lways** for session |
| Destructive (`rm -rf /`, `mkfs`) | Always blocked |
| File writes (safe types) | Auto-approve |
| Hardware actuation | Approval-gated; `hw_estop` always allowed |

### Configuration

```bash
leap config list                  # all settings
leap config set memory.working_max_tokens 12000
```

Inside the TUI: `/config list`, `/config set …` — hot-reloads the active session.

<details>
<summary>Common keys</summary>

| Key | Default | Description |
|-----|---------|-------------|
| `llm.model` | — | Primary model |
| `llm.base_url` | — | OpenAI-compatible endpoint |
| `llm.context_length` | `1000000` | Context budget |
| `memory.working_max_tokens` | `12000` | Working memory budget |
| `tools.terminal_session_enabled` | `false` | Persistent terminal sessions |
| `dashboard.enabled` | `true` | LeapBoard |
| `evolution.enabled` | `false` | Self-evolution |

</details>

---

## Advanced

<details>
<summary><strong>Self-Evolution</strong></summary>

Disabled by default. When enabled, LeapFlow observes environment deltas and evolves its plugin composition under governance:

```bash
leap config set evolution.enabled true
```

Pipeline: observation → adaptation verdict → capability gap → resolution-first check → proposal → LLM code generation → multi-stage validation → dual approval → install at DRAFT trust → progressive accrual.

Even when disabled, the world model still produces verdicts and distils knowledge. See [`docs/world_model_plugin_harness_evolution.md`](docs/world_model_plugin_harness_evolution.md).

</details>

<details>
<summary><strong>Plugin Self-Management</strong></summary>

The agent manages its own plugins at runtime: `plugin_list`, `plugin_generate`, `plugin_install`, `plugin_reload`, `plugin_disable`, `plugin_remove`, `plugin_rollback`.

All mutations require HIGH-risk approval. Trust: `DRAFT → CANDIDATE → VERIFIED → PRODUCTION`. Untrusted plugins run sandboxed (subprocess JSON-RPC). Marketplace plugins are Ed25519-signature verified.

</details>

<details>
<summary><strong>Adaptive-Depth Execution</strong></summary>

The agent loop adapts depth to task difficulty. Hard tasks earn wider budgets; recursive subagents run isolated OODA frames.

Config: `agent.subagent_full_loop`, `agent.reentry_enabled`, `agent.calibration_enabled`. Inspect with `/orient`. Design doc: [`docs/design/adaptive_depth_ooda.md`](docs/design/adaptive_depth_ooda.md).

</details>

<details>
<summary><strong>Workflow Copilot (Preview)</strong></summary>

Multi-tier next-action predictor: L0 hash match (<1ms) → L1 n-gram (<5ms) → L2 embedding (<50ms) → L3 LLM+RAG (200–2000ms). Display overlay pending; predictions are computed and cached internally.

</details>

<details>
<summary><strong>Native Execution Backend (macOS)</strong></summary>

For desktop perception — screen capture, accessibility tree, input injection:

```bash
brew install trycua/tap/cua-driver && leap host doctor
```

Without this, use `--mock-host`.

</details>

<details>
<summary><strong>Multi-Profile</strong></summary>

```bash
LEAPFLOW_PROFILE=work leap
```

Each profile owns its own daemon, config, credentials, and databases under `~/.leapflow/profiles/<name>/`.

</details>

<details>
<summary><strong>Custom Gateway Platform</strong></summary>

Drop a YAML manifest in `~/.leapflow/profiles/<profile>/gateway/manifests/`, implement the adapter Protocol (`connect`, `send`, `disconnect`), and say *"connect to my_platform"* in the TUI.

</details>

<details>
<summary><strong>CLI Reference</strong></summary>

| Command | Description |
|---------|-------------|
| `leap` | Interactive TUI |
| `leap "question"` | One-shot chat |
| `leap teach [goal]` | Record and distill a skill |
| `leap run [prompt]` | Execute a matched skill |
| `leap skills list\|show\|export\|import` | Skill library |
| `leap evolve` | Run the learning boundary |
| `leap board` | LeapBoard dashboard |
| `leap hw list\|serve\|verify\|estop\|…` | Hardware / LHP node |
| `leap daemon start\|stop\|restart\|status` | Manage leapd |
| `leap config list\|set\|show` | Configuration |
| `leap host doctor` | Execution backend check |

Flags: `--mock-host`, `--thinking`

</details>

---

## Development

```bash
make setup && make test && make lint
```

<details>
<summary>More</summary>

```bash
make test-unit              # mock layer, parallel
make test-e2e               # real journeys
make test-impact BASE=origin/main
make test-live              # real provider (needs API key)
```

```
src/leapflow/
├── cli/              ├── engine/           ├── hardware/
├── robot/            ├── tools/            ├── plugins/
├── llm/              ├── memory/           ├── perception/
├── signal_fusion/    ├── causal/           ├── world_model/
├── evolution/        ├── learning/         ├── skills/
├── copilot/          ├── monitor/          ├── dashboard/
├── gateway/          ├── platform/         ├── security/
├── daemon/           ├── scheduler/        ├── domain/
├── storage/          ├── recording/        └── analysis/
```

</details>

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `cua-driver not found` | `brew install trycua/tap/cua-driver` |
| `MCP connection failed` | `leap host doctor` |
| `LLM API key is empty` | `/config llm set --api-key …` |
| Accessibility denied | System Settings → Privacy & Security → grant |
| Device `writable: 0` | `leap hw verify` — unverified devices are read-only |
| `permit_required` | `leap hw permit-key` and configure issuer keys |

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

<div align="center">

**[ModelScope](https://github.com/modelscope)** · [Star](https://github.com/modelscope/leapflow/stargazers) · [Issues](https://github.com/modelscope/leapflow/issues) · [Discussions](https://github.com/modelscope/leapflow/discussions)

*LeapFlow — Learning and Evolving from Actual Practice.*

</div>
