# Adaptive Plugin Experiment Workspace

This scratch workspace contains the next-generation adaptive plugin experiment.
It replaces the old lifecycle-only plugin experiment: mechanical lifecycle checks
now live in `tests/`, while this directory focuses on adaptive decision
transparency.

## Current P0 Experiment

`temp/plugin_exp/scripts/adaptive_plugin_exp.py` runs a deterministic scenario
matrix over the adaptive plugin decision layer:

```text
environment fingerprint
→ capability requirements
→ candidate plugin scoring
→ hard exclusions
→ selected plugin set
→ declarative capability plan
→ JSON + Markdown + HTML reports
```

The script does not call an LLM, network, daemon process, approval modal, or real
plugin installation path. It uses synthetic candidates to stress the resolver and
plan logic without coupling this experiment to framework runtime side effects.

## Native DeepSeek Harness Compatibility Experiment

`temp/plugin_exp/scripts/native_dsh_plugin_exp.py` complements the synthetic
matrix with real artifacts from a local DeepSeek Harness checkout. It extracts the
canonical dynamic `REVERSE_TOOL_CODE` and composition fixtures, copies selected
published package manifests/build outputs, and runs LeapFlow's production path:

```text
source inspection → compatibility verdict → isolated profile install
→ restricted Node discovery → tool invocation → wrapper rediscovery/reload → remove
```

The default matrix covers:

| Class | Real source | Expected result |
|---|---|---|
| Directly adaptable | `cordis-host-runner` `REVERSE_TOOL_CODE` | Installs, invokes, reloads, and removes. |
| Partial | The same Host tool plus the runner's real Client fixture | Host tool runs; Client half is explicitly skipped. |
| Adapted package | Real dynamic fixture wrapped as a self-contained pre-built package | Standard DSH package path executes end to end. |
| Architecturally unsuitable | `packages/core/agent-loop` | Rejected before approval by the pluggability taxonomy. |
| Relevant but not self-contained | `packages/fs/tool-fs` | Rejected for npm/peer dependencies and unsupported services. |
| Unsupported composition | Real `CONSUMER_CODE` requiring `greeter` | Declared service dependency is rejected statically. |
| UI-only | Real Client fixture with no public Host tool | Rejected as non-executable in P0. |
| Unbuilt or incomplete | Real TypeScript source / package missing its compiled entry | Source inspection rejects the artifact. |
| Capability attack | Dynamic tool requesting `curl ...; id` | Installs, but invocation is denied before `web_fetch`. |

Run it from the LeapFlow repository root, using the project experiment environment:

```bash
conda run -n leap python temp/plugin_exp/scripts/native_dsh_plugin_exp.py
```

Options:

```bash
python temp/plugin_exp/scripts/native_dsh_plugin_exp.py \
  --harness-root /path/to/deepseek-harness \
  --user-data-root ~/.leapflow \
  --keep-work
```

Each run writes JSON and Markdown evidence to
`temp/plugin_exp/reports/<timestamp>-native-dsh-plugin-exp.*`. Runtime files live
under `temp/plugin_exp/work/native-dsh/<timestamp>/` and are deleted unless
`--keep-work` is set. The registry and profile are isolated from the active user
profile. The default profile's cache and LLM configuration are probed with direct
read-only YAML/existence checks; secret references are not resolved, secret values
are never emitted or copied, and this deterministic experiment sends zero LLM or
network requests.

## P0 Scenario Matrix

| Scenario | Purpose |
|---|---|
| `file_ops_only` | Baseline Python workspace with only `file.ops`; shell-dependent candidates are excluded. |
| `shell_enabled` | Adds `shell.exec`; verifies environment capability changes candidate eligibility. |
| `trust_flip` | Alters trust and reliability evidence so selection flips to another plugin. |
| `risk_limit_read_only` | Enforces a read-only risk ceiling and excludes an external candidate. |
| `missing_dependency` | Selects a tool whose required capability has no provider; plan is not executable. |
| `dependency_cycle` | Selects mutually dependent tools and verifies cycle detection. |
| `unmet_requirement` | Requests an unsupported capability and verifies it remains unmet. |
| `node_workspace_marker` | Changes workspace markers and verifies environment fingerprint changes. |
| `unknown_tool_ingestion` | Converts repeated `unknown_tool` evidence into a `CapabilityRequirement` and resolves it. |

## Run

From the repository root:

```bash
python temp/plugin_exp/scripts/adaptive_plugin_exp.py
python temp/plugin_exp/scripts/adaptive_plugin_exp.py --closed-loop
python temp/plugin_exp/scripts/adaptive_plugin_exp.py --closed-loop --no-live-generation
python temp/plugin_exp/scripts/adaptive_plugin_exp.py --autonomous-long-run
python temp/plugin_exp/scripts/adaptive_plugin_exp.py --autonomous-long-run --no-live-generation
```

Outputs:

- JSON: `temp/plugin_exp/reports/<timestamp>-adaptive-plugin-matrix.json`
- Markdown: `temp/plugin_exp/reports/<timestamp>-adaptive-plugin-matrix.md`
- HTML dashboard: `temp/plugin_exp/reports/<timestamp>-adaptive-plugin-matrix.html`
- Full metadata gaps: `temp/plugin_exp/reports/<timestamp>-real-registry-metadata-gaps.md`
- Store record: `temp/plugin_exp/work/capability_plans.json`

The Markdown and HTML reports now separate built-in, profile-scoped, and external
plugin metadata coverage. This keeps framework regressions visible even when the
active profile contains experimental plugins whose `ToolMetadata` declarations are
not part of the repository.

`--closed-loop` adds an isolated real registry mutation experiment. By default it
uses the real default profile LLM configuration from `~/.leapflow` to generate a
plugin, validates that generated code, installs it through the same
`self_management.plugin_install` handler used by LeapFlow, resolves the capability
plan again, executes the new tool, disables it, removes it, and records every
phase in the report. Use `--no-live-generation` to run the deterministic fixture
variant. The default profile under `~/.leapflow` is read for LLM configuration but
is not mutated by this closed-loop run.

## Strategy And Prioritized TODOs

### P0 — Deterministic decision matrix (implemented here)

- Build scenario matrix entirely under `temp/plugin_exp`.
- Cover environment A/B, trust/reliability flip, risk hard exclusion, missing
  dependency, cycle detection, unmet requirement, and workspace-marker deltas.
- Emit JSON, Markdown, HTML dashboard, and full metadata-gap reports.
- Do not modify framework runtime code.

### P1 — Real registry candidate source (snapshot implemented)

- The experiment now reads `get_registry()` and `candidates_from_registry()` to
  produce a live candidate metadata coverage snapshot.
- Current output includes candidate count, plugin count, coverage ratios,
  conflict count, top metadata gaps, and source-split coverage for built-in vs
  profile-scoped tools.
- Current framework-side metadata pass leaves built-in tools at 100% declared
  `provides_capabilities`; remaining gaps in a local run are profile plugin
  quality issues to feed into later self-improvement scenarios.
- Remaining work: replace or augment selected scenarios with real registry
  candidates once built-in tools have enough declarative metadata.
- Framework-side changes may be needed if built-in ToolMetadata lacks capability
  declarations; confirm before changing runtime plugins.

### P1 — Unknown-tool evidence ingestion (synthetic implemented)

- The scenario matrix now includes `unknown_tool_ingestion`.
- It feeds repeated synthetic `unknown_tool` payloads through
  `CapabilityGapDetector.requirements_from_tool_results()`.
- The resulting `CapabilityRequirement(origin="unknown_tool")` is resolved by
  the same `CapabilityResolver` and included in the plan/report output.
- Real engine failure integration still requires confirmation before changing
  framework-side observation or turn plumbing.

### P2 — Runtime registry mutation smoke (implemented in `--closed-loop`)

- Install/disable/remove a live-generated plugin by default and prove adaptive
  decisions change with live catalog state.
- The experiment uses real `~/.leapflow` LLM configuration for generation, then an
  isolated registry and temporary profile-scoped plugin directory under
  `temp/plugin_exp/work`; it does not mutate the default user profile.
- `--no-live-generation` keeps the deterministic fixture path available for
  offline debugging.
- Each phase is written to `capability_plans.json` and rendered in the Markdown
  and HTML reports as a closed-loop mutation timeline.

### P2 — LeapBoard and slash smoke

- Render `dashboard/templates/capability.yaml` using stored capability-plan data.
- Exercise `/plugin plan --latest` against a seeded profile store.
- Slash/board behavior changes require human confirmation before shipping.

### P3 — Long-run autonomous governance (implemented in `--autonomous-long-run`)

- Repeated structured `unknown_tool` observations are persisted to a durable
  observation store and aggregated into requirements.
- Requirements are enqueued in a proposal queue and passed through
  `AdaptiveEvolutionPolicy`.
- The experiment performs live generation by default, runs the isolated closed
  loop, records probation outcomes, verifies trust promotion, and injects failure
  streak evidence to drive quarantine.
- `--no-live-generation` keeps a deterministic offline variant.

### P3 — Closed-loop autonomous evolution

- Connect real observation signals, resolver output, approval policy,
  `plugin_generate`, `plugin_install`, and post-use trust/reliability feedback.
- Keep first-time install behind Progressive Trust and ApprovalGate.
- This requires framework-side orchestration changes and must be designed with a
  separate approval/safety review.
