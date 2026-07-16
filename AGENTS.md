# AGENTS.md

This document is the LeapFlow engineering collaboration contract. It is not only a style guide: it defines the design, runtime, UX, safety, and verification rules that every implementation change must follow.

## Design Philosophy

1. **Signal-Driven Intelligence** — All agent intelligence derives from observing real-world signals, not from hardcoded rules. If a behavior cannot be learned from signals, it is not in scope.

2. **Context Pipeline as Core** — Signal → Filter (SNR) → Compress (intent-preserving) → Store (multi-layer) → Retrieve (goal-dependent) → Decide. Every feature and every external signal source, including IM collaboration events, must map to this pipeline before it can drive action.

3. **Progressive Trust** — Never auto-execute on first encounter. Earn autonomy through repeated success: DRAFT → CANDIDATE → VERIFIED → PRODUCTION.

4. **Occam's Razor** — The simplest correct solution wins. Reject complexity that doesn't directly serve user value. Every abstraction must pay for itself.

5. **LLM-Native Design** — Design for LLM reasoning first. Protocols over classes. Declarative over imperative. Context over configuration.

6. **User-Centric Reliability** — User experience is part of correctness. Every change must keep common paths easy, predictable, recoverable, and must not degrade adjacent workflows.

## Code Quality Requirements

- SOLID principles are non-negotiable; implementations must be cohesive, well-factored, and easy to reason about
- Occam's Razor: maximize elegance and efficiency, reject unnecessary complexity
- Design for generalization and universality; prefer reusable domain concepts over one-off special cases
- Easy to extend, avoid hardcoding and hard rules
- Industrial-grade robustness: every external call has timeout, retry, and fallback
- Structured error propagation: failures must flow as typed envelopes (`FailureEnvelope`) with source, category, recoverability, and side-effect state — never as ad-hoc dicts, bare strings, or unclassified exceptions
- User experience is a first-class quality bar: optimize for clarity, ease of use, fast feedback, and graceful recovery
- All comments and docstrings in English
- Type annotations on all public APIs
- No bare except — always specify exception types

## Architecture Principles

- **System Boundary Awareness**: LeapFlow is a multi-entry, multi-module runtime. Changes must account for the affected path across CLI/TUI, leapd, engine, skills/tools, LLM, storage, memory, gateway, hub, and platform adapters.
- **TUI as the Primary User Entry**: The interactive TUI is the default product surface. Preserve streaming feedback, command queue behavior, approval prompts, status bar accuracy, long-input robustness, history, and session continuity.
- **TUI Command Clarity**: Global task-control commands stay short and unambiguous (`/cancel`, `/skip`, `/pause`, `/resume`, `/queue`, `/drop`); teach-mode controls must use the `/teach ...` namespace and should not keep bare compatibility aliases during early iteration.
- **TUI Prompt Ownership**: Input prompt and placeholder rendering must have a single owner. Avoid duplicate prompt sources; placeholder text stays visually subordinate, offset after the prompt, and disappears as soon as the user types.
- **leapd Runtime Consistency**: Daemon-backed behavior must preserve lifecycle correctness: start, stop, restart, status, RPC streaming, cancellation, pending approvals, runtime config reload, multi-client state, and version consistency.
- **Progressive Context Disclosure (PCD)**: Keep one unified execution loop, but never default every turn to full disclosure. Each LLM call must use the smallest sufficient PromptAssemblyPlan for tools, memory, history, reasoning, streaming, and risk; upgrade progressively only when observable signals require it.
- **Gateway as Signal Boundary**: External IM/platform integrations are not just messaging features; they extend LeapFlow's Observe/Orient boundary into collaboration environments. Inbound platform events must enter as structured signals (`BackendEvent` → normalized domain event/message), pass SNR filtering and privacy/safety gates, then feed memory, decision, and action paths according to their classification.
- **Transport-Lifecycle Separation**: Short-lived actions (`ExecutionBackend`/`CliBackend`) and long-lived observations (`BackendEventSource`) are separate responsibilities. Do not implement streaming subscribers, webhooks, polling loops, or CLI NDJSON consumers inside one-shot action execution code.
- **Platform-Neutral Gateway Core**: Gateway core owns protocols, lifecycle, routing, session isolation, approval, audit, and memory integration. Platform adapters own authentication, send semantics, event-source configuration, and schema normalization. Core modules must not import platform SDKs directly.
- **Platform vs App Business Boundary**: Platform layers may define stable contracts and governance primitives (`ActionSpec`, `ActionFailure`, `ActionAuthSpec`, `CapabilityHealthLedger`, approval/feasibility gates, audit, and metadata propagation). Third-party app or vendor specifics — SDK/CLI wire formats, scope names, auth commands, console URLs, error JSON shapes, resource naming, and recovery playbooks — must live in that app's action pack, adapter, backend, or normalizer, never in gateway core.
- **Dependency Inversion**: Core logic depends on Protocol abstractions, never on concrete implementations
- **Protocol over ABC**: Use `typing.Protocol` with `runtime_checkable` for all extension points
- **Event-Driven Communication**: Modules interact through typed events on EventBus, not direct imports
- **Immutable Domain Types**: Use `@dataclass(frozen=True)` or `NamedTuple` for domain objects
- **Config-Driven Behavior**: Thresholds, intervals, feature flags, model budgets, platform capabilities, hub backends, gateway manifests, and paths must be configurable through Settings/env/config layers.
- **Graceful Degradation**: Every optional component (LLM, Hub) can be absent without crash
- **Single Source of Truth**: DuckDB for persistence, EventBus for communication, Settings for configuration
- **Inbound Signal Classification**: Platform events must be classified before they activate the agent. Message/callback events may enter Decide; signal/lifecycle events should be stored or routed without triggering LLM by default; ignored events must be explicit (e.g. self-message, duplicate, blocked scope).
- **Single Recovery Decision Point**: All agent loop errors (LLM, tool, system, security) enter one `RecoveryCoordinator`. No parallel decision paths, no scattered if/break logic. The pipeline is always: `FailureEnvelope` → `RecoveryDecision` → `StrategyOutcome` feedback.
- **Side-Effect Gating**: Recovery actions are gated by `SideEffectState`. Committed or partial side effects block automatic retry; only user-mediated or checkpoint-based resumption is permitted after state mutation.
- **Budget-Constrained Recovery**: Turn-level deadlines, per-category limits, and a global recovery budget prevent infinite retry loops. Every recovery action has an explicit cost; exhaustion triggers a clean halt or user escalation.
- **Recovery Strategy as Protocol**: Recovery strategies implement a `RecoveryStrategy` Protocol (`can_apply` + `decide`), registered by priority, composable, and extensible without modifying the coordinator.

## Path Tree, Configuration, and Secrets Rules

- **Path tree is a product contract**: every LeapFlow-managed path must be declared by `PathLayout`, `ProfileLayout`, `CacheLayout`, or a child layout object. Runtime code must consume layout APIs, never assemble managed paths with ad-hoc string joins.
- **Profile is the runtime boundary**: `profiles/<profile>/` owns profile metadata, config, DBs, memory, skills, gateway state, approval state, audit logs, runtime files, cache roots, and profile-scoped secrets. Cross-profile access requires an explicit layout object.
- **Workspace is context, not ownership**: workspace-local files are limited to `.leapflow/config.yaml` and `.leapflow/workspace.yaml`; profile data and caches stay under the active profile and are addressed by workspace/session ids.
- **Config is layered, not scattered**: durable settings live in `config/user.yaml`, `profiles/<profile>/config/*.yaml`, and optional workspace config. `LEAPFLOW_*` values are process overrides only; env files are not a supported configuration source.
- **`leap config` is the user-facing control plane**: every durable, user-writable setting must be discoverable and mutable through `ConfigService`, `leap config`, and TUI `/config`; do not add one-off setup commands, hidden YAML-only knobs, or new persistent env-first flows.
- **Config catalog is the discovery contract**: writable config fields must expose key, effective value, type, scopes, hot-reload semantics, category, value hint, and description. `leap config keys` stays compact and script-friendly; `leap config list` and `/config list` are the human-readable catalog; `leap config show <key>` and `/config show <key>` are the single-field detail views.
- **TUI config parity is mandatory**: `/config` mirrors `leap config`, supports active-session reload when possible, and must remain self-discoverable through slash completion for subcommands, keys, and simple values. Any new config subcommand must update CLI parser, TUI payload/rendering, completion, README, and tests together.
- **Secrets are refs, never durable plaintext**: long-lived LLM, VLM, aux-provider, Gateway, and Hub credentials must be stored in the vault and referenced as `secret://profile/...` or `secret://global/...`. Config may contain refs, never tokens.
- **Cache declares scope and sensitivity**: cache paths must route through `CacheLayout`/`CacheManager` with `profile`, `workspace`, or `session` scope. Session visual/video/VLM/signal artifacts are sensitive, non-syncable, and TTL/quota managed by index.
- **Safety follows path semantics**: daemon sockets, pid/lock files, runtime state, DuckDB files, vault files, approval grants, audit logs, and memory stores must flow through layout descriptors, path sensitivity, risk, approval, and redaction gates.
- **No legacy aliases**: do not reintroduce global `.env` as persistent config, flat cache roots, profile-root gateway config, inline credential files, `.credential_key`, or `run/` runtime paths.

## Implementation Guidelines

- Define the Protocol first — the contract is the design
- Implement against the Protocol, never against another implementation
- Consider affected user journeys before changing shared flows; do not introduce regressions, broken links, or worse experiences in adjacent paths
- Keep common paths transparent: long-running work must stream progress, surface recoverable errors clearly, and avoid silent stalls.
- For context assembly, prefer manifest-driven progressive disclosure over shortcuts or intent-handler sprawl: expose compact capability indexes, selected schemas, and targeted memory only when the current plan needs them.
- For gateway or IM work, define the signal contract first: event source, normalizer/classifier, trigger policy, session routing, memory/audit path, and outbound action path. Default inbound activation to least privilege (`mention_only` or equivalent), filter self-generated messages before LLM invocation, and keep cross-chat or proactive sends behind Progressive Trust and ApprovalGate.
- Avoid rule-based natural-language fitting by default. Do not add keyword/action-verb/alias enumerations, intent-handler taxonomies, or brittle routing rules when LLM-native capability disclosure, manifests, schemas, protocols, or configuration-driven contracts can solve the problem. If a rule-based method is truly unavoidable for a stable protocol boundary, offline fallback, or safety hard gate, explain the necessity, scope, alternatives, and rollback path to a human and obtain explicit second confirmation before implementation.
- Preserve security and audit paths: dangerous actions, file writes, outbound messages, credentials, and path access must flow through the existing policy, approval, redaction, and audit mechanisms.
- Preserve gateway safety boundaries: inbound credentials stay in CredentialVault; outbound send/write/execute actions go through ApprovalGate; bot self-messages and duplicate events are filtered before routing; platform-specific metadata must remain in `metadata` escape hatches instead of polluting core message types.
- Keep App Connector governance thin: platform core should consume normalized contracts and failures, while app-specific auth scopes, CLI/SDK error parsing, vendor recovery steps, and command templates remain in action packs, adapters, or backend-specific helpers. If a new platform requires changing gateway core business rules, first refactor toward a protocol hook or app-side classifier.
- Maintain backward-compatible migrations for persistent state, configuration, skills, trajectories, sessions, and profile data.
- Write unit tests before or alongside the implementation
- Integrate via EventBus events, not direct function calls between modules
- Every module must be importable standalone without side effects
- No placeholder stubs — implement fully or do not add the code
- ANSI output must check `sys.stdout.isatty()` before emitting escape codes
- For error recovery, route all failures through the `RecoveryCoordinator` — classify into a `FailureEnvelope`, receive a `RecoveryDecision` with an explainable `reason` and `strategy_key`, then feed the outcome back. Never handle errors with ad-hoc if/break in the loop body.
- Recovery strategies are standalone Protocol implementations with `can_apply()` + `decide()`. Add new strategies by registration, never by modifying the coordinator's decision logic.
- When automatic recovery exhausts its budget or encounters non-recoverable failures, emit a structured `InteractionRequest` (typed action, severity, suggested actions, timeout behavior, resumption key) — not raw text appended to conversation.

## Review Requirements

- **Deep review for large changes**: When a change substantially affects architecture, runtime behavior, user flows, persistence, safety, or multiple modules, perform an additional deep review before considering the work complete.
- **Human confirmation for TUI changes**: Any TUI layout or interaction-logic change requires a second human confirmation before it is considered ready.
- **Human confirmation for slash-command paths**: Any change that adds, removes, renames, reroutes, or alters the behavior of a slash command (`/...`) — across the registry, router, in-process REPL, daemon REPL, `command_execute`, completion, and rendering — requires a second human confirmation before it is considered ready. This applies especially to user-experience-facing behavior (dispatch, prompts, confirmations, output, browser/dashboard launches, and error/recovery messaging), which must never be shipped on a single pass.
- **Design goal check**: Verify that the implementation actually achieves the intended design goal and is not just a local patch.
- **Optimality check**: Evaluate whether the solution is the simplest robust design, avoids unnecessary abstractions, and fits the existing architecture.
- **Regression impact check**: Inspect affected modules and user journeys for logic bugs, degraded UX, broken compatibility, slower feedback, weaker diagnostics, or worse failure recovery.
- **SOLID and extensibility check**: Look for responsibility leaks, tight coupling, hardcoded paths/thresholds/rules, magic strings, and choices that reduce generalization or future extension.
- **Fix what the review finds**: If the review identifies correctness, design, UX, SOLID, hardcoding, or extensibility issues, fix and simplify them directly rather than only reporting them.

## Testing Philosophy

- **Unit tests must be hermetic**: no network, no LLM calls
- **py_compile all modified files**: syntax errors caught before test run
- **Import chain verification**: every new module must be importable standalone
- **Existing tests must not regress**: all tests must pass after every change
- **User-facing flows must not regress**: preserve or improve usability, feedback clarity, and failure recovery for impacted paths
- **Verification sequence**: compile → import → unit test → integration (if applicable)
- **Behavior contracts over snapshots**: assert invariants, not frozen values
- **Mock at boundaries only**: mock external I/O (network, disk), never internal logic
- **Change-scoped validation**: Run the most specific relevant tests first, then broaden only as needed: CLI/TUI changes require CLI/TUI tests; leapd changes require daemon RPC/lifecycle tests; storage or memory changes require persistence tests; gateway, IM, event-source, or approval changes require connector lifecycle, event normalization, routing, idempotency, self-message filtering, security/approval, and failure-recovery tests; skills, learning, perception, and copilot changes require their lifecycle or pipeline tests.
- **Recovery strategy isolation**: Each `RecoveryStrategy` must be testable in isolation — verify `can_apply` predicates, `decide` outputs, and side-effect-state gating independently of the coordinator and other strategies.
- **Budget boundary tests**: Verify that recovery budgets exhaust correctly (per-category, per-turn, deadline), that exhaustion produces a deterministic halt decision, and that cost accounting is exact.

## What to Avoid

- Over-engineering: if you need 3+ files for a simple feature, rethink
- Premature optimization: measure first, optimize only bottlenecks
- God objects: no class should exceed 500 lines; approaching this limit requires checking whether policy, state, rendering, protocol, storage, or adapter responsibilities should be split.
- Magic strings: use constants or enums
- Blocking the event loop: all IO must be async or `run_in_executor`
- Hardcoded paths, URLs, thresholds without config escape hatch
- Chinese comments in source code (English only)
- Speculative infrastructure: no hooks or extension points without a concrete consumer
- Mixing long-lived event observation into short-lived action execution; `CliBackend` is for bounded commands, while streaming CLI consumers, webhooks, polling, and WebSocket subscriptions belong behind `BackendEventSource`-style contracts.
- Activating IM agents on all inbound messages by default, skipping self-message filtering, or allowing cross-chat/proactive sends before Progressive Trust and approval policies explicitly permit them.
- Putting third-party app business code into platform core: do not add vendor scopes, lark-cli/SDK JSON parsing, auth command construction, console-specific recovery instructions, or resource-specific branching to gateway-wide modules such as action registries, capability ledgers, approval gates, or engine recovery paths.
- Shortcut-style natural-language fitting and large intent-handler taxonomies; use stable runtime gates plus capability manifests instead. Rule-based keyword/action-verb/alias matching is prohibited by default and requires explicit human second confirmation before implementation when unavoidable.
- Scattered if/break/continue recovery decisions inside the agent loop body; all error handling enters the `RecoveryCoordinator` as a `FailureEnvelope`
- Magic retry counts or unbounded retry loops without budget constraints and deadline enforcement
- Feeding unstructured error text back to the LLM without classification, recoverability assessment, or side-effect awareness
- Multiple parallel error-handling paths for the same failure domain (LLM errors in one handler, tool errors in another, security errors in a third); use a unified classification and coordination pipeline
- Bare `except:` clauses — always specify the exception type
- `# TODO: implement` stubs — implement or don't commit

## Naming Conventions

- **Files**: `snake_case.py` — noun for types (`signal_event.py`), verb for actions (`compress_context.py`)
- **Classes**: `PascalCase` — Protocols suffixed with purpose (`SignalSource`, `SkillStore`)
- **Functions/Methods**: `snake_case` — verb-first (`filter_noise`, `retrieve_context`)
- **Constants**: `UPPER_SNAKE_CASE` — grouped in module-level or dedicated `constants.py`
- **Private**: single underscore prefix (`_internal_method`) — never double underscore
- **Modules/Packages**: short, singular nouns (`perception`, `causal`, `memory`)
- **Events**: past-tense domain verbs (`SignalReceived`, `SkillVerified`, `ContextCompressed`)
- **Config keys**: `dot.separated.lowercase` in env/settings (`copilot.idle_threshold_ms`)
