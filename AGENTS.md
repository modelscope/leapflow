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

## Path Tree, Configuration, and Secrets Rules

- **Layout is the only path authority**: all LeapFlow-managed paths must come from `PathLayout`, `ProfileLayout`, or their child layout objects. Business modules must not construct paths such as `profile_dir / "cache"`, `profile_dir / "approval"`, `data_dir / ".env"`, or `profile_dir / "gateway.yaml"`.
- **Profile is the runtime boundary**: profile metadata, config, DBs, memory, skills, gateway state, approval state, audit logs, cache, runtime files, and profile-scoped secrets live under `profiles/<profile>/` and must be addressed through `ProfileLayout`.
- **Persistent config is structured YAML**: long-lived configuration lives in `config/user.yaml`, `profiles/<profile>/config/*.yaml`, and optional `<workspace>/.leapflow/config.yaml`. `LEAPFLOW_*` process environment values and explicit override files are temporary highest-priority overrides, not persistent config stores.
- **Secrets are never durable plaintext config**: long-lived LLM, VLM, aux provider, Gateway, and Hub credentials must use `secret://global/...` or `secret://profile/...` references resolved through the vault layer. Config files may store refs, never tokens.
- **Cache is scope-aware**: cache paths must declare `profile`, `workspace`, or `session` scope and route through `CacheLayout`/`CacheManager`. Session visual/video/VLM/signal artifacts are sensitive, non-syncable by default, and must be eligible for index-driven TTL/quota cleanup.
- **Runtime and control files are protected**: daemon sockets, pid/lock files, runtime state, DuckDB files, vault files, approval grants/audit, and memory stores must flow through path sensitivity, risk, approval, and redaction metadata before file tools can access them.
- **No legacy aliases unless explicitly approved**: do not reintroduce global `.env` as a main config source, flat cache directories, profile-root gateway config, `.credential_key`, or `run/` runtime paths.

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

## Review Requirements

- **Deep review for large changes**: When a change substantially affects architecture, runtime behavior, user flows, persistence, safety, or multiple modules, perform an additional deep review before considering the work complete.
- **Human confirmation for TUI changes**: Any TUI layout or interaction-logic change requires a second human confirmation before it is considered ready.
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
