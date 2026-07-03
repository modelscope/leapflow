# AGENTS.md

## Design Philosophy

1. **Signal-Driven Intelligence** — All agent intelligence derives from observing real-world signals, not from hardcoded rules. If a behavior cannot be learned from signals, it is not in scope.

2. **Context Pipeline as Core** — Signal → Filter (SNR) → Compress (intent-preserving) → Store (multi-layer) → Retrieve (goal-dependent) → Decide. Every feature must map to this pipeline.

3. **Progressive Trust** — Never auto-execute on first encounter. Earn autonomy through repeated success: DRAFT → CANDIDATE → VERIFIED → PRODUCTION.

4. **Occam's Razor** — The simplest correct solution wins. Reject complexity that doesn't directly serve user value. Every abstraction must pay for itself.

5. **LLM-Native Design** — Design for LLM reasoning first. Protocols over classes. Declarative over imperative. Context over configuration.

## Code Quality Requirements

- SOLID principles are non-negotiable
- Occam's Razor: maximize elegance and efficiency, reject unnecessary complexity
- Design for generalization and universality
- Easy to extend, avoid hardcoding and hard rules
- Industrial-grade robustness: every external call has timeout, retry, and fallback
- All comments and docstrings in English
- Type annotations on all public APIs
- No bare except — always specify exception types

## Architecture Principles

- **Dependency Inversion**: Core logic depends on Protocol abstractions, never on concrete implementations
- **Protocol over ABC**: Use `typing.Protocol` with `runtime_checkable` for all extension points
- **Event-Driven Communication**: Modules interact through typed events on EventBus, not direct imports
- **Immutable Domain Types**: Use `@dataclass(frozen=True)` or `NamedTuple` for domain objects
- **Config-Driven Behavior**: Thresholds, intervals, feature flags — all configurable via env vars
- **Graceful Degradation**: Every optional component (LLM, Hub, OS Host) can be absent without crash
- **Single Source of Truth**: DuckDB for persistence, EventBus for communication, Settings for configuration

## Implementation Guidelines

- Define the Protocol first — the contract is the design
- Implement against the Protocol, never against another implementation
- Write unit tests before or alongside the implementation
- Integrate via EventBus events, not direct function calls between modules
- Every module must be importable standalone without side effects
- No placeholder stubs — implement fully or do not add the code
- ANSI output must check `sys.stdout.isatty()` before emitting escape codes

## Testing Philosophy

- **Unit tests must be hermetic**: no network, no OS Host, no LLM calls
- **py_compile all modified files**: syntax errors caught before test run
- **Import chain verification**: every new module must be importable standalone
- **Existing tests must not regress**: all tests must pass after every change
- **Verification sequence**: compile → import → unit test → integration (if applicable)
- **Behavior contracts over snapshots**: assert invariants, not frozen values
- **Mock at boundaries only**: mock external I/O (network, disk, OS Host), never internal logic

## What to Avoid

- Over-engineering: if you need 3+ files for a simple feature, rethink
- Premature optimization: measure first, optimize only bottlenecks
- God objects: no class should exceed 300 lines
- Magic strings: use constants or enums
- Blocking the event loop: all IO must be async or `run_in_executor`
- Hardcoded paths, URLs, thresholds without config escape hatch
- Chinese comments in source code (English only)
- Speculative infrastructure: no hooks or extension points without a concrete consumer
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
