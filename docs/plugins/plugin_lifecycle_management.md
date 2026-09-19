# Plugin Lifecycle Management Strategy

> **Scope**: ALL plugins — built-in AND third-party — across Tool, Gateway, LLM-provider, and Signal-Source subsystems.  
> **Audience**: LeapFlow operators, plugin authors, and platform engineers.  
> **Date**: 2026-08-19  
> **Package**: the plugin subsystem is the first-class `leapflow.plugins` package — contracts (`protocol.py`), registry (`registry.py`), lifecycle (`scoped_registry.py`), built-ins (`tool_plugins/`), isolation (`sandbox/`), distribution (`marketplace/`); tool behaviour stays in `leapflow.tools`.  
> **Companion document**: `docs/plugins/third_party_plugin_development.md` (interface/API details and development guide — referenced, not duplicated here).

---

## Terminology

| Term | Definition |
|------|-----------|
| **PluginFiber** | A per-plugin lifecycle state-machine instance (`domain/plugin_fiber.py`). Tracks runtime state transitions (PENDING/DRAFT/LOADING/ACTIVE/FAILED/UNLOADING/DISPOSED) and owns an EffectScope for deterministic cleanup. |
| **EffectScope** | Hierarchical, LIFO-ordered cleanup collector (`domain/effect_scope.py`). Guarantees safe teardown on dispose. |
| **Trust Level** | Progressive reliability gradient (DRAFT → CANDIDATE → VERIFIED → PRODUCTION) earned by consecutive successes, persisted in DuckDB. |
| **Generation Counter** | Module-level monotonic integer; each new PluginFiber receives a unique generation. Engine caches key on `(id(plugin), generation)` to detect reloads. |
| **ScopedRegistry** | Composition wrapper (`plugins/scoped_registry.py`) that binds PluginFibers to the underlying ToolPluginRegistry so dispose == unregister. |
| **Publish** | `ToolPluginRegistry.publish_plugin_tools(plugin)` — how a plugin registered after boot (install, hot-reload) enters the live catalog and handler table without a full reassemble. |
| **Per-turn snapshot** | Each engine turn copies `dict(registry.tool_handlers)` at turn start. Mid-turn reload/disable cannot disrupt in-flight execution. |
| **ApprovalGate** | Security gate classifying mutation actions at `RiskLevel.HIGH` with `allow_permanent=False` (`security/risk.py`). |
| **PluginHealthProducer** | Monitor-subsystem producer emitting `Finding` alerts on trust degradation or error-rate spike (advisory only). |
| **PluginAdvisor** | Stateless scoring engine computing promote/investigate/demote recommendations from trust + stats. |

---

## 1. Lifecycle Model — Three Orthogonal Axes

A plugin's state is the **composition** of three independent axes: Runtime, Trust, and Operational. Any combination is valid (e.g., a PRODUCTION-trust plugin can be operationally disabled, with its fiber DISPOSED).

### 1.1 Runtime Lifecycle (PluginFiber)

```
┌─────────┐ begin_loading() ┌─────────┐  activate()  ┌────────┐ begin_unload() ┌───────────┐ dispose() ┌──────────┐
│ PENDING │────────────────→│ LOADING │─────────────→│ ACTIVE │───────────────→│ UNLOADING │─────────→│ DISPOSED │
└─────────┘                 └─────────┘              └────────┘                └───────────┘          └──────────┘
    │                            │                                                                        ▲
    │                            │ (init fails)                                                           │
    │                            ▼                                                                        │
    │                      ┌────────┐                                                                     │
    │                      │ FAILED │─────────────── dispose() (early cleanup) ───────────────────────────→│
    │                      └────────┘                                                                     │
    │                         ▲  │                                                                        │
    │                         │  │ retry()/begin_loading()                                                │
    │                         │  ▼                                                                        │
    │                      ┌─────────┐                                                                    │
    │                      │ LOADING │ (retry path → ACTIVE)                                              │
    │                      └─────────┘                                                                    │
    │                                                                                                     │
    └──── activate() (fast path: no async init) ──→ ACTIVE                                               │
    └──── dispose() (early cleanup) ──────────────────────────────────────────────────────────────────────→┘
```

**State table**:

| State | Meaning | Transitions out |
|-------|---------|----------------|
| `PENDING` | Created, awaiting activation or async init | `DRAFT`, `LOADING`, `ACTIVE` (trusted fast path), `DISPOSED` |
| `DRAFT` | Isolated candidate; no live handlers are published | `ACTIVE`, `DISPOSED` |
| `LOADING` | Async initialization in progress (dependency resolution) | `ACTIVE`, `FAILED`, `DISPOSED` |
| `ACTIVE` | Fully operational, tools registered and available | `UNLOADING` |
| `FAILED` | Initialization failed; retryable via `retry()`/`begin_loading()` | `LOADING`, `DISPOSED` |
| `UNLOADING` | Graceful teardown in progress | `DISPOSED` |
| `DISPOSED` | Terminal; EffectScope cleaned, no longer usable | *(none)* |

**Key properties**:
- Transitions are enforced by `_VALID_TRANSITIONS` dict — illegal transitions raise `IllegalStateTransition`.
- Each fiber has a monotonically increasing `generation` (from `_next_generation()`).
- Dispose is idempotent and exception-safe (each effect runs in try/except; failures are logged, not propagated).
- Children scopes are disposed before parent effects (reverse creation order).
- The `FAILED` state captures initialization errors and supports retry (`FAILED → LOADING → ACTIVE`) for async-init paths. Current built-in/profile ToolPlugin registration mostly uses the fast path `PENDING → ACTIVE`.
- Cleanup is scope-based: any resource explicitly registered on `fiber.scope` is disposed automatically with the fiber. EventBus or interceptor cleanup must be wired through such an effect before it becomes scope-bound.

**Concurrency model**: Single-threaded asyncio. The module-level `_generation_counter` and fiber state mutations have no explicit lock — correctness relies on the cooperative event loop. Comment in source: *"if multi-threaded plugin lifecycle management is added later, this counter must be guarded by a threading.Lock"*.

### 1.2 Trust Lifecycle (Progressive Trust)

```
                 ≥5 consecutive       ≥20 consecutive      ≥50 consecutive
                   successes             successes             successes
┌───────┐      ┌───────────┐        ┌──────────┐        ┌────────────┐
│ DRAFT │─────→│ CANDIDATE │───────→│ VERIFIED │───────→│ PRODUCTION │
└───────┘      └───────────┘        └──────────┘        └────────────┘
    ▲               │                    │                    │
    │               │ ≥3 consec.         │ ≥3 consec.        │ ≥3 consec.
    │               │ failures           │ failures          │ failures
    │               ▼                    ▼                    ▼
    │          [demote -1]           [demote -1]         [demote -1]
    │
    └──── hard failure (internal_defect) at ANY level ──→ FREEZE to DRAFT (permanent)
```

**Actors & data flow**:
1. `_execute_general_tool()` records `(tool_name, ok, duration)` → `TurnUsageTracker`.
2. `TurnUsageTracker` forwards to `PluginUsageTracker` (process-global, cross-turn accumulator).
3. `PluginUsageTracker.record()` resolves tool → plugin_id via a lazy reverse index (rebuilt on registry version change), then forwards to `PluginTrustLedger.record_success()` / `record_failure()`.
4. `_PersistingTrustLedger` (subclass) flushes to DuckDB **only on level transitions** (not per-call), keeping writes off the hot path.
5. `atexit` handler `persist_plugin_trust_state()` ensures final counter state survives orderly process exit.

**Persistence**:
- Store: `PluginStatsStore` → DuckDB singleton table `plugin_trust_state` (JSON blob).
- Location: `profiles/<profile>/db/plugin_stats.duckdb`.
- Restored at first `_wire_plugin_stats_sink()` call (session factory boot).

### 1.3 Operational State

| State | Meaning | Code-enforced? |
|-------|---------|:--------------:|
| **enabled** | Fiber is ACTIVE, tools registered in catalog | ✅ ENFORCED |
| **disabled** | Fiber DISPOSED via `plugin_disable` tool; tools removed | ✅ ENFORCED |
| **disabled-at-boot** | Listed in `Settings.disabled_plugins`; skipped by `get_all_plugins()` | ✅ ENFORCED (confirmed: `plugins/tool_plugins/__init__.py`) |
| **quarantined** | Disabled + trust frozen to DRAFT + flagged for investigation | ⚠️ **RECOMMENDED POLICY** (not automated; requires human `plugin_disable` + hard failure record) |
| **removed** | Fiber DISPOSED via `plugin_remove`; tools unregistered and optional profile source file deleted | ✅ ENFORCED |

### 1.4 Composite State Transition Table

| Operational | Fiber State | Trust Level | Meaning |
|-------------|-------------|-------------|---------|
| enabled | ACTIVE | DRAFT | Newly installed, untrusted, operating |
| enabled | ACTIVE | PRODUCTION | Fully trusted, auto-approves reload |
| disabled | DISPOSED | (any) | Not executing; trust state preserved |
| quarantined | DISPOSED | DRAFT (frozen) | Under investigation; cannot promote |
| removed | DISPOSED | (orphaned in DB) | Clean removal; trust row is residual |

---

## 2. Built-in vs Third-Party Differences

| Dimension | Built-in Plugins | Third-Party Plugins |
|-----------|-----------------|---------------------|
| **Discovery** | Hardcoded module list in `plugins/tool_plugins/__init__.py` → `get_all_plugins()` | Profile-dir install (`plugin_install` tool) or marketplace fetch |
| **Boot sequence** | `discover_builtin()` → `register()` → `bind_runtime()` → `assemble()` → `adopt_existing_plugins()` | `plugin_install` → staging → bounded sandbox smoke/behavior tests → DRAFT fiber → atomic publish |
| **Initial trust** | Implicitly DRAFT (but never demoted/frozen in practice — no failure path for well-tested built-ins) | Explicitly DRAFT; must earn promotion through usage |
| **Fiber creation** | `adopt_existing_plugins()` at first `get_scoped_registry()` access; starts in ACTIVE | `create_draft_fiber()` → `stage_plugin()` → `promote_draft()` after tests |
| **Approval** | None for registration (they ARE the system); mutations still gated | ALL mutations gated (HIGH risk, no permanent grants) |
| **Isolation** | In-process (same asyncio loop) | Optionally sandboxed (subprocess JSON-RPC via `SandboxHost`); `requires_sandbox` manifest flag defaults `True` |
| **Reload** | `reload(plugin_id)` via scoped registry; version bump + cache invalidation | Same mechanism, but PRODUCTION trust → auto-approve; below PRODUCTION → explicit approval |
| **Disabling** | Permitted (except `self_management`); removes tools until `plugin_enable` | Same mechanism; human-gated |
| **Self-protection** | `self_management` plugin refuses self-disable | N/A |

---

## 3. Governance Matrix

| Action | Actor | Approval Gate | Auto-approve condition | Config flags | Audit trail |
|--------|-------|---------------|----------------------|--------------|-------------|
| **plugin_list** | Agent tool | None | Always | — | No (read-only) |
| **plugin_status** | Agent tool | None | Always | — | No (read-only) |
| **plugin_versions** | Agent tool | None | Always | `ProfileLayout.plugin_versions_dir` | No (read-only) |
| **plugin_propose** | Agent tool | None | Always (proposal only, no runtime mutation) | Event-sourced lifecycle store | Yes — `proposal.created` |
| **assess_compatibility** | Agent tool | None | Always (read-only manifest assessment; no file/runtime mutation) | — | No (read-only) |
| **plugin_generate** | Agent tool | Content approval | Never for proposal-backed generation | `plugin_generation_enabled`; LLM provider; CAS | Yes — generated artifact and approval lifecycle events |
| **/plugin generate** | User (slash command) | None (user invocation = consent) | Always auto-approved (user-initiated); installs at DRAFT trust level | `plugin_generation_enabled` must be `True`; needs an LLM provider | Yes — install action descriptor recorded |
| **plugin_install** | Agent tool | `ApprovalGate` → HIGH, `allow_permanent=False` | Never (always requires human) | `plugin_install_dir`, event lifecycle/version stores, `plugin_marketplace_root/url`, `plugin_marketplace_trusted_pubkeys` | Yes — action descriptor and lifecycle event recorded |
| **plugin_rollback** | Agent tool | `ApprovalGate` → HIGH, `allow_permanent=False` | Never (always requires human) | `ProfileLayout.plugin_versions_dir` | Yes |
| **plugin_reload** | Agent tool | `ApprovalGate` → HIGH, `allow_permanent=False` | Trust == PRODUCTION (auto-approved) | proposal/version stores when behavior tests or version labels are used | Yes |
| **plugin_disable** | Agent tool | `ApprovalGate` → HIGH, `allow_permanent=False` | Never (always requires human) | — | Yes |
| **plugin_enable** | Agent tool | `ApprovalGate` → HIGH, `allow_permanent=False` | Never (always requires human) | — | Yes |
| **plugin_remove** | Agent tool | `ApprovalGate` → HIGH, `allow_permanent=False` | Never (always requires human) | — | Yes |
| **marketplace uninstall** | `MarketplaceClient.uninstall()` | File-only low-level primitive; prefer `plugin_remove` for live registry cleanup | N/A | — | File deletion only |
| **boot-time disable** | Config | N/A (pre-registration) | Automatic (from `Settings.disabled_plugins`) | `disabled_plugins` | Logged at INFO |
| **/plugin** slash | Human (TUI/CLI) | None (read-only) | Always | — | — |

**Fail-closed guarantee**: When no `plugin_approval_gate` is installed (non-daemon mode), ALL mutation tools return an error. The system never falls open.

**Risk classification** (`security/risk.py:222`): Any action with `metadata.platform == "plugin_management"` is forced to `RiskLevel.HIGH` + `allow_permanent=False`, preventing permanent "always allow" grants. This is defense-in-depth — even if caller metadata is misconfigured.

---

## 4. Real-World Scenario Playbooks

### 4.1 First-Time Install of an Untrusted Third-Party Plugin

**Trigger**: User or agent decides a new capability is needed.

**Sequence**:
1. **Generate**: `plugin_generate(proposal_id="...")` → LLM produces code → `PluginValidator` multi-stage check → profile CAS write → explicit proposal-content approval. No live registry mutation occurs.
2. **Install request**: `plugin_install(proposal_id="...")` resolves the approved CAS artifact and requests a separate mutation approval. Direct code/marketplace installs still use the mutation gate.
   - **Compatibility pre-gate (marketplace path only)**: the resolved manifest is run through `assess_plugin()` (the Compatibility Assessment Engine) *before* anything else. An `INCOMPATIBLE` verdict is **rejected here with a structured error, before any file write**; an `ADAPTABLE` verdict proceeds and its adaptation notes are attached to the install result.
3. **Approval gate**: `ActionDescriptor.platform_action("plugin_management", "install", {...})` → `gate.evaluate()` → user prompted (HIGH risk, one-time).
4. **Duplicate check**: If `plugin_id` already exists in registry → immediate rejection with error.
5. **Validation**: `PluginValidator.validate()` re-runs (defense-in-depth, even for marketplace code).
6. **File write**: Code written to `ProfileLayout.plugins_dir / <plugin_id>.py`.
7. **Sandbox smoke test**: `SandboxHost` starts subprocess → loads plugin → `ping()` + `list_tools()` → verifies conformance in isolation.
8. **Registration**: `_register_inprocess()` — import module → `scoped_register(plugin, fiber)` → `fiber.activate()`.
9. **Behavior tests**: If installation is linked to a `PluginProposal` with `test_cases`, the live plugin handlers must return the expected subsets before the install is accepted.
10. **Version snapshot**: Code installs can record a version label or content hash under `ProfileLayout.plugin_versions_dir`; active pointer metadata retains the proposal id when present.
11. **Runtime dep injection**: `bind_runtime(**last_bound_deps)` distributes existing deps to new plugin.
12. **Version bump**: `registry.notify_mutation()` → engine cache invalidated → next turn sees new tools.
13. **Trust**: Starts at DRAFT. First 5 successful calls → CANDIDATE.

**Rollback on failure** (at any step 5–10):
- Fiber disposed (EffectScope cleans registered tools).
- Module removed from `sys.modules`.
- File deleted from plugins dir.

**Guardrails**:
- Ed25519 signature verification (marketplace path, when `trusted_pubkeys` configured).
- SHA-256 checksum integrity.
- Sandbox timeout (30s default per invoke).
- No permanent approval grants possible.

| Step | Automated today? |
|------|:----------------:|
| Validation | ✅ |
| Approval prompt | ✅ |
| Sandbox smoke | ✅ |
| Rollback | ✅ |
| Trust accrual | ✅ |

### 4.2 Routine Hot-Upgrade / Reload

**Trigger**: Plugin source file updated (bug fix, new tool added) — agent or human calls `plugin_reload(plugin_id="...")`.

**Sequence**:
1. **Approval check**: If trust == PRODUCTION → auto-approved. Otherwise → human approval required (HIGH risk).
2. **Dispose old fiber**: `old_fiber.begin_unload()` → `old_fiber.dispose()` → EffectScope cleanup removes old tools from registry.
3. **Re-import**: `importlib.reload(sys.modules[module_path])` → fresh module instance.
4. **Register new instance**: `scoped_register(fresh_plugin, new_fiber)` → new tools added.
5. **Activate**: `new_fiber.activate()`.
6. **Re-inject deps**: `bind_runtime(**registry.last_bound_deps)`.
7. **Version bump**: `registry.notify_mutation()`.

**In-flight turn safety**: Per-turn snapshot guarantees. The snapshot is `dict(registry.tool_handlers)` copied at turn start. An in-flight turn holds a reference to old handlers; the reload mutates the registry underneath but the old turn's dict is unaffected. New turns after reload pick up fresh handlers.

**Cache invalidation**: Engine caches tool catalog with key `((id(dp), dp.version), len(tool_definitions))`. Since the new plugin instance has a different `id()` and the version counter is bumped, the next turn rebuilds the catalog.

**What persists**: Trust level carries over (keyed by `plugin_id`, not by object identity). Usage stats deque continues accumulating.

| Aspect | Automated today? |
|--------|:----------------:|
| Approval (PRODUCTION) | ✅ auto |
| Approval (below PRODUCTION) | ✅ human prompt |
| Dispose + re-register | ✅ |
| Snapshot isolation | ✅ |
| Cache invalidation | ✅ |
| Dep re-injection | ✅ |

### 4.3 Plugin Misbehavior / Error-Rate Spike

**Trigger**: Plugin's error rate exceeds 25% (rolling window, min 5 calls).

**Detection chain**:
1. `PluginUsageTracker` records failures → `PluginTrustLedger` accumulates consecutive failures → after 3 → demotion.
2. `PluginHealthProducer.observe()` (polled every ~5 min by MonitorManager):
   - Detects trust degradation (level drop since last observation) → emits `Finding(severity=NOTABLE)`.
   - Detects error rate > 25% → emits `Finding(severity=ALERT)` with suggested actions: inspect + disable.
3. `PluginAdvisor.recommend()` (on-demand, triggered by `plugin_status` query):
   - Error rate > 30% + trust ≥ VERIFIED → recommends "demote".
   - Error rate > 20% → recommends "investigate".

**Response** (current state):
- ⚠️ **Advisory only**. PluginHealthProducer does NOT auto-disable. It surfaces `SuggestedAction(name="plugin_disable", kind="approval")` in the Finding, requiring human or agent to act.
- Trust demotion IS automatic (3 consecutive failures → drop one level).
- Human decision: `plugin_disable(plugin_id="...")` → approval prompt → fiber disposed.

**RECOMMENDED POLICY** (not enforced by code today):
- Auto-quarantine threshold: If trust drops to DRAFT AND error rate > 50% within a window → auto-emit `plugin_disable` recommendation with `kind="urgent"`.
- Alert escalation: repeated ALERT findings for same plugin within 3 observation cycles → escalate to operator notification.

| Aspect | Automated today? |
|--------|:----------------:|
| Trust demotion on failures | ✅ |
| Health finding emission | ✅ |
| Advisor recommendation | ✅ (on query) |
| Auto-disable | ❌ Advisory only |
| Quarantine workflow | ❌ Manual |

### 4.4 Security Incident — Malicious or Compromised Plugin

**Trigger**: Operator discovers a plugin is exfiltrating data or executing unauthorized actions.

**Immediate response**:
1. **Disable**: `plugin_disable(plugin_id="...")` → approval (always required, even in emergency) → fiber DISPOSED → tools removed from registry.
2. **Hard failure record**: If discovered through tool execution (e.g., `_execute_general_tool` catches an internal defect): `trust_ledger.record_failure(plugin_id, hard=True)` → FROZEN to DRAFT permanently.
3. **Audit inspection**: Check approval logs, usage stats, finding history.

**Full removal**:
- `plugin_remove(plugin_id, delete_source=True)` performs the live lifecycle operation:
  - disposes the fiber,
  - unregisters the plugin and tools from the live registry,
  - drops reload metadata and `sys.modules` entry,
  - deletes the profile-scoped source file when requested.
- `MarketplaceClient.uninstall(name)` remains a low-level file deletion primitive; use `plugin_remove` for live runtime cleanup.

**Correct removal sequence**:
1. `plugin_remove(plugin_id)` → disposes fiber, removes from registry, deletes source file.
2. Optional daemon restart verifies the plugin does not reappear.

**Rollback**: If wrongly accused → `plugin_enable(plugin_id)` re-imports and re-registers. Trust state remains frozen (requires manual trust ledger reset via DuckDB or code intervention — no tool exposes unfreezing today).

| Aspect | Automated today? |
|--------|:----------------:|
| Disable (fiber dispose) | ✅ (with approval) |
| Hard freeze trust | ✅ (on internal_defect) |
| File deletion | ✅ (`plugin_remove(delete_source=True)` or low-level marketplace uninstall) |
| Live fiber disposal on remove | ✅ |
| Trust unfreeze | ❌ No exposed tool |

### 4.5 Duplicate Plugin ID / Version Conflict

**Trigger**: Attempting to register a plugin whose `plugin_id` matches an existing entry.

**Response**: `ToolPluginRegistry.register()` raises `ValueError` immediately. The install handler catches this and returns a structured error message. No partial state is left.

**Version conflict in marketplace**: Manifest includes `version` field; `MarketplaceClient.install()` fetches by name, not version. If the same name with a different version is installed, it overwrites the file. To upgrade without conflict: `plugin_reload` after file replacement.

| Aspect | Automated today? |
|--------|:----------------:|
| Duplicate rejection | ✅ |
| Version conflict prevention | ⚠️ Partial (no version comparison logic) |

### 4.6 Resource Governance

| Resource | Mechanism | Default | Enforced? |
|----------|-----------|---------|:---------:|
| Tool execution timeout | `_execute_general_tool` wraps the shared `invoke_tool_handler(...)` call with `asyncio.wait_for` | Engine-level timeout (configurable) | ✅ |
| Sandbox invoke timeout | `SandboxHost.invoke_timeout_s` | 30s | ✅ |
| Sandbox subprocess lifecycle | `SandboxHost.stop()` kills worker process | — | ✅ |
| Usage deque memory | `deque(maxlen=500)` per tool in `PluginUsageTracker` | 500 samples | ✅ |
| Plugin generation gating | `Settings.plugin_generation_enabled` | `True` (enabled; set false to disable synthesis) | ✅ |
| Per-plugin resource quota | — | — | ❌ Not implemented |
| Tool call rate limiting | — | — | ❌ Not implemented |

### 4.7 Deprecation & Clean Removal

**Intended sequence**:
1. Mark plugin as deprecated (no formal mechanism today — operational convention).
2. `plugin_disable(plugin_id)` → fiber DISPOSED → EffectScope runs LIFO cleanup → tools unregistered.
3. Delete source file from `ProfileLayout.plugins_dir`.
4. *Trust state persists as orphaned row in DuckDB* — not automatically cleaned. This is by design (audit trail), but `PluginStatsStore` has no GC mechanism.

**What EffectScope guarantees**:
- All registered effects fire in reverse order.
- Exception-safe: one failing cleanup does not prevent remaining cleanups.
- Child scopes are disposed before parent scope.
- Idempotent: calling `dispose()` again is a no-op.

**Residuals after removal**:
- Trust state row in DuckDB (harmless but accumulates).
- Usage deque entries in `PluginUsageTracker._samples` (keyed by tool name — will not match new tools unless same names reused; bounded by maxlen).
- `_fibers` dict entry in `ScopedToolRegistry` (fiber marked DISPOSED; not pruned).

### 4.8 Restart & Persistence

| What | Survives restart? | Mechanism |
|------|:-----------------:|-----------|
| Trust levels + streak counters | ✅ | `PluginStatsStore` DuckDB, loaded at `_wire_plugin_stats_sink()` |
| Frozen (hard-failed) set | ✅ | Serialized in trust state JSON |
| Usage sample deques | ❌ | In-memory only; bounded deque resets to empty |
| Fiber objects | ❌ | Recreated at boot via `adopt_existing_plugins()` (built-ins) or re-install (third-party) |
| Installed third-party files | ✅ | Filesystem under `ProfileLayout.plugins_dir` |
| Third-party re-registration | ✅ | Profile-scoped plugin files are discovered from `ProfileLayout.plugins_dir` at registry boot, respecting `disabled_plugins` |
| `disabled_plugins` config | ✅ | `config.yaml` / Settings |

**Profile discovery**: third-party plugins installed via `plugin_install` are written to `ProfileLayout.plugins_dir`. At registry boot, `discover_profile_plugins()` scans that directory, loads each `.py` file with a file-backed import spec, attaches source-path metadata for reload, and registers plugins not blocked by `disabled_plugins`.

### 4.9 Multi-Instance / Concurrent TUI

**Architecture fact**: Plugins are **process-global** on the daemon. The `ToolPluginRegistry` is a module-level singleton; `ScopedToolRegistry` wraps it. All TUI sessions connected to the same daemon share one plugin set.

**Implications**:
- `plugin_disable` removes tools for ALL sessions (current and future turns).
- `plugin_reload` upgrades the plugin for ALL sessions.
- In-flight turns (any session) are safe due to per-turn handler snapshot.
- Trust accrual is global (all sessions contribute to the same `PluginUsageTracker`).
- `plugin_install` adds tools visible to ALL sessions after their next turn.

**Boundary**:
- **Process-scope**: Plugin registration, trust ledger, usage tracker, fiber state.
- **Session-scope**: Per-turn handler snapshot (isolated), `TurnUsageTracker` (per-session, forwards to global).
- **Workspace-scope**: None for plugins today. A workspace cannot have its own plugin set (plugins are profile-scoped).

---

## 5. Observability & Audit

### 5.1 Introspection Tools (Agent-accessible)

| Tool | Output | Requires approval? |
|------|--------|:------------------:|
| `plugin_list` | All plugins across Tool/Gateway/LLM subsystems plus a live `capability_report` covering plugin support, self-evolution readiness, runtime dependencies, and limitations | No |
| `plugin_status(plugin_id)` | Detailed: tools list, dependencies, fiber state, generation, trust level, usage stats, advisor recommendation | No |
| `/plugin` (TUI slash) | Human-readable list | No |
| `/plugin status <id>` (TUI slash) | Human-readable detail | No |

### 5.2 Health Metrics (Monitor Subsystem)

`PluginHealthProducer` (domain: `plugin_health`, polled ~5 min):
- **Trust degradation finding**: Emitted when trust level drops between observations. Severity: NOTABLE.
- **High error rate finding**: Emitted when error rate > 25% (min 5 calls). Severity: ALERT. Includes suggested actions (inspect, disable).
- **Dedup**: Keyed by `trust_degrade:<plugin_id>:<level>` and `error_rate:<plugin_id>`.

### 5.3 Usage Statistics

`PluginUsageTracker` maintains per-tool rolling stats:
- Total calls, successes, failures.
- Average duration (ms), P95 duration.
- Error rate (ratio).
- Window: last 500 samples per tool (configurable).

Accessed via `plugin_status` tool or `PluginAdvisor.recommend()`.

### 5.4 Audit Trail

| Event | Audit mechanism |
|-------|----------------|
| Mutation approval (install/reload/disable/enable) | `ApprovalGate` records `ActionDescriptor` + decision in approval audit log |
| Trust level transitions | `_PersistingTrustLedger._flush()` writes to DuckDB (durable record of level at transition time) |
| Hard failure freeze | Persisted in trust state `frozen` set |
| Health findings | MonitorManager's finding history (in-memory; not persisted beyond session) |
| Plugin generation attempts | Not persisted (ephemeral LLM call) |

### 5.5 Current Limits

- No long-term finding persistence (findings are session-scoped in MonitorManager).
- No audit log of individual tool call results per plugin (only aggregate stats).
- No dashboard UI for plugin health (would require LeapBoard integration).
- Trust state is a single JSON blob — no time-series history of trust transitions.

---

## 6. Policy Defaults & Recommendations

### 6.1 Recommended Default Thresholds

| Parameter | Current default | Recommendation | Rationale |
|-----------|:--------------:|:--------------:|-----------|
| `candidate_at` | 5 | 5 | Low bar for initial promotion; reasonable for discovery |
| `verified_at` | 20 | 20 | Enough signal to confirm basic reliability |
| `production_at` | 50 | 50 | High bar for auto-approve privilege |
| `demote_after` | 3 | 3 | Quick response to regressions |
| Error rate alert threshold | 25% | 25% | Below would be noisy; above misses real issues |
| Advisor investigate threshold | 20% | 20% | Proportional early warning |
| Advisor demote threshold | 30% | 30% | Action-worthy signal |
| Sandbox invoke timeout | 30s | 30s | Generous for network-bound tools; prevents hangs |
| Usage deque maxlen | 500 | 500 | ~10 hours of moderate use; low memory footprint |
| Health poll interval | ~5 min | 5 min | Balance between responsiveness and overhead |
| `plugin_generation_enabled` | `True` | `False` in restricted production profiles; `True` in dev/demo profiles | Current code default enables generation, while install/rollback/disable remain approval-gated |

### 6.2 Gaps & Future Hardening

The following items are identified from code analysis as **partial or unwired**. They represent the recommended hardening roadmap:

| # | Gap | Impact | Recommended fix |
|---|-----|--------|-----------------|
| 1 | **Auto-quarantine on health breach** | PluginHealthProducer only advises; a truly misbehaving plugin runs until human acts | Wire `PluginHealthProducer` → `RecoveryCoordinator` with a `plugin_quarantine` strategy that emits `plugin_disable` with `InteractionRequest` for urgent human confirmation |
| 2 | **ActiveSignalSource not fiber-managed** | Signal sources bypass PluginFiber lifecycle; no EffectScope cleanup | Integrate `ActiveSourceManager` with fiber system (already noted in source as "future extension") |
| 4 | **ScopedLLMProviderRegistry lacks `adopt_existing_plugins()`** | Built-in LLM providers have no fibers at boot | Add adoption logic mirroring `ScopedToolRegistry` |
| 5 | **No entry-point discovery for ToolPlugins** | Third-party tools cannot be discovered via `pip install`; profile-dir and marketplace installs are supported | Implement `setuptools` entry_point group `leapflow.tool_plugins` with discovery at boot |
| 7 | **No per-plugin resource quotas** | A misbehaving plugin can consume unlimited CPU/memory | Add configurable per-plugin timeout and call-rate ceiling |
| 8 | **Trust unfreeze not exposed** | A hard-failed plugin can never recover without DB intervention | Add `plugin_unfreeze` tool (gated, HIGH risk) or admin slash command |
| 9 | **No time-series trust history** | Only current state is persisted; cannot audit historical transitions | Extend `PluginStatsStore` with an append-only transitions table |
| 10 | **Gateway adapter lifecycle not fiber-wired** | `GatewayAdapterPlugin` lacks scoped reload/disable mechanics | Extend `ScopedToolRegistry` pattern to gateway adapters |
| 11 | **Fiber dict never pruned** | DISPOSED fibers remain in `_fibers` dict indefinitely | Add `prune_disposed()` method or periodic GC |

---

## 7. Open Questions

These require human/product input and are not answerable from code alone:

1. **Should auto-quarantine be opt-in or opt-out?** If opt-in: which profile types enable it? If opt-out: what is the override config key?

2. **Third-party plugin boot-time discovery**: Should `plugins_dir` contents be auto-loaded at daemon start, or should there be a `registered_plugins.json` manifest that the user explicitly curates?

3. **Trust reset mechanism**: Should operators have a way to manually reset a frozen plugin's trust (clear the `_frozen` set)? Via tool, slash command, or config edit? What approval level?

4. **Cross-profile plugin sharing**: Today plugins are profile-scoped. Should a "global plugins" directory exist (under `~/.leapflow/plugins/`) for plugins shared across profiles?

5. **Marketplace governance**: For the HTTP marketplace, who operates the signing authority? Is the local-directory marketplace sufficient for enterprise deployments, or is a hosted registry needed?

6. **Version pinning**: Should `PluginManifest.min_leapflow_version` be enforced at install time? What about max version? Should version conflicts between plugins be checked (dependency resolution)?

7. **Multi-daemon coordination**: If multiple daemons run under the same profile (not currently supported but architecturally possible), how should trust state writes be coordinated? DuckDB's single-writer model may conflict.

8. **Observability persistence**: Should health findings be persisted to DuckDB for post-mortem analysis? Current in-memory-only model loses incident history on restart.

---

## Appendix A: Key Source File Map

| Responsibility | File |
|---------------|------|
| Fiber state machine | `src/leapflow/domain/plugin_fiber.py` |
| EffectScope (LIFO cleanup) | `src/leapflow/domain/effect_scope.py` |
| Scoped registry (lifecycle-aware tool registration) | `src/leapflow/plugins/scoped_registry.py` |
| Core tool registry | `src/leapflow/plugins/registry.py` |
| Plugin contracts (ToolPlugin / ToolMetadata) | `src/leapflow/plugins/protocol.py` |
| Plugin subsystem public API | `src/leapflow/plugins/__init__.py` |
| Plugin discovery (built-in) | `src/leapflow/plugins/tool_plugins/__init__.py` |
| Self-management tools (12 tools) | `src/leapflow/plugins/tool_plugins/self_management.py` |
| Proposal domain records | `src/leapflow/domain/plugin_proposal.py` |
| Proposal persistence | `src/leapflow/storage/capability_proposal_queue.py` (`EvolutionCapabilityProposalStore`) |
| Behavior test execution | `src/leapflow/learning/plugin_behavior_tests.py` |
| Version snapshot store | `src/leapflow/storage/plugin_version_store.py` |
| Trust ledger | `src/leapflow/learning/plugin_trust.py` |
| Usage tracker | `src/leapflow/learning/plugin_stats.py` |
| Advisor (scoring engine) | `src/leapflow/learning/plugin_advisor.py` |
| Trust persistence (DuckDB) | `src/leapflow/learning/plugin_stats_store.py` |
| Health producer (monitor) | `src/leapflow/monitor/plugin_health_producer.py` |
| Session factory (wiring) | `src/leapflow/engine/session_factory.py` |
| Risk classification | `src/leapflow/security/risk.py` |
| Sandbox host | `src/leapflow/plugins/sandbox/sandbox_host.py` |
| Marketplace client | `src/leapflow/plugins/marketplace/client.py` |
| Plugin generator + validator | `src/leapflow/learning/plugin_generator.py` |
| Settings (config flags) | `src/leapflow/config.py` |
| Profile layout (plugins_dir) | `src/leapflow/layout.py` |

---

## Appendix B: Enforcement Status Summary

| Behavior | Status |
|----------|--------|
| Fiber state transitions (`PENDING→ACTIVE` fast path; `LOADING`/`FAILED` retry primitives available) | ✅ ENFORCED |
| EffectScope LIFO cleanup on dispose | ✅ ENFORCED |
| Per-turn handler snapshot (in-flight safety) | ✅ ENFORCED |
| Generation counter + cache invalidation on reload | ✅ ENFORCED |
| Trust promotion on consecutive successes | ✅ ENFORCED |
| Trust demotion on consecutive failures | ✅ ENFORCED |
| Hard failure → permanent DRAFT freeze | ✅ ENFORCED |
| Trust persistence to DuckDB (on transitions + atexit) | ✅ ENFORCED |
| Approval gate for mutations (HIGH risk, no permanent) | ✅ ENFORCED |
| Fail-closed when no gate installed | ✅ ENFORCED |
| `disabled_plugins` config respected at boot | ✅ ENFORCED |
| Duplicate plugin_id rejection | ✅ ENFORCED |
| Sandbox isolation for installs | ✅ ENFORCED |
| Ed25519 signature + SHA-256 checksum verification | ✅ ENFORCED (when pubkeys configured) |
| Self-management cannot self-disable | ✅ ENFORCED |
| PRODUCTION trust auto-approves reload | ✅ ENFORCED |
| Health finding emission (error rate + trust degrade) | ✅ ENFORCED (advisory) |
| Auto-quarantine on health breach | ❌ NOT ENFORCED (recommended policy) |
| Uninstall disposes live fiber | ✅ ENFORCED through `plugin_remove` |
| Third-party re-discovery on restart | ✅ ENFORCED for profile-scoped `.py` plugins |
| Per-plugin resource quotas | ❌ NOT ENFORCED (roadmap) |
| ActiveSignalSource fiber management | ❌ NOT ENFORCED (future extension) |
| Gateway adapter fiber lifecycle | ❌ NOT ENFORCED (unwired) |
| Trust state GC for removed plugins | ❌ NOT ENFORCED (no mechanism) |
