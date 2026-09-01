# Third-Party Plugin Development Specification

> **Audience**: External developers building plugins for LeapFlow.  
> **Authoritative source**: Derived from production code at `src/leapflow/plugins/` (contracts, registry, lifecycle, sandbox, marketplace), `src/leapflow/tools/` (tool behaviour), `src/leapflow/domain/`, `src/leapflow/learning/`, and `src/leapflow/gateway/`.

---

## 1. Overview & Philosophy

LeapFlow treats **everything as a plugin**. The agent's capabilities — tool execution, LLM access, platform adapters, signal ingestion, computer vision, and frame storage — are all governed by `typing.Protocol` contracts with `@runtime_checkable`. Third-party code extends LeapFlow by satisfying one of these Protocols and registering with the appropriate registry.

### 1.0 Where the plugin subsystem lives

`leapflow.plugins` is a first-class package and the only owner of plugin
contracts, discovery, lifecycle, isolation, and distribution:

```
src/leapflow/plugins/
├── __init__.py         # public API: get_registry / get_scoped_registry / reload_plugin
├── protocol.py         # ToolPlugin Protocol + ToolMetadata (SSOT per tool)
├── registry.py         # ToolPluginRegistry — discovery, DI, assembly, runtime gates
├── scoped_registry.py  # ScopedToolRegistry — PluginFiber lifecycle, hot-reload
├── tool_plugins/       # built-in ToolPlugin declarations (the only layer that
│                       # imports tool implementations)
├── sandbox/            # subprocess isolation for untrusted plugins
└── marketplace/        # manifest, client, HTTP source, prototype server
```

`leapflow.tools` holds what tools *do* (file ops, shell, terminal, web, SCM,
config, gateway dispatch) plus the Tool Capability Contract in
`tools/name_resolver.py`. The dependency direction is one-way and enforced by
`tests/test_architecture_contracts.py`: plugin core never imports a tool module;
`tool_plugins/` is the single place allowed to wrap one.

### 1.1 Extension Protocols

| Protocol | Module | Purpose |
|----------|--------|---------|
| `ToolPlugin` | `plugins/protocol.py` | Register callable tools exposed to the LLM agent |
| `GatewayAdapterPlugin` | `gateway/adapter_registry.py` | Factory for IM/platform adapters (Feishu, Telegram, etc.) |
| `LLMProviderPlugin` | `llm/provider_registry.py` | Register alternative LLM backends |
| `SignalSource` | `perception/signal_source.py` | Stateless event → signal transform |
| `ActiveSignalSource` | `perception/active_signal_source.py` | Long-running signal emitter (webhook listener, polling bot) |
| `CVProcessor` | `perception/cv_processor.py` | Frame-pair visual diff processing |
| `HardwareContextProvider` | `hardware/providers/__init__.py` | Discover physical devices and declare their channels |
| `HardwareTransport` | `hardware/transport.py` | Execute reads/writes against one device (six methods) |
| `FrameTransport` | `hardware/transport.py` | Optional side protocol: a device that produces frames |

Additionally, `FrameStore` (`perception/storage/frame_store.py`) is a `@runtime_checkable` Protocol for pluggable frame persistence backends.

**When to use each:**

- **ToolPlugin** — You want the LLM agent to invoke your functionality as a tool call (most common).
- **GatewayAdapterPlugin** — You are integrating a new IM/collaboration platform.
- **LLMProviderPlugin** — You are adding a new LLM API backend (e.g., a private deployment).
- **SignalSource** — You need to normalize external events into LeapFlow's signal pipeline (stateless, transform-only).
- **ActiveSignalSource** — You need a long-running listener that emits signals (websocket, polling loop).
- **CVProcessor** — You are implementing a visual diff algorithm for the perception subsystem.
- **HardwareContextProvider / HardwareTransport** — You are adding a peripheral. Both have
  their own entry-point groups (`leapflow.hardware.providers`,
  `leapflow.hardware.transports`), so `pip install` is enough. See
  [`hardware_peripherals_board.md`](hardware_peripherals_board.md) for the full contract,
  including how a declared channel becomes a LeapBoard preview or control with no board
  code, and [`hardware_init_calibration.md`](hardware_init_calibration.md) for declaring
  readiness preconditions.

---

## 2. Interface Specification

### 2.1 ToolPlugin Protocol

```python
from typing import Any, Protocol, runtime_checkable

@runtime_checkable
class ToolPlugin(Protocol):
    @property
    def plugin_id(self) -> str: ...

    @property
    def category(self) -> str: ...

    @property
    def tools(self) -> list[ToolMetadata]: ...

    @property
    def dependencies(self) -> list[str]: ...

    def bind_runtime(self, **deps: Any) -> None: ...
```

| Attribute | Description |
|-----------|-------------|
| `plugin_id` | Globally unique string (e.g., `"weather_lookup"`). Used for registry keys, trust tracking, fiber IDs. |
| `category` | Category label for PCD grouping (e.g., `"general"`, `"system"`, `"integration"`). Must match `x_leapflow.category` on tools. |
| `tools` | List of `ToolMetadata` instances — the **single source of truth** for tool schemas, handlers, and metadata. |
| `dependencies` | List of runtime dependency names this plugin requires (e.g., `["memory_manager", "file_read_gate"]`). |
| `bind_runtime` | Receives injected dependencies matching the `dependencies` list. Ignore unknown kwargs. |

### 2.2 ToolMetadata

```python
from dataclasses import dataclass, field
from typing import Any, Callable

@dataclass(frozen=True)
class ToolMetadata:
    name: str
    description: str
    parameters_schema: dict[str, Any]   # OpenAI JSON Schema format
    handler: Callable[..., Any]
    x_leapflow: dict[str, Any] = field(default_factory=dict)
    mutates_state: bool = False

    def to_openai_schema(self) -> dict[str, Any]:
        """Generate OpenAI function-calling schema dict."""
        ...
```

**`to_openai_schema()` output shape:**

```json
{
  "type": "function",
  "function": {
    "name": "tool_name",
    "description": "What the tool does",
    "parameters": { "type": "object", "properties": {...}, "required": [...] },
    "x_leapflow": {
      "category": "integration",
      "mutates_state": true,
      "risk_level": "medium"
    }
  }
}
```

When `mutates_state=True`, `to_openai_schema()` folds it into `x_leapflow.mutates_state` so schema-only consumers can classify side-effecting tools without accessing the metadata object.

**`x_leapflow` well-known keys:**

`x_leapflow` is required for every generated ToolMetadata and must be a dict.
`category` and `risk_level` are mandatory; the validator rejects `None`, missing
category/risk, malformed schemas, non-callable handlers, and mutating tools that
omit approval/idempotency metadata.

| Key | Type | Purpose |
|-----|------|---------|
| `category` | `str` | PCD disclosure grouping |
| `risk_level` | `str` | `"read_only"` / `"low"` / `"medium"` / `"high"` |
| `schema_cost` | `str` | `"low"` / `"medium"` / `"high"` — token cost hint for PCD |
| `requires_approval` | `bool` | Whether the engine gates this tool behind approval |
| `mutates_state` | `bool` | Auto-populated from the field when `True` |

### 2.3 GatewayAdapterPlugin Protocol

```python
@runtime_checkable
class GatewayAdapterPlugin(Protocol):
    @property
    def platform_id(self) -> str: ...

    @property
    def display_name(self) -> str: ...

    @property
    def adapter_class_path(self) -> str: ...  # "module.path:ClassName"

    @property
    def config_schema(self) -> Dict[str, Any]: ...

    def create_adapter(self, config: Dict[str, Any]) -> PlatformAdapter: ...
```

### 2.4 LLMProviderPlugin Protocol

```python
@runtime_checkable
class LLMProviderPlugin(Protocol):
    @property
    def provider_id(self) -> str: ...

    @property
    def display_name(self) -> str: ...

    @property
    def supported_models(self) -> List[str]: ...

    @property
    def capabilities(self) -> Dict[str, Any]: ...
    # Keys: supports_streaming, supports_tools, supports_vision,
    #        supports_thinking, max_context_length, credential_rotation

    def create_provider(self, config: Dict[str, Any]) -> LLMProvider: ...
```

LLM provider plugins support **entry_point discovery** via setuptools group `"leapflow.llm_providers"`. This is the only Protocol that supports entry_point-based discovery.

### 2.5 SignalSource Protocol

```python
@runtime_checkable
class SignalSource(Protocol):
    @property
    def channel_id(self) -> str: ...

    @property
    def event_types(self) -> FrozenSet[str]: ...

    @property
    def bypasses_privacy(self) -> bool: ...

    def transform(self, event_type: str, payload: Dict[str, Any],
                  context: SignalTransformContext) -> Optional[InteractionSignal]: ...
```

Stateless; not fiber-managed. Registered with `SignalSourceRegistry`.

### 2.6 ActiveSignalSource Protocol

```python
EmitCallback = Callable[[InteractionSignal], None]

@runtime_checkable
class ActiveSignalSource(Protocol):
    @property
    def source_id(self) -> str: ...

    @property
    def channel_id(self) -> str: ...

    async def start(self, emit: EmitCallback) -> None: ...
    async def stop(self) -> None: ...
```

Managed by `ActiveSourceManager` (bounded asyncio queue, per-source task, thread-safe emit callback). **Note:** ActiveSignalSource is not yet integrated with PluginFiber lifecycle; lifecycle is owned by `PerceptionSession` directly.

### 2.7 CVProcessor Protocol

```python
@runtime_checkable
class CVProcessor(Protocol):
    @property
    def processor_id(self) -> str: ...

    @property
    def description(self) -> str: ...

    def process(self, frame_a: bytes, frame_b: bytes, **kwargs: Any) -> Dict[str, Any]: ...
```

---

## 3. Development Standards

### 3.1 Plugin ID Naming

- Must be globally unique across the registry.
- Use lowercase `snake_case` (e.g., `"weather_lookup"`, `"jira_integration"`).
- Duplicates at registration time raise `ValueError`.

### 3.2 Dependency Injection via `bind_runtime`

Plugins declare needed services in `dependencies` and receive them through `bind_runtime(**deps)`. This is **late dependency injection** — plugins must not import runtime services at module level.

```python
@property
def dependencies(self) -> list[str]:
    return ["memory_manager", "file_read_gate"]

def bind_runtime(self, **deps: Any) -> None:
    if "memory_manager" in deps:
        self._memory = deps["memory_manager"]
    if "file_read_gate" in deps:
        self._gate = deps["file_read_gate"]
```

Available dependency names (wired by the daemon):
`plugin_approval_gate`, `llm_provider`, `plugin_generation_enabled`, `plugin_install_dir`, `marketplace_client`, `marketplace_trusted_pubkeys`, `memory_manager`, `gateway_server`, `research_ledger`, `reentry_scheduler`, `file_read_gate`, `file_write_gate`, `desktop_gate`, `capability_catalog_provider`, `subagent_manager`.

### 3.3 Side-Effect-Free Import

Plugin modules **must not** perform I/O, network calls, or state mutation at import time. Use the lazy `__getattr__` pattern for heavy optional imports:

```python
def __getattr__(name: str):
    if name == "heavy_client":
        import some_heavy_sdk
        return some_heavy_sdk.Client()
    raise AttributeError(name)
```

### 3.4 Handler Contract

All tool handlers must:

1. Be `async` (signature: `async def handler(params: dict) -> dict`).
2. Accept a single `dict` of parameters matching `parameters_schema`.
3. Return a structured `dict` result (never raw strings).
4. **Never raise** for expected errors — return `{"ok": False, "error": "..."}`.
5. Reserve exceptions for truly unexpected internal failures.

### 3.5 Mutating Tools and Approval

Tools that produce side effects must set `mutates_state=True` on their `ToolMetadata`. The engine routes mutating tool calls through the `ApprovalGate`. To declare risk level and trigger approval:

```python
ToolMetadata(
    name="delete_resource",
    description="Delete a cloud resource permanently.",
    parameters_schema={...},
    handler=handle_delete,
    mutates_state=True,
    x_leapflow={
        "category": "cloud_ops",
        "risk_level": "high",
        "requires_approval": True,
    },
)
```

For platform actions (gateway send, external API write), use `ActionDescriptor.platform_action(platform, action, metadata)` within the handler to explicitly request gate evaluation.

### 3.6 Code Quality

- English docstrings and comments.
- Type annotations on all public APIs.
- No bare `except` — always specify exception types.
- No global mutable state besides the module-level `plugin` instance.

---

## 4. Execution Chain

The following is the ordered sequence from plugin source to tool invocation:

### Step 1: Discovery & Registration

1. **Built-in discovery**: `ToolPluginRegistry.discover_builtin()` imports `leapflow.plugins.tool_plugins.get_all_plugins()`, which lazily imports each plugin module and collects `plugin` instances. Plugins listed in `Settings.disabled_plugins` are skipped.
2. **Registration**: `registry.register(plugin)` stores the plugin keyed by `plugin_id`, validates Protocol conformance, bumps `_version`.

### Step 2: Dependency Injection

3. **`registry.bind_runtime(**deps)`**: Iterates all plugins; for each, filters deps to only those declared in `plugin.dependencies`, then calls `plugin.bind_runtime(**relevant_deps)`. Tracks `_last_bound_deps` for re-injection on hot-reload.

### Step 3: Assembly

4. **`registry.assemble()`**: One-shot pass over all plugins → all tools. For each `ToolMetadata`: calls `to_openai_schema()` to produce the LLM-facing schema, maps `tool.name → tool.handler` into `_tool_handlers`. Plugins that arrive *after* assembly (install, hot-reload) publish their tools through `registry.publish_plugin_tools(plugin)`, which returns the published names and bumps the version counter.

### Step 4: PluginFiber Lifecycle

5. **`ScopedToolRegistry.adopt_existing_plugins()`**: Called on first `leapflow.plugins.get_scoped_registry()` access. It creates a `PluginFiber` for every already-registered plugin and uses the fast path `PENDING → ACTIVE` for the current built-in/profile ToolPlugin runtime. The `PluginFiber` domain type also supports `LOADING` and `FAILED` retry states for future async initialization paths, but the scoped registry does not yet run a dependency-driven async activation loop.

Fiber domain state machine: `PENDING → LOADING → ACTIVE → UNLOADING → DISPOSED` (with `LOADING → FAILED → LOADING` retry path). Current ToolPlugin registration uses the fast path `PENDING → ACTIVE`; `LOADING`/`FAILED` are available primitives, not automatic dependency orchestration.

### Step 5: Per-Turn Engine Assembly

6. **`engine._unified_tool_catalog()`**: Produces the OpenAI schema list sent to the LLM. Cache key: `((id(desktop_plugin), desktop_plugin.version), len(registry.tool_definitions))`. Invalidated when registry version changes.
7. **`engine._unified_tool_handlers()`**: Returns `dict(registry.tool_handlers)` — a **fresh copy per turn**. This per-turn snapshot is the concurrency-safety mechanism: in-flight turns keep their snapshot; a hot-reload mid-session only affects turns started after the reload.

### Step 6: Tool Dispatch

8. **`_execute_general_tool(tool_call, handlers)`**: Resolves the tool name (canonical/normalized resolution via `ToolRegistry.resolve()`), looks up the handler in the per-turn snapshot, and invokes it through the shared handler adapter (`invoke_tool_handler`). The adapter supports both generated `**kwargs` handlers and older `params: dict` handlers, so native function calls with `{}` work for no-argument tools such as `plugin_list`.
9. **Approval gate**: If the tool's metadata declares `mutates_state=True` or requires approval, the engine checks the approval gate before execution.

### Step 7: Usage & Trust Recording

10. **`TurnUsageTracker.record_tool_call(name, ok, duration)`**: Records per-turn sample.
11. **Forwarding**: If `_plugin_stats_sink` is set (wired by `session_factory._wire_plugin_stats_sink()`), forwards to `PluginUsageTracker.record()`.
12. **Trust update**: `PluginUsageTracker` resolves `tool_name → plugin_id` via a lazy reverse index (invalidated by registry `_version`), then calls `PluginTrustLedger.record_success()` or `record_failure()`.
13. **Persistence**: `PluginStatsStore` (DuckDB) persists trust state. An `atexit` handler ensures durability on shutdown.

### Why Per-Turn Snapshots Make Hot-Reload Safe

The engine calls `dict(registry.tool_handlers)` at the start of each turn, creating an isolated copy. A `ScopedToolRegistry.reload(plugin_id)` in a concurrent session:
- Disposes the old fiber (removes tools from the registry).
- Re-imports and re-registers the fresh plugin.
- Bumps `_version` → invalidates cache for future turns.

But the currently-executing turn still holds its snapshot with the old handlers and finishes safely. LeapFlow's single-threaded asyncio model guarantees no pre-emption mid-turn.

### Tool Execution Pipeline & Interceptors

Tool dispatch is wrapped by a **`ToolExecutionPipeline`** that implements a waterfall (middleware) pattern. Before and after actual handler invocation, registered `ToolInterceptor` instances run composable pre/post hooks.

```python
from typing import Any, Protocol, runtime_checkable

@runtime_checkable
class ToolInterceptor(Protocol):
    async def before(self, context: dict[str, Any]) -> dict[str, Any]: ...
    async def after(self, context: dict[str, Any], result: Any) -> Any: ...
```

**Registering an interceptor from a third-party plugin:**

```python
from leapflow.plugins import get_registry

def bind_runtime(self, **deps: Any) -> None:
    registry = get_registry()
    registry.tool_pipeline.register(self._my_interceptor)  # priority comes from interceptor.priority
```

Interceptors are process-global once registered on `registry.tool_pipeline`. If a plugin registers one dynamically, it must also register an explicit cleanup effect on its `PluginFiber`/`EffectScope` (or unregister it during disposal); automatic scope-bound interceptor removal is a planned convenience, not current runtime behavior.

Typical interceptor use cases include audit logging, execution timeout, approval gating, rate limiting, and result redaction. The repository currently ships the pipeline primitives and example audit/timeout interceptors; production plugins must register and unregister any additional interceptor explicitly.

### Dependency Binding and Activation

Plugins declare `dependencies`, and `ToolPluginRegistry.bind_runtime()` distributes matching runtime dependencies in topological plugin order. Current ToolPlugin activation still uses the `ScopedToolRegistry` fast path (`PENDING → ACTIVE`) after registration; plugins that require a dependency should degrade gracefully in their handler when the dependency is not bound. A future async activation loop may use the `LOADING`/`FAILED` states for dependency-driven retries, but that is not yet automatic.

---

## 5. Deployment

### 5.1 Built-in Package Plugins

Plugins shipped with LeapFlow live in `src/leapflow/plugins/tool_plugins/`. Each module defines a module-level `plugin = MyPlugin()` instance. Discovery is via the `_BUILTIN_PLUGIN_MODULES` tuple in `plugins/tool_plugins/__init__.py`.

**To add a new built-in plugin**: Add the module path to `_BUILTIN_PLUGIN_MODULES` and ensure the module exposes `plugin`.

### 5.2 Profile-Scoped Install (Dynamic)

Third-party plugins are installed into the active profile's plugin directory:

```
~/.leapflow/profiles/<profile>/plugins/<entry_point>.py
```

Installation is performed by the `plugin_install` tool (part of the `self_management` plugin). The path is derived from `ProfileLayout.plugins_dir` or overridden by `Settings.plugin_install_dir`. Profile-scoped `.py` plugins are discovered on registry boot by `discover_profile_plugins()` and loaded with file-backed import specs so reload does not depend on global `sys.path`.

**Install flow**: Validated code → write to profile plugins dir → sandbox smoke test → dynamic import → register in live registry → activate fiber.

### 5.2a Generating plugins via slash command (`/plugin generate`)

For a zero-boilerplate path, the TUI/CLI exposes `/plugin generate <description>`,
which synthesizes, validates, and installs a plugin directly from a natural-language
description:

```text
/plugin generate a tool that fetches the current weather for a city
```

- **Zero-prompt happy path** — the description is sent to the configured LLM, the
  generated code is run through the staged validators (syntax → import → Protocol
  conformance → sandbox smoke test), and on success the plugin is installed into the
  profile plugins dir and hot-loaded. A single bounded refinement retry runs on a
  refinable validation failure (syntax/protocol/structure).
- **`--preview`** — generate and validate only, then return the code for inspection
  without installing. Use this to review before committing.
- **`--dry-run`** — validate the generated code without writing it to disk.
- **`--id <plugin_id>`** — override the auto-derived plugin id (a slug of the
  description). A colliding id is rejected cleanly.

Generation is controlled by `plugin.generation_enabled` (enabled by default in current config; disable via `/config set plugin.generation_enabled false`) and requires an LLM provider. Installation still remains a separate approval-gated action.

**Difference from the `plugin_generate` agent tool**: `/plugin generate` is a
*user-initiated* control-plane command — the user's invocation is the consent, so it
runs without an approval gate and produces a DRAFT-trust plugin. The `plugin_generate`
tool is *agent-initiated*: the agent proposes generation from capability-gap evidence,
and every mutation routes through the `ApprovalGate` at HIGH risk. Both share the same
generator, validators, and install path.

### 5.3 LLM Provider Entry Points

Only `LLMProviderPlugin` supports setuptools entry_point discovery:

```toml
# pyproject.toml of the external package
[project.entry-points."leapflow.llm_providers"]
my_provider = "my_package.provider:plugin"
```

`LLMProviderRegistry.discover_entry_points()` loads these at startup.

> **Important**: `ToolPlugin` supports built-in package discovery, profile-scoped file discovery, marketplace install, and explicit registration. It does NOT yet support setuptools entry_point discovery. `GatewayAdapterPlugin` remains registered through the gateway adapter registry.

### 5.4 Marketplace Distribution

#### PluginManifest

```python
@dataclass(frozen=True)
class PluginManifest:
    name: str                       # unique plugin identifier
    version: str                    # semver
    author: str
    description: str
    entry_point: str                # module filename (without .py)
    plugin_type: str = "tool"       # "tool" | "active_signal_source" | "gateway" | "llm"
    source_url: str = ""
    checksum_sha256: str = ""       # SHA-256 of the source code
    requires_sandbox: bool = True   # untrusted by default
    dependencies: List[str] = field(default_factory=list)
    min_leapflow_version: str = ""
    signature: str = ""             # hex Ed25519 signature
    signer_pubkey: str = ""         # hex Ed25519 public key
```

#### Integrity: SHA-256 Checksum

```python
checksum = PluginManifest.compute_checksum(code_bytes)
manifest.verify_checksum(code_bytes)  # -> bool
```

#### Authenticity: Ed25519 Signing

```python
# Generate keypair (author does this once)
private_hex, public_hex = PluginManifest.generate_keypair()

# Sign (author signs before publishing)
signed_manifest = manifest.sign(code_bytes, private_hex)

# Verify (client checks on install)
signed_manifest.verify_signature(code_bytes, trusted_pubkeys={"<public_hex>"})
```

Canonical signed payload: `name|version|entry_point|checksum_sha256` (UTF-8 encoded).

#### Marketplace Sources

| Source | Class | Discovery |
|--------|-------|-----------|
| Local directory | `LocalDirectorySource` | `<root>/<plugin_name>/manifest.json` + `<entry_point>.py` |
| HTTP registry | `HttpMarketplaceSource` | `GET /plugins/`, `GET /plugins/<name>/manifest.json`, `GET /plugins/<name>/<entry_point>.py` |

#### MarketplaceClient

```python
client = MarketplaceClient(source=LocalDirectorySource(root), install_dir=path)
manifests = client.discover()
result = client.install("plugin_name", verify=True, trusted_pubkeys={"..."})
```

> **Caveat — Marketplace HTTP server**: The HTTP server (`plugins/marketplace/server.py`) is a prototype. Production readiness is not guaranteed.

> **Removal**: Use the `plugin_remove(plugin_id, delete_source=true)` self-management tool for live removal. It disposes the fiber, unregisters tools, drops reload metadata, and optionally deletes the profile-scoped source file. `MarketplaceClient.uninstall()` remains a low-level file deletion primitive and does not by itself operate on live runtime state.

### 5.5 Configuration Keys

| Key | Default | Purpose |
|-----|---------|---------|
| `disabled_plugins` | `()` | Tuple of `plugin_id`s to skip at discovery time |
| `plugin_generation_enabled` | `True` | Gate for LLM-driven code generation; set false to disable synthesis |
| `plugin_install_dir` | `None` (→ `ProfileLayout.plugins_dir`) | Override install directory |
| `plugin_marketplace_root` | `None` | Local directory marketplace source |
| `plugin_marketplace_url` | `None` | HTTP marketplace URL (takes precedence over local) |
| `plugin_marketplace_trusted_pubkeys` | `()` | Hex Ed25519 public keys for signature verification |

### 5.6 Slash Commands

- `/plugin` — List all registered plugins.
- `/plugin status <id>` — Show plugin details and trust level.

Mutating operations (install, rollback, reload, disable, remove, enable) are only available through the self-management tools in daemon mode, and require explicit approval. Read-only governance tools such as `plugin_versions` can inspect recorded profile-scoped source snapshots without approval.

### 5.7 Compatibility Assessment (Pre-Install Gate)

Before a marketplace install writes any file, LeapFlow runs the **Compatibility Assessment Engine** (`leapflow.learning.compatibility`) against the resolved manifest. The engine is a six-stage pipeline (manifest parsing → category resolution → interface analysis → dependency checking → execution model → security classification) that produces a `CompatibilityReport` with a final verdict. Marketplace installs are gated on it: an `INCOMPATIBLE` verdict is **rejected before file write** with a structured error; an `ADAPTABLE` verdict proceeds and surfaces its `adaptation_notes` alongside the install result.

**Verdict meaning for developers:**

| Verdict | What it means for you |
|---------|-----------------------|
| `COMPATIBLE` | Direct install; no modification needed. |
| `ADAPTABLE` | A thin bridge/shim is needed and is **auto-generated** (see `adapter_generator`). |
| `PARTIAL` | Only a subset of the plugin's features is usable; the unusable surfaces are documented in the report. |
| `INCOMPATIBLE` | The plugin targets a system layer LeapFlow does not expose (e.g. `agent-loop`, `session`, `context`, `storage`); LeapFlow **cannot host** it. |

**Pre-check a manifest manually** with the `assess_compatibility` tool (read-only, no approval). It accepts a manifest dict in LeapFlow or DSH format and returns the verdict, target protocol, adaptation notes, adapter spec, and per-stage results:

```python
from leapflow.learning.compatibility import assess_plugin

report = assess_plugin(manifest_dict)          # dict, LeapFlow or DSH format
print(report.final_verdict.value)               # "compatible" | "adaptable" | "partial" | "incompatible"
print(report.is_installable())                  # True unless INCOMPATIBLE
```

**File-path loading:** `assess_plugin()` also accepts a path (string or `Path`) to a manifest JSON file; it is read from disk and flows through the same pipeline:

```python
report = assess_plugin("/path/to/manifest.json")
```

**DSH → LeapFlow interop:** `convert_dsh_to_leapflow()` translates a deepseek-harness `package.json`-style manifest into a LeapFlow `PluginManifest`-compatible dict (name normalization, `main` → `entry_point`, `requires_sandbox=True`, original manifest preserved under `x_dsh_original`):

```python
from leapflow.learning.compatibility.manifest_converter import convert_dsh_to_leapflow

leapflow_manifest = convert_dsh_to_leapflow(dsh_package_json)
```

---

## 6. Security Model

### 6.1 Validation Pipeline

| Stage | When | What |
|-------|------|------|
| Syntax | Generate-time | `ast.parse()` — valid Python |
| Structure | Generate-time | AST check: module-level `plugin` assignment; flag dangerous patterns (`os.system`, `eval`, `exec`) |
| Import | Generate-time | Temp-file import in throwaway namespace; Protocol conformance check |
| Sandbox smoke | Install-time | First tool invoked in isolated subprocess via `SandboxHost` |
| Human approval | Install-time | `ApprovalGate` evaluation (see below) |

### 6.2 Approval Gating

All plugin mutation operations are classified as **HIGH risk** with `allow_permanent=False`:

```python
# From security/risk.py
platform "plugin_management" → RiskLevel.HIGH, allow_permanent=False
```

This means:
- Every install/reload/disable/enable requires explicit approval per invocation.
- Permanent grants are never issued for plugin mutations.
- No gate installed (non-daemon mode) → **fail-closed** (mutations denied).

### 6.3 Progressive Trust

Plugins earn trust through consistent successful execution:

| Level | Consecutive Successes Required | Behavior |
|-------|-------------------------------|----------|
| `DRAFT` | 0 (initial) | Untrusted; full sandbox |
| `CANDIDATE` | 5 | Partially trusted |
| `VERIFIED` | 20 | Established reliability |
| `PRODUCTION` | 50 | Auto-approves `plugin_reload` |

**Demotion**: 3 consecutive failures → demote one level.  
**Hard failure** (internal defect): Immediate freeze to `DRAFT` permanently.

### 6.4 Sandbox Isolation

Untrusted plugins (`requires_sandbox=True`, the default) execute in a subprocess:

- `SandboxHost` launches a worker subprocess (`leapflow.plugins.sandbox.worker`).
- Communication: JSON-RPC over stdin/stdout.
- Timeout: 30s default per invocation.
- `SandboxedToolPlugin` wraps the plugin Protocol with proxied handlers.
- Sandboxed plugins receive **no host-side runtime dependencies** (empty `bind_runtime`).

---

## 7. End-to-End Example

### 7.1 Author a ToolPlugin

```python
"""Weather lookup plugin — demonstrates a minimal third-party ToolPlugin."""
from __future__ import annotations
from typing import Any
from leapflow.plugins.protocol import ToolMetadata, ToolPlugin


class WeatherPlugin:
    """Provides a single tool to look up weather data."""

    @property
    def plugin_id(self) -> str:
        return "weather_lookup"

    @property
    def category(self) -> str:
        return "integration"

    @property
    def tools(self) -> list[ToolMetadata]:
        return [
            ToolMetadata(
                name="get_weather",
                description="Get current weather for a city.",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "City name",
                        },
                    },
                    "required": ["city"],
                },
                handler=self._handle_get_weather,
                x_leapflow={
                    "category": "integration",
                    "risk_level": "read_only",
                    "schema_cost": "low",
                },
            ),
        ]

    @property
    def dependencies(self) -> list[str]:
        return []  # No runtime deps needed

    def bind_runtime(self, **deps: Any) -> None:
        pass  # No-op

    async def _handle_get_weather(self, params: dict) -> dict:
        """Handler: fetch weather data."""
        city = params.get("city", "")
        if not city:
            return {"ok": False, "error": "city parameter is required"}
        # Real implementation would call a weather API here
        return {
            "ok": True,
            "city": city,
            "temperature_c": 22,
            "condition": "partly_cloudy",
        }


# Module-level instance — REQUIRED for discovery
plugin = WeatherPlugin()
```

### 7.2 Validate

The `PluginValidator` runs automatically during `plugin_generate` or can be triggered programmatically:

```python
from leapflow.learning.plugin_generator import PluginValidator

validator = PluginValidator()
result = await validator.validate("weather_lookup", source_code)
# result.ok == True, result.stage == "passed", result.exposed_tools == ["get_weather"]
```

### 7.3 Install (Code Path)

Using the `plugin_install` tool (requires daemon mode + approval):

```
Agent: I'll install the weather plugin.
→ plugin_install(code="<validated source>", plugin_id="weather_lookup")
→ ApprovalGate: HIGH risk, requires user confirmation
→ User approves
→ Code written to ~/.leapflow/profiles/default/plugins/weather_lookup.py
→ Sandbox smoke test passes
→ Dynamic import → register → fiber created (ACTIVE)
→ Result: {"ok": true, "plugin_id": "weather_lookup", "tools": ["get_weather"]}
```

### 7.4 Invoke

Once installed, the LLM agent can call the tool naturally:

```
User: What's the weather in Tokyo?
Agent: [calls get_weather(city="Tokyo")]
→ Engine: _unified_tool_handlers() snapshot includes "get_weather"
→ _execute_general_tool → handler invoked → result returned
→ TurnUsageTracker.record_tool_call("get_weather", ok=True, duration_ms=45)
```

### 7.5 Observe Trust Accrual

```
→ PluginUsageTracker.record("get_weather", ok=True, 45.0)
→ Resolves "get_weather" → plugin_id "weather_lookup"
→ PluginTrustLedger.record_success("weather_lookup")
→ After 5 consecutive successes: DRAFT → CANDIDATE
→ After 20: CANDIDATE → VERIFIED
→ After 50: VERIFIED → PRODUCTION (reload auto-approves)
```

Query trust via `plugin_status("weather_lookup")` to see current level and advisor recommendations.

---

## 8. Reference Tables

### 8.1 Module Path Index

| Subsystem | Key File(s) |
|-----------|-------------|
| ToolPlugin Protocol | `src/leapflow/plugins/protocol.py` |
| ToolPluginRegistry | `src/leapflow/plugins/registry.py` |
| ScopedToolRegistry | `src/leapflow/plugins/scoped_registry.py` |
| Built-in plugin discovery | `src/leapflow/plugins/tool_plugins/__init__.py` |
| EffectScope | `src/leapflow/domain/effect_scope.py` |
| PluginFiber | `src/leapflow/domain/plugin_fiber.py` |
| Self-management tools | `src/leapflow/plugins/tool_plugins/self_management.py` |
| Sandbox host + protocol | `src/leapflow/plugins/sandbox/sandbox_host.py`, `plugins/sandbox/protocol.py` |
| Marketplace manifest | `src/leapflow/plugins/marketplace/manifest.py` |
| Marketplace client | `src/leapflow/plugins/marketplace/client.py` |
| HTTP marketplace source | `src/leapflow/plugins/marketplace/http_source.py` |
| Marketplace server (prototype) | `src/leapflow/plugins/marketplace/server.py` |
| Plugin generator + validator | `src/leapflow/learning/plugin_generator.py` |
| Trust ledger | `src/leapflow/learning/plugin_trust.py` |
| Usage tracker | `src/leapflow/learning/plugin_stats.py` |
| Plugin advisor | `src/leapflow/learning/plugin_advisor.py` |
| Stats persistence (DuckDB) | `src/leapflow/learning/plugin_stats_store.py` |
| Health producer | `src/leapflow/monitor/plugin_health_producer.py` |
| GatewayAdapterPlugin | `src/leapflow/gateway/adapter_registry.py` |
| LLMProviderPlugin | `src/leapflow/llm/provider_registry.py` |
| Settings (config keys) | `src/leapflow/config.py` |
| Profile layout (plugins_dir) | `src/leapflow/layout.py` |

### 8.2 Configuration Key Table

| Config Key | Type | Default | Description |
|------------|------|---------|-------------|
| `disabled_plugins` | `tuple[str, ...]` | `()` | Plugin IDs to skip during built-in discovery |
| `plugin_generation_enabled` | `bool` | `True` | Enable LLM-driven plugin code generation; set false to disable synthesis |
| `plugin_install_dir` | `str \| None` | `None` | Override profile plugins dir path |
| `plugin_marketplace_root` | `str \| None` | `None` | Local marketplace directory |
| `plugin_marketplace_url` | `str \| None` | `None` | HTTP marketplace URL |
| `plugin_marketplace_trusted_pubkeys` | `tuple[str, ...]` | `()` | Trusted Ed25519 public keys (hex) |

### 8.3 Test Files for Contributors

| Subsystem | Test File |
|-----------|-----------|
| Plugin reload / lifecycle | `tests/test_plugin_reload.py` |
| Self-management tools | `tests/test_self_management.py` |
| Sandbox | `tests/test_plugin_sandbox.py` |
| Marketplace + signing | `tests/test_plugin_marketplace.py`, `tests/test_marketplace_signing.py` |
| Generator + validator | `tests/test_plugin_generator.py` |
| Trust / learning | `tests/test_plugin_learning.py` |
| Stats persistence | `tests/test_plugin_stats_persistence.py` |
| Scoped registry | `tests/test_scoped_registry.py` |
| Full fiberization | `tests/test_full_fiberization.py` |
| Effect scope | `tests/test_effect_scope.py` |
| Architecture contracts | `tests/test_architecture_contracts.py` |
| LLM provider registry | `tests/test_llm_provider_registry.py` |
| Gateway adapters | `tests/test_gateway_adapters.py`, `tests/test_gateway_adapter_registry.py` |
| CV plugins | `tests/test_cv_plugins.py` |
| Active signal sources | `tests/test_active_signal_source.py` |
| Marketplace HTTP server | `tests/test_marketplace_server.py` |
| Monitor (health producer) | `tests/test_monitor_subsystem.py` |

---

## Appendix: Roadmap / Not Yet Available

The following features exist in code but are **partial, prototype, or unwired**:

| Feature | Status |
|---------|--------|
| Entry-point discovery for `ToolPlugin` / `GatewayAdapterPlugin` | Not implemented. Only `LLMProviderPlugin` uses entry_points. |
| `ActiveSignalSource` fiber integration | Not wired. Lifecycle owned by `PerceptionSession`, not `PluginFiber`. |
| Marketplace HTTP server | Prototype (`asyncio` HTTP, no auth/rate-limiting). |
| `PluginHealthProducer` → automatic remediation | Advisory-only. Detects anomalies but does not auto-disable plugins. |
| `MarketplaceClient.uninstall()` live unregister | Deletes file only; does not dispose fiber or remove from live registry. |
| Gateway adapter fiber lifecycle (scoped reload) | Protocol exists but scoped fiber-based reload for gateway adapters is not confirmed wired. |
