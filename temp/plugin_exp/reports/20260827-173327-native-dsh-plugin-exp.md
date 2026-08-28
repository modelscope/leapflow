# Native DSH Plugin Compatibility Experiment

- Result: **PASS**
- Passed: 10/10
- DeepSeek Harness: `2df279bfd21e1985ebfa85c3afd21790d918c2ab`
- Node: `v25.2.1`
- LLM requests: `0` (configuration and cache were probed read-only)

## Strategy

The matrix uses real source and built artifacts from the local DeepSeek Harness checkout. It separates structural rejection, architectural rejection, runtime discovery, lifecycle persistence, and capability enforcement so a static verdict cannot masquerade as execution proof.

| Case | Expected class | Static verdict | Install | Runtime | Result |
|---|---|---|---|---|---|
| `native_dynamic_reverse` | adaptable | adaptable | ok | ok | PASS |
| `native_dynamic_reverse_partial` | partial | partial | ok | ok | PASS |
| `adapted_prebuilt_reverse` | adaptable | adaptable | ok | ok | PASS |
| `native-agent-loop` | incompatible | incompatible | rejected | n/a | PASS |
| `native-tool-fs` | incompatible | incompatible | rejected | n/a | PASS |
| `native_cross_plugin_service_consumer` | incompatible | incompatible | rejected | n/a | PASS |
| `native_ui_only` | incompatible | incompatible | rejected | n/a | PASS |
| `native_unbuilt_typescript` | inspection error | SourceInspectionError | rejected | n/a | PASS |
| `native_missing_entry` | inspection error | SourceInspectionError | rejected | n/a | PASS |
| `dynamic_shell_injection` | adaptable | adaptable | ok | denied safely | PASS |

## Evidence

### native_dynamic_reverse
- Source: `packages/extensions/cordis-host-runner/tests/helpers.ts#REVERSE_TOOL_CODE`
- Purpose: Real dynamic host tool copied from the DSH runner conformance suite.
- Static: `{"source_kind": "cordis_dynamic_export", "verdict": "adaptable", "installable": false, "installable_candidate": true, "category": "tools", "dependencies": [], "permissions": [], "blockers": [], "limitations": [], "components": [{"kind": "host", "status": "candidate", "reason": "Dynamic Cordis host requires restricted runtime discovery"}], "rejection_reason": ""}`
- Lifecycle: `{"install": {"ok": true, "action": "install", "plugin_id": "native_dynamic_reverse", "installed_tools": ["reverse_text"], "state": "active", "version": "native-exp-v1", "source_kind": "cordis_dynamic_export", "bundle_sha256": "39ea7a2973a4c2d2105db2deb2dd430fcb3d7c6fbef1db797a0b8e5ef0532789", "descriptor_path": "/Users/jason/work/github/leapflow/temp/plugin_exp/work/native-dsh/20260827-173327/profile/plugins/dsh/native_dynamic_reverse/descriptor.json", "verdict": "adaptable", "limitations": [], "client_components": []}, "invoke": {"ok": true, "result": "wolfpael"}, "status": {"ok": true, "plugin_id": "native_dynamic_reverse", "category": "bridge", "dependencies": [], "tools": [{"name": "reverse_text", "description": "Reverse a string."}], "fiber": {"state": "active", "generation": 1}, "dsh": {"source_kind": "cordis_dynamic_export", "bundle_sha256": "39ea7a2973a4c2d2105db2deb2dd430fcb3d7c6fbef1db797a0b8e5ef0532789", "entry_point": "host.runtime.cjs", "verdict": "adaptable", "limitations": [], "client_components": [], "runtime": "node"}}, "restart_invoke": {"ok": true, "result": "wolfpael"}, "reload": {"ok": true, "action": "reload", "plugin_id": "native_dynamic_reverse", "new_generation": 2, "state": "active", "version": ""}, "remove": {"ok": true, "action": "remove", "plugin_id": "native_dynamic_reverse", "state": "disposed", "source_path": "/Users/jason/work/github/leapflow/temp/plugin_exp/work/native-dsh/20260827-173327/profile/plugins/native_dynamic_reverse.py", "source_deleted": true}, "cleanup": {"wrapper_absent": true, "bundle_absent": true, "registry_absent": true}}`

### native_dynamic_reverse_partial
- Source: `packages/extensions/cordis-host-runner/tests/helpers.ts#REVERSE_TOOL_CODE`
- Purpose: Real dynamic host tool with a skipped browser half.
- Static: `{"source_kind": "cordis_dynamic_export", "verdict": "partial", "installable": false, "installable_candidate": true, "category": "tools", "dependencies": [], "permissions": [], "blockers": [], "limitations": ["client.js UI was detected and will be skipped; only safe host tools can be installed"], "components": [{"kind": "host", "status": "candidate", "reason": "Dynamic Cordis host requires restricted runtime discovery"}, {"kind": "client", "status": "unsupported", "reason": "Cordis React/slots client UI is not executable in LeapFlow P0"}], "rejection_reason": ""}`
- Lifecycle: `{"install": {"ok": true, "action": "install", "plugin_id": "native_dynamic_reverse_partial", "installed_tools": ["reverse_text"], "state": "active", "version": "native-exp-v1", "source_kind": "cordis_dynamic_export", "bundle_sha256": "89db83d73e1f6afdb97a721684ec9c427fcd9a04c0ee5f6919af527fa2cd906a", "descriptor_path": "/Users/jason/work/github/leapflow/temp/plugin_exp/work/native-dsh/20260827-173327/profile/plugins/dsh/native_dynamic_reverse_partial/descriptor.json", "verdict": "partial", "limitations": ["client.js UI was detected and will be skipped; only safe host tools can be installed"], "client_components": [{"name": "client", "status": "unsupported", "reason": "Cordis React/slots client UI is not executable in LeapFlow P0", "slots": []}]}, "invoke": {"ok": true, "result": "wolfpael"}, "status": {"ok": true, "plugin_id": "native_dynamic_reverse_partial", "category": "bridge", "dependencies": [], "tools": [{"name": "reverse_text", "description": "Reverse a string."}], "fiber": {"state": "active", "generation": 3}, "dsh": {"source_kind": "cordis_dynamic_export", "bundle_sha256": "89db83d73e1f6afdb97a721684ec9c427fcd9a04c0ee5f6919af527fa2cd906a", "entry_point": "host.runtime.cjs", "verdict": "partial", "limitations": ["client.js UI was detected and will be skipped; only safe host tools can be installed"], "client_components": [{"name": "client", "status": "unsupported", "reason": "Cordis React/slots client UI is not executable in LeapFlow P0", "slots": []}], "runtime": "node"}}, "restart_invoke": {"ok": true, "result": "wolfpael"}, "reload": {"ok": true, "action": "reload", "plugin_id": "native_dynamic_reverse_partial", "new_generation": 4, "state": "active", "version": ""}, "remove": {"ok": true, "action": "remove", "plugin_id": "native_dynamic_reverse_partial", "state": "disposed", "source_path": "/Users/jason/work/github/leapflow/temp/plugin_exp/work/native-dsh/20260827-173327/profile/plugins/native_dynamic_reverse_partial.py", "source_deleted": true}, "cleanup": {"wrapper_absent": true, "bundle_absent": true, "registry_absent": true}}`

### adapted_prebuilt_reverse
- Source: `packages/extensions/cordis-host-runner/tests/helpers.ts#REVERSE_TOOL_CODE`
- Purpose: Real dynamic fixture adapted into a self-contained pre-built DSH package.
- Static: `{"source_kind": "dsh_package", "verdict": "adaptable", "installable": false, "installable_candidate": true, "category": "tools", "dependencies": [], "permissions": [], "blockers": [], "limitations": [], "components": [{"kind": "host", "status": "candidate", "reason": "Pre-built JavaScript entry requires restricted Node runtime discovery"}], "rejection_reason": ""}`
- Lifecycle: `{"install": {"ok": true, "action": "install", "plugin_id": "adapted_prebuilt_reverse", "installed_tools": ["reverse_text"], "state": "active", "version": "native-exp-v1", "source_kind": "dsh_package", "bundle_sha256": "3fa4be60a2bf1393f2fe365975f2bd8029c21e025fa5fbcb7bc701682520fe32", "descriptor_path": "/Users/jason/work/github/leapflow/temp/plugin_exp/work/native-dsh/20260827-173327/profile/plugins/dsh/adapted_prebuilt_reverse/descriptor.json", "verdict": "adaptable", "limitations": [], "client_components": []}, "invoke": {"ok": true, "result": "wolfpael"}, "status": {"ok": true, "plugin_id": "adapted_prebuilt_reverse", "category": "bridge", "dependencies": [], "tools": [{"name": "reverse_text", "description": "Reverse a string."}], "fiber": {"state": "active", "generation": 5}, "dsh": {"source_kind": "dsh_package", "bundle_sha256": "3fa4be60a2bf1393f2fe365975f2bd8029c21e025fa5fbcb7bc701682520fe32", "entry_point": "index.cjs", "verdict": "adaptable", "limitations": [], "client_components": [], "runtime": "node"}}, "restart_invoke": {"ok": true, "result": "wolfpael"}, "reload": {"ok": true, "action": "reload", "plugin_id": "adapted_prebuilt_reverse", "new_generation": 6, "state": "active", "version": ""}, "remove": {"ok": true, "action": "remove", "plugin_id": "adapted_prebuilt_reverse", "state": "disposed", "source_path": "/Users/jason/work/github/leapflow/temp/plugin_exp/work/native-dsh/20260827-173327/profile/plugins/adapted_prebuilt_reverse.py", "source_deleted": true}, "cleanup": {"wrapper_absent": true, "bundle_absent": true, "registry_absent": true}}`

### native-agent-loop
- Source: `packages/core/agent-loop`
- Purpose: Native agent-loop package must not replace LeapFlow's OODA loop.
- Static: `{"source_kind": "dsh_package", "verdict": "incompatible", "installable": false, "installable_candidate": false, "category": "agent-loop", "dependencies": ["@deepseek-ai/cordis", "@deepseek-ai/dsh-agent", "@deepseek-ai/dsh-invariants", "@deepseek-ai/dsh-llm", "@deepseek-ai/dsh-scope", "@deepseek-ai/dsh-session", "@deepseek-ai/dsh-session-persistence", "@deepseek-ai/dsh-settings", "@deepseek-ai/dsh-system-prompt", "@deepseek-ai/dsh-tools", "@deepseek-ai/schemastery"], "permissions": ["unknown.service.agents", "unknown.service.llm", "unknown.service.sessionPersistence", "unknown.service.sessions", "unknown.service.systemPrompt"], "blockers": ["P0 does not install npm dependencies; provide a self-contained pre-built bundle", "P0 does not expose required DSH host service: agents", "P0 does not expose required DSH host service: llm", "P0 does not expose required DSH host service: sessionPersistence", "P0 does not expose required DSH host service: sessions", "P0 does not expose required DSH host service: systemPrompt"], "limitations": [], "components": [{"kind": "host", "status": "candidate", "reason": "Pre-built JavaScript entry requires restricted Node runtime discovery"}], "rejection_reason": "LeapFlow engine is a single hardened OODA execution loop with PCD; replacing it breaks session safety, recovery, and context invariants"}`
- Lifecycle: `{"install": {"ok": false, "error": "LeapFlow engine is a single hardened OODA execution loop with PCD; replacing it breaks session safety, recovery, and context invariants", "verdict": "incompatible"}}`

### native-tool-fs
- Source: `packages/fs/tool-fs`
- Purpose: Tool domain is relevant, but the real package needs npm and Cordis services.
- Static: `{"source_kind": "dsh_package", "verdict": "incompatible", "installable": false, "installable_candidate": false, "category": "fs", "dependencies": ["@deepseek-ai/cordis", "@deepseek-ai/dsh-attachment", "@deepseek-ai/dsh-fs", "@deepseek-ai/dsh-invariants", "@deepseek-ai/dsh-llm", "@deepseek-ai/dsh-sandbox", "@deepseek-ai/dsh-sandbox-policy", "@deepseek-ai/dsh-session", "@deepseek-ai/dsh-system-prompt", "@deepseek-ai/dsh-tools", "@deepseek-ai/dsh-user-approval", "@deepseek-ai/schemastery", "diff"], "permissions": ["unknown.service.approval", "unknown.service.attachments", "unknown.service.fs", "unknown.service.llm", "unknown.service.sandboxPolicy", "unknown.service.systemPrompt"], "blockers": ["P0 does not install npm dependencies; provide a self-contained pre-built bundle", "P0 does not expose required DSH host service: approval", "P0 does not expose required DSH host service: attachments", "P0 does not expose required DSH host service: fs", "P0 does not expose required DSH host service: llm", "P0 does not expose required DSH host service: sandboxPolicy", "P0 does not expose required DSH host service: systemPrompt"], "limitations": [], "components": [{"kind": "host", "status": "candidate", "reason": "Pre-built JavaScript entry requires restricted Node runtime discovery"}], "rejection_reason": "Blocking or unavailable dependencies cannot be satisfied in P0: ['@deepseek-ai/cordis', '@deepseek-ai/dsh-attachment', '@deepseek-ai/dsh-fs', '@deepseek-ai/dsh-invariants', '@deepseek-ai/dsh-llm', '@deepseek-ai/dsh-sandbox', '@deepseek-ai/dsh-sandbox-policy', '@deepseek-ai/dsh-session', '@deepseek-ai/dsh-system-prompt', '@deepseek-ai/dsh-tools', '@deepseek-ai/dsh-user-approval', '@deepseek-ai/schemastery', 'diff']. DSH packages must be self-contained pre-built bundles; npm install/build and architecture-bound DSH services are not available."}`
- Lifecycle: `{"install": {"ok": false, "error": "Blocking or unavailable dependencies cannot be satisfied in P0: ['@deepseek-ai/cordis', '@deepseek-ai/dsh-attachment', '@deepseek-ai/dsh-fs', '@deepseek-ai/dsh-invariants', '@deepseek-ai/dsh-llm', '@deepseek-ai/dsh-sandbox', '@deepseek-ai/dsh-sandbox-policy', '@deepseek-ai/dsh-session', '@deepseek-ai/dsh-system-prompt', '@deepseek-ai/dsh-tools', '@deepseek-ai/dsh-user-approval', '@deepseek-ai/schemastery', 'diff']. DSH packages must be self-contained pre-built bundles; npm install/build and architecture-bound DSH services are not available.", "verdict": "incompatible"}}`

### native_cross_plugin_service_consumer
- Source: `packages/extensions/cordis-host-runner/tests/helpers.ts#CONSUMER_CODE`
- Purpose: Real DSH composition fixture requiring a greeter service absent in LeapFlow P0.
- Static: `{"source_kind": "cordis_dynamic_export", "verdict": "incompatible", "installable": false, "installable_candidate": false, "category": "tools", "dependencies": [], "permissions": ["unknown.service.greeter"], "blockers": ["P0 does not expose required DSH host service: greeter"], "limitations": [], "components": [{"kind": "host", "status": "candidate", "reason": "Dynamic Cordis host requires restricted runtime discovery"}], "rejection_reason": "P0 does not expose required DSH host service: greeter"}`
- Lifecycle: `{"install": {"ok": false, "error": "P0 does not expose required DSH host service: greeter", "verdict": "incompatible"}}`

### native_ui_only
- Source: `packages/extensions/cordis-host-runner/tests/helpers.ts#CLIENT_CODE`
- Purpose: Real DSH browser-half fixture with no model-visible host tool.
- Static: `{"source_kind": "cordis_dynamic_export", "verdict": "incompatible", "installable": false, "installable_candidate": false, "category": "tools", "dependencies": [], "permissions": [], "blockers": ["Dynamic export contains no statically visible registerTool call; P0 does not publish private handler channels as LeapFlow tools"], "limitations": ["client.js UI was detected and will be skipped; only safe host tools can be installed"], "components": [{"kind": "host", "status": "candidate", "reason": "Dynamic Cordis host requires restricted runtime discovery"}, {"kind": "client", "status": "unsupported", "reason": "Cordis React/slots client UI is not executable in LeapFlow P0"}], "rejection_reason": "Dynamic export contains no statically visible registerTool call; P0 does not publish private handler channels as LeapFlow tools"}`
- Lifecycle: `{"install": {"ok": false, "error": "Dynamic export contains no statically visible registerTool call; P0 does not publish private handler channels as LeapFlow tools", "verdict": "incompatible"}}`

### native_unbuilt_typescript
- Source: `packages/context/time-context/src/index.ts`
- Purpose: Real native TypeScript source without a pre-built JavaScript entry.
- Static: `{"inspection_error": "P0 requires a pre-built JavaScript entry; TypeScript build/install is not supported", "error_type": "SourceInspectionError"}`
- Lifecycle: `{"install": {"ok": false, "error": "DSH source assessment failed: P0 requires a pre-built JavaScript entry; TypeScript build/install is not supported"}}`

### native_missing_entry
- Source: `packages/bundle/base/package.json`
- Purpose: Real native package manifest whose compiled entry is absent from the artifact.
- Static: `{"inspection_error": "DSH entry point does not exist: lib/index.js", "error_type": "SourceInspectionError"}`
- Lifecycle: `{"install": {"ok": false, "error": "DSH source assessment failed: DSH entry point does not exist: lib/index.js"}}`

### dynamic_shell_injection
- Source: `packages/extensions/cordis-host-runner/src/sandbox.ts#HOST_BUILTIN_INSPECTION`
- Purpose: Dynamic host tool whose runtime command violates the strict curl grammar.
- Static: `{"source_kind": "cordis_dynamic_export", "verdict": "adaptable", "installable": false, "installable_candidate": true, "category": "tools", "dependencies": [], "permissions": ["compat.shell.curl_get", "network.outbound"], "blockers": [], "limitations": [], "components": [{"kind": "host", "status": "candidate", "reason": "Dynamic Cordis host requires restricted runtime discovery"}], "rejection_reason": ""}`
- Lifecycle: `{"install": {"ok": true, "action": "install", "plugin_id": "dynamic_shell_injection", "installed_tools": ["unsafe_shell_probe"], "state": "active", "version": "native-exp-security", "source_kind": "cordis_dynamic_export", "bundle_sha256": "97b3cbccc4de0ef09f754e5617dd33997679f04256d9d278b903618aabd8bef1", "descriptor_path": "/Users/jason/work/github/leapflow/temp/plugin_exp/work/native-dsh/20260827-173327/profile/plugins/dsh/dynamic_shell_injection/descriptor.json", "verdict": "adaptable", "limitations": [], "client_components": []}, "invoke": {"ok": false, "error": "DSH shell compatibility only permits: curl -sS -m <1..120> '<http(s) URL>' [| iconv -f GB18030 -t UTF-8]", "error_type": "Error", "retryable": false}, "fetch_count": 0, "remove": {"ok": true, "action": "remove", "plugin_id": "dynamic_shell_injection", "state": "disposed", "source_path": "/Users/jason/work/github/leapflow/temp/plugin_exp/work/native-dsh/20260827-173327/profile/plugins/dynamic_shell_injection.py", "source_deleted": true}}`

## Conclusions

- A real dynamic `reverse_text` host tool executes through LeapFlow and survives wrapper rediscovery/reload.
- A host+client package is installable only as PARTIAL; the browser half is persisted as skipped metadata.
- Agent-loop replacement, unknown injected services, npm/peer dependencies, UI-only packages, missing builds, and missing entries fail before approval.
- A syntactically valid plugin cannot turn the shell shim into raw command execution; the forbidden command is rejected before `web_fetch`.
- The experiment does not need an LLM. Existing user LLM/cache configuration is probed without resolving secrets, and no credential value is emitted.
