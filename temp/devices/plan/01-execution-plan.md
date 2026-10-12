# 01 — Execution Plan (Completed Baseline and Remaining Work)

> Rule: one architecture, incremental delivery. Every implementation increment ends
> with local regression and a Jetson real-board gate. State-changing Jetson work is
> executed only after operation-specific approval.

## 1. Frozen architecture and completed baseline

| ID | Deliverable | Status | Evidence |
|---|---|---|---|
| P0 | Architecture decisions, signal classification and LHP v1 contract | Complete | `00-architecture.md`, `02-lhp-wire-protocol-v1.md` |
| P1 | LHP protocol/server/clients, LHP transport, registry external sources, fleet wiring, `leap hw serve` and startup registration | Complete | `05-review-notes.md` |
| RV1 | First full code review; six findings fixed | Complete | `05-review-notes.md` |
| R1 | Mac operator ↔ Jetson node real-board regression R1.1–R1.10 | Complete | `06-r1-results.md` |
| RV2 | Post-R1 review and optimization; token logging, hot-reload manager replacement, private binding exposure, hello parity, heartbeat concurrency, module split, dynamic trust registration | Complete | `05-review-notes.md` R2-1–R2-9; J-RV2 passed |
| J-RV2 | Real-board revalidation of RV2 hot reload and outage/recovery | Complete | `08-j-rv2-results.md`; R2-9 found/fixed/reverified |

The completed P1 scope deliberately excludes subscription streaming, frame bytes,
TLS/pairing, settings-backed fleet configuration and connection multiplexing.

## 2. Execution order and dependencies

```text
Stage A: RV2 board gate
   ↓
Stage B: HCP contract/spec + ROS2 tests and direct backend hardening
   ↓
Stage C: Fleet/Fabric lifecycle completion
   ↓
Stage D: Jetson ROS2 sidecar
   ↓
Stage E: typed telemetry + preview federation
   ↓
Stage F: governance, identity and audit
   ↓
Stage G: robot/peripheral convergence and safety completion
```

Stages B and C may be developed in parallel after Stage A, but Stage D consumes both.
Cross-cutting Stage I in `10-human-approval-deployment.md` runs with C/F: interaction
lifecycle work aligns with C, and node-verifiable execution permits align with F. Any real
writable remote channel blocks on I4/I5 or stays inside the explicit trusted-v1 boundary.
No camera or actuator write is used to validate infrastructure that can be proven with
fake ROS2 nodes or read-only topics.

## Stage A — RV2 real-board gate (complete)

### A1. Completed scope

Execution evidence is retained in `08-j-rv2-results.md`.

1. Synchronized the reviewed source changes to `~/leapflow-src`.
2. Keep the approved conda environment `~/miniconda3/envs/leap` and editable install.
3. Start the LHP node in foreground.
4. Start/restart the Mac daemon with the yahboom fleet declaration.
5. Trigger one plugin reload and prove the same started `FleetManager` remains active.
6. Stop/start the node and prove degraded/offline and `back online` transitions.
7. Stop the edge node after evidence capture.

### A2. Acceptance

- `hw_fleet_status` still shows `yahboom` after reload;
- no second unstarted manager replaces the active manager;
- heartbeat transition logs are present and contain no token/config secret;
- node-private `transport.config` and `control_bindings` are absent from `list`;
- all processes return to the user-approved final state.

### A3. Approval boundary

Source synchronization, editable-install refresh, edge node start/stop and Mac daemon
restart are separate state-changing approvals. No package, service, network, USB,
camera or actuator changes are included.

## Stage B — HCP contract and direct ROS2 hardening

Authoritative details: `07-ros2-hcp-integration.md` R0/R1.

| ID | Task | Deliverable | Acceptance |
|---|---|---|---|
| B1 | Freeze normative HCP v1 semantics and compatibility | HCP v1 specification + fixtures | hc.v0/v1/future-optional round-trip |
| B2 | Classify every hc.v1 field as enforced, consumed or advisory | coverage matrix + tests | no field silently implies enforcement |
| B3 | Freeze typed ROS2 binding schema | validated config model | invalid timeout/QoS/type rejected early |
| B4 | Split ROS2 implementation by backend/codec | files remain below soft limits | no behavior fork in HCP adapter |
| B5 | Correct service completion and timeouts | direct backend fix | success, unavailable, timeout and ambiguous effect tests |
| B6 | Implement actions and cancellation | action adapter | rejection/accept/cancel side effects proven |
| B7 | Add BatchTransport | coherent reads + ordered writes | partial failure aggregation tests |
| B8 | Add fake-rclpy suite | dedicated tests | runs without ROS installation |

Jetson gate B-J: read-only ROS graph, type and QoS inventory. Direct-rclpy tests run
only in a compatible interpreter and do not justify modifying the board environment.

## Stage C — Fleet to Fabric lifecycle completion

| ID | Task | Acceptance |
|---|---|---|
| C1 | Settings-backed node declarations with env override compatibility | config reload test; no secret logging |
| C2 | Retry/backoff with jitter and bounded startup latency | multiple unreachable nodes cannot block online nodes |
| C3 | Re-handshake and device-set reconciliation | add/remove/change is transactional |
| C4 | Keep-last-degraded on loss and clean recovery | device identity remains stable |
| C5 | Dynamic trust registration/unregistration | remote trust settings follow reconciliation |
| C6 | Route comm loss through per-device degradation policy | max loss, fallback and halt policy tests |
| C7 | Correct `hw_fleet_discover/status` semantics | endpoint and device-level health are visible |
| C8 | Capability routing by affordance, health and trust | deterministic route tests |
| C9 | Fleet isolation benchmark | `fleet_isolation.py` passes |
| C10 | Graceful daemon shutdown/restart with an active fabric | exits within deadline without `--force`; no lingering executor/task/thread |

Jetson gate C-J: node disappearance, changed declaration and recovery using simulated
or read-only channels. Physical write/halt validation is not implied.

## Stage D — Jetson ROS2 sidecar

Authoritative design: `07-ros2-hcp-integration.md` §3 and R2.

| ID | Task | Acceptance |
|---|---|---|
| D1 | Versioned bounded Unix-socket ROS primitive protocol | codec/security tests |
| D2 | System-Python-3.10 rclpy sidecar | topic/service/action fake-node tests |
| D3 | Python-3.13 sidecar backend for `ROS2Transport` | same HCP suite as direct backend |
| D4 | Peer validation, restrictive socket permissions and no TCP binding | security tests |
| D5 | Crash/restart supervision and bounded reconnect | kill/recover test; no blind replay |
| D6 | Plugin discovery and lifecycle | injected/disposed through normal machinery |

Jetson gate D-J starts the sidecar in foreground first. Package installation and
service enablement require separate approvals and are not part of initial validation.

## Stage E — Typed telemetry and preview federation

| ID | Task | Acceptance |
|---|---|---|
| E1 | LHP subscribe/unsubscribe capability negotiation | old peers continue polling |
| E2 | Sequence epoch, bounded queues and backpressure | drops are observable; memory bounded |
| E3 | Host sampling uses read_batch/subscription where available | one coherent remote sample |
| E4 | Connection multiplexing | requests and subscriptions share lifecycle safely |
| E5 | Governed HCP preview descriptor | endpoint/topic/media type are explicit |
| E6 | LeapBoard preview uses existing Foxglove stream | no frame base64 in LHP JSON |
| E7 | Optional node reading buffer and replay | bounded outage merge with epochs |

Jetson gate E-J is read-only RealSense/Foxglove validation. The existing camera node
and bridge on port 8765 are not restarted or reconfigured without approval.

## Stage F — Governance, identity and audit

| ID | Task | Acceptance |
|---|---|---|
| F1 | Pairing token → unique per-node credential → remote mTLS profile | rotation/revocation tests |
| F2 | Node operation evidence upload and audit merge | end-to-end causal record |
| F3 | Conservative cross-node trust aggregation | lowest proven trust wins |
| F4 | `PhysicalEnvironmentSource` wiring | typed relevant deltas only |
| F5 | Requirement resolution before proposal | satisfiable requirements are durable no-ops |
| F6 | Evolution ledger provenance includes node/plugin/version | rejected and rollback branches queryable |

## Stage G — Peripheral convergence and safety completion

| ID | Task | Acceptance |
|---|---|---|
| G1 | Robot provider/transport converges on HCP | no parallel robot contract |
| G2 | FK and collision-zone enforcement consumer | unsafe command denied node-side |
| G3 | Episode export and learning chain | governed evidence round-trip |
| G4 | Voice input as a SignalSource plugin | classification before activation |
| G5 | Observe-only restricted-node profile | deny-write default verified |

## Cross-cutting Stage I — Human approval deployment hardening

Authoritative design: `10-human-approval-deployment.md`.

| ID | Task | Acceptance |
|---|---|---|
| I0 | Freeze edge-only/dual topology and current baseline | plan/code/test audit complete (`09`, `10`) |
| I1 | Plugin `ApprovalChannel` + exactly-once `InteractionBroker` | TUI behavior preserved; multi-channel race contained |
| I2 | Durable pending ownership and restart reconciliation | no orphan or implicit approval after disconnect/restart |
| I3 | Local web/operator and governed IM channels | authenticated approver identity and failover |
| I4 | LHP v2 signed per-action execution permits | wrong node/action/expiry/replay denied before transport |
| I5 | Permit-required remote writes + explicit migration profiles | no permit means `side_effect_state=none` |
| I6 | Independent host/edge agent authority rules | action origin has exactly one authority/audit chain |

No Jetson action is needed for I0. I1–I5 use simulated/deny-write channels first. A physical
write requires a later, separate operation-specific approval after deterministic gates pass.

## 3. LLM configuration reuse on the Jetson

The Mac profile's non-secret LLM configuration may be reused. The execution is:

1. copy `profiles/default/config/llm.yaml` only;
2. preserve `secret://profile/llm/primary/api_key`;
3. create that reference in the board's own vault through the secure CLI prompt;
4. do not copy the host vault key and never place plaintext in a command, log or
   evidence file;
5. validate with one minimal request and redacted logs.

Copying configuration and provisioning a secret are separate sensitive board changes
and require explicit approval.

## 4. Verification policy

Each stage must pass, in order:

1. focused unit/contract tests;
2. affected hardware/plugin/architecture suites;
3. non-e2e regression with baseline failures separated;
4. lint/type diagnostics for changed files;
5. one recorded Jetson gate with exact versions, commands, outcomes and rollback.

A stage is not complete when only mocks pass. Conversely, real-board validation must
not be used to bypass deterministic local tests.

## 5. Known risks and controls

| Risk | Control |
|---|---|
| ROS2 Humble rclpy is Python 3.10 while LeapFlow is Python 3.13 | local sidecar; do not force ABI mixing |
| Service/action timeout after possible execution | evidence-correct `UNKNOWN`; verify before retry |
| Remote bindings expose paths/tokens/topology | strip transport config and control bindings at LHP boundary |
| Clock/sequence reset across node restart | wall-clock observation + local monotonic + session epoch in Stage E |
| Stream overload | bounded queues, declared rates, backpressure and drop counters |
| Camera traffic starves control | Foxglove/dedicated frame path; LHP remains bounded control/telemetry |
| New ROS bridge becomes a second fabric | sidecar is node-local ROS primitive adapter only |
| Board validation destabilizes working camera/network | read-only first; operation-specific approvals and rollback |
