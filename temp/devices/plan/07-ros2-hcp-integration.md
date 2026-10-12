# 07 — ROS2 and HCP Integration Plan

> Decision status: **approved future roadmap; not implemented**. Stages B–G remain
> pending, and every real-board mutation keeps its operation-specific approval gate.
> Scope: complete the ROS2 southbound path, verify the HCP contract end to end,
> and connect both to the existing LHP fabric without making ROS2 a second fabric.

## 1. Layering decision

The dependency direction remains fixed:

```text
Host LeapFlow
  HCP registry / tools / safety / trust
       │
       └── LHP v1 fabric ──> Jetson LeapFlow node (Python 3.13)
                                  │
                                  ├── HCP governance and local registry
                                  ├── ROS2 proxy transport (Unix socket)
                                  │      └── ROS2 sidecar (system Python 3.10 + rclpy)
                                  │             └── topics / services / actions
                                  └── stream descriptor ──> existing Foxglove :8765
```

- **HCP** is the device contract: declaration (`HardwareContext`) plus executable
  transport contracts (`HardwareTransport`, `BatchTransport`, `FrameTransport`).
- **ROS2** is a southbound device transport on a node. It does not perform fleet
  identity, host admission, cross-node trust, or fabric routing.
- **LHP** is the cross-node serialization and control plane for HCP. It remains the
  only LeapFlow fabric protocol.
- **Foxglove** remains the high-bandwidth camera/preview path. Raw image frames are
  not base64-encoded into LHP JSON frames.

This avoids parallel governance paths and preserves one safety/trust decision point
inside the Jetson LeapFlow process.

## 2. Coverage audit

### 2.1 ROS2 — current status: partial

| Area | Current implementation | Gap / decision |
|---|---|---|
| HCP six methods | `ROS2Transport` implements open/close/read/write/probe/halt | Keep and harden |
| Topic read/write | Subscription cache + publisher | Add explicit message types and per-channel QoS; remove unsafe String fallback for governed writes |
| Services | Client is created | Current future wait is not a valid bounded rclpy wait; replace with executor-safe completion + configurable timeout |
| Actions | Config and docstring claim support | No action client exists; implement goal/result/cancel and map uncertain effects correctly |
| Batch | Not implemented | Add `BatchTransport`; one coherent cache snapshot for reads and ordered/contained writes |
| Stream | Callback is hidden behind polled `read()` | Add HCP stream subscription for typed telemetry in P3 |
| Frames | Not implemented | Return governed stream descriptors; use Foxglove for camera bytes |
| QoS | BEST_EFFORT + depth only | Declare reliability, durability, history, deadline and sensor-data preset per channel |
| Timeouts | 5 s / 10 s and stale=2 s hardcoded | Settings/config fields with validation |
| Domain | Ambient environment only | Support explicit `domain_id`; sidecar owns ROS context initialization |
| Testing | Contract case is skipped without rclpy; no dedicated suite | Fake-rclpy unit suite + opt-in Jetson live tests |
| Runtime compatibility | Direct `rclpy` import | Jetson ROS2 Humble rclpy is system Python 3.10; LeapFlow runs in conda Python 3.13, so direct mode is not viable there |

### 2.2 HCP — current status: substantial, not complete

| Area | Covered | Remaining work |
|---|---|---|
| Versioning | hc.v0/hc.v1 parse and round-trip | Freeze a normative HCP v1 document and compatibility rules |
| Core declarations | channels, envelopes, interlocks, provenance | Add explicit schema invariants and conformance fixtures |
| hc.v1 extensions | kinematics, capabilities, safety, trust, degradation, control bindings | Verify every field has a runtime consumer or is explicitly advisory |
| Cross-node declaration | LHP `list` round-trips declarations | Node-private `transport.config` and `control_bindings` are stripped (RV2 R2-3/R2-7) |
| Safety | Node re-runs `check_safety_policy` before writes | Add vector/joint/FK/collision enforcement in P5; declaration alone is not enforcement |
| Trust | Local contexts are registered with `HardwareTrustGate` | RV2 now registers startup fleet contexts too; P2 must register/unregister changed remote contexts transactionally |
| Degradation | `DegradationCoordinator` consumes policy in stream paths | Connect fleet communication loss and recovery to per-device degradation policy |
| Batch | Contract and LHP serialization exist | Add ROS2 implementation and consistency semantics |
| Streaming/frame | Side protocols exist | No fabric subscription protocol or remote frame descriptor contract yet |
| Control bindings | YAML resolver maps `ros2_topics` to `TransportRef(kind="ros2")` | Extend resolver for sidecar backend and typed bindings; never export raw bindings over LHP |

## 3. Jetson runtime boundary

### 3.1 Constraint

The board has ROS2 Humble packages and `rclpy` for system Python 3.10, while the
approved LeapFlow deployment is `~/miniconda3/envs/leap` on Python 3.13. Installing
or importing the system `rclpy` wheel into that environment is unsupported and must
not be used as the solution.

### 3.2 Selected design: local ROS2 sidecar

Add a discoverable ROS2 transport backend with two modes:

1. `backend: direct` — existing in-process rclpy behavior for compatible hosts.
2. `backend: sidecar` — the Jetson production path:
   - a small ROS2 adapter runs under system Python 3.10;
   - communication uses a local Unix-domain socket, bounded JSON control messages,
     peer-credential checks and restrictive filesystem permissions;
   - the sidecar exposes ROS primitives only (graph/type discovery, subscribe,
     publish, service, action, cancel), not HCP admission or safety decisions;
   - `ROS2Transport` in LeapFlow Python 3.13 remains the HCP adapter and sole owner
     of HCP outcome classification;
   - the sidecar is started/stopped explicitly for validation first. Service
     enablement is a separate approval-gated operation.

The sidecar protocol must have its own version, maximum message size, request IDs,
timeouts and cancellation. It is node-local and must never be exposed on TCP.

### 3.3 Rejected alternatives

- **Force rclpy into Python 3.13:** ABI mismatch and unsupported deployment.
- **Make ROS2 the fabric:** duplicates LHP identity, admission, safety and trust.
- **Use rosbridge as the default control path:** adds another network service and
  authentication surface; useful as an optional third-party interop plugin only.
- **Use Foxglove for control:** its current role is observation/preview. Do not mix
  safety-critical actuation with the visualization endpoint.
- **Encode camera frames in LHP JSON:** inefficient and conflicts with bounded JSON
  control framing. LHP transports descriptors and low-rate metadata instead.

## 4. Implementation work packages

### R0 — Contract and test freeze

1. Add a normative HCP v1 specification covering field semantics, version
   negotiation, unknown fields, immutable identity, units, time, quality, sequence,
   side-effect classification and private bindings.
2. Add HCP fixtures for hc.v0, hc.v1 and unknown future optional fields.
3. Add ROS2 binding schema: backend, topics/services/actions, explicit types, QoS,
   stale threshold, availability/call/action timeouts, domain ID and stream policy.
4. Add architecture tests: ROS2 modules may depend on HCP; HCP core may not import
   ROS2, LHP or vendor SDKs.

Acceptance: schema round-trips; malformed contracts fail with typed errors; no
secrets or node-private bindings appear in LHP list payloads.

### R1 — Harden direct ROS2 transport

1. Split the 743-line transport before adding sidecar/action logic: common HCP
   adapter, direct-rclpy backend and codec/QoS helpers.
2. Correct service future completion and timeout semantics.
3. Implement action goal/result/cancel with side-effect states:
   - rejection before acceptance = `NONE`;
   - accepted goal or ambiguous timeout = `UNKNOWN`/`COMMITTED` as evidence allows;
   - cancellation must report whether motion may still be active.
4. Add validated configurable timeouts, stale threshold, QoS and domain ID.
5. Implement BatchTransport and deterministic partial-failure aggregation.
6. Add fake-rclpy tests with no ROS installation requirement.

Acceptance: dedicated tests cover open/close races, topic typing, service timeout,
action rejection/accept/cancel, halt independence and batch partial failure.

### R2 — Sidecar backend

1. Define the versioned local sidecar protocol and security limits.
2. Implement the Python 3.10 ROS adapter as a separate plugin/package boundary.
3. Implement the Python 3.13 Unix-socket backend behind the same ROS2 transport.
4. Add crash/restart supervision, bounded reconnect backoff and actionable health.
5. Preserve side-effect uncertainty across disconnects; never blind-retry accepted
   services/actions.

Acceptance: direct and sidecar backends pass the same transport contract; killing
the sidecar degrades the device without crashing LeapFlow; recovery rebinds without
losing node/session identity.

### R3 — HCP runtime completion

1. Register/unregister trust settings whenever fleet reconciliation adds/removes or
   changes remote contexts, not only at initial startup.
2. Route fleet communication loss into `DegradationCoordinator` using each remote
   context's `max_comm_loss_s`, fallback and halt policy.
3. Separate enforceable fields from advisory metadata; reject unsafe declarations
   that claim enforcement with no runtime consumer.
4. Add vector/joint and coordinate-frame schema only with the robot/FK consumer;
   avoid speculative fields.
5. Emit typed lifecycle/audit events for admission, degradation, recovery, trust
   changes and policy rejection.

Acceptance: hc.v1 semantics survive YAML → resolver → Jetson registry → LHP list →
host registry, while private bindings do not; tests prove trust/degradation behavior.

### R4 — Typed telemetry and preview federation

1. Add LHP subscribe/unsubscribe for low-rate typed readings with sequence,
   backpressure, bounded queues and resume epoch.
2. Keep RealSense image bytes on the existing Foxglove endpoint. Expose a governed
   HCP frame/preview descriptor containing topic, media type and authorized endpoint.
3. LeapBoard consumes the descriptor and never infers the Jetson endpoint.
4. Add capability negotiation through the hello `capabilities` field.

Acceptance: telemetry loss is detectable; slow consumers cannot exhaust node
memory; camera preview remains at current rates without burdening LHP control.

## 5. Real-board validation gates

Every implemented work package requires a Jetson gate. Read-only inspection is
allowed; each state-changing board operation is requested separately before use.

| Gate | Validation | State-changing approval |
|---|---|---|
| J-RV2 | Deploy current RV2 code; hot reload retains yahboom; stop/start shows degraded and back-online transitions | Code sync and LHP node stop/start |
| J-R0 | Read ROS graph/types/QoS and capture compatibility evidence | None unless a process must be started |
| J-R1 | Direct backend only where interpreter/rclpy are compatible | Starting test nodes; no actuator writes |
| J-R2 | Start sidecar in foreground, Unix socket integration, crash/recovery | Deploy sidecar and start/stop processes |
| J-R3 | Trust/degradation with simulated or read-only channels first | Any physical write/halt test is separate |
| J-R4 | Read-only RealSense topic/preview validation | Do not restart/reconfigure `realsense2_camera_node` or `foxglove_bridge` without approval |

Package/driver installation, service enablement, network/USB changes, peripheral
changes, reboot/power, firmware and physical writes are never bundled into a general
approval.

## 6. Reusing the macOS LLM configuration

The host profile currently uses:

- config file: `~/.leapflow/profiles/default/config/llm.yaml`;
- provider-compatible base URL and model settings in that file;
- API-key reference: `secret://profile/llm/primary/api_key`;
- encrypted vault: profile-local `secrets/vault.json` + `vault.key`.

Selected migration method:

1. Copy only the non-secret `llm.yaml` after board deployment approval.
2. Create the same secret reference on the board with
   `leap config secret set secret://profile/llm/primary/api_key` and its secure
   prompt; do not put the plaintext in shell arguments, logs, plan files or rsync.
3. Do not copy the host `vault.key`; generate/use the board vault and re-encrypt the
   value locally.
4. Validate with a minimal provider call, then inspect logs only for redacted
   provider/model/error metadata.

Transferring config and provisioning the board secret are state-changing and
sensitive operations; they require explicit approval. Secret values must never be
returned as validation evidence.

## 7. Definition of done

ROS2/HCP integration is complete only when:

- the direct and Jetson sidecar paths satisfy the same HCP conformance suite;
- services and actions have evidence-correct side-effect states;
- all hc.v1 fields used in production have verified runtime consumers;
- private node bindings never cross LHP;
- LHP remains the only fabric and ROS2 remains node-local;
- high-bandwidth frames use a bounded dedicated path;
- Mac↔Jetson tests pass for lifecycle, outage/recovery, trust/degradation and
  read-only camera preview;
- every approval-gated operation and its rollback is recorded in the regression
  result document.
