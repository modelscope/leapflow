# 10 — Human Approval Across Edge-Only and Dual Deployments

> Decision status: selected architecture target; current behavior and implementation gaps
> are explicitly separated below.
> Principle: approval authority follows the action-owning session/control plane. Physical
> safety remains local to the hardware node. Neither can replace the other.

## 1. Terms and invariants

- **Control-plane owner:** the LeapFlow runtime that owns the `AgentEngine` session and
  decides to request an action. It owns approval policy and interaction lifecycle.
- **Hardware node:** the runtime that owns the device declaration, transport and local
  `SafetyPolicy`. `leap hw serve` is a node data plane, not an approval UI.
- **Approval authority:** `ApprovalOrchestrator` plus grants/audit. Exactly one authority
  resolves any action; topology must never create two independent prompts for one write.
- **Approval channel:** a user-facing plugin that presents a structured request and returns
  an authenticated decision. TUI, local web, IM and a dedicated operator console are
  channels, not policy engines.
- **Execution permit:** a short-lived, single-use authorization bound to one canonical
  mutating action. It proves to a remote node that the control-plane authority allowed the
  exact operation; it does not bypass node safety.

Non-negotiable invariants:

1. feasibility, declaration, envelope, rate, reachability and interlock checks run before
   asking a human;
2. missing/unhealthy approval authority fails closed with an actionable response;
3. ordinary non-private reads do not prompt; declared privacy reads require consent;
4. irreversible actions and `dispense` never inherit reusable consent;
5. emergency stop never waits for approval or network round trips and remains node-local;
6. an approval is bound to action, session, workspace, node/device/channel and value/band;
7. duplicate, stale, wrong-session or already-consumed decisions are rejected;
8. node `SafetyPolicy` is authoritative even when a valid permit exists;
9. uncertain physical effects are verified before retry, never authorized by replaying the
   earlier decision;
10. all request, delivery, decision, permit, execution and refusal states are auditable.

## 2. Current implementation baseline

### 2.1 Direct interactive runtime

`LeapContext` constructs `_TUIApprovalGate` → `SessionAwareGate` →
`ApprovalOrchestrator`, composes the hardware risk classifier, and binds the orchestrator
to `HardwareContextPlugin`. This supports a full edge-only deployment when an interactive
TUI is actually attached to the edge runtime.

### 2.2 Daemon-backed runtime

`ApprovalCoordinator.install_gate()` replaces the direct gate with
`_DaemonApprovalGate`. A watched turn/RPC installs a per-request `ContextVar` route;
`approval_request` is streamed to that client, and `approval.resolve` completes the
pending future. Pending requests do not time out while their owner is live, and are denied
when the owning turn, stream or connection ends. This is correct for one live client but
is in-memory and route-local.

### 2.3 LHP node runtime

The LHP shared token is connection authentication. `LhpNodeServer` independently verifies
writability and `SafetyPolicy`, then writes. It does not receive or validate a human
approval record. Therefore:

- dual deployment currently relies on the authenticated host being the trusted approval
  control plane;
- a standalone `leap hw serve` process cannot ask a human and must not originate writes;
- possession of the v1 token is effectively authority to submit any node-safe write.

That last property is the boundary to remove before remote nodes are exposed beyond a
single trusted operator domain.

## 3. Deployment behavior

| Topology | Action owner | Human path | Mutating behavior when path is absent |
|---|---|---|---|
| Edge-only, direct TUI | Edge process | local TUI gate | deny; reads/estop remain available |
| Edge-only, edge `leapd` + local client | Edge daemon | daemon stream to local TUI/web | deny/cancel when route disappears |
| Edge-only, headless | Edge daemon | configured ApprovalChannel plugin (IM/web/operator console) | park durably if owner is checkpointed; otherwise deny |
| Host + edge node | Host daemon/session | host TUI/web/IM; one decision, then signed permit to edge | no permit → node rejects mutation |
| Both host and edge run independent agents | runtime owning the originating session | that runtime's authority; no proxy prompt by the other agent | no authority/channel → that action fails closed |
| Network partition | whichever side owns the session | existing local channel only | no new remote mutation; local estop/safety continue |

A dual deployment must not send a raw “please approve” request from node to host after the
host has already approved the action. The host decides once and sends a verifiable permit.
If an edge-local agent originates a different action, its edge authority decides it.

## 4. Target architecture

```text
AgentEngine/session
      │ ActionDescriptor
      ▼
ApprovalOrchestrator (existing policy/risk/grants/audit authority)
      │ ASK
      ▼
InteractionBroker (core lifecycle; durable pending state; exactly-once resolve)
      ├── ApprovalChannel plugin: TUI
      ├── ApprovalChannel plugin: local web/operator console
      └── ApprovalChannel plugin: governed IM gateway
      │ authenticated decision
      ▼
ExecutionPermitIssuer (control-plane credential)
      │ permit + exact LHP write/write_batch
      ▼
ExecutionPermitVerifier (node) → local SafetyPolicy → device transport
      │ outcome/readback/audit evidence
      └──────────────────────────────► control-plane audit
```

### 4.1 `ApprovalChannel` plugin

Add a `runtime_checkable` Protocol discovered and disposed through plugin machinery:

```python
class ApprovalChannel(Protocol):
    channel_id: str
    async def present(self, interaction: PendingInteraction) -> DeliveryReceipt: ...
    async def withdraw(self, interaction_id: str) -> None: ...
    async def health(self) -> ChannelHealth: ...
```

The channel never evaluates risk and never executes tools. It authenticates the responder,
renders the request, and submits a signed/authorized decision to the broker. Vendor-specific
IM identities, scopes, callbacks and URLs stay in gateway adapters/action packs.

### 4.2 `InteractionBroker`

The broker is the single lifecycle owner for human interactions. It stores immutable
request content plus state transitions:

```text
CREATED → DELIVERED → RESOLVED_ALLOW | RESOLVED_DENY | CANCELLED | OWNER_GONE
```

Required fields include `interaction_id`, `request_id`, `session_id`, `workspace_id`,
`action_digest`, risk/explanation, allowed choices, owner liveness/checkpoint, channel
deliveries, approver identity/role, decision time and audit correlation ID.

Resolution uses compare-and-set: the first authorized decision wins; later channel replies
receive “already resolved”. A channel outage does not silently become consent. Pending
requests have no human-response timeout while a live/checkpointed owner exists. If the owner
cannot resume, the broker cancels them fail-closed.

### 4.3 Execution permit

For every remote mutation, issue a signed permit containing at least:

- issuer and key ID; subject node identity;
- session/workspace and approval/audit correlation IDs;
- canonical action digest covering method, device, channel(s), value(s), effect and declared
  envelope band;
- issuance and short execution expiry; nonce/idempotency key;
- allowed use count (`1` for irreversible actions); policy/grant scope;
- approver identity reference without leaking channel secrets.

The node verifies signature, node audience, digest, expiry, nonce and replay cache before
its existing write precheck. Permit validation failure is provably `side_effect_state=none`.
A permit is authorization evidence, not a retry token. Batch permits cover the entire ordered
batch and cannot be split.

LHP v1 is frozen, so this belongs in LHP v2 (or a separately negotiated capability that
increments the protocol contract). Do not silently add optional fields to v1 writes and call
them secure. Until v2 is deployed, remote write is allowed only inside the explicitly trusted
v1 control-plane boundary; observe-only nodes should use deny-write admission.

## 5. UX and recovery rules

1. Present what will happen: device/location, channel, value/unit, envelope position,
   reversibility, interlocks, readback plan and reason.
2. Offer only policy-valid scopes. Personal/environment reads never get permanent consent;
   irreversible writes are once-only.
3. Show delivery state in every surface and allow another authenticated channel to answer
   the same pending interaction without creating another request.
4. On disconnect, retain the request only if the action owner is durably checkpointed and
   can resume safely. Otherwise cancel and explain that no effect was attempted.
5. On daemon restart, reconcile pending records: resume checkpointed owners, cancel all
   others, and withdraw stale channel notifications.
6. On execution timeout/connection loss after permit acceptance, mark effect `UNKNOWN` and
   require readback/inspection. Never mint a second permit automatically.
7. A channel can be absent without crashing LeapFlow. Mutations fail closed; monitoring,
   diagnostics and emergency stop remain usable.

## 6. Implementation plan (cross-cutting Stage I)

| ID | Work | Dependency | Acceptance |
|---|---|---|---|
| I0 | Freeze this topology contract; correct stale comments/docs; add deployment-mode test matrix | now | docs and current tests agree |
| I1 | Add `ApprovalChannel` Protocol and `InteractionBroker`; adapt TUI/daemon route as the first channel | before headless edge mutation | existing UX unchanged; multi-channel CAS tests |
| I2 | Add durable pending store and owner checkpoint/reconcile semantics | I1 + Stage C lifecycle | restart/disconnect tests leave no orphan or implicit allow |
| I3 | Add local web/operator channel and governed IM channel plugins | I1 | authenticated identity, channel health and fallback tests |
| I4 | Add node/authority identity, key rotation and signed execution permits in LHP v2 | Stage F identity | wrong-node/digest/expiry/replay tests all reject before transport |
| I5 | Enforce permit-required remote mutation; retain explicit observe-only and trusted-v1 migration modes | I4 | no permit means `NONE`; compatibility is explicit, never inferred |
| I6 | Add dual-authority/failover rules for independently acting edge agents | I2–I5 | action origin has exactly one authority and one audit chain |

I1 and I2 should be implemented alongside Stage C because both depend on daemon lifecycle.
I4 and I5 belong with Stage F identity/mTLS. I3 may proceed after I1. Any Stage B–G work
that introduces a real writable remote channel must block on I4/I5 or explicitly remain in
the trusted-v1 test boundary.

## 7. Verification plan

### Deterministic local/CI gates

- direct TUI approval for privacy read, reversible write and irreversible write;
- daemon streamed approval with two concurrent workspaces and strict session isolation;
- no watched route, disconnected client and owner cancellation all fail closed;
- two channels race to answer; one wins and one receives already-resolved;
- broker restart resumes only checkpointed owners;
- permit signature, audience, action digest, expiry, nonce and replay rejection;
- node safety rejects a valid permit for an unsafe value;
- post-send timeout becomes `UNKNOWN` and cannot auto-replay;
- emergency stop remains independent of broker/network health.

### Jetson gates

No board action is needed to approve this plan. After implementation:

1. I1–I3: run an edge `leapd` with a simulated writable device; verify local and remote
   approval channels without physical effects.
2. I4–I5: deploy the permit-capable node and prove valid permit, missing permit, altered
   action, replay and expiry paths against a mock/deny-write channel.
3. Only after those pass, request a separate operation-specific approval for one reversible,
   bounded physical write with readback. No camera, network, driver, service or peripheral
   change is implied by that approval.

Every source sync, package change, process start/stop, service/network change and physical
write on the Jetson remains separately approval-gated with stated command, impact and
rollback.

## 8. Chosen solution and rejected alternatives

Chosen: one session-owned approval authority, pluggable interaction channels, durable
exactly-once lifecycle, and signed per-action permits enforced by the node.

Rejected:

- **Prompt independently on host and node:** duplicate decisions can disagree and leave no
  single audit authority.
- **Treat the LHP shared token as approval:** it authenticates a client, not human intent for
  one action, and is replayable across actions.
- **Put vendor IM logic in hardware/fabric core:** violates platform neutrality and plugin
  boundaries.
- **Auto-approve when no user channel is reachable:** converts an availability problem into
  a safety failure.
- **Make every read prompt:** creates approval fatigue; privacy classification already gives
  a signal-driven boundary.
- **Require the cloud for edge-only approval:** breaks local autonomy and failure recovery;
  local channels must remain possible.
