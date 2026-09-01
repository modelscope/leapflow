# Declaring Device Readiness: Init, Homing, and Calibration

> **Audience**: Device declaration authors and hardware app-pack developers.
> **Authoritative source**: Derived from production code at
> `src/leapflow/hardware/context.py` (declarative protocol),
> `src/leapflow/hardware/tools.py` (write path, `_write` / `_not_ready` / dry-run),
> `src/leapflow/hardware/registry.py` (admission, V7 unverified policy), and
> `src/leapflow/security/permission_failures.py` (`build_readiness_failure`).
> **Scope**: This document covers **one** pattern — declaring that a device must
> reach a ready state (homed / initialized / calibrated) before a channel can be
> commanded, and how that declaration is enforced fail-closed. It does **not**
> describe a device state machine, calibration procedures, or persisted
> calibration data; those are deliberately out of scope (see §6).

---

## 1. What "readiness" means in the Hardware Context Protocol

A device is often unsafe or meaningless to command until it has completed an
initialization routine: a robot arm must be **homed** before its joints know
where they are; a depth camera must be **calibrated** before a captured frame
has any metric meaning. The Hardware Context Protocol (HCP) expresses this as a
**declared precondition on a writable channel**, not as procedural code and not
as a new field.

The key architectural fact — enforced by
`tests/test_architecture_contracts.py` — is that `context.py` carries **no
transport, vendor, or upstream-standard concept**. Readiness is therefore
declared with the two primitives that already exist:

- [`Envelope.requires_interlocks`](../../src/leapflow/hardware/context.py) — a
  tuple of interlock ids that must hold before a write to that channel is
  permitted.
- [`Interlock`](../../src/leapflow/hardware/context.py) — a deterministic
  channel comparison (`channel_id` + `operator` + `value`) that points at a
  **readiness channel** reporting whether the device has finished its init /
  homing / calibration routine.

There is **no** `DevicePhase`, `CalibrationState`, or `Procedure` type in the
current implementation. Readiness is nothing more than "a readable channel says
the device is ready, and a writable channel refuses to be commanded until it
does."

---

## 2. How readiness is enforced (the write path)

When a model calls `hw_actuate` / `hw_configure` / `hw_dispense`, the handler
`HardwareTools._write` (`src/leapflow/hardware/tools.py`) runs a fixed sequence
of feasibility checks **before** any approval prompt is shown, honoring the
platform rule that *feasibility precedes consent*:

1. resolve device + channel,
2. channel is writable,
3. effect class matches the tool,
4. `hw_describe` was called first (when `require_describe_before_write`),
5. value lies inside the declared envelope,
6. rate limit (`max_rate`) is respected,
7. device is reachable,
8. **evaluate `requires_interlocks`** via `_failed_interlocks`,
9. build the `ActionDescriptor`,
10. (dry-run stops here — see §4),
11. **if any readiness interlock is unmet → hard stop `_not_ready`** (this
    section),
12. approval gate,
13. execute against the transport.

### 2.1 `not_ready` is a fail-closed hard stop

If step 8 finds any unmet interlock, `_write` returns
`_not_ready(...)` **before consent is sought**. That refusal is built by the
single shared authority
[`build_readiness_failure`](../../src/leapflow/security/permission_failures.py),
so the engine and TUI report it identically. The payload is:

```json
{
  "ok": false,
  "device_id": "robot_arm_r1",
  "channel_id": "joint_shoulder",
  "failure_code": "not_ready",
  "failure_class": "device_not_ready",
  "blocks_approval": true,
  "retryable": true,
  "recoverability": "ready_state_required",
  "error": "robot_arm_r1.joint_shoulder is not ready to command: 'homed' requires homed_state == True. Bring robot_arm_r1 to its declared ready state -- run its initialization / homing / calibration routine so every precondition above holds, confirm it by reading the source channel back, then re-issue the same command. No approval was requested because the command cannot succeed until the device is ready.",
  "repair": {
    "kind": "device_readiness",
    "device_id": "robot_arm_r1",
    "channel_id": "joint_shoulder",
    "unmet": [
      {
        "interlock_id": "homed",
        "channel_id": "homed_state",
        "operator": "eq",
        "value": true,
        "description": "The arm must complete homing before any joint is commanded.",
        "declared": true
      }
    ]
  },
  "side_effect_state": "none"
}
```

Properties that matter:

- **`blocks_approval: true`** makes it a hard stop under
  `is_permission_hard_stop_payload`: the turn surfaces the deterministic repair
  instruction instead of giving the LLM another chance to retry or invent a way
  around it. No approval prompt is ever shown.
- **`retryable: true`** because the *identical* command becomes feasible once
  the preconditions hold — the fix is to make the device ready, not to change
  the command.
- **`side_effect_state: "none"`** — nothing reached the device.
- The **`error`** prose and the machine-readable **`repair.unmet`** array carry
  the same information, so both a human and an automated caller know exactly
  which precondition failed and on which source channel to confirm it.

### 2.2 Readiness fails closed on every uncertainty

`_failed_interlocks` treats a missing interlock, an unreadable source channel,
or a source read that raises **all** as unsatisfied: "cannot check" and "not
satisfied" carry the same consequence. An interlock named on a channel's
`requires_interlocks` but absent from the device's `interlocks` list is reported
with `"declared": false`, because the repair differs (fix the declaration, not
the device). The risk classifier keeps its own interlock hardline as
defense-in-depth for any descriptor built outside this path.

---

## 3. Authorizing calibration with `verified_by`

Readiness (§2) answers "has the device finished its routine?" A separate
question is "does a human vouch for this device declaration at all?" — which is
what authorizes writes in the first place. That is
[`ContextProvenance.verified_by`](../../src/leapflow/hardware/context.py) and
the **V7 admission rule** in `registry.py`.

- `ContextProvenance.is_verified` is simply `bool(verified_by.strip())`.
- Under the default policy `unverified_context_policy = "deny_write"`
  (`HardwareSettings`), an **unverified** context has **every writable channel
  demoted to read-only** at admission time (rule V7). A subsequent write then
  fails with `channel_not_writable`, not `not_ready` — the two are distinct
  causes.

Verification is stored **out of band** from the declaration on purpose: the
person who confirms a device must not edit the file they are confirming, or the
confirmation would be self-attested. The YAML provider reads a sibling
`verified.json` mapping `device_id → verifier` and stamps
`provenance.verified_by` on load (`YamlContextProvider._apply_verification`).

```json
// verified.json (sibling of the devices directory)
{
  "depth_cam_d1": "alice@lab (intrinsics+hand-eye checked 2026-08-30)"
}
```

For calibration-bearing devices this doubles as the **calibration
authorization**: a device whose captured data is only trustworthy after
calibration should ship **unverified**, so its writable channels stay demoted
until a human records — in `verified.json` — that calibration was performed and
checked. Set `verified_by` and the writable channels are admitted; leave it
empty and they are not.

---

## 4. Previewing with `dry_run`

Every write tool accepts `dry_run: true`
(`hw_actuate` / `hw_configure` / `hw_dispense`). A dry run executes **all**
feasibility checks in §2 (resolution, writability, effect class, describe,
envelope, rate, reachability, **and interlocks**), builds the approval
descriptor, and then **stops without seeking consent or touching the device**.
It is safe against an irreversible channel because nothing is written; the
result is a `WriteOutcome` with `preview: true` and `side_effect_state: "none"`.

The returned `plan` reports the command that *would* be issued together with the
outcome of every pre-consent check — including readiness:

```json
{
  "device_id": "robot_arm_r1",
  "channel_id": "joint_shoulder",
  "ok": false,
  "side_effect_state": "none",
  "preview": true,
  "failure_code": "interlocks_unsatisfied",
  "error": "Interlocks ['homed'] are not satisfied for robot_arm_r1.joint_shoulder, so the real command would be refused. Restore the interlock conditions before commanding it.",
  "plan": {
    "value_in_envelope": true,
    "interlocks_satisfied": false,
    "interlocks_failed": ["homed"]
  }
}
```

`plan.ok` is `true` only when the value is inside the envelope **and** every
interlock holds — the same two conditions that would otherwise let it reach
approval. This makes `dry_run` the recommended way to confirm both intent and
readiness before committing an irreversible physical effect.

> **Dispense note:** `effect=dispense` is treated as an irreversible external
> output regardless of the channel's `reversible` flag — a substance that has
> left the device cannot be un-dispensed.  Consequently, dispense writes
> **never** receive session-level or profile-level reusable consent
> (`allow_permanent` is always `false`); each dispense command is confirmed
> individually.

---

## 5. Complete examples

Device declarations are YAML files whose structure mirrors
`HardwareContext.to_dict()` / `from_mapping`. `hc_version` must be `hc.v0`.
`Interlock.operator` is one of `eq` / `ne` / `lt` / `le` / `gt` / `ge`
(default `eq`); `value` defaults to `true`.

### 5.1 Robot arm homing

A `homed_state` readiness channel reports whether homing has completed; a motion
channel declares `requires_interlocks: [homed]`, so a joint cannot be commanded
until the arm is homed.

```yaml
hc_version: hc.v0
device_id: robot_arm_r1
display_name: Bench robot arm R1
vendor: ExampleRobotics
model: RA-6
location: bench-3
halt_supported: true
notes: >-
  Six-axis arm. Joints must be homed before any motion command; homing establishes
  the absolute joint origin the motion channels are expressed against.

transport:
  kind: cli
  # Homing sequence, DH parameters, and the homing routine itself live here in the
  # transport/app-pack layer -- never in the HCP core declaration. See section 6.
  config:
    endpoint: "robotctl"
    home_command: "home --all"

channels:
  # Readiness channel: readable boolean the interlock points at.
  - channel_id: homed_state
    direction: read
    quantity: homing_state
    effect: read
    description: True once the arm has completed its homing routine.

  # Motion channel: refuses to be commanded until 'homed' holds.
  - channel_id: joint_shoulder
    direction: readwrite
    quantity: angle
    unit: deg
    effect: actuate
    verify_after_write: true
    envelope:
      declared: true
      min_value: -170.0
      max_value: 170.0
      max_rate: 45.0
      quantization: 0.01
      settling_time_s: 0.5
      reversible: true
      requires_interlocks:
        - homed
    description: Shoulder joint angle. Homing must complete before commanding it.

interlocks:
  - interlock_id: homed
    channel_id: homed_state
    operator: eq
    value: true
    description: The arm must complete homing before any joint is commanded.
```

Commanding `joint_shoulder` while `homed_state` reads `false` returns the
`not_ready` hard stop from §2.1; once homing completes and `homed_state` reads
`true`, the identical command proceeds to the approval gate.

### 5.2 Depth camera calibration

A `calibrated` readiness channel gates a `capture` channel, and the device is
declared **unverified** so its writable channel stays demoted until a human
records the calibration in `verified.json` (§3).

```yaml
hc_version: hc.v0
device_id: depth_cam_d1
display_name: Depth camera D1
vendor: ExampleVision
model: DC-2
location: cell-1
halt_supported: false
notes: >-
  Structured-light depth camera. A captured frame is only metrically meaningful
  after intrinsic + extrinsic (hand-eye) calibration has been performed and a human
  has recorded it. Ships unverified: the capture channel is admitted only once a
  verifier is recorded out of band.

transport:
  kind: cli
  # Intrinsic/extrinsic/hand-eye calibration algorithms, the camera-matrix format,
  # and convergence criteria all live here -- not in the HCP core. See section 6.
  config:
    endpoint: "depthcamctl"
    calibrate_command: "calibrate --hand-eye"

provenance:
  source: declared
  # verified_by is intentionally empty here. It is stamped out of band from
  # verified.json (device_id -> verifier) so the confirmation is not self-attested;
  # until then, V7 admission demotes 'capture' to read-only.
  verified_by: ""

channels:
  # Readiness channel: readable boolean the interlock points at.
  - channel_id: calibrated
    direction: read
    quantity: calibration_state
    effect: read
    description: True once intrinsic + hand-eye calibration has completed.

  # Capture channel: refuses to fire until calibration holds, and is admitted
  # writable only once the device is verified.
  - channel_id: capture
    direction: readwrite
    quantity: frame_request
    effect: actuate
    envelope:
      declared: true
      reversible: true
      requires_interlocks:
        - calibrated
    description: Trigger a depth-frame capture. Requires completed calibration.

interlocks:
  - interlock_id: calibrated
    channel_id: calibrated
    operator: eq
    value: true
    description: Calibration must complete before a capture is trusted.
```

Two independent gates apply here:

- **Authorization (V7 / `verified_by`)** — until `verified.json` records a
  verifier for `depth_cam_d1`, `capture` is demoted to read-only at admission;
  commanding it fails `channel_not_writable`.
- **Readiness (`requires_interlocks`)** — once verified, `capture` still refuses
  to fire with `not_ready` until `calibrated` reads `true`.

### 5.3 Temperature controller with tolerance and first-order settling (macOS host)

A real-device declaration (macOS host driver) demonstrating the `tolerance` and
`settling_model` fields added in Stage C (G-1 / G-2 protocol revisions).

`tolerance` declares the **absolute precision** of a channel. When `tolerance > 0`,
`normalized_delta` divides by `tolerance` instead of the envelope span — so a
tight-tolerance channel on a wide envelope reports error faithfully instead of
appearing misleadingly small (see §8 Q15 in the research document).

`settling_model: first_order` + `settling_tau_s` expresses a first-order
exponential settling behavior. The effective settling time is `5 * tau` (99 %
convergence), replacing the fixed `settling_time_s` wait for channels where a
scalar step-time is inadequate. When both `settling_time_s` and `settling_tau_s`
are declared, the system takes `max(settling_time_s, 5 * settling_tau_s)`.

```yaml
hc_version: hc.v0
device_id: temp_ctrl_t1
display_name: Peltier temperature controller T1
vendor: ExampleThermal
model: PTC-200
location: bench-1
halt_supported: true
notes: >-
  Peltier-based temperature controller with first-order thermal response. The
  heatsink sensor has 0.5 °C absolute precision (tolerance), and the PID loop
  settles exponentially with τ ≈ 2 s to the setpoint. Declaration via the
  leapflow_host macOS driver.

transport:
  kind: leapflow_host
  config:
    endpoint: "localhost:9710"
    bus: i2c
    device_address: 0x48

provenance:
  source: declared
  verified_by: "operator@lab (PID tuned, tolerance verified 2026-08-28)"

channels:
  # Readiness channel: PID loop reports stable
  - channel_id: pid_stable
    direction: read
    quantity: controller_state
    effect: read
    description: True once the PID loop has achieved stable regulation.

  # Temperature setpoint: first-order settling, tolerance-normalised
  - channel_id: setpoint
    direction: readwrite
    quantity: temperature
    unit: degC
    effect: configure
    verify_after_write: true
    envelope:
      declared: true
      min_value: 4.0
      max_value: 85.0
      max_rate: 5.0
      quantization: 0.1
      tolerance: 0.5
      settling_model: first_order
      settling_tau_s: 2.0
      settling_time_s: 3.0
      reversible: true
      requires_interlocks:
        - pid_ready
    description: >-
      Temperature setpoint in °C. tolerance=0.5 means normalized_delta divides
      by 0.5 (not by span 81); settling uses 5τ = 10 s (> settling_time_s 3 s,
      so effective = 10 s). PID must be stable before commanding.

  # Heatsink readback: read-only sensor with tolerance for observation scoring
  - channel_id: heatsink_temp
    direction: read
    quantity: temperature
    unit: degC
    effect: read
    envelope:
      declared: true
      min_value: -10.0
      max_value: 100.0
      tolerance: 0.5
    description: >-
      Heatsink temperature readback. tolerance=0.5 is used for
      observation scoring: a 0.3 °C deviation scores 0.3/0.5 = 0.6
      instead of 0.3/110 ≈ 0.003.

interlocks:
  - interlock_id: pid_ready
    channel_id: pid_stable
    operator: eq
    value: true
    description: PID controller must be stable before setpoint changes.
```

Key points demonstrated:

- **`tolerance: 0.5`** on `setpoint` and `heatsink_temp` — `normalized_delta`
  divides by 0.5 instead of `max_value − min_value`. A 0.3 °C error scores
  0.6, not 0.003.
- **`settling_model: first_order`** + **`settling_tau_s: 2.0`** — the effective
  settling wait is `max(settling_time_s, 5 × settling_tau_s)` = `max(3, 10)` =
  10 s. An observation arriving before 10 s after the command is not scored.
- **`settling_model` defaults to `"step"`** and **`settling_tau_s` defaults to
  `0.0`**: existing declarations without these fields behave exactly as before.

---

## 6. Boundary: what belongs to the driver / app-pack, not the HCP core

This is a direct application of AGENTS.md's **Platform vs App Business
Boundary**. The HCP core (`hardware/context.py` and the write path) owns only:

- the **readiness declaration** (`requires_interlocks` + `Interlock` pointing at
  a readiness channel),
- the **fail-closed gate** (`not_ready` before consent; unverified → V7
  demotion),
- **observability** of the refusal (the shared `build_readiness_failure`
  payload; `preview` plans).

Everything about *how* a device becomes ready is **third-party / app-pack**
concern, declared through `transport.config` and implemented in the
driver/transport, never in the HCP core declaration:

| Belongs to driver / transport / app-pack | Not in HCP core |
|---|---|
| The homing motion sequence and its ordering | — |
| Camera intrinsic / extrinsic / hand-eye calibration algorithms | — |
| DH-parameter tables, camera-matrix / distortion formats | — |
| Convergence criteria and tolerances for a calibration run | — |
| The command that actually runs the routine (e.g. `home --all`) | — |

The HCP core neither runs these routines nor understands their formats. It only
observes, through a declared readiness channel, whether they have *finished*,
and refuses writes until they have.

`CalibrationStore` (`hardware/calibration_store.py`) belongs to the
**storage / governance layer**: it persists versioned calibration results
(parameters, matrices, poses) to the profile's `instrument.duckdb` and surfaces
`last_calibrated_at` through `hw_describe`, but it does not know what a
calibration *is* — the algorithms, matrix formats, convergence criteria, and
hand-eye procedures remain driver / app-pack concerns, consistent with the
boundary above.

---

## 7. Out of scope (subsequent evolution)

This document covers **only** the readiness-gating pattern that ships today.

| Capability | Status |
|---|---|
| Persisted calibration data (parameters, matrices, poses) | **Implemented — IC-7** (`CalibrationStore` in `instrument.duckdb`, versioned; `hw_describe` outputs `last_calibrated_at`) |
| A full device state machine (`DevicePhase` / `CalibrationState`) | Not implemented — future (IC-5) |
| Multi-step `Procedure` orchestration for init/homing/calibration | Not implemented — future (IC-8) |
| Reference frames / pose representation | Not implemented — future (IC-10) |

Today, "readiness" is exactly a readable channel plus an interlock, backed by
versioned calibration storage. There is no state-machine type and no procedure
runner behind these primitives yet.
