# LeapCalib

LeapCalib provides evidence contracts, offline replay, camera-extrinsic estimation,
and bounded task-local response estimates. Internals use meters/radians;
LeapFlow skills use millimeters/degrees.

**Offline candidate ≠ usable robot calibration.** `leap-calib` does not capture
hardware, move robots, or activate results. For real-hardware perception and
manipulation, follow the TUI workflow below. Camera-motion trials remain
shadow-only; they do not correct joint zeros.

## 1. Install and connect

Use the same LeapFlow revision on the host and the hardware-owning node. From the
repository root, in an activated Python 3.11+ environment:

```sh
# Host: CLI, TUI, and offline estimation.
python -m pip install -e '.[leapcalib]'
leap-calib --help

# Node (and any machine exporting the board): add vision support.
python -m pip install -e '.[leapcalib,vision]'
```

Install compatible vendor SDKs **only on the node**; these extras do not install
arm/camera drivers. Configure the host's LLM via the [main quickstart](../../README.md#quick-start).

For remote hardware, run `leap hw permit-key` on the host. It creates/reuses a
private signing key and prints its public PEM. Save **only the public PEM block**
as `host-permit.pub.pem` on the node. Choose a shared LHP token, replace the
paths/address/token below, and use a trusted network or authenticated tunnel:

```sh
# Node: after completing the declarations in step 2 below.
export LEAPFLOW_LHP_TOKEN='YOUR_SHARED_TOKEN'
export LEAPFLOW_LHP_PERMIT_ISSUER_KEYS='/absolute/path/host-permit.pub.pem'
leap hw serve --node-id edge01 --host 0.0.0.0 --port 8767

# Host: use the same token and the node's reachable address.
export EDGE_LHP_TOKEN='YOUR_SHARED_TOKEN'
export LEAPFLOW_FLEET_NODES='[{"node_id":"edge01","endpoint":"tcp://NODE_ADDRESS:8767","token_env":"EDGE_LHP_TOKEN"}]'
leap daemon restart
leap hw list
leap
```

Reuse an existing fleet if configured. Service changes require attended, idle
hardware. Board calibration needs no separate `leap-calib serve`; a connection
token alone does not authorize motion.

## 2. Prepare the workcell

On the node, use the selected profile's directories (default profile shown):

```text
~/.leapflow/profiles/default/hardware/devices/       # One device YAML per device
~/.leapflow/profiles/default/hardware/workcells/     # Your workcell YAML
```

Use the [workcell example](../leapflow/hardware/skills/workcell.py): set actual
device IDs, arm/gripper/camera roles, `tcp` and camera frame procedures, workspace
procedure, measured `workspace.bootstrap_box`, and task acceptance limits. Camera
parent: `flange` for wrist-mounted; `base` for fixed. Add `calibration_target`
with the board's square/marker dimensions, counts, dictionary, and legacy layout.
Never copy another robot's physical limits. On the node:

```sh
leap hw validate --workcells
leap hw verify YOUR_ARM_DEVICE_ID --as YOUR_OPERATOR_NAME
```

Verify each declaration needing writes. Validation prints admission results;
verification updates `hardware/verified.json`, **not camera accuracy**. Now start
the node from step 1. If already running, request a governed workcell reload in
the TUI; editing YAML alone does not reload it.

## 3. Export and mount the board

Run on a machine with the vision extra; use the node's actual workcell YAML
(copy it to the host if exporting there):

```sh
leap hw board --workcell /absolute/path/workcell.yaml --out calibration-board.svg
```

**Output:** a new SVG with exact dimensions, labels, and a 100 mm ruler; no
hardware is opened or existing file overwritten. Print at **100% / actual size**,
not fit-to-page. Measure the ruler and a square; mount on rigid, flat backing.

- Wrist camera: keep the board stationary in the workcell.
- Fixed camera: attach the board rigidly to the flange/gripper.

Keep the board centered in view and the entire planned motion area clear.

## 4. Calibrate from the TUI

These are **chat requests**, not shell/slash commands. Use the qualified workcell
from `hw_skill_list` (e.g. `edge01/bench`) and its camera role (e.g. `camera.wrist`).
Submit one step at a time; review each preview before approving.

| Step | What to ask in TUI chat | Result |
|---|---|---|
| Inspect | "List hardware and skills for YOUR_WORKCELL; describe readiness and missing calibration. Do not move." | Live device, skill, and frame status; no motion. |
| Bring up | "Run calib.bringup for YOUR_WORKCELL; show the gripper-travel preview first." | Bring-up record; gripper travel can open/close the jaws. |
| TCP | "Run calib.tcp with method=declared and offset=[X,Y,Z] mm from my measured flange-to-tool offset." | Versioned TCP transform; this step does not move. |
| Workspace | "Run calib.workspace with my measured x/y bounds, table_z, clearance_mm, and z_max; preview verification first." | Workspace record tied to the TCP; verification visits eight corners. |
| Camera | "Run calib.handeye for YOUR_WORKCELL with camera=CAMERA_ROLE; show the complete plan before approval." | Camera transform **and** exact-version held-out verification. |

Replace TCP/workspace placeholders with measurements; stop and measure if unknown.
Hand-eye previews the fit ring, held-out depth grid, and return. Both checks must
pass before writes. Keep the board in the same mount. After failure, inspect arm
state before another run: motion may already have occurred.

**Stored output:** node profile `db/instrument.duckdb`, keyed by workcell/procedure.
Camera records contain transform, intrinsics, target mount, training poses, and
fit residuals; verification contains p95/RMS/max depth errors and the exact camera
version. No manual matrix import or automatic `result.json` export is needed.

## 5. Use the result for real-hardware tasks

Remain in the same TUI/fleet; skills resolve the persisted calibration automatically.

1. Ask: "Check YOUR_WORKCELL camera/TCP/workspace readiness and measured accuracy
   versus required limits. Do not move." Proceed only with usable frames.
2. Ask: `Run perceive.locate with camera=CAMERA_ROLE and queries=["red cup"].`
   Add: "Report base-frame coordinates without moving." Requires a configured detector;
   returns fresh object IDs and geometry in mm, not a motion command.
3. Ask: "Run grasp.propose with object=OBJECT_ID; show candidates without moving."
   Replace `OBJECT_ID` with the fresh detection ID; output is grasp poses/IDs.
4. If `motion.sequence` is available, ask: "Plan a slow pick-and-lift using the
   selected grasp's pregrasp/grasp/retreat, opening/closing the gripper as needed.
   Show all steps and guards; do not execute before approval." Resolve readiness
   refusals, review, and approve separately. Then request `hw_skill_status` and
   `arm.state` to inspect the outcome; command delivery alone is not grasp success.

Calibration does **not** certify gripper pads, grasp success, clearance, kinematics,
or policies. Resolve the chosen skill's additional prerequisites; `grasp.cycle`
needs multiple cameras and separate grasp qualification. Re-observe expired IDs.

Later, request `calib.verify camera=CAMERA_ROLE` with the original board mounting.
After mount/intrinsics/TCP changes, restore affected calibration before precision
use. Measured p95 is setup-specific, not a universal guarantee. Fixed-camera
attestation/bridge details: [operator guide](../../docs/plugins/hardware_init_calibration.md#8-calibration-quickstart-operator).

## Optional: offline estimates

For an **existing, integrity-checked evidence bundle** produced by an adapter:

```sh
leap-calib inspect bundle.json
leap-calib replay bundle.json --only-accepted
leap-calib estimate bundle.json --method metric_motion.v1 \
  --request request.json --output candidate.json
```

`inspect`/`replay` print reports. `estimate` writes `candidate.json`: digest,
supported/retained parameters, holdout predictions, and limitations. Supply
`request.json` with the actual mounting `prior` (`rotation`, `translation_m`,
`parent`, `child`), optionally `prior_uncertainty`/`config`; see the
[bundle contract](validation/bundle.py) and [method options](estimation/handeye.py).
`shadow`/`budget` are also compute-only; use `leap-calib COMMAND --help`.

<details>
<summary>Board-free node experiments (shadow-only)</summary>

On the node before startup, run
`leap config set hardware.calibration_worker_enabled true`. Declare
`calibration_acquisition`, `calibration_reference`, and, for motion collection,
`calibration_motion`. Use [governed skills](../leapflow/hardware/skills/builtin/governed_calibration.py):

- `calib.collect`: pixel ROI `[x,y,width,height]` and frame count → stationary
  private captures/digests. `calib.align_reference` registers a task-local scene.
- `calib.collect_poses`: approved raw-joint poses, campaign reference, and
  clearance evidence → paired captures; moves and stays at the final pose.
- `calib.assess_motion`: acquisition pairs, predeclared training/holdout split,
  mount identity, and prior → shadow candidate/assessment; no motion or activation.

Raw captures expire in the private node cache; indices/assessments persist in
`instrument.duckdb`. TUI digests are **not** downloadable `bundle.json`. Numerical
success does not install a usable frame or replace steps 3–5. Never copy a shadow
candidate into active records.

</details>

## If a step is refused

- **Board scale mismatch / missing depth:** measure/reprint the board, check its
  declaration and depth alignment; usable depth is required for verification.
- **Unreachable poses / poor fit:** reposition within the safe area, improve
  visibility/rigidity, and follow the reported bounded parameter advice.
- **`insufficient_accuracy` / stale dependencies:** repeat combined hand-eye or
  use an independently verified reference; approval cannot bypass these checks.
- **Unverified device / missing permit:** fix declaration verification and the
  host public-key installation; never weaken safety or accuracy limits to pass.
