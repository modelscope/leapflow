# 08 — J-RV2 Real-Board Validation Results

> Status: **retained real-board evidence**; not an active runbook.
> Date: 2026-10-12
> Topology: macOS LeapFlow daemon ↔ Jetson Orin NX (`yahboom`) over LHP TCP/8767.
> Result: **passed after one additional reload defect was found and fixed**.

## 1. Approved operations

The user separately approved:

- synchronization of the reviewed source delta to `~/leapflow-src`;
- foreground start/stop of the Jetson LHP node;
- restart of the Mac LeapFlow daemon;
- two one-time daemon approvals for `plugin reload hardware_context` (the first
  reproduced the defect; the second verified the correction).

No package, driver, service, network, USB, camera, peripheral or physical actuator
change was performed.

## 2. Deployment and startup

| Check | Result |
|---|---|
| Source synchronization | Passed |
| Jetson imports `fleet_config` / `fleet_tools` | `RV2_IMPORT_OK` |
| Edge runtime | 9 devices, protocol 1, token required |
| Mac daemon fleet registration | 9 remote devices admitted; registry total 10 |
| Fleet tool publication before reload | `hw_fleet_discover`, `hw_fleet_halt_all`, `hw_fleet_status` present |

## 3. Reload defect found by the first board gate

The first real daemon reload returned an active generation but showed both symptoms:

1. a second `FleetManager initialized` line appeared with no following node
   registration; and
2. the live tool catalog lost the ordinary hardware and fleet tools.

The earlier unit fix covered repeated dependency binding on the **same plugin
instance**, but a true `ScopedToolRegistry.reload()` creates a **fresh plugin
instance**. The generic reload path discovered and published `plugin.tools` before
replaying runtime dependencies. `HardwareContextPlugin.tools` is dependency-generated,
so the published catalog was empty; its live FleetManager/heartbeat ownership was
also not transferred.

Correction (RV2 R2-9):

- `ScopedToolRegistry.reload()` now invokes optional state adoption and injects only
  the fresh plugin's relevant dependencies before registration/tool publication;
- it no longer rebinds every plugin after one plugin reload;
- `HardwareContextPlugin` steals the live registry, fleet manager, heartbeat and
  health monitor from the retiring instance, while rebuilding fresh tool handlers;
- the retiring instance is detached from transferred supervisors so its later
  teardown cannot close them.

Regression evidence: plugin reload + LHP/fleet + governance + transport contract +
CLI + architecture/inert wiring suites: **401 passed, 2 skipped**; changed-file ruff
passed.

## 4. Corrected hot-reload gate

After redeployment and daemon restart:

| Check | Result |
|---|---|
| Reload request | Approved once through daemon high-risk gate |
| Reload result | active generation 28 |
| Second FleetManager initialization after reload | **Absent** |
| Fleet tools after reload | all 3 present |
| Registry after reload | 10 devices total; `yahboom__*` devices retained |
| Remote identity | unchanged |

This verifies that the started manager and its heartbeat remain owned after a true
fresh-instance plugin reload.

## 5. Outage and recovery gate

| Transition | Evidence |
|---|---|
| Node stopped | approved `pkill -f '[l]eap hw serve'` |
| online → degraded | `15:01:33`, 10.1 s since last heartbeat |
| degraded → offline | `15:01:54`, 30.3 s since last heartbeat |
| Node restarted | edge runtime became ready with the same 9 devices |
| offline → online | `15:02:14`, `Fleet: node 'yahboom' back online` |
| Final state | edge node stopped; no matching process remains |

The Mac daemon remains running with the yahboom declaration and will report the node
offline while the approved final node state is stopped.

## 6. Credential handling incident and remediation

The first foreground SSH launcher echoed the test token as part of the spawned command
line. The value is therefore treated as compromised and is not reproduced here.
Immediate remediation:

1. stopped the node using that token;
2. generated a replacement token;
3. restarted using `LEAPFLOW_LHP_TOKEN` loaded from a file, so the replacement did
   not appear in command arguments or output;
4. updated the Mac fleet declaration to the replacement;
5. verified both the local and Jetson token files are mode `0600`.

The replacement value was never printed. The restricted Jetson token file remains at
`~/.leapflow/.lhp-token`; deleting it is a separate filesystem mutation and was not
performed.

## 7. Additional observation

Both normal Mac daemon restart attempts completed most logged shutdown cleanup but
did not exit within the lifecycle deadline; `daemon stop --force` was required before
a clean start. This did not affect J-RV2 correctness, but active-fabric graceful
shutdown must be added to Stage C as a separate lifecycle regression item.

## 8. Gate conclusion

J-RV2 is complete:

- private binding protection and trust registration are deployed;
- true plugin reload retains the live fleet and publishes the complete catalog;
- degraded, offline and back-online transitions are observed on real hardware;
- the Jetson is returned to a stopped node state;
- no physical hardware writes or camera/ROS2 reconfiguration occurred.
