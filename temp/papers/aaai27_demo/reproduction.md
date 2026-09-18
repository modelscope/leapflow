# AAAI Demo Reproduction Guide

## Evidence Levels

The demo keeps three intentionally separate evidence lanes:

- **L1 controlled mechanism evidence** is the CE-X four-arm counterfactual run. It supports no-op, rejection, catalog reuse, and acquisition-hypothesis routing under isolated profiles.
- **Real headless structural seam** uses two offscreen PyQt6 `BaseLeapApp` versions, role-aware state envelopes, `LeapSpaceEnvironmentSource`, and the shipped `CapabilityObservationService`. It proves a real widget rename can become an opt-in `environment_probe` requirement; it has no framebuffer and is not GUI-agent e2e.
- **Conditional signal archive** consists of a real Leapspace signal-mode trajectory, state/event evidence, and an unmodified `expect()` result declared in one archive manifest. Do not call this lane live or end-to-end unless every declared artifact is present and the verdict is PASS.
- **L3 runtime-governance evidence** requires the real daemon, approval route, lifecycle/trust trajectory, and rollback evidence. It is not inferred from L1, the headless seam, or a signal archive.

Never describe a missing lane as completed. The generated evidence bundle enumerates its own `claims.not_supported` list and the reviewer console renders those boundaries verbatim.

## Prerequisites

- A completed CE-X run directory containing `manifest.json`, `records.jsonl`, and `summary.json`.
- Python with the repository dependencies available.
- A PyQt6-enabled environment for the real offscreen structural-seam export. The repository's default lightweight environment may skip this lane.
- Optionally, a declared signal archive manifest with a recording, cursor path, state snapshot, event log, and `expect()` stdout.
- A Linux/KVM/AT-SPI environment is required to create new full GUI footage. On a host without that environment, use the L1 lane plus the real headless seam; do not synthesize a screen recording.

## Create a CE-X Run

```bash
PYTHONPATH=temp/leapspace_exp/evo-02 uv run python temp/leapspace_exp/evo-02/orchestrator/run_c2.py
```

The command prints the run directory. Preserve the entire directory as a raw experiment artifact. It contains substitutions that delimit the L1 claim, including the absence of real approval, installation, and independent GUI oracle coverage.

## Export the Real Headless Structural Seam

Run this only in the PyQt6-enabled environment. The output is write-once and contains hashes for the real widget envelopes, durable observation, requirement identity, benign negative control, and default refusal.

```bash
PYTHONPATH=temp/papers/aaai27_demo:temp/leapspace_exp/evo-02 uv run python -m aaai_demo.cli headless-seam \
  --output-dir temp/papers/aaai27_demo/demo/headless/<run-id>
```

## Declare a Signal Archive

A signal archive manifest is a JSON object with `schema_version: 1`, `kind: "leapspace_signal_archive"`, `run_id`, `expect: {"verdict": "pass", "exit_code": 0}`, and relative artifact entries of kinds `recording`, `cursor_path`, `state_snapshot`, `event_log`, and `expect_stdout`. Each artifact has `path` and optional `sensitive`. Missing files or a non-PASS verdict render the archive `incomplete`; they never become a successful L2 claim.

## Export the Evidence Bundle

Use a new output directory for every export. The exporter refuses to replace a bundle or its files.

```bash
PYTHONPATH=temp/papers/aaai27_demo:temp/leapspace_exp/evo-02 uv run python -m aaai_demo.cli bundle \
  --run-dir temp/leapspace_exp/runs/<c2-run-id> \
  --output-dir temp/papers/aaai27_demo/demo/evidence/<c2-run-id> \
  --drift-fixture temp/papers/aaai27_demo/demo_fixtures/headless-chat-probe-drifts.json \
  --headless-seam temp/papers/aaai27_demo/demo/headless/<run-id>/headless_seam.json \
  --signal-archive /absolute/path/signal_archive.json
```

Omit `--signal-archive` when no complete archive exists; the bundle will show `not_supplied`. `--trajectory-dir` remains a legacy media index and does not establish an independent oracle. Add `--include-media` only after confirming that the media is safe to place in the evidence package.

The bundle includes hashes for the CE-X source files, headless seam and signal-archive status, a compact causal trace, per-arm evidence boundaries, and the copied fixture. It intentionally labels CE-X effect confirmation as a handler-reported outcome rather than an independent `expect()` oracle.

## Render the Read-Only Console and Video EDL

```bash
PYTHONPATH=temp/papers/aaai27_demo uv run python -m aaai_demo.cli render \
  --bundle temp/papers/aaai27_demo/demo/evidence/<c2-run-id>/bundle.json \
  --output-dir temp/papers/aaai27_demo/demo/render/<c2-run-id>
```

Open `index.html` locally. It has four synchronized columns: Leapspace evidence, Observe/Orient, Decide/PCD, and Governance/Effect. The matrix gives no-op, rejection, reuse, and acquisition-hypothesis equal visual weight. The view distinguishes the legacy trajectory index, declared signal archive, and real headless seam. `storyboard.json` is the 300-second edit decision list; it includes narration constraints and prohibited claims.

The hidden LeapBoard `causal_trace` template is an alternative read-only lens over a live `framework_evolution` finding. It is hidden from ordinary navigation so it cannot broaden the normal product surface. Invoke it explicitly only for the experiment/audit workflow.

## Evolution Live Lens

`evolution_live` is a hidden, read-only LeapBoard lens over the same authoritative `framework_evolution` snapshot. Runtime traces are projected into Environment, Decision, and Governance lanes through `evolution.presentation` WebSocket messages. On reconnect, the browser fetches the authoritative snapshot again; the lens never writes a proposal, resolves approval, or runs a replay against the runtime.

## Provisional Accepted-Demo Poster Source

A poster is **not** a replacement for the required initial video/slides material. It is an accepted-demo station aid: the official call provides a poster board but defers final exhibit format information to the Chairs. Generate only the responsive source until that information arrives.

```bash
PYTHONPATH=temp/papers/aaai27_demo uv run python -m aaai_demo.cli poster \
  --bundle temp/papers/aaai27_demo/demo/evidence/<c2-run-id>/bundle.json \
  --output-dir temp/papers/aaai27_demo/demo/poster/<c2-run-id>
```

The output records `3:4 portrait` only as a provisional design ratio and explicitly marks the official print size as pending. Do not print or claim a final format before the Chairs provide the station specification.

## Backup Reel

After reviewing clip provenance and continuity, concatenate archived clips without re-encoding:

```bash
PYTHONPATH=temp/papers/aaai27_demo uv run python -m aaai_demo.cli backup-reel \
  --clip /absolute/path/intro.mp4 \
  --clip /absolute/path/trajectory.mp4 \
  --output temp/papers/aaai27_demo/demo/backup-90s.mp4
```

The helper uses `ffmpeg -n`, so an existing output is never replaced. The main five-minute video is assembled from the 300-second EDL with reviewed overlays and narration; the tool does not fabricate unavailable signal-archive, e2e, approval, or L3 footage.

## Verification

```bash
uv run pytest temp/papers/aaai27_demo/tests -q
uv run pytest tests/test_dashboard_sdui.py tests/test_dashboard_view.py -q
PYTHONPATH=src:temp/papers/aaai27_demo:temp/leapspace_exp/evo-02 \
  conda run -n leap pytest temp/papers/aaai27_demo/tests/test_demo_artifacts.py -q
```

Before recording or submission, verify all of the following:

1. Every visible claim maps to a hash-listed source artifact.
2. The four CE-X arms have four distinct profile roots and the expected action for each arm.
3. No-op and rejected arms appear in the rendered matrix and causal trace.
4. Any actual GUI footage has a declared signal archive, PASS `expect()` output, and aligned state/event artifacts.
5. Any headless structural seam record contains the rename, benign control, default refusal, and `environment_probe` requirement.
6. Any real mutation has its own approval record; L1 acquisition hypotheses must remain labelled as non-install evidence.
7. The video is at most 300 seconds, and the dashboard, paper captions, and evidence bundle use the same evidence language.
