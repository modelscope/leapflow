# Journey cassette replay — environment isolation

Journeys run against a real `leapd` subprocess with the LLM boundary served by a
cassette proxy (replay / seed / record / live).  Deterministic replay depends on
every provider request fingerprinting identically across machines and runs, so any
environmental source of non-determinism must be isolated.

## Standard environment variables

| Variable | Default in journeys | Reason |
|---|---|---|
| `LEAPFLOW_MEMORY_INTEGRATION_ENABLED` | `0` | Memory prefetch injects a `## Recent Context` block into the system prompt from the profile's signal store.  That store picks up ambient desktop events (app-focus, clipboard, filesystem) whose top-k ordering is timing- and environment-dependent.  The block reaches the model, enters the cassette fingerprint, and makes replay miss when the ambient signal differs from the seed run. |
| `LEAPFLOW_COPILOT_ENABLED` | `0` | Copilot autonomy may inject unscripted LLM turns, breaking the scripted-turn ↔ cassette contract. |

These are set in each journey's `extra_env` dict.  Any new journey that scripts
hardware, board, or multi-turn flows should carry the same pair unless it
explicitly tests the memory or copilot layer.

## Memory-prefetch PCD non-determinism

The signal-based prefetch in `_prefetch_and_freeze_memory` queries the PCD
(Persistent Context Database) for recent signals that match the user's request
keywords.  The returned entries depend on:

1. **Ambient signals** — desktop focus events, clipboard snapshots, filesystem
   changes — which differ between CI, developer laptops, and headless test
   runners.
2. **Top-k ordering** — when several signals score similarly, the ranking is
   unstable across runs.
3. **Signal timestamps** — epoch-second scrubbers normalise individual values
   but cannot normalise *which* signals appear and *how many*.

Because the prefetched block is injected into the system prompt, any variation
changes the provider request body and invalidates the cassette fingerprint.

### Current mitigation

`LEAPFLOW_MEMORY_INTEGRATION_ENABLED=0` disables both narrative memory and
signal prefetch, removing the non-deterministic block entirely.

### Future direction (PCD-layer)

A signal-level SNR / replayability classification would let the PCD mark signals
as replay-safe (deterministic, derived from the declared workspace) vs.
replay-unstable (ambient, hardware-dependent).  With that classification,
prefetch could filter to safe signals during `LEAPFLOW_TEST_LLM_MODE=replay`,
allowing journeys to exercise the memory layer without sacrificing determinism.
