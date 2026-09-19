# Errors

## [ERR-20260918-P2S] sandbox-resource-default

**Logged**: 2026-09-18T13:52:25Z
**Priority**: high
**Status**: resolved
**Area**: backend

### Summary
A default RLIMIT_AS ceiling caused Python sandbox workers to exit before responding on macOS.

### Error
```
Sandbox smoke test failed: worker did not respond
```

### Context
- The plugin sandbox applied a 512 MiB address-space limit before importing runtime modules.
- macOS virtual address-space accounting exceeded the limit during normal worker startup.

### Suggested Fix
Keep memory limits configurable but disabled by default unless validated for the deployment platform.

### Metadata
- Reproducible: yes
- Related Files: src/leapflow/plugins/sandbox/worker.py, src/leapflow/config.py

### Resolution
- **Resolved**: 2026-09-18T13:52:25Z
- **Notes**: Default memory limit changed to zero; explicit configured limits remain enforced.

---

## [ERR-20260918-LNT] repository-wide-ruff-baseline

**Logged**: 2026-09-18T13:52:25Z
**Priority**: medium
**Status**: resolved
**Area**: tests

### Summary
Repository-wide Ruff validation reports pre-existing unused imports outside the evolution implementation surface.

### Error
```
ruff check src tests: 33 F401/F811 findings
```

### Context
- Targeted Ruff checks for all files changed by the evolution plan pass.
- The remaining findings are in unrelated perception, platform, marketplace, and older test modules.

### Suggested Fix
Run a dedicated repository-wide lint cleanup and verify affected modules with their focused tests.

### Metadata
- Reproducible: yes
- Related Files: src/leapflow/perception/signal_source.py, src/leapflow/platform/event_bus.py, tests/test_signal_source.py

### Resolution
- **Resolved**: 2026-09-18T13:52:25Z
- **Notes**: Applied safe Ruff fixes and replaced the remaining assigned lambdas with local functions; repository-wide Ruff now passes.

---

## [ERR-20260918-EVP] event-projection-migration-regressions

**Logged**: 2026-09-18T14:30:00Z
**Priority**: medium
**Status**: resolved
**Area**: tests

### Summary
Targeted tests exposed stale constructor and lazy JSON-store assumptions during the event-derived knowledge migration.

### Error
```
3 failed: proposal_sink constructor argument, lazy JSON knowledge binding, missing injected event knowledge store
```

### Context
- DurableTeacherWorker now requires an event-backed proposal queue and live resolver.
- AgentEngine no longer creates a JSON knowledge store on the hot path.
- A production-wiring unit fixture did not inject the new event-derived knowledge projection.

### Suggested Fix
Update tests and fixtures to use the event store, inject the knowledge projection explicitly, and assert fail-closed acquisition resolution.

### Metadata
- Reproducible: yes
- Related Files: tests/test_durable_teacher.py, tests/test_distilled_knowledge.py, tests/test_degradation_feedback_loop.py

### Resolution
- **Resolved**: 2026-09-19T00:30:00Z
- **Notes**: Updated worker tests for atomic proposal preparation, replaced lazy JSON binding with explicit event-projection injection, and fixed production governor fixtures. Full suite passes.

---
