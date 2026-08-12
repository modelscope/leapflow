# LeapFlow 0.0.9 Release Notes

LeapFlow 0.0.9 focuses on real-time signal awareness, long-running task reliability, richer TUI feedback, safer approval controls, and stronger cross-session context recall.

## Features
- Added a real-time signal pipeline with event bridging, signal metrics, noise filtering, and mock signal generators for repeatable testing.
- Introduced LeapBoard signal observability improvements, including a redesigned signals board, server-driven UI updates, and signal templates.
- Added Semantic Focus Plane support for entity-aware context tracking and reference resolution.
- Added cross-session task history awareness with automatic session summaries and proactive history injection.
- Added TUI thinking display so LLM reasoning is surfaced in-place during execution and summarized in the final panel.
- Enabled terminal sessions by default for richer interactive workflows.

## Enhancements
- Hardened long-task convergence with false-progress detection, repeated-read guards, periodic checkpoints, and pre-compression knowledge extraction.
- Improved progressive context disclosure and tool-category handling for smaller, more accurate prompt assembly.
- Rebuilt the tool registry dynamically when late-registered tools arrive.
- Softened workspace boundary handling through approval-gated flows instead of hard refusal.
- Improved daemon build-staleness diagnostics and kept status hot paths non-blocking.
- Expanded runtime/build metadata reporting and configuration visibility.

## Fixes
- Fixed daemon-mode approval bypass behavior, including session-wide “Allow ALL” propagation.
- Fixed TUI thinking rendering so reasoning appears during active LLM rounds and is not duplicated around tool execution.
- Fixed session summary persistence and cross-session recall edge cases.
- Fixed `code_search` behavior and guidance when no regex pattern is provided.
- Fixed CUA client heartbeat/runtime issues and teach-mode runtime bugs.
- Fixed empty tool-response handling and shell/terminal tool-name drift.
- Hardened error classification and removed brittle engine keyword-rule paths.

## Docs and Tests
- Updated README release news for v0.0.9.
- Added coverage for context disclosure, semantic focus, signal buffering/noise, dashboard views, gateway flows, daemon transport, and build metadata.
- Reseeded journey cassettes and synchronized derived LLM response fixtures for deterministic replay.
- Added test harness isolation for background Copilot predictions to keep journey cassettes free of ambient desktop signals.
