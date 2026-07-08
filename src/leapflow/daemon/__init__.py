"""LeapFlow daemon (leapd) — centralized process for DuckDB + runtime.

The daemon architecture follows the "single process owns all mutable state"
principle. In single-window mode, the daemon runs in-process (zero IPC
overhead). In multi-window mode, a detached leapd process is spawned and
CLI instances connect as thin clients over Unix socket.

Submodules:

- ``protocol``: JSON-RPC message types and ``LeapService`` interface
- ``lifecycle``: PID/lock file management, health checks, process spawning
"""
