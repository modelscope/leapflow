"""Telemetry taps: optional, opt-in observation points for runtime facts.

A tap is a module-level sink plus a one-line emit function. Absent a sink every
probe is a no-op, so a tap can be placed at a hot or low-level site without
imposing a dependency or a cost on it.
"""

__all__: list[str] = []
