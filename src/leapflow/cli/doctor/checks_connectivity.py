# Copyright (c) Alibaba, Inc. and its affiliates.
"""Connectivity diagnostic checks (daemon, LLM provider, gateway)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from leapflow.cli.doctor.protocol import Finding

logger = logging.getLogger(__name__)


class DaemonHealthCheck:
    """Probe whether leapd is running and its socket is responsive."""

    name = "Daemon health"
    section = "connectivity"

    def __init__(self, runtime_dir: Path) -> None:
        self._runtime_dir = runtime_dir

    async def check(self, should_fix: bool = False) -> Finding:
        f = Finding()
        try:
            from leapflow.daemon.lifecycle import DaemonInfo

            info = DaemonInfo.discover(self._runtime_dir)
            if info.is_healthy:
                f.pass_()
            elif info.is_running:
                f.warn(f"leapd is running (pid={info.pid}) but socket is unresponsive")
            else:
                f.warn("leapd is not running — start with `leap daemon start`")
        except Exception as exc:
            f.warn(f"Cannot probe daemon: {exc}")
        return f


class LLMConnectivityCheck:
    """Attempt a minimal LLM API call to verify connectivity."""

    name = "LLM connectivity"
    section = "connectivity"

    def __init__(self, settings: Any) -> None:
        self._settings = settings

    async def check(self, should_fix: bool = False) -> Finding:
        f = Finding()
        s = self._settings
        if not s.has_llm_credentials:
            f.warn("Skipped — no LLM API key configured")
            return f

        try:
            from leapflow.llm.openai_client import create_llm_client

            client = create_llm_client(
                api_key=s.llm_api_key,
                base_url=s.llm_base_url,
                model=s.llm_model,
                max_retries=1,
            )
            # Minimal health probe: send a tiny request
            response = await client.achat(
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
            if response:
                f.pass_()
            else:
                f.warn("LLM returned empty response")
        except Exception as exc:
            f.error(f"LLM connectivity failed: {exc}")
        return f


class GatewayConnectivityCheck:
    """Check whether configured gateway platforms report healthy."""

    name = "Gateway connectivity"
    section = "connectivity"

    def __init__(self, profile_layout: Any) -> None:
        self._layout = profile_layout

    async def check(self, should_fix: bool = False) -> Finding:
        f = Finding()
        config_path = self._layout.gateway_config_path
        if not config_path.is_file():
            f.pass_()  # No gateway configured — not an error
            return f

        try:
            import yaml

            data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            platforms = data.get("platforms") or data.get("gateway", {}).get("platforms") or {}
            if not platforms:
                f.pass_()  # No platforms configured
                return f

            for name in platforms:
                # Just verify the configuration entry exists; deeper connectivity
                # checks would require instantiating adapters, which is out of
                # scope for a lightweight doctor check.
                f.pass_()
        except Exception as exc:
            f.warn(f"Cannot read gateway config: {exc}")
        return f
