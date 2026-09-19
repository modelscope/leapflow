# Copyright (c) Alibaba, Inc. and its affiliates.
"""Phase 2 quarantine recovery path tests."""
from __future__ import annotations

import pytest

from leapflow.learning.plugin_trust import PluginTrustLedger, PluginTrustLevel


class TestUnfreeze:
    """PluginTrustLedger.unfreeze() resets a frozen plugin for re-probation."""

    def test_unfreeze_resets_trust_to_draft(self) -> None:
        ledger = PluginTrustLedger(candidate_at=2, demote_after=1)
        pid = "quarantined_plugin"

        # Earn some trust first, then hard-freeze
        for _ in range(3):
            ledger.record_success(pid)
        assert ledger.level(pid) == PluginTrustLevel.CANDIDATE

        ledger.record_failure(pid, hard=True)
        assert ledger.is_frozen(pid)
        assert ledger.level(pid) == PluginTrustLevel.DRAFT

        # Unfreeze
        result = ledger.unfreeze(pid)
        assert result is True
        assert not ledger.is_frozen(pid)
        assert ledger.level(pid) == PluginTrustLevel.DRAFT
        # Counters are zeroed — verify by recording success and checking promotion
        assert ledger._consecutive_ok.get(pid, 0) == 0
        assert ledger._consecutive_fail.get(pid, 0) == 0

        # Verify record_success works again after unfreeze
        ledger.record_success(pid)
        assert ledger._consecutive_ok[pid] == 1

    def test_unfreeze_nonexistent_returns_false(self) -> None:
        ledger = PluginTrustLedger()
        assert ledger.unfreeze("never_frozen") is False

    def test_unfreeze_nonfrozen_plugin_returns_false(self) -> None:
        ledger = PluginTrustLedger()
        pid = "normal_plugin"
        ledger.record_success(pid)
        assert ledger.unfreeze(pid) is False


class TestUnquarantineToolMetadata:
    """Verify plugin_unquarantine tool metadata meets governance requirements."""

    def test_unquarantine_requires_approval(self) -> None:
        from leapflow.plugins.tool_plugins.self_management import SelfManagementPlugin

        plugin = SelfManagementPlugin()
        tool = None
        for t in plugin.tools:
            if t.name == "plugin_unquarantine":
                tool = t
                break
        assert tool is not None, "plugin_unquarantine tool not found"
        assert tool.x_leapflow["risk_level"] == "high"
        assert tool.x_leapflow["category"] == "plugin_management"
        assert tool.mutates_state is True
