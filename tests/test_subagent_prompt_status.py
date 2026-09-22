# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for Part A: subagent status injection into prompt volatile context.

Covers:
- SubagentManager.has_active() / render_active_status()
- PromptAssembler._active_subagent_status_section() PCD gating
"""
from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

from leapflow.engine.subagent import SubagentManager
from leapflow.engine.context.context_disclosure import (
    DisclosureLevel,
    PromptAssemblyPlan,
)


# ═══════════════════════════════════════════════════════════════════
# SubagentManager.has_active / render_active_status
# ═══════════════════════════════════════════════════════════════════


class TestSubagentManagerHasActive:
    """has_active() correctly reflects in-flight subagent state."""

    def test_no_active_subagents(self) -> None:
        mgr = SubagentManager()
        assert mgr.has_active() is False

    def test_has_active_when_tasks_registered(self) -> None:
        mgr = SubagentManager()
        # Simulate an in-flight task by inserting directly into _active
        fake_task = MagicMock()
        fake_task.done.return_value = False
        mgr._active["sub_abc123"] = fake_task
        assert mgr.has_active() is True

    def test_has_active_becomes_false_after_cleanup(self) -> None:
        mgr = SubagentManager()
        fake_task = MagicMock()
        mgr._active["sub_abc123"] = fake_task
        assert mgr.has_active() is True
        mgr._active.pop("sub_abc123")
        assert mgr.has_active() is False


class TestSubagentManagerRenderStatus:
    """render_active_status() returns correct content."""

    def test_empty_when_no_active(self) -> None:
        mgr = SubagentManager()
        assert mgr.render_active_status() == ""

    def test_renders_section_with_active_tasks(self) -> None:
        mgr = SubagentManager()
        fake_task1 = MagicMock()
        fake_task1.get_name.return_value = "subagent:sub_abc123"
        fake_task2 = MagicMock()
        fake_task2.get_name.return_value = "subagent:sub_def456"
        mgr._active["sub_abc123"] = fake_task1
        mgr._active["sub_def456"] = fake_task2

        status = mgr.render_active_status()
        assert "## Active Delegated Tasks" in status
        assert "sub_abc123" in status
        assert "sub_def456" in status
        assert "running" in status

    def test_zero_cost_when_empty(self) -> None:
        """Calling render_active_status with no active tasks does no work."""
        mgr = SubagentManager()
        # Should return empty string immediately
        result = mgr.render_active_status()
        assert result == ""


# ═══════════════════════════════════════════════════════════════════
# PromptAssembler._active_subagent_status_section (PCD gating)
# ═══════════════════════════════════════════════════════════════════


class TestActiveSubagentStatusSection:
    """The prompt section is gated by PCD level and active subagent state."""

    def _make_assembler(self) -> Any:
        """Create a minimal PromptAssembler with a mock engine."""
        from leapflow.engine.prompt_assembler import PromptAssembler

        engine = MagicMock()
        return PromptAssembler(engine)

    def test_no_section_at_core_level(self) -> None:
        """CORE level: no section even when subagents are active."""
        assembler = self._make_assembler()
        plan = PromptAssemblyPlan(level=DisclosureLevel.CORE)
        result = assembler._active_subagent_status_section(plan)
        assert result == ""

    def test_no_section_when_no_manager(self) -> None:
        """EXPANDED level but no SubagentManager available."""
        assembler = self._make_assembler()
        plan = PromptAssemblyPlan(level=DisclosureLevel.EXPANDED)
        with patch("leapflow.plugins.get_registry") as mock_reg:
            mock_reg.return_value._subagent_manager = None
            result = assembler._active_subagent_status_section(plan)
        assert result == ""

    def test_no_section_when_no_active_subagents(self) -> None:
        """EXPANDED level, manager exists, but no active subagents."""
        assembler = self._make_assembler()
        plan = PromptAssemblyPlan(level=DisclosureLevel.EXPANDED)
        mock_manager = MagicMock()
        mock_manager.has_active.return_value = False
        with patch("leapflow.plugins.get_registry") as mock_reg:
            mock_reg.return_value._subagent_manager = mock_manager
            result = assembler._active_subagent_status_section(plan)
        assert result == ""
        mock_manager.has_active.assert_called_once()

    def test_section_injected_at_expanded_with_active(self) -> None:
        """EXPANDED level + active subagents → section appears."""
        assembler = self._make_assembler()
        plan = PromptAssemblyPlan(level=DisclosureLevel.EXPANDED)
        mock_manager = MagicMock()
        mock_manager.has_active.return_value = True
        mock_manager.render_active_status.return_value = (
            "## Active Delegated Tasks\n- sub_abc: running"
        )
        with patch("leapflow.plugins.get_registry") as mock_reg:
            mock_reg.return_value._subagent_manager = mock_manager
            result = assembler._active_subagent_status_section(plan)
        assert "Active Delegated Tasks" in result
        assert "sub_abc" in result

    def test_section_injected_at_full_with_active(self) -> None:
        """FULL level + active subagents → section appears."""
        assembler = self._make_assembler()
        plan = PromptAssemblyPlan(level=DisclosureLevel.FULL)
        mock_manager = MagicMock()
        mock_manager.has_active.return_value = True
        mock_manager.render_active_status.return_value = (
            "## Active Delegated Tasks\n- sub_xyz: running"
        )
        with patch("leapflow.plugins.get_registry") as mock_reg:
            mock_reg.return_value._subagent_manager = mock_manager
            result = assembler._active_subagent_status_section(plan)
        assert "Active Delegated Tasks" in result

    def test_graceful_on_registry_import_error(self) -> None:
        """Exception in registry access → empty string, no crash."""
        assembler = self._make_assembler()
        plan = PromptAssemblyPlan(level=DisclosureLevel.EXPANDED)
        with patch(
            "leapflow.plugins.get_registry",
            side_effect=ImportError("mocked"),
        ):
            result = assembler._active_subagent_status_section(plan)
        assert result == ""
