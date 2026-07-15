from __future__ import annotations

from pathlib import Path

from leapflow.security.actions import ActionDescriptor
from leapflow.security.path_sensitivity import (
    classify_path_sensitivity,
    configure_path_sensitivity_roots,
)
from leapflow.security.risk import DefaultRiskClassifier, RiskLevel


def test_path_sensitivity_uses_configured_layout_root(tmp_path) -> None:
    data_root = tmp_path / "custom-leap-home"
    configure_path_sensitivity_roots((data_root,))
    try:
        vault_path = data_root / "profiles" / "default" / "secrets" / "vault.key"
        sensitivity = classify_path_sensitivity(vault_path)

        assert sensitivity.category == "secret_vault"
        assert sensitivity.scope == "profile"
        assert sensitivity.owner_component == "secrets"
        assert sensitivity.syncable is False
        assert sensitivity.hardline is True
    finally:
        configure_path_sensitivity_roots((Path("~/.leapflow").expanduser(),))


def test_risk_classifier_uses_configured_layout_root_for_shell_config_mentions(tmp_path) -> None:
    data_root = tmp_path / "custom-leap-home"
    configure_path_sensitivity_roots((data_root,))
    try:
        assessment = DefaultRiskClassifier().assess(
            ActionDescriptor.shell(f"cat {data_root / 'config' / 'user.yaml'}")
        )

        assert assessment.level == RiskLevel.HIGH
        assert "sensitive_config_reference" in assessment.reasons
        assert assessment.allow_permanent is False
    finally:
        configure_path_sensitivity_roots((Path("~/.leapflow").expanduser(),))


def test_path_sensitivity_classifies_new_layout_categories(tmp_path) -> None:
    data_root = tmp_path / "custom-leap-home"
    configure_path_sensitivity_roots((data_root,))
    try:
        cases = {
            data_root / "config" / "mcp_servers.json": "mcp_config",
            data_root / "profiles" / "default" / "history" / "tui_history": "history",
            data_root / "profiles" / "default" / "cache" / "workspaces" / "ws-1" / "workspace.yaml": "workspace_manifest",
            data_root / "profiles" / "default" / "cache" / "workspaces" / "ws-1" / "sessions" / "session-1" / "video" / "recording.mp4": "cache_sensitive",
        }
        for path, category in cases.items():
            sensitivity = classify_path_sensitivity(path)
            assert sensitivity.category == category
            assert sensitivity.requires_approval is True
            assert sensitivity.syncable is False if category in {"history", "cache_sensitive"} else sensitivity.syncable in {True, False}

        workspace_config = tmp_path / "workspace" / ".leapflow" / "config.yaml"
        sensitivity = classify_path_sensitivity(workspace_config)
        assert sensitivity.category == "config"
        assert sensitivity.scope == "workspace"
    finally:
        configure_path_sensitivity_roots((Path("~/.leapflow").expanduser(),))


def test_risk_classifier_blocks_permanent_allow_for_layout_sensitive_writes(tmp_path) -> None:
    action = ActionDescriptor.file_write(
        str(tmp_path / "workspace" / ".leapflow" / "workspace.yaml"),
        "version: 1\n",
        metadata={"sensitivity_category": "workspace_manifest"},
    )

    assessment = DefaultRiskClassifier().assess(action)

    assert assessment.level == RiskLevel.HIGH
    assert assessment.allow_permanent is False
    assert assessment.reasons == ("workspace_manifest_write",)
