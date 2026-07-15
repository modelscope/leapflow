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
