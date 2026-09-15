# Copyright (c) Alibaba, Inc. and its affiliates.
from __future__ import annotations

from leapflow.config_loader import load_config_bundle
from leapflow.layout import build_layout
from leapflow.security.secrets import FernetSecretVault, secret_ref


def test_secret_vault_atomic_save_preserves_existing_file_on_replace_failure(monkeypatch, tmp_path) -> None:
    import leapflow.security.secrets as secrets_module

    vault = FernetSecretVault(tmp_path / "vault.json", tmp_path / "vault.key")
    first_ref = secret_ref("profile", "llm", "primary", "api_key")
    second_ref = secret_ref("profile", "llm", "aux", "api_key")
    vault.set(first_ref, "sk-original", metadata={"owner": "test"})

    def fail_replace(_src, _dst) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(secrets_module.os, "replace", fail_replace)

    try:
        vault.set(second_ref, "sk-new", metadata={"owner": "test"})
    except OSError:
        pass
    else:
        raise AssertionError("vault.set should propagate atomic replace failures")

    assert vault.get(first_ref) == "sk-original"
    assert vault.get(second_ref) is None


def test_config_loader_resolves_profile_secret_refs_without_writing_plaintext(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("LEAPFLOW_LLM_API_KEY", raising=False)

    layout = build_layout(tmp_path / "leap-home")
    profile_layout = layout.ensure(profile_id="default")
    api_key_ref = secret_ref("profile", "llm", "primary", "api_key")
    vault = FernetSecretVault(profile_layout.secrets.vault_path, profile_layout.secrets.key_path)
    vault.set(api_key_ref, "sk-profile-vault", metadata={"owner": "test"})
    profile_layout.llm_config_path.write_text(
        "llm:\n"
        f"  api_key_ref: {api_key_ref}\n"
        "  base_url: https://vault.example.invalid/v1\n"
        "  model: vault-model\n",
        encoding="utf-8",
    )

    bundle = load_config_bundle(layout, profile_layout, tmp_path)

    assert bundle.env["LEAPFLOW_LLM_API_KEY"] == "sk-profile-vault"
    assert bundle.env["LEAPFLOW_LLM_API_KEY_REF"] == api_key_ref
    assert "sk-profile-vault" not in profile_layout.llm_config_path.read_text(encoding="utf-8")
    assert layout.mcp_servers_path in bundle.watched_paths
    assert layout.workspace_config_path(tmp_path) in bundle.watched_paths

    monkeypatch.setenv("LEAPFLOW_LLM_API_KEY", "sk-process-override")
    overridden = load_config_bundle(layout, profile_layout, tmp_path)

    assert overridden.env["LEAPFLOW_LLM_API_KEY"] == "sk-process-override"


def test_config_loader_warns_on_missing_secret_ref_and_bad_section(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("LEAPFLOW_LLM_API_KEY", raising=False)
    layout = build_layout(tmp_path / "leap-home")
    profile_layout = layout.ensure(profile_id="default")
    missing_ref = secret_ref("profile", "llm", "primary", "missing")
    profile_layout.llm_config_path.write_text(
        "llm:\n"
        f"  api_key_ref: {missing_ref}\n"
        "cache: invalid-shape\n",
        encoding="utf-8",
    )

    bundle = load_config_bundle(layout, profile_layout, tmp_path)

    assert bundle.env["LEAPFLOW_LLM_API_KEY_REF"] == missing_ref
    assert "LEAPFLOW_LLM_API_KEY" not in bundle.env
    assert any("Missing secret ref" in warning for warning in bundle.warnings)
    assert any("section 'cache' must be a mapping" in warning for warning in bundle.warnings)


def test_mask_secret_reveals_only_a_short_suffix() -> None:
    from leapflow.config_service import _mask_secret

    assert _mask_secret("") == "missing"
    assert _mask_secret("   ") == "missing"
    # Long enough to reveal the last 3 characters as a recognizable hint.
    assert _mask_secret("sk-1234567890abc") == "***abc"
    # Too short to safely reveal any characters.
    assert _mask_secret("short") == "***"
    # Hardened threshold: even an 8-char secret is fully masked (revealing 3 of
    # 8 would expose too much of a short secret).
    assert _mask_secret("12345678") == "***"
    # Never leaks the full value.
    assert "1234567890" not in _mask_secret("sk-1234567890abc")


def test_daemon_log_level_config_chain(monkeypatch, tmp_path) -> None:
    """daemon.log_level flows through all three layers: Settings default,
    env override, and the discoverable config catalog (restart-required)."""
    from conftest import make_settings
    from leapflow.config import _build_settings_from_env
    from leapflow.config_service import _build_field_specs

    # Layer 1: Settings default keeps daemon file logs at INFO.
    settings = make_settings(str(tmp_path))
    assert settings.daemon_log_level == "INFO"

    # Layer 2: catalog exposes the key with restart-required semantics.
    specs = _build_field_specs()
    assert "daemon.log_level" in specs
    spec = specs["daemon.log_level"]
    assert spec.setting_name == "daemon_log_level"

    # Layer 3: LEAPFLOW_DAEMON_LOG_LEVEL overrides (the same channel the
    # config loader uses when flattening daemon.yaml values).
    monkeypatch.setenv("LEAPFLOW_DATA_DIR", str(tmp_path / "home"))
    monkeypatch.setenv("LEAPFLOW_DAEMON_LOG_LEVEL", "DEBUG")
    built = _build_settings_from_env()
    assert built.daemon_log_level == "DEBUG"


def test_logging_setup_is_idempotent_and_redacting(monkeypatch) -> None:
    """Centralized init: one tagged redacting handler on root, re-init only
    adjusts the level and never stacks duplicate handlers."""
    import logging

    from leapflow import logging_setup
    from leapflow.security.redact import RedactingFormatter

    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    try:
        for handler in original_handlers:
            root.removeHandler(handler)

        logging_setup.init_logging("INFO")
        owned = [h for h in root.handlers if getattr(h, "_leapflow_log_handler", False)]
        assert len(owned) == 1
        assert isinstance(owned[0].formatter, RedactingFormatter)
        assert root.level == logging.INFO

        # Re-init: level changes, handler count does not.
        logging_setup.init_logging("DEBUG")
        owned_again = [h for h in root.handlers if getattr(h, "_leapflow_log_handler", False)]
        assert owned_again == owned
        assert root.level == logging.DEBUG

        # Unknown level falls back to the surface default, never raises.
        logging_setup.init_logging("NOT-A-LEVEL")
        assert root.level == logging.WARNING
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
        for handler in original_handlers:
            root.addHandler(handler)
        root.setLevel(original_level)


def test_surface_level_policy_cli_quiet_daemon_verbose(tmp_path) -> None:
    """Level policy lives in one place: CLI/TUI defaults quiet (WARNING),
    the daemon surface defaults verbose (INFO) for leapd.log evidence."""
    import logging

    from leapflow import logging_setup

    resolved: list[int] = []

    class _Settings:
        log_level = ""
        daemon_log_level = ""

    original = logging_setup.init_logging
    try:
        logging_setup.init_logging = lambda level, *, default=logging.WARNING: resolved.append(
            logging_setup.resolve_level(level, default=default)
        )
        logging_setup.init_cli_logging(_Settings())
        logging_setup.init_daemon_logging(_Settings())
    finally:
        logging_setup.init_logging = original

    assert resolved == [logging.WARNING, logging.INFO]


def test_daemon_serve_applies_daemon_log_level(monkeypatch, tmp_path) -> None:
    """`leap daemon serve` must configure logging from daemon.log_level so
    INFO field evidence lands in leapd.log (previously nothing configured
    logging in the daemon process and only WARNING+ was emitted)."""
    import asyncio

    from leapflow.cli.commands import daemon as daemon_module

    applied: list[object] = []

    async def fake_serve(settings, *, mock_host=False):
        return 0

    monkeypatch.setattr(
        "leapflow.logging_setup.init_daemon_logging", lambda settings: applied.append(settings)
    )
    monkeypatch.setattr("leapflow.daemon.server.serve_daemon", fake_serve)

    class Settings:
        daemon_log_level = "DEBUG"

    settings = Settings()
    assert asyncio.run(daemon_module._serve(settings, mock_host=True)) == 0
    assert applied == [settings]
