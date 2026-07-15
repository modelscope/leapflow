from __future__ import annotations

from leapflow.config_loader import load_config_bundle
from leapflow.layout import build_layout
from leapflow.security.secrets import FernetSecretVault, secret_ref


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
