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
