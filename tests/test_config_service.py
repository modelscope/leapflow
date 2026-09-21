# Copyright (c) Alibaba, Inc. and its affiliates.
"""Behavioral tests for ConfigService: set/unset/configure_llm/secret CRUD,
snapshot rendering, key normalization, value coercion, atomic YAML writes,
and secret masking.

Uses real layout/vault objects rooted under tmp_path; no production FS access.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
import yaml

from leapflow.config_service import (
    ConfigService,
    _coerce_value,
    _mask_secret,
    _normalize_key,
    _write_yaml_atomic,
)


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture()
def config_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Isolated Settings + ConfigService rooted under tmp_path."""
    home = tmp_path / "leapdata"
    monkeypatch.setenv("LEAPFLOW_DATA_DIR", str(home))
    monkeypatch.setenv("LEAPFLOW_WORKSPACE_ROOT", str(tmp_path / "workspace"))
    (tmp_path / "workspace").mkdir()
    # Reset the global singleton so get_settings() rebuilds from the tmp env.
    import leapflow.config as _cfg

    prev = getattr(_cfg, "_settings_instance", None)
    _cfg._settings_instance = None
    try:
        settings = _cfg.get_settings()
        yield settings, ConfigService(settings)
    finally:
        _cfg._settings_instance = prev


# ═══════════════════════════════════════════════════════════════════
# _normalize_key
# ═══════════════════════════════════════════════════════════════════


class TestNormalizeKey:
    """Test env-var and underscore-to-dot key normalization."""

    def test_leapflow_prefix_stripped(self) -> None:
        assert _normalize_key("LEAPFLOW_LLM_MODEL") == "llm.model"

    def test_leapflow_prefix_single_segment(self) -> None:
        assert _normalize_key("LEAPFLOW_RUNTIME") == "runtime"

    def test_bare_underscore_becomes_dot(self) -> None:
        assert _normalize_key("runtime_log_level") == "runtime.log_level"

    def test_dotted_key_passes_through(self) -> None:
        assert _normalize_key("llm.model") == "llm.model"

    def test_whitespace_is_trimmed(self) -> None:
        assert _normalize_key("  llm.model  ") == "llm.model"


# ═══════════════════════════════════════════════════════════════════
# _coerce_value
# ═══════════════════════════════════════════════════════════════════


class TestCoerceValue:
    """Test type coercion for config values."""

    def test_bool_truthy_strings(self) -> None:
        for text in ("true", "yes", "1", "on", "True", "YES"):
            assert _coerce_value(text, bool) is True

    def test_bool_falsy_strings(self) -> None:
        for text in ("false", "no", "0", "off"):
            assert _coerce_value(text, bool) is False

    def test_bool_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Expected boolean"):
            _coerce_value("maybe", bool)

    def test_int_coercion(self) -> None:
        assert _coerce_value(" 42 ", int) == 42

    def test_float_coercion(self) -> None:
        assert _coerce_value("3.14", float) == pytest.approx(3.14)

    def test_dict_from_yaml_string(self) -> None:
        result = _coerce_value('{"a": 1}', dict)
        assert result == {"a": 1}

    def test_dict_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Expected mapping"):
            _coerce_value("just-a-string", dict)

    def test_list_from_csv(self) -> None:
        result = _coerce_value("a, b, c", list)
        assert result == ["a", "b", "c"]

    def test_list_passthrough(self) -> None:
        result = _coerce_value(["x", "y"], list)
        assert result == ["x", "y"]


# ═══════════════════════════════════════════════════════════════════
# _mask_secret
# ═══════════════════════════════════════════════════════════════════


class TestMaskSecret:
    """Test secret masking renders safely for any length."""

    def test_empty_renders_missing(self) -> None:
        assert _mask_secret("") == "missing"
        assert _mask_secret(None) == "missing"  # type: ignore[arg-type]

    def test_short_secret_fully_masked(self) -> None:
        # Fewer than 16 chars: suffix must not leak
        assert _mask_secret("sk-short") == "***"

    def test_long_secret_reveals_last_three(self) -> None:
        long_key = "sk-1234567890abcdef"
        result = _mask_secret(long_key)
        assert result.startswith("***")
        assert result.endswith(long_key[-3:])
        assert len(result) == 6  # "***" + 3 suffix chars


# ═══════════════════════════════════════════════════════════════════
# ConfigService.set / unset
# ═══════════════════════════════════════════════════════════════════


class TestConfigSetUnset:
    """Test YAML persistence and scope validation for set/unset."""

    def test_set_writes_yaml_and_returns_changed_key(self, config_env) -> None:
        settings, svc = config_env
        result = svc.set("llm.model", "test-model-x")

        assert result.ok is True
        assert "llm.model" in result.changed_keys
        assert result.path is not None
        # Verify on-disk YAML
        data = yaml.safe_load(result.path.read_text("utf-8"))
        assert data["llm"]["model"] == "test-model-x"

    def test_set_rejects_unsupported_scope(self, config_env) -> None:
        _, svc = config_env
        with pytest.raises(ValueError, match="does not support scope"):
            svc.set("llm.api_key", "val", scope="workspace")

    def test_set_restart_required_section_emits_warning(self, config_env) -> None:
        _, svc = config_env
        result = svc.set("daemon.log_level", "DEBUG")
        assert result.ok
        assert any("restart" in w for w in result.warnings)

    def test_set_hot_reload_section_no_warning(self, config_env) -> None:
        _, svc = config_env
        result = svc.set("runtime.log_level", "DEBUG")
        assert result.ok
        assert result.warnings == ()

    def test_unset_removes_key_preserving_siblings(self, config_env) -> None:
        _, svc = config_env
        svc.set("llm.model", "keep-this")
        svc.set("llm.base_url", "https://example.invalid/v1")

        result = svc.unset("llm.base_url")
        assert result.ok
        data = yaml.safe_load(result.path.read_text("utf-8"))
        assert "base_url" not in data.get("llm", {})
        assert data["llm"]["model"] == "keep-this"


# ═══════════════════════════════════════════════════════════════════
# ConfigService.configure_llm
# ═══════════════════════════════════════════════════════════════════


class TestConfigureLlm:
    """Test the batch LLM configuration helper."""

    def test_batch_values_applied(self, config_env) -> None:
        _, svc = config_env
        result = svc.configure_llm(model="my-model", base_url="https://api.example.com/v1")

        assert result.ok
        assert "llm.model" in result.changed_keys
        assert "llm.base_url" in result.changed_keys

    def test_api_key_goes_through_secret_vault(self, config_env) -> None:
        settings, svc = config_env
        result = svc.configure_llm(api_key="sk-super-secret")

        assert result.ok
        assert "llm.api_key" in result.changed_keys
        # The YAML must store a ref, never the plaintext key.
        data = yaml.safe_load(result.path.read_text("utf-8"))
        ref_value = data.get("llm", {}).get("api_key_ref", "")
        assert ref_value.startswith("secret://"), f"Expected secret ref, got {ref_value!r}"

    def test_no_changes_returns_not_ok(self, config_env) -> None:
        _, svc = config_env
        result = svc.configure_llm()
        assert result.ok is False
        assert "No LLM config changes" in result.message


# ═══════════════════════════════════════════════════════════════════
# ConfigService.set_secret / delete_secret
# ═══════════════════════════════════════════════════════════════════


class TestSecretCrud:
    """Test secret CRUD through ConfigService."""

    def test_set_and_get_secret_roundtrip(self, config_env) -> None:
        _, svc = config_env
        svc.set_secret("test/mykey", "hunter2", scope="profile")
        value = svc.get_secret("test/mykey", scope="profile", reveal=True)
        assert value == "hunter2"

    def test_delete_secret(self, config_env) -> None:
        _, svc = config_env
        svc.set_secret("test/delme", "tmp", scope="profile")
        result = svc.delete_secret("test/delme", scope="profile")
        assert result.ok

        with pytest.raises(KeyError):
            svc.get_secret("test/delme", scope="profile")

    def test_get_secret_without_reveal(self, config_env) -> None:
        _, svc = config_env
        svc.set_secret("test/hidden", "s3cret", scope="profile")
        msg = svc.get_secret("test/hidden", scope="profile", reveal=False)
        # Must NOT contain the plaintext
        assert "s3cret" not in msg
        assert "is set" in msg


# ═══════════════════════════════════════════════════════════════════
# ConfigService.snapshot
# ═══════════════════════════════════════════════════════════════════


class TestSnapshot:
    """Test snapshot field enumeration and secret masking."""

    def test_snapshot_enumerates_all_writable_keys(self, config_env) -> None:
        _, svc = config_env
        snap = svc.snapshot()
        snap_keys = {v.key for v in snap.values}
        writable = set(svc.writable_keys())
        assert snap_keys == writable

    def test_snapshot_masks_secrets(self, config_env) -> None:
        _, svc = config_env
        snap = svc.snapshot()
        secrets = [v for v in snap.values if v.secret]
        for sv in secrets:
            # Secret values must be masked or show 'missing'
            assert sv.value in ("missing", "***") or sv.value.startswith("***")


# ═══════════════════════════════════════════════════════════════════
# Atomic YAML write and permissions
# ═══════════════════════════════════════════════════════════════════


class TestAtomicYamlWrite:
    """Test _write_yaml_atomic creates the file with correct content/mode."""

    def test_write_creates_parent_and_correct_content(self, tmp_path: Path) -> None:
        target = tmp_path / "sub" / "dir" / "config.yaml"
        _write_yaml_atomic(target, {"section": {"key": "val"}})

        assert target.exists()
        data = yaml.safe_load(target.read_text("utf-8"))
        assert data["section"]["key"] == "val"

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permissions only")
    def test_written_file_has_0600_permissions(self, tmp_path: Path) -> None:
        target = tmp_path / "secrets.yaml"
        _write_yaml_atomic(target, {"x": 1})

        mode = stat.S_IMODE(target.stat().st_mode)
        assert mode == 0o600
