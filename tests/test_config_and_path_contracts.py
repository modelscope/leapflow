# Copyright (c) Alibaba, Inc. and its affiliates.
"""End-to-end guards for the config control plane and the path tree contract.

AGENTS.md devotes a whole section to these, but coverage was scattered: each
feature asserted its own key in its own test file (``agent.reentry_enabled`` in
one, ``daemon.log_level`` in another), so a new field could ship without a
description, without hot-reload semantics, or writing outside the profile, and
no test would notice. These tests assert the contract over *every* field and
*every* managed path instead of one at a time.

Covered contracts:
- `leap config` is the user-facing control plane (discoverable + mutable)
- Config catalog is the discovery contract (complete per-field metadata)
- Secrets are refs, never durable plaintext
- Path tree is a product contract (layout-owned, profile-scoped)
- Graceful degradation (optional components may be absent)
"""

from __future__ import annotations

import dataclasses
import pathlib

import pytest

from leapflow.config_service import (
    _FIELD_DESCRIPTIONS,
    ConfigService,
    _build_field_specs,
)
from leapflow.layout import build_layout

_VALID_HOT_RELOAD = frozenset({"yes", "partial", "restart-required"})
_VALID_SCOPES = frozenset({"profile", "workspace", "global"})


@pytest.fixture()
def layout(tmp_path: pathlib.Path):
    """A real layout rooted in a temp dir (never the user's home)."""
    return build_layout(tmp_path / "leap-home")


# ── Config catalog is the discovery contract ─────────────────────────────


def test_catalog_is_not_empty() -> None:
    """A collapsed catalog would make every assertion below vacuous."""
    assert len(_build_field_specs()) > 100


def test_every_writable_field_declares_valid_discovery_metadata() -> None:
    """Catalog metadata must be well-formed for every field.

    This guards hand-written spec mistakes on newly added settings (an invalid
    ``hot_reload`` string, an unknown scope, a key that is not dot.separated).
    Fields with generated fallbacks are deliberately not asserted non-empty:
    ``description`` falls back to "Configure <key>." and ``value_hint`` is empty
    for plain numeric settings, so such checks would pass unconditionally. The
    parts that actually need prose are covered by the two tests below.
    """
    problems: list[str] = []
    for name, spec in _build_field_specs().items():
        if not spec.key or "." not in spec.key:
            problems.append(f"{name}: key must be dot.separated, got {spec.key!r}")
        if spec.value_type is None:
            problems.append(f"{name}: missing value_type")
        if not spec.scopes:
            problems.append(f"{name}: must declare at least one scope")
        if not _VALID_SCOPES.issuperset(spec.scopes):
            problems.append(f"{name}: unknown scope in {spec.scopes}")
        if spec.hot_reload not in _VALID_HOT_RELOAD:
            problems.append(f"{name}: hot_reload={spec.hot_reload!r} not in {sorted(_VALID_HOT_RELOAD)}")
        if not spec.category:
            problems.append(f"{name}: missing category")

    assert problems == [], "config catalog contract violations:\n  " + "\n  ".join(problems)


def test_restart_required_fields_explain_the_restart() -> None:
    """A setting that needs a restart must say so in hand-written prose.

    The generated fallback ("Configure daemon request ledger ttl s.") never
    mentions the restart, so a user edits the value, sees no effect, and has no
    way to find out why.
    """
    problems: list[str] = []
    for spec in _build_field_specs().values():
        if spec.hot_reload != "restart-required":
            continue
        description = _FIELD_DESCRIPTIONS.get(spec.key, "")
        if not description:
            problems.append(f"{spec.key}: relies on the generated description")
        elif "restart" not in description.lower():
            problems.append(f"{spec.key}: description does not mention the restart")

    assert problems == [], "restart-required fields must be explained:\n  " + "\n  ".join(problems)


def test_secret_fields_have_hand_written_descriptions() -> None:
    """Credential fields must explain provenance, not just restate the key."""
    missing = [
        spec.key
        for spec in _build_field_specs().values()
        if spec.secret and not _FIELD_DESCRIPTIONS.get(spec.key)
    ]

    assert missing == [], f"secret fields need an explicit description: {missing}"


def test_catalog_keys_are_unique() -> None:
    """Two specs sharing a key would make `leap config set` ambiguous."""
    keys = [spec.key for spec in _build_field_specs().values()]

    duplicates = {key for key in keys if keys.count(key) > 1}
    assert duplicates == set(), f"duplicate config keys: {sorted(duplicates)}"


def test_secret_fields_declare_a_vault_ref_and_are_profile_scoped() -> None:
    """Credentials must resolve to a vault ref, never to a durable value.

    Also profile-scoped: a workspace-writable credential would leak into a
    shared repo.
    """
    problems: list[str] = []
    for name, spec in _build_field_specs().items():
        if not spec.secret:
            continue
        if not spec.ref_name:
            problems.append(f"{name}: secret field must declare ref_name")
        if tuple(spec.scopes) != ("profile",):
            problems.append(f"{name}: secret must be profile-scoped, got {spec.scopes}")

    assert problems == [], "secret field contract violations:\n  " + "\n  ".join(problems)


def test_writable_keys_matches_the_catalog(monkeypatch, tmp_path) -> None:
    """``leap config keys`` and ``leap config list`` must not drift apart."""
    monkeypatch.setenv("LEAPFLOW_HOME", str(tmp_path / "home"))
    from leapflow.config import get_settings

    service = ConfigService(get_settings())

    assert set(service.writable_keys()) == {s.key for s in _build_field_specs().values()}


def test_describe_exposes_every_field_the_contract_requires(monkeypatch, tmp_path) -> None:
    """``leap config show <key>`` must render the full discovery contract.

    AGENTS.md enumerates exactly what a writable field has to expose; assert the
    view carries all of it so a trimmed-down renderer cannot pass.
    """
    monkeypatch.setenv("LEAPFLOW_HOME", str(tmp_path / "home"))
    from leapflow.config import get_settings

    view = ConfigService(get_settings()).describe("runtime.log_level")

    required = {
        "key", "value", "value_type", "scopes",
        "hot_reload", "category", "value_hint", "description",
    }
    present = {field.name for field in dataclasses.fields(view)}
    assert required <= present, f"missing from the field view: {sorted(required - present)}"

    assert view.key == "runtime.log_level"
    assert view.hot_reload in _VALID_HOT_RELOAD
    assert view.description
    # The hint is what makes an enum-like value self-discoverable in the TUI.
    assert view.value_hint


def test_signal_noise_config_fields_are_discoverable(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LEAPFLOW_HOME", str(tmp_path / "home"))
    from leapflow.config import get_settings

    service = ConfigService(get_settings())

    for key in (
        "signal.noise_gate_enabled",
        "signal.noise_same_source_cooldown_s",
        "signal.noise_allow_fs_outside_workspace",
        "signal.noise_path_fragments",
        "signal.noise_dir_names",
        "signal.noise_suffixes",
    ):
        view = service.describe(key)
        assert view.category == "Signal Fusion"
        assert view.description
        assert view.hot_reload in _VALID_HOT_RELOAD


def test_unknown_key_is_rejected_rather_than_silently_accepted(monkeypatch, tmp_path) -> None:
    """Both read and write reject an unknown key loudly.

    Silently accepting a typo'd key would strand the user's setting in a file
    nothing ever reads.
    """
    monkeypatch.setenv("LEAPFLOW_HOME", str(tmp_path / "home"))
    from leapflow.config import get_settings

    service = ConfigService(get_settings())

    with pytest.raises(ValueError, match="Unknown config key"):
        service.describe("definitely.not.a.real.key")
    with pytest.raises(ValueError, match="Unsupported config key"):
        service.set("definitely.not.a.real.key", "x")


# ── Path tree is a product contract ──────────────────────────────────────


def test_managed_paths_stay_under_the_layout_root(layout) -> None:
    """Every layout-declared path must live under the layout root.

    Guards against an ad-hoc join escaping into ``~`` or the CWD.
    """
    root = layout.root.resolve()
    candidates = {
        "user_config_path": layout.user_config_path,
        "mcp_servers_path": layout.mcp_servers_path,
        "policy_config_path": layout.policy_config_path,
        "profiles_dir": layout.profiles_dir,
        "logs_dir": layout.logs_dir,
        "defaults_lock_path": layout.defaults_lock_path,
    }

    for name, path in candidates.items():
        assert pathlib.Path(path).resolve().is_relative_to(root), (
            f"{name} ({path}) escapes the layout root {root}"
        )


def test_profile_owns_its_state_and_never_the_workspace(layout, tmp_path) -> None:
    """Profile data must not be written into the user's workspace."""
    profile = layout.ensure(profile_id="default")
    workspace = tmp_path / "some-workspace"
    workspace.mkdir()

    profile_root = pathlib.Path(profile.config_dir).resolve().parent
    owned = [
        profile.duckdb_path, profile.conversation_db_path, profile.audit_log_path,
        profile.memory_dir, profile.config_dir, profile.gateway_config_path,
        profile.llm_config_path, profile.approval_config_path,
    ]

    for path in owned:
        resolved = pathlib.Path(path).resolve()
        assert resolved.is_relative_to(profile_root), f"{path} is not profile-owned"
        assert not resolved.is_relative_to(workspace.resolve()), (
            f"{path} leaked into the workspace"
        )


def test_workspace_footprint_is_limited_to_declared_files(layout, tmp_path) -> None:
    """Workspace-local files are limited to config + manifest, in .leapflow/."""
    workspace = tmp_path / "ws"
    workspace.mkdir()

    config_path = pathlib.Path(layout.workspace_config_path(workspace)).resolve()
    manifest_path = pathlib.Path(layout.workspace_manifest_path(workspace)).resolve()

    for path in (config_path, manifest_path):
        assert path.is_relative_to(workspace.resolve())
        assert path.parent.name == ".leapflow", f"{path} must live in .leapflow/"
    assert {config_path.name, manifest_path.name} == {"config.yaml", "workspace.yaml"}


def test_profiles_are_mutually_isolated(layout) -> None:
    """Two profiles must not share DBs, memory, secrets, or audit logs."""
    first = layout.ensure(profile_id="default")
    second = layout.ensure(profile_id="work")

    pairs = [
        (first.duckdb_path, second.duckdb_path),
        (first.memory_dir, second.memory_dir),
        (first.audit_log_path, second.audit_log_path),
        (first.config_dir, second.config_dir),
        (first.secrets.vault_path, second.secrets.vault_path),
    ]
    for left, right in pairs:
        assert pathlib.Path(left).resolve() != pathlib.Path(right).resolve(), (
            f"profiles share {left}"
        )


def test_no_legacy_run_or_flat_cache_paths(layout) -> None:
    """Retired layouts must not come back (``run/``, flat cache roots)."""
    profile = layout.ensure(profile_id="default")
    described = {
        "duckdb": str(profile.duckdb_path),
        "audit": str(profile.audit_log_path),
        "cache": str(profile.cache.profile_dir),
    }

    for name, path in described.items():
        assert "/run/" not in path, f"{name} uses the retired run/ path: {path}"
    # Cache must be scoped under the profile, not a shared flat root.
    assert pathlib.Path(profile.cache.profile_dir).resolve().is_relative_to(
        pathlib.Path(profile.config_dir).resolve().parent
    )


def test_secret_vault_is_profile_scoped_and_separate_from_config(layout) -> None:
    """Vault files must not sit inside the readable config directory."""
    profile = layout.ensure(profile_id="default")

    vault = pathlib.Path(profile.secrets.vault_path).resolve()
    key = pathlib.Path(profile.secrets.key_path).resolve()
    config_dir = pathlib.Path(profile.config_dir).resolve()

    assert not vault.is_relative_to(config_dir), "vault must not live in config/"
    assert not key.is_relative_to(config_dir), "vault key must not live in config/"
    assert vault != key, "key material must be separate from the vault payload"


# ── Graceful degradation ─────────────────────────────────────────────────


def test_config_service_works_without_any_existing_config(monkeypatch, tmp_path) -> None:
    """First run: no files on disk yet, and the control plane still answers."""
    monkeypatch.setenv("LEAPFLOW_HOME", str(tmp_path / "pristine"))
    from leapflow.config import get_settings

    service = ConfigService(get_settings())

    assert service.writable_keys(), "catalog must be available before any file exists"
    assert service.describe("runtime.log_level") is not None


def test_dashboard_view_builds_with_no_watches_or_findings() -> None:
    """A dashboard with an empty backend must render, not raise.

    This is the degraded path a fresh board hits before the first analysis.
    """
    import asyncio

    from leapflow.dashboard.intent import DashboardIntent
    from leapflow.dashboard.service import DashboardViewBuilder

    class _EmptyProvider:
        async def watches(self) -> list[dict]:
            return []

        async def findings(self, *, watch_id: str = "", limit: int = 50) -> list[dict]:
            return []

    spec = asyncio.run(
        DashboardViewBuilder().build(DashboardIntent.from_params({}), _EmptyProvider())
    )

    assert isinstance(spec, dict)
    assert "root" in spec, "an empty board must still produce a renderable ViewSpec"


def test_field_spec_is_immutable() -> None:
    """The catalog is shared read-only state; a mutable spec invites drift."""
    spec = next(iter(_build_field_specs().values()))

    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.description = "rewritten"  # type: ignore[misc]
