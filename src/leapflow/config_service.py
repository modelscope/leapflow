"""User-facing configuration control plane for LeapFlow."""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, fields, replace
from getpass import getpass
from pathlib import Path
from types import UnionType
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints

import yaml

from leapflow.config import Settings
from leapflow.security.secrets import ScopedSecretResolver, secret_ref, secret_scope

ConfigScope = Literal["profile", "workspace", "user"]
SecretScope = Literal["profile", "global"]


@dataclass(frozen=True)
class ConfigFieldSpec:
    """Writable config field contract."""

    key: str
    section: str
    name: str
    value_type: Any
    profile_file: str
    setting_name: str | None = None
    scopes: tuple[ConfigScope, ...] = ("profile", "workspace")
    secret: bool = False
    ref_name: str | None = None
    category: str = "Runtime"
    description: str = ""
    value_hint: str = ""
    hot_reload: str = "yes"
    examples: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConfigMutationResult:
    """Result of a config mutation."""

    ok: bool
    message: str
    changed_keys: tuple[str, ...] = ()
    path: Path | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConfigValueView:
    """Rendered config value with source metadata."""

    key: str
    value: str
    source: str = "effective"
    secret: bool = False


@dataclass(frozen=True)
class ConfigFieldView:
    """Human-readable field metadata for config discovery."""

    key: str
    value: str
    value_type: str
    category: str
    scopes: tuple[str, ...]
    hot_reload: str
    secret: bool
    description: str
    value_hint: str = ""
    examples: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConfigSnapshot:
    """A redacted snapshot of user-facing configuration."""

    values: tuple[ConfigValueView, ...]
    sources: tuple[str, ...]
    warnings: tuple[str, ...] = ()


_BOOTSTRAP_ONLY_SETTINGS = frozenset({
    "data_dir",
    "profile",
    "workspace_root",
    "layout",
    "profile_layout",
    "profile_manifest",
    "config_sources",
    "watched_config_paths",
    "config_warnings",
    "duckdb_path",
    "runtime_dir",
    "audit_log_path",
    "skills_dir",
    "visual_frame_cache_dir",
    "video_cache_dir",
    "perceptual_field_config",
})

_SECRET_SETTINGS = frozenset({"llm_api_key", "vlm_api_key", "llm_aux_api_key"})

_FIELD_DESCRIPTIONS = {
    "llm.api_key": "Primary LLM API key stored in the local secret vault.",
    "llm.aux_api_key": "Auxiliary LLM provider API key stored in the local secret vault.",
    "llm.base_url": "OpenAI-compatible endpoint for the primary LLM provider.",
    "llm.model": "Primary LLM model used by chat, planning, and tool reasoning.",
    "llm.context_length": "Runtime context budget in tokens shown in the TUI status bar.",
    "llm.max_retries": "Retry attempts for transient LLM provider failures.",
    "vlm.api_key": "VLM API key stored in the local secret vault.",
    "runtime.mock_host": "Use the in-process mock host when native OS control is unavailable.",
    "runtime.log_level": "Logging verbosity for CLI, TUI, and runtime diagnostics.",
    "memory.working_max_tokens": "Token budget for working memory injected into active reasoning.",
    "visual.track_enabled": "Enable screenshot-based visual perception for the active profile.",
    "recording.mode": "Default recording pipeline used during teaching and observation.",
    "scheduler.tick_seconds": "Scheduler polling interval in seconds.",
    "dashboard.enabled": "Enable the local monitoring web dashboard.",
    "dashboard.bind": "Address the dashboard web server binds to (keep loopback).",
    "dashboard.port": "TCP port for the local dashboard web server.",
    "dashboard.auto_open": "Open the default browser automatically when launching the dashboard.",
    "dashboard.token_ref": "Secret ref for the local dashboard access token.",
    "stream.output": "Stream assistant tokens and progress in interactive sessions.",
    "verbose.progress": "Show inline execution progress for tools and runtime steps.",
}

_SECTION_CATEGORIES = {
    "llm": "LLM Provider",
    "vlm": "Perception",
    "memory": "Memory",
    "visual": "Perception",
    "text": "Perception",
    "clipboard": "Perception",
    "perceptual": "Perception",
    "attention": "Perception",
    "recording": "Recording",
    "video": "Recording",
    "learnability": "Learning",
    "learn": "Learning",
    "skill": "Skills",
    "hub": "Hub",
    "gateway": "Gateway",
    "privacy": "Safety",
    "approval": "Safety",
    "cache": "Storage",
    "runtime": "Runtime",
    "mock": "Runtime",
    "log": "Runtime",
    "scheduler": "Scheduler",
    "dashboard": "Dashboard",
    "monitor": "Dashboard",
    "copilot": "Copilot",
    "react": "Execution Loop",
    "tool": "Execution Loop",
    "context": "Execution Loop",
    "stream": "Interactive UX",
    "verbose": "Interactive UX",
    "signal": "Signal Fusion",
    "surprise": "Signal Fusion",
    "causal": "World Model",
    "prediction": "World Model",
    "curiosity": "World Model",
    "replay": "World Model",
}

_VALUE_HINTS = {
    "runtime.log_level": "DEBUG|INFO|WARNING|ERROR",
    "recording.mode": "video|default|vision_only",
    "signal.channels": "all or comma-separated channel names",
}

_PARTIAL_RELOAD_SECTIONS = frozenset({"runtime", "mock", "gateway", "hub", "scheduler", "observer", "cua", "use", "dashboard"})

_PROFILE_FILE_BY_SECTION = {
    "llm": "llm.yaml",
    "hub": "hub.yaml",
    "gateway": "gateway.yaml",
    "privacy": "privacy.yaml",
    "approval": "approval.yaml",
    "cache": "cache.yaml",
}

_PERCEPTION_SECTIONS = frozenset({
    "visual",
    "vlm",
    "text",
    "clipboard",
    "perceptual",
    "attention",
    "recording",
    "causal",
    "heuristic",
    "prediction",
    "curiosity",
    "replay",
    "semantic",
    "budget",
    "ast",
    "mhms",
    "surprise",
    "video",
    "learnability",
    "signal",
})

_EXPLICIT_SPECS = {
    "llm_api_key": ConfigFieldSpec(
        "llm.api_key",
        "llm",
        "api_key_ref",
        str,
        "llm.yaml",
        setting_name="llm_api_key",
        scopes=("profile",),
        secret=True,
        ref_name="llm/primary/api_key",
    ),
    "mock_host": ConfigFieldSpec(
        "runtime.mock_host",
        "mock",
        "host",
        bool,
        "runtime.yaml",
        setting_name="mock_host",
    ),
    "log_level": ConfigFieldSpec(
        "runtime.log_level",
        "log",
        "level",
        str,
        "runtime.yaml",
        setting_name="log_level",
    ),
}


def _build_field_specs() -> dict[str, ConfigFieldSpec]:
    type_hints = get_type_hints(Settings)
    specs: dict[str, ConfigFieldSpec] = {}
    for item in fields(Settings):
        name = item.name
        if name in _BOOTSTRAP_ONLY_SETTINGS:
            continue
        if name in _EXPLICIT_SPECS:
            spec = _EXPLICIT_SPECS[name]
        else:
            spec = _spec_from_setting(name, type_hints.get(name, item.type))
            if spec is None:
                continue
        spec = _with_metadata(spec)
        specs[spec.key] = spec
    return dict(sorted(specs.items()))


def _spec_from_setting(name: str, value_type: Any) -> ConfigFieldSpec | None:
    parts = name.split("_", 1)
    if len(parts) == 1:
        return None
    section, field_name = parts
    if section == "llm" and field_name == "api_key":
        return None
    if name in _SECRET_SETTINGS:
        return ConfigFieldSpec(
            key=f"{section}.{field_name}",
            section=section,
            name=f"{field_name}_ref",
            value_type=str,
            profile_file=_profile_file_for_section(section),
            setting_name=name,
            scopes=("profile",),
            secret=True,
            ref_name=f"{section}/{field_name}",
        )
    return ConfigFieldSpec(
        key=f"{section}.{field_name}",
        section=section,
        name=field_name,
        value_type=value_type,
        profile_file=_profile_file_for_section(section),
        setting_name=name,
    )


def _profile_file_for_section(section: str) -> str:
    if section in _PROFILE_FILE_BY_SECTION:
        return _PROFILE_FILE_BY_SECTION[section]
    if section in _PERCEPTION_SECTIONS:
        return "perception.yaml"
    return "runtime.yaml"



class ConfigService:
    """Read and mutate LeapFlow configuration through layout-owned paths."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._secrets = ScopedSecretResolver(settings.layout, settings.profile_layout)

    def snapshot(self) -> ConfigSnapshot:
        values = tuple(self.get(key) for key in sorted(_FIELD_SPECS))
        return ConfigSnapshot(
            values=values,
            sources=tuple(self._settings.config_sources),
            warnings=tuple(self._settings.config_warnings),
        )

    def sources(self) -> tuple[str, ...]:
        return tuple(self._settings.config_sources)

    def writable_keys(self) -> tuple[str, ...]:
        return tuple(sorted(_FIELD_SPECS))

    def describe(self, key: str) -> ConfigFieldView:
        """Return metadata and effective value for one writable config field."""
        normalized = _normalize_key(key)
        spec = _FIELD_SPECS.get(normalized)
        if spec is None:
            raise ValueError(f"Unknown config key: {key}")
        return self._field_view(spec)

    def list_fields(self, category: str | None = None) -> tuple[ConfigFieldView, ...]:
        """List writable config fields with human-readable metadata."""
        normalized = category.strip().lower() if category else ""
        fields_list = []
        for spec in _FIELD_SPECS.values():
            if normalized and normalized not in spec.category.lower() and not spec.key.lower().startswith(f"{normalized}."):
                continue
            fields_list.append(self._field_view(spec))
        return tuple(sorted(fields_list, key=lambda item: (item.category, item.key)))

    def get(self, key: str) -> ConfigValueView:
        normalized = _normalize_key(key)
        spec = _FIELD_SPECS.get(normalized)
        if spec is None:
            raise ValueError(f"Unknown config key: {key}")
        if spec.secret:
            value = str(getattr(self._settings, spec.setting_name or "", "") or "")
            return ConfigValueView(normalized, "set" if value.strip() else "missing", secret=True)
        value = getattr(self._settings, spec.setting_name or "", None)
        return ConfigValueView(normalized, _format_value(value), secret=False)

    def set(self, key: str, value: object, *, scope: ConfigScope = "profile") -> ConfigMutationResult:
        normalized = _normalize_key(key)
        spec = _FIELD_SPECS.get(normalized)
        if spec is None:
            raise ValueError(f"Unsupported config key: {key}")
        if scope not in spec.scopes:
            raise ValueError(f"Config key {normalized} does not support scope: {scope}")
        if spec.secret:
            return self._set_secret_backed_field(spec, str(value), scope=scope)
        coerced = _coerce_value(value, spec.value_type)
        path = self._path_for_spec(spec, scope)
        data = _read_yaml(path)
        section = dict(data.get(spec.section) or {})
        section[spec.name] = coerced
        data[spec.section] = section
        _write_yaml_atomic(path, data)
        return ConfigMutationResult(True, f"Updated {normalized}", (normalized,), path)

    def unset(self, key: str, *, scope: ConfigScope = "profile") -> ConfigMutationResult:
        normalized = _normalize_key(key)
        spec = _FIELD_SPECS.get(normalized)
        if spec is None:
            raise ValueError(f"Unsupported config key: {key}")
        if scope not in spec.scopes:
            raise ValueError(f"Config key {normalized} does not support scope: {scope}")
        path = self._path_for_spec(spec, scope)
        data = _read_yaml(path)
        section = dict(data.get(spec.section) or {})
        section.pop(spec.name, None)
        data[spec.section] = section
        _write_yaml_atomic(path, data)
        return ConfigMutationResult(True, f"Unset {normalized}", (normalized,), path)

    def configure_llm(
        self,
        *,
        api_key: str | None = None,
        ask_api_key: bool = False,
        base_url: str | None = None,
        model: str | None = None,
        context_length: int | None = None,
        max_retries: int | None = None,
        scope: ConfigScope = "profile",
    ) -> ConfigMutationResult:
        changed: list[str] = []
        path: Path | None = None
        if ask_api_key:
            api_key = getpass("LLM API key: ")
        if api_key is not None:
            result = self.set("llm.api_key", api_key, scope=scope)
            changed.extend(result.changed_keys)
            path = result.path
        for key, item in (
            ("llm.base_url", base_url.rstrip("/") if base_url is not None else None),
            ("llm.model", model),
            ("llm.context_length", context_length),
            ("llm.max_retries", max_retries),
        ):
            if item is None:
                continue
            result = self.set(key, item, scope=scope)
            changed.extend(result.changed_keys)
            path = result.path
        if not changed:
            return ConfigMutationResult(False, "No LLM config changes requested", (), path)
        return ConfigMutationResult(True, "LLM config updated", tuple(dict.fromkeys(changed)), path)

    def list_secrets(self) -> tuple[str, ...]:
        return self._secrets.list_refs()

    def set_secret(
        self,
        ref: str,
        value: str | None = None,
        *,
        scope: SecretScope = "profile",
        reveal_name: bool = True,
    ) -> ConfigMutationResult:
        normalized = normalize_secret_ref(ref, default_scope=scope)
        secret_value = value if value is not None else getpass(f"Value for {normalized}: ")
        self._secrets.set(normalized, secret_value, metadata={"source": "config"})
        name = normalized if reveal_name else "secret"
        return ConfigMutationResult(True, f"Saved {name}", (normalized,))

    def get_secret(self, ref: str, *, scope: SecretScope = "profile", reveal: bool = False) -> str:
        normalized = normalize_secret_ref(ref, default_scope=scope)
        value = self._secrets.get(normalized)
        if value is None:
            raise KeyError(normalized)
        return value if reveal else f"{normalized} is set"

    def delete_secret(self, ref: str, *, scope: SecretScope = "profile") -> ConfigMutationResult:
        normalized = normalize_secret_ref(ref, default_scope=scope)
        self._secrets.delete(normalized)
        return ConfigMutationResult(True, f"Deleted {normalized}", (normalized,))

    def _set_secret_backed_field(
        self,
        spec: ConfigFieldSpec,
        value: str,
        *,
        scope: ConfigScope,
    ) -> ConfigMutationResult:
        secret_path = spec.ref_name or spec.key.replace(".", "/")
        ref = secret_ref("profile", *[part for part in secret_path.split("/") if part])
        self._secrets.set(ref, value, metadata={"source": "config", "owner": spec.section})
        path = self._path_for_spec(spec, scope)
        data = _read_yaml(path)
        section = dict(data.get(spec.section) or {})
        section[spec.name] = ref
        data[spec.section] = section
        _write_yaml_atomic(path, data)
        return ConfigMutationResult(True, f"Updated {spec.key}", (spec.key, f"{spec.key}_ref"), path)

    def _path_for_spec(self, spec: ConfigFieldSpec, scope: ConfigScope) -> Path:
        if scope == "profile":
            return self._settings.profile_layout.config_dir / spec.profile_file
        if scope == "workspace":
            return self._settings.layout.workspace_config_path(self._settings.workspace_root)
        if scope == "user":
            return self._settings.layout.user_config_path
        raise ValueError(f"Unsupported config scope: {scope}")

    def _field_view(self, spec: ConfigFieldSpec) -> ConfigFieldView:
        value = self.get(spec.key)
        return ConfigFieldView(
            key=spec.key,
            value=value.value,
            value_type="secret" if spec.secret else _type_label(spec.value_type),
            category=spec.category,
            scopes=tuple(spec.scopes),
            hot_reload=spec.hot_reload,
            secret=spec.secret,
            description=spec.description,
            value_hint=spec.value_hint,
            examples=spec.examples,
        )


def _with_metadata(spec: ConfigFieldSpec) -> ConfigFieldSpec:
    return replace(
        spec,
        category=_category_for_spec(spec),
        description=_FIELD_DESCRIPTIONS.get(spec.key, _default_description(spec.key)),
        value_hint=_VALUE_HINTS.get(spec.key, _default_value_hint(spec)),
        hot_reload="partial" if spec.section in _PARTIAL_RELOAD_SECTIONS else "yes",
        examples=_examples_for_key(spec.key),
    )


def _category_for_spec(spec: ConfigFieldSpec) -> str:
    if spec.key.startswith("runtime."):
        return "Runtime"
    return _SECTION_CATEGORIES.get(spec.section, _title_words(spec.section))


def _default_description(key: str) -> str:
    words = key.replace(".", " ").replace("_", " ")
    return f"Configure {words}."


def _default_value_hint(spec: ConfigFieldSpec) -> str:
    value_type = _unwrap_optional(spec.value_type)
    origin = get_origin(value_type)
    if spec.secret:
        return "secret value"
    if value_type is bool:
        return "true|false"
    if value_type in (tuple, list, frozenset) or origin in (tuple, list, frozenset):
        return "comma-separated values"
    if value_type is dict or origin is dict:
        return "YAML/JSON mapping"
    return ""


def _examples_for_key(key: str) -> tuple[str, ...]:
    examples = {
        "llm.model": ("leap config set llm.model qwen3.7-plus",),
        "llm.context_length": ("leap config set llm.context_length 1000000",),
        "runtime.mock_host": ("leap config set runtime.mock_host true",),
        "runtime.log_level": ("leap config set runtime.log_level DEBUG",),
        "memory.working_max_tokens": ("leap config set memory.working_max_tokens 12000",),
        "visual.track_enabled": ("leap config set visual.track_enabled true",),
    }
    return examples.get(key, ())


def _title_words(value: str) -> str:
    return " ".join(part.capitalize() for part in value.replace("_", " ").split())


def _type_label(value_type: Any) -> str:
    normalized = _unwrap_optional(value_type)
    origin = get_origin(normalized)
    if normalized in (str, int, float, bool):
        return normalized.__name__
    if normalized is Path:
        return "path"
    if normalized is dict or origin is dict:
        return "dict"
    if normalized in (tuple, list, frozenset) or origin in (tuple, list, frozenset):
        return "list"
    if origin is Literal:
        return "|".join(str(item) for item in get_args(normalized))
    return getattr(normalized, "__name__", str(normalized).replace("typing.", ""))


def _unwrap_optional(value_type: Any) -> Any:
    origin = get_origin(value_type)
    if origin in (UnionType, Union):
        args = tuple(arg for arg in get_args(value_type) if arg is not type(None))
        if len(args) == 1:
            return args[0]
    return value_type


_FIELD_SPECS: dict[str, ConfigFieldSpec] = _build_field_specs()


def normalize_secret_ref(raw: str, *, default_scope: SecretScope = "profile") -> str:
    """Normalize a CLI/TUI secret shorthand into a secret:// ref."""
    value = raw.strip()
    if value.startswith("secret://"):
        secret_scope(value)
        return value
    return secret_ref(default_scope, *_secret_name_parts(value))


def _secret_name_parts(value: str) -> list[str]:
    if "/" in value:
        parts = [part for part in value.split("/") if part]
    else:
        parts = [part for part in value.split(".") if part]
    if not parts:
        raise ValueError("Secret ref cannot be empty")
    return parts


def _normalize_key(key: str) -> str:
    value = key.strip()
    if value.startswith("LEAPFLOW_"):
        parts = value[len("LEAPFLOW_"):].lower().split("_", 1)
        return parts[0] if len(parts) == 1 else f"{parts[0]}.{parts[1]}"
    if "_" in value and "." not in value:
        parts = value.lower().split("_", 1)
        return f"{parts[0]}.{parts[1]}"
    return value


def _coerce_value(value: object, value_type: Any) -> object:
    value_type = _unwrap_optional(value_type)
    origin = get_origin(value_type)
    if origin in (tuple, list, frozenset) or value_type in (tuple, list, frozenset):
        if isinstance(value, (list, tuple, set, frozenset)):
            return list(value)
        return [part.strip() for part in str(value).split(",") if part.strip()]
    if origin is dict or value_type is dict:
        parsed = yaml.safe_load(str(value))
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected mapping value, got: {value}")
        return parsed
    if value_type is bool:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"Expected boolean value, got: {value}")
    if value_type is int:
        return int(str(value).strip())
    if value_type is float:
        return float(str(value).strip())
    if value_type is Path:
        return str(value).strip()
    if value_type is str:
        return str(value).strip()
    text = str(value).strip()
    if text.startswith("{") or text.startswith("["):
        return yaml.safe_load(text)
    return text


def _format_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list, set, frozenset)):
        return ",".join(str(item) for item in value)
    if isinstance(value, dict):
        return yaml.safe_dump(value, sort_keys=True, default_flow_style=True).strip()
    return "" if value is None else str(value)


def _read_yaml(path: Path) -> dict[str, Any]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError, ValueError):
        return {"version": 1}
    if not isinstance(loaded, dict):
        return {"version": 1}
    loaded.setdefault("version", 1)
    return dict(loaded)


def _write_yaml_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, sort_keys=False, allow_unicode=True)
        os.replace(tmp_path, path)
        if os.name != "nt":
            path.chmod(0o600)
    except Exception:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise
