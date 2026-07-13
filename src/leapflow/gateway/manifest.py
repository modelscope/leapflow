"""Platform manifest loading and discovery.

A manifest is a YAML file that declares:

- What credentials a platform requires (for conversational config)
- Setup instructions for the user (multilingual)
- Adapter module / class to instantiate
- Validation method to call

Manifests are discovered from a prioritised list of directories
(built-in → user profile → pip packages) so that new platforms can
be added without touching core code.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Manifest domain types (all frozen)
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CredentialField:
    """A single credential field declared by a platform."""

    key: str
    label: str
    required: bool = True
    secret: bool = False
    help_zh: str = ""
    help_en: str = ""


@dataclass(frozen=True)
class OptionField:
    """An optional configuration field with a default value."""

    key: str
    label: str
    field_type: str = "string"
    choices: tuple = ()
    default: Any = None
    required: bool = False
    advanced: bool = False
    depends_on: Dict[str, Any] = field(default_factory=dict)
    help_zh: str = ""
    help_en: str = ""


@dataclass(frozen=True)
class SetupGuide:
    """User-facing setup instructions (bilingual)."""

    summary_zh: str = ""
    summary_en: str = ""
    steps_zh: tuple = ()
    steps_en: tuple = ()
    console_url: str = ""


@dataclass(frozen=True)
class AdapterSpec:
    """How to dynamically instantiate the adapter."""

    module: str
    class_name: str
    dependencies: tuple = ()


@dataclass(frozen=True)
class PlatformManifest:
    """Complete declaration of a platform's integration requirements."""

    platform_id: str
    display_name: str
    description: str = ""
    category: str = "im"
    credentials: tuple = ()
    options: tuple = ()
    setup_guide: SetupGuide = field(default_factory=SetupGuide)
    validation_method: str = ""
    validation_timeout_s: float = 10.0
    adapter: Optional[AdapterSpec] = None
    backend: Dict[str, Any] = field(default_factory=dict)
    actions: Dict[str, Any] = field(default_factory=dict)
    extra_fields: Dict[str, Any] = field(default_factory=dict)
    source_path: str = ""


# ═══════════════════════════════════════════════════════════════
# Manifest loader
# ═══════════════════════════════════════════════════════════════

class ManifestLoader:
    """Discovers and loads platform manifests from YAML files.

    Search order (later wins on ID collision):
    1. Built-in manifests shipped with leapflow
    2. Extra directories (user profile, pip packages)
    """

    def __init__(self, extra_dirs: Optional[List[Path]] = None) -> None:
        self._search_paths: List[Path] = [
            Path(__file__).parent / "manifests",
        ]
        if extra_dirs:
            self._search_paths.extend(extra_dirs)

    def discover(self) -> Dict[str, PlatformManifest]:
        """Scan all search paths and return discovered manifests."""
        manifests: Dict[str, PlatformManifest] = {}
        for search_path in self._search_paths:
            if not search_path.is_dir():
                continue
            for yaml_file in sorted(search_path.glob("*.yaml")):
                try:
                    manifest = self._parse(yaml_file)
                    manifests[manifest.platform_id] = manifest
                except Exception:
                    logger.warning(
                        "Failed to parse manifest: %s", yaml_file, exc_info=True,
                    )
        return manifests

    # ── Internal ─────────────────────────────────────────────

    def _parse(self, path: Path) -> PlatformManifest:
        """Parse a single YAML file into a ``PlatformManifest``."""
        import yaml  # lazy — not needed if no manifests exist

        raw = yaml.safe_load(path.read_text(encoding="utf-8"))

        credentials = tuple(
            CredentialField(
                key=c["key"],
                label=c["label"],
                required=c.get("required", True),
                secret=c.get("secret", False),
                help_zh=c.get("help_zh", ""),
                help_en=c.get("help_en", ""),
            )
            for c in raw.get("credentials", [])
        )

        options = tuple(
            OptionField(
                key=o["key"],
                label=o["label"],
                field_type=o.get("type", "string"),
                choices=tuple(o.get("choices", [])),
                default=o.get("default"),
                required=o.get("required", False),
                advanced=o.get("advanced", False),
                depends_on=o.get("depends_on", {}),
                help_zh=o.get("help_zh", ""),
                help_en=o.get("help_en", ""),
            )
            for o in raw.get("options", [])
        )

        guide_raw = raw.get("setup_guide", {})
        setup_guide = SetupGuide(
            summary_zh=guide_raw.get("summary_zh", ""),
            summary_en=guide_raw.get("summary_en", ""),
            steps_zh=tuple(guide_raw.get("steps_zh", [])),
            steps_en=tuple(guide_raw.get("steps_en", [])),
            console_url=guide_raw.get("console_url", ""),
        )

        adapter_raw = raw.get("adapter")
        adapter = None
        if adapter_raw:
            adapter = AdapterSpec(
                module=adapter_raw["module"],
                class_name=adapter_raw["class"],
                dependencies=tuple(adapter_raw.get("dependencies", [])),
            )

        validation_raw = raw.get("validation", {})
        known_keys = {
            "platform_id", "display_name", "description", "category",
            "credentials", "options", "setup_guide", "validation",
            "adapter", "backend", "actions",
        }
        return PlatformManifest(
            platform_id=raw["platform_id"],
            display_name=raw.get("display_name", raw["platform_id"]),
            description=raw.get("description", ""),
            category=raw.get("category", "im"),
            credentials=credentials,
            options=options,
            setup_guide=setup_guide,
            validation_method=validation_raw.get("method", ""),
            validation_timeout_s=float(validation_raw.get("timeout_s", 10)),
            adapter=adapter,
            backend=dict(raw.get("backend") or {}),
            actions=dict(raw.get("actions") or {}),
            extra_fields={key: value for key, value in raw.items() if key not in known_keys},
            source_path=str(path),
        )
