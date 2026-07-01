"""Serialize/deserialize SkillBundle for hub transport.

Converts between SkillLibraryStore records and portable SkillBundle format.
Uses YAML for manifest serialization with JSON fallback if PyYAML is unavailable.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Dict

from leapflow.hub.protocol import SkillBundle, SkillManifest

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Attempt YAML import with graceful fallback
try:
    import yaml

    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False
    logger.debug("PyYAML not available; using JSON fallback for manifest serialization")


# ─── YAML Helpers ────────────────────────────────────────────────────────────


def _manifest_to_dict(manifest: SkillManifest) -> dict:
    """Convert SkillManifest dataclass to a plain dict."""
    return {
        "name": manifest.name,
        "version": manifest.version,
        "description": manifest.description,
        "parameters": manifest.parameters,
        "triggers": manifest.triggers,
        "source_tag": manifest.source_tag,
        "tier": manifest.tier,
        "leapflow_min_version": manifest.leapflow_min_version,
        "created_at": manifest.created_at,
        "author": manifest.author,
        "hub_type": manifest.hub_type,
        "repo_id": manifest.repo_id,
    }


def _dict_to_manifest(data: dict) -> SkillManifest:
    """Reconstruct SkillManifest from a plain dict."""
    return SkillManifest(
        name=data.get("name", ""),
        version=data.get("version", "0.1.0"),
        description=data.get("description", ""),
        parameters=data.get("parameters", []),
        triggers=data.get("triggers", []),
        source_tag=data.get("source_tag", "learned"),
        tier=data.get("tier", 1),
        leapflow_min_version=data.get("leapflow_min_version", "0.1.0"),
        created_at=data.get("created_at", ""),
        author=data.get("author", ""),
        hub_type=data.get("hub_type", ""),
        repo_id=data.get("repo_id", ""),
    )


def _serialize_manifest(manifest: SkillManifest) -> str:
    """Serialize manifest to YAML (or JSON fallback)."""
    data = _manifest_to_dict(manifest)
    if _YAML_AVAILABLE:
        return yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)
    return json.dumps(data, indent=2, ensure_ascii=False)


def _deserialize_manifest(text: str) -> SkillManifest:
    """Deserialize manifest from YAML or JSON."""
    if _YAML_AVAILABLE:
        data = yaml.safe_load(text)
    else:
        # Try JSON first, fallback to basic parsing
        data = json.loads(text)
    return _dict_to_manifest(data if data else {})


# ─── SkillSerializer ─────────────────────────────────────────────────────────


class SkillSerializer:
    """Convert between SkillLibraryStore records and portable SkillBundle format."""

    def export_skill(self, stored_skill: dict) -> SkillBundle:
        """Export a local skill to portable bundle format.

        Args:
            stored_skill: Dict-like object from SkillLibraryStore with fields:
                name, version, description, source_code, parameters,
                triggers, trajectory_skeleton, copilot_prior, etc.

        Returns:
            SkillBundle ready for push to a Hub.
        """
        manifest = SkillManifest(
            name=stored_skill.get("name", ""),
            version=stored_skill.get("version", "0.1.0"),
            description=stored_skill.get("description", ""),
            parameters=stored_skill.get("parameters", []),
            triggers=stored_skill.get("triggers", []),
            source_tag=stored_skill.get("source_tag", "learned"),
            tier=stored_skill.get("tier", 1),
            leapflow_min_version=stored_skill.get("leapflow_min_version", "0.1.0"),
            created_at=stored_skill.get("created_at", ""),
            author=stored_skill.get("author", ""),
        )

        return SkillBundle(
            manifest=manifest,
            source_code=stored_skill.get("source_code", ""),
            trajectory_skeleton=stored_skill.get("trajectory_skeleton", ""),
            copilot_prior=stored_skill.get("copilot_prior", ""),
            readme=stored_skill.get("readme", ""),
        )

    def import_skill(self, bundle: SkillBundle) -> dict:
        """Convert bundle back to fields suitable for SkillLibraryStore.save().

        Returns:
            Dict with all fields needed by the local skill store.
        """
        m = bundle.manifest
        return {
            "name": m.name,
            "version": m.version,
            "description": m.description,
            "parameters": m.parameters,
            "triggers": m.triggers,
            "source_tag": m.source_tag,
            "tier": m.tier,
            "leapflow_min_version": m.leapflow_min_version,
            "created_at": m.created_at,
            "author": m.author,
            "source_code": bundle.source_code,
            "trajectory_skeleton": bundle.trajectory_skeleton,
            "copilot_prior": bundle.copilot_prior,
            "readme": bundle.readme,
            "hub_type": m.hub_type,
            "repo_id": m.repo_id,
        }

    def bundle_to_files(self, bundle: SkillBundle) -> Dict[str, str]:
        """Flatten bundle to file map (for upload_folder).

        Returns:
            Dict mapping filename to content string:
            {"manifest.yaml": ..., "skill.py": ..., "README.md": ...}
        """
        ext = "yaml" if _YAML_AVAILABLE else "json"
        files: Dict[str, str] = {}

        # Manifest
        files[f"manifest.{ext}"] = _serialize_manifest(bundle.manifest)

        # Source code
        if bundle.source_code:
            files["skill.py"] = bundle.source_code

        # Trajectory skeleton
        if bundle.trajectory_skeleton:
            files["trajectory.json"] = bundle.trajectory_skeleton

        # Copilot prior
        if bundle.copilot_prior:
            files["copilot_prior.json"] = bundle.copilot_prior

        # README
        if bundle.readme:
            files["README.md"] = bundle.readme

        return files

    def files_to_bundle(self, files: Dict[str, str | bytes]) -> SkillBundle:
        """Reconstruct bundle from downloaded file map.

        Args:
            files: Dict of filename -> content (str or bytes).

        Returns:
            Reconstructed SkillBundle.
        """
        # Decode bytes if needed
        decoded: Dict[str, str] = {}
        for name, content in files.items():
            if isinstance(content, bytes):
                decoded[name] = content.decode("utf-8")
            else:
                decoded[name] = content

        # Find manifest (try yaml first, then json)
        manifest_text = ""
        for candidate in ("manifest.yaml", "manifest.yml", "manifest.json"):
            if candidate in decoded:
                manifest_text = decoded[candidate]
                break

        if not manifest_text:
            logger.warning("No manifest file found in bundle; using empty manifest")
            manifest = SkillManifest(name="unknown")
        else:
            manifest = _deserialize_manifest(manifest_text)

        return SkillBundle(
            manifest=manifest,
            source_code=decoded.get("skill.py", ""),
            trajectory_skeleton=decoded.get("trajectory.json", ""),
            copilot_prior=decoded.get("copilot_prior.json", ""),
            readme=decoded.get("README.md", ""),
        )
