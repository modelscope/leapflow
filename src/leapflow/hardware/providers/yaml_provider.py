"""Declaration-file provider: reads hardware contexts from YAML on disk.

The default and, before an upstream standard is available, the only source of
device knowledge. Declarations are durable user assets under the profile: they
encode physical operating limits a person is accountable for, so this provider
never writes them and never rewrites them.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

import yaml

from leapflow.hardware.context import ContextSource, HardwareContext
from leapflow.hardware.providers import ProviderError

logger = logging.getLogger(__name__)

_DECLARATION_SUFFIXES = (".yaml", ".yml")


class YamlContextProvider:
    """Supplies hardware contexts parsed from a declarations directory."""

    kind = "yaml"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        config = config or {}
        root = config.get("devices_dir") or config.get("root")
        if not root:
            raise ProviderError(
                "yaml provider requires config.devices_dir",
                failure_code="devices_dir_missing",
            )
        self._root = Path(str(root)).expanduser()
        self._verified: dict[str, str] = _load_verified(config.get("verified_path"))

    def discover(self) -> tuple[HardwareContext, ...]:
        """Return every parseable declaration under the configured directory.

        Never connects to a device: discovery must work while the hardware is
        powered off, and must not block boot on device I/O.

        A malformed file is skipped with a logged reason rather than aborting the
        scan, so one bad declaration cannot make every other device disappear.
        """
        if not self._root.is_dir():
            return ()
        contexts: list[HardwareContext] = []
        for path in sorted(self._root.iterdir()):
            if path.suffix.lower() not in _DECLARATION_SUFFIXES or not path.is_file():
                continue
            payload = _read_yaml(path)
            if payload is None:
                continue
            context = HardwareContext.from_mapping(payload)
            contexts.append(self._apply_verification(context))
        return tuple(contexts)

    def _apply_verification(self, context: HardwareContext) -> HardwareContext:
        """Attach an out-of-band human confirmation, if one exists.

        Verification is stored separately from the declaration on purpose: the
        person who confirms a context must not have to edit the file they are
        confirming, or the confirmation would be self-attested.

        ``replace`` rather than a field-by-field rebuild: the hand-written copy
        silently dropped every field added to ``HardwareContext`` after it, which
        is a defect that only shows up as a missing value on a board.
        """
        verifier = self._verified.get(context.device_id, "")
        if not verifier or context.provenance.is_verified:
            return context
        from dataclasses import replace

        return replace(
            context,
            provenance=replace(context.provenance, verified_by=verifier),
        )


def _read_yaml(path: Path) -> dict[str, Any] | None:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("Skipping unreadable hardware declaration %s: %s", path, exc)
        return None
    if not isinstance(raw, Mapping):
        logger.warning("Skipping hardware declaration %s: top level is not a mapping", path)
        return None
    payload = dict(raw)
    payload.setdefault("device_id", path.stem)
    payload.setdefault("provenance", {}).setdefault("source", ContextSource.DECLARED.value)
    return payload


def _load_verified(path: Any) -> dict[str, str]:
    """Read ``{device_id: verifier}`` confirmations, tolerating absence."""
    if not path:
        return {}
    target = Path(str(path)).expanduser()
    if not target.is_file():
        return {}
    try:
        raw = yaml.safe_load(target.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("Ignoring unreadable hardware verification file %s: %s", target, exc)
        return {}
    if not isinstance(raw, Mapping):
        return {}
    return {str(key): str(value) for key, value in raw.items() if value}


def build_provider(config: Mapping[str, Any] | None = None) -> YamlContextProvider:
    """Factory registered in the provider table."""
    return YamlContextProvider(config)


__all__ = ["YamlContextProvider", "build_provider"]
