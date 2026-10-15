# Copyright (c) Alibaba, Inc. and its affiliates.
"""Declaration-file provider: reads hardware contexts from YAML on disk.

The default and, before an upstream standard is available, the only source of
device knowledge. Declarations are durable user assets under the profile: they
encode physical operating limits a person is accountable for, so this provider
never writes them and never rewrites them.
"""

from __future__ import annotations

import hashlib
import json
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
            context = self._apply_verification(context)
            context = _apply_control_bindings(context)
            contexts.append(context)
        return tuple(contexts)

    def _apply_verification(self, context: HardwareContext) -> HardwareContext:
        """Attach an out-of-band human confirmation, if one exists.

        Verification is stored separately from the declaration on purpose: the
        person who confirms a context must not have to edit the file they are
        confirming, or the confirmation would be self-attested.

        New format (digest-bound): ``{device_id: {verifier, declaration_digest, verified_at}}``.
        Old format (legacy): ``{device_id: verifier_string}``.

        Digest-bound entries whose digest does not match the current declaration
        are treated as unbound: the context is demoted (V7) with a warning.
        Old string entries are treated as unbound (no digest) with a warning.
        """
        entry = self._verified.get(context.device_id)
        if entry is None or context.provenance.is_verified:
            return context

        from dataclasses import replace

        if isinstance(entry, str):
            # Legacy format: plain verifier string, no digest binding
            if not entry:
                return context
            logger.warning(
                "verified.json: device %r uses legacy string format (no digest binding); "
                "run 'leap hw verify %s' to create a digest-bound entry",
                context.device_id, context.device_id,
            )
            return replace(
                context,
                provenance=replace(context.provenance, verified_by=entry),
            )

        if isinstance(entry, Mapping):
            verifier = str(entry.get("verifier") or "")
            if not verifier:
                return context
            stored_digest = str(entry.get("declaration_digest") or "")
            if stored_digest:
                current_digest = _declaration_digest(context)
                if current_digest != stored_digest:
                    logger.warning(
                        "verified.json: device %r declaration digest mismatch "
                        "(stored=%s, current=%s); verification demoted",
                        context.device_id, stored_digest[:12], current_digest[:12],
                    )
                    return context  # demoted: declaration changed since verification
            return replace(
                context,
                provenance=replace(context.provenance, verified_by=verifier),
            )

        return context


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


def _load_verified(path: Any) -> dict[str, Any]:
    """Read verification confirmations, tolerating absence.

    Supports two formats:
    - New (digest-bound): ``{device_id: {verifier, declaration_digest, verified_at}}``
    - Legacy: ``{device_id: verifier_string}``

    Returns a dict of ``{device_id: entry}`` where entry is either a string
    (legacy) or a dict (new format).
    """
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
    result: dict[str, Any] = {}
    for key, value in raw.items():
        if not value:
            continue
        if isinstance(value, str):
            result[str(key)] = value  # legacy format
        elif isinstance(value, Mapping):
            result[str(key)] = dict(value)  # new digest-bound format
        else:
            result[str(key)] = str(value)
    return result


def _declaration_digest(context: HardwareContext) -> str:
    """Compute a sha256 digest of the context's wire form for verification binding.

    Uses the same canonical form as the LHP server's _declarations_digest,
    ensuring consistency between local and remote verification.
    """
    payload = context.to_dict()
    # Remove transport config (node-private) and control_bindings
    if "transport" in payload:
        payload["transport"] = {"kind": payload.get("transport", {}).get("kind", ""), "config": {}}
    payload.pop("control_bindings", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def declaration_digest(context: HardwareContext) -> str:
    """Public API for computing a context's declaration digest."""
    return _declaration_digest(context)


def build_provider(config: Mapping[str, Any] | None = None) -> YamlContextProvider:
    """Factory registered in the provider table."""
    return YamlContextProvider(config)


def _apply_control_bindings(context: HardwareContext) -> HardwareContext:
    """Resolve control_bindings into a TransportRef when the YAML declares them.

    Rules:
    - If no ``control_bindings`` section, return *context* unchanged.
    - If the YAML already declares an explicit ``transport.kind``, the bindings
      are kept as supplementary metadata but do **not** override the transport.
    - Otherwise the resolver picks the best available binding and sets it as
      the context's transport.
    """
    if not context.control_bindings:
        return context

    # Explicit transport already declared — nothing to override.
    if context.transport.kind:
        logger.debug(
            "control_bindings: device %s already has transport.kind=%r; "
            "bindings kept as metadata only",
            context.device_id,
            context.transport.kind,
        )
        return context

    # Function-local import to avoid a module-level circular dependency.
    from leapflow.hardware.control_binding_resolver import ControlBindingResolver

    resolver = ControlBindingResolver()
    bindings = resolver.parse(context.control_bindings)
    if not bindings:
        return context

    resolved = resolver.resolve(bindings)
    if resolved is None:
        return context

    if not resolved.available:
        logger.info(
            "control_bindings: best binding %s for device %s resolved to "
            "unavailable transport (%s); transport left empty",
            resolved.binding.binding_type,
            context.device_id,
            resolved.reason,
        )
        return context

    from dataclasses import replace

    logger.debug(
        "control_bindings: device %s transport set to %s from %s binding",
        context.device_id,
        resolved.transport_ref.kind,
        resolved.binding.binding_type,
    )
    return replace(context, transport=resolved.transport_ref)


__all__ = ["YamlContextProvider", "build_provider"]
