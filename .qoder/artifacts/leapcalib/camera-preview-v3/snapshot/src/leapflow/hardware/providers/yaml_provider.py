# Copyright (c) Alibaba, Inc. and its affiliates.
"""Declaration-file provider: reads hardware contexts from YAML on disk.

The default and, before an upstream standard is available, the only source of
device knowledge. Declarations are durable user assets under the profile: they
encode physical operating limits a person is accountable for, so this provider
never writes them and never rewrites them.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import yaml

from leapflow.hardware.context import ContextSource, HardwareContext
from leapflow.hardware.providers import ProviderError
from leapflow.hardware.attestation import read_verified, verification_stamp

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
        self._verified: Mapping[str, Any] = read_verified(config.get("verified_path"))

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
        person confirming a context must not have to edit the file they are
        confirming, or the confirmation would be self-attested.  Which
        confirmations apply -- and what makes one apply -- is decided once, in
        ``hardware.attestation``, so a device cannot be admitted here under a
        weaker rule than the fleet applies to the same declaration.
        """
        if context.provenance.is_verified:
            return context
        verifier, verified_at = verification_stamp(self._verified, context.device_id, context)
        if not verifier:
            return context
        return replace(
            context,
            provenance=replace(
                context.provenance,
                verified_by=verifier,
                verified_at=verified_at,
            ),
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
