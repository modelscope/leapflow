# Copyright (c) Alibaba, Inc. and its affiliates.
"""Registry for selection policy plugins: register, discover, build from config.

Shaped after :class:`~leapflow.llm.provider_registry.LLMProviderRegistry`, which is
the established pattern for a core extension point: a Protocol, a registry populated
at startup, and config-driven instantiation. Deliberately *not* shaped after the tool
plugin pipeline -- a policy runs inside a turn and needs host services, so it must not
be sandboxed, approval-gated, or trust-graded.

One difference from the provider registry is intentional. Registering a policy id that
already exists is **refused**, not silently replaced. The provider registry allows
override because swapping an LLM backend is an operator's prerogative; a selection
policy silently replaced by a third-party package would change how the framework
chooses its own tools with nothing on the record. First registration wins and the
challenger is reported, which is the same arbitration the tool-name namespace uses.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional

from leapflow.plugins.selection_policy import (
    PolicyDeps,
    PolicyDescriptor,
    SelectionPolicy,
    SelectionPolicyPlugin,
)

logger = logging.getLogger(__name__)

#: Config key naming the active policy, and the id shipped as the default.
DEFAULT_POLICY_ID = "greedy"

#: setuptools group a third-party package advertises a policy under.
ENTRY_POINT_GROUP = "leapflow.selection_policies"


class SelectionPolicyRegistry:
    """Central registry of available selection policies.

    Not thread-safe: populated at startup and read afterwards, like every other
    extension-point registry here.
    """

    def __init__(self) -> None:
        self._plugins: Dict[str, SelectionPolicyPlugin] = {}
        self._rejected: List[Dict[str, str]] = []
        self._active: SelectionPolicy | None = None
        # The configuration the active instance was built from, so a change to any
        # ``selection.*`` value -- not just the policy id -- rebuilds it.
        self._built_from: Dict[str, Any] | None = None

    # ── registration ──────────────────────────────────────────────────────

    def register(self, plugin: SelectionPolicyPlugin) -> bool:
        """Register a policy plugin. First registration of an id wins.

        Returns ``True`` when accepted. A rejection is recorded rather than raised:
        one colliding package must not prevent every other policy from registering,
        which is the same reason tool-name arbitration is non-fatal.
        """
        policy_id = str(plugin.policy_id)
        if not policy_id:
            logger.warning("selection_policy: refusing a plugin with no policy_id")
            return False
        incumbent = self._plugins.get(policy_id)
        if incumbent is not None:
            self._rejected.append({
                "policy_id": policy_id,
                "kept": str(incumbent.display_name),
                "rejected": str(plugin.display_name),
            })
            logger.warning(
                "selection_policy: '%s' already registered by %s; rejecting %s",
                policy_id, incumbent.display_name, plugin.display_name,
            )
            return False
        self._plugins[policy_id] = plugin
        logger.debug("selection_policy: registered '%s'", policy_id)
        return True

    def discover_entry_points(self) -> int:
        """Register policies advertised by installed packages. Returns the count.

        A package that fails to load is skipped with a warning: a broken third-party
        strategy must not stop the framework from choosing tools at all.
        """
        found = 0
        try:
            from importlib.metadata import entry_points

            for entry in entry_points(group=ENTRY_POINT_GROUP):
                try:
                    if self.register(entry.load()()):
                        found += 1
                except Exception:  # noqa: BLE001 - one bad package, not a startup failure
                    logger.warning(
                        "selection_policy: entry point %r failed to load", entry.name,
                        exc_info=True,
                    )
        except Exception:  # noqa: BLE001
            logger.debug("selection_policy: entry point discovery unavailable", exc_info=True)
        return found

    # ── construction ──────────────────────────────────────────────────────

    def create(
        self,
        policy_id: str,
        params: Mapping[str, Any] | None = None,
        deps: PolicyDeps | None = None,
    ) -> Optional[SelectionPolicy]:
        """Build one policy by id, or ``None`` when it is unknown or fails."""
        plugin = self._plugins.get(str(policy_id))
        if plugin is None:
            logger.warning(
                "selection_policy: %r is not registered; available: %s",
                policy_id, ", ".join(self.available()) or "(none)",
            )
            return None
        try:
            return plugin.create(dict(params or {}), deps or PolicyDeps())
        except Exception:  # noqa: BLE001 - a broken policy must not break selection
            logger.warning("selection_policy: %r failed to build", policy_id, exc_info=True)
            return None

    def create_from_config(
        self, config: Mapping[str, Any], deps: PolicyDeps | None = None
    ) -> Optional[SelectionPolicy]:
        """Build the configured policy, falling back to the default id.

        Reads ``selection_policy`` for the id and ``policy_params`` for the shared
        parameter dict every policy filters for itself, so adding a strategy needs no
        change to the settings schema.
        """
        policy_id = str(config.get("selection_policy") or DEFAULT_POLICY_ID)
        params = dict(config.get("policy_params") or {})
        policy = self.create(policy_id, params, deps)
        if policy is None and policy_id != DEFAULT_POLICY_ID:
            # An unknown id in config must not leave the resolver without a policy:
            # falling back keeps tool selection working while the log names the
            # misconfiguration.
            logger.warning(
                "selection_policy: falling back to %r after %r could not be built",
                DEFAULT_POLICY_ID, policy_id,
            )
            return self.create(DEFAULT_POLICY_ID, {}, deps)
        return policy

    def activate(
        self, deps: PolicyDeps | None = None, *, settings: Any = None
    ) -> Optional[SelectionPolicy]:
        """Build the configured policy once and keep it as the process's active one.

        Cached for correctness, not speed. A learning policy carries a posterior, so
        building a fresh instance per call would hand every observation to a throwaway
        object and nothing would ever accumulate -- silently, since each individual
        call looks fine. ``LLMProviderRegistry`` caches its instances for the same
        reason a provider holds a session.

        ``settings`` is the *live* configuration, pushed in by the caller. It has to be
        pushed: ``get_settings()`` is a boot snapshot with no refresh path, so a
        component that reads it for a mutable setting reads a value from process start
        forever, and ``selection.policy`` would present itself as hot-reloadable
        through ``leap config`` while never taking effect. The whole object rather than
        individual values, so a new policy parameter needs no new argument here.

        Switching is self-correcting rather than driven by an external invalidation
        hook, so it holds for every entry point -- in-process CLI, daemon, tests --
        without each having to remember to call it. A *stateful* policy loses its
        in-memory state on a switch, which is why posteriors belong in a store rather
        than the instance.

        Called by the component that *selects*, because that is the one holding the
        live services a policy may read. Reporters use :meth:`current` instead, so a
        cold-path caller can never install a policy with no dependencies.
        """
        config = (
            {
                "selection_policy": getattr(settings, "selection_policy", "") or DEFAULT_POLICY_ID,
                "policy_params": policy_params_from_settings(settings),
            }
            if settings is not None
            else _settings_config()
        )
        wanted = str(config.get("selection_policy") or DEFAULT_POLICY_ID)
        if (
            self._active is not None
            # Compared on the whole effective configuration, not just the id: every
            # ``selection.*`` key presents itself as hot-reloadable through
            # ``leap config``, so changing an exploration coefficient or an
            # experiment's arms has to take effect too. Comparing ids alone left
            # those silently pinned to their process-start values.
            and self._built_from != config
            # Only when the wanted id is registered. An unknown id already fell back
            # to the default and logged; rebuilding on every activation would thrash
            # and re-log for the life of the misconfiguration.
            and wanted in self._plugins
        ):
            logger.info(
                "selection_policy: rebuilding %r after a configuration change",
                self._active.policy_id,
            )
            self._active = None
        if self._active is None:
            # A rebuilt policy loses its in-memory state, which is why a posterior
            # belongs in the durable store rather than the instance.
            self._active = self.create_from_config(config, deps)
            self._built_from = dict(config)
        return self._active

    def current(self) -> Optional[SelectionPolicy]:
        """The active policy, or ``None`` if nothing has selected yet.

        Deliberately non-creating. A reporter that built the policy would cache one
        with no host dependencies and win the race against the real owner; and with
        nothing selected yet there is by definition no decision to report on.
        """
        return self._active

    # ── introspection ─────────────────────────────────────────────────────

    def available(self) -> List[str]:
        return sorted(self._plugins)

    def describe(self) -> List[PolicyDescriptor]:
        """Every registered policy. Read by the config catalog's value hint, so an
        operator setting ``selection.policy`` can discover the ids that exist --
        including ones a third-party package registered.
        """
        return [
            PolicyDescriptor(policy_id=pid, display_name=str(plugin.display_name))
            for pid, plugin in sorted(self._plugins.items())
        ]

    @property
    def rejected(self) -> List[Dict[str, str]]:
        """Collisions, surfaced rather than silently dropped."""
        return list(self._rejected)


_registry: SelectionPolicyRegistry | None = None


def _settings_config() -> Dict[str, Any]:
    """The policy selection read from settings, in one place.

    Both the selecting component and the reporting one must agree on which policy is
    active; reading the key in two places is how they would come to disagree.

    Translates the flat, ``leap config``-discoverable ``selection_*`` settings into the
    per-policy ``params`` each plugin reads. Flat keys are the durable surface -- a
    nested dict in ``Settings`` would be a YAML-only knob, which the config contract
    forbids -- while ``params`` stays open so a policy arriving through the entry point
    group has somewhere to be configured from without adding fields to ``Settings``.
    """
    try:
        from leapflow.config import get_settings

        settings = get_settings()
    except Exception:  # noqa: BLE001 - an unreadable config still selects tools
        logger.debug("selection_policy: settings unavailable, using the default", exc_info=True)
        return {"selection_policy": DEFAULT_POLICY_ID, "policy_params": {}}

    return {
        "selection_policy": getattr(settings, "selection_policy", "") or DEFAULT_POLICY_ID,
        "policy_params": policy_params_from_settings(settings),
    }


def policy_params_from_settings(settings: Any) -> Dict[str, Any]:
    """The shared parameter dict every policy filters for itself.

    Empty for the shipped set: ``GreedyPolicy`` takes no configuration. Kept as the
    seam's contract rather than removed, because a third-party policy registered
    through the entry point group cannot add typed fields to ``Settings`` and this open
    dict is its only configuration path.
    """
    params: Dict[str, Any] = {}
    extra = getattr(settings, "policy_params", None)
    if isinstance(extra, Mapping):
        params.update({str(k): v for k, v in extra.items()})
    return params


def get_selection_policy_registry() -> SelectionPolicyRegistry:
    """The process-wide registry, populated with built-ins on first use.

    A singleton for the same reason the plugin and provider registries are: the set
    of available policies is a property of the process, and the cold-path sweep needs
    to reach the same policy the resolver used in order to report back to it.
    """
    global _registry
    if _registry is None:
        _registry = SelectionPolicyRegistry()
        from leapflow.plugins._builtin_policies import register_builtin_policies

        register_builtin_policies(_registry)
        _registry.discover_entry_points()
    return _registry


def reset_selection_policy_registry() -> None:
    """Drop the singleton, including any active instance. For tests."""
    global _registry
    _registry = None


__all__ = [
    "DEFAULT_POLICY_ID",
    "ENTRY_POINT_GROUP",
    "SelectionPolicyRegistry",
    "get_selection_policy_registry",
    "reset_selection_policy_registry",
]
