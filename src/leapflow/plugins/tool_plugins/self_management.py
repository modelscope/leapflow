# Copyright (c) Alibaba, Inc. and its affiliates.
"""Self-Management plugin — lets the Agent introspect and manage its own plugin composition.

This is the Phase 2.4 Self-Modification MVP. It exposes thirteen tools:

Read-only governance (no approval needed):
    - plugin_list     : list all registered plugins across Tool/Gateway/LLM subsystems
    - plugin_status   : detailed info about one plugin (tools, deps, fiber state, generation)
    - plugin_versions : inspect recorded profile plugin versions and the active pointer
    - plugin_propose  : create a side-effect-free proposal from capability-gap evidence
    - assess_compatibility : assess foreign plugin manifest compatibility with LeapFlow

Governed generation (proposal content approval, no installation yet):
    - plugin_generate : describe a capability need; the LLM produces conformant
                        plugin code, stores it in CAS, validates it, and requests
                        content approval. Installation is a second gated step.

State-mutating (REQUIRES approval — routed through the plugin_approval_gate):
    - plugin_install  : write validated code (from plugin_generate) or a
                        marketplace payload into the profile-scoped plugins
                        directory and load it dynamically. This mutates the
                        filesystem and the live registry.
    - plugin_rollback : restore a recorded source snapshot and hot-reload it
    - plugin_reload   : hot-reload a plugin
    - plugin_disable  : dispose a plugin's fiber (removes its tools)
    - plugin_remove   : terminally remove a plugin and optionally delete source
    - plugin_enable   : re-enable a previously disabled plugin

Concurrency note: plugin_install, plugin_disable, plugin_reload, and plugin_enable
operate at the process-global registry level; changes affect all sessions in this
daemon, not just the current conversation. In-flight turns keep using their
per-turn handler snapshot so they finish safely; only NEW turns started after the
change see the new plugin set.

Approval note: In non-daemon (in-process CLI) mode, no plugin_approval_gate is
installed, so mutation tools will always fail-closed. Self-modification is
available only in daemon mode where the ApprovalCoordinator wires the gate.

LLM co-evolution note: plugin_generate depends on an optional llm_provider that
is wired via bind_runtime. If unavailable (e.g. no LLM credentials configured or
the container has not propagated one yet), the tool reports the missing
dependency instead of pretending to have generated code.

Design principle: this is the Agent's window into its own architecture. It must
be transparent (introspection is free) but safe (mutation requires explicit
approval, and self-modification is classified HIGH risk with no permanent grants).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from leapflow.plugins.protocol import ToolMetadata

logger = logging.getLogger(__name__)



def _declared_capabilities(proposal: Any) -> tuple[str, ...]:
    """The capability names a proposal was raised for, from its own evidence.

    Read off ``GapEvidence.metadata`` rather than re-derived, because the producer
    already recorded it there: ``capability_gap_detector`` puts ``intent.capability``
    into the metadata of a world-model proposal. Re-deriving it from the summary would
    be inferring a capability name from text, which the observation layer forbids.

    Empty for a proposal that carries none -- notably the ``unknown_tool`` path, whose
    "capability" is the missing tool's invented name and therefore not a name any tool
    should declare. Generating with no declaration is still better than generating with
    a wrong one.
    """
    if proposal is None:
        return ()
    names: list[str] = []
    for evidence in getattr(proposal, "evidence", ()) or ():
        for key, value in dict(getattr(evidence, "metadata", ()) or ()).items():
            if str(key) == "capability" and str(value).strip():
                candidate = str(value).strip()
                if candidate not in names:
                    names.append(candidate)
    return tuple(names)

class SelfManagementPlugin:
    """ToolPlugin exposing the Agent's own plugin management surface."""

    def __init__(self) -> None:
        self._plugin_approval_gate: Any = None
        # Optional: an LLM provider (leapflow.llm.LLMProvider-like) used by
        # plugin_generate. Wired opportunistically via bind_runtime — the tool
        # degrades gracefully when it is absent so introspection and mutation
        # paths never break because generation is offline.
        self._llm_provider: Any = None
        # Opt-in switch for LLM-driven plugin generation. Wired from
        # Settings.plugin_generation_enabled by the daemon; defaults to False
        # so an unattended profile cannot spend tokens synthesizing plugins.
        self._plugin_generation_enabled: bool = False
        # Profile-scoped directory where plugin_install writes plugin code and
        # loads it dynamically. Injected via bind_runtime by the daemon
        # approval coordinator (derived from ProfileLayout). None -> resolved
        # lazily from the active profile layout so in-process CLI mode still
        # installs into a profile-scoped path rather than the package dir.
        self._plugin_install_dir: Optional[str] = None
        self._plugin_staging_dir: Optional[str] = None
        # Optional MarketplaceClient used by the marketplace_name install branch.
        # None when no marketplace is configured; the branch then returns a
        # structured error.
        self._marketplace_client: Any = None
        # Hex-encoded Ed25519 public keys trusted to sign marketplace plugins.
        # When non-empty, marketplace installs require a valid signature.
        self._trusted_pubkeys: set[str] = set()
        # Sole acquisition-lifecycle ledger (PENDING -> GENERATED -> APPROVED ->
        # INSTALLED -> PROBATION -> VERIFIED/QUARANTINED). Rich review content is
        # embedded in the same record; AdaptiveEvolutionPolicy and LifecycleGovernor
        # both operate on this store.
        self._capability_lifecycle_store: Any = None
        self._proposal_orchestrator: Any = None
        self._evolution_outbox: Any = None
        self._evolution_profile_id: str = ""
        # Optional version store; lazily resolved from ProfileLayout.plugin_versions_dir.
        self._plugin_version_store: Any = None
        # Optional adaptive capability decision store; lazily resolved from
        # ProfileLayout.capability_plans_path.
        self._capability_plan_store: Any = None

    @property
    def plugin_id(self) -> str:
        return "self_management"

    @property
    def category(self) -> str:
        return "system"

    @property
    def dependencies(self) -> list[str]:
        return [
            "plugin_approval_gate",
            "llm_provider",
            "plugin_generation_enabled",
            "plugin_install_dir",
            "plugin_staging_dir",
            "marketplace_client",
            "marketplace_trusted_pubkeys",
            "plugin_version_store",
            "capability_plan_store",
            "capability_lifecycle_store",
            "proposal_orchestrator",
            "evolution_outbox",
            "evolution_profile_id",
        ]

    def bind_runtime(self, **deps: Any) -> None:
        if "plugin_approval_gate" in deps:
            self._plugin_approval_gate = deps["plugin_approval_gate"]
        if "llm_provider" in deps:
            self._llm_provider = deps["llm_provider"]
        if "plugin_generation_enabled" in deps:
            self._plugin_generation_enabled = bool(deps["plugin_generation_enabled"])
        if "plugin_install_dir" in deps:
            value = deps["plugin_install_dir"]
            self._plugin_install_dir = str(value) if value else None
        if "plugin_staging_dir" in deps:
            value = deps["plugin_staging_dir"]
            self._plugin_staging_dir = str(value) if value else None
        if "marketplace_client" in deps:
            self._marketplace_client = deps["marketplace_client"]
        if "marketplace_trusted_pubkeys" in deps:
            raw = deps["marketplace_trusted_pubkeys"] or ()
            self._trusted_pubkeys = {str(k).strip() for k in raw if str(k).strip()}
        if "plugin_version_store" in deps:
            self._plugin_version_store = deps["plugin_version_store"]
        if "capability_plan_store" in deps:
            self._capability_plan_store = deps["capability_plan_store"]
        if "capability_lifecycle_store" in deps:
            self._capability_lifecycle_store = deps["capability_lifecycle_store"]
        if "proposal_orchestrator" in deps:
            self._proposal_orchestrator = deps["proposal_orchestrator"]
        if "evolution_outbox" in deps:
            self._evolution_outbox = deps["evolution_outbox"]
        if "evolution_profile_id" in deps:
            self._evolution_profile_id = str(deps["evolution_profile_id"] or "")

    # ── Read-only introspection ────────────────────────────

    async def _plugin_list_handler(self, **kwargs: Any) -> Dict[str, Any]:
        """List all registered plugins across Tool/Gateway/LLM subsystems."""
        from leapflow.plugins import get_registry, get_scoped_registry

        try:
            reg = get_registry()
            scoped = get_scoped_registry()

            plugins_info: list[dict[str, Any]] = []
            for plugin_id, plugin in reg.plugins.items():
                fiber = scoped.get_fiber(plugin_id)
                plugins_info.append(
                    {
                        "plugin_id": plugin_id,
                        "category": plugin.category,
                        "tool_count": len(plugin.tools),
                        "state": fiber.state.value if fiber else "unmanaged",
                        "generation": fiber.generation if fiber else None,
                    }
                )

            # Cross-subsystem: Gateway adapters
            gateway_adapters: list[dict[str, Any]] = []
            try:
                from leapflow.gateway.adapters import BUILTIN_PLUGINS

                for bp in BUILTIN_PLUGINS:
                    gateway_adapters.append(
                        {
                            "platform_id": bp.platform_id,
                            "display_name": bp.display_name,
                            "subsystem": "gateway",
                        }
                    )
            except (ImportError, AttributeError):
                pass

            # Cross-subsystem: LLM providers
            llm_providers: list[dict[str, Any]] = []
            try:
                from leapflow.llm.provider_registry import (
                    get_default_registry as get_llm_registry,
                )

                llm_reg = get_llm_registry()
                for plugin_meta in llm_reg.list_plugins():
                    llm_providers.append(
                        {
                            "provider_id": plugin_meta.get("provider_id", "unknown"),
                            "display_name": plugin_meta.get("display_name", ""),
                            "subsystem": "llm",
                        }
                    )
            except (ImportError, AttributeError, RuntimeError):
                pass

            return {
                "ok": True,
                "subsystem": "tools",
                "plugin_count": len(plugins_info),
                "plugins": plugins_info,
                "categories": sorted(reg.categories),
                # Cross-subsystem introspection (additive)
                "gateway_adapters": gateway_adapters,
                "llm_providers": llm_providers,
                "total_count": len(plugins_info) + len(gateway_adapters) + len(llm_providers),
                "capability_report": self._build_capability_report(
                    reg,
                    scoped,
                    plugins_info,
                    gateway_adapters,
                    llm_providers,
                ),
            }
        except (RuntimeError, AttributeError) as exc:
            logger.warning("plugin_list failed: %s", exc, exc_info=True)
            return {"ok": False, "error": f"plugin_list failed: {exc}"}

    def _build_capability_report(
        self,
        reg: Any,
        scoped: Any,
        plugins_info: list[dict[str, Any]],
        gateway_adapters: list[dict[str, Any]],
        llm_providers: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Build a live, evidence-backed capability report for self-questions."""
        tool_categories: dict[str, dict[str, Any]] = {}
        self_management_tools: list[str] = []
        mutation_tools: list[str] = []
        approval_required_tools: list[str] = []
        read_only_tools: list[str] = []

        for tool in reg.all_metadata:
            metadata = dict(tool.x_leapflow or {})
            category = str(metadata.get("category") or "general")
            bucket = tool_categories.setdefault(
                category,
                {
                    "tool_count": 0,
                    "tools": [],
                    "approval_required_count": 0,
                    "mutating_count": 0,
                },
            )
            bucket["tool_count"] += 1
            bucket["tools"].append(tool.name)
            if bool(tool.mutates_state):
                mutation_tools.append(tool.name)
                bucket["mutating_count"] += 1
            else:
                read_only_tools.append(tool.name)
            if metadata.get("requires_approval") is True:
                approval_required_tools.append(tool.name)
                bucket["approval_required_count"] += 1
            if tool.name.startswith("plugin_") or tool.name == "assess_compatibility":
                self_management_tools.append(tool.name)

        for bucket in tool_categories.values():
            bucket["tools"] = sorted(bucket["tools"])

        profile_layout = self._profile_layout_or_none()
        install_dir = self._safe_install_dir()
        dependency_state = {
            "approval_gate_bound": self._plugin_approval_gate is not None,
            "llm_provider_bound": self._llm_provider is not None,
            "plugin_generation_enabled": self._plugin_generation_enabled,
            "plugin_install_dir": install_dir,
            "marketplace_configured": self._marketplace_client is not None,
            "trusted_marketplace_pubkeys": len(self._trusted_pubkeys),
            "proposal_store_available": self._capability_lifecycle_store is not None,
            "version_store_available": (
                self._plugin_version_store is not None or profile_layout is not None
            ),
            "capability_plan_store_available": (
                self._capability_plan_store is not None or profile_layout is not None
            ),
        }
        limitations = self._capability_limitations(dependency_state)

        return {
            "source": "live_runtime_registry",
            "registry": {
                "version": reg.version,
                "plugin_count": len(plugins_info),
                "tool_count": len(reg.tool_handlers),
                "fiber_count": len(scoped.fibers),
                "categories": sorted(tool_categories),
                "capability_conflicts": [
                    {
                        "tool_name": c.tool_name,
                        "kept_plugin": c.kept_plugin,
                        "rejected_plugin": c.rejected_plugin,
                    }
                    for c in getattr(reg, "conflicts", [])
                ],
            },
            "plugins_supported": {
                "supported": "self_management" in reg.plugins,
                "evidence_tools": sorted(self_management_tools),
                "profile_installs": bool(install_dir),
                "hot_reload": "plugin_reload" in self_management_tools,
                "versioning": "plugin_versions" in self_management_tools
                and "plugin_rollback" in self_management_tools,
                "compatibility_assessment": "assess_compatibility" in self_management_tools,
            },
            "self_evolution": {
                "proposal_flow": "plugin_propose" in self_management_tools,
                "generation_tool": "plugin_generate" in self_management_tools,
                "generation_ready": self._plugin_generation_enabled
                and self._llm_provider is not None,
                "install_tool": "plugin_install" in self_management_tools,
                "rollback_tool": "plugin_rollback" in self_management_tools,
                "behavior_test_gate": True,
            },
            "runtime_dependencies": dependency_state,
            "tool_categories": dict(sorted(tool_categories.items())),
            "read_only_tool_count": len(read_only_tools),
            "mutation_tool_count": len(mutation_tools),
            "approval_required_tools": sorted(approval_required_tools),
            "gateway_adapter_count": len(gateway_adapters),
            "llm_provider_count": len(llm_providers),
            "limitations": limitations,
            "answering_guidance": [
                "Use this live report as the evidence source for questions about LeapFlow capabilities.",
                (
                    "State configuration-dependent capabilities as available only when "
                    "their dependency flags are ready."
                ),
                "If this report is unavailable, say that live capability verification failed instead of guessing.",
            ],
        }

    def _profile_layout_or_none(self) -> Any:
        try:
            from leapflow.config import get_settings

            return getattr(get_settings(), "profile_layout", None)
        except (RuntimeError, AttributeError, ImportError):
            return None

    def _safe_install_dir(self) -> str:
        try:
            return str(self._resolve_install_dir())
        except (RuntimeError, AttributeError, ImportError):
            return ""

    @staticmethod
    def _capability_limitations(dependency_state: dict[str, Any]) -> list[str]:
        limitations: list[str] = []
        if not dependency_state["approval_gate_bound"]:
            limitations.append("Mutation tools fail closed until plugin_approval_gate is bound.")
        if not dependency_state["llm_provider_bound"]:
            limitations.append("plugin_generate cannot run until an LLM provider is bound.")
        if not dependency_state["plugin_generation_enabled"]:
            limitations.append("plugin_generate is disabled by configuration.")
        if not dependency_state["marketplace_configured"]:
            limitations.append("Marketplace installs require a configured marketplace client.")
        if not dependency_state["proposal_store_available"]:
            limitations.append(
                "Plugin proposals require the daemon-injected evolution lifecycle store."
            )
        if not dependency_state["version_store_available"]:
            limitations.append(
                "Plugin versioning requires a profile layout or injected version store."
            )
        if not dependency_state["capability_plan_store_available"]:
            limitations.append(
                "Adaptive capability plan history requires a profile layout or injected plan store."
            )
        return limitations

    async def _plugin_status_handler(self, plugin_id: str, **kwargs: Any) -> Dict[str, Any]:
        """Detailed information about a specific plugin."""
        from leapflow.plugins import get_registry, get_scoped_registry

        try:
            reg = get_registry()
            plugin = reg.get_plugin(plugin_id)
            if plugin is None:
                return {"ok": False, "error": f"Plugin '{plugin_id}' not registered"}

            scoped = get_scoped_registry()
            fiber = scoped.get_fiber(plugin_id)

            response = {
                "ok": True,
                "plugin_id": plugin_id,
                "category": plugin.category,
                "dependencies": list(plugin.dependencies),
                "tools": [{"name": t.name, "description": t.description} for t in plugin.tools],
                "fiber": {
                    "state": fiber.state.value if fiber else "unmanaged",
                    "generation": fiber.generation if fiber else None,
                },
            }
            descriptor = getattr(plugin, "descriptor", None)
            if descriptor is not None and hasattr(descriptor, "to_dict"):
                descriptor_data = descriptor.to_dict()
                response["dsh"] = {
                    "source_kind": descriptor_data.get("source_kind"),
                    "bundle_sha256": descriptor_data.get("bundle_sha256"),
                    "entry_point": descriptor_data.get("entry_point"),
                    "verdict": (
                        "partial"
                        if descriptor_data.get("client_components")
                        else "adaptable"
                    ),
                    "limitations": descriptor_data.get("limitations", []),
                    "client_components": descriptor_data.get("client_components", []),
                    "runtime": "node",
                }

            # Learning-driven trust and recommendation (purely additive)
            try:
                from leapflow.learning.plugin_advisor import get_default_advisor

                advisor = get_default_advisor()
                if advisor is not None:
                    trust = advisor._trust_ledger.level(plugin_id)
                    response["trust_level"] = trust.name
                    rec = advisor.recommend(plugin_id)
                    if rec is not None:
                        response["recommendation"] = {
                            "action": rec.action,
                            "reason": rec.reason,
                            "confidence": rec.confidence,
                        }
            except (ImportError, AttributeError, RuntimeError):
                pass  # Learning integration not wired — degrade gracefully

            return response
        except (RuntimeError, AttributeError) as exc:
            logger.warning("plugin_status failed for %s: %s", plugin_id, exc, exc_info=True)
            return {"ok": False, "error": f"plugin_status failed: {exc}"}

    async def _plugin_plan_handler(
        self, limit: int = 5, latest: bool = False, **kwargs: Any
    ) -> Dict[str, Any]:
        """Inspect stored adaptive capability decisions and plans."""
        try:
            store = self._capability_plan_store_resolved()
            if latest:
                record = store.latest()
                return {
                    "ok": True,
                    "store_path": str(getattr(store, "path", "")),
                    "latest": record,
                    "records": [record] if record else [],
                }
            records = store.list_records(limit=max(1, int(limit or 5)))
            return {
                "ok": True,
                "store_path": str(getattr(store, "path", "")),
                "records": records,
                "count": len(records),
            }
        except (RuntimeError, AttributeError, OSError, ValueError) as exc:
            logger.warning("plugin_plan failed: %s", exc, exc_info=True)
            return {"ok": False, "error": f"plugin_plan failed: {exc}"}

    # ── Generation (produces code, does NOT install) ──────────

    async def _plugin_propose_handler(
        self,
        requested_capability: str,
        plugin_id: str = "",
        proposed_tools: list[str] | None = None,
        test_cases: list[dict[str, Any]] | None = None,
        risk_level: str = "read_only",
        evidence: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Create a side-effect-free plugin proposal from explicit evidence."""
        try:
            from leapflow.learning.capability_gap_detector import CapabilityGapDetector
        except ImportError as exc:
            return {"ok": False, "error": f"Capability gap detector unavailable: {exc}"}

        detector = CapabilityGapDetector()
        try:
            proposal = None
            if evidence and evidence.get("error_type") == "unknown_tool":
                proposal = detector.proposal_from_unknown_tool(
                    evidence,
                    requested_capability=requested_capability,
                )
            if proposal is None:
                proposal = detector.proposal_from_capability_request(
                    requested_capability,
                    plugin_id=plugin_id,
                    proposed_tool_names=tuple(proposed_tools or ()),
                    risk_level=risk_level,  # type: ignore[arg-type]
                    evidence_summary=str((evidence or {}).get("summary") or ""),
                )
        except (TypeError, ValueError) as exc:
            return {"ok": False, "error": f"Proposal failed: {exc}"}

        if test_cases:
            try:
                from leapflow.domain.plugin_proposal import BehaviorTestCase, PluginProposal

                parsed_tests = tuple(
                    BehaviorTestCase.create(
                        str(item.get("tool_name") or ""),
                        arguments=dict(item.get("arguments") or {}),
                        expected_subset=dict(item.get("expected_subset") or {}),
                        description=str(item.get("description") or ""),
                    )
                    for item in test_cases
                    if isinstance(item, dict)
                )
                proposal = PluginProposal(
                    proposal_id=proposal.proposal_id,
                    plugin_id=proposal.plugin_id,
                    capability_summary=proposal.capability_summary,
                    gap_type=proposal.gap_type,
                    risk_level=proposal.risk_level,
                    status=proposal.status,
                    evidence=proposal.evidence,
                    proposed_tools=proposal.proposed_tools,
                    test_cases=parsed_tests,
                    created_at=proposal.created_at,
                )
            except (TypeError, ValueError) as exc:
                return {"ok": False, "error": f"Proposal test case parsing failed: {exc}"}

        lifecycle_id = self._open_lifecycle_record(proposal, requested_capability)
        if not lifecycle_id:
            return {
                "ok": False,
                "error": "Proposal persistence failed: lifecycle store unavailable",
            }

        return {
            "ok": True,
            "action": "propose",
            "proposal": proposal.to_dict(),
            "lifecycle_proposal_id": lifecycle_id,
            "next_actions": [
                "Review proposal fields and risk level.",
                "If acceptable, call plugin_generate with proposal_id to preserve review metadata.",
                "Install generated code separately with plugin_install(proposal_id=...) after validation and approval.",
            ],
        }

    async def _plugin_generate_handler(
        self, plugin_id: str = "", description: str = "", proposal_id: str = "", **kwargs: Any
    ) -> Dict[str, Any]:
        """Generate a new plugin via LLM and validate it. Returns validated code (does NOT install).

        This is the LLM co-evolution entry point: describe a capability need,
        the LLM generates conformant plugin code, and it's rigorously validated.
        Installation is a SEPARATE approval-gated step (plugin_install).
        """
        provides_capabilities: tuple[str, ...] = ()
        source = ""
        if proposal_id:
            source, resolved_plugin_id, resolved_description, provides_capabilities = (
                self._resolve_generation_source(proposal_id)
            )
            if not source:
                return {"ok": False, "error": f"Plugin proposal '{proposal_id}' not found"}
            plugin_id = plugin_id or resolved_plugin_id
            description = description or resolved_description
        if not plugin_id or not description:
            return {
                "ok": False,
                "error": "plugin_id and description are required unless proposal_id is provided",
            }

        if not self._plugin_generation_enabled:
            return {
                "ok": False,
                "error": (
                    "Plugin generation is disabled. "
                    "Set plugin_generation_enabled=true in config to opt in."
                ),
            }

        try:
            from leapflow.learning.plugin_generator import PluginGenerator, PluginGenerationRequest
        except ImportError as exc:
            return {"ok": False, "error": f"Generation module unavailable: {exc}"}

        if self._llm_provider is None:
            return {
                "ok": False,
                "error": (
                    "No LLM provider available for plugin generation. "
                    "Wire an llm_provider into self_management via bind_runtime "
                    "(requires daemon-mode with LLM credentials configured)."
                ),
            }

        try:
            generator = PluginGenerator(llm_provider=self._llm_provider)
            request = PluginGenerationRequest(
                plugin_id=plugin_id,
                description=description,
                provides_capabilities=provides_capabilities,
            )
            result = await generator.generate_and_validate(request)
            if proposal_id:
                lifecycle_id = self._lifecycle_proposal_id(source, proposal_id)
                result["proposal_id"] = proposal_id
                result["lifecycle_proposal_id"] = lifecycle_id
                if result.get("ok") and lifecycle_id and self._proposal_orchestrator is not None:
                    item = self._proposal_orchestrator.register_generated(
                        lifecycle_id,
                        str(result.get("code") or ""),
                        validation={
                            "ok": True,
                            "stage": "passed",
                            "compatibility_ok": True,
                            "target_protocol": "ToolPlugin",
                            "exposed_tools": list(result.get("exposed_tools") or ()),
                        },
                    )
                    content_approval = await self._proposal_orchestrator.approve_content(
                        lifecycle_id
                    )
                    result.update(
                        {
                            "generated_code_ref": item.generated_code_ref,
                            "content_approved": content_approval.approved,
                            "content_approval_id": content_approval.approval_id,
                        }
                    )
                    if not content_approval.approved:
                        result.update(
                            {
                                "ok": False,
                                "error": content_approval.denial_message,
                                "requires_approval": True,
                            }
                        )
                elif result.get("ok"):
                    result.update(
                        {
                            "ok": False,
                            "error": "proposal orchestration unavailable; content approval cannot be recorded",
                            "requires_approval": True,
                        }
                    )
            return result
        except (AttributeError, KeyError, PermissionError, RuntimeError, ValueError) as exc:
            return {"ok": False, "error": f"Generation failed: {exc}"}

    def _resolve_generation_source(
        self, proposal_id: str
    ) -> tuple[str, str, str, tuple[str, ...]]:
        """Resolve either canonical lifecycle id or review alias from one store."""
        try:
            store = self._lifecycle_store()
            item = store.get(proposal_id)
            source = "lifecycle"
            if item is None:
                item = store.find_by_metadata("review_proposal_id", proposal_id)
                source = "review"
        except (RuntimeError, OSError, ValueError, AttributeError):
            item = None
            source = ""
        if item is None:
            return ("", "", "", ())
        metadata = dict(item.metadata or {})
        review_payload = metadata.get("review_proposal")
        if isinstance(review_payload, Mapping):
            from leapflow.domain.plugin_proposal import PluginProposal

            review = PluginProposal.from_dict(review_payload)
            return (
                source,
                str(review.plugin_id),
                str(review.capability_summary),
                _declared_capabilities(review),
            )
        requirements = [dict(requirement) for requirement in (item.requirements or ())]
        capability = str((requirements[0].get("capability") if requirements else "") or "")
        plugin_id = str(metadata.get("plugin_id") or "")
        description = str(
            metadata.get("capability_summary")
            or (requirements[0].get("evidence") if requirements else "")
            or capability
        )
        provides = (capability,) if capability else ()
        return (source, plugin_id, description, provides)

    def _lifecycle_proposal_id(self, source: str, proposal_id: str) -> str:
        if source == "lifecycle":
            return proposal_id
        if source == "review":
            try:
                item = self._lifecycle_store().find_by_metadata(
                    "review_proposal_id", proposal_id
                )
            except (RuntimeError, OSError, ValueError, AttributeError):
                item = None
            return str(getattr(item, "proposal_id", "") or "")
        return ""

    # ── Compatibility assessment (read-only) ─────────────────

    async def _assess_compatibility_handler(
        self,
        manifest: dict | None = None,
        source_path: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Assess a foreign manifest or real DSH source bundle."""
        if manifest is None:
            manifest = kwargs.get("manifest")
        source_path = str(source_path or kwargs.get("source_path") or "")
        if manifest and source_path:
            return {"ok": False, "error": "Provide either manifest or source_path, not both"}
        if not source_path and (not manifest or not isinstance(manifest, dict)):
            return {
                "ok": False,
                "error": "manifest (dict) or source_path (DSH bundle directory) is required",
            }
        if source_path:
            from pathlib import Path

            from leapflow.tools.execution_context import require_workspace_access

            scope_error = await require_workspace_access(
                Path(source_path).expanduser().resolve(),
                operation="assess_compatibility source",
                effect="read",
            )
            if scope_error:
                return scope_error

        try:
            from leapflow.learning.compatibility import assess_plugin

            report = assess_plugin(source_path or manifest)
            plan = report.execution_plan
            return {
                "ok": True,
                "final_verdict": report.final_verdict.value,
                "is_installable": report.is_installable(),
                "installable_candidate": bool(plan and plan.installable_candidate),
                "runtime_ready": bool(plan and plan.runtime_ready),
                "target_protocol": report.target_protocol,
                "rejection_reason": report.rejection_reason,
                "adaptation_notes": report.adaptation_notes,
                "adapter_spec": {
                    "source_interface": report.adapter_spec.source_interface,
                    "target_protocol": report.adapter_spec.target_protocol,
                    "bridge_type": report.adapter_spec.bridge_type,
                    "shim_methods": report.adapter_spec.shim_methods,
                    "estimated_complexity": report.adapter_spec.estimated_complexity,
                }
                if report.adapter_spec
                else None,
                "execution_plan": self._compatibility_plan_payload(plan),
                "stages": [
                    {
                        "stage_name": s.stage_name,
                        "passed": s.passed,
                        "verdict": s.verdict.value if s.verdict else None,
                        "details": s.details,
                    }
                    for s in report.stages
                ],
                "manifest_name": report.manifest.name,
                "manifest_version": report.manifest.version,
            }
        except (ImportError, AttributeError, OSError, TypeError, ValueError) as exc:
            logger.warning("assess_compatibility failed: %s", exc, exc_info=True)
            return {"ok": False, "error": f"Assessment failed: {exc}"}

    @staticmethod
    def _compatibility_plan_payload(plan: Any) -> Dict[str, Any] | None:
        if plan is None:
            return None
        return {
            "source_kind": plan.source_kind.value,
            "source_root": plan.source_root,
            "entry_point": plan.entry_point,
            "runtime": plan.runtime,
            "bundle_sha256": plan.bundle_sha256,
            "source_files": list(plan.source_files),
            "requires_discovery": plan.requires_discovery,
            "runtime_ready": plan.runtime_ready,
            "blockers": list(plan.blockers),
            "limitations": list(plan.limitations),
            "components": [
                {
                    "name": item.name,
                    "kind": item.kind.value,
                    "status": item.status.value,
                    "reason": item.reason,
                    "entry_point": item.entry_point,
                    "metadata": dict(item.metadata),
                }
                for item in plan.components
            ],
        }

    # ── State-mutating (requires approval) ─────────────────

    async def _plugin_install_handler(
        self,
        plugin_id: str = "",
        code: str = "",
        marketplace_name: str = "",
        source_path: str = "",
        proposal_id: str = "",
        version_label: str = "",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Install Python code/marketplace content or a real DSH source bundle."""
        proposal = None
        lifecycle_id = ""
        if proposal_id:
            try:
                store = self._lifecycle_store()
                lifecycle = store.get(proposal_id)
                if lifecycle is None:
                    lifecycle = store.find_by_metadata("review_proposal_id", proposal_id)
            except (RuntimeError, OSError, ValueError, AttributeError):
                lifecycle = None
            if lifecycle is None:
                return {"ok": False, "error": f"Plugin proposal '{proposal_id}' not found"}
            lifecycle_id = str(lifecycle.proposal_id)
            metadata = dict(lifecycle.metadata or {})
            review_payload = metadata.get("review_proposal")
            if isinstance(review_payload, Mapping):
                from leapflow.domain.plugin_proposal import PluginProposal

                proposal = PluginProposal.from_dict(review_payload)
                plugin_id = plugin_id or proposal.plugin_id
            else:
                plugin_id = plugin_id or str(metadata.get("plugin_id") or "")
        source_path = str(source_path or kwargs.get("source_path") or "")
        if (
            lifecycle_id
            and not code
            and not marketplace_name
            and not source_path
            and self._proposal_orchestrator is not None
        ):
            try:
                code = self._proposal_orchestrator.generated_code(lifecycle_id)
            except (KeyError, OSError, RuntimeError, ValueError) as exc:
                return {"ok": False, "error": str(exc)}
        modes = sum(bool(value) for value in (code, marketplace_name, source_path))
        if modes != 1:
            return {
                "ok": False,
                "error": "Provide exactly one of code, marketplace_name, or source_path",
            }

        source_metadata: dict[str, Any] = {}
        if source_path:
            try:
                from pathlib import Path

                from leapflow.learning.compatibility import assess_plugin, inspect_plugin_source
                from leapflow.plugins.dsh import normalize_plugin_id
                from leapflow.tools.execution_context import require_workspace_access

                scope_error = await require_workspace_access(
                    Path(source_path).expanduser().resolve(),
                    operation="plugin_install source",
                    effect="read",
                )
                if scope_error:
                    return scope_error
                inspection = inspect_plugin_source(source_path)
                report = assess_plugin(source_path)
                plan = report.execution_plan
                if plan is None or not plan.installable_candidate:
                    return {
                        "ok": False,
                        "error": report.rejection_reason or "; ".join(plan.blockers if plan else ()),
                        "verdict": report.final_verdict.value,
                    }
                plugin_id = normalize_plugin_id(plugin_id or inspection.manifest.name)
                source_metadata = {
                    "source_kind": plan.source_kind.value,
                    "source_path": str(Path(source_path).expanduser().resolve()),
                    "bundle_sha256": plan.bundle_sha256,
                    "verdict": report.final_verdict.value,
                    "permissions": list(plan.permissions),
                    "limitations": list(plan.limitations),
                    "components": [
                        {
                            "name": item.name,
                            "kind": item.kind.value,
                            "status": item.status.value,
                            "reason": item.reason,
                        }
                        for item in plan.components
                    ],
                }
            except (ImportError, OSError, TypeError, ValueError) as exc:
                return {"ok": False, "error": f"DSH source assessment failed: {exc}"}

        if not plugin_id:
            return {"ok": False, "error": "plugin_id is required unless source_path or proposal_id is provided"}

        if lifecycle_id:
            if self._proposal_orchestrator is None:
                return {
                    "ok": False,
                    "error": "proposal orchestration unavailable; mutation approval cannot be recorded",
                    "requires_approval": True,
                }
            try:
                mutation_approval = await self._proposal_orchestrator.authorize_mutation(
                    lifecycle_id
                )
            except (KeyError, PermissionError, ValueError) as exc:
                return {"ok": False, "error": str(exc), "requires_approval": True}
            if not mutation_approval.approved:
                return {
                    "ok": False,
                    "error": mutation_approval.denial_message,
                    "requires_approval": True,
                }
        else:
            approved, denial = await self._check_approval(
                "install", plugin_id, proposal_id=proposal_id, metadata=source_metadata,
            )
            if not approved:
                return {"ok": False, "error": denial, "requires_approval": True}

        from leapflow.plugins import get_registry

        # R1: reject a duplicate plugin_id BEFORE creating any fiber or writing
        # any file, so a re-install cannot leave a half-initialized fiber.
        if get_registry().get_plugin(plugin_id) is not None:
            return {
                "ok": False,
                "error": (
                    f"Plugin '{plugin_id}' is already registered; "
                    "use plugin_reload or choose a new id"
                ),
            }

        if code and marketplace_name:
            return {"ok": False, "error": "Provide either code or marketplace_name, not both"}

        try:
            if source_path:
                result = await self._install_from_dsh_source(
                    plugin_id,
                    source_path,
                    version_label=version_label,
                    expected_bundle_sha256=str(source_metadata.get("bundle_sha256") or ""),
                )
            elif code:
                result = await self._install_from_code(
                    plugin_id, code, proposal=proposal, version_label=version_label
                )
            elif marketplace_name:
                # Run compatibility gate for marketplace installs (BLOCKING)
                result = await self._install_from_marketplace_with_gate(plugin_id, marketplace_name)
            else:
                return {"ok": False, "error": "Must provide code, marketplace_name, or source_path"}
            if proposal_id:
                result["proposal_id"] = proposal_id
            if lifecycle_id and self._proposal_orchestrator is not None:
                self._proposal_orchestrator.record_installed(lifecycle_id, result)
                result["lifecycle_proposal_id"] = lifecycle_id
            elif result.get("ok"):
                from leapflow.domain.event_types import EvolutionEventType

                persisted = await self._emit_plugin_event(
                    EvolutionEventType.PLUGIN_INSTALLED,
                    plugin_id=plugin_id,
                    version_id=str(result.get("version") or version_label),
                    payload={
                        "action": "install",
                        "installed_tools": list(result.get("installed_tools") or ()),
                    },
                    dedup_suffix=str(
                        result.get("version")
                        or source_metadata.get("bundle_sha256")
                        or "installed"
                    ),
                )
                if not persisted:
                    result["audit_incomplete"] = True
            return result
        except (ImportError, AttributeError, OSError, RuntimeError, ValueError) as exc:
            logger.warning("plugin_install failed for %s: %s", plugin_id, exc, exc_info=True)
            return {"ok": False, "error": f"Install failed: {exc}"}

    def _resolve_install_dir(self) -> "Path":
        """Resolve the profile-scoped directory for installed plugin code."""
        from pathlib import Path

        if self._plugin_install_dir:
            return Path(self._plugin_install_dir)
        from leapflow.config import get_settings

        settings = get_settings()
        profile_layout = getattr(settings, "profile_layout", None)
        if profile_layout is not None:
            return profile_layout.plugins_dir
        return Path(settings.layout.root) / "plugins"

    def _resolve_staging_dir(self) -> "Path":
        """Resolve the profile-owned quarantine directory for candidate code."""
        from pathlib import Path

        if self._plugin_staging_dir:
            return Path(self._plugin_staging_dir)
        if self._plugin_install_dir:
            return Path(self._plugin_install_dir) / ".staging"
        from leapflow.config import get_settings

        profile_layout = getattr(get_settings(), "profile_layout", None)
        if profile_layout is not None:
            return profile_layout.plugin_staging_dir
        return self._resolve_install_dir() / ".staging"

    @staticmethod
    def _sandbox_settings() -> dict[str, int | float]:
        """Return bounded sandbox configuration from the effective settings."""
        from leapflow.config import get_settings

        settings = get_settings()
        return {
            "invoke_timeout_s": max(
                0.1, float(getattr(settings, "plugin_sandbox_invoke_timeout_s", 15.0))
            ),
            "shutdown_timeout_s": max(
                0.1, float(getattr(settings, "plugin_sandbox_shutdown_timeout_s", 3.0))
            ),
            "cpu_time_s": max(
                0, int(getattr(settings, "plugin_sandbox_cpu_time_s", 30))
            ),
            "max_memory_bytes": max(
                0, int(getattr(settings, "plugin_sandbox_max_memory_mb", 0))
            )
            * 1024
            * 1024,
        }

    def _lifecycle_store(self) -> Any:
        """Resolve the profile-scoped acquisition-lifecycle ledger."""
        if self._capability_lifecycle_store is not None:
            return self._capability_lifecycle_store
        raise RuntimeError("capability lifecycle store was not injected by the runtime")

    def _open_lifecycle_record(self, proposal: Any, capability: str) -> str:
        """Open a PENDING lifecycle record correlated with a review proposal.

        This is what makes the trust/probation/quarantine tier reachable: without a
        lifecycle record there is nothing for ``AdaptiveEvolutionPolicy`` to decide
        about or for ``LifecycleGovernor`` to transition. Returns the lifecycle
        proposal id, or ``""`` when no ledger is available.

        Failures are contained: a bookkeeping write must never fail the proposal
        the caller actually asked for.
        """
        try:
            from leapflow.domain.capability_requirement import CapabilityRequirement

            requirement = CapabilityRequirement.create(
                capability or proposal.plugin_id,
                "explicit_request",
                evidence=proposal.capability_summary,
                max_risk_level=proposal.risk_level,
                requirement_id=f"req-review-{proposal.proposal_id}",
            )
            item = self._lifecycle_store().enqueue(
                requirements=[requirement],
                risk={"risk_level": proposal.risk_level},
                source="plugin_propose",
                metadata={
                    "plugin_id": proposal.plugin_id,
                    "review_proposal_id": proposal.proposal_id,
                    "review_proposal": proposal.to_dict(),
                },
            )
            self._trace_lifecycle_opened(item, proposal, requirement)
            return str(item.proposal_id)
        except (RuntimeError, OSError, ValueError, TypeError, AttributeError):
            logger.debug("self_management: lifecycle record not opened", exc_info=True)
            return ""

    @staticmethod
    def _trace_lifecycle_opened(item: Any, proposal: Any, requirement: Any) -> None:
        """Emit the one durable sign that the governance tier was actually driven.

        The queue records the item, but not what it was opened *for*: the link from a
        review proposal and a requirement to a lifecycle record lives only here. That
        link is what distinguishes "trust, probation and quarantine exist" from
        "something reached them" -- a distinction that mattered, because this
        machinery was for a long time unreachable in production and invisible while
        it was.
        """
        try:
            from leapflow.domain.evolution_trace import EvolutionStage
            from leapflow.telemetry.evolution_tap import emit_trace, is_enabled

            if not is_enabled():
                return
            emit_trace(
                EvolutionStage.DECIDE,
                "lifecycle_opened",
                correlation={
                    "lifecycle_proposal_id": str(getattr(item, "proposal_id", "")),
                    "review_proposal_id": str(getattr(proposal, "proposal_id", "")),
                    "requirement_id": str(getattr(requirement, "requirement_id", "")),
                    "plugin_id": str(getattr(proposal, "plugin_id", "")),
                },
                summary=(
                    f"lifecycle record opened for {getattr(proposal, 'plugin_id', '')}"
                ),
                detail={
                    "source": "plugin_propose",
                    "risk_level": str(getattr(proposal, "risk_level", "")),
                    "status": str(getattr(item, "status", "")),
                    "capability": str(getattr(requirement, "capability", "")),
                },
            )
        except Exception:  # noqa: BLE001 - bookkeeping must not fail the proposal
            logger.debug("self_management: evolution trace failed", exc_info=True)

    def _version_store(self) -> Any:
        """Resolve the profile-scoped plugin version store."""
        if self._plugin_version_store is not None:
            return self._plugin_version_store
        from leapflow.config import get_settings
        from leapflow.storage.plugin_version_store import PluginVersionStore

        settings = get_settings()
        profile_layout = getattr(settings, "profile_layout", None)
        if profile_layout is None:
            raise RuntimeError("profile_layout is required for plugin version storage")
        self._plugin_version_store = PluginVersionStore(profile_layout.plugin_versions_dir)
        return self._plugin_version_store

    def _capability_plan_store_resolved(self) -> Any:
        """Resolve the profile-scoped adaptive capability decision store."""
        if self._capability_plan_store is not None:
            return self._capability_plan_store
        from leapflow.config import get_settings
        from leapflow.storage.capability_plan_store import JsonCapabilityPlanStore

        settings = get_settings()
        profile_layout = getattr(settings, "profile_layout", None)
        if profile_layout is None:
            raise RuntimeError("profile_layout is required for capability plan storage")
        self._capability_plan_store = JsonCapabilityPlanStore(profile_layout.capability_plans_path)
        return self._capability_plan_store

    async def _install_from_code(
        self, plugin_id: str, code: str, *, proposal: Any = None, version_label: str = ""
    ) -> Dict[str, Any]:
        """Validate in quarantine, then atomically publish one DRAFT fiber."""
        import os
        import shutil
        import sys
        import tempfile

        from leapflow.learning.plugin_generator import PluginValidator
        from leapflow.plugins import get_scoped_registry

        validator = PluginValidator()
        vresult = await validator.validate(plugin_id, code)
        if not vresult.ok:
            return {
                "ok": False,
                "error": f"Code failed re-validation at stage '{vresult.stage}': {vresult.error}",
            }

        install_dir = self._resolve_install_dir()
        staging_root = self._resolve_staging_dir()
        install_dir.mkdir(parents=True, exist_ok=True)
        staging_root.mkdir(parents=True, exist_ok=True)
        staging_dir = Path(tempfile.mkdtemp(prefix=f"{plugin_id}-", dir=staging_root))
        staged_target = staging_dir / f"{plugin_id}.py"
        target = install_dir / f"{plugin_id}.py"
        previous_source = target.read_bytes() if target.exists() else None
        previous_module = sys.modules.get(plugin_id)
        scoped = get_scoped_registry()
        promoted = False
        try:
            staged_target.write_text(code, encoding="utf-8")
            test_cases = tuple(getattr(proposal, "test_cases", ()) or ())
            ok, error, observations = await self._sandbox_validate_candidate(
                plugin_id,
                staging_dir,
                test_cases=test_cases,
            )
            if not ok:
                return {"ok": False, "error": error, "behavior_tests": observations}

            new_plugin, load_err = self._load_from_path(plugin_id, staged_target)
            if new_plugin is None:
                return {"ok": False, "error": load_err}
            fiber = scoped.create_draft_fiber(plugin_id)
            scoped.stage_plugin(new_plugin, fiber)
            setattr(new_plugin, "__leapflow_plugin_path__", str(target))
            os.replace(staged_target, target)
            fiber = scoped.promote_draft(plugin_id)
            promoted = True

            version_info = self._version_store().record_source(
                plugin_id,
                target,
                version=version_label,
                metadata={
                    "source": "plugin_install",
                    "proposal_id": getattr(proposal, "proposal_id", ""),
                },
            )
            result: Dict[str, Any] = {
                "ok": True,
                "action": "install",
                "plugin_id": plugin_id,
                "installed_tools": [tool.name for tool in new_plugin.tools],
                "state": fiber.state.value,
                "shadow_validated": True,
                "behavior_tests": observations,
                "version": version_info.get("version", ""),
            }
            self._record_acquisition(plugin_id)
            self._trace_artifact_installed(plugin_id, proposal, version_info, result)
            return result
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as exc:
            if promoted:
                try:
                    scoped.dispose_plugin(plugin_id, prune_metadata=True)
                except (KeyError, RuntimeError, ValueError):
                    pass
            else:
                scoped.discard_draft(plugin_id)
            if previous_source is None:
                self._safe_unlink(target)
            else:
                target.write_bytes(previous_source)
            if previous_module is None:
                sys.modules.pop(plugin_id, None)
            else:
                sys.modules[plugin_id] = previous_module
            return {"ok": False, "error": f"Install transaction failed: {exc}"}
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)

    async def _emit_plugin_event(
        self,
        event_type: str,
        *,
        plugin_id: str,
        proposal_id: str = "",
        version_id: str = "",
        payload: Mapping[str, Any] | None = None,
        dedup_suffix: str,
    ) -> bool:
        """Persist a plugin mutation fact through the daemon-owned outbox."""
        outbox = self._evolution_outbox
        if outbox is None or not self._evolution_profile_id:
            return False
        from leapflow.domain.evolution_event import EvolutionContext, EvolutionEvent

        event = EvolutionEvent.create(
            event_type,
            context=EvolutionContext(
                profile_id=self._evolution_profile_id,
                proposal_id=proposal_id,
                plugin_id=plugin_id,
                version_id=version_id,
                correlation_id=proposal_id or plugin_id,
            ),
            payload=dict(payload or {}),
            producer="plugin.self_management",
            privacy_class="profile",
            dedup_key=f"{event_type}:{plugin_id}:{dedup_suffix}",
        )
        try:
            await outbox.publish(event, critical=True)
            return True
        except Exception:  # noqa: BLE001 - mutation already happened; report audit gap
            logger.error("plugin mutation event could not be persisted", exc_info=True)
            return False

    @staticmethod
    def _record_acquisition(plugin_id: str) -> None:
        """Note the acquisition for the cold-path co-evolution sweep; never raises."""
        try:
            from leapflow.evolution.observations import record_acquisition

            record_acquisition(plugin_id)
        except Exception:  # noqa: BLE001 - bookkeeping must not fail an install
            logger.debug("acquisition not recorded for %s", plugin_id, exc_info=True)

    @staticmethod
    def _trace_artifact_installed(
        plugin_id: str, proposal: Any, version_info: Any, result: Any
    ) -> None:
        """Record the artifact identity behind a completed acquisition.

        A capability transition has to be reconstructable end to end, and the piece no
        store held was the link from the causal proposal to the *artifact* that ended
        up registered. The version store knows the digest; the proposal knows why. This
        joins them so an installed plugin can always be traced back to the requirement
        that asked for it.
        """
        try:
            from leapflow.domain.evolution_trace import EvolutionStage
            from leapflow.telemetry.evolution_tap import emit_trace, is_enabled

            if not is_enabled():
                return
            info = dict(version_info or {})
            emit_trace(
                EvolutionStage.ACT,
                "artifact_installed",
                correlation={
                    "plugin_id": str(plugin_id),
                    "proposal_id": str(getattr(proposal, "proposal_id", "")),
                },
                summary=f"installed {plugin_id} v{info.get('version', '')}",
                detail={
                    "plugin_id": str(plugin_id),
                    "version": str(info.get("version", "")),
                    "digest": str(
                        info.get("checksum_sha256")
                        or info.get("sha256")
                        or info.get("digest")
                        or ""
                    ),
                    "capability": str(getattr(proposal, "capability", "")),
                    "installed_tools": list(dict(result or {}).get("installed_tools", ()) or ()),
                },
            )
        except Exception:  # noqa: BLE001 - telemetry must never fail an install
            logger.debug("artifact install trace failed", exc_info=True)

    def _resolve_dsh_install_dir(self) -> "Path":
        """Resolve the profile-owned directory for DSH source bundles."""
        from pathlib import Path

        from leapflow.config import get_settings

        if self._plugin_install_dir:
            return Path(self._plugin_install_dir) / "dsh"
        settings = get_settings()
        profile_layout = getattr(settings, "profile_layout", None)
        if profile_layout is not None:
            return profile_layout.dsh_plugins_dir
        raise RuntimeError("profile_layout is required for DSH plugin storage")

    async def _install_from_dsh_source(
        self,
        plugin_id: str,
        source_path: str,
        *,
        version_label: str = "",
        expected_bundle_sha256: str = "",
    ) -> Dict[str, Any]:
        """Install a real DSH bundle through restricted Node discovery."""
        import os
        import shutil
        import uuid

        from leapflow.config import get_settings
        from leapflow.learning.plugin_generator import PluginValidator
        from leapflow.plugins.dsh import prepare_dsh_installation
        from leapflow.plugins.dsh.bundle import promote_staging_bundle

        install_dir = self._resolve_install_dir()
        dsh_dir = self._resolve_dsh_install_dir()
        settings = get_settings()
        prepared = await prepare_dsh_installation(
            source_path,
            plugin_id=plugin_id,
            plugins_dir=install_dir,
            dsh_plugins_dir=dsh_dir,
            settings=settings,
        )
        if (
            expected_bundle_sha256
            and prepared.descriptor.bundle_sha256 != expected_bundle_sha256
        ):
            prepared.cleanup()
            return {
                "ok": False,
                "error": "DSH source changed after approval; installation was not attempted",
                "failure_code": "source_changed_after_approval",
            }
        promoted = False
        committed = False
        descriptor_path = prepared.final_root / "descriptor.json"
        try:
            install_dir.mkdir(parents=True, exist_ok=True)
            promote_staging_bundle(prepared.staging_root, prepared.final_root)
            promoted = True
            descriptor_temp = descriptor_path.with_name(
                f".{descriptor_path.name}.{uuid.uuid4().hex}.tmp"
            )
            try:
                descriptor_temp.write_text(prepared.descriptor.to_json(), encoding="utf-8")
                os.replace(descriptor_temp, descriptor_path)
            finally:
                descriptor_temp.unlink(missing_ok=True)

            validator = PluginValidator()
            validation = await validator.validate(
                prepared.plugin_id, prepared.wrapper_source
            )
            if not validation.ok:
                return {
                    "ok": False,
                    "error": (
                        f"DSH wrapper failed validation at stage '{validation.stage}': "
                        f"{validation.error}"
                    ),
                }

            temp_wrapper = prepared.wrapper_path.with_name(
                f".{prepared.wrapper_path.name}.{uuid.uuid4().hex}.tmp"
            )
            try:
                temp_wrapper.write_text(prepared.wrapper_source, encoding="utf-8")
                os.replace(temp_wrapper, prepared.wrapper_path)
            finally:
                temp_wrapper.unlink(missing_ok=True)

            result = self._register_inprocess(
                prepared.plugin_id, prepared.plugin_id, prepared.wrapper_path
            )
            if not result.get("ok"):
                return result
            try:
                version_store = self._version_store()
                version_info = version_store.record_bundle(
                    prepared.plugin_id,
                    prepared.wrapper_path,
                    prepared.final_root,
                    version=version_label,
                    metadata={
                        "source": "dsh_source",
                        "bundle_sha256": prepared.descriptor.bundle_sha256,
                        "source_kind": prepared.descriptor.source_kind,
                        "descriptor_path": str(descriptor_path),
                        "verdict": prepared.compatibility.final_verdict.value,
                        "limitations": list(prepared.descriptor.limitations),
                        "installed_tools": [
                            tool.name for tool in prepared.descriptor.tools
                        ],
                    },
                )
                result["version"] = version_info.get("version", "")
            except (RuntimeError, OSError, ValueError, AttributeError) as exc:
                logger.debug("DSH version recording skipped: %s", exc, exc_info=True)
            result.update(
                {
                    "source_kind": prepared.descriptor.source_kind,
                    "bundle_sha256": prepared.descriptor.bundle_sha256,
                    "descriptor_path": str(descriptor_path),
                    "installed_tools": [tool.name for tool in prepared.descriptor.tools],
                    "verdict": prepared.compatibility.final_verdict.value,
                    "limitations": list(prepared.descriptor.limitations),
                    "client_components": list(prepared.descriptor.client_components),
                }
            )
            committed = True
            return result
        finally:
            prepared.cleanup()
            if promoted and not committed:
                # Installation is a single transaction from the user's point of
                # view. A validation/registration failure must leave neither a
                # wrapper nor a managed bundle (and no partially registered fiber).
                try:
                    from leapflow.plugins import get_scoped_registry

                    scoped = get_scoped_registry()
                    if scoped.get_fiber(prepared.plugin_id) is not None:
                        scoped.dispose_plugin(prepared.plugin_id, prune_metadata=True)
                except (ImportError, KeyError, RuntimeError, AttributeError):
                    logger.debug(
                        "DSH install rollback found no live fiber for %s",
                        prepared.plugin_id,
                        exc_info=True,
                    )
                self._safe_unlink(prepared.wrapper_path)
                shutil.rmtree(prepared.final_root, ignore_errors=True)

    async def _install_from_marketplace_with_gate(
        self, plugin_id: str, marketplace_name: str
    ) -> Dict[str, Any]:
        """Install from marketplace with compatibility gate pre-check.

        Runs assess_plugin() on the resolved manifest before attempting install.
        If verdict is INCOMPATIBLE → returns structured error without install.
        If ADAPTABLE → includes adaptation_notes alongside the install result.
        """
        client = self._marketplace_client
        if client is None:
            return {
                "ok": False,
                "error": (
                    "Marketplace not configured "
                    "(set plugin_marketplace_root or plugin_marketplace_url)"
                ),
            }

        # Resolve manifest for compatibility check. Installation never proceeds
        # when the decision-bearing manifest is unavailable: a compatibility
        # gate that cannot run must not become an open door.
        try:
            manifest_data = client.resolve_manifest(marketplace_name)
        except (OSError, ValueError, RuntimeError, AttributeError) as exc:
            logger.warning(
                "Marketplace manifest resolution failed for %s: %s",
                marketplace_name,
                exc,
                exc_info=True,
            )
            return {
                "ok": False,
                "error": (
                    f"Compatibility manifest for '{marketplace_name}' could not be resolved; "
                    "installation was not attempted"
                ),
                "failure_code": "compatibility_manifest_unavailable",
            }
        if not isinstance(manifest_data, dict):
            return {
                "ok": False,
                "error": (
                    f"Compatibility manifest for '{marketplace_name}' is missing or invalid; "
                    "installation was not attempted"
                ),
                "failure_code": "compatibility_manifest_unavailable",
            }

        compatibility_notes: list[str] = []
        try:
            from leapflow.learning.compatibility import assess_plugin

            report = assess_plugin(manifest_data)
            if report.final_verdict.value == "incompatible":
                return {
                    "ok": False,
                    "error": (
                        f"Compatibility gate: plugin '{marketplace_name}' is not installable "
                        f"by the Python marketplace path. Reason: {report.rejection_reason}"
                    ),
                    "verdict": report.final_verdict.value,
                    "rejection_reason": report.rejection_reason,
                }
            if report.manifest.source_language.lower() in {"javascript", "typescript"}:
                return {
                    "ok": False,
                    "error": (
                        "DSH marketplace bundles are not supported in P0; install a local "
                        "pre-built source bundle with plugin_install(source_path=...)"
                    ),
                    "failure_code": "dsh_marketplace_unsupported",
                }
            if not report.is_installable():
                return {
                    "ok": False,
                    "error": (
                        f"Compatibility gate: plugin '{marketplace_name}' is not installable "
                        f"by the Python marketplace path. Reason: {report.rejection_reason or 'runtime compatibility not proven'}"
                    ),
                    "verdict": report.final_verdict.value,
                    "rejection_reason": report.rejection_reason,
                }
            if report.adaptation_notes:
                compatibility_notes = list(report.adaptation_notes)
        except (ImportError, AttributeError, TypeError, ValueError) as exc:
            logger.warning(
                "Compatibility gate failed closed for %s: %s",
                marketplace_name,
                exc,
                exc_info=True,
            )
            return {
                "ok": False,
                "error": (
                    f"Compatibility gate failed for '{marketplace_name}'; "
                    "installation was not attempted"
                ),
                "failure_code": "compatibility_gate_failed",
            }

        result = await self._install_from_marketplace(plugin_id, marketplace_name)
        if compatibility_notes and result.get("ok"):
            result["compatibility_notes"] = compatibility_notes
        return result

    async def _install_from_marketplace(
        self, plugin_id: str, marketplace_name: str
    ) -> Dict[str, Any]:
        """Install via the configured MarketplaceClient with verification + smoke test."""
        from pathlib import Path

        client = self._marketplace_client
        if client is None:
            return {
                "ok": False,
                "error": (
                    "Marketplace not configured "
                    "(set plugin_marketplace_root or plugin_marketplace_url)"
                ),
            }

        try:
            result = client.install(
                marketplace_name,
                verify=True,
                trusted_pubkeys=(self._trusted_pubkeys or None),
            )
        except (OSError, ValueError, RuntimeError) as exc:
            return {"ok": False, "error": f"Marketplace install failed: {exc}"}

        if not result.get("ok"):
            return {"ok": False, "error": result.get("error", "Marketplace install failed")}

        installed_path = Path(str(result["installed_path"]))
        module_name = installed_path.stem
        requires_sandbox = bool(result.get("requires_sandbox"))

        smoke_ok, smoke_err = await self._sandbox_smoke_test(module_name, installed_path.parent)
        if not smoke_ok:
            self._safe_unlink(installed_path)
            return {"ok": False, "error": smoke_err}

        if requires_sandbox:
            return await self._register_sandboxed(plugin_id, module_name, installed_path)
        return self._register_inprocess(plugin_id, module_name, installed_path)

    async def _sandbox_smoke_test(
        self, module_name: str, install_dir: "Path", *, timeout_s: float | None = None
    ) -> tuple[bool, str]:
        """Load a module in the bounded subprocess and invoke its first tool."""
        limits = self._sandbox_settings()
        if timeout_s is not None:
            limits["invoke_timeout_s"] = max(0.1, float(timeout_s))
        ok, error, _ = await self._sandbox_validate_candidate(
            module_name,
            install_dir,
            test_cases=(),
            limits=limits,
        )
        return ok, error

    async def _sandbox_validate_candidate(
        self,
        module_name: str,
        install_dir: "Path",
        *,
        test_cases: tuple[Any, ...],
        limits: dict[str, int | float] | None = None,
    ) -> tuple[bool, str, list[dict[str, Any]]]:
        """Run smoke and proposal behavior tests without publishing host handlers."""
        from leapflow.plugins.sandbox.sandbox_host import SandboxHost

        sandbox_limits = dict(limits or self._sandbox_settings())
        host = SandboxHost(
            module_name,
            python_paths=(str(install_dir),),
            **sandbox_limits,
        )
        started = False
        observations: list[dict[str, Any]] = []
        try:
            await host.start()
            started = True
        except (OSError, RuntimeError, ValueError) as exc:
            return False, f"Sandbox smoke test error: {exc}", observations
        if not started:
            return False, "Sandbox smoke test failed: worker did not start", observations

        try:
            if not await host.ping():
                return False, "Sandbox smoke test failed: worker did not respond", observations
            tool_names = await host.list_tools()
            if not tool_names:
                return (
                    False,
                    "Sandbox smoke test failed: plugin exposed no tools "
                    "(likely failed to import in isolation)",
                    observations,
                )
            smoke = await host.invoke(tool_names[0], {})
            if not smoke.ok and not smoke.error_type:
                return False, f"Sandbox smoke test failed: {smoke.error}", observations
            for index, case in enumerate(test_cases):
                tool_name = str(getattr(case, "tool_name", "") or "")
                if tool_name not in tool_names:
                    return (
                        False,
                        f"Behavior tests failed: behavior test {index}: "
                        f"tool {tool_name!r} not exposed",
                        observations,
                    )
                arguments = dict(getattr(case, "arguments", {}) or {})
                expected = dict(getattr(case, "expected_subset", {}) or {})
                response = await host.invoke(tool_name, arguments)
                if not response.ok:
                    return (
                        False,
                        f"Behavior tests failed: behavior test {index}: "
                        f"handler raised {response.error_type or 'SandboxError'}: {response.error}",
                        observations,
                    )
                observations.append(
                    {"tool_name": tool_name, "arguments": arguments, "result": response.result}
                )
                if not isinstance(response.result, dict):
                    return (
                        False,
                        f"Behavior tests failed: behavior test {index}: result is not a dict",
                        observations,
                    )
                for key, expected_value in expected.items():
                    if response.result.get(key) != expected_value:
                        return (
                            False,
                            f"Behavior tests failed: behavior test {index}: expected "
                            f"{key}={expected_value!r}, got {response.result.get(key)!r}",
                            observations,
                        )
            return True, "", observations
        finally:
            try:
                await host.stop()
            except (OSError, RuntimeError):
                pass

    def _register_inprocess(
        self, plugin_id: str, module_name: str, target: "Path"
    ) -> Dict[str, Any]:
        """Dynamically load the installed module and register it on the registry.

        On any failure the fiber is disposed, the module removed from
        ``sys.modules``, and the written file deleted — no partial state remains.
        """
        import sys

        from leapflow.plugins import get_scoped_registry

        new_plugin, load_err = self._load_from_path(module_name, target)
        if new_plugin is None:
            self._safe_unlink(target)
            return {"ok": False, "error": load_err}

        scoped = get_scoped_registry()
        fiber = scoped.create_draft_fiber(plugin_id)
        try:
            scoped.stage_plugin(new_plugin, fiber)
            fiber = scoped.promote_draft(plugin_id)
            installed_tools = [tool.name for tool in new_plugin.tools]
        except (RuntimeError, ValueError, AttributeError, TypeError) as exc:
            scoped.discard_draft(plugin_id)
            sys.modules.pop(module_name, None)
            self._safe_unlink(target)
            return {"ok": False, "error": f"Registration failed: {exc}"}

        return {
            "ok": True,
            "action": "install",
            "plugin_id": plugin_id,
            "installed_tools": installed_tools,
            "state": fiber.state.value,
        }

    async def _register_sandboxed(
        self, plugin_id: str, module_name: str, installed_path: "Path"
    ) -> Dict[str, Any]:
        """Register a marketplace plugin that must run isolated in a subprocess.

        The untrusted code is never imported in-process: tool names come from
        the sandbox worker and every handler proxies to it via
        SandboxedToolPlugin. The worker is stopped when the fiber is disposed.
        """
        from leapflow.plugins import get_scoped_registry
        from leapflow.plugins.protocol import ToolMetadata
        from leapflow.plugins.sandbox.sandbox_host import SandboxHost, SandboxedToolPlugin

        install_dir = installed_path.parent
        host = SandboxHost(
            module_name,
            python_paths=(str(install_dir),),
            **self._sandbox_settings(),
        )
        started = False
        try:
            await host.start()
            started = True
        except (OSError, RuntimeError, ValueError) as exc:
            return {"ok": False, "error": f"Sandbox start failed: {exc}"}
        if not started:
            return {"ok": False, "error": "Sandbox start failed"}

        tool_names = await host.list_tools()
        if not tool_names:
            await host.stop()
            self._safe_unlink(installed_path)
            return {"ok": False, "error": "Sandboxed plugin exposed no tools"}

        metadatas = [
            ToolMetadata(
                name=name,
                description=f"Sandboxed marketplace tool '{name}' from plugin '{plugin_id}'.",
                parameters_schema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": True,
                },
                handler=self._noop_handler,
                x_leapflow={"category": "marketplace", "risk_level": "high"},
                mutates_state=True,
            )
            for name in tool_names
        ]
        sandboxed = SandboxedToolPlugin(plugin_id, "marketplace", metadatas, host)

        scoped = get_scoped_registry()
        fiber = scoped.create_draft_fiber(plugin_id)
        try:
            scoped.stage_plugin(sandboxed, fiber)
            fiber.scope.async_effect(host.stop)
            fiber = scoped.promote_draft(plugin_id)
            installed_tools = [tool.name for tool in sandboxed.tools]
        except (RuntimeError, ValueError, AttributeError, TypeError) as exc:
            scoped.discard_draft(plugin_id)
            await host.stop()
            self._safe_unlink(installed_path)
            return {"ok": False, "error": f"Sandboxed registration failed: {exc}"}

        return {
            "ok": True,
            "action": "install",
            "plugin_id": plugin_id,
            "installed_tools": installed_tools,
            "state": fiber.state.value,
            "sandboxed": True,
        }

    @staticmethod
    async def _noop_handler(**kwargs: Any) -> Dict[str, Any]:
        """Placeholder handler replaced by SandboxedToolPlugin's proxy at wrap time."""
        return {"ok": False, "error": "handler not bound"}

    def _load_from_path(self, module_name: str, path: "Path") -> "tuple[Any, str]":
        """Load a plugin module from a file path and register it in sys.modules.

        Registering under ``module_name`` (which becomes the plugin class's
        ``__module__``) lets the scoped registry's reload() find it later via
        ``importlib.reload(sys.modules[module_name])`` — file-path modules keep
        a valid loader spec, so reload/enable work for installed plugins.

        Returns (plugin_obj, "") on success or (None, error) on failure.
        """
        import importlib.util
        import sys

        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                return None, f"Cannot create import spec for {path}"
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        except Exception as exc:  # noqa: BLE001 - importing installed plugin code can raise anything
            sys.modules.pop(module_name, None)
            return None, f"Failed to load installed module: {exc}"

        plugin_obj = getattr(module, "plugin", None)
        if plugin_obj is None:
            sys.modules.pop(module_name, None)
            return None, "Installed module has no 'plugin' attribute"
        try:
            setattr(plugin_obj, "__leapflow_plugin_path__", str(path))
            setattr(plugin_obj, "__leapflow_plugin_module__", module_name)
        except Exception:
            logger.debug(
                "Cannot attach plugin source path metadata for %s", module_name, exc_info=True
            )
        return plugin_obj, ""

    @staticmethod
    def _safe_unlink(path: "Path") -> None:
        """Remove a written plugin file, ignoring absence/IO errors."""
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    def _active_snapshot_path(self, plugin_id: str) -> "Path | None":
        """Return the active version snapshot path, if one is recorded."""
        try:
            active = self._version_store().active(plugin_id)
        except (RuntimeError, OSError, ValueError, AttributeError):
            return None
        if not isinstance(active, dict):
            return None
        raw_path = str(active.get("snapshot_path") or "")
        if not raw_path:
            return None
        path = Path(raw_path)
        return path if path.exists() else None

    def _active_proposal_tests(self, plugin_id: str) -> tuple[str, tuple[Any, ...], str]:
        """Return behavior tests linked to the plugin's active proposal, if any."""
        try:
            active = self._version_store().active(plugin_id)
        except (RuntimeError, OSError, ValueError, AttributeError) as exc:
            logger.debug(
                "Cannot read active plugin version for %s: %s", plugin_id, exc, exc_info=True
            )
            return "", (), ""
        if not isinstance(active, dict):
            return "", (), ""
        metadata = active.get("metadata")
        if not isinstance(metadata, dict):
            return "", (), ""
        proposal_id = str(metadata.get("proposal_id") or "")
        if not proposal_id:
            return "", (), ""
        try:
            store = self._lifecycle_store()
            lifecycle = store.get(proposal_id)
            if lifecycle is None:
                lifecycle = store.find_by_metadata("review_proposal_id", proposal_id)
        except (RuntimeError, OSError, ValueError, AttributeError) as exc:
            return (
                proposal_id,
                (),
                f"Plugin proposal '{proposal_id}' unavailable for behavior tests: {exc}",
            )
        if lifecycle is None:
            return proposal_id, (), f"Plugin proposal '{proposal_id}' not found for behavior tests"
        review_payload = dict(lifecycle.metadata or {}).get("review_proposal")
        if not isinstance(review_payload, Mapping):
            return proposal_id, (), ""
        from leapflow.domain.plugin_proposal import PluginProposal

        proposal = PluginProposal.from_dict(review_payload)
        return proposal_id, tuple(proposal.test_cases), ""

    async def _run_behavior_tests_for_plugin(
        self, plugin_id: str, test_cases: tuple[Any, ...]
    ) -> tuple[bool, str, list[dict[str, Any]]]:
        """Execute behavior tests against the currently registered plugin instance."""
        if not test_cases:
            return True, "", []
        from leapflow.learning.plugin_behavior_tests import run_plugin_behavior_tests
        from leapflow.plugins import get_registry

        plugin = get_registry().get_plugin(plugin_id)
        if plugin is None:
            return False, f"Plugin '{plugin_id}' is not registered for behavior tests", []
        return await run_plugin_behavior_tests(plugin, test_cases)

    def _restore_plugin_source(
        self,
        plugin_id: str,
        source_path: "Path | None",
        snapshot_path: "Path | None",
    ) -> str:
        """Restore a previous source snapshot and reload it; return an error string on failure."""
        if source_path is None or snapshot_path is None:
            return "no previous source snapshot is available"
        try:
            source_path.parent.mkdir(parents=True, exist_ok=True)
            source_path.write_bytes(snapshot_path.read_bytes())
            from leapflow.plugins import reload_plugin

            reload_plugin(plugin_id)
            return ""
        except (OSError, RuntimeError, KeyError, AttributeError) as exc:
            logger.warning(
                "plugin rollback after failed behavior tests failed: %s", exc, exc_info=True
            )
            return str(exc)

    async def _plugin_versions_handler(self, plugin_id: str, **kwargs: Any) -> Dict[str, Any]:
        """List recorded versions and the active pointer for a profile plugin."""
        try:
            store = self._version_store()
            return {
                "ok": True,
                "plugin_id": plugin_id,
                "active": store.active(plugin_id),
                "versions": store.versions(plugin_id),
            }
        except (RuntimeError, OSError, ValueError, AttributeError) as exc:
            return {"ok": False, "error": f"Version query failed: {exc}"}

    async def _plugin_rollback_handler(
        self, plugin_id: str, version: str, **kwargs: Any
    ) -> Dict[str, Any]:
        """Rollback a profile plugin to a recorded source snapshot and reload it."""
        try:
            from leapflow.plugins.dsh import normalize_plugin_id

            is_dsh = (
                normalize_plugin_id(plugin_id) == plugin_id
                and (self._resolve_dsh_install_dir() / plugin_id).is_dir()
            )
        except (ImportError, ValueError):
            is_dsh = False
        if is_dsh:
            return await self._plugin_rollback_dsh(
                plugin_id, version, **kwargs
            )
        approved, denial = await self._check_approval("rollback", plugin_id)
        if not approved:
            return {"ok": False, "error": denial, "requires_approval": True}
        from leapflow.plugins import reload_plugin

        target = self._resolve_install_dir() / f"{plugin_id}.py"
        version_store = self._version_store()
        metadata_snapshot = version_store.snapshot_state(plugin_id)
        source_snapshot = target.read_bytes() if target.exists() else None
        try:
            entry = version_store.rollback(plugin_id, version, target)
            fiber = reload_plugin(plugin_id)
            response = {
                "ok": True,
                "action": "rollback",
                "plugin_id": plugin_id,
                "version": entry.get("version", version),
                "state": fiber.state.value,
                "new_generation": fiber.generation,
            }
            from leapflow.domain.event_types import EvolutionEventType

            persisted = await self._emit_plugin_event(
                EvolutionEventType.PLUGIN_ROLLED_BACK,
                plugin_id=plugin_id,
                version_id=str(entry.get("version") or version),
                payload=response,
                dedup_suffix=f"{entry.get('version', version)}:{fiber.generation}",
            )
            if not persisted:
                response["audit_incomplete"] = True
            return response
        except (KeyError, RuntimeError, OSError, AttributeError) as exc:
            restoration_error = ""
            try:
                version_store.restore_source(target, source_snapshot)
                version_store.restore_state(plugin_id, metadata_snapshot)
                reload_plugin(plugin_id)
            except (KeyError, RuntimeError, OSError, AttributeError) as restore_exc:
                restoration_error = str(restore_exc)
                logger.error(
                    "plugin_rollback could not restore the previous runtime: %s",
                    restore_exc,
                    exc_info=True,
                )
            logger.warning("plugin_rollback failed: %s", exc, exc_info=True)
            response = {
                "ok": False,
                "error": f"Rollback failed: {exc}",
                "rolled_back": restoration_error == "",
            }
            if restoration_error:
                response["rollback_error"] = restoration_error
            return response

    async def _plugin_rollback_dsh(
        self, plugin_id: str, version: str, **kwargs: Any
    ) -> Dict[str, Any]:
        """Rollback a DSH bundle plugin to a recorded directory-level snapshot."""
        approved, denial = await self._check_approval("rollback", plugin_id)
        if not approved:
            return {"ok": False, "error": denial, "requires_approval": True}

        from leapflow.plugins import reload_plugin

        version_store = self._version_store()
        wrapper_target = self._resolve_install_dir() / f"{plugin_id}.py"
        bundle_target = self._resolve_dsh_install_dir() / plugin_id
        metadata_snapshot = version_store.snapshot_state(plugin_id)
        wrapper_snapshot = wrapper_target.read_bytes() if wrapper_target.exists() else None
        try:
            _result = version_store.rollback_bundle(
                plugin_id, version, wrapper_target, bundle_target
            )
            fiber = reload_plugin(plugin_id)
            response: Dict[str, Any] = {
                "ok": True,
                "action": "rollback",
                "plugin_id": plugin_id,
                "version": version,
                "bundle": True,
                "state": fiber.state.value,
                "new_generation": fiber.generation,
            }
            from leapflow.domain.event_types import EvolutionEventType

            persisted = await self._emit_plugin_event(
                EvolutionEventType.PLUGIN_ROLLED_BACK,
                plugin_id=plugin_id,
                version_id=version,
                payload=response,
                dedup_suffix=f"{version}:{fiber.generation}",
            )
            if not persisted:
                response["audit_incomplete"] = True
            return response
        except (KeyError, RuntimeError, OSError, AttributeError, FileNotFoundError) as exc:
            restoration_error = ""
            try:
                version_store.restore_source(wrapper_target, wrapper_snapshot)
                version_store.restore_state(plugin_id, metadata_snapshot)
                reload_plugin(plugin_id)
            except (KeyError, RuntimeError, OSError, AttributeError) as restore_exc:
                restoration_error = str(restore_exc)
                logger.error(
                    "DSH bundle rollback could not restore the previous runtime: %s",
                    restore_exc,
                    exc_info=True,
                )
            logger.warning("DSH bundle rollback failed: %s", exc, exc_info=True)
            response = {
                "ok": False,
                "error": f"DSH bundle rollback failed: {exc}",
                "rolled_back": restoration_error == "",
            }
            if restoration_error:
                response["rollback_error"] = restoration_error
            return response

    async def _plugin_enable_handler(self, plugin_id: str, **kwargs: Any) -> Dict[str, Any]:
        """Re-enable a previously disabled plugin. REQUIRES approval.

        This calls reload_plugin internally, which re-imports the module
        and registers a fresh instance with a new fiber.
        """
        if plugin_id == "self_management":
            return {"ok": False, "error": "Cannot enable self_management (already active)"}

        approved, denial = await self._check_approval("enable", plugin_id)
        if not approved:
            return {"ok": False, "error": denial, "requires_approval": True}

        try:
            from leapflow.plugins import reload_plugin

            new_fiber = reload_plugin(plugin_id)
            return {
                "ok": True,
                "action": "enable",
                "plugin_id": plugin_id,
                "new_generation": new_fiber.generation,
                "state": new_fiber.state.value,
            }
        except KeyError:
            return {"ok": False, "error": f"Plugin '{plugin_id}' not found in scoped registry"}
        except RuntimeError as exc:
            return {"ok": False, "error": f"Enable failed: {exc}"}

    async def _check_approval(
        self,
        action: str,
        plugin_id: str,
        *,
        proposal_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> tuple[bool, str]:
        """Consult the plugin approval gate. Returns (approved, denial_message).

        Progressive Trust: PRODUCTION-level plugins get auto-approved for
        'reload' (which is idempotent). 'disable' and 'enable' always require
        human approval regardless of trust level.
        """
        # Progressive Trust: auto-approve reload for PRODUCTION-level plugins
        if action == "reload":
            try:
                from leapflow.learning.plugin_advisor import get_default_advisor

                advisor = get_default_advisor()
                if advisor is not None:
                    trust = advisor._trust_ledger.level(plugin_id)
                    if trust.name == "PRODUCTION":
                        logger.info(
                            "Auto-approving '%s' on plugin '%s' (trust: PRODUCTION)",
                            action,
                            plugin_id,
                        )
                        return True, ""
            except (ImportError, AttributeError, RuntimeError):
                pass  # Learning not wired — fall through to gate

        # Standard gate check
        if self._plugin_approval_gate is None:
            # No gate installed: for safety, deny mutation
            return False, (
                f"Plugin action '{action}' on '{plugin_id}' blocked: "
                "no approval gate configured. Configure a plugin_approval_gate "
                "in the daemon approval coordinator to enable self-modification."
            )
        try:
            from leapflow.security.actions import ActionDescriptor

            descriptor = ActionDescriptor.platform_action(
                "plugin_management",
                action,
                {"plugin_id": plugin_id},
                metadata={
                    "effect": "write",
                    "risk_level": "high",
                    "category": "self_modification",
                    "proposal_id": proposal_id,
                    **(metadata or {}),
                },
            )
            result = await self._plugin_approval_gate.evaluate(descriptor)
            if getattr(result, "approved", False):
                return True, ""
            message = str(
                getattr(result, "denial_message", "")
                or f"Plugin action '{action}' on '{plugin_id}' requires approval (denied)"
            )
            return False, message
        except (ImportError, AttributeError, RuntimeError) as exc:
            logger.warning("approval check failed: %s", exc, exc_info=True)
            return False, f"Plugin action '{action}' blocked: approval check error"

    async def _plugin_reload_handler(
        self, plugin_id: str, version_label: str = "", **kwargs: Any
    ) -> Dict[str, Any]:
        """Hot-reload a plugin. REQUIRES approval."""
        approved, denial = await self._check_approval("reload", plugin_id)
        if not approved:
            return {"ok": False, "error": denial, "requires_approval": True}

        try:
            from leapflow.plugins import get_scoped_registry, reload_plugin

            scoped = get_scoped_registry()
            source_path = scoped.get_plugin_file(plugin_id)
            previous_snapshot = self._active_snapshot_path(plugin_id)
            proposal_id, test_cases, test_error = self._active_proposal_tests(plugin_id)
            if test_error:
                return {"ok": False, "error": test_error}

            new_fiber = reload_plugin(plugin_id)
            behavior_observations: list[dict[str, Any]] = []
            if test_cases:
                ok, error, behavior_observations = await self._run_behavior_tests_for_plugin(
                    plugin_id, test_cases
                )
                if not ok:
                    restore_error = self._restore_plugin_source(
                        plugin_id, source_path, previous_snapshot
                    )
                    response: Dict[str, Any] = {
                        "ok": False,
                        "error": f"Behavior tests failed: {error}",
                        "plugin_id": plugin_id,
                        "proposal_id": proposal_id,
                        "behavior_tests": behavior_observations,
                        "rolled_back": restore_error == "",
                    }
                    if restore_error:
                        response["rollback_error"] = restore_error
                    return response

            version = ""
            if version_label:
                source_path = scoped.get_plugin_file(plugin_id)
                if source_path is not None:
                    version_info = self._version_store().record_source(
                        plugin_id,
                        source_path,
                        version=version_label,
                        metadata={"source": "plugin_reload", "proposal_id": proposal_id},
                    )
                    version = str(version_info.get("version") or "")
            response = {
                "ok": True,
                "action": "reload",
                "plugin_id": plugin_id,
                "new_generation": new_fiber.generation,
                "state": new_fiber.state.value,
                "version": version,
            }
            if behavior_observations:
                response["proposal_id"] = proposal_id
                response["behavior_tests"] = behavior_observations
            return response
        except KeyError:
            return {"ok": False, "error": f"Plugin '{plugin_id}' not scoped-registered"}
        except RuntimeError as exc:
            return {"ok": False, "error": f"Reload failed: {exc}"}

    async def _plugin_unquarantine_handler(self, plugin_id: str, **kwargs: Any) -> Dict[str, Any]:
        """Restore a quarantined plugin to probation for re-evaluation."""
        if plugin_id == "self_management":
            return {"ok": False, "error": "Cannot unquarantine self_management (not quarantined)"}

        # Approval gate — forces HIGH risk via platform="plugin_management"
        approved, denial = await self._check_approval(
            "unquarantine", plugin_id,
            metadata={"platform": "plugin_management"},
        )
        if not approved:
            return {"ok": False, "error": denial, "requires_approval": True}

        # Confirm the plugin is actually in QUARANTINED status via proposal store
        lifecycle_store = self._capability_lifecycle_store
        proposal = None
        quarantine_reason = ""
        if lifecycle_store is not None:
            try:
                for item in lifecycle_store.list_items(status="QUARANTINED", limit=0):
                    pid = str(item.metadata.get("plugin_id") or "")
                    if pid == plugin_id:
                        proposal = item
                        quarantine_reason = str(item.metadata.get("terminal_reason") or "")
                        break
            except (AttributeError, RuntimeError) as exc:
                logger.warning("unquarantine: proposal lookup failed: %s", exc)

        if proposal is None:
            return {
                "ok": False,
                "error": f"Plugin '{plugin_id}' is not in QUARANTINED status",
            }

        # Unfreeze the trust ledger — resets to DRAFT
        unfrozen = False
        try:
            from leapflow.learning.plugin_advisor import get_default_advisor

            advisor = get_default_advisor()
            if advisor is not None:
                unfrozen = advisor._trust_ledger.unfreeze(plugin_id)
        except (ImportError, AttributeError, RuntimeError) as exc:
            logger.warning("unquarantine: trust unfreeze failed: %s", exc)

        # Transition proposal: QUARANTINED -> PROBATION
        try:
            lifecycle_store.transition(
                proposal.proposal_id,
                "PROBATION",
                metadata={"unquarantine_reason": "manual_recovery"},
            )
        except (ValueError, KeyError, RuntimeError) as exc:
            return {
                "ok": False,
                "error": f"Proposal transition failed: {exc}",
                "trust_unfrozen": unfrozen,
            }

        # Reload the plugin fiber
        reload_ok = False
        reload_error = ""
        try:
            from leapflow.plugins import reload_plugin

            reload_plugin(plugin_id)
            reload_ok = True
        except (KeyError, RuntimeError, ImportError) as exc:
            reload_error = str(exc)
            logger.warning("unquarantine: reload failed: %s", exc)

        # Emit PLUGIN_UNQUARANTINED event
        try:
            from leapflow.domain.event_types import EvolutionEventType

            await self._emit_plugin_event(
                EvolutionEventType.PLUGIN_UNQUARANTINED,
                plugin_id=plugin_id,
                proposal_id=proposal.proposal_id,
                payload={
                    "plugin_id": plugin_id,
                    "original_quarantine_reason": quarantine_reason,
                    "trust_unfrozen": unfrozen,
                    "reload_ok": reload_ok,
                },
                dedup_suffix=f"{proposal.proposal_id}:unquarantine",
            )
        except Exception:  # noqa: BLE001
            logger.debug("unquarantine event emission failed", exc_info=True)

        response: Dict[str, Any] = {
            "ok": True,
            "plugin_id": plugin_id,
            "status": "PROBATION",
            "trust": "DRAFT",
            "trust_unfrozen": unfrozen,
            "reload_ok": reload_ok,
        }
        if reload_error:
            response["reload_error"] = reload_error
        return response

    async def _plugin_disable_handler(self, plugin_id: str, **kwargs: Any) -> Dict[str, Any]:
        """Disable a plugin by disposing its fiber. REQUIRES approval.

        Note: this removes the plugin's tools from the registry until process restart
        or explicit re-enable (not yet implemented).
        """
        # Protect against self-destruction
        if plugin_id == "self_management":
            return {
                "ok": False,
                "error": "Cannot disable self_management plugin (would remove this tool)",
            }

        approved, denial = await self._check_approval("disable", plugin_id)
        if not approved:
            return {"ok": False, "error": denial, "requires_approval": True}

        try:
            from leapflow.plugins import get_scoped_registry

            scoped = get_scoped_registry()
            fiber = scoped.dispose_plugin(plugin_id)

            return {
                "ok": True,
                "action": "disable",
                "plugin_id": plugin_id,
                "state": fiber.state.value,
            }
        except KeyError as exc:
            return {"ok": False, "error": str(exc)}
        except (RuntimeError, AttributeError) as exc:
            logger.warning("plugin_disable failed: %s", exc, exc_info=True)
            return {"ok": False, "error": f"Disable failed: {exc}"}

    async def _plugin_remove_handler(
        self, plugin_id: str, delete_source: bool = True, **kwargs: Any
    ) -> Dict[str, Any]:
        """Terminally remove a plugin: dispose fiber, unregister tools, delete source."""
        if plugin_id == "self_management":
            return {
                "ok": False,
                "error": "Cannot remove self_management plugin (would remove this tool)",
            }

        approved, denial = await self._check_approval("remove", plugin_id)
        if not approved:
            return {"ok": False, "error": denial, "requires_approval": True}

        try:
            import sys

            from leapflow.plugins import get_registry, get_scoped_registry

            scoped = get_scoped_registry()
            source_path = scoped.get_plugin_file(plugin_id)
            module_path = scoped.get_plugin_module(plugin_id)
            plugin = get_registry().get_plugin(plugin_id)
            dsh_bundle = None
            try:
                from leapflow.plugins.dsh import normalize_plugin_id

                if normalize_plugin_id(plugin_id) == plugin_id:
                    dsh_bundle = self._resolve_dsh_install_dir().resolve() / plugin_id
            except (ImportError, ValueError):
                pass
            descriptor = getattr(plugin, "descriptor", None) if plugin is not None else None
            if descriptor is not None:
                raw_root = str(getattr(descriptor, "bundle_root", "") or "")
                if raw_root:
                    candidate = Path(raw_root).expanduser().resolve()
                    managed_root = self._resolve_dsh_install_dir().resolve()
                    try:
                        candidate.relative_to(managed_root)
                    except ValueError:
                        candidate = None
                    if candidate is not None:
                        dsh_bundle = candidate
            fiber = scoped.dispose_plugin(plugin_id, prune_metadata=True)
            if module_path:
                sys.modules.pop(module_path, None)
            source_deleted = False
            if delete_source:
                import shutil

                target = source_path or (self._resolve_install_dir() / f"{plugin_id}.py")
                if target.exists():
                    target.unlink()
                    source_deleted = True
                if dsh_bundle is not None and dsh_bundle.is_dir():
                    shutil.rmtree(dsh_bundle)
                    source_deleted = True
            return {
                "ok": True,
                "action": "remove",
                "plugin_id": plugin_id,
                "state": fiber.state.value,
                "source_path": str(source_path or ""),
                "source_deleted": source_deleted,
            }
        except KeyError as exc:
            return {"ok": False, "error": str(exc)}
        except (RuntimeError, AttributeError, OSError) as exc:
            logger.warning("plugin_remove failed: %s", exc, exc_info=True)
            return {"ok": False, "error": f"Remove failed: {exc}"}

    # ── Tool metadata ──────────────────────────────────────

    @property
    def tools(self) -> list[ToolMetadata]:
        return [
            ToolMetadata(
                name="plugin_list",
                description=(
                    "List the live plugin registry and cross-subsystem capability evidence. "
                    "Use this before answering questions about whether LeapFlow supports plugins, "
                    "self-evolution, plugin installation, hot reload, versioning, or other runtime capabilities."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
                handler=self._plugin_list_handler,
                x_leapflow={
                    "category": "system",
                    "risk_level": "read_only",
                    "schema_cost": "low",
                    "requires_approval": False,
                    "summary": "list live plugins and self capability evidence",
                },
                provides_capabilities=("plugin.list",),
            ),
            ToolMetadata(
                name="plugin_status",
                description=(
                    "Get detailed status of a specific plugin: its declared category, "
                    "runtime dependencies, contributed tools, and fiber lifecycle state."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "plugin_id": {
                            "type": "string",
                            "description": "The plugin identifier (e.g. 'file_ops', 'web_access').",
                        },
                    },
                    "required": ["plugin_id"],
                },
                handler=self._plugin_status_handler,
                x_leapflow={
                    "category": "system",
                    "risk_level": "read_only",
                    "schema_cost": "low",
                    "requires_approval": False,
                    "summary": "inspect one plugin's details",
                },
                provides_capabilities=("plugin.status",),
            ),
            ToolMetadata(
                name="plugin_versions",
                description="List recorded source versions and active pointer for a profile-scoped plugin.",
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "plugin_id": {
                            "type": "string",
                            "description": "The plugin identifier to inspect.",
                        },
                    },
                    "required": ["plugin_id"],
                },
                handler=self._plugin_versions_handler,
                x_leapflow={
                    "category": "system",
                    "risk_level": "read_only",
                    "schema_cost": "low",
                    "requires_approval": False,
                    "summary": "list plugin source versions",
                },
                provides_capabilities=("plugin.versions",),
            ),
            ToolMetadata(
                name="plugin_propose",
                description=(
                    "Create a side-effect-free PluginProposal from explicit capability-gap evidence. "
                    "Use this before plugin_generate when a missing capability should be reviewed. "
                    "Does not call an LLM, write files, or install anything."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "requested_capability": {
                            "type": "string",
                            "description": "Capability the plugin should provide.",
                        },
                        "plugin_id": {
                            "type": "string",
                            "description": "Optional proposed plugin id; auto-derived when omitted.",
                        },
                        "proposed_tools": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional proposed tool names.",
                        },
                        "test_cases": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "Optional behavior tests: {tool_name, arguments, expected_subset}.",
                        },
                        "risk_level": {
                            "type": "string",
                            "enum": ["read_only", "low", "medium", "high", "mutating", "external"],
                            "description": "Risk classification for the proposed plugin.",
                        },
                        "evidence": {
                            "type": "object",
                            "description": "Optional structured evidence such as an unknown_tool result.",
                        },
                    },
                    "required": ["requested_capability"],
                },
                handler=self._plugin_propose_handler,
                x_leapflow={
                    "category": "system",
                    "risk_level": "read_only",
                    "schema_cost": "medium",
                    "requires_approval": False,
                    "effect_scope": "none",
                    "idempotency_scope": "turn",
                    "summary": "create a reviewable plugin proposal without side effects",
                },
                provides_capabilities=("plugin.propose",),
            ),
            ToolMetadata(
                name="assess_compatibility",
                description=(
                    "Assess whether a foreign plugin manifest or real DSH source bundle "
                    "is compatible with LeapFlow. A source bundle assessment is static: "
                    "runtime_ready stays false until restricted Node discovery during "
                    "plugin_install. Returns component-level verdicts and limitations."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "manifest": {
                            "type": "object",
                            "description": "Plugin manifest to assess (LeapFlow or package.json-like DSH format). Mutually exclusive with source_path.",
                        },
                        "source_path": {
                            "type": "string",
                            "description": "Path to a DSH package directory or dynamic Cordis export (meta.json + host.js). Mutually exclusive with manifest.",
                        },
                    },
                    "required": [],
                },
                handler=self._assess_compatibility_handler,
                x_leapflow={
                    "category": "plugin_management",
                    "risk_level": "none",
                    "schema_cost": "low",
                    "requires_approval": False,
                    "effect": "read",
                    "summary": "assess foreign plugin manifest compatibility",
                },
                mutates_state=False,
                provides_capabilities=("plugin.compatibility_check",),
            ),
            ToolMetadata(
                name="plugin_generate",
                description=(
                    "Generate a new ToolPlugin from a natural-language capability "
                    "description. The LLM produces code that conforms to the "
                    "ToolPlugin Protocol; it is then rigorously validated "
                    "(syntax, structure, import, protocol conformance). The "
                    "generated source is stored in CAS and receives explicit content "
                    "approval. It DOES NOT install the plugin — installation requires "
                    "a second, mutation-specific approval via plugin_install."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "plugin_id": {
                            "type": "string",
                            "description": "Identifier for the new plugin and profile-scoped module filename.",
                        },
                        "description": {
                            "type": "string",
                            "description": "Natural-language description of the capability the plugin should provide.",
                        },
                        "proposal_id": {
                            "type": "string",
                            "description": "Optional lifecycle proposal id or review alias; fills plugin_id/description when omitted.",
                        },
                    },
                    "required": [],
                },
                handler=self._plugin_generate_handler,
                x_leapflow={
                    "category": "system",
                    "risk_level": "medium",
                    "schema_cost": "medium",
                    "requires_approval": False,
                    "effect_scope": "none",
                    "idempotency_scope": "turn",
                    "summary": "generate, validate, persist, and approve proposal content",
                },
                provides_capabilities=("plugin.generate",),
            ),
            ToolMetadata(
                name="plugin_install",
                description=(
                    "Install a plugin from exactly one source: validated Python code, "
                    "the configured Python marketplace, or a real DSH source bundle. "
                    "DSH bundles run restricted Node discovery before registration; "
                    "only public host tools are installed and client UI limitations are "
                    "reported. Writes profile-scoped state and mutates the process-global "
                    "registry. REQUIRES APPROVAL."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "plugin_id": {
                            "type": "string",
                            "description": "Identifier of the plugin to install.",
                        },
                        "code": {
                            "type": "string",
                            "description": "Validated plugin source code (typically from plugin_generate). Mutually exclusive with marketplace_name.",
                        },
                        "marketplace_name": {
                            "type": "string",
                            "description": "Python marketplace entry name. Mutually exclusive with code and source_path.",
                        },
                        "source_path": {
                            "type": "string",
                            "description": "Local DSH package or dynamic Cordis export directory. Mutually exclusive with code and marketplace_name.",
                        },
                        "proposal_id": {
                            "type": "string",
                            "description": "Optional lifecycle proposal id or review alias; requires prior content approval.",
                        },
                        "version_label": {
                            "type": "string",
                            "description": "Optional version id to record for code installs.",
                        },
                    },
                    "required": [],
                },
                handler=self._plugin_install_handler,
                x_leapflow={
                    "category": "system",
                    "risk_level": "high",
                    "schema_cost": "medium",
                    "requires_approval": True,
                    "effect_scope": "persistent",
                    "idempotency_scope": "session",
                    "summary": "install a Python or restricted DSH plugin (approval required)",
                },
                mutates_state=True,
                provides_capabilities=("plugin.install",),
                requires_platform_capabilities=("file.ops",),
            ),
            ToolMetadata(
                name="plugin_rollback",
                description=(
                    "Rollback a profile-scoped plugin to a recorded source version and reload it. "
                    "REQUIRES APPROVAL."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "plugin_id": {
                            "type": "string",
                            "description": "The plugin identifier to rollback.",
                        },
                        "version": {
                            "type": "string",
                            "description": "Recorded version id to restore.",
                        },
                    },
                    "required": ["plugin_id", "version"],
                },
                handler=self._plugin_rollback_handler,
                x_leapflow={
                    "category": "system",
                    "risk_level": "high",
                    "schema_cost": "medium",
                    "requires_approval": True,
                    "effect_scope": "persistent",
                    "idempotency_scope": "session",
                    "summary": "rollback a plugin to a recorded version (approval required)",
                },
                mutates_state=True,
                provides_capabilities=("plugin.rollback",),
                requires_platform_capabilities=("file.ops",),
            ),
            ToolMetadata(
                name="plugin_reload",
                description=(
                    "Hot-reload a plugin at runtime. Disposes the old plugin fiber, "
                    "re-imports its module, and registers a fresh instance. Existing "
                    "in-flight turns are unaffected (snapshot isolation). "
                    "REQUIRES APPROVAL — this is a self-modification action."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "plugin_id": {
                            "type": "string",
                            "description": "The plugin identifier to reload.",
                        },
                        "version_label": {
                            "type": "string",
                            "description": "Optional version id to record after reload.",
                        },
                    },
                    "required": ["plugin_id"],
                },
                handler=self._plugin_reload_handler,
                x_leapflow={
                    "category": "system",
                    "risk_level": "high",
                    "schema_cost": "medium",
                    "requires_approval": True,
                    "effect_scope": "local",
                    "idempotency_scope": "turn",
                    "summary": "hot-reload a plugin (approval required)",
                },
                mutates_state=True,
                provides_capabilities=("plugin.reload",),
                requires_platform_capabilities=("file.ops",),
            ),
            ToolMetadata(
                name="plugin_disable",
                description=(
                    "Disable a plugin by disposing its fiber, removing its tools "
                    "from the runtime registry. Cannot disable self_management itself. "
                    "REQUIRES APPROVAL — this is a self-modification action."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "plugin_id": {
                            "type": "string",
                            "description": "The plugin identifier to disable.",
                        },
                    },
                    "required": ["plugin_id"],
                },
                handler=self._plugin_disable_handler,
                x_leapflow={
                    "category": "system",
                    "risk_level": "high",
                    "schema_cost": "medium",
                    "requires_approval": True,
                    "effect_scope": "local",
                    "idempotency_scope": "session",
                    "summary": "disable a plugin (approval required)",
                },
                mutates_state=True,
                provides_capabilities=("plugin.disable",),
                requires_capabilities=("plugin.list",),
                requires_platform_capabilities=("file.ops",),
            ),
            ToolMetadata(
                name="plugin_remove",
                description=(
                    "Terminally remove a plugin: dispose its fiber, unregister its tools, "
                    "remove reload metadata, and optionally delete its profile-scoped source file. "
                    "Cannot remove self_management itself. REQUIRES APPROVAL."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "plugin_id": {
                            "type": "string",
                            "description": "The plugin identifier to remove.",
                        },
                        "delete_source": {
                            "type": "boolean",
                            "description": "Delete the profile-scoped source file as part of removal (default true).",
                        },
                    },
                    "required": ["plugin_id"],
                },
                handler=self._plugin_remove_handler,
                x_leapflow={
                    "category": "system",
                    "risk_level": "high",
                    "schema_cost": "medium",
                    "requires_approval": True,
                    "effect_scope": "persistent",
                    "idempotency_scope": "session",
                    "summary": "remove a plugin and optionally delete its source (approval required)",
                },
                mutates_state=True,
                provides_capabilities=("plugin.remove",),
                requires_platform_capabilities=("file.ops",),
            ),
            ToolMetadata(
                name="plugin_unquarantine",
                description=(
                    "Restore a quarantined plugin to probation status for re-evaluation. "
                    "Unfreezes the trust ledger, transitions the proposal back to PROBATION, "
                    "and reloads the plugin. REQUIRES APPROVAL."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "plugin_id": {
                            "type": "string",
                            "description": "The plugin identifier to unquarantine.",
                        },
                    },
                    "required": ["plugin_id"],
                },
                handler=self._plugin_unquarantine_handler,
                x_leapflow={
                    "category": "plugin_management",
                    "risk_level": "high",
                    "schema_cost": "medium",
                    "requires_approval": True,
                    "effect_scope": "local",
                    "idempotency_scope": "session",
                    "summary": "restore a quarantined plugin to probation (approval required)",
                },
                mutates_state=True,
                provides_capabilities=("plugin.unquarantine",),
                requires_platform_capabilities=("file.ops",),
            ),
            ToolMetadata(
                name="plugin_enable",
                description=(
                    "Re-enable a previously disabled plugin by reloading its module "
                    "and registering a fresh instance. REQUIRES APPROVAL."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "plugin_id": {
                            "type": "string",
                            "description": "The plugin identifier to re-enable.",
                        },
                    },
                    "required": ["plugin_id"],
                },
                handler=self._plugin_enable_handler,
                x_leapflow={
                    "category": "system",
                    "risk_level": "high",
                    "schema_cost": "medium",
                    "requires_approval": True,
                    "effect_scope": "local",
                    "idempotency_scope": "turn",
                    "summary": "re-enable a disabled plugin (approval required)",
                },
                mutates_state=True,
                provides_capabilities=("plugin.enable",),
                requires_platform_capabilities=("file.ops",),
            ),
        ]


plugin = SelfManagementPlugin()
