"""Risk assessment for structured approval actions."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

from leapflow.security.actions import ActionDescriptor, ActionEffect, ActionKind
from leapflow.security.path_sensitivity import configured_path_sensitivity_roots


class RiskLevel(str, Enum):
    """Risk levels used by approval policy."""

    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(frozen=True)
class RiskAssessment:
    """Structured risk assessment for a pending action."""

    level: RiskLevel
    score: float = 0.0
    reasons: tuple[str, ...] = ()
    explanation: str = ""
    hardline: bool = False
    allow_permanent: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["level"] = self.level.value
        data["reasons"] = list(self.reasons)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RiskAssessment":
        raw_level = str(data.get("level") or RiskLevel.MEDIUM.value)
        try:
            level = RiskLevel(raw_level)
        except ValueError:
            level = RiskLevel.MEDIUM
        return cls(
            level=level,
            score=float(data.get("score") or 0.0),
            reasons=tuple(str(item) for item in data.get("reasons") or ()),
            explanation=str(data.get("explanation") or ""),
            hardline=bool(data.get("hardline", False)),
            allow_permanent=bool(data.get("allow_permanent", True)),
            metadata=dict(data.get("metadata") or {}),
        )


@runtime_checkable
class RiskClassifier(Protocol):
    """Classifies action risk without prompting the user."""

    def assess(self, action: ActionDescriptor) -> RiskAssessment: ...


class CompositeRiskClassifier:
    """Dispatches by action-kind prefix, delegating everything else.

    ``ApprovalOrchestrator`` holds a single classifier slot, so a new action
    domain cannot simply register itself alongside the default classifier.
    Composing rather than replacing keeps the fallback the authority for every
    kind it already owns: adding a domain must never change how a shell command
    is assessed.

    This class carries no domain knowledge of its own, which is the point -- a
    second domain registers a prefix instead of forking it. Longest prefix wins,
    so ``device.actuate.`` can refine ``device.`` without ordering surprises.
    """

    def __init__(
        self,
        *,
        fallback: RiskClassifier,
        by_prefix: Mapping[str, RiskClassifier] = MappingProxyType({}),
    ) -> None:
        self._fallback = fallback
        # Sorted longest-first so a more specific prefix is never shadowed by a
        # shorter one that happens to be inserted earlier.
        self._by_prefix: tuple[tuple[str, RiskClassifier], ...] = tuple(
            sorted(dict(by_prefix).items(), key=lambda item: len(item[0]), reverse=True)
        )

    def assess(self, action: ActionDescriptor) -> RiskAssessment:
        kind = str(getattr(action, "kind", "") or "")
        for prefix, classifier in self._by_prefix:
            if kind.startswith(prefix):
                return classifier.assess(action)
        return self._fallback.assess(action)


class DefaultRiskClassifier:
    """Small, explainable risk classifier for core LeapFlow actions."""

    _SHELL_HARDLINE: tuple[tuple[re.Pattern[str], str], ...] = (
        (re.compile(r"\brm\s+.*-[^\s]*r[^\s]*f.*\s/\s*$", re.IGNORECASE), "recursive_delete_root"),
        (re.compile(r"\bmkfs\b", re.IGNORECASE), "format_filesystem"),
        (re.compile(r"\bdd\s+.*of=/dev/", re.IGNORECASE), "raw_device_write"),
        (re.compile(r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:"), "fork_bomb"),
        (re.compile(r"\bshutdown\b|\breboot\b|\bhalt\b|\bpoweroff\b", re.IGNORECASE), "system_shutdown"),
    )
    _SHELL_HIGH: tuple[tuple[re.Pattern[str], str], ...] = (
        (re.compile(r"\b(?:python[23]?|perl|ruby|node|bash|sh|zsh|ksh)\s+<<", re.IGNORECASE), "script_execution_via_heredoc"),
        (re.compile(r"\b(curl|wget)\b.*\|\s*(?:ba)?sh", re.IGNORECASE), "remote_script_execution"),
        (re.compile(r"\bsudo\b.*(?:-S|--stdin|--askpass|-A)\b", re.IGNORECASE), "privileged_command_with_password_path"),
        (re.compile(r"\bgit\s+push\b.*(?:--force|-f)\b", re.IGNORECASE), "git_force_push"),
        (re.compile(r"\bsystemctl\s+(?:stop|restart|disable|mask)\b", re.IGNORECASE), "service_lifecycle_change"),
    )
    _SHELL_MEDIUM: tuple[tuple[re.Pattern[str], str], ...] = (
        (re.compile(r"\bsudo\b", re.IGNORECASE), "privileged_command"),
        (re.compile(r"\brm\s+-r\b", re.IGNORECASE), "recursive_delete"),
        (re.compile(r"\bchmod\s+[0-7]*7[0-7]*\b", re.IGNORECASE), "broad_permission_change"),
        (re.compile(r"\b(curl|wget|ssh|scp|rsync|nc|ncat)\b", re.IGNORECASE), "external_network_access"),
        (re.compile(r"\b(pip|npm|brew)\s+install\b", re.IGNORECASE), "dependency_install"),
    )
    _SENSITIVE_NAMES = frozenset({
        ".env", ".env.local", ".env.production", "credentials.json",
        "secrets.yaml", "secrets.yml", "vault.json", "vault.key",
        ".npmrc", ".pypirc", ".netrc",
    })
    _SENSITIVE_PARTS = ("/.ssh/", "/.gnupg/", "/.aws/", "/.kube/")

    def assess(self, action: ActionDescriptor) -> RiskAssessment:
        if action.kind == ActionKind.SHELL_COMMAND.value:
            return self._assess_shell(action)
        if action.kind == ActionKind.FILE_READ.value:
            return self._assess_file_read(action)
        if action.kind == ActionKind.FILE_WRITE.value:
            return self._assess_file_write(action)
        if action.kind == ActionKind.NETWORK_FETCH.value:
            return self._assess_network_fetch(action)
        if action.kind == ActionKind.WORKSPACE_ESCAPE.value:
            return self._assess_workspace_escape(action)
        if action.kind == ActionKind.MCP_TOOL.value:
            return self._assess_mcp_tool(action)
        if action.kind == ActionKind.GATEWAY_SEND.value:
            return RiskAssessment(
                level=RiskLevel.HIGH,
                score=0.72,
                reasons=("external_message_send",),
                explanation="This sends content to an external platform conversation.",
                allow_permanent=False,
            )
        if action.kind == ActionKind.PLATFORM_ACTION.value:
            explicit_level = self._platform_risk_level(action)
            if explicit_level is not None:
                return explicit_level
            if action.effect == "read":
                return RiskAssessment(
                    level=RiskLevel.MEDIUM,
                    score=0.5,
                    reasons=("platform_data_access",),
                    explanation="This reads data from an external platform.",
                    metadata={"backend_kind": action.metadata.get("backend_kind", "")},
                )
            return RiskAssessment(
                level=RiskLevel.HIGH,
                score=0.74,
                reasons=("external_platform_action",),
                explanation="This performs an action through an external platform backend.",
                allow_permanent=False,
                metadata={"backend_kind": action.metadata.get("backend_kind", "")},
            )
        if action.kind in {
            ActionKind.SCHEDULER_ARM.value,
            ActionKind.SKILL_PROMOTE.value,
            ActionKind.APP_INSTALL.value,
            ActionKind.RUNTIME_CONFIGURE.value,
        }:
            return RiskAssessment(
                level=RiskLevel.HIGH,
                score=0.75,
                reasons=("long_lived_or_runtime_changing_action",),
                explanation="This action can change future runtime behavior or run without the current terminal.",
                allow_permanent=False,
            )
        return RiskAssessment(level=RiskLevel.MEDIUM, score=0.5, reasons=("external_action",))

    # Target categories that mean "this request would reach infrastructure the
    # user did not intend to expose": loopback holds the daemon socket and the
    # dashboard, link-local holds cloud instance metadata (credential theft), and
    # private ranges hold the rest of the user's network.
    _INTERNAL_TARGETS = frozenset({
        "loopback", "private", "link_local", "metadata", "unspecified", "reserved",
    })

    def _assess_network_fetch(self, action: ActionDescriptor) -> RiskAssessment:
        """Assess an outbound HTTP read.

        Deliberately never CRITICAL and never hardline: policy turns both into an
        outright block, and the product decision is that internal targets stay
        reachable through an explicit human approval rather than being refused.
        HIGH plus ``allow_permanent=False`` is what expresses "ask every session,
        never remember forever".
        """
        category = str(action.metadata.get("target_category") or "unknown")
        scheme = str(action.metadata.get("scheme") or "").lower()
        origin = str(action.resource or "")

        if action.metadata.get("has_credentials"):
            return RiskAssessment(
                level=RiskLevel.HIGH,
                score=0.8,
                reasons=("url_embedded_credentials",),
                explanation=(
                    "The URL carries credentials, so fetching it would send them to "
                    f"{origin} and record them in this turn."
                ),
                allow_permanent=False,
                metadata={"origin": origin, "target_category": category},
            )

        if category in self._INTERNAL_TARGETS:
            return RiskAssessment(
                level=RiskLevel.HIGH,
                score=0.82,
                reasons=(f"{category}_network_target", "internal_service_access"),
                explanation=(
                    f"{origin} resolves to a {category.replace('_', ' ')} address, which "
                    "can reach services on this machine or private network rather than "
                    "the public internet."
                ),
                allow_permanent=False,
                metadata={"origin": origin, "target_category": category},
            )

        if scheme and scheme != "https":
            return RiskAssessment(
                level=RiskLevel.LOW,
                score=0.3,
                reasons=("public_network_read", "plaintext_transport"),
                explanation=f"Reads {origin} over plaintext {scheme}.",
                metadata={"origin": origin, "target_category": category},
            )

        return RiskAssessment(
            level=RiskLevel.LOW,
            score=0.15,
            reasons=("public_network_read",),
            explanation=f"Reads public web content from {origin}.",
            metadata={"origin": origin, "target_category": category},
        )

    @staticmethod
    def _platform_risk_level(action: ActionDescriptor) -> RiskAssessment | None:
        # Self-modification is always HIGH risk, no permanent grants allowed.
        # This is a defense-in-depth guard: even if a caller forgets to set
        # explicit risk metadata, plugin_management actions are treated as
        # HIGH so that "always allow" grants are never offered for actions
        # that change the agent's own composition.
        if action.metadata.get("platform") == "plugin_management":
            return RiskAssessment(
                level=RiskLevel.HIGH,
                score=0.78,
                reasons=("agent_self_modification",),
                explanation="This action changes the agent's own plugin composition.",
                hardline=False,
                allow_permanent=False,
                metadata={
                    "backend_kind": action.metadata.get("backend_kind", ""),
                    "platform": action.metadata.get("platform", ""),
                    "action": action.metadata.get("action", ""),
                },
            )
        raw = str(action.metadata.get("risk_level") or "").lower()
        if not raw:
            return None
        try:
            level = RiskLevel(raw)
        except ValueError:
            return None
        score_by_level = {
            RiskLevel.SAFE: 0.05,
            RiskLevel.LOW: 0.2,
            RiskLevel.MEDIUM: 0.5,
            RiskLevel.HIGH: 0.78,
            RiskLevel.CRITICAL: 0.95,
        }
        return RiskAssessment(
            level=level,
            score=score_by_level[level],
            reasons=("registered_platform_action",),
            explanation="This risk level comes from the registered platform action spec.",
            hardline=level == RiskLevel.CRITICAL,
            allow_permanent=level not in {RiskLevel.HIGH, RiskLevel.CRITICAL},
            metadata={
                "backend_kind": action.metadata.get("backend_kind", ""),
                "platform": action.metadata.get("platform", ""),
                "action": action.metadata.get("action", ""),
            },
        )

    def _assess_shell(self, action: ActionDescriptor) -> RiskAssessment:
        command = action.detail
        for pattern, reason in self._SHELL_HARDLINE:
            if pattern.search(command):
                return RiskAssessment(
                    level=RiskLevel.CRITICAL,
                    score=1.0,
                    reasons=(reason,),
                    explanation="This command can irreversibly damage the host and is never run by LeapFlow.",
                    hardline=True,
                    allow_permanent=False,
                )
        reasons = self._matched_reasons(command, self._SHELL_HIGH)
        if reasons:
            if self._mentions_sensitive_config(command):
                reasons.append("writes_or_reads_sensitive_config")
            return RiskAssessment(
                level=RiskLevel.HIGH,
                score=0.82,
                reasons=tuple(reasons),
                explanation="This shell command executes code, changes runtime state, or reaches sensitive resources.",
                allow_permanent=False,
            )
        if self._mentions_sensitive_config(command):
            return RiskAssessment(
                level=RiskLevel.HIGH,
                score=0.8,
                reasons=("sensitive_config_reference",),
                explanation="This command references LeapFlow configuration, secrets, or profile data.",
                allow_permanent=False,
            )
        reasons = self._matched_reasons(command, self._SHELL_MEDIUM)
        if reasons:
            return RiskAssessment(
                level=RiskLevel.MEDIUM,
                score=0.58,
                reasons=tuple(reasons),
                explanation="This shell command has side effects or reaches external systems.",
            )
        return RiskAssessment(level=RiskLevel.LOW, score=0.15, reasons=("ordinary_shell_command",))

    def _assess_workspace_escape(self, action: ActionDescriptor) -> RiskAssessment:
        """Assess a tool reaching outside the active workspace.

        Never LOW, and never below the policy's ASK threshold: crossing the
        boundary is precisely the case a human must see. Rating it by the target
        path's own sensitivity would let an ordinary file in another project fall
        through as low risk, and the policy engine auto-allows low risk — turning
        a refusal into silent cross-workspace access.

        The effect sets the tier, so listing a sibling repo is not weighed the
        same as writing into it. Write and execute additionally refuse permanent
        grants: "always allow" for mutating another workspace is a standing
        licence the user is unlikely to have meant.
        """
        operation = str((action.metadata or {}).get("operation") or "tool")
        if action.effect in {ActionEffect.WRITE.value, ActionEffect.EXECUTE.value,
                             ActionEffect.DELETE.value}:
            return RiskAssessment(
                level=RiskLevel.HIGH,
                score=0.7,
                reasons=("workspace_escape_mutating",),
                explanation=(
                    f"{operation} would modify or execute against a path outside "
                    "the active workspace."
                ),
                allow_permanent=False,
                metadata={"operation": operation, "effect": action.effect},
            )
        return RiskAssessment(
            level=RiskLevel.MEDIUM,
            score=0.5,
            reasons=("workspace_escape_read",),
            explanation=(
                f"{operation} would read a path outside the active workspace."
            ),
            metadata={"operation": operation, "effect": action.effect},
        )

    def _assess_file_read(self, action: ActionDescriptor) -> RiskAssessment:
        path = Path(action.resource).expanduser()
        name = path.name.lower()
        normalized = str(path).replace("\\", "/").lower()
        meta = action.metadata or {}
        category = str(meta.get("sensitivity_category") or "")
        if category in {"credential", "secret_vault"}:
            return RiskAssessment(
                level=RiskLevel.HIGH,
                score=0.82,
                reasons=("credential_file_read",),
                explanation="This reads credentials, tokens, or security-sensitive configuration. Content will be redacted.",
                allow_permanent=False,
            )
        if category in (
            "approval_state", "audit_log", "memory_store", "leapflow_profile_data",
            "runtime_state", "config", "mcp_config", "workspace_manifest", "history",
            "cache_sensitive",
        ):
            return RiskAssessment(
                level=RiskLevel.HIGH,
                score=0.75,
                reasons=(f"{category}_read",),
                explanation="This reads LeapFlow internal data. Content will be redacted.",
                allow_permanent=False,
            )
        if name in self._SENSITIVE_NAMES or any(part in normalized for part in self._SENSITIVE_PARTS):
            return RiskAssessment(
                level=RiskLevel.HIGH,
                score=0.78,
                reasons=("sensitive_file_read",),
                explanation="This reads credentials, configuration, or security-sensitive files.",
                allow_permanent=False,
            )
        return RiskAssessment(level=RiskLevel.LOW, score=0.1, reasons=("ordinary_file_read",))

    def _assess_file_write(self, action: ActionDescriptor) -> RiskAssessment:
        path = Path(action.resource).expanduser()
        name = path.name.lower()
        normalized = str(path).replace("\\", "/").lower()
        size = int(action.metadata.get("bytes") or 0)
        category = str(action.metadata.get("sensitivity_category") or "")
        if any(normalized.startswith(prefix) for prefix in ("/system", "/usr", "/bin", "/sbin", "/etc")):
            return RiskAssessment(
                level=RiskLevel.CRITICAL,
                score=0.95,
                reasons=("system_path_write",),
                explanation="This writes to an operating-system controlled path.",
                hardline=True,
                allow_permanent=False,
            )
        if category in {"runtime_database", "database", "runtime_control", "secret_vault"}:
            return RiskAssessment(
                level=RiskLevel.CRITICAL,
                score=0.95,
                reasons=(f"{category}_write",),
                explanation="This writes to protected LeapFlow runtime, database, or secret storage.",
                hardline=True,
                allow_permanent=False,
            )
        if category in {"config", "mcp_config", "workspace_manifest", "history", "cache_sensitive", "approval_state"}:
            return RiskAssessment(
                level=RiskLevel.HIGH,
                score=0.8,
                reasons=(f"{category}_write",),
                explanation="This writes to sensitive LeapFlow runtime configuration or state.",
                allow_permanent=False,
            )
        if name in self._SENSITIVE_NAMES or any(part in normalized for part in self._SENSITIVE_PARTS):
            return RiskAssessment(
                level=RiskLevel.HIGH,
                score=0.8,
                reasons=("sensitive_file_write",),
                explanation="This writes to credentials, configuration, or security-sensitive files.",
                allow_permanent=False,
            )
        if size > 20_000:
            return RiskAssessment(
                level=RiskLevel.MEDIUM,
                score=0.5,
                reasons=("large_file_write",),
                explanation="This writes a non-trivial amount of content.",
            )
        return RiskAssessment(level=RiskLevel.LOW, score=0.2, reasons=("ordinary_file_write",))

    def _assess_mcp_tool(self, action: ActionDescriptor) -> RiskAssessment:
        """Assess a tool supplied by an external MCP server.

        The MCP protocol does not tell us what a tool does, so the only honest basis is
        provenance: this is third-party code reached over a local transport, running with
        the agent's own privileges. That places it at the same tier as any other external
        action, and *not* at the tier its own description claims -- the description was
        written by the server being assessed.

        A server may declare ``readOnlyHint``, and that declaration is honoured the same
        way a plugin's declared metadata is: it is the server's own contract, and treating
        a self-declared read as a write would gate documentation lookups into uselessness.
        Absence of the hint is not a claim of read-only, so it stays at the higher tier.

        ``allow_permanent`` remains available. A person who trusts a server enough to
        configure it should be able to say so once; refusing that would mean prompting on
        every call, which is how a gate gets switched off by the person it protects.
        """
        metadata = action.metadata or {}
        server = str(metadata.get("server") or "unknown")
        if bool(metadata.get("read_only", False)):
            return RiskAssessment(
                level=RiskLevel.LOW,
                score=0.25,
                reasons=("mcp_tool_read_only",),
                explanation=(
                    f"This calls a tool the {server!r} MCP server declares as read-only. "
                    "It still runs third-party code on this machine."
                ),
            )
        return RiskAssessment(
            level=RiskLevel.HIGH,
            score=0.7,
            reasons=("mcp_tool_undeclared_effect",),
            explanation=(
                f"This calls a tool on the {server!r} MCP server that does not declare "
                "itself read-only, so its effect is unknown. It runs third-party code "
                "with this agent's privileges."
            ),
        )

    @staticmethod
    def _matched_reasons(command: str, rules: tuple[tuple[re.Pattern[str], str], ...]) -> list[str]:
        return [reason for pattern, reason in rules if pattern.search(command)]

    @staticmethod
    def _mentions_sensitive_config(command: str) -> bool:
        lowered = command.lower().replace("\\", "/")
        configured_roots = tuple(
            str(root).lower().replace("\\", "/").rstrip("/")
            for root in configured_path_sensitivity_roots()
        )
        return any(
            token in lowered
            for token in (
                ".env", "vault.json", "vault.key", "secrets.yaml", "config/user.yaml",
                "mcp_servers.json", "workspace.yaml", ".leapflow/config.yaml", "tui_history",
                "profiles/", "config.yaml", *configured_roots,
            )
        )
