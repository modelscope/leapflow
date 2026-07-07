"""Privacy policy — configuration-driven data retention, opt-out, and audit.

Design principles (from Active Learning Design doc):
- Local processing, user-controlled, auditable records
- Opt-in per observer + EpisodicMemory TTL auto-expiry

Ensures that continuous observation respects user privacy boundaries.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Protocol, Set, runtime_checkable

logger = logging.getLogger(__name__)

_SENSITIVE_PATTERNS = re.compile(
    r"(?i)(password|passwd|secret|token|api[_-]?key|private[_-]?key"
    r"|credential|authorization|bearer\s)",
)


@dataclass(frozen=True)
class DataRetentionConfig:
    """Data retention policy configuration."""
    episodic_ttl_s: float = 86400.0 * 7
    trajectory_ttl_s: float = 86400.0 * 30
    audit_ttl_s: float = 86400.0 * 90
    max_episodic_entries: int = 10000
    max_trajectory_entries: int = 1000


@dataclass(frozen=True)
class PrivacyPolicy:
    """Immutable privacy policy snapshot.

    Controls what is observed, how long data is retained, and provides
    user-facing opt-out granularity at the observer level.
    """
    disabled_observers: frozenset[str] = frozenset()
    retention: DataRetentionConfig = DataRetentionConfig()
    exclude_apps: frozenset[str] = frozenset()
    exclude_paths: frozenset[str] = frozenset()
    audit_all_access: bool = True
    redact_passwords: bool = True
    redact_clipboard_sensitive: bool = True

    def is_observer_allowed(self, observer_name: str) -> bool:
        return observer_name not in self.disabled_observers

    def is_app_allowed(self, app_id: str) -> bool:
        if not self.exclude_apps:
            return True
        return app_id.lower() not in {a.lower() for a in self.exclude_apps}

    def is_path_allowed(self, path: str) -> bool:
        return not any(path.startswith(excluded) for excluded in self.exclude_paths)

    def should_redact(self, content: str) -> bool:
        if not self.redact_passwords and not self.redact_clipboard_sensitive:
            return False
        return bool(_SENSITIVE_PATTERNS.search(content))


@runtime_checkable
class EventPrivacyFilter(Protocol):
    """Protocol for privacy filtering in the EventBus pipeline.

    Implementations decide whether an event should be ingested into memory
    and optionally redact sensitive content before storage.
    """

    def should_ingest(self, event_type: str, source: str, payload: Dict[str, Any]) -> bool:
        """Return False to block the event from entering memory."""
        ...

    def redact_payload(self, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Return a sanitized copy of the payload for memory storage."""
        ...


class PrivacyManager:
    """Manages privacy policy enforcement and data lifecycle.

    Implements EventPrivacyFilter so it can be injected directly into EventBus.
    """

    def __init__(self, policy: PrivacyPolicy = PrivacyPolicy()) -> None:
        self._policy = policy
        self._access_log: List[Dict[str, Any]] = []

    @property
    def policy(self) -> PrivacyPolicy:
        return self._policy

    def update_policy(self, policy: PrivacyPolicy) -> None:
        """Replace the current policy with a new immutable snapshot."""
        self._policy = policy
        logger.info("Privacy policy updated")

    # ── EventPrivacyFilter implementation ──

    def should_ingest(self, event_type: str, source: str, payload: Dict[str, Any]) -> bool:
        """Deny ingestion for excluded apps and paths."""
        if event_type in ("app.focus_change", "ui.action", "context.change"):
            app_id = payload.get("bundle_id", "") or payload.get("app_bundle_id", "") or source
            if app_id and not self._policy.is_app_allowed(app_id):
                return False

        if event_type == "fs.change":
            path = payload.get("path", "") or source
            if path and not self._policy.is_path_allowed(path):
                return False

        return True

    def redact_payload(self, event_type: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Redact sensitive fields from clipboard and text content."""
        if event_type == "clipboard.change" and self._policy.redact_clipboard_sensitive:
            text = payload.get("text", "")
            if text and self._policy.should_redact(text):
                redacted = dict(payload)
                redacted["text"] = "[REDACTED]"
                redacted["redacted"] = True
                return redacted

        for key in ("content", "text", "value"):
            value = payload.get(key, "")
            if isinstance(value, str) and value and self._policy.should_redact(value):
                redacted = dict(payload)
                redacted[key] = "[REDACTED]"
                redacted["redacted"] = True
                return redacted

        return payload

    # ── Observer opt-out ──

    def is_observer_allowed(self, observer_name: str) -> bool:
        return self._policy.is_observer_allowed(observer_name)

    # ── Audit ──

    def record_access(self, accessor: str, data_type: str, purpose: str) -> None:
        if self._policy.audit_all_access:
            entry = {
                "ts": time.time(),
                "accessor": accessor,
                "data_type": data_type,
                "purpose": purpose,
            }
            self._access_log.append(entry)
            if len(self._access_log) > 10000:
                self._access_log = self._access_log[-5000:]

    def get_privacy_report(self) -> Dict[str, Any]:
        return {
            "disabled_observers": sorted(self._policy.disabled_observers),
            "excluded_apps": sorted(self._policy.exclude_apps),
            "data_retention": {
                "episodic_days": self._policy.retention.episodic_ttl_s / 86400,
                "trajectory_days": self._policy.retention.trajectory_ttl_s / 86400,
                "audit_days": self._policy.retention.audit_ttl_s / 86400,
            },
            "recent_accesses": len(self._access_log),
            "redaction_enabled": self._policy.redact_passwords,
            "clipboard_redaction": self._policy.redact_clipboard_sensitive,
        }
