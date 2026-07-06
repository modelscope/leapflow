"""Privacy policy — configuration-driven data retention, opt-out, and audit.

Design principles (from Active Learning Design doc):
- "本地处理端侧推理"
- "用户完全掌控可审计的记录"
- "默认 opt-in per observer + EpisodicMemory TTL 自动过期"

Ensures that 24/7 observation doesn't become "无节制的隐私侵犯".
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataRetentionConfig:
    """Data retention policy configuration."""
    episodic_ttl_s: float = 86400.0 * 7    # 7 days for episodic memory
    trajectory_ttl_s: float = 86400.0 * 30  # 30 days for trajectories
    audit_ttl_s: float = 86400.0 * 90      # 90 days for audit logs
    max_episodic_entries: int = 10000       # Hard cap on episodic entries
    max_trajectory_entries: int = 1000      # Hard cap on trajectories


@dataclass
class PrivacyPolicy:
    """User-configurable privacy policy for observation and data collection.
    
    Controls what is observed, how long data is retained, and provides
    user-facing opt-out granularity at the observer level.
    """
    # Observer opt-out (per observer name)
    disabled_observers: Set[str] = field(default_factory=set)
    
    # Data retention
    retention: DataRetentionConfig = field(default_factory=DataRetentionConfig)
    
    # Content filtering
    exclude_apps: FrozenSet[str] = field(default_factory=frozenset)  # Apps to never observe
    exclude_paths: FrozenSet[str] = field(default_factory=frozenset)  # Paths to never watch
    
    # Audit
    audit_all_access: bool = True  # Log all memory/data access
    
    # Sensitivity
    redact_passwords: bool = True  # Redact password-like fields from events
    redact_clipboard_sensitive: bool = True  # Redact clipboard if contains sensitive patterns

    def is_observer_allowed(self, observer_name: str) -> bool:
        """Check if an observer is allowed to run."""
        return observer_name not in self.disabled_observers

    def is_app_allowed(self, app_name: str) -> bool:
        """Check if an app is allowed to be observed."""
        return app_name.lower() not in {a.lower() for a in self.exclude_apps}

    def is_path_allowed(self, path: str) -> bool:
        """Check if a path is allowed to be watched."""
        return not any(path.startswith(excluded) for excluded in self.exclude_paths)

    def should_redact(self, content: str) -> bool:
        """Check if content should be redacted based on sensitivity patterns."""
        if not self.redact_passwords:
            return False
        # Simple heuristic: check for common sensitive patterns
        sensitive_indicators = ("password", "secret", "token", "api_key", "private_key")
        content_lower = content.lower()
        return any(indicator in content_lower for indicator in sensitive_indicators)


class PrivacyManager:
    """Manages privacy policy enforcement and data lifecycle.
    
    Responsibilities:
    - Enforce observer opt-out
    - Apply data retention policies (TTL expiry)
    - Audit data access
    - Provide user-facing privacy report
    """

    def __init__(self, policy: PrivacyPolicy = PrivacyPolicy()) -> None:
        self._policy = policy
        self._access_log: List[Dict[str, Any]] = []

    @property
    def policy(self) -> PrivacyPolicy:
        return self._policy

    def update_policy(self, **kwargs: Any) -> None:
        """Update policy fields dynamically."""
        for key, value in kwargs.items():
            if hasattr(self._policy, key):
                setattr(self._policy, key, value)
        logger.info("Privacy policy updated: %s", list(kwargs.keys()))

    def disable_observer(self, observer_name: str) -> None:
        """User opt-out: disable a specific observer."""
        self._policy.disabled_observers.add(observer_name)
        logger.info("Observer '%s' disabled by user", observer_name)

    def enable_observer(self, observer_name: str) -> None:
        """User opt-in: re-enable a specific observer."""
        self._policy.disabled_observers.discard(observer_name)
        logger.info("Observer '%s' re-enabled by user", observer_name)

    def record_access(self, accessor: str, data_type: str, purpose: str) -> None:
        """Audit: record a data access event."""
        if self._policy.audit_all_access:
            entry = {
                "ts": time.time(),
                "accessor": accessor,
                "data_type": data_type,
                "purpose": purpose,
            }
            self._access_log.append(entry)
            # Keep bounded
            if len(self._access_log) > 10000:
                self._access_log = self._access_log[-5000:]

    def redact_if_sensitive(self, content: str) -> str:
        """Redact content if it matches sensitivity patterns."""
        if self._policy.should_redact(content):
            return "[REDACTED — sensitive content]"
        return content

    def get_privacy_report(self) -> Dict[str, Any]:
        """Generate a user-facing privacy report."""
        return {
            "disabled_observers": list(self._policy.disabled_observers),
            "excluded_apps": list(self._policy.exclude_apps),
            "data_retention": {
                "episodic_days": self._policy.retention.episodic_ttl_s / 86400,
                "trajectory_days": self._policy.retention.trajectory_ttl_s / 86400,
                "audit_days": self._policy.retention.audit_ttl_s / 86400,
            },
            "recent_accesses": len(self._access_log),
            "redaction_enabled": self._policy.redact_passwords,
        }
