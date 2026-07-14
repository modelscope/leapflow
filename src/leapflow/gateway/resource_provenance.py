"""Resource provenance tracking for platform actions.

Maintains a session-scoped pool of resource identifiers (chat_id, message_id,
file_key, etc.) that have been observed from successful API responses. Used to
detect hallucinated resource references before they reach the approval gate.

Design:
- Platform-neutral: no Feishu-specific logic.
- Populated automatically from successful action results that contain fields
  matching declared ``resource_fields`` from any action spec.
- Queried before side-effect actions to determine provenance of referenced
  resources: VERIFIED (seen from API), UNVERIFIED (pool populated but ID not
  found), or UNKNOWN (pool never populated for this resource type).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Set, Sequence

logger = logging.getLogger(__name__)


class ProvenanceStatus(str, Enum):
    """Result of checking a resource identifier's provenance."""

    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ProvenanceResult:
    """Outcome of a provenance check for a single resource field."""

    field_name: str
    value: str
    status: ProvenanceStatus
    known_values_count: int = 0

    @property
    def is_safe(self) -> bool:
        """True if the resource is verified or provenance cannot be determined."""
        return self.status in (ProvenanceStatus.VERIFIED, ProvenanceStatus.UNKNOWN)


class ResourceProvenancePool:
    """Session-scoped pool of known resource identifiers per platform.

    Thread-safety: single-threaded async usage expected; no locking.
    """

    def __init__(self) -> None:
        # Key: (platform, resource_field_name) → set of known values
        self._pool: Dict[tuple[str, str], Set[str]] = {}

    def register(
        self,
        platform: str,
        resource_field: str,
        value: str,
    ) -> None:
        """Register a resource identifier observed from a successful API call."""
        if not value:
            return
        key = (platform, resource_field)
        if key not in self._pool:
            self._pool[key] = set()
        self._pool[key].add(value)

    def register_from_result(
        self,
        platform: str,
        resource_fields: Sequence[str],
        result_data: Any,
    ) -> int:
        """Extract and register resource IDs from an action result.

        Recursively scans the result data for keys matching declared
        resource_fields and registers all found string values.

        Returns the number of values registered.
        """
        if not resource_fields or not result_data:
            return 0
        target_fields = set(resource_fields)
        found = _extract_values(result_data, target_fields)
        count = 0
        for field_name, values in found.items():
            for value in values:
                self.register(platform, field_name, value)
                count += 1
        if count > 0:
            logger.debug(
                "resource_provenance.registered platform=%s fields=%s count=%d",
                platform, list(found.keys()), count,
            )
        return count

    def check(
        self,
        platform: str,
        resource_field: str,
        value: str,
    ) -> ProvenanceResult:
        """Check whether a resource identifier has known provenance."""
        key = (platform, resource_field)
        pool = self._pool.get(key)
        if pool is None:
            return ProvenanceResult(
                field_name=resource_field,
                value=value,
                status=ProvenanceStatus.UNKNOWN,
                known_values_count=0,
            )
        if value in pool:
            return ProvenanceResult(
                field_name=resource_field,
                value=value,
                status=ProvenanceStatus.VERIFIED,
                known_values_count=len(pool),
            )
        return ProvenanceResult(
            field_name=resource_field,
            value=value,
            status=ProvenanceStatus.UNVERIFIED,
            known_values_count=len(pool),
        )

    def check_payload(
        self,
        platform: str,
        resource_fields: Sequence[str],
        payload: Mapping[str, Any],
    ) -> list[ProvenanceResult]:
        """Check provenance of all resource fields referenced in a payload.

        Returns a list of ProvenanceResult for each resource field that has a
        non-empty value in the payload.
        """
        results: list[ProvenanceResult] = []
        for field_name in resource_fields:
            value = str(payload.get(field_name) or "")
            if not value:
                continue
            results.append(self.check(platform, field_name, value))
        return results

    def has_data(self, platform: str, resource_field: str) -> bool:
        """Return True if any values have been registered for this resource type."""
        key = (platform, resource_field)
        pool = self._pool.get(key)
        return pool is not None and len(pool) > 0

    def clear(self, platform: str | None = None) -> None:
        """Clear pool entries for a platform, or all entries if platform is None."""
        if platform is None:
            self._pool.clear()
        else:
            keys = [k for k in self._pool if k[0] == platform]
            for key in keys:
                del self._pool[key]

    def summary(self) -> Dict[str, Dict[str, int]]:
        """Return a compact summary: platform → field → count."""
        result: Dict[str, Dict[str, int]] = {}
        for (platform, field_name), values in self._pool.items():
            if platform not in result:
                result[platform] = {}
            result[platform][field_name] = len(values)
        return result


def _extract_values(
    data: Any,
    target_fields: set[str],
    *,
    _depth: int = 0,
) -> Dict[str, list[str]]:
    """Recursively extract values for target field names from nested data."""
    if _depth > 10:
        return {}
    found: Dict[str, list[str]] = {}
    if isinstance(data, Mapping):
        for key, value in data.items():
            str_key = str(key)
            if str_key in target_fields and isinstance(value, str) and value:
                found.setdefault(str_key, []).append(value)
            elif isinstance(value, (dict, list)):
                nested = _extract_values(value, target_fields, _depth=_depth + 1)
                for nk, nv in nested.items():
                    found.setdefault(nk, []).extend(nv)
    elif isinstance(data, (list, tuple)):
        for item in data:
            nested = _extract_values(item, target_fields, _depth=_depth + 1)
            for nk, nv in nested.items():
                found.setdefault(nk, []).extend(nv)
    return found
