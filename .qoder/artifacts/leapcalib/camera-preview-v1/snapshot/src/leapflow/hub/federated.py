# Copyright (c) Alibaba, Inc. and its affiliates.
"""Federated Hub Router — parallel multi-backend search and pull aggregation.

Routes search/list queries to multiple Hub backends concurrently, aggregates
and deduplicates results.  This is a composition layer above HubBackend; it
does not modify the Protocol itself.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from leapflow.hub.protocol import HubBackend, SkillBundle, SkillSummary

logger = logging.getLogger(__name__)

# Per-backend query timeout (seconds).
_DEFAULT_BACKEND_TIMEOUT: float = 30.0


# ─── Result Container ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FederatedSearchResult:
    """Aggregated search outcome across multiple backends."""

    results: Tuple[SkillSummary, ...] = ()
    failed_backends: Tuple[str, ...] = ()
    total_backends_queried: int = 0


# ─── Deduplication Helpers ────────────────────────────────────────────────────


def _semver_tuple(version: str) -> Tuple[int, ...]:
    """Parse a version string into a comparable int tuple."""
    cleaned = version.lstrip("vV")
    parts: list[int] = []
    for segment in cleaned.split("."):
        if segment.isdigit():
            parts.append(int(segment))
    return tuple(parts) if parts else (0,)


def _prefer_best(existing: SkillSummary, challenger: SkillSummary) -> SkillSummary:
    """Given two summaries for the same skill, return the better one.

    Preference order:
    1. Higher semantic version.
    2. More downloads (popularity signal).
    3. Keep existing (first-seen wins as tie-breaker).
    """
    ev = _semver_tuple(existing.version)
    cv = _semver_tuple(challenger.version)
    if cv > ev:
        return challenger
    if cv < ev:
        return existing
    # Versions equal — prefer more downloads.
    if challenger.downloads > existing.downloads:
        return challenger
    return existing


def _deduplicate(summaries: List[SkillSummary]) -> List[SkillSummary]:
    """Deduplicate by (name, version-agnostic key) keeping the best variant."""
    best_by_name: Dict[str, SkillSummary] = {}
    for s in summaries:
        key = s.name
        if key in best_by_name:
            best_by_name[key] = _prefer_best(best_by_name[key], s)
        else:
            best_by_name[key] = s
    return list(best_by_name.values())


# ─── FederatedHubRouter ──────────────────────────────────────────────────────


class FederatedHubRouter:
    """Routes search/list queries to multiple Hub backends in parallel,
    aggregates and deduplicates results.

    Usage::

        router = FederatedHubRouter()
        router.add_backend("modelscope", ms_backend)
        router.add_backend("github", gh_backend)
        results = await router.search("file management")
    """

    def __init__(
        self,
        *,
        timeout: float = _DEFAULT_BACKEND_TIMEOUT,
    ) -> None:
        """Initialize the router.

        Args:
            timeout: Per-backend query timeout in seconds.
        """
        self._backends: List[Tuple[str, HubBackend]] = []
        self._timeout = timeout

    # ─── Registration ─────────────────────────────────────────────────────

    def add_backend(self, hub_type: str, backend: HubBackend) -> None:
        """Register a backend for federated queries.

        Args:
            hub_type: Backend identifier (e.g. 'modelscope', 'github').
            backend: A concrete HubBackend instance.
        """
        # Avoid duplicate registrations for the same hub_type.
        for existing_type, _ in self._backends:
            if existing_type == hub_type:
                logger.debug(
                    "Backend '%s' already registered — skipping duplicate", hub_type
                )
                return
        self._backends.append((hub_type, backend))
        logger.debug("Federated router: added backend '%s'", hub_type)

    @property
    def backend_count(self) -> int:
        """Return the number of registered backends."""
        return len(self._backends)

    @property
    def backend_types(self) -> List[str]:
        """Return registered backend type names in registration order."""
        return [ht for ht, _ in self._backends]

    # ─── Search ───────────────────────────────────────────────────────────

    async def search(
        self,
        query: str,
        owner: Optional[str] = None,
    ) -> FederatedSearchResult:
        """Search all registered backends concurrently and return merged results.

        Failed backends are logged but never block results from healthy ones.

        Args:
            query: Free-text search query.
            owner: Optional owner/org filter applied to every backend.

        Returns:
            FederatedSearchResult with deduplicated summaries.
        """
        if not self._backends:
            logger.warning("FederatedHubRouter.search called with no backends registered")
            return FederatedSearchResult(total_backends_queried=0)

        async def _query_one(hub_type: str, backend: HubBackend) -> List[SkillSummary]:
            """Query a single backend with timeout protection."""
            try:
                return await asyncio.wait_for(
                    backend.list_remote_skills(owner=owner, query=query),
                    timeout=self._timeout,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Federated search: backend '%s' timed out after %.1fs",
                    hub_type,
                    self._timeout,
                )
                raise
            except Exception:
                logger.warning(
                    "Federated search: backend '%s' failed",
                    hub_type,
                    exc_info=True,
                )
                raise

        tasks = [
            _query_one(hub_type, backend) for hub_type, backend in self._backends
        ]
        raw_results = await asyncio.gather(*tasks, return_exceptions=True)

        # Collect successes and failures.
        all_summaries: List[SkillSummary] = []
        failed: List[str] = []
        for (hub_type, _backend), result in zip(self._backends, raw_results):
            if isinstance(result, BaseException):
                failed.append(hub_type)
            elif isinstance(result, list):
                all_summaries.extend(result)

        deduplicated = _deduplicate(all_summaries)

        # Sort by relevance proxy: downloads desc, then name asc.
        deduplicated.sort(key=lambda s: (-s.downloads, s.name))

        return FederatedSearchResult(
            results=tuple(deduplicated),
            failed_backends=tuple(failed),
            total_backends_queried=len(self._backends),
        )

    # ─── Pull Best Match ──────────────────────────────────────────────────

    async def pull_best(
        self,
        name_or_query: str,
        version: Optional[str] = None,
    ) -> Optional[SkillBundle]:
        """Search across backends and pull the best matching skill bundle.

        Finds the best match by name across all backends, then pulls from
        the backend that owns it.

        Args:
            name_or_query: Skill name or search query.
            version: Specific version to pull (None = latest).

        Returns:
            SkillBundle if found, None otherwise.
        """
        search_result = await self.search(name_or_query)
        if not search_result.results:
            logger.info("Federated pull_best: no results for '%s'", name_or_query)
            return None

        best = search_result.results[0]

        # Find the backend that owns this result.
        target_backend: Optional[HubBackend] = None
        for hub_type, backend in self._backends:
            if hub_type == best.hub_type:
                target_backend = backend
                break

        if target_backend is None:
            # Fallback: try the first available backend with the repo_id.
            for _hub_type, backend in self._backends:
                try:
                    bundle = await asyncio.wait_for(
                        backend.pull_skill(best.repo_id, version),
                        timeout=self._timeout,
                    )
                    return bundle
                except Exception:
                    continue
            logger.warning(
                "Federated pull_best: could not pull '%s' from any backend",
                best.repo_id,
            )
            return None

        try:
            return await asyncio.wait_for(
                target_backend.pull_skill(best.repo_id, version),
                timeout=self._timeout,
            )
        except Exception:
            logger.warning(
                "Federated pull_best: pull from '%s' failed for '%s'",
                best.hub_type,
                best.repo_id,
                exc_info=True,
            )
            return None
