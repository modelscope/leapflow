# Copyright (c) Alibaba, Inc. and its affiliates.
"""Skill Marketplace — curated directory with categories, featured/trending skills.

Provides a browsable marketplace layer on top of HubClient, with offline
support via a local JSON manifest cache.  The marketplace composes existing
Hub infrastructure (search, pull) and never replaces it.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from leapflow.hub.protocol import SkillSummary

logger = logging.getLogger(__name__)


# ─── Data Types ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MarketplaceCategory:
    """A single browsable category in the skill marketplace."""

    name: str
    description: str
    icon: str
    skill_count: int = 0


@dataclass(frozen=True)
class MarketplaceEntry:
    """Extended skill listing enriched with marketplace metadata.

    Carries all fields from SkillSummary plus curation signals
    (featured, trending, rating, install_count, etc.).
    """

    repo_id: str
    name: str
    description: str = ""
    version: str = ""
    downloads: int = 0
    hub_type: str = ""
    # ── marketplace extensions ──
    featured: bool = False
    trending: bool = False
    rating: float = 0.0
    install_count: int = 0
    updated_at: str = ""
    categories: Tuple[str, ...] = ()
    author: str = ""

    @classmethod
    def from_summary(
        cls,
        summary: SkillSummary,
        *,
        featured: bool = False,
        trending: bool = False,
        rating: float = 0.0,
        install_count: int = 0,
        updated_at: str = "",
        categories: Tuple[str, ...] = (),
        author: str = "",
    ) -> MarketplaceEntry:
        """Construct a MarketplaceEntry from an existing SkillSummary."""
        return cls(
            repo_id=summary.repo_id,
            name=summary.name,
            description=summary.description,
            version=summary.version,
            downloads=summary.downloads,
            hub_type=summary.hub_type,
            featured=featured,
            trending=trending,
            rating=rating,
            install_count=install_count,
            updated_at=updated_at,
            categories=categories,
            author=author,
        )


# ─── Predefined Categories ──────────────────────────────────────────────────

_DEFAULT_CATEGORIES: Tuple[MarketplaceCategory, ...] = (
    MarketplaceCategory("development", "Software development and coding tools", "🛠️"),
    MarketplaceCategory("research", "Academic and scientific research aids", "🔬"),
    MarketplaceCategory("automation", "Workflow and task automation", "⚙️"),
    MarketplaceCategory("productivity", "Personal and team productivity boosters", "📈"),
    MarketplaceCategory("security", "Security scanning and hardening", "🔒"),
    MarketplaceCategory("operations", "DevOps, SRE, and infrastructure management", "🖥️"),
    MarketplaceCategory("analysis", "Data analysis and visualization", "📊"),
    MarketplaceCategory("integration", "Third-party service connectors and bridges", "🔗"),
)


# ─── Marketplace ─────────────────────────────────────────────────────────────


class SkillMarketplace:
    """Curated marketplace layer on top of HubClient.

    Provides category browsing, featured/trending discovery, install
    statistics tracking, and offline support via a local JSON cache.

    Args:
        hub_client: HubClient instance for hub operations.
        cache_path: Path to the local marketplace cache JSON file.
        manifest_url: Optional URL to a remote JSON catalog (reserved for
            future remote manifest fetching).
    """

    def __init__(
        self,
        hub_client: Any,
        *,
        cache_path: Optional[Path] = None,
        manifest_url: str = "",
    ) -> None:
        self._hub = hub_client
        self._manifest_url = manifest_url
        self._cache_path = cache_path
        self._entries: List[MarketplaceEntry] = []
        self._install_stats: Dict[str, int] = {}
        self._loaded = False

    # ── Category API ──────────────────────────────────────────────────────

    def get_categories(self) -> List[MarketplaceCategory]:
        """Return the predefined marketplace categories with live skill counts."""
        self._ensure_loaded()
        counts: Dict[str, int] = {}
        for entry in self._entries:
            for cat in entry.categories:
                counts[cat] = counts.get(cat, 0) + 1
        return [
            MarketplaceCategory(
                name=c.name,
                description=c.description,
                icon=c.icon,
                skill_count=counts.get(c.name, 0),
            )
            for c in _DEFAULT_CATEGORIES
        ]

    # ── Discovery API ─────────────────────────────────────────────────────

    def get_featured(self) -> List[MarketplaceEntry]:
        """Return entries marked as featured or trending."""
        self._ensure_loaded()
        return [e for e in self._entries if e.featured or e.trending]

    def get_by_category(self, category: str) -> List[MarketplaceEntry]:
        """Filter entries belonging to *category*."""
        self._ensure_loaded()
        return [e for e in self._entries if category in e.categories]

    def get_trending(self, limit: int = 10) -> List[MarketplaceEntry]:
        """Top skills sorted by recent installs / downloads."""
        self._ensure_loaded()
        ranked = sorted(
            self._entries,
            key=lambda e: (e.install_count, e.downloads),
            reverse=True,
        )
        return ranked[:limit]

    def search(
        self,
        query: str,
        category: Optional[str] = None,
    ) -> List[MarketplaceEntry]:
        """Enhanced local search with optional category filter and relevance ranking."""
        self._ensure_loaded()
        q = query.lower()
        candidates = self._entries
        if category:
            candidates = [e for e in candidates if category in e.categories]

        def _score(entry: MarketplaceEntry) -> float:
            score = 0.0
            if q in entry.name.lower():
                score += 10.0
            if q in entry.description.lower():
                score += 5.0
            if entry.featured:
                score += 3.0
            if entry.trending:
                score += 2.0
            score += min(entry.downloads / 1000.0, 5.0)
            return score

        scored = [(e, _score(e)) for e in candidates]
        scored = [(e, s) for e, s in scored if s > 0]
        scored.sort(key=lambda t: t[1], reverse=True)
        return [e for e, _ in scored]

    # ── Install ───────────────────────────────────────────────────────────

    async def install(self, name_or_entry: str | MarketplaceEntry) -> str:
        """Install a skill from the marketplace via hub.pull().

        Records install statistics locally.

        Args:
            name_or_entry: Either a skill name / repo_id string or a
                MarketplaceEntry.

        Returns:
            Human-readable status message.
        """
        if isinstance(name_or_entry, MarketplaceEntry):
            if name_or_entry.hub_type:
                repo_id = f"{name_or_entry.hub_type}://{name_or_entry.repo_id}"
            else:
                repo_id = name_or_entry.repo_id
            display = name_or_entry.name
        else:
            repo_id = name_or_entry
            display = name_or_entry

        try:
            bundle = await self._hub.pull(repo_id)
            self._record_install(display)
            return (
                f"Installed '{bundle.manifest.name}' "
                f"v{bundle.manifest.version} from {repo_id}."
            )
        except Exception as exc:
            return f"Install failed for '{display}': {type(exc).__name__}: {exc}"

    # ── Refresh / Cache ───────────────────────────────────────────────────

    async def refresh(self) -> int:
        """Fetch latest catalog from hub backends and update local cache.

        Returns:
            Number of entries in the refreshed catalog.
        """
        new_entries: List[MarketplaceEntry] = []
        try:
            results = await self._hub.federated_search("")
            summaries: Sequence[SkillSummary] = (
                list(results.results) if hasattr(results, "results") else results
            )
            for s in summaries:
                cats = self._infer_categories(s)
                entry = MarketplaceEntry.from_summary(
                    s,
                    categories=tuple(cats),
                    install_count=self._install_stats.get(s.name, 0),
                    trending=s.downloads > 50,
                )
                new_entries.append(entry)
        except Exception as exc:
            logger.warning("Marketplace refresh from hub failed: %s", exc)

        if new_entries:
            self._entries = new_entries
            self._save_cache()
        self._loaded = True
        return len(self._entries)

    # ── Private Helpers ───────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Load from local cache if not already loaded."""
        if self._loaded:
            return
        self._load_cache()
        self._loaded = True

    def _load_cache(self) -> None:
        """Read marketplace_cache.json from disk."""
        if self._cache_path is None or not self._cache_path.exists():
            return
        try:
            raw = json.loads(self._cache_path.read_text(encoding="utf-8"))
            entries = raw.get("entries", [])
            if not isinstance(entries, list):
                logger.warning("Marketplace cache entries malformed; ignoring cache")
                return

            for item in entries:
                # Convert categories list to tuple for frozen dataclass
                if isinstance(item, dict) and isinstance(item.get("categories"), list):
                    item["categories"] = tuple(item["categories"])
                try:
                    self._entries.append(MarketplaceEntry(**item))
                except (TypeError, AttributeError):
                    logger.debug("Skipping malformed cache entry: %r", item)
            self._install_stats = raw.get("install_stats", {})
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load marketplace cache: %s", exc)

    def _save_cache(self) -> None:
        """Persist current entries and install stats to disk."""
        if self._cache_path is None:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "version": 1,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "entries": [_entry_to_dict(e) for e in self._entries],
                "install_stats": self._install_stats,
            }
            self._cache_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Failed to save marketplace cache: %s", exc)

    def _record_install(self, name: str) -> None:
        """Increment local install counter and persist."""
        self._install_stats[name] = self._install_stats.get(name, 0) + 1
        self._save_cache()

    @staticmethod
    def _infer_categories(summary: SkillSummary) -> List[str]:
        """Heuristic category inference from skill name/description."""
        text = f"{summary.name} {summary.description}".lower()
        cats: List[str] = []
        _KEYWORD_MAP: Dict[str, List[str]] = {
            "development": ["code", "develop", "build", "compile", "debug", "lint", "refactor"],
            "research": ["research", "paper", "academic", "experiment", "science"],
            "automation": ["automat", "workflow", "pipeline", "schedule", "batch"],
            "productivity": ["productiv", "organiz", "note", "calendar", "todo"],
            "security": ["secur", "vulnerab", "audit", "scan", "cve", "encrypt"],
            "operations": ["devops", "deploy", "monitor", "infra", "docker", "k8s", "ci/cd"],
            "analysis": ["analy", "data", "visual", "chart", "statistic", "metric"],
            "integration": ["integrat", "connect", "bridge", "api", "webhook", "sync"],
        }
        for cat, keywords in _KEYWORD_MAP.items():
            if any(kw in text for kw in keywords):
                cats.append(cat)
        return cats or ["development"]


def _entry_to_dict(entry: MarketplaceEntry) -> Dict[str, Any]:
    """Serialize a MarketplaceEntry to a JSON-safe dict."""
    d = asdict(entry)
    # Ensure categories is a list for JSON serialization
    d["categories"] = list(d.get("categories", ()))
    return d
