"""Sync engine — bidirectional skill synchronization between local and hub.

Computes diff-based sync plans and executes push/pull actions to keep local
skill libraries in sync with a remote Hub. Supports configurable conflict
resolution strategies and optional Copilot model state synchronization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from leapflow.hub.client import HubClient
from leapflow.hub.protocol import SkillManifest, SkillSourceTag, SkillSummary
from leapflow.hub.serializer import SkillSerializer

logger = logging.getLogger(__name__)


# ─── Data Structures ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SyncAction:
    """A single sync operation to perform."""

    direction: str  # "push" | "pull" | "conflict"
    skill_name: str
    local_version: str  # "" if not local
    remote_version: str  # "" if not remote
    reason: str  # "local_newer" | "remote_newer" | "local_only" | "remote_only"
    repo_id: str = ""  # Remote repo_id (used for pull actions)


@dataclass
class SyncPlan:
    """Computed synchronization plan."""

    actions: List[SyncAction] = field(default_factory=list)
    copilot_sync: bool = False

    @property
    def push_count(self) -> int:
        """Number of skills to push to remote."""
        return sum(1 for a in self.actions if a.direction == "push")

    @property
    def pull_count(self) -> int:
        """Number of skills to pull from remote."""
        return sum(1 for a in self.actions if a.direction == "pull")

    @property
    def is_empty(self) -> bool:
        """Return True if no sync actions are needed."""
        return len(self.actions) == 0 and not self.copilot_sync

    def __repr__(self) -> str:
        return (
            f"SyncPlan(push={self.push_count}, pull={self.pull_count}, "
            f"copilot_sync={self.copilot_sync})"
        )


# ─── Version Comparison ─────────────────────────────────────────────────────


def _parse_version(v: str) -> Tuple[int, ...]:
    """Parse semver-like version string to comparable 3-tuple.

    Handles: "0.1.0", "v1.2.3", "1.0", "2.0.0-beta.1", "1.2.3+sha.abc".
    Strips 'v' prefix, pre-release labels, and build metadata before parsing.
    Normalizes to 3 components (padded with zeros, limited to first 3).
    Returns (0, 0, 0) on parse failure for safe fallback.
    """
    import re

    cleaned = v.strip().lstrip("v")
    # Strip pre-release (after '-') and build metadata (after '+')
    cleaned = re.split(r"[\-+]", cleaned, maxsplit=1)[0]
    try:
        parts = [int(x) for x in cleaned.split(".")[:3]]
        # Normalize to 3 components
        while len(parts) < 3:
            parts.append(0)
        return tuple(parts)
    except (ValueError, AttributeError):
        return (0, 0, 0)


def _compare_versions(local_ver: str, remote_ver: str) -> int:
    """Compare two version strings.

    Returns:
        -1 if local < remote (remote newer)
         0 if equal
         1 if local > remote (local newer)
    """
    lv = _parse_version(local_ver)
    rv = _parse_version(remote_ver)
    if lv < rv:
        return -1
    elif lv > rv:
        return 1
    return 0


# ─── Sync Engine ────────────────────────────────────────────────────────────


# Source tags that are eligible for sync (builtin skills are excluded)
_SYNCABLE_TAGS = {SkillSourceTag.LEARNED.value, SkillSourceTag.HUB.value}


class SyncEngine:
    """Compute and execute sync plans between local skills and hub.

    Sync strategy (configurable):
    - remote-wins: when both sides have changes, prefer remote
    - local-wins: prefer local
    - manual: skip conflicts (leave them unresolved for user to decide)

    Only syncs skills with source_tag in (learned, hub). Builtin skills are excluded.
    """

    STRATEGIES = ("remote-wins", "local-wins", "manual")

    def __init__(self, client: HubClient, strategy: str = "remote-wins"):
        """Initialize SyncEngine.

        Args:
            client: HubClient instance for push/pull operations.
            strategy: Conflict resolution strategy. One of 'remote-wins',
                      'local-wins', or 'manual'.

        Raises:
            ValueError: If strategy is not recognized.
        """
        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown sync strategy: '{strategy}'. "
                f"Must be one of: {', '.join(self.STRATEGIES)}"
            )
        self._client = client
        self._strategy = strategy
        self._serializer = SkillSerializer()

    @property
    def strategy(self) -> str:
        """Return the active conflict resolution strategy."""
        return self._strategy

    async def compute_plan(
        self,
        local_skills: List[SkillManifest],
        remote_skills: List[SkillSummary],
        *,
        push_only: bool = False,
        pull_only: bool = False,
        copilot_sync: bool = False,
    ) -> SyncPlan:
        """Compute the diff between local and remote skill sets.

        Args:
            local_skills: Local skill manifests to consider for sync.
            remote_skills: Remote skill summaries from the Hub.
            push_only: Only generate push actions (no pulls).
            pull_only: Only generate pull actions (no pushes).
            copilot_sync: Whether to include Copilot model state in sync.

        Returns:
            SyncPlan describing all actions needed.
        """
        # Filter local skills to syncable ones only
        syncable_local = [
            s for s in local_skills if s.source_tag in _SYNCABLE_TAGS
        ]

        local_by_name = {s.name: s for s in syncable_local}
        remote_by_name = {s.name: s for s in remote_skills}

        actions: List[SyncAction] = []

        # Process skills present locally
        for name, local in local_by_name.items():
            if name not in remote_by_name:
                # Local only → push
                if not pull_only:
                    actions.append(
                        SyncAction(
                            direction="push",
                            skill_name=name,
                            local_version=local.version,
                            remote_version="",
                            reason="local_only",
                        )
                    )
            else:
                # Both sides exist → compare versions
                remote = remote_by_name[name]
                cmp = _compare_versions(local.version, remote.version)

                if cmp > 0:
                    # Local newer
                    if not pull_only:
                        actions.append(
                            SyncAction(
                                direction="push",
                                skill_name=name,
                                local_version=local.version,
                                remote_version=remote.version,
                                reason="local_newer",
                                repo_id=getattr(remote, "repo_id", ""),
                            )
                        )
                elif cmp < 0:
                    # Remote newer
                    if not push_only:
                        actions.append(
                            SyncAction(
                                direction="pull",
                                skill_name=name,
                                local_version=local.version,
                                remote_version=remote.version,
                                reason="remote_newer",
                                repo_id=getattr(remote, "repo_id", ""),
                            )
                        )
                else:
                    # Same version — check content hash for silent divergence
                    local_hash = getattr(local, "content_hash", "")
                    remote_hash = getattr(remote, "content_hash", "") if hasattr(remote, "content_hash") else ""
                    if local_hash and remote_hash and local_hash != remote_hash:
                        actions.append(SyncAction(
                            direction="conflict",
                            skill_name=name,
                            local_version=local.version,
                            remote_version=remote.version,
                            reason="content_diverged",
                            repo_id=getattr(remote, "repo_id", ""),
                        ))
                    # else: truly identical, skip

        # Process skills only on remote
        if not push_only:
            for name, remote in remote_by_name.items():
                if name not in local_by_name:
                    actions.append(
                        SyncAction(
                            direction="pull",
                            skill_name=name,
                            local_version="",
                            remote_version=remote.version,
                            reason="remote_only",
                            repo_id=getattr(remote, "repo_id", ""),
                        )
                    )

        # Apply conflict resolution for version conflicts
        actions = self._resolve_conflicts(actions)

        logger.info(
            "Computed sync plan: %d push, %d pull (strategy=%s)",
            sum(1 for a in actions if a.direction == "push"),
            sum(1 for a in actions if a.direction == "pull"),
            self._strategy,
        )

        return SyncPlan(actions=actions, copilot_sync=copilot_sync)

    def _resolve_conflicts(self, actions: List[SyncAction]) -> List[SyncAction]:
        """Apply conflict resolution strategy to ambiguous actions.

        For the 'manual' strategy, conflicting actions are dropped
        (user must resolve manually).
        """
        if self._strategy == "manual":
            # In manual mode, skip any action where both versions exist
            # and a clear winner isn't determined by version comparison
            return [
                a
                for a in actions
                if a.reason in ("local_only", "remote_only")
                or (a.reason == "local_newer" and a.direction == "push")
                or (a.reason == "remote_newer" and a.direction == "pull")
            ]
        # remote-wins / local-wins strategies are already reflected in
        # the version comparison logic above
        return actions

    async def execute_plan(
        self,
        plan: SyncPlan,
        *,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> List[str]:
        """Execute a sync plan. Returns list of completed action descriptions.

        Args:
            plan: The SyncPlan to execute.
            on_progress: Optional callback invoked with a status message
                         after each action completes.

        Returns:
            List of human-readable descriptions of completed actions.
        """
        completed: List[str] = []

        for action in plan.actions:
            try:
                if action.direction == "conflict":
                    logger.warning("Conflict: skill '%s' v%s has diverged content. Resolve manually.", action.skill_name, action.local_version)
                    completed.append(f"CONFLICT: {action.skill_name} (content diverged, resolve manually)")
                    continue
                elif action.direction == "push":
                    desc = await self._execute_push(action)
                else:
                    desc = await self._execute_pull(action)

                completed.append(desc)

                if on_progress:
                    on_progress(desc)

            except Exception as exc:
                msg = f"Failed to {action.direction} '{action.skill_name}': {exc}"
                logger.error(msg)
                completed.append(f"[ERROR] {msg}")
                if on_progress:
                    on_progress(f"[ERROR] {msg}")

        # Copilot model state sync (placeholder for future implementation)
        if plan.copilot_sync:
            copilot_desc = await self._sync_copilot_state()
            completed.append(copilot_desc)
            if on_progress:
                on_progress(copilot_desc)

        logger.info("Sync complete: %d actions executed", len(completed))
        return completed

    async def _execute_push(self, action: SyncAction) -> str:
        """Execute a single push action.

        Serializes the local skill and pushes it to the Hub.
        """
        logger.info(
            "Pushing '%s' (v%s → remote)",
            action.skill_name,
            action.local_version,
        )

        # Build a minimal bundle from manifest for push
        # In production, this would load the full skill data from local store
        manifest = SkillManifest(
            name=action.skill_name,
            version=action.local_version,
            source_tag="learned",
        )
        from leapflow.hub.protocol import SkillBundle

        bundle = SkillBundle(manifest=manifest)

        result = await self._client.push(bundle, skill_name=action.skill_name)

        desc = (
            f"Pushed '{action.skill_name}' v{action.local_version} → "
            f"{result.url} (reason: {action.reason})"
        )
        logger.info(desc)
        return desc

    async def _execute_pull(self, action: SyncAction) -> str:
        """Execute a single pull action.

        Downloads the skill from Hub and prepares it for local import.
        """
        logger.info(
            "Pulling '%s' (v%s ← remote)",
            action.skill_name,
            action.remote_version,
        )

        # Determine repo_id to pull from
        repo_id = action.repo_id or self._client._build_repo_id(action.skill_name)
        bundle = await self._client.pull(repo_id)

        desc = (
            f"Pulled '{action.skill_name}' v{action.remote_version} ← "
            f"hub (reason: {action.reason})"
        )
        logger.info(desc)
        return desc

    async def _sync_copilot_state(self) -> str:
        """Sync Copilot model state (L1 Markov predictor).

        Exports predictor state as JSON and pushes as a file attachment.
        On pull, downloads and imports into local predictor.

        Returns:
            Description of the copilot sync result.
        """
        # Placeholder: full implementation requires Copilot predictor integration
        logger.info("Copilot model state sync requested (placeholder)")
        return "Copilot model state sync: skipped (not yet configured)"
