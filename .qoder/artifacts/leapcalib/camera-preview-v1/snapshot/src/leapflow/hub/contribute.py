# Copyright (c) Alibaba, Inc. and its affiliates.
"""Community contribution pipeline — prepare, submit, and track skill contributions.

Provides a structured workflow for contributing skills back to the Hub:
DRAFT → SUBMITTED → UNDER_REVIEW → APPROVED/REJECTED → PUBLISHED.
Contributions are tracked locally in a JSON file per profile.
"""

from __future__ import annotations

import enum
import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ─── Data Types ──────────────────────────────────────────────────────────────


@enum.unique
class ContributionStatus(enum.Enum):
    """Lifecycle status of a community skill contribution."""

    DRAFT = "draft"
    SUBMITTED = "submitted"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    PUBLISHED = "published"


@dataclass(frozen=True)
class ContributionRecord:
    """Persistent record tracking one skill contribution."""

    skill_name: str
    author: str
    status: str  # ContributionStatus value
    submitted_at: str = ""
    review_notes: str = ""
    hub_type: str = ""
    repo_id: str = ""


# ─── Contributor Pipeline ────────────────────────────────────────────────────


class CommunityContributor:
    """End-to-end community contribution pipeline.

    Validates, sanitizes, and submits local skills to the Hub,
    then tracks the contribution lifecycle in a local JSON store.

    Args:
        hub_client: HubClient instance for push operations.
        store_path: Path to the contributions JSON file.
    """

    def __init__(
        self,
        hub_client: Any,
        *,
        store_path: Optional[Path] = None,
    ) -> None:
        self._hub = hub_client
        self._store_path = store_path
        self._records: Dict[str, ContributionRecord] = {}
        self._loaded = False

    # ── Public API ────────────────────────────────────────────────────────

    async def prepare(self, skill_name: str, *, ctx: Any = None) -> str:
        """Validate a local skill and generate a sanitized SkillBundle.

        Runs ContentSanitizer to detect secrets/PII before submission.

        Args:
            skill_name: Name of the local skill to prepare.
            ctx: Optional runtime context with skill_lib access.

        Returns:
            Human-readable preparation report.
        """
        from leapflow.hub.security import ContentSanitizer
        from leapflow.hub.serializer import SkillSerializer

        # Load skill data from context
        stored_dict = self._load_skill_from_ctx(skill_name, ctx)
        if isinstance(stored_dict, str):
            return stored_dict  # error message

        # Serialize to bundle
        serializer = SkillSerializer()
        bundle = serializer.export_skill(stored_dict)

        # Sanitize
        sanitizer = ContentSanitizer()
        warnings = sanitizer.scan(bundle)

        high = sum(1 for w in warnings if w.severity == "high")
        medium = sum(1 for w in warnings if w.severity == "medium")

        # Create or update draft record
        self._ensure_loaded()
        record = ContributionRecord(
            skill_name=skill_name,
            author=stored_dict.get("author", ""),
            status=ContributionStatus.DRAFT.value,
        )
        self._records[skill_name] = record
        self._save_store()

        lines = [f"Prepared '{skill_name}' for contribution."]
        if warnings:
            lines.append(f"  Sanitization: {high} high, {medium} medium, "
                         f"{len(warnings) - high - medium} low warning(s).")
            if high > 0:
                lines.append("  ⚠ High-risk issues must be resolved before submission.")
                for w in warnings:
                    if w.severity == "high":
                        lines.append(f"    - {w.detail}")
        else:
            lines.append("  ✓ No sanitization warnings — ready to submit.")
        lines.append(f"  Status: {ContributionStatus.DRAFT.value}")
        return "\n".join(lines)

    async def submit(
        self,
        skill_name: str,
        hub_type: str = "github",
        *,
        ctx: Any = None,
    ) -> str:
        """Push skill to hub with PUBLIC visibility and create contribution record.

        Args:
            skill_name: Name of the local skill to submit.
            hub_type: Target hub backend (default: 'github').
            ctx: Optional runtime context with skill_lib access.

        Returns:
            Human-readable submission result.
        """
        from leapflow.hub.protocol import Visibility
        from leapflow.hub.security import ContentSanitizer
        from leapflow.hub.serializer import SkillSerializer

        stored_dict = self._load_skill_from_ctx(skill_name, ctx)
        if isinstance(stored_dict, str):
            return stored_dict

        serializer = SkillSerializer()
        bundle = serializer.export_skill(stored_dict)

        # Pre-flight sanitization check
        sanitizer = ContentSanitizer()
        warnings = sanitizer.scan(bundle)
        high_warnings = [w for w in warnings if w.severity == "high"]
        if high_warnings:
            return (
                f"Cannot submit '{skill_name}': {len(high_warnings)} high-risk "
                f"sanitization warning(s). Run prepare() first to review."
            )

        # Push to hub
        try:
            result = await self._hub.push(
                bundle,
                skill_name=skill_name,
                visibility=Visibility.PUBLIC,
            )
        except Exception as exc:
            return f"Submission failed: {type(exc).__name__}: {exc}"

        # Record contribution
        self._ensure_loaded()
        actual_hub_type = getattr(self._hub, 'hub_type', hub_type)
        record = ContributionRecord(
            skill_name=skill_name,
            author=stored_dict.get("author", ""),
            status=ContributionStatus.SUBMITTED.value,
            submitted_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            hub_type=actual_hub_type,
            repo_id=result.repo_id,
        )
        self._records[skill_name] = record
        self._save_store()

        return (
            f"Submitted '{skill_name}' to {result.repo_id} ({actual_hub_type}).\n"
            f"  Version: v{result.version}\n"
            f"  URL: {result.url}\n"
            f"  Status: {ContributionStatus.SUBMITTED.value}"
        )

    def check_status(self, skill_name: str) -> ContributionStatus:
        """Return the current ContributionStatus for a skill.

        Args:
            skill_name: Name of the contributed skill.

        Returns:
            Current status enum value.

        Raises:
            KeyError: If no contribution record exists for the skill.
        """
        self._ensure_loaded()
        record = self._records.get(skill_name)
        if record is None:
            raise KeyError(f"No contribution record for '{skill_name}'.")
        return ContributionStatus(record.status)

    def list_my_contributions(self) -> List[ContributionRecord]:
        """Return all contribution records from local store."""
        self._ensure_loaded()
        return list(self._records.values())

    # ── Private Helpers ───────────────────────────────────────────────────

    @staticmethod
    def _load_skill_from_ctx(
        skill_name: str, ctx: Any
    ) -> Dict[str, Any] | str:
        """Load skill data from context's skill library.

        Returns dict on success, or an error string on failure.
        """
        if ctx is None or not hasattr(ctx, "skill_lib") or ctx.skill_lib is None:
            return "Error: Skill library context not available."

        try:
            stored = ctx.skill_lib.load_skill_by_title(skill_name)
            if stored is None:
                return f"Error: Skill '{skill_name}' not found in local library."
            return {
                "name": getattr(stored, "title", skill_name),
                "version": getattr(stored, "version", "0.1.0"),
                "description": getattr(stored, "description", ""),
                "source_code": getattr(stored, "source_code", ""),
                "parameters": getattr(stored, "parameters", []),
                "triggers": list(getattr(stored, "trigger_phrases", [])),
                "trajectory_skeleton": getattr(stored, "trajectory_skeleton", ""),
                "copilot_prior": getattr(stored, "copilot_prior", ""),
                "readme": getattr(stored, "readme", f"# {skill_name}\n"),
                "source_tag": getattr(stored, "source_tag", "learned"),
                "tier": getattr(stored, "tier", 1),
                "author": getattr(stored, "author", ""),
            }
        except Exception as exc:
            return f"Error loading skill '{skill_name}': {exc}"

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._load_store()
        self._loaded = True

    def _load_store(self) -> None:
        """Read contributions.json from disk."""
        if self._store_path is None or not self._store_path.exists():
            return
        try:
            raw = json.loads(self._store_path.read_text(encoding="utf-8"))
            for item in raw.get("contributions", []):
                try:
                    record = ContributionRecord(**item)
                    self._records[record.skill_name] = record
                except (TypeError, AttributeError):
                    logger.debug("Skipping malformed contribution entry: %r", item)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load contributions store: %s", exc)

    def _save_store(self) -> None:
        """Persist contribution records to disk."""
        if self._store_path is None:
            return
        try:
            self._store_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "version": 1,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "contributions": [asdict(r) for r in self._records.values()],
            }
            self._store_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Failed to save contributions store: %s", exc)
