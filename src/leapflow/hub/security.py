"""Content sanitization and security audit for hub operations.

Scans SkillBundle content for sensitive data (before push) and dangerous
operations (after pull) using regex-based pattern matching.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, List, Pattern

if TYPE_CHECKING:
    from leapflow.hub.protocol import SkillBundle

logger = logging.getLogger(__name__)


# ─── Data Types ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SanitizationWarning:
    """A single sanitization finding."""

    severity: str  # "high" | "medium" | "low"
    category: str  # "absolute_path" | "token" | "pii" | "dangerous_code"
    detail: str
    line_number: int | None = None


# ─── Content Sanitizer ───────────────────────────────────────────────────────


class ContentSanitizer:
    """Scan skill content for sensitive data before push."""

    # Patterns: (compiled regex, severity, category, detail template)
    PATTERNS: ClassVar[List[tuple]] = [
        # Absolute paths (Unix & Windows)
        (
            re.compile(r"(/Users/[^\s\"']+|/home/[^\s\"']+|[A-Z]:\\[^\s\"']+)"),
            "medium",
            "absolute_path",
            "Absolute path detected: {match}",
        ),
        # API tokens / secrets (generic patterns)
        (
            re.compile(
                r"(?i)(api[_-]?key|api[_-]?token|secret[_-]?key|access[_-]?token"
                r"|auth[_-]?token)\s*[=:]\s*[\"']?[a-zA-Z0-9_\-]{16,}[\"']?"
            ),
            "high",
            "token",
            "Potential secret/token: {match}",
        ),
        # Bearer tokens
        (
            re.compile(r"Bearer\s+[a-zA-Z0-9_\-.]{20,}"),
            "high",
            "token",
            "Bearer token detected: {match}",
        ),
        # Email addresses
        (
            re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"),
            "low",
            "pii",
            "Email address: {match}",
        ),
        # Phone numbers (international and local formats)
        (
            re.compile(r"(?<!\d)(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}(?!\d)"),
            "low",
            "pii",
            "Possible phone number: {match}",
        ),
    ]

    def scan(self, bundle: SkillBundle) -> List[SanitizationWarning]:
        """Scan bundle content for potential sensitive data leaks.

        Inspects source_code, trajectory_skeleton, copilot_prior, and readme.
        """
        warnings: List[SanitizationWarning] = []

        content_fields = [
            ("source_code", bundle.source_code),
            ("trajectory_skeleton", bundle.trajectory_skeleton),
            ("copilot_prior", bundle.copilot_prior),
            ("readme", bundle.readme),
        ]

        for field_name, content in content_fields:
            if not content:
                continue
            self._scan_content(content, field_name, warnings)

        if warnings:
            logger.warning(
                "Sanitization scan found %d warning(s) in bundle '%s'",
                len(warnings),
                bundle.manifest.name,
            )
        return warnings

    def _scan_content(
        self,
        content: str,
        field_name: str,
        warnings: List[SanitizationWarning],
    ) -> None:
        """Scan a single content field line by line."""
        lines = content.splitlines()
        for line_idx, line in enumerate(lines, start=1):
            for pattern, severity, category, detail_tpl in self.PATTERNS:
                for match in pattern.finditer(line):
                    matched_text = match.group(0)
                    # Truncate long matches for readability
                    display = matched_text[:80] + "..." if len(matched_text) > 80 else matched_text
                    warnings.append(
                        SanitizationWarning(
                            severity=severity,
                            category=category,
                            detail=f"[{field_name}] {detail_tpl.format(match=display)}",
                            line_number=line_idx,
                        )
                    )


# ─── Security Auditor ────────────────────────────────────────────────────────


class SecurityAuditor:
    """Audit pulled skill code for dangerous operations."""

    # Dangerous code patterns: (compiled regex, severity, detail)
    DANGEROUS_PATTERNS: ClassVar[List[tuple]] = [
        (
            re.compile(r"\bos\.system\s*\("),
            "high",
            "os.system() call — arbitrary command execution",
        ),
        (
            re.compile(r"\bsubprocess\b"),
            "high",
            "subprocess usage — arbitrary command execution",
        ),
        (
            re.compile(r"\bexec\s*\("),
            "high",
            "exec() call — arbitrary code execution",
        ),
        (
            re.compile(r"\beval\s*\("),
            "high",
            "eval() call — arbitrary code execution",
        ),
        (
            re.compile(r"\b__import__\s*\("),
            "medium",
            "__import__() — dynamic module import",
        ),
        (
            re.compile(r"\brequests\.(get|post|put|delete|patch)\s*\("),
            "medium",
            "Network request — potential data exfiltration",
        ),
        (
            re.compile(r"\bshutil\.rmtree\s*\("),
            "high",
            "shutil.rmtree() — recursive directory deletion",
        ),
        (
            re.compile(r"\bos\.remove\s*\("),
            "medium",
            "os.remove() — file deletion",
        ),
        (
            re.compile(r"\bos\.chmod\s*\("),
            "medium",
            "os.chmod() — permission modification",
        ),
        (
            re.compile(r"\bos\.unlink\s*\("),
            "medium",
            "os.unlink() — file deletion",
        ),
        (
            re.compile(r"\bopen\s*\(.+,\s*['\"]w"),
            "low",
            "File write operation detected",
        ),
    ]

    def audit(self, bundle: SkillBundle) -> List[SanitizationWarning]:
        """Audit bundle code for potentially dangerous operations.

        Inspects source_code and trajectory_skeleton for dangerous patterns.
        """
        warnings: List[SanitizationWarning] = []

        audit_fields = [
            ("source_code", bundle.source_code),
            ("trajectory_skeleton", bundle.trajectory_skeleton),
        ]

        for field_name, content in audit_fields:
            if not content:
                continue
            self._audit_content(content, field_name, warnings)

        if warnings:
            logger.warning(
                "Security audit found %d warning(s) in bundle '%s'",
                len(warnings),
                bundle.manifest.name,
            )
        return warnings

    def _audit_content(
        self,
        content: str,
        field_name: str,
        warnings: List[SanitizationWarning],
    ) -> None:
        """Audit a single content field for dangerous patterns."""
        lines = content.splitlines()
        for line_idx, line in enumerate(lines, start=1):
            # Skip comment lines
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            for pattern, severity, detail in self.DANGEROUS_PATTERNS:
                if pattern.search(line):
                    warnings.append(
                        SanitizationWarning(
                            severity=severity,
                            category="dangerous_code",
                            detail=f"[{field_name}] {detail}",
                            line_number=line_idx,
                        )
                    )
