"""Standard Skill Document model (Anthropic Agent Skills format).

Provides the data model, renderer, and parser for SKILL.md files that conform
to the Anthropic standard: YAML frontmatter + Markdown body with progressive
disclosure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ParameterDoc:
    """A skill parameter declaration."""

    name: str
    type: str = "str"
    required: bool = False
    default: Optional[str] = None
    description: str = ""


@dataclass
class ExampleDoc:
    """A usage example showing trigger → actions → result."""

    trigger: str
    actions: List[str] = field(default_factory=list)
    result: str = ""


@dataclass
class ErrorHandlingEntry:
    """A structured error-recovery rule."""

    pattern: str
    signal: str = ""
    recovery: str = ""
    script: str = ""


@dataclass
class ProvenanceEntry:
    """A source trajectory reference."""

    trajectory_id: str
    date: str = ""
    notes: str = ""
    reference: str = ""


@dataclass
class SkillDocument:
    """Full representation of a standard Agent Skill document."""

    name: str
    description: str
    goal: str = ""
    allowed_tools: str = ""
    parameters: List[ParameterDoc] = field(default_factory=list)
    instructions: List[str] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    error_handling: List[ErrorHandlingEntry] = field(default_factory=list)
    examples: List[ExampleDoc] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    procedure_graph: str = ""
    provenance: List[ProvenanceEntry] = field(default_factory=list)

    source_trajectory_id: str = ""
    source_episode_id: str = ""
    learned_pattern: str = ""


def _escape_pipe(text: str) -> str:
    """Escape pipe characters in Markdown table cell content."""
    return text.replace("|", "\\|")


def _split_table_row(line: str) -> List[str]:
    """Split a Markdown table row into cells, respecting escaped pipes."""
    inner = line.strip().strip("|")
    cells: List[str] = []
    current: List[str] = []
    i = 0
    while i < len(inner):
        if inner[i] == "\\" and i + 1 < len(inner) and inner[i + 1] == "|":
            current.append("|")
            i += 2
        elif inner[i] == "|":
            cells.append("".join(current))
            current = []
            i += 1
        else:
            current.append(inner[i])
            i += 1
    cells.append("".join(current))
    return cells


class SkillDocRenderer:
    """Renders a SkillDocument to standard SKILL.md format."""

    def render(self, doc: SkillDocument) -> str:
        frontmatter = self._render_frontmatter(doc)
        body = self._render_body(doc)
        return f"---\n{frontmatter}---\n\n{body}"

    def _render_frontmatter(self, doc: SkillDocument) -> str:
        data: Dict[str, Any] = {
            "name": doc.name,
            "description": doc.description,
        }
        if doc.allowed_tools:
            data["allowed-tools"] = doc.allowed_tools
        if doc.metadata:
            data["metadata"] = doc.metadata
        return yaml.dump(data, default_flow_style=False, allow_unicode=True, sort_keys=False)

    def _render_body(self, doc: SkillDocument) -> str:
        sections: List[str] = []

        title = doc.name.replace("-", " ").title()
        sections.append(f"# {title}\n")

        if doc.goal:
            sections.append(f"## Goal\n\n{doc.goal}\n")

        if doc.parameters:
            lines = ["## Parameters\n"]
            for p in doc.parameters:
                req = " (required)" if p.required else ""
                default = f" Default: `{p.default}`" if p.default else ""
                lines.append(f"- **{p.name}** (`{p.type}`{req}): {p.description}{default}")
            sections.append("\n".join(lines) + "\n")

        if doc.procedure_graph:
            sections.append(
                f"## Procedure\n\n```mermaid\n{doc.procedure_graph}\n```\n"
            )

        if doc.instructions:
            lines = ["## Instructions\n"]
            for i, step in enumerate(doc.instructions, 1):
                lines.append(f"### Step {i}\n\n{step}\n")
            sections.append("\n".join(lines))

        if doc.preconditions:
            lines = ["## Preconditions\n"]
            for cond in doc.preconditions:
                lines.append(f"- {cond}")
            sections.append("\n".join(lines) + "\n")

        if doc.postconditions:
            lines = ["## Postconditions\n"]
            for cond in doc.postconditions:
                lines.append(f"- {cond}")
            sections.append("\n".join(lines) + "\n")

        if doc.error_handling:
            lines = [
                "## Error Handling\n",
                "| Pattern | Signal | Recovery | Script |",
                "|---------|--------|----------|--------|",
            ]
            for entry in doc.error_handling:
                lines.append(
                    f"| {_escape_pipe(entry.pattern)} "
                    f"| {_escape_pipe(entry.signal)} "
                    f"| {_escape_pipe(entry.recovery)} "
                    f"| {_escape_pipe(entry.script)} |"
                )
            sections.append("\n".join(lines) + "\n")

        if doc.examples:
            lines = ["## Examples\n"]
            for j, ex in enumerate(doc.examples, 1):
                lines.append(f"### Example {j}\n")
                lines.append(f"User says: \"{ex.trigger}\"\n")
                if ex.actions:
                    lines.append("Actions:")
                    for a in ex.actions:
                        lines.append(f"1. {a}")
                if ex.result:
                    lines.append(f"\nResult: {ex.result}")
                lines.append("")
            sections.append("\n".join(lines))

        if doc.provenance:
            lines = [
                "## Provenance\n",
                "| Trajectory ID | Date | Notes | Reference |",
                "|---------------|------|-------|-----------|",
            ]
            for entry in doc.provenance:
                ref = f"[summary]({entry.reference})" if entry.reference else ""
                lines.append(
                    f"| {_escape_pipe(entry.trajectory_id)} "
                    f"| {_escape_pipe(entry.date)} "
                    f"| {_escape_pipe(entry.notes)} "
                    f"| {ref} |"
                )
            sections.append("\n".join(lines) + "\n")

        return "\n".join(sections)


class SkillDocParser:
    """Parses a SKILL.md string back into a SkillDocument."""

    def parse(self, content: str) -> SkillDocument:
        frontmatter, body = self._split_frontmatter(content)
        fm = yaml.safe_load(frontmatter) or {}

        doc = SkillDocument(
            name=fm.get("name", ""),
            description=fm.get("description", ""),
            allowed_tools=fm.get("allowed-tools", ""),
            metadata=fm.get("metadata", {}),
        )

        self._parse_body(body, doc)
        return doc

    def _split_frontmatter(self, content: str) -> tuple[str, str]:
        match = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)", content, re.DOTALL)
        if match:
            return match.group(1), match.group(2)
        return "", content

    def _parse_body(self, body: str, doc: SkillDocument) -> None:
        current_section = ""
        current_content: List[str] = []

        for line in body.split("\n"):
            if line.startswith("## "):
                if current_section:
                    self._assign_section(doc, current_section, current_content)
                current_section = line[3:].strip()
                current_content = []
            else:
                current_content.append(line)

        if current_section:
            self._assign_section(doc, current_section, current_content)

    def _assign_section(self, doc: SkillDocument, section: str, lines: List[str]) -> None:
        text = "\n".join(lines).strip()
        section_lower = section.lower()

        if section_lower == "goal":
            doc.goal = text
        elif section_lower == "parameters":
            doc.parameters = self._parse_parameters(lines)
        elif section_lower == "procedure":
            doc.procedure_graph = self._parse_mermaid(lines)
        elif section_lower == "instructions":
            doc.instructions = self._parse_steps(lines)
        elif section_lower == "preconditions":
            doc.preconditions = self._parse_list(lines)
        elif section_lower == "postconditions":
            doc.postconditions = self._parse_list(lines)
        elif section_lower == "error handling":
            doc.error_handling = self._parse_error_handling(lines)
        elif section_lower == "examples":
            doc.examples = self._parse_examples(lines)
        elif section_lower == "provenance":
            doc.provenance = self._parse_provenance(lines)

    def _parse_parameters(self, lines: List[str]) -> List[ParameterDoc]:
        params: List[ParameterDoc] = []
        for line in lines:
            match = re.match(
                r"^- \*\*(\w+)\*\*\s*\(`(\w+)`(\s*\(required\))?\):\s*(.*)",
                line.strip(),
            )
            if match:
                name, ptype, req, desc = match.groups()
                params.append(ParameterDoc(
                    name=name,
                    type=ptype,
                    required=bool(req),
                    description=desc.strip(),
                ))
        return params

    def _parse_steps(self, lines: List[str]) -> List[str]:
        steps: List[str] = []
        current: List[str] = []
        for line in lines:
            if line.startswith("### Step"):
                if current:
                    steps.append("\n".join(current).strip())
                current = []
            else:
                current.append(line)
        if current:
            steps.append("\n".join(current).strip())
        return [s for s in steps if s]

    def _parse_list(self, lines: List[str]) -> List[str]:
        items: List[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("- "):
                items.append(stripped[2:])
        return items

    def _parse_mermaid(self, lines: List[str]) -> str:
        """Extract Mermaid code block content."""
        inside = False
        mermaid_lines: List[str] = []
        for line in lines:
            stripped = line.strip()
            if stripped == "```mermaid":
                inside = True
            elif stripped == "```" and inside:
                break
            elif inside:
                mermaid_lines.append(line)
        return "\n".join(mermaid_lines).strip()

    def _parse_error_handling(self, lines: List[str]) -> List[ErrorHandlingEntry]:
        """Parse error handling as Markdown table or bullet list fallback."""
        entries = self._parse_md_table(lines)
        if entries:
            return [
                ErrorHandlingEntry(
                    pattern=row.get("pattern", row.get("异常模式", "")),
                    signal=row.get("signal", row.get("检测信号", "")),
                    recovery=row.get("recovery", row.get("恢复动作", "")),
                    script=row.get("script", row.get("脚本", "")),
                )
                for row in entries
            ]
        return [
            ErrorHandlingEntry(pattern=item)
            for item in self._parse_list(lines)
        ]

    def _parse_provenance(self, lines: List[str]) -> List[ProvenanceEntry]:
        """Parse provenance as Markdown table."""
        entries = self._parse_md_table(lines)
        return [
            ProvenanceEntry(
                trajectory_id=row.get("trajectory id", row.get("轨迹 id", "")),
                date=row.get("date", row.get("日期", "")),
                notes=row.get("notes", row.get("路径特点", "")),
                reference=self._extract_link_target(
                    row.get("reference", row.get("参考", ""))
                ),
            )
            for row in entries
        ]

    def _parse_md_table(self, lines: List[str]) -> List[Dict[str, str]]:
        """Generic Markdown table parser. Returns list of row dicts keyed by lowercase header."""
        header_line = ""
        data_lines: List[str] = []
        found_separator = False
        for line in lines:
            stripped = line.strip()
            if not stripped or not stripped.startswith("|"):
                if found_separator:
                    continue
                continue
            if re.match(r"^\|[\s\-:|]+\|$", stripped):
                found_separator = True
                continue
            if not found_separator:
                header_line = stripped
            else:
                data_lines.append(stripped)

        if not header_line or not data_lines:
            return []

        headers = [h.strip().lower() for h in _split_table_row(header_line)]
        rows: List[Dict[str, str]] = []
        for dl in data_lines:
            cells = [c.strip() for c in _split_table_row(dl)]
            row = {headers[i]: cells[i] for i in range(min(len(headers), len(cells)))}
            rows.append(row)
        return rows

    @staticmethod
    def _extract_link_target(text: str) -> str:
        """Extract URL from a Markdown link ``[text](url)``."""
        match = re.search(r"\[.*?\]\((.+?)\)", text)
        return match.group(1) if match else text.strip()

    def _parse_examples(self, lines: List[str]) -> List[ExampleDoc]:
        examples: List[ExampleDoc] = []
        current_trigger = ""
        current_actions: List[str] = []
        current_result = ""

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("### Example"):
                if current_trigger:
                    examples.append(ExampleDoc(
                        trigger=current_trigger,
                        actions=current_actions,
                        result=current_result,
                    ))
                current_trigger = ""
                current_actions = []
                current_result = ""
            elif stripped.startswith("User says:"):
                match = re.search(r'"(.+)"', stripped)
                current_trigger = match.group(1) if match else stripped[10:].strip()
            elif re.match(r"^\d+\.\s", stripped):
                current_actions.append(re.sub(r"^\d+\.\s*", "", stripped))
            elif stripped.startswith("Result:"):
                current_result = stripped[7:].strip()

        if current_trigger:
            examples.append(ExampleDoc(
                trigger=current_trigger,
                actions=current_actions,
                result=current_result,
            ))
        return examples


def title_to_kebab(title: str) -> str:
    """Convert a skill title to kebab-case name.

    Handles ASCII titles directly. For CJK, uses transliteration-free
    approach: keeps alphanumeric chars, replaces separators with hyphens.
    """
    normalized = re.sub(r"[^\w\s-]", "", title.lower())
    normalized = re.sub(r"[\s_]+", "-", normalized)
    normalized = re.sub(r"-+", "-", normalized).strip("-")
    if not normalized or not re.match(r"[a-z]", normalized):
        normalized = f"skill-{normalized}" if normalized else "unnamed-skill"
    return normalized
