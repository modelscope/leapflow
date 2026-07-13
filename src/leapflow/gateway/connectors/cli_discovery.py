"""CLI help-based command discovery for App Connector backends.

Discovers available commands and arguments by invoking ``<binary> --help``
recursively, parsing the output into structured data, and generating
draft :class:`ActionSpec` objects.  The parser uses heuristics that cover
the most common CLI frameworks (Click, Cobra, argparse) without requiring
LLM calls — the LLM can later match user intent against the discovered
command index exposed in the prompt context.

Design
~~~~~~
Three layers cooperate:

1. **HelpParser** — stateless parser that turns raw ``--help`` text into
   structured :class:`HelpParseResult` objects.
2. **CliDiscovery** — async runner that invokes the binary, feeds output
   into :class:`HelpParser`, and builds a command tree.
3. **Draft ActionSpec builder** — converts discovered commands into
   ``ActionSpec`` objects with conservative safety defaults.

All discovered commands default to ``risk_level="high"`` and
``effect="execute"`` to enforce approval before first use.
"""
from __future__ import annotations

import asyncio
import hashlib
import re
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from leapflow.gateway.connectors.protocol import ActionSpec, BackendKind

# ── Data types ────────────────────────────────────────────────────────

_EFFECT_HINTS: dict[str, str] = {
    "list": "read", "get": "read", "show": "read", "search": "read",
    "view": "read", "info": "read", "status": "read", "describe": "read",
    "find": "read", "query": "read", "read": "read", "cat": "read",
    "inspect": "read", "dump": "read", "export": "read", "count": "read",
    "send": "send", "post": "send", "notify": "send", "publish": "send",
    "forward": "send", "reply": "send", "broadcast": "send",
    "create": "write", "add": "write", "set": "write", "update": "write",
    "put": "write", "edit": "write", "modify": "write", "append": "write",
    "write": "write", "insert": "write", "patch": "write", "upload": "write",
    "delete": "execute", "remove": "execute", "drop": "execute",
    "purge": "execute", "reset": "execute", "destroy": "execute",
    "revoke": "execute",
}

_DANGEROUS_PATTERNS = re.compile(
    r"\b(delete|remove|drop|purge|reset|destroy|revoke|wipe|"
    r"force|admin|sudo|dangerous)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class HelpEntry:
    """One command or subcommand discovered from ``--help`` output."""

    name: str
    description: str = ""


@dataclass(frozen=True)
class HelpArgument:
    """One CLI argument/flag discovered from ``--help`` output."""

    name: str
    flag: str = ""
    description: str = ""
    required: bool = False
    type_hint: str = "string"
    default: str = ""


@dataclass(frozen=True)
class HelpParseResult:
    """Structured parse of a single ``--help`` invocation."""

    binary: str
    prefix: tuple[str, ...] = ()
    description: str = ""
    subcommands: tuple[HelpEntry, ...] = ()
    arguments: tuple[HelpArgument, ...] = ()
    raw_text: str = ""


@dataclass(frozen=True)
class DiscoveredCommand:
    """A fully resolved CLI command path with argument metadata."""

    binary: str
    argv_prefix: tuple[str, ...]
    description: str = ""
    arguments: tuple[HelpArgument, ...] = ()
    group: str = ""
    depth: int = 0
    discovered_at: float = field(default_factory=time.time)


# ── Help text parser ─────────────────────────────────────────────────

_SECTION_HEADER_RE = re.compile(
    r"^(?:available\s+|positional\s+|optional\s+)?"
    r"(commands|subcommands|options|flags|arguments|usage)"
    r"\s*:?\s*$",
    re.IGNORECASE,
)

_COMMAND_LINE_RE = re.compile(
    r"^\s{2,}(\S+)\s{2,}(.+)$",
)

_FLAG_RE = re.compile(
    r"^\s{2,}"
    r"(?:(-\w),?\s+)?"
    r"(--[\w][\w-]*)"
    r"(?:\s+(\S+))?"
    r"\s{2,}(.*)$",
)

_REQUIRED_MARKER_RE = re.compile(r"\brequired\b", re.IGNORECASE)


class HelpParser:
    """Parse ``--help`` output into structured data.

    Handles the most common CLI help formats without requiring a specific
    framework.  Works with Click, Cobra (Go), argparse, clap (Rust), and
    similar two-column layouts.
    """

    def parse(
        self,
        raw: str,
        *,
        binary: str = "",
        prefix: Sequence[str] = (),
    ) -> HelpParseResult:
        lines = raw.splitlines()
        description = self._extract_description(lines)
        subcommands = self._extract_subcommands(lines)
        arguments = self._extract_arguments(lines)
        return HelpParseResult(
            binary=binary,
            prefix=tuple(prefix),
            description=description,
            subcommands=tuple(subcommands),
            arguments=tuple(arguments),
            raw_text=raw,
        )

    def _extract_description(self, lines: list[str]) -> str:
        """Extract the top-level description before any section header."""
        parts: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                if parts:
                    break
                continue
            if _SECTION_HEADER_RE.match(stripped):
                break
            if stripped.lower().startswith("usage:"):
                continue
            parts.append(stripped)
        return " ".join(parts)

    def _extract_subcommands(self, lines: list[str]) -> list[HelpEntry]:
        entries: list[HelpEntry] = []
        in_section = False
        for line in lines:
            stripped = line.strip()
            if _SECTION_HEADER_RE.match(stripped):
                header_lower = stripped.lower()
                in_section = (
                    "command" in header_lower
                    or "subcommand" in header_lower
                    or "positional" in header_lower
                )
                continue
            if in_section:
                if not stripped:
                    if entries:
                        break
                    continue
                if stripped.startswith("{") and stripped.endswith("}"):
                    continue
                match = _COMMAND_LINE_RE.match(line)
                if match:
                    name = match.group(1).strip()
                    desc = match.group(2).strip()
                    if not name.startswith("-"):
                        entries.append(HelpEntry(name=name, description=desc))
        return entries

    def _extract_arguments(self, lines: list[str]) -> list[HelpArgument]:
        args: list[HelpArgument] = []
        in_section = False
        for line in lines:
            stripped = line.strip()
            if _SECTION_HEADER_RE.match(stripped):
                header_lower = stripped.lower()
                in_section = any(
                    kw in header_lower
                    for kw in ("option", "flag", "argument")
                )
                continue
            if in_section:
                if not stripped:
                    if args:
                        break
                    continue
                match = _FLAG_RE.match(line)
                if match:
                    short = match.group(1) or ""
                    long_flag = match.group(2)
                    value_hint = match.group(3) or ""
                    desc = match.group(4).strip()
                    if long_flag in ("--help", "--version"):
                        continue
                    name = long_flag.lstrip("-").replace("-", "_")
                    required = bool(_REQUIRED_MARKER_RE.search(desc))
                    type_hint = self._infer_type(value_hint, desc)
                    default = self._extract_default(desc)
                    args.append(HelpArgument(
                        name=name,
                        flag=long_flag,
                        description=desc,
                        required=required,
                        type_hint=type_hint,
                        default=default,
                    ))
        return args

    def _infer_type(self, value_hint: str, desc: str) -> str:
        combined = f"{value_hint} {desc}".lower()
        if any(kw in combined for kw in ("int", "number", "count", "size", "port")):
            return "integer"
        if any(kw in combined for kw in ("bool", "true", "false", "enable", "disable")):
            return "boolean"
        return "string"

    def _extract_default(self, desc: str) -> str:
        match = re.search(r"default[:\s]+['\"]?([^'\")\]]+)", desc, re.IGNORECASE)
        return match.group(1).strip() if match else ""


# ── Async discovery engine ───────────────────────────────────────────

_DEFAULT_CACHE_TTL_S = 3600.0


@dataclass
class _CacheEntry:
    result: HelpParseResult
    expires_at: float


class CliDiscovery:
    """Discover CLI commands by recursively invoking ``--help``.

    Results are cached in-memory with a configurable TTL. The discovery
    tree is built lazily — call :meth:`discover_group` only when needed
    rather than pre-scanning the entire command namespace.
    """

    def __init__(
        self,
        *,
        binary: str,
        profile: str = "",
        identity: str = "",
        default_args: Sequence[str] = (),
        timeout_s: float = 10.0,
        cache_ttl_s: float = _DEFAULT_CACHE_TTL_S,
    ) -> None:
        self._binary = binary
        self._profile = profile
        self._identity = identity
        self._default_args = tuple(default_args)
        self._timeout_s = timeout_s
        self._cache_ttl_s = cache_ttl_s
        self._cache: dict[str, _CacheEntry] = {}
        self._parser = HelpParser()

    @property
    def binary(self) -> str:
        return self._binary

    def invalidate(self) -> None:
        """Clear all cached help results."""
        self._cache.clear()

    async def discover(
        self,
        *prefix: str,
    ) -> HelpParseResult:
        """Run ``<binary> [prefix...] --help`` and return parsed result."""
        cache_key = _cache_key(self._binary, prefix)
        entry = self._cache.get(cache_key)
        if entry is not None and entry.expires_at > time.monotonic():
            return entry.result

        argv = self._build_argv(prefix)
        raw = await self._run_help(argv)
        result = self._parser.parse(raw, binary=self._binary, prefix=prefix)
        self._cache[cache_key] = _CacheEntry(
            result=result,
            expires_at=time.monotonic() + self._cache_ttl_s,
        )
        return result

    async def discover_group(
        self,
        group: str,
        *,
        max_subcommands: int = 50,
    ) -> list[DiscoveredCommand]:
        """Discover all commands under a top-level group.

        Performs a two-level traversal: ``<binary> <group> --help`` to list
        subcommands, then ``<binary> <group> <sub> --help`` for each to
        get argument details.
        """
        group_result = await self.discover(group)
        commands: list[DiscoveredCommand] = []

        if group_result.subcommands:
            tasks = [
                self.discover(group, sub.name)
                for sub in group_result.subcommands[:max_subcommands]
            ]
            sub_results = await asyncio.gather(*tasks, return_exceptions=True)
            for sub_entry, sub_result in zip(
                group_result.subcommands[:max_subcommands], sub_results
            ):
                if isinstance(sub_result, BaseException):
                    continue
                commands.append(DiscoveredCommand(
                    binary=self._binary,
                    argv_prefix=(group, sub_entry.name),
                    description=sub_entry.description or sub_result.description,
                    arguments=sub_result.arguments,
                    group=group,
                    depth=2,
                ))
        elif group_result.arguments:
            commands.append(DiscoveredCommand(
                binary=self._binary,
                argv_prefix=(group,),
                description=group_result.description,
                arguments=group_result.arguments,
                group=group,
                depth=1,
            ))

        return commands

    async def discover_tree(
        self,
        *,
        max_groups: int = 30,
        max_subcommands: int = 50,
    ) -> list[DiscoveredCommand]:
        """Full two-level discovery: top-level groups → subcommands."""
        root = await self.discover()
        if not root.subcommands:
            return []

        tasks = [
            self.discover_group(
                group_entry.name,
                max_subcommands=max_subcommands,
            )
            for group_entry in root.subcommands[:max_groups]
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        all_commands: list[DiscoveredCommand] = []
        for result in results:
            if isinstance(result, BaseException):
                continue
            all_commands.extend(result)
        return all_commands

    def to_action_spec(
        self,
        command: DiscoveredCommand,
        *,
        domain: str = "",
    ) -> ActionSpec:
        """Convert a discovered command into a draft ActionSpec."""
        action_name = _action_name(command, domain=domain)
        argv_template = _build_argv_template(command)
        schema = _build_schema(command)
        effect = _infer_effect(command)
        risk = "critical" if _DANGEROUS_PATTERNS.search(
            f"{' '.join(command.argv_prefix)} {command.description}"
        ) else "high"

        return ActionSpec(
            name=action_name,
            backend_kind=BackendKind.CLI.value,
            description=command.description or f"Discovered CLI command: {' '.join(command.argv_prefix)}",
            effect=effect,
            schema=schema,
            backend_config={
                "argv": tuple(argv_template),
                "timeout_s": 30,
                "discovered": True,
            },
            risk_level=risk,
            output_policy="summary",
        )

    def _build_argv(self, prefix: Sequence[str]) -> list[str]:
        argv = [self._binary]
        if self._profile:
            argv.extend(["--profile", self._profile])
        if self._identity:
            argv.extend(["--as", self._identity])
        argv.extend(self._default_args)
        argv.extend(prefix)
        argv.append("--help")
        return argv

    async def _run_help(self, argv: Sequence[str]) -> str:
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self._timeout_s,
            )
        except FileNotFoundError:
            return ""
        except asyncio.TimeoutError:
            if proc is not None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            return ""
        except OSError:
            return ""
        return (stdout or stderr or b"").decode("utf-8", errors="replace")


# ── Spec builders ────────────────────────────────────────────────────

def _cache_key(binary: str, prefix: Sequence[str]) -> str:
    raw = f"{binary}:{':'.join(prefix)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _action_name(command: DiscoveredCommand, *, domain: str = "") -> str:
    """Build a ``domain.operation`` name from the command path."""
    parts = list(command.argv_prefix)
    if not parts:
        return "unknown.unknown"
    if domain:
        group = domain
    else:
        group = parts[0].lstrip("+").replace("-", "_")
    operation = "_".join(
        p.lstrip("+").replace("-", "_") for p in parts[1:]
    ) if len(parts) > 1 else parts[0].lstrip("+").replace("-", "_")
    return f"{group}.{operation}"


def _build_argv_template(command: DiscoveredCommand) -> list[str]:
    """Build an argv template with ``{placeholder}`` tokens from arguments."""
    argv: list[str] = list(command.argv_prefix)
    for arg in command.arguments:
        if arg.flag and arg.required:
            argv.append(arg.flag)
            argv.append("{" + arg.name + "}")
    return argv


def _build_schema(command: DiscoveredCommand) -> dict[str, Any]:
    """Build a JSON Schema subset from discovered arguments."""
    required: list[str] = []
    properties: dict[str, dict[str, str]] = {}
    for arg in command.arguments:
        prop: dict[str, str] = {"type": arg.type_hint}
        if arg.description:
            prop["description"] = arg.description
        properties[arg.name] = prop
        if arg.required:
            required.append(arg.name)
    schema: dict[str, Any] = {}
    if required:
        schema["required"] = required
    if properties:
        schema["properties"] = properties
    return schema


def _infer_effect(command: DiscoveredCommand) -> str:
    """Infer the action effect from the command name/description.

    Uses word-boundary matching to avoid false positives (e.g. "reset"
    must not match the "set" keyword).
    """
    parts = " ".join(command.argv_prefix).lower()
    for keyword, effect in _EFFECT_HINTS.items():
        if re.search(r"(?:^|[\W_])" + re.escape(keyword) + r"(?:$|[\W_])", parts):
            return effect
    desc_lower = command.description.lower()
    for keyword, effect in _EFFECT_HINTS.items():
        if re.search(r"(?:^|[\W_])" + re.escape(keyword) + r"(?:$|[\W_])", desc_lower):
            return effect
    return "execute"
