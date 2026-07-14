"""CLI execution backend for App Connector actions."""
from __future__ import annotations

import asyncio
import json
import shutil
from contextlib import suppress
from typing import Any, Mapping, Sequence

from leapflow.gateway.connectors.cli_discovery import CliDiscovery
from leapflow.gateway.connectors.protocol import (
    ActionPreview,
    ActionResult,
    ActionSpec,
    BackendKind,
    BackendStatus,
)
from leapflow.security.redact import redact_sensitive_text


class CliBackend:
    """Execute registered platform actions through an official CLI binary.

    Implements :class:`ActionDiscovery` via an embedded :class:`CliDiscovery`
    instance that can discover unregistered commands through ``--help``
    introspection.
    """

    kind = BackendKind.CLI.value

    def __init__(
        self,
        *,
        binary: str,
        profile: str = "",
        identity: str = "",
        default_args: Sequence[str] = (),
        timeout_s: float = 30.0,
        discovery_cache_ttl_s: float = 3600.0,
    ) -> None:
        self._binary = binary
        self._profile = profile
        self._identity = identity
        self._default_args = tuple(default_args)
        self._timeout_s = timeout_s
        self._discovery = CliDiscovery(
            binary=binary,
            profile=profile,
            identity=identity,
            default_args=default_args,
            timeout_s=min(timeout_s, 10.0),
            cache_ttl_s=discovery_cache_ttl_s,
        )

    @property
    def binary(self) -> str:
        return self._binary

    @property
    def profile(self) -> str:
        return self._profile

    @property
    def identity(self) -> str:
        return self._identity

    async def status(self) -> BackendStatus:
        binary_path = shutil.which(self._binary)
        if binary_path is None:
            return BackendStatus(
                ok=False,
                backend_kind=self.kind,
                detail=f"CLI binary not found: {self._binary}",
                metadata={
                    **self._base_metadata(),
                    "recoverable": True,
                    "auth_status": "unknown",
                    "recovery_hint": f"Install '{self._binary}' and ensure it is available on PATH.",
                    "next_steps": [
                        f"Install the official CLI binary '{self._binary}'.",
                        f"Run '{self._binary} auth login --json' to authorize the selected profile.",
                        f"Run '{self._binary} auth status --json' to verify the connection.",
                    ],
                },
            )
        command = [self._binary, *self._profile_args(), "auth", "status", "--json"]
        result = await self._run_json(command, timeout_s=min(self._timeout_s, 10.0))
        metadata = {**self._base_metadata(binary_path=binary_path), **dict(result.data)}
        if result.ok:
            metadata.setdefault("auth_status", "authorized")
            return BackendStatus(
                ok=True,
                backend_kind=self.kind,
                detail=result.error,
                metadata=metadata,
            )
        metadata.update(self._recovery_metadata(result.error))
        metadata.setdefault("detail", result.error)
        metadata.setdefault("auth_status", "not_ready")
        return BackendStatus(
            ok=False,
            backend_kind=self.kind,
            detail=result.error,
            metadata=metadata,
        )

    async def authenticate(self, payload: Mapping[str, Any]) -> BackendStatus:
        domain = str(payload.get("domain") or "")
        args = [self._binary, *self._profile_args(), "auth", "login"]
        if domain:
            args.extend(["--domain", domain])
        args.extend(["--json"])
        result = await self._run_json(args, timeout_s=self._timeout_s)
        return BackendStatus(
            ok=result.ok,
            backend_kind=self.kind,
            detail=result.error,
            metadata=dict(result.data),
        )

    async def execute(
        self,
        spec: ActionSpec,
        payload: Mapping[str, Any],
    ) -> ActionResult:
        argv_result = self._build_action_command(spec, payload)
        if isinstance(argv_result, ActionResult):
            return argv_result
        result = await self._run_json(
            argv_result,
            timeout_s=float(spec.backend_config.get("timeout_s") or self._timeout_s),
        )
        if not result.ok and result.failure is None:
            failure = self._classify_failure(spec, result.error, result.raw)
            return ActionResult(
                ok=False,
                error=result.error,
                raw=result.raw,
                failure=failure,
            )
        return result

    async def preview(
        self,
        spec: ActionSpec,
        payload: Mapping[str, Any],
    ) -> ActionPreview:
        """Return a side-effect-free preview of the registered CLI action."""
        argv_result = self._build_action_command(spec, payload, dry_run=True)
        if isinstance(argv_result, ActionResult):
            return ActionPreview(ok=False, error=argv_result.error)
        redacted = [redact_sensitive_text(str(arg), force=True) for arg in argv_result]
        return ActionPreview(
            ok=True,
            summary=self._preview_summary(spec, payload),
            data={
                "backend_kind": self.kind,
                "action": spec.name,
                "argv": redacted,
                "dry_run": bool(spec.backend_config.get("dry_run_argv")),
                "profile": self._profile or "default",
                "identity": self._identity,
                "binary": self._binary,
            },
        )

    async def discover_actions(
        self,
        *,
        groups: Sequence[str] = (),
    ) -> list[ActionSpec]:
        """Discover available CLI commands via ``--help`` introspection.

        When *groups* is provided, only those command groups are explored.
        Otherwise performs a full top-level discovery.
        """
        if groups:
            all_commands = []
            for group in groups:
                all_commands.extend(await self._discovery.discover_group(group))
        else:
            all_commands = await self._discovery.discover_tree()
        return [self._discovery.to_action_spec(cmd) for cmd in all_commands]

    @property
    def cli_discovery(self) -> CliDiscovery:
        """Access the underlying discovery engine for advanced use."""
        return self._discovery

    def _build_action_command(
        self,
        spec: ActionSpec,
        payload: Mapping[str, Any],
        *,
        dry_run: bool = False,
    ) -> list[str] | ActionResult:
        argv_template = spec.backend_config.get("dry_run_argv" if dry_run else "argv")
        if argv_template is None and dry_run:
            argv_template = spec.backend_config.get("argv")
        if not isinstance(argv_template, Sequence) or isinstance(argv_template, (str, bytes)):
            return ActionResult(ok=False, error=f"Action has no CLI argv template: {spec.name}")
        argv = [self._render_arg(str(arg), payload) for arg in argv_template]
        command = [self._binary, *self._profile_args(), *self._identity_args(), *self._default_args, *argv]
        output_args = spec.backend_config.get("output_args", ())
        if isinstance(output_args, Sequence) and not isinstance(output_args, (str, bytes)):
            command.extend(str(a) for a in output_args)
        else:
            return ActionResult(ok=False, error=f"Action has invalid CLI output_args: {spec.name}")
        return command

    def _profile_args(self) -> tuple[str, ...]:
        return ("--profile", self._profile) if self._profile else ()

    def _identity_args(self) -> tuple[str, ...]:
        return ("--as", self._identity) if self._identity else ()

    def _base_metadata(self, *, binary_path: str = "") -> dict[str, Any]:
        return {
            "binary": self._binary,
            "binary_path": binary_path,
            "profile": self._profile or "default",
            "identity": self._identity,
            "backend_kind": self.kind,
        }

    def _classify_failure(
        self,
        spec: ActionSpec,
        error: str,
        raw: Mapping[str, Any],
    ) -> "ActionFailure":
        """Classify lark-cli backend errors into ActionFailure."""
        from leapflow.gateway.backends.lark_cli_errors import classify_lark_cli_failure

        return classify_lark_cli_failure(
            spec,
            error,
            raw,
            binary=self._binary,
            profile=self._profile,
            identity=self._identity,
        )

    def _recovery_metadata(self, error: str) -> dict[str, Any]:
        status_command = " ".join([self._binary, *self._profile_args(), "auth", "status", "--json"])
        login_command = " ".join([self._binary, *self._profile_args(), "auth", "login", "--json"])
        lowered = error.lower()
        if (
            "unknown flag" in lowered
            or "unknown command" in lowered
            or "unknown shorthand flag" in lowered
            or "accepts " in lowered
        ):
            return {
                "recoverable": False,
                "failure_code": "cli_contract_mismatch",
                "recovery_hint": (
                    f"CLI command contract mismatch — run '{self._binary} --version' to check "
                    "the installed version and verify supported flags."
                ),
                "next_steps": [
                    f"{self._binary} --version",
                    status_command,
                    "Update the action pack argv template if the CLI version has changed.",
                ],
            }
        hint = "Authorize the selected CLI profile, then retry the platform connection."
        if "scope" in lowered or "permission" in lowered:
            hint = "Grant the missing permission scopes to the selected CLI identity, then retry."
        elif "unauthor" in lowered or "login" in lowered or "token" in lowered:
            hint = "Run CLI login for the selected profile, then retry the connection."
        return {
            "recoverable": True,
            "failure_code": "auth_missing",
            "recovery_hint": hint,
            "next_steps": [
                login_command,
                status_command,
                "Retry platform_connect after the CLI reports an authorized profile.",
            ],
        }

    def _preview_summary(self, spec: ActionSpec, payload: Mapping[str, Any]) -> str:
        template = spec.backend_config.get("approval_summary")
        if isinstance(template, str) and template:
            context: dict[str, Any] = {
                **dict(payload),
                "action": spec.name,
                "binary": self._binary,
                "profile": self._profile or "default",
                "identity": self._identity or "default identity",
                "backend_kind": self.kind,
            }
            return self._render_arg(template, context)
        target = spec.description or f"Run {spec.name}"
        return f"{target} via {self._binary} profile {self._profile or 'default'}"

    @staticmethod
    def _render_arg(template: str, payload: Mapping[str, Any]) -> str:
        value = template
        for key, raw in payload.items():
            value = value.replace("{" + str(key) + "}", str(raw))
        return value

    async def _run_json(self, argv: Sequence[str], *, timeout_s: float) -> ActionResult:
        proc: asyncio.subprocess.Process | None = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except FileNotFoundError:
            return ActionResult(ok=False, error=f"CLI binary not found: {self._binary}")
        except asyncio.TimeoutError:
            if proc is not None:
                with suppress(ProcessLookupError):
                    proc.kill()
                with suppress(ProcessLookupError, asyncio.TimeoutError):
                    await asyncio.wait_for(proc.wait(), timeout=2.0)
            return ActionResult(ok=False, error=f"CLI command timed out after {timeout_s}s")
        except OSError as exc:
            return ActionResult(ok=False, error=redact_sensitive_text(str(exc), force=True))

        raw_stdout = stdout.decode("utf-8", errors="replace").strip()
        raw_stderr = stderr.decode("utf-8", errors="replace").strip()
        parsed = self._parse_json(raw_stdout or raw_stderr)
        if proc.returncode != 0:
            error = self._extract_error(parsed) or raw_stderr or raw_stdout or f"exit {proc.returncode}"
            return ActionResult(ok=False, error=redact_sensitive_text(error, force=True), raw=parsed)
        if parsed and parsed.get("ok") is False:
            return ActionResult(
                ok=False,
                error=redact_sensitive_text(self._extract_error(parsed), force=True),
                raw=parsed,
            )
        data = parsed.get("data") if isinstance(parsed.get("data"), dict) else parsed
        resource_id = self._extract_resource_id(data)
        return ActionResult(ok=True, data=data, resource_id=resource_id, raw=parsed)

    @staticmethod
    def _parse_json(raw: str) -> dict[str, Any]:
        if not raw:
            return {}
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            return {"text": raw}
        return value if isinstance(value, dict) else {"value": value}

    @staticmethod
    def _extract_error(parsed: Mapping[str, Any]) -> str:
        error = parsed.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or error.get("hint") or error)
        if error:
            return str(error)
        if parsed.get("text"):
            return str(parsed["text"])
        return ""

    @staticmethod
    def _extract_resource_id(data: Mapping[str, Any]) -> str:
        for key in ("message_id", "guid", "id", "token"):
            value = data.get(key)
            if value:
                return str(value)
        nested = data.get("message") if isinstance(data.get("message"), dict) else {}
        if isinstance(nested, dict) and nested.get("message_id"):
            return str(nested["message_id"])
        return ""
