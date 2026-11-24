# Copyright (c) Alibaba, Inc. and its affiliates.
"""Shared CLI/TUI specialist catalog operations and explicit approved fixture checks."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from leapflow.config_service import ConfigMutationResult, ConfigService


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise ValueError(message)


def add_service_arguments(parser: argparse.ArgumentParser) -> None:
    actions = parser.add_subparsers(dest="service_action")
    show = actions.add_parser("show", help="Show profile model declarations (no endpoint calls)")
    show.add_argument("service_id", nargs="?", default="")
    set_parser = actions.add_parser("set", help="Validate and atomically upsert one service")
    set_parser.add_argument("service_id")
    set_parser.add_argument("--file", required=True, help="YAML/JSON declaration file")
    set_parser.add_argument("--scope", choices=("profile",), default="profile")
    unset = actions.add_parser("unset", help="Remove future admission for one service")
    unset.add_argument("service_id")
    unset.add_argument("--scope", choices=("profile",), default="profile")
    for operation in ("enable", "disable"):
        toggle = actions.add_parser(operation, help=f"{operation.capitalize()} future use; never stop/start serving processes")
        toggle.add_argument("service_id")
        toggle.add_argument("--scope", choices=("profile",), default="profile")
    check = actions.add_parser("check", help="Validate locally, or explicitly check an approved --case; never motion")
    check.add_argument("service_id", nargs="?", default="")
    check.add_argument("--case", default="", help="Bounded synthetic/recorded JSON task fixture")


def parse_service_arguments(tokens: list[str]) -> argparse.Namespace:
    parser = _Parser(prog="config llm services", add_help=False)
    add_service_arguments(parser)
    return parser.parse_args(tokens or ["show"])


def read_declaration(path: str) -> Any:
    """Read a bounded operator-supplied file; never log YAML content on failure."""
    try:
        with Path(path).expanduser().open("rb") as handle:
            content = handle.read(8388609)
        if len(content) > 8388608:
            raise ValueError("Declaration file exceeds 8 MiB")
        return yaml.safe_load(content)
    except (OSError, yaml.YAMLError, UnicodeError) as exc:
        raise ValueError("Could not read declaration; supply a valid bounded YAML/JSON file") from exc


def execute_service_config(service: ConfigService, args: argparse.Namespace) -> dict[str, Any] | ConfigMutationResult:
    action = args.service_action or "show"
    name = getattr(args, "service_id", "")
    if action == "set":
        return service.set_model_service(name, read_declaration(args.file))
    if action == "unset":
        return service.unset_model_service(name)
    if action in {"enable", "disable"}:
        return service.set_model_service_enabled(name, action == "enable")
    if action == "check":
        if getattr(args, "case", ""):
            raise ValueError("Fixture checks require the asynchronous governed command path")
        return service.check_model_services(name)
    catalog = service.model_services()
    if name:
        if name not in catalog:
            raise ValueError(f"Unknown model service: {name}")
        catalog = {name: catalog[name]}
    return {"ok": True, "mode": "local_only", "declarations": catalog, "endpoint_probed": False}


async def execute_fixture(settings: Any, args: argparse.Namespace, *, client: Any = None,
                          gate: Any = None, session_id: str = "",
                          workspace_root: str | None = None) -> dict[str, Any]:
    """Compose one owner only for CLI; TUI/daemon reuse their injected profile owner."""
    from leapflow.llm.services.runtime import ModelServiceRuntime
    from leapflow.tools.execution_context import ToolExecutionContext, set_tool_context, reset_tool_context
    from leapflow.tools.model_fixture import check_fixture
    from leapflow.tools.model_tools import failure_payload
    from leapflow.llm.services.protocol import ServiceError

    if client is not None and not callable(getattr(gate, "evaluate", None)):
        return failure_payload(ServiceError("Fixture approval unavailable; use a governed interactive session",
                                            code="permission_denied", category="security"))
    root = workspace_root if workspace_root is not None else str(settings.workspace_root)
    if not root:
        return failure_payload(ServiceError("Fixture check requires the caller's active workspace; start a session turn first",
                                            code="model_session_unavailable"))
    owned = client is None
    if gate is None:
        from leapflow.cli.context import _TUIApprovalGate
        from leapflow.security.approval import SessionAwareGate
        from leapflow.security.grants import ApprovalAuditLog, JsonApprovalGrantStore
        from leapflow.security.orchestrator import ApprovalOrchestrator
        from leapflow.security.policy import ApprovalPolicyEngine

        layout = settings.profile_layout.approval
        gate = ApprovalOrchestrator(SessionAwareGate(_TUIApprovalGate()),
                                    policy=ApprovalPolicyEngine(bypass=settings.approval_bypass),
                                    grants=JsonApprovalGrantStore(layout.grants_path),
                                    audit=ApprovalAuditLog(layout.audit_path))
    if owned:
        client = ModelServiceRuntime.from_settings(settings)
    token = set_tool_context(ToolExecutionContext.from_strings(workspace_root=root,
                                                              session_id=session_id, orchestrator=gate))
    try:
        path = Path(args.case).expanduser()
        if not path.is_absolute():
            path = Path(root) / path
        return await check_fixture(client, args.service_id, path, gate)
    finally:
        reset_tool_context(token)
        if owned:
            await client.aclose()
