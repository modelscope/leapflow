"""Vault CLI for managing LeapFlow secret refs."""
from __future__ import annotations

import argparse
from getpass import getpass

from leapflow.config import load_config
from leapflow.security.secrets import ScopedSecretResolver, secret_ref, secret_scope


def normalize_secret_ref(raw: str, *, default_scope: str = "profile") -> str:
    """Normalize a CLI secret name into a secret:// ref."""
    value = raw.strip()
    if value.startswith("secret://"):
        secret_scope(value)
        return value
    return secret_ref(default_scope, *_secret_name_parts(value))


def _secret_name_parts(value: str) -> list[str]:
    if "/" in value:
        parts = [part for part in value.split("/") if part]
    else:
        parts = [part for part in value.split(".") if part]
    if not parts:
        raise ValueError("Secret ref cannot be empty")
    return parts


def cmd_vault(args: argparse.Namespace) -> int:
    """Manage secrets in the active layout vault."""
    settings = load_config()
    resolver = ScopedSecretResolver(settings.layout, settings.profile_layout)
    action = getattr(args, "vault_action", None) or "list"

    if action == "list":
        for ref in resolver.list_refs():
            print(ref)
        return 0

    raw_ref = getattr(args, "ref", "")
    if not raw_ref:
        print("Missing secret ref. Use: leap vault <set|get|delete> <ref>")
        return 2
    try:
        ref = normalize_secret_ref(raw_ref, default_scope=getattr(args, "scope", "profile"))
    except ValueError as exc:
        print(f"Invalid secret ref: {exc}")
        return 2

    if action == "set":
        value = getattr(args, "value", None)
        if value is None:
            value = getpass(f"Value for {ref}: ")
        resolver.set(ref, value, metadata={"source": "cli"})
        print(ref)
        return 0

    if action == "get":
        value = resolver.get(ref)
        if value is None:
            print(f"Secret not found: {ref}")
            return 1
        if getattr(args, "reveal", False):
            print(value)
        else:
            print(f"{ref} is set")
        return 0

    if action == "delete":
        resolver.delete(ref)
        print(f"Deleted {ref}")
        return 0

    print(f"Unknown vault action: {action}")
    return 2
