"""Unified config CLI for LeapFlow."""
from __future__ import annotations

import argparse
import json
import logging
from typing import Any

from leapflow.config import load_config
from leapflow.config_service import ConfigService


def cmd_config(args: argparse.Namespace) -> int:
    """Manage LeapFlow configuration through the unified control plane."""
    action = getattr(args, "config_action", None) or "show"
    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.WARNING)
    try:
        settings = load_config()
    finally:
        logging.disable(previous_disable_level)
    service = ConfigService(settings)
    try:
        if action == "show":
            key = getattr(args, "key", None)
            if key:
                _print_field_detail(service.describe(str(key)))
            else:
                _print_snapshot(service)
            return 0
        if action == "keys":
            for key in service.writable_keys():
                print(key)
            return 0
        if action == "list":
            items = service.list_fields(getattr(args, "category", None))
            if bool(getattr(args, "json", False)):
                print(json.dumps([_field_to_dict(item) for item in items], indent=2, ensure_ascii=False))
            else:
                _print_field_list(items)
            return 0
        if action == "sources":
            for source in service.sources():
                print(source)
            return 0
        if action == "get":
            view = service.get(str(args.key))
            print(f"{view.key}={view.value}")
            return 0
        if action == "set":
            result = service.set(str(args.key), args.value, scope=getattr(args, "scope", "profile"))
            _print_result(result.message, result.changed_keys, result.warnings)
            return 0
        if action == "unset":
            result = service.unset(str(args.key), scope=getattr(args, "scope", "profile"))
            _print_result(result.message, result.changed_keys, result.warnings)
            return 0
        if action == "llm":
            return _cmd_llm(service, args)
        if action == "secret":
            return _cmd_secret(service, args)
    except (KeyError, ValueError, RuntimeError) as exc:
        print(f"Config error: {exc}")
        return 2
    print(f"Unknown config action: {action}")
    return 2


def _cmd_llm(service: ConfigService, args: argparse.Namespace) -> int:
    llm_action = getattr(args, "llm_action", None) or "show"
    if llm_action == "show":
        snapshot = service.snapshot()
        for item in snapshot.values:
            if item.key.startswith("llm."):
                print(f"{item.key}={item.value}")
        return 0
    if llm_action == "set":
        result = service.configure_llm(
            api_key=getattr(args, "api_key", None),
            ask_api_key=bool(getattr(args, "ask_api_key", False)),
            base_url=getattr(args, "base_url", None),
            model=getattr(args, "model", None),
            context_length=getattr(args, "context_length", None),
            max_retries=getattr(args, "max_retries", None),
            scope=getattr(args, "scope", "profile"),
        )
        _print_result(result.message, result.changed_keys, result.warnings)
        return 0 if result.ok else 1
    if llm_action == "key":
        result = service.configure_llm(ask_api_key=True, scope=getattr(args, "scope", "profile"))
        _print_result(result.message, result.changed_keys, result.warnings)
        return 0
    print(f"Unknown llm config action: {llm_action}")
    return 2


def _cmd_secret(service: ConfigService, args: argparse.Namespace) -> int:
    secret_action = getattr(args, "secret_action", None) or "list"
    if secret_action == "list":
        for ref in service.list_secrets():
            print(ref)
        return 0
    if secret_action == "set":
        result = service.set_secret(args.ref, getattr(args, "value", None), scope=getattr(args, "scope", "profile"))
        print(result.message)
        return 0
    if secret_action == "get":
        print(service.get_secret(args.ref, scope=getattr(args, "scope", "profile"), reveal=bool(getattr(args, "reveal", False))))
        return 0
    if secret_action == "delete":
        result = service.delete_secret(args.ref, scope=getattr(args, "scope", "profile"))
        print(result.message)
        return 0
    print(f"Unknown secret config action: {secret_action}")
    return 2


def _print_snapshot(service: ConfigService) -> None:
    snapshot = service.snapshot()
    for item in snapshot.values:
        print(f"{item.key}={item.value}")
    if snapshot.warnings:
        print("warnings:")
        for warning in snapshot.warnings:
            print(f"  - {warning}")


def _print_field_detail(item: Any) -> None:
    scope = ",".join(item.scopes)
    print(f"{item.key}")
    print(f"  value: {item.value}")
    print(f"  type: {item.value_type}")
    print(f"  category: {item.category}")
    print(f"  scope: {scope}")
    print(f"  reload: {item.hot_reload}")
    print(f"  secret: {'true' if item.secret else 'false'}")
    if item.value_hint:
        print(f"  value hint: {item.value_hint}")
    print(f"  description: {item.description}")
    if item.examples:
        print(f"  example: {item.examples[0]}")


def _print_field_list(items: tuple[Any, ...]) -> None:
    if not items:
        print("No writable config fields found.")
        return
    print("Writable config fields:")
    for item in items:
        scope = ",".join(item.scopes)
        print(f"- {item.key}={item.value}")
        print(f"  type: {item.value_type} | scope: {scope} | reload: {item.hot_reload}")
        if item.value_hint:
            print(f"  value: {item.value_hint}")
        print(f"  description: {item.description}")
        if item.examples:
            print(f"  example: {item.examples[0]}")


def _field_to_dict(item: Any) -> dict[str, Any]:
    return {
        "key": item.key,
        "value": item.value,
        "type": item.value_type,
        "category": item.category,
        "scopes": list(item.scopes),
        "hot_reload": item.hot_reload,
        "secret": item.secret,
        "description": item.description,
        "value_hint": item.value_hint,
        "examples": list(item.examples),
    }


def _print_result(
    message: str,
    changed_keys: tuple[str, ...],
    warnings: tuple[str, ...] = (),
) -> None:
    print(message)
    for key in changed_keys:
        print(f"  {key}")
    for warning in warnings:
        print(f"warning: {warning}")
