# Copyright (c) Alibaba, Inc. and its affiliates.
"""Invocation adapter for ToolMetadata handlers.

Tool handlers historically used two call shapes:
- generated and self-management tools accept keyword arguments (``**kwargs``),
- older built-ins accept one JSON-object argument named ``params``/``args``.

The engine calls through this module so the runtime has one explicit contract
adapter instead of relying on fragile ``handler(args)`` positional calls.
"""
from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any

_MAPPING_ARGUMENT_NAMES = frozenset({"params", "args", "arguments", "payload"})


class ToolHandlerInvocationError(TypeError):
    """Raised when a tool handler cannot be called with JSON-object arguments."""


async def invoke_tool_handler(handler: Callable[..., Any], arguments: Mapping[str, Any] | None) -> Any:
    """Invoke ``handler`` with the runtime's JSON-object argument payload.

    The function selects the call style from the handler signature rather than
    catching ``TypeError`` from the invocation. That preserves real tool bugs as
    real failures instead of retrying them through another call convention.
    """
    args = _coerce_arguments(arguments)
    signature = inspect.signature(handler)
    params = tuple(signature.parameters.values())

    if _accepts_keyword_arguments(params):
        return await _maybe_await(handler(**args))

    if not params:
        if args:
            raise ToolHandlerInvocationError(
                "Tool handler accepts no arguments but received a non-empty argument object"
            )
        return await _maybe_await(handler())

    if _accepts_single_mapping_argument(params):
        return await _maybe_await(handler(args))

    try:
        signature.bind(**args)
    except TypeError as exc:
        raise ToolHandlerInvocationError(
            f"Tool handler signature {signature} is incompatible with provided arguments"
        ) from exc
    return await _maybe_await(handler(**args))


def _coerce_arguments(arguments: Mapping[str, Any] | None) -> dict[str, Any]:
    if arguments is None:
        return {}
    if not isinstance(arguments, Mapping):
        raise ToolHandlerInvocationError("Tool arguments must be a JSON object")
    return dict(arguments)


def _accepts_keyword_arguments(params: tuple[inspect.Parameter, ...]) -> bool:
    return any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params)


def _accepts_single_mapping_argument(params: tuple[inspect.Parameter, ...]) -> bool:
    positional = [
        param
        for param in params
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if not positional:
        return False
    first, *rest = positional
    if first.name not in _MAPPING_ARGUMENT_NAMES:
        return False
    return all(param.default is not inspect.Parameter.empty for param in rest)


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value
