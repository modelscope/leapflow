# Copyright (c) Alibaba, Inc. and its affiliates.
"""Bounded local request inputs and integrity-linked compute-only result artifacts."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from leapcalib.contracts.failures import (
    CalibrationFailure,
    FAILURE_SCHEMA,
    FAILURE_EVIDENCE_CORRUPT,
)
from leapcalib.contracts.identity import digest_of

REQUEST_MAX_BYTES = 1024 * 1024


def read_request(path: str | os.PathLike[str]) -> dict[str, Any]:
    with Path(path).open("rb") as source:
        encoded = source.read(REQUEST_MAX_BYTES + 1)
    if len(encoded) > REQUEST_MAX_BYTES:
        raise CalibrationFailure(FAILURE_SCHEMA, "numerical request exceeds byte budget")
    try:
        payload = json.loads(encoded.decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("request must be an object")
        digest_of(payload)
    except (ValueError, UnicodeDecodeError, TypeError, RecursionError) as error:
        raise CalibrationFailure(FAILURE_SCHEMA, "invalid numerical request JSON") from error
    return payload


def result_payload(content: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "leapcalib.result.v1",
        "content": dict(content),
        "digest": digest_of(content),
    }


def write_result(
    path: str | os.PathLike[str], payload: Mapping[str, Any], *, overwrite: bool = False
) -> Path:
    """Write explicitly requested private local output, never a deployment database."""
    encoded = json.dumps(payload, sort_keys=True, allow_nan=False, separators=(",", ":"))
    target = Path(path)
    flags = os.O_WRONLY | os.O_CREAT | (os.O_TRUNC if overwrite else os.O_EXCL)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(target, flags, 0o600)
    except FileExistsError as error:
        raise CalibrationFailure(
            FAILURE_EVIDENCE_CORRUPT, "result exists; explicitly request overwrite"
        ) from error
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        os.fchmod(output.fileno(), 0o600)
        output.write(encoded)
    return target


__all__ = ["read_request", "result_payload", "write_result"]
