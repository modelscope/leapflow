# Copyright (c) Alibaba, Inc. and its affiliates.
"""Node write authority: the write mode and the permit check every actuating method passes.

One object answers "may this request move hardware?" for the whole node, so
``write``, ``write_batch``, ``control.open``, ``control.update``, and
``skill.execute`` cannot drift into different rules. A refusal is always a
``WriteOutcome`` with side effect NONE: nothing was attempted.

The digest a permit must carry is computed *here*, from what the node received
(``request_digest``), unless the method binds a content digest of its own --
``skill.execute`` binds the approved plan. A permit is never checked against
the digest it carries itself; that would accept any validly signed permit for
any request.
"""

from __future__ import annotations

from typing import Any, Mapping

from leapflow.hardware.lhp.permit import PERMIT_PARAM, ExecutionPermit, request_digest
from leapflow.hardware.lhp.protocol import FAILURE_PERMIT_REQUIRED, FAILURE_WRITES_DISABLED
from leapflow.hardware.transport import SIDE_EFFECT_NONE, WriteOutcome

WRITE_MODE_PERMIT_REQUIRED = "permit_required"
WRITE_MODE_TRUSTED_V1 = "trusted_v1"
WRITE_MODE_OBSERVE_ONLY = "observe_only"

VALID_WRITE_MODES = frozenset(
    {WRITE_MODE_PERMIT_REQUIRED, WRITE_MODE_TRUSTED_V1, WRITE_MODE_OBSERVE_ONLY}
)


def _refused(message: str, code: str) -> WriteOutcome:
    return WriteOutcome(
        ok=False, side_effect_state=SIDE_EFFECT_NONE, error=message, failure_code=code
    )


class WriteAuthority:
    """Write-mode and permit gate shared by every actuating LHP method."""

    def __init__(self, mode: str, verifier: Any | None = None) -> None:
        if mode not in VALID_WRITE_MODES:
            raise ValueError(f"write_mode must be one of {sorted(VALID_WRITE_MODES)}, got {mode!r}")
        self._mode = mode
        self._verifier = verifier

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def verifier(self) -> Any | None:
        return self._verifier

    def check(
        self, method: str, params: Mapping[str, Any], *, expected_digest: str = ""
    ) -> WriteOutcome | None:
        """Return a refusal, or None when *method* with *params* may actuate.

        ``expected_digest`` overrides the request digest for methods that bind
        content rather than wire parameters.
        """
        if self._mode == WRITE_MODE_OBSERVE_ONLY:
            return _refused(
                "node is in observe-only mode; writes are disabled", FAILURE_WRITES_DISABLED
            )
        if self._mode != WRITE_MODE_PERMIT_REQUIRED:
            return None
        return self._verify(params, expected_digest or request_digest(method, params))

    def _verify(self, params: Mapping[str, Any], expected: str) -> WriteOutcome | None:
        verifier = self._verifier
        permit_data = params.get(PERMIT_PARAM)
        if permit_data is None or verifier is None:
            trusted = verifier is not None and bool(getattr(verifier, "has_trusted_keys", False))
            return _refused(
                "node requires a signed execution permit for writes"
                if trusted
                else "node has no trusted issuer keys; all writes are refused "
                "until permit issuer keys are configured",
                FAILURE_PERMIT_REQUIRED,
            )
        if not isinstance(permit_data, Mapping):
            return _refused("malformed permit: not an object", FAILURE_PERMIT_REQUIRED)
        try:
            permit = ExecutionPermit.from_dict(permit_data)
        except (TypeError, ValueError, KeyError) as exc:
            return _refused(f"malformed permit: {exc}", FAILURE_PERMIT_REQUIRED)
        rejection = verifier.verify(permit, expected_digest=expected)
        if rejection is not None:
            return _refused(rejection.message, rejection.code)
        return None


__all__ = [
    "VALID_WRITE_MODES",
    "WRITE_MODE_OBSERVE_ONLY",
    "WRITE_MODE_PERMIT_REQUIRED",
    "WRITE_MODE_TRUSTED_V1",
    "WriteAuthority",
]
