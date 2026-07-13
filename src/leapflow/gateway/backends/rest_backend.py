"""REST execution backend for App Connector actions."""
from __future__ import annotations

from typing import Any, Mapping

from leapflow.gateway.adapters.common import JsonHttpClient, UrlLibJsonHttpClient
from leapflow.gateway.connectors.protocol import (
    ActionPreview,
    ActionResult,
    ActionSpec,
    BackendKind,
    BackendStatus,
)
from leapflow.security.redact import redact_sensitive_text


class RestBackend:
    """Execute registered platform actions through official REST APIs."""

    kind = BackendKind.REST.value

    def __init__(
        self,
        *,
        base_url: str,
        token: str = "",
        headers: Mapping[str, str] | None = None,
        http_client: JsonHttpClient | None = None,
        timeout_s: float = 30.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._headers = dict(headers or {})
        self._http = http_client or UrlLibJsonHttpClient()
        self._timeout_s = timeout_s

    async def status(self) -> BackendStatus:
        return BackendStatus(
            ok=bool(self._base_url),
            backend_kind=self.kind,
            detail="" if self._base_url else "REST base_url is required",
        )

    async def authenticate(self, payload: Mapping[str, Any]) -> BackendStatus:
        token = str(payload.get("token") or self._token or "")
        return BackendStatus(
            ok=bool(token),
            backend_kind=self.kind,
            detail="" if token else "REST token is required",
        )

    async def execute(
        self,
        spec: ActionSpec,
        payload: Mapping[str, Any],
    ) -> ActionResult:
        config = spec.backend_config
        method = str(config.get("method") or "GET").upper()
        path = self._render(str(config.get("path") or ""), payload)
        body_template = config.get("json_body")
        json_body = self._render_mapping(body_template, payload) if isinstance(body_template, Mapping) else dict(payload)
        headers = dict(self._headers)
        if self._token:
            headers.setdefault("Authorization", f"Bearer {self._token}")
        url = f"{self._base_url}/{path.lstrip('/')}"
        try:
            status, data = await self._http.request_json(
                method,
                url,
                json_body=json_body if method != "GET" else None,
                headers=headers,
                timeout_s=float(config.get("timeout_s") or self._timeout_s),
            )
        except Exception as exc:
            return ActionResult(ok=False, error=redact_sensitive_text(str(exc), force=True))
        if status >= 400:
            return ActionResult(ok=False, error=redact_sensitive_text(str(data), force=True), raw=data)
        return ActionResult(ok=True, data=data, resource_id=self._resource_id(data), raw=data)

    async def preview(
        self,
        spec: ActionSpec,
        payload: Mapping[str, Any],
    ) -> ActionPreview:
        """Return a side-effect-free REST request preview."""
        config = spec.backend_config
        method = str(config.get("method") or "GET").upper()
        path = self._render(str(config.get("path") or ""), payload)
        return ActionPreview(
            ok=True,
            summary=f"{method} {path}",
            data={"backend_kind": self.kind, "action": spec.name, "method": method, "path": path},
        )

    @classmethod
    def _render_mapping(cls, template: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in template.items():
            if isinstance(value, str):
                result[str(key)] = cls._render(value, payload)
            else:
                result[str(key)] = value
        return result

    @staticmethod
    def _render(template: str, payload: Mapping[str, Any]) -> str:
        value = template
        for key, raw in payload.items():
            value = value.replace("{" + str(key) + "}", str(raw))
        return value

    @staticmethod
    def _resource_id(data: Mapping[str, Any]) -> str:
        for key in ("id", "message_id", "guid", "token"):
            value = data.get(key)
            if value:
                return str(value)
        return ""
