# Copyright (c) Alibaba, Inc. and its affiliates.
"""Legacy model HTTP binding, with its existing privacy and retry semantics."""
from __future__ import annotations

import asyncio
import ipaddress
import math
from typing import Any, Mapping
from urllib.parse import urlsplit

import httpx

from leapflow.hardware.transport import TransportError


class HttpModelBackend:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.url = str(config.get("url") or "").strip().rstrip("/")
        self.allow_remote = config.get("allow_remote", False)
        self.timeout_s = _number(config, "timeout_s", 5.0)
        self.max_retries = int(_number(config, "max_retries", 1))
        self.retry_backoff_s = _number(config, "retry_backoff_s", 0.2)
        self.client: httpx.AsyncClient | None = None

    async def open(self) -> None:
        parts = urlsplit(self.url)
        problem = ""
        if parts.scheme not in ("http", "https") or not parts.hostname:
            problem = f"model_service url must be http(s)://host:port, got {self.url!r}"
        elif type(self.allow_remote) is not bool:
            problem = "allow_remote must be boolean"
        elif not self.allow_remote and not _is_loopback(parts.hostname):
            problem = "model_service url is not loopback; set allow_remote only for a deliberately remote server"
        elif (not math.isfinite(self.timeout_s) or self.timeout_s <= 0 or self.max_retries < 0
              or not math.isfinite(self.retry_backoff_s) or self.retry_backoff_s < 0):
            problem = "model_service needs timeout_s > 0, max_retries >= 0, retry_backoff_s >= 0"
        if problem:
            raise TransportError(problem, failure_code="model_service_misconfigured")
        self.client = httpx.AsyncClient(base_url=self.url, timeout=self.timeout_s, trust_env=False)

    async def close(self) -> None:
        client, self.client = self.client, None
        if client is not None:
            try:
                await client.aclose()
            except (httpx.HTTPError, RuntimeError):
                pass

    async def describe(self) -> Mapping[str, Any]:
        return await self._call("GET", "/v1/health", retries=0)

    async def infer(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        return await self._call("POST", "/v1/infer", json=dict(request))

    async def _call(self, method: str, path: str, *, json: Any = None,
                    retries: int | None = None) -> dict[str, Any]:
        client = self.client
        if client is None:
            raise TransportError("model service is not open", failure_code="transport_not_open")
        budget = self.max_retries if retries is None else retries
        attempt = 0
        while True:
            outcome = await self._attempt(client, method, path, json)
            if isinstance(outcome, dict):
                return outcome
            if not isinstance(outcome, _TransientError) or attempt >= budget:
                raise outcome
            attempt += 1
            await asyncio.sleep(self.retry_backoff_s * attempt)

    async def _attempt(self, client: httpx.AsyncClient, method: str,
                       path: str, json: Any) -> dict[str, Any] | TransportError:
        try:
            response = await client.request(method, path, json=json)
        except httpx.TimeoutException as exc:
            return _TransientError(f"model service timed out after {self.timeout_s}s: {exc!r}",
                                   failure_code="model_timeout")
        except httpx.TransportError as exc:
            return _TransientError(f"model service at {self.url} is unreachable: {exc!r}",
                                   failure_code="model_unreachable")
        if response.status_code in {409, 503}:
            return _TransientError(f"model service answered {response.status_code}: {_detail(response)}",
                                   failure_code="model_busy" if response.status_code == 409 else "model_unavailable")
        if response.status_code == 422:
            return TransportError(f"request does not match the model: {_detail(response)}",
                                  failure_code="model_schema_mismatch")
        if response.status_code >= 400:
            return TransportError(f"model service failed with {response.status_code}: {_detail(response)}",
                                  failure_code="model_error")
        try:
            payload = response.json()
        except ValueError as exc:
            return TransportError(f"model service returned non-JSON: {exc}", failure_code="model_bad_response")
        if not isinstance(payload, dict):
            return TransportError("model service returned a non-object body", failure_code="model_bad_response")
        return payload


class _TransientError(TransportError):
    """Transient stateless inference failure; only this backend owns HTTP retries."""


def _detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(body, Mapping):
        return str(body.get("error") or body.get("detail") or body)[:200]
    return str(body)[:200]


def _is_loopback(host: str) -> bool:
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _number(config: Mapping[str, Any], key: str, default: float) -> float:
    value = config.get(key)
    return default if value is None else float(value)
