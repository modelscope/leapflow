# Copyright (c) Alibaba, Inc. and its affiliates.
"""Bounded single-origin HTTP for specialist SDK clients, before SDK parsing."""
from __future__ import annotations

import httpx

from leapflow.llm.services.protocol import ServiceError, ServiceSpec


class BoundedTransport(httpx.AsyncBaseTransport):
    """Buffer bounded identity-encoded responses; never follow a redirect or proxy."""

    def __init__(self, spec: ServiceSpec, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self._spec = spec
        self._origin = httpx.URL(spec.base_url)
        self._inner = transport or httpx.AsyncHTTPTransport(trust_env=False, retries=0)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        url = request.url
        if (url.scheme, url.host, url.port) != (self._origin.scheme, self._origin.host, self._origin.port):
            raise ServiceError("Provider attempted a different endpoint origin", code="model_origin_changed", category="security")
        body = bytearray()
        async for chunk in request.stream:
            if len(body) + len(chunk) > self._spec.limits.max_request_bytes:
                raise ServiceError("Encoded request exceeds its byte budget", code="model_request_too_large")
            body.extend(chunk)
        # The SDK normally supplies buffered JSON. Also bound a streaming body
        # before allowing the inner transport to consume it.
        request.stream = httpx.ByteStream(bytes(body))
        request.headers["accept-encoding"] = "identity"
        response = await self._inner.handle_async_request(request)
        try:
            if 300 <= response.status_code < 400:
                raise ServiceError("Model endpoint redirects are refused; configure the final approved origin",
                                   code="model_redirect", category="security")
            # Refuse unsolicited compression instead of allocating an unbounded decompressor.
            if response.headers.get("content-encoding", "identity").lower() != "identity":
                raise ServiceError("Compressed model response refused; serve identity encoding",
                                   code="model_response_encoding")
            length = response.headers.get("content-length", "")
            if length.isdigit() and int(length) > self._spec.limits.max_response_bytes:
                raise ServiceError("Model response exceeds its byte budget", code="model_response_too_large")
            data = bytearray()
            chunks = response.aiter_bytes() if response.is_stream_consumed else response.aiter_raw()
            async for chunk in chunks:
                if len(data) + len(chunk) > self._spec.limits.max_response_bytes:
                    raise ServiceError("Model response exceeds its byte budget", code="model_response_too_large")
                data.extend(chunk)
            return httpx.Response(response.status_code, headers=response.headers,
                                  content=bytes(data), extensions=response.extensions)
        finally:
            await response.aclose()

    async def aclose(self) -> None:
        await self._inner.aclose()


def build_client(spec: ServiceSpec) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=BoundedTransport(spec), trust_env=False,
                            follow_redirects=False, timeout=spec.limits.deadline_s)
