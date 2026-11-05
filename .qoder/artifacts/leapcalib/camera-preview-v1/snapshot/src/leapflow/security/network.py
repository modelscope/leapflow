# Copyright (c) Alibaba, Inc. and its affiliates.
"""Outbound URL classification for the network egress gate.

Splitting this out of ``risk.py`` keeps the risk classifier synchronous and
I/O-free: deciding whether a host is internal requires DNS resolution, which is
I/O and must not run inside an approval decision. The tool resolves the target
here (off the event loop), then passes the verdict as action metadata for the
classifier to judge.

Name resolution matters, not just literal addresses: a public hostname can point
at ``127.0.0.1`` or a cloud metadata address, so a gate that only inspected the
literal host would pass exactly the requests worth stopping.
"""

from __future__ import annotations

import asyncio
import ipaddress
import socket
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urlsplit

# Only these reach the network. Everything else (file://, gopher://, data:, ...)
# is refused as a malformed request rather than judged for risk: a fetch tool
# that could read local files would bypass the workspace boundary entirely.
ALLOWED_SCHEMES: frozenset[str] = frozenset({"http", "https"})

# Cloud instance metadata services. Named explicitly, separately from the ranges
# that enclose them, because the consequence is specific and worth stating in an
# approval prompt: these endpoints hand out instance credentials.
_METADATA_ADDRESSES: frozenset[str] = frozenset({
    "169.254.169.254",       # AWS / GCP / Azure / OpenStack IMDS
    "fd00:ec2::254",         # AWS IMDSv2 over IPv6
    "100.100.100.200",       # Alibaba Cloud ECS metadata
})

_DEFAULT_PORTS = {"http": 80, "https": 443}


class UrlRejected(ValueError):
    """Raised when a URL cannot be fetched at all, regardless of approval."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class NetworkTarget:
    """A resolved fetch target and the verdict the approval path needs."""

    url: str
    scheme: str
    host: str
    port: int
    origin: str
    category: str
    addresses: tuple[str, ...] = ()
    has_credentials: bool = False

    @property
    def is_internal(self) -> bool:
        """Whether the target reaches something other than the public internet."""
        return self.category != "public"

    def to_metadata(self) -> dict[str, object]:
        """Return the fields the risk classifier and audit trail consume."""
        return {
            "scheme": self.scheme,
            "host": self.host,
            "port": self.port,
            "origin": self.origin,
            "target_category": self.category,
            "has_credentials": self.has_credentials,
            "resolved_addresses": list(self.addresses),
        }


def _category_for_address(address: str) -> str:
    """Classify one resolved IP address into a trust category.

    "public" is decided by ``is_global`` rather than by ``not is_private``: the
    private-address test misses ranges that are unroutable but not RFC1918, most
    consequentially the shared address space (100.64.0.0/10) that carries Alibaba
    Cloud's metadata service. Anything the stdlib does not consider globally
    routable is therefore treated as internal, which fails closed for ranges we
    have not enumerated.
    """
    if address in _METADATA_ADDRESSES:
        return "metadata"
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        # An address the resolver returned but ipaddress cannot parse is not
        # something to optimistically treat as public.
        return "reserved"
    if ip.is_loopback:
        return "loopback"
    if ip.is_link_local:
        return "link_local"
    if ip.is_unspecified:
        return "unspecified"
    if ip.is_private:
        return "private"
    if not ip.is_global:
        return "reserved"
    return "public"


def _worst_category(categories: Iterable[str]) -> str:
    """Return the most restrictive category among resolved addresses.

    A hostname with both a public and a loopback record must be treated as
    loopback: the connection may take either, so the safe reading is the worse
    one.
    """
    # Materialized before the scan: ``categories`` may be a generator, and
    # re-evaluating it inside the comprehension would consume it on the first
    # probe and then read as empty for every remaining category.
    present = set(categories)
    order = ("metadata", "unspecified", "loopback", "link_local", "private", "reserved", "public")
    ranked = [c for c in order if c in present]
    return ranked[0] if ranked else "unknown"


def _split_url(url: str) -> tuple[str, str, int, bool]:
    parts = urlsplit(url.strip())
    scheme = (parts.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise UrlRejected(
            "unsupported_scheme",
            f"Only http and https URLs can be fetched; got {scheme or 'no scheme'!r}.",
        )
    host = (parts.hostname or "").strip()
    if not host:
        raise UrlRejected("missing_host", "The URL has no host component.")
    try:
        port = parts.port or _DEFAULT_PORTS[scheme]
    except ValueError as exc:  # malformed port, e.g. https://host:notaport/
        raise UrlRejected("invalid_port", f"The URL port is not a number: {exc}") from exc
    return host, scheme, port, bool(parts.username or parts.password)


async def classify_url(url: str, *, resolve: bool = True) -> NetworkTarget:
    """Return the classified target for ``url``.

    ``resolve=False`` skips DNS and classifies from the literal host only. That
    is for offline/unit contexts; leaving it on is what catches a public name
    pointing at an internal address.
    """
    host, scheme, port, has_credentials = _split_url(url)
    default_port = _DEFAULT_PORTS[scheme]
    origin = f"{scheme}://{host}" if port == default_port else f"{scheme}://{host}:{port}"

    literal = host.strip("[]")
    try:
        ipaddress.ip_address(literal)
    except ValueError:
        addresses: tuple[str, ...] = ()
        if resolve:
            addresses = await _resolve(host, port)
    else:
        addresses = (literal,)

    if addresses:
        category = _worst_category(_category_for_address(a) for a in addresses)
    elif resolve:
        # Resolution failed. Report it as a target we could not vet rather than
        # letting the request through unclassified.
        raise UrlRejected("dns_error", f"Could not resolve host {host!r}.")
    else:
        category = "unknown"

    return NetworkTarget(
        url=url,
        scheme=scheme,
        host=host,
        port=port,
        origin=origin,
        category=category,
        addresses=addresses,
        has_credentials=has_credentials,
    )


async def _resolve(host: str, port: int) -> tuple[str, ...]:
    """Resolve ``host`` to its addresses without blocking the event loop."""
    loop = asyncio.get_running_loop()
    try:
        infos = await loop.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except (socket.gaierror, OSError):
        return ()
    seen: list[str] = []
    for info in infos:
        address = str(info[4][0])
        if address not in seen:
            seen.append(address)
    return tuple(seen)


__all__ = [
    "ALLOWED_SCHEMES",
    "NetworkTarget",
    "UrlRejected",
    "classify_url",
]
