# Copyright (c) Alibaba, Inc. and its affiliates.
"""HTTP-based marketplace source for remote plugin discovery and download.

Fetches plugin manifests and code from a remote HTTP(S) registry endpoint.
Complements ``LocalDirectorySource`` for production use behind a real registry.

Expected server API::

    GET /manifests.json                        → list of PluginManifest JSON objects
    GET /plugins/<name>/<entry_point>.py       → plugin source code

The response for a manifest is either the full manifest URL declared in
``PluginManifest.source_url`` or, absent that, the derived path above.

Security:
    - HTTPS verification is on by default (``verify_ssl=True``).
    - Every request has a timeout (``timeout_s``, default 30s) so a slow or
      hanging registry cannot stall the caller.
    - Checksum verification stays in ``MarketplaceClient`` after fetch: this
      module only transports bytes.
"""

from __future__ import annotations

import json
import logging
import ssl
from typing import List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from leapflow.plugins.marketplace.manifest import PluginManifest

logger = logging.getLogger(__name__)


class HttpMarketplaceSource:
    """Fetches plugins from a remote HTTP(S) marketplace registry.

    Satisfies the ``MarketplaceSource`` Protocol declared in
    ``leapflow.plugins.marketplace.client``.
    """

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 30.0,
        verify_ssl: bool = True,
        auth_token: Optional[str] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._verify_ssl = verify_ssl
        self._auth_token = auth_token

    def list_manifests(self) -> List[PluginManifest]:
        """Fetch and parse the manifest index from the remote server."""
        url = f"{self._base_url}/manifests.json"
        data = self._fetch(url)
        if data is None:
            return []
        try:
            raw_list = json.loads(data)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Failed to decode marketplace manifests: %s", exc)
            return []

        if not isinstance(raw_list, list):
            logger.warning(
                "Marketplace at %s returned non-list manifest index (type=%s)",
                url,
                type(raw_list).__name__,
            )
            return []

        manifests: List[PluginManifest] = []
        for item in raw_list:
            try:
                manifests.append(PluginManifest.from_json(json.dumps(item)))
            except (TypeError, ValueError, KeyError) as exc:
                logger.warning("Skipping malformed manifest entry: %s", exc)
                continue
        return manifests

    def fetch_code(self, manifest: PluginManifest) -> Optional[bytes]:
        """Download the plugin source code identified by ``manifest``."""
        url = manifest.source_url or (
            f"{self._base_url}/plugins/{manifest.name}/{manifest.entry_point}.py"
        )
        return self._fetch(url)

    def _fetch(self, url: str) -> Optional[bytes]:
        """HTTP GET with timeout and optional bearer auth. ``None`` on failure."""
        req = Request(url)
        if self._auth_token:
            req.add_header("Authorization", f"Bearer {self._auth_token}")

        context: Optional[ssl.SSLContext] = None
        if url.lower().startswith("https://") and not self._verify_ssl:
            # Explicit opt-out. Deliberately unsafe; only for internal registries.
            context = ssl.create_default_context()
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE

        try:
            with urlopen(req, timeout=self._timeout_s, context=context) as resp:
                return resp.read()
        except HTTPError as exc:
            logger.warning("HTTP %s fetching %s", exc.code, url)
            return None
        except (URLError, OSError, TimeoutError) as exc:
            logger.warning("Failed to fetch %s: %s", url, exc)
            return None
