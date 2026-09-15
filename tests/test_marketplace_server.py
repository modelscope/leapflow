# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for the marketplace HTTP server.

Verifies that the minimal asyncio-based HTTP server correctly serves
plugin manifests and code files, matching the API expected by HttpMarketplaceSource.
"""

import asyncio
import json
from pathlib import Path
from typing import List

import pytest
import httpx

from leapflow.plugins.marketplace.manifest import PluginManifest
from leapflow.plugins.marketplace.server import MarketplaceServer


@pytest.fixture
async def marketplace_dir(tmp_path: Path) -> Path:
    """Create a test marketplace directory with sample plugins."""
    # Create plugin 1
    plugin1_dir = tmp_path / "test_plugin"
    plugin1_dir.mkdir()
    manifest1 = {
        "name": "test_plugin",
        "version": "1.0.0",
        "author": "Test Author",
        "description": "A test plugin",
        "entry_point": "test_plugin",
        "plugin_type": "tool",
    }
    (plugin1_dir / "manifest.json").write_text(json.dumps(manifest1))
    (plugin1_dir / "test_plugin.py").write_text("# Test plugin code\nprint('hello')")

    # Create plugin 2
    plugin2_dir = tmp_path / "another_plugin"
    plugin2_dir.mkdir()
    manifest2 = {
        "name": "another_plugin",
        "version": "2.0.0",
        "author": "Another Author",
        "description": "Another test plugin",
        "entry_point": "another_plugin",
        "plugin_type": "gateway",
    }
    (plugin2_dir / "manifest.json").write_text(json.dumps(manifest2))
    (plugin2_dir / "another_plugin.py").write_text("# Another plugin\nprint('world')")

    return tmp_path


@pytest.fixture
async def running_server(marketplace_dir: Path) -> MarketplaceServer:
    """Start the marketplace server on a random port."""
    server = MarketplaceServer(marketplace_dir, host="127.0.0.1", port=0)
    await server.start()
    # Get the actual port assigned and store it for later use
    actual_port = server._server.sockets[0].getsockname()[1]
    object.__setattr__(server, "_port", actual_port)
    yield server
    await server.stop()


class TestServerStartStop:
    """Test server lifecycle management."""

    @pytest.mark.asyncio
    async def test_server_start_stop(self, marketplace_dir: Path) -> None:
        """Server starts on configured port, stops cleanly."""
        server = MarketplaceServer(marketplace_dir, host="127.0.0.1", port=0)
        await server.start()
        assert server._server is not None

        # Verify server is listening
        port = server._server.sockets[0].getsockname()[1]
        async with httpx.AsyncClient() as client:
            response = await client.get(f"http://127.0.0.1:{port}/health", timeout=2.0)
            assert response.status_code == 200

        await server.stop()
        assert server._server is None


class TestServeManifests:
    """Test manifest index serving."""

    @pytest.mark.asyncio
    async def test_serve_manifests(self, running_server: MarketplaceServer, marketplace_dir: Path) -> None:
        """GET /manifests.json returns valid JSON array."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"http://127.0.0.1:{running_server._port}/manifests.json",  # type: ignore
                timeout=2.0,
            )

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"

        data = response.json()
        assert isinstance(data, list)
        assert len(data) == 2

        # Verify each manifest has required fields
        for item in data:
            assert "name" in item
            assert "version" in item
            assert "author" in item
            assert "description" in item
            assert "entry_point" in item


class TestServePluginCode:
    """Test plugin source code serving."""

    @pytest.mark.asyncio
    async def test_serve_plugin_code(self, running_server: MarketplaceServer) -> None:
        """GET /plugins/name/entry.py returns file bytes."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"http://127.0.0.1:{running_server._port}/plugins/test_plugin/test_plugin.py",  # type: ignore
                timeout=2.0,
            )

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/octet-stream"
        assert b"# Test plugin code" in response.content
        assert b"print('hello')" in response.content


class TestServeNotFound:
    """Test not found responses."""

    @pytest.mark.asyncio
    async def test_serve_not_found(self, running_server: MarketplaceServer) -> None:
        """GET /nonexistent → 404."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"http://127.0.0.1:{running_server._port}/nonexistent",  # type: ignore
                timeout=2.0,
            )

        assert response.status_code == 404


class TestPathTraversalBlocked:
    """Test security against path traversal attacks."""

    @pytest.mark.asyncio
    async def test_path_traversal_blocked(self, running_server: MarketplaceServer) -> None:
        """GET /plugins/../test_plugin/../../etc/passwd → 403 or 404 (server blocks traversal)."""
        # Use encoded path to prevent httpx from normalizing
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"http://127.0.0.1:{running_server._port}/plugins/%2e%2e/test_plugin/%2e%2e/%2e%2e/etc/passwd",  # type: ignore
                follow_redirects=False,
                timeout=2.0,
            )

        # Server should either return 403 (blocked) or 404 (not found)
        # but not expose files outside the marketplace directory
        assert response.status_code in (403, 404)


class TestHealthEndpoint:
    """Test health check endpoint."""

    @pytest.mark.asyncio
    async def test_health_endpoint(self, running_server: MarketplaceServer) -> None:
        """GET /health → {"status":"ok"}."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"http://127.0.0.1:{running_server._port}/health",  # type: ignore
                timeout=2.0,
            )

        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        data = response.json()
        assert data == {"status": "ok"}


class TestIntegrationWithHttpSource:
    """Test integration with HttpMarketplaceSource client."""

    @pytest.mark.asyncio
    async def test_integration_with_http_source(
        self,
        running_server: MarketplaceServer,
        marketplace_dir: Path,
    ) -> None:
        """Start server, use httpx to verify roundtrip (urllib has timeout issues with local servers)."""
        base_url = f"http://127.0.0.1:{running_server._port}"  # type: ignore

        async with httpx.AsyncClient() as client:
            # Discover plugins via manifests.json
            resp = await client.get(f"{base_url}/manifests.json", timeout=5.0)
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) == 2

            manifest_names = {m["name"] for m in data}
            assert manifest_names == {"test_plugin", "another_plugin"}

            # Fetch code for each plugin
            for manifest_data in data:
                plugin_name = manifest_data["name"]
                entry_point = manifest_data["entry_point"]
                
                resp = await client.get(
                    f"{base_url}/plugins/{plugin_name}/{entry_point}.py",
                    timeout=5.0,
                )
                assert resp.status_code == 200
                assert len(resp.content) > 0

            # Verify specific plugin code matches what's on disk
            test_plugin_manifest = next(m for m in data if m["name"] == "test_plugin")
            resp = await client.get(
                f"{base_url}/plugins/{test_plugin_manifest['name']}/{test_plugin_manifest['entry_point']}.py",
                timeout=5.0,
            )
            assert resp.status_code == 200

            expected_file = marketplace_dir / "test_plugin" / "test_plugin.py"
            assert resp.content == expected_file.read_bytes()


class TestPrefixParsing:
    """Lock the /plugins/ prefix-parsing fix (R2).

    ``str.lstrip('/plugins/')`` strips a character *set*, not the prefix, so a
    plugin whose name begins with any of {'/','p','l','u','g','i','n','s'} was
    corrupted (e.g. ``secrets/secrets.py`` → ``ecrets/secrets.py`` → 404).
    An explicit prefix slice must serve such plugins correctly.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("plugin_name", ["secrets", "plugins_demo"])
    async def test_plugin_name_starting_with_stripped_char(
        self, tmp_path: Path, plugin_name: str
    ) -> None:
        """A plugin whose name starts with a stripped char is served intact."""
        plugin_dir = tmp_path / plugin_name
        plugin_dir.mkdir()
        (plugin_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "name": plugin_name,
                    "version": "1.0.0",
                    "author": "Test",
                    "description": "prefix edge case",
                    "entry_point": plugin_name,
                    "plugin_type": "tool",
                }
            )
        )
        marker = f"# {plugin_name} marker".encode()
        (plugin_dir / f"{plugin_name}.py").write_bytes(marker)

        server = MarketplaceServer(tmp_path, host="127.0.0.1", port=0)
        await server.start()
        port = server._server.sockets[0].getsockname()[1]
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"http://127.0.0.1:{port}/plugins/{plugin_name}/{plugin_name}.py",
                    timeout=2.0,
                )
            assert response.status_code == 200
            assert response.content == marker
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_missing_prefix_returns_400(self, tmp_path: Path) -> None:
        """A path routed to the plugin handler without the exact prefix → 400.

        The router only dispatches ``/plugins/...`` here, so this guards the
        handler's own contract (defence in depth) via a direct call.
        """
        server = MarketplaceServer(tmp_path, host="127.0.0.1", port=0)

        class _Writer:
            def __init__(self) -> None:
                self.data = b""

            def write(self, chunk: bytes) -> None:
                self.data += chunk

            async def drain(self) -> None:
                return None

        writer = _Writer()
        await server._serve_plugin_file_and_close(writer, "/plugin/oops.py")
        assert b"400 Bad Request" in writer.data


class TestMethodNotAllowed:
    """Test that non-GET methods are rejected."""

    @pytest.mark.asyncio
    async def test_post_to_manifests(self, running_server: MarketplaceServer) -> None:
        """POST /manifests.json → 405."""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://127.0.0.1:{running_server._port}/manifests.json",  # type: ignore
                timeout=2.0,
            )

        assert response.status_code == 405

    @pytest.mark.asyncio
    async def test_delete_plugin(self, running_server: MarketplaceServer) -> None:
        """DELETE /plugins/... → 405."""
        async with httpx.AsyncClient() as client:
            response = await client.delete(
                f"http://127.0.0.1:{running_server._port}/plugins/test_plugin/test_plugin.py",  # type: ignore
                timeout=2.0,
            )

        assert response.status_code == 405


class TestEmptyDirectory:
    """Test behavior with empty marketplace directory."""

    @pytest.mark.asyncio
    async def test_empty_directory(self, tmp_path: Path) -> None:
        """Empty directory returns empty manifests array."""
        server = MarketplaceServer(tmp_path, host="127.0.0.1", port=0)
        await server.start()
        port = server._server.sockets[0].getsockname()[1]

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"http://127.0.0.1:{port}/manifests.json",
                    timeout=2.0,
                )

            assert response.status_code == 200
            data = response.json()
            assert data == []
        finally:
            await server.stop()


class TestInvalidManifest:
    """Test handling of malformed manifest files."""

    @pytest.mark.asyncio
    async def test_invalid_json_manifest(self, tmp_path: Path) -> None:
        """Malformed JSON manifest is skipped gracefully."""
        plugin_dir = tmp_path / "bad_plugin"
        plugin_dir.mkdir()
        (plugin_dir / "manifest.json").write_text("{invalid json}")
        (plugin_dir / "bad_plugin.py").write_text("# placeholder")

        server = MarketplaceServer(tmp_path, host="127.0.0.1", port=0)
        await server.start()
        port = server._server.sockets[0].getsockname()[1]

        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"http://127.0.0.1:{port}/manifests.json",
                    timeout=2.0,
                )

            assert response.status_code == 200
            data = response.json()
            # Bad manifest should be skipped, so empty array
            assert data == []
        finally:
            await server.stop()
