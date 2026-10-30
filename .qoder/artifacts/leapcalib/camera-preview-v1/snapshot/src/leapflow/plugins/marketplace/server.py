# Copyright (c) Alibaba, Inc. and its affiliates.
"""Plugin Marketplace HTTP server.

A minimal asyncio-based HTTP server that serves plugin manifests and code
from a local directory, implementing the same API that HttpMarketplaceSource
expects.

Usage:
    python -m leapflow.plugins.marketplace.server --port 8080 --dir ./marketplace

Directory layout (same as LocalDirectorySource):
    <dir>/
        <plugin_name>/
            manifest.json
            <entry_point>.py

This is a development/testing server. For production use, deploy behind
a reverse proxy with TLS, rate limiting, and proper auth.
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class MarketplaceServer:
    """Async HTTP server for plugin marketplace."""

    def __init__(self, directory: Path, *, host: str = "127.0.0.1", port: int = 8080) -> None:
        self._directory = Path(directory).resolve()
        self._host = host
        self._port = port
        self._server: Optional[asyncio.Server] = None

    async def start(self) -> None:
        """Start serving."""
        self._server = await asyncio.start_server(
            self._handle_request, self._host, self._port
        )
        logger.info("Marketplace server running on http://%s:%d (serving from %s)",
                   self._host, self._port, self._directory)

    async def stop(self) -> None:
        """Stop serving."""
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def _handle_request(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Minimal HTTP request handler."""
        try:
            # Read request line
            request_line = await asyncio.wait_for(reader.readline(), timeout=5.0)
            if not request_line:
                return

            # Parse method and path
            parts = request_line.decode().strip().split(" ")
            method = parts[0] if parts else "GET"
            path = parts[1] if len(parts) > 1 else "/"

            # Read headers (discard) - handle both CRLF and LF line endings
            while True:
                header = await asyncio.wait_for(reader.readline(), timeout=5.0)
                # Handle empty line that marks end of headers
                if header in (b"\r\n", b"\n", b""):
                    break
                # Stop if we get a blank line
                if header.strip() == b"":
                    break

            # Route
            if method != "GET":
                await self._respond_and_close(writer, 405, b"Method Not Allowed")
            elif path == "/manifests.json":
                await self._serve_manifests_and_close(writer)
            elif path.startswith("/plugins/"):
                await self._serve_plugin_file_and_close(writer, path)
            elif path == "/health":
                await self._respond_and_close(writer, 200, b'{"status":"ok"}', content_type="application/json")
            else:
                await self._respond_and_close(writer, 404, b"Not Found")
        except (asyncio.TimeoutError, ConnectionResetError, OSError):
            pass
        finally:
            # Force close the connection immediately after responding
            try:
                writer.close()
                # Don't wait for wait_closed() - just close immediately
                # This ensures urllib can read the full response
            except (OSError, RuntimeError):
                pass

    async def _respond_and_close(self, writer: asyncio.StreamWriter, status: int, body: bytes, *, content_type: str = "text/plain") -> None:
        """Write an HTTP response and close the connection."""
        status_text = {
            200: "OK",
            400: "Bad Request",
            403: "Forbidden",
            404: "Not Found",
            405: "Method Not Allowed",
            500: "Internal Server Error",
        }.get(status, "Unknown")

        header = (
            f"HTTP/1.1 {status} {status_text}\r\n"
            f"Content-Type: {content_type}\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Connection: close\r\n"
            f"\r\n"
        ).encode()
        # Write all data at once to ensure atomicity
        writer.write(header + body)
        await writer.drain()

    async def _serve_manifests_and_close(self, writer: asyncio.StreamWriter) -> None:
        """Serve the combined manifest index and close connection."""
        manifests = []
        if self._directory.exists():
            for plugin_dir in sorted(self._directory.iterdir()):
                if not plugin_dir.is_dir():
                    continue
                manifest_file = plugin_dir / "manifest.json"
                if manifest_file.exists():
                    try:
                        data = json.loads(manifest_file.read_text())
                        manifests.append(data)
                    except (json.JSONDecodeError, OSError) as exc:
                        logger.warning("Failed to load manifest from %s: %s", manifest_file, exc)
                        continue

        body = json.dumps(manifests, indent=2).encode()
        await self._respond_and_close(writer, 200, body, content_type="application/json")

    async def _serve_plugin_file_and_close(self, writer: asyncio.StreamWriter, path: str) -> None:
        """Serve a plugin source file: /plugins/<name>/<file>.py and close connection."""
        # path is like /plugins/my_plugin/my_plugin.py. Strip the exact route
        # prefix, never a character set: ``str.lstrip('/plugins/')`` removes any
        # leading char in {'/','p','l','u','g','i','n','s'}, so a plugin whose
        # name starts with one of those (e.g. "secrets") would be corrupted into
        # a different, potentially traversal-adjacent path.
        prefix = "/plugins/"
        if not path.startswith(prefix):
            await self._respond_and_close(writer, 400, b"Bad Request")
            return
        relative = path[len(prefix):]
        file_path = self._directory / relative

        # Security: prevent path traversal
        try:
            resolved = file_path.resolve()
            if not str(resolved).startswith(str(self._directory)):
                await self._respond_and_close(writer, 403, b"Forbidden")
                return
        except (OSError, ValueError):
            await self._respond_and_close(writer, 400, b"Bad Request")
            return

        if not file_path.exists() or not file_path.is_file():
            await self._respond_and_close(writer, 404, b"Not Found")
            return

        try:
            body = file_path.read_bytes()
            await self._respond_and_close(writer, 200, body, content_type="application/octet-stream")
        except OSError as exc:
            logger.warning("Failed to read file %s: %s", file_path, exc)
            await self._respond_and_close(writer, 500, b"Internal Server Error")


async def main(directory: str, host: str = "127.0.0.1", port: int = 8080) -> None:
    """Run the marketplace server until interrupted."""
    server = MarketplaceServer(Path(directory), host=host, port=port)
    await server.start()
    try:
        await asyncio.Event().wait()  # Run forever
    except asyncio.CancelledError:
        pass
    finally:
        await server.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LeapFlow Plugin Marketplace Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on (default: 8080)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--dir", required=True, help="Directory containing plugin packages")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    asyncio.run(main(args.dir, args.host, args.port))
