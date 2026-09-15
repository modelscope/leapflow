# Copyright (c) Alibaba, Inc. and its affiliates.
"""Web access plugin — read-only HTTP fetch for the agent loop."""

from __future__ import annotations

from leapflow.plugins.protocol import ToolMetadata


class WebAccessPlugin:
    """First-class read-only HTTP access (replaces curl through shell_run)."""

    @property
    def plugin_id(self) -> str:
        return "web_access"

    @property
    def category(self) -> str:
        return "network"

    @property
    def tools(self) -> list[ToolMetadata]:
        from leapflow.tools.web_fetch import web_fetch

        return [
            ToolMetadata(
                name="web_fetch",
                description=(
                    "Read a URL over HTTP(S) and get back extracted, context-sized content: "
                    "parsed JSON for API endpoints, readable text plus links for web pages. "
                    "Use this for anything on the internet \u2014 prices, docs, releases, articles "
                    "\u2014 instead of running curl through shell_run: it reports real HTTP status "
                    "codes, retries rate limits on its own, and is a plain read so a retry is "
                    "always safe. For JSON APIs pass `select` with a dotted path (e.g. "
                    "'chart.result.0.meta') to return just that part instead of the whole "
                    "payload."
                ),
                parameters_schema={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "http(s) URL to read"},
                        "select": {
                            "type": "string",
                            "description": (
                                "Optional dotted path into a JSON response, list indices "
                                "allowed, e.g. 'chart.result.0.meta.regularMarketPrice'"
                            ),
                        },
                        "timeout": {
                            "type": "number",
                            "description": "Timeout in seconds (default from config)",
                        },
                        "max_bytes": {
                            "type": "integer",
                            "description": "Response size cap in bytes",
                        },
                    },
                    "required": ["url"],
                },
                handler=web_fetch,
                x_leapflow={
                    "category": "network",
                    "risk_level": "read_only",
                    "schema_cost": "low",
                    "requires_approval": False,
                    "mutates_state": False,
                    "idempotency_scope": "turn",
                },
                provides_capabilities=("network.http_get",),
            ),
        ]

    @property
    def dependencies(self) -> list[str]:
        return []

    def bind_runtime(self, **deps: object) -> None:
        pass


# Module-level instance for plugin discovery
plugin = WebAccessPlugin()
