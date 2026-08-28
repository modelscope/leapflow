"""Native ToolPlugin wrapper for an installed restricted DSH bundle."""
from __future__ import annotations

from typing import Any

from leapflow.plugins.dsh.capabilities import DshCapabilityBroker
from leapflow.plugins.dsh.descriptor import DshPluginDescriptor, DshToolDescriptor
from leapflow.plugins.dsh.node_host import DshNodeHost, DshRuntimeUnavailable
from leapflow.plugins.protocol import ToolMetadata


class DshBridgePlugin:
    """Expose runtime-discovered DSH tools through LeapFlow's ToolPlugin contract.

    Each invocation gets its own Node worker. That keeps plugin globals isolated
    between sessions and avoids an idle untrusted process. P1 may introduce a
    supervised per-session pool after lifecycle and resource evidence exists.
    """

    def __init__(self, descriptor: dict[str, Any] | DshPluginDescriptor) -> None:
        self._descriptor = (
            descriptor
            if isinstance(descriptor, DshPluginDescriptor)
            else DshPluginDescriptor.from_dict(descriptor)
        )
        self._web_fetch: Any = None
        self._tools = [self._metadata(item) for item in self._descriptor.tools]

    @property
    def plugin_id(self) -> str:
        return self._descriptor.plugin_id

    @property
    def category(self) -> str:
        return self._descriptor.category

    @property
    def dependencies(self) -> list[str]:
        return []

    @property
    def tools(self) -> list[ToolMetadata]:
        return list(self._tools)

    @property
    def descriptor(self) -> DshPluginDescriptor:
        return self._descriptor

    def bind_runtime(self, **deps: Any) -> None:
        # Tests can inject a hermetic web_fetch; production uses the governed
        # implementation lazily through DshCapabilityBroker.
        if "web_fetch" in deps:
            self._web_fetch = deps["web_fetch"]

    def _metadata(self, tool: DshToolDescriptor) -> ToolMetadata:
        async def _handler(**kwargs: Any) -> Any:
            return await self._invoke(tool.name, kwargs)

        return ToolMetadata(
            name=tool.name,
            description=tool.description,
            parameters_schema=dict(tool.parameters_schema),
            handler=_handler,
            x_leapflow={
                "category": "bridge",
                "runtime": "node",
                "bridge": "dsh_ndjson_v1",
                "risk_level": "external" if self._descriptor.permissions else "medium",
                "requires_approval": False,
                "execution_policy": "parallel_safe",
                "source_bundle_sha256": self._descriptor.bundle_sha256,
                "limitations": list(self._descriptor.limitations),
            },
            mutates_state=False,
        )

    async def _invoke(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        try:
            self._descriptor.verify_integrity()
        except (OSError, ValueError) as exc:
            return {
                "ok": False,
                "error": str(exc),
                "error_type": "integrity_error",
                "retryable": False,
            }
        settings = _settings()
        broker = DshCapabilityBroker(web_fetch=self._web_fetch)
        host = DshNodeHost(
            self._descriptor.bundle_root,
            source_kind=self._descriptor.source_kind,
            entry_point=self._descriptor.entry_point,
            broker=broker,
            invoke_timeout_s=float(
                getattr(settings, "plugins_dsh_invoke_timeout_s", 30.0)
            ),
            discovery_timeout_s=float(
                getattr(settings, "plugins_dsh_discovery_timeout_s", 10.0)
            ),
            max_line_bytes=int(
                getattr(settings, "plugins_dsh_max_message_bytes", 1_000_000)
            ),
            max_stderr_bytes=int(
                getattr(settings, "plugins_dsh_max_stderr_bytes", 64_000)
            ),
            max_memory_mb=int(
                getattr(settings, "plugins_dsh_max_memory_mb", 128)
            ),
        )
        try:
            response = await host.invoke(tool_name, arguments)
        except DshRuntimeUnavailable as exc:
            return {
                "ok": False,
                "error": str(exc),
                "error_type": "runtime_unavailable",
                "retryable": False,
            }
        finally:
            await host.stop()
        if not response.ok:
            return {
                "ok": False,
                "error": response.error,
                "error_type": response.error_type or "dsh_runtime_error",
                "retryable": response.error_type in {"timeout", "worker_closed"},
            }
        if isinstance(response.result, dict):
            result = dict(response.result)
            result.setdefault("ok", True)
            return result
        return {"ok": True, "result": response.result}


def _settings() -> Any:
    try:
        from leapflow.config import get_settings

        return get_settings()
    except (ImportError, RuntimeError):
        return object()
