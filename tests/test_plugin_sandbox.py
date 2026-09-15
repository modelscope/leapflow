# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for the plugin sandbox (process isolation for untrusted plugins)."""

from __future__ import annotations

import asyncio
import sys
from typing import Any

import pytest

from leapflow.plugins.sandbox.protocol import SandboxRequest, SandboxResponse


# ---------------------------------------------------------------------------
# Protocol unit tests
# ---------------------------------------------------------------------------


class TestSandboxProtocol:
    """SandboxRequest/Response serialization roundtrips."""

    def test_request_roundtrip_basic(self) -> None:
        req = SandboxRequest(
            request_id="abc-123",
            method="invoke_tool",
            tool_name="text_search",
            arguments={"params": {"text": "hello", "pattern": "ell"}},
        )
        json_str = req.to_json()
        restored = SandboxRequest.from_json(json_str)
        assert restored.request_id == "abc-123"
        assert restored.method == "invoke_tool"
        assert restored.tool_name == "text_search"
        assert restored.arguments == {"params": {"text": "hello", "pattern": "ell"}}

    def test_request_roundtrip_defaults(self) -> None:
        req = SandboxRequest(request_id="x", method="ping")
        restored = SandboxRequest.from_json(req.to_json())
        assert restored.tool_name == ""
        assert restored.arguments == {}

    def test_response_roundtrip_ok(self) -> None:
        resp = SandboxResponse(
            request_id="r1", ok=True, result={"count": 2, "matches": [[0, "a"]]}
        )
        restored = SandboxResponse.from_json(resp.to_json())
        assert restored.ok is True
        assert restored.result["count"] == 2
        assert restored.error == ""

    def test_response_roundtrip_error(self) -> None:
        resp = SandboxResponse(
            request_id="r2",
            ok=False,
            error="Tool not found: bogus",
            error_type="KeyError",
        )
        restored = SandboxResponse.from_json(resp.to_json())
        assert restored.ok is False
        assert "bogus" in restored.error
        assert restored.error_type == "KeyError"

    def test_request_is_frozen(self) -> None:
        req = SandboxRequest(request_id="f", method="ping")
        with pytest.raises(Exception):  # FrozenInstanceError
            req.request_id = "changed"  # type: ignore[misc]

    def test_response_is_frozen(self) -> None:
        resp = SandboxResponse(request_id="f", ok=True)
        with pytest.raises(Exception):
            resp.ok = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Integration tests (subprocess)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sandbox_host_start_stop() -> None:
    """Launch worker, ping it, then stop gracefully."""
    from leapflow.plugins.sandbox.sandbox_host import SandboxHost

    host = SandboxHost("leapflow.plugins.tool_plugins.text_utils")
    await host.start()
    try:
        ok = await host.ping()
        assert ok is True, "Ping should succeed after start"
    finally:
        await host.stop()
    # After stop, ping should fail
    ok = await host.ping()
    assert ok is False


@pytest.mark.asyncio
async def test_sandbox_invoke_tool() -> None:
    """Invoke text_search through the sandbox and verify the result."""
    from leapflow.plugins.sandbox.sandbox_host import SandboxHost

    host = SandboxHost("leapflow.plugins.tool_plugins.text_utils")
    await host.start()
    try:
        resp = await host.invoke(
            "text_search", {"params": {"text": "hello world", "pattern": "world"}}
        )
        assert resp.ok is True
        assert resp.result["ok"] is True
        assert resp.result["count"] == 1
        assert resp.result["matches"][0][1] == "world"
    finally:
        await host.stop()


@pytest.mark.asyncio
async def test_sandbox_invoke_timeout() -> None:
    """A tool that hangs should produce a timeout response."""
    import os
    import tempfile

    from leapflow.plugins.sandbox.sandbox_host import SandboxHost

    # Create a temp plugin that has a hanging tool
    plugin_code = '''
"""Hanging plugin for timeout testing."""
import asyncio
from leapflow.plugins.protocol import ToolMetadata

async def hang_forever(params):
    await asyncio.sleep(9999)

class HangPlugin:
    @property
    def plugin_id(self): return "hang"
    @property
    def category(self): return "test"
    @property
    def tools(self):
        return [ToolMetadata(
            name="hang_tool",
            description="Hangs forever",
            parameters_schema={"type": "object", "properties": {}},
            handler=hang_forever,
        )]
    @property
    def dependencies(self): return []
    def bind_runtime(self, **deps): pass

plugin = HangPlugin()
'''
    # Write to a temp file in a discoverable location
    tmp_dir = tempfile.mkdtemp()
    plugin_file = os.path.join(tmp_dir, "hang_plugin.py")
    with open(plugin_file, "w") as f:
        f.write(plugin_code)

    # Add tmp_dir to sys.path so the subprocess can import it
    # We'll use a different approach: write a wrapper that adds to path
    wrapper_code = f'''
import sys
sys.path.insert(0, {tmp_dir!r})
from hang_plugin import plugin
'''
    wrapper_file = os.path.join(tmp_dir, "hang_wrapper.py")
    with open(wrapper_file, "w") as f:
        f.write(wrapper_code)

    # Use a very short timeout
    host = SandboxHost("hang_plugin", invoke_timeout_s=0.5)
    # Manually start with custom PYTHONPATH
    host._proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "leapflow.plugins.sandbox.worker",
        "hang_plugin",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "PYTHONPATH": tmp_dir},
    )
    try:
        resp = await host.invoke("hang_tool", {"params": {}})
        assert resp.ok is False
        assert "timed out" in resp.error
    finally:
        await host.stop()


@pytest.mark.asyncio
async def test_sandbox_invoke_error_isolated() -> None:
    """A tool that raises an exception returns an error but host survives."""
    import os
    import tempfile

    from leapflow.plugins.sandbox.sandbox_host import SandboxHost

    plugin_code = '''
"""Plugin with a crashing tool."""
from leapflow.plugins.protocol import ToolMetadata

def crash_tool(params):
    raise ValueError("intentional test crash")

def ok_tool(params):
    return {"ok": True, "value": 42}

class CrashPlugin:
    @property
    def plugin_id(self): return "crash"
    @property
    def category(self): return "test"
    @property
    def tools(self):
        return [
            ToolMetadata(
                name="crash_tool",
                description="Always crashes",
                parameters_schema={"type": "object", "properties": {}},
                handler=crash_tool,
            ),
            ToolMetadata(
                name="ok_tool",
                description="Always works",
                parameters_schema={"type": "object", "properties": {}},
                handler=ok_tool,
            ),
        ]
    @property
    def dependencies(self): return []
    def bind_runtime(self, **deps): pass

plugin = CrashPlugin()
'''
    tmp_dir = tempfile.mkdtemp()
    plugin_file = os.path.join(tmp_dir, "crash_plugin.py")
    with open(plugin_file, "w") as f:
        f.write(plugin_code)

    host = SandboxHost("crash_plugin", invoke_timeout_s=5.0)
    host._proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "leapflow.plugins.sandbox.worker",
        "crash_plugin",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, "PYTHONPATH": tmp_dir},
    )
    try:
        # Tool that crashes
        resp = await host.invoke("crash_tool", {"params": {}})
        assert resp.ok is False
        assert "intentional test crash" in resp.error
        assert resp.error_type == "ValueError"

        # Host should still be alive — invoke another tool
        resp2 = await host.invoke("ok_tool", {"params": {}})
        assert resp2.ok is True
        assert resp2.result["value"] == 42
    finally:
        await host.stop()


@pytest.mark.asyncio
async def test_sandbox_list_tools() -> None:
    """list_tools returns the tool names loaded in the sandbox."""
    from leapflow.plugins.sandbox.sandbox_host import SandboxHost

    host = SandboxHost("leapflow.plugins.tool_plugins.text_utils")
    await host.start()
    try:
        tools = await host.list_tools()
        assert "text_search" in tools
        assert "text_replace" in tools
    finally:
        await host.stop()


# ---------------------------------------------------------------------------
# SandboxedToolPlugin protocol conformance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sandboxed_plugin_protocol_conformance() -> None:
    """SandboxedToolPlugin satisfies the ToolPlugin Protocol."""
    from leapflow.plugins.protocol import ToolMetadata, ToolPlugin
    from leapflow.plugins.sandbox.sandbox_host import SandboxedToolPlugin, SandboxHost

    host = SandboxHost("leapflow.plugins.tool_plugins.text_utils")
    metadatas = [
        ToolMetadata(
            name="test_tool",
            description="A test tool",
            parameters_schema={"type": "object", "properties": {}},
            handler=lambda **kw: None,
            x_leapflow={"category": "test"},
            mutates_state=False,
        ),
    ]
    sandboxed = SandboxedToolPlugin(
        plugin_id="test_sandboxed",
        category="test",
        tool_metadatas=metadatas,
        host=host,
    )

    # Protocol conformance checks
    assert isinstance(sandboxed, ToolPlugin)
    assert sandboxed.plugin_id == "test_sandboxed"
    assert sandboxed.category == "test"
    assert len(sandboxed.tools) == 1
    assert sandboxed.tools[0].name == "test_tool"
    assert sandboxed.dependencies == []
    # bind_runtime should not raise
    sandboxed.bind_runtime(some_dep="value")

    # The handler should be async (proxied)
    import inspect

    assert inspect.iscoroutinefunction(sandboxed.tools[0].handler)
