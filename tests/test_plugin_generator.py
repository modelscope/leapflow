# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for LLM-driven plugin generation and validation pipeline."""

from __future__ import annotations

import pytest

from leapflow.learning.plugin_generator import (
    PluginGenerationRequest,
    PluginGenerator,
    PluginValidator,
)


pytestmark = pytest.mark.unit


# ── Fixture: a realistic, well-formed ToolPlugin as a canned code string ──

VALID_ECHO_PLUGIN_CODE = '''
from typing import Any
from leapflow.plugins.protocol import ToolMetadata


class EchoPlugin:
    @property
    def plugin_id(self) -> str:
        return "echo_test"

    @property
    def category(self) -> str:
        return "custom"

    @property
    def dependencies(self) -> list:
        return []

    def bind_runtime(self, **deps: Any) -> None:
        pass

    @property
    def tools(self) -> list:
        return [
            ToolMetadata(
                name="echo",
                description="Echo the input arguments back",
                parameters_schema={"type": "object", "properties": {}},
                handler=self._echo,
                x_leapflow={"category": "custom", "risk_level": "read_only"},
            )
        ]

    async def _echo(self, **kwargs: Any) -> dict:
        return {"ok": True, "echo": kwargs}


plugin = EchoPlugin()
'''


class _FakeLLM:
    """Minimal LLM stub matching the ``achat(messages)`` shape."""

    def __init__(self, response: str) -> None:
        self._response = response
        self.calls: list[list[dict]] = []

    async def achat(self, messages):  # type: ignore[no-untyped-def]
        self.calls.append(messages)
        return self._response


# ── Validator: stage 1 (syntax) ──


@pytest.mark.asyncio
async def test_validator_rejects_syntax_error() -> None:
    validator = PluginValidator()
    result = await validator.validate("bad_syntax", "def broken(:\n  pass")
    assert not result.ok
    assert result.stage == "syntax"
    assert "Syntax error" in result.error


# ── Validator: stage 2 (structure) ──


@pytest.mark.asyncio
async def test_validator_rejects_missing_plugin() -> None:
    validator = PluginValidator()
    # Parses fine but never assigns `plugin`
    result = await validator.validate("no_plugin", "x = 1\ny = 2\n")
    assert not result.ok
    assert result.stage == "structure"
    assert "plugin" in result.error


@pytest.mark.asyncio
async def test_validator_rejects_dangerous_eval() -> None:
    validator = PluginValidator()
    code = "plugin = eval('1+1')\n"
    result = await validator.validate("evil", code)
    assert not result.ok
    assert result.stage == "structure"
    assert "eval" in result.error


@pytest.mark.asyncio
async def test_validator_rejects_dangerous_os_system() -> None:
    validator = PluginValidator()
    code = "import os\nos.system('echo hi')\nplugin = None\n"
    result = await validator.validate("evil2", code)
    assert not result.ok
    assert result.stage == "structure"
    assert "system" in result.error


# ── Validator: stage 3+4 (runtime import + protocol) ──


@pytest.mark.asyncio
async def test_validator_accepts_valid_plugin() -> None:
    validator = PluginValidator()
    result = await validator.validate("echo_test", VALID_ECHO_PLUGIN_CODE)
    assert result.ok, f"Expected pass, got stage={result.stage}, error={result.error}"
    assert result.stage == "passed"
    assert result.exposed_tools == ["echo"]


@pytest.mark.asyncio
async def test_validator_rejects_none_x_leapflow() -> None:
    validator = PluginValidator()
    result = await validator.validate(
        "echo_test",
        VALID_ECHO_PLUGIN_CODE.replace(
            'x_leapflow={"category": "custom", "risk_level": "read_only"},',
            'x_leapflow=None,',
        ),
    )
    assert not result.ok
    assert "x_leapflow must be a dict" in result.error


@pytest.mark.asyncio
async def test_validator_rejects_bad_parameters_schema() -> None:
    validator = PluginValidator()
    result = await validator.validate(
        "echo_test",
        VALID_ECHO_PLUGIN_CODE.replace(
            'parameters_schema={"type": "object", "properties": {}},',
            'parameters_schema={"type": "array"},',
        ),
    )
    assert not result.ok
    assert "parameters_schema.type" in result.error


@pytest.mark.asyncio
async def test_validator_rejects_sync_handler() -> None:
    validator = PluginValidator()
    result = await validator.validate(
        "echo_test",
        VALID_ECHO_PLUGIN_CODE.replace("async def _echo", "def _echo"),
    )
    assert not result.ok
    assert "handler must be an async function" in result.error


@pytest.mark.asyncio
async def test_validator_rejects_mutating_tool_without_approval_metadata() -> None:
    validator = PluginValidator()
    result = await validator.validate(
        "echo_test",
        VALID_ECHO_PLUGIN_CODE.replace(
            'x_leapflow={"category": "custom", "risk_level": "read_only"},',
            'x_leapflow={"category": "custom", "risk_level": "high"},\n                mutates_state=True,',
        ),
    )
    assert not result.ok
    assert "requires_approval" in result.error


@pytest.mark.asyncio
async def test_validator_rejects_non_protocol() -> None:
    """A module-level `plugin` that doesn't satisfy the ToolPlugin Protocol fails."""
    validator = PluginValidator()
    # `plugin` is a bare dict — no protocol conformance
    code = "plugin = {'not': 'a plugin'}\n"
    result = await validator.validate("not_a_plugin", code)
    assert not result.ok
    assert result.stage == "protocol"


# ── Generator: prompt shape ──


def test_generation_prompt_includes_plugin_id() -> None:
    generator = PluginGenerator(llm_provider=None)
    request = PluginGenerationRequest(
        plugin_id="my_special_id_42",
        description="Do a thing",
    )
    prompt = generator.build_generation_prompt(request)
    assert "my_special_id_42" in prompt
    assert "Do a thing" in prompt
    assert "ToolPlugin" in prompt


# ── Generator: code extraction ──


def test_extract_code_strips_markdown_fences() -> None:
    generator = PluginGenerator(llm_provider=None)
    fenced = "```python\nplugin = None\n```"
    assert generator._extract_code(fenced) == "plugin = None"

    fenced_no_lang = "```\nplugin = None\n```"
    assert generator._extract_code(fenced_no_lang) == "plugin = None"

    unfenced = "plugin = None"
    assert generator._extract_code(unfenced) == "plugin = None"


# ── Generator: end-to-end orchestration ──


@pytest.mark.asyncio
async def test_generate_without_llm_returns_error() -> None:
    generator = PluginGenerator(llm_provider=None)
    request = PluginGenerationRequest(plugin_id="x", description="y")
    result = await generator.generate_and_validate(request)
    assert result["ok"] is False
    assert "LLM" in result["error"]


@pytest.mark.asyncio
async def test_generate_with_fake_llm_success() -> None:
    fake = _FakeLLM(response=VALID_ECHO_PLUGIN_CODE)
    generator = PluginGenerator(llm_provider=fake)
    request = PluginGenerationRequest(
        plugin_id="echo_test",
        description="An echo tool",
    )
    result = await generator.generate_and_validate(request)
    assert result["ok"] is True, result
    assert result["plugin_id"] == "echo_test"
    assert result["exposed_tools"] == ["echo"]
    assert result["requires_approval"] is True
    assert "code" in result
    # The LLM was actually invoked once
    assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_generate_with_fake_llm_invalid_code() -> None:
    """LLM returns dangerous code → validation surfaces the failure, no install path."""
    fake = _FakeLLM(response="import os\nos.system('rm -rf /')\nplugin = None\n")
    generator = PluginGenerator(llm_provider=fake)
    request = PluginGenerationRequest(plugin_id="malicious", description="bad thing")
    result = await generator.generate_and_validate(request)
    assert result["ok"] is False
    assert result["stage"] == "structure"
    # The code is surfaced for debugging
    assert "code" in result


@pytest.mark.asyncio
async def test_generate_extracts_fenced_llm_output() -> None:
    """LLM often wraps output in ```python fences; the generator strips them."""
    fenced = f"```python\n{VALID_ECHO_PLUGIN_CODE}\n```"
    fake = _FakeLLM(response=fenced)
    generator = PluginGenerator(llm_provider=fake)
    request = PluginGenerationRequest(plugin_id="echo_test", description="an echo")
    result = await generator.generate_and_validate(request)
    assert result["ok"] is True, result
