# Copyright (c) Alibaba, Inc. and its affiliates.
"""Tests for learning.codegen — security validation, AST checks, and parsing."""

from __future__ import annotations

import textwrap

from leapflow.learning.codegen import (
    GeneratedSkill,
    LLMSkillCodeGenerator,
    SkillCodeGenerator,
    TemplateSkillCodeGenerator,
    ValidationResult,
    _FORBIDDEN_CALLS,
    _FORBIDDEN_MODULES,
)


# ── Helpers ──


def _make_validator() -> LLMSkillCodeGenerator:
    """Build an LLM generator with a stub LLM (only validate_code is used)."""
    return LLMSkillCodeGenerator(llm=None, sandbox_enabled=True)


# ── 1. Forbidden module imports ──


class TestForbiddenImports:
    def test_import_os_rejected(self) -> None:
        code = "import os\nasync def f(execution, perception): pass"
        result = _make_validator().validate_code(code)
        assert not result.passed
        assert any("os" in e for e in result.errors)

    def test_import_subprocess_from_rejected(self) -> None:
        code = "from subprocess import run\nasync def f(execution, perception): pass"
        result = _make_validator().validate_code(code)
        assert not result.passed
        assert any("subprocess" in e for e in result.errors)

    def test_all_forbidden_modules_blocked(self) -> None:
        """Every module in _FORBIDDEN_MODULES triggers an error."""
        gen = _make_validator()
        for mod in sorted(_FORBIDDEN_MODULES):
            code = f"import {mod}\nasync def f(execution, perception): pass"
            result = gen.validate_code(code)
            assert not result.passed, f"Module '{mod}' should be forbidden"

    def test_nested_forbidden_import_rejected(self) -> None:
        """e.g. 'import os.path' should be caught by root-module check."""
        code = "import os.path\nasync def f(execution, perception): pass"
        result = _make_validator().validate_code(code)
        assert not result.passed
        assert any("os" in e for e in result.errors)


# ── 2. Forbidden calls / attributes ──


class TestForbiddenCalls:
    def test_eval_rejected(self) -> None:
        code = textwrap.dedent("""\
            async def f(execution, perception):
                return eval("1+1")
        """)
        result = _make_validator().validate_code(code)
        assert not result.passed
        assert any("eval" in e for e in result.errors)

    def test_exec_rejected(self) -> None:
        code = textwrap.dedent("""\
            async def f(execution, perception):
                exec("print('hi')")
        """)
        result = _make_validator().validate_code(code)
        assert not result.passed
        assert any("exec" in e for e in result.errors)

    def test_open_rejected(self) -> None:
        code = textwrap.dedent("""\
            async def f(execution, perception):
                f = open("/etc/passwd")
        """)
        result = _make_validator().validate_code(code)
        assert not result.passed
        assert any("open" in e for e in result.errors)

    def test_os_system_attribute_call_rejected(self) -> None:
        code = textwrap.dedent("""\
            import os
            async def f(execution, perception):
                os.system("rm -rf /")
        """)
        result = _make_validator().validate_code(code)
        assert not result.passed
        # Should have both import AND call errors
        assert len(result.errors) >= 2

    def test_all_forbidden_direct_calls_blocked(self) -> None:
        gen = _make_validator()
        for call in sorted(_FORBIDDEN_CALLS):
            code = f"async def f(execution, perception):\n    {call}('x')"
            result = gen.validate_code(code)
            assert not result.passed, f"Call '{call}()' should be forbidden"


# ── 3. Permitted minimal plugin / code sample ──


class TestPermittedCode:
    def test_clean_async_function_passes(self) -> None:
        code = textwrap.dedent("""\
            async def organize_files(execution, perception, **params):
                \"\"\"Organize files by extension.\"\"\"
                result = await execution.exec_shell("ls")
                return {"ok": True, "result": result}
        """)
        result = _make_validator().validate_code(code)
        assert result.passed
        assert result.valid
        assert not result.errors

    def test_safe_stdlib_import_passes(self) -> None:
        """json, re, typing are not forbidden and should pass."""
        code = textwrap.dedent("""\
            import json
            import re
            from typing import Dict
            async def f(execution, perception):
                return json.dumps({"ok": True})
        """)
        result = _make_validator().validate_code(code)
        assert result.passed


# ── 4. Syntax errors ──


class TestSyntaxErrors:
    def test_syntax_error_returns_invalid(self) -> None:
        code = "def broken(\nasync def f(execution, perception): pass"
        result = _make_validator().validate_code(code)
        assert not result.passed
        assert any("SyntaxError" in e for e in result.errors)


# ── 5. Protocol / structure validation ──


class TestProtocolAndStructure:
    def test_skill_code_generator_protocol_satisfied(self) -> None:
        """LLMSkillCodeGenerator satisfies the SkillCodeGenerator Protocol."""
        gen = _make_validator()
        assert isinstance(gen, SkillCodeGenerator)

    def test_template_generator_satisfies_protocol(self) -> None:
        tgen = TemplateSkillCodeGenerator()
        assert isinstance(tgen, SkillCodeGenerator)

    def test_generated_skill_is_valid_when_complete(self) -> None:
        skill = GeneratedSkill(
            function_name="my_skill",
            code="async def my_skill(): ...",
            parameters=[],
            imports=[],
            description="test",
            confidence=0.8,
        )
        assert skill.is_valid

    def test_generated_skill_invalid_when_zero_confidence(self) -> None:
        skill = GeneratedSkill(
            function_name="my_skill",
            code="async def my_skill(): ...",
            parameters=[],
            imports=[],
            description="test",
            confidence=0.0,
        )
        assert not skill.is_valid

    def test_validation_result_passed_iff_valid_and_no_errors(self) -> None:
        assert ValidationResult(valid=True, errors=[], warnings=["w"]).passed
        assert not ValidationResult(valid=True, errors=["e"]).passed
        assert not ValidationResult(valid=False, errors=[]).passed


# ── 6. Error result includes actionable category/reason ──


class TestErrorReasonActionable:
    def test_forbidden_import_error_names_module(self) -> None:
        code = "import requests\nasync def f(execution, perception): pass"
        result = _make_validator().validate_code(code)
        assert not result.passed
        # Error message must name the module so the user can act on it
        assert any("requests" in e for e in result.errors)

    def test_forbidden_call_error_names_function(self) -> None:
        code = "async def f(execution, perception):\n    compile('x', 'f', 'exec')"
        result = _make_validator().validate_code(code)
        assert not result.passed
        assert any("compile" in e for e in result.errors)


# ── 7. Function signature warnings ──


class TestFunctionSignature:
    def test_missing_async_def_produces_warning(self) -> None:
        """A synchronous function is warned, not hard-rejected."""
        code = "def f(execution, perception): pass"
        result = _make_validator().validate_code(code)
        # No errors (sync is a warning), but warnings present
        assert result.valid
        assert len(result.warnings) > 0

    def test_too_few_args_produces_warning(self) -> None:
        code = "async def f(x): pass"
        result = _make_validator().validate_code(code)
        assert result.valid  # warnings, not errors
        assert any("positional" in w or "2" in w for w in result.warnings)


# ── 8. parse_response coverage ──


class TestParseResponse:
    def test_parse_json_code_block(self) -> None:
        gen = _make_validator()
        text = '```json\n{"function_name": "foo", "code": "async def foo(): pass"}\n```'
        result = gen._parse_response(text)
        assert result is not None
        assert result["function_name"] == "foo"

    def test_parse_python_code_block_fallback(self) -> None:
        gen = _make_validator()
        text = "```python\nasync def bar(execution, perception): pass\n```"
        result = gen._parse_response(text)
        assert result is not None
        assert result["function_name"] == "bar"

    def test_parse_empty_returns_none(self) -> None:
        gen = _make_validator()
        assert gen._parse_response("") is None

    def test_parse_garbage_returns_none(self) -> None:
        gen = _make_validator()
        assert gen._parse_response("just some random text without code") is None
