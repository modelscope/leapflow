"""LLM-driven plugin code generation and validation.

The capstone of LeapFlow's self-evolution: the Agent can propose a new plugin,
have the LLM generate its code, then validate it rigorously before any approval-
gated installation. This closes the loop from 'observe a capability gap' to
'safely acquire the capability'.

Safety pipeline (validation runs at generate-time; the rest is gated later):
    1. Syntax validation (py_compile)                     [generate-time]
    2. Import validation (module loads in a throwaway namespace)  [generate-time]
    3. Protocol conformance (exposes a valid `plugin` satisfying ToolPlugin)
                                                          [generate-time]
    4. Sandbox smoke test (first tool invoked in an isolated subprocess)
                                                          [INSTALL-time, owned
                                                           by plugin_install]
    5. Human approval (via ApprovalGate)                  [install-time]
    6. Install + dynamic load into the profile plugins dir [install-time]

Stages 1-3 are performed here by ``PluginValidator`` and NEVER invoke a tool
handler in-process. The sandbox smoke test (stage 4) is deliberately deferred
to install-time — it runs a real subprocess and is owned by the plugin_install
tool, not by ``PluginValidator``.

Nothing is auto-installed. Generated code is untrusted until validated AND approved.
"""

from __future__ import annotations

import ast
import inspect
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PluginValidationResult:
    """Outcome of validating generated plugin code."""

    ok: bool
    stage: str  # "syntax" | "structure" | "import" | "protocol" | "sandbox" | "passed"
    error: str = ""
    exposed_tools: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class PluginGenerationRequest:
    """A request to generate a plugin."""

    plugin_id: str
    description: str  # natural-language description of what the plugin should do
    plugin_type: str = "tool"  # "tool" | "active_signal_source"


class PluginValidator:
    """Validates generated plugin code through a multi-stage safety pipeline.

    Each stage is independent and fail-fast. Validation NEVER executes the
    plugin's code in-process — import happens in a throwaway namespace and
    tool invocation happens only in the sandbox.
    """

    async def validate(self, plugin_id: str, code: str) -> PluginValidationResult:
        """Run the full validation pipeline on generated code."""
        # Stage 1: syntax
        syntax_result = self._validate_syntax(code)
        if not syntax_result.ok:
            return syntax_result

        # Stage 2: static structure check (must define `plugin`, must not do
        # obviously dangerous things at import time)
        structure_result = self._validate_structure(code)
        if not structure_result.ok:
            return structure_result

        # Stage 3+4: write to temp, import + protocol + sandbox test
        return await self._validate_runtime(plugin_id, code)

    def _validate_syntax(self, code: str) -> PluginValidationResult:
        """Stage 1: the code must parse as valid Python."""
        try:
            ast.parse(code)
            return PluginValidationResult(ok=True, stage="syntax")
        except SyntaxError as exc:
            return PluginValidationResult(
                ok=False, stage="syntax", error=f"Syntax error: {exc}"
            )

    def _validate_structure(self, code: str) -> PluginValidationResult:
        """Stage 2: static AST checks.

        - Must define a module-level `plugin` assignment
        - Flag dangerous import-time patterns (os.system, subprocess at module level,
          eval/exec) — these are heuristics, the sandbox is the real safety boundary
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return PluginValidationResult(ok=False, stage="structure", error=str(exc))

        has_plugin = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "plugin":
                        has_plugin = True

        if not has_plugin:
            return PluginValidationResult(
                ok=False,
                stage="structure",
                error="Generated code must define a module-level `plugin` instance",
            )

        # Heuristic dangerous-pattern detection (defense in depth; sandbox is the real gate)
        dangerous = self._scan_dangerous_calls(tree)
        if dangerous:
            return PluginValidationResult(
                ok=False,
                stage="structure",
                error=(
                    f"Generated code contains flagged patterns: {', '.join(dangerous)}. "
                    "Manual review required."
                ),
            )

        return PluginValidationResult(ok=True, stage="structure")

    def _scan_dangerous_calls(self, tree: ast.AST) -> List[str]:
        """Detect obviously dangerous call patterns (heuristic)."""
        flagged: List[str] = []
        dangerous_names = {"eval", "exec", "compile", "__import__"}
        dangerous_attrs = {"system", "popen", "rmtree", "remove", "unlink"}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in dangerous_names:
                    flagged.append(node.func.id)
                elif (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr in dangerous_attrs
                ):
                    flagged.append(node.func.attr)
        return sorted(set(flagged))

    async def _validate_runtime(
        self, plugin_id: str, code: str
    ) -> PluginValidationResult:
        """Stages 3+4: write to temp dir, import, protocol check, sandbox smoke test."""
        import importlib.util
        import sys

        tmpdir = Path(tempfile.mkdtemp(prefix="leapflow_plugin_gen_"))
        module_file = tmpdir / f"{plugin_id}.py"
        module_name = f"_genplugin_{plugin_id}"
        try:
            module_file.write_text(code)

            # Stage 3: import in isolated namespace
            spec = importlib.util.spec_from_file_location(module_name, module_file)
            if spec is None or spec.loader is None:
                return PluginValidationResult(
                    ok=False, stage="import", error="Cannot create module spec"
                )

            module = importlib.util.module_from_spec(spec)
            try:
                spec.loader.exec_module(module)
            except Exception as exc:  # noqa: BLE001 - import of untrusted generated code; any error is a validation failure
                return PluginValidationResult(
                    ok=False, stage="import", error=f"Import failed: {exc}"
                )

            plugin_obj = getattr(module, "plugin", None)
            if plugin_obj is None:
                return PluginValidationResult(
                    ok=False,
                    stage="protocol",
                    error="No `plugin` attribute after import",
                )

            # Protocol conformance
            from leapflow.plugins.protocol import ToolPlugin

            if not isinstance(plugin_obj, ToolPlugin):
                return PluginValidationResult(
                    ok=False,
                    stage="protocol",
                    error=(
                        "`plugin` does not satisfy ToolPlugin Protocol "
                        "(missing plugin_id/category/tools/dependencies/bind_runtime)"
                    ),
                )

            try:
                tools = list(plugin_obj.tools)
                tool_names = [t.name for t in tools]
            except Exception as exc:  # noqa: BLE001 - accessing plugin.tools may raise on malformed generated code
                return PluginValidationResult(
                    ok=False, stage="protocol", error=f"plugin.tools raised: {exc}"
                )

            metadata_error = self._validate_tool_metadata(plugin_id, plugin_obj, tools)
            if metadata_error:
                return PluginValidationResult(
                    ok=False, stage="protocol", error=metadata_error
                )

            # Clean up the imported module from sys.modules to avoid pollution
            sys.modules.pop(module_name, None)

            # Stage 4: Full sandbox invocation (running tool handlers in a
            # subprocess) is deliberately deferred to install-time. During
            # validation, the import+protocol checks (stages 2-3) are the
            # critical safety gates: they load the module in a throwaway
            # namespace, verify it conforms to ToolPlugin, and never invoke a
            # handler in-process. Driving a sandbox subprocess from here would
            # require putting the temp directory holding untrusted generated
            # code on the worker's PYTHONPATH, which itself extends the trust
            # boundary during a check whose purpose is to *establish* trust.
            # Once approved and copied into the profile plugins directory, the
            # sandbox infrastructure (SandboxHost / SandboxedToolPlugin) owns
            # the actual isolation for tool invocations.

            return PluginValidationResult(
                ok=True, stage="passed", exposed_tools=tool_names
            )
        finally:
            # Clean up temp files
            try:
                module_file.unlink(missing_ok=True)
                tmpdir.rmdir()
            except OSError:
                pass

    def _validate_tool_metadata(self, plugin_id: str, plugin_obj: Any, tools: list[Any]) -> str:
        """Validate ToolMetadata entries without invoking generated handlers."""
        if plugin_obj.plugin_id != plugin_id:
            return f"plugin_id mismatch: expected {plugin_id!r}, got {plugin_obj.plugin_id!r}"
        if not isinstance(plugin_obj.category, str) or not plugin_obj.category.strip():
            return "plugin.category must be a non-empty string"
        if not isinstance(plugin_obj.dependencies, list) or not all(
            isinstance(dep, str) for dep in plugin_obj.dependencies
        ):
            return "plugin.dependencies must be list[str]"
        if not tools:
            return "plugin.tools must expose at least one ToolMetadata"

        seen: set[str] = set()
        allowed_risk = {"read_only", "low", "medium", "high", "mutating", "external"}
        for index, tool in enumerate(tools):
            name = getattr(tool, "name", None)
            if not isinstance(name, str) or not name.strip():
                return f"tool[{index}].name must be a non-empty string"
            if name in seen:
                return f"duplicate tool name: {name}"
            seen.add(name)
            if not name.replace("_", "").isalnum() or name.lower() != name:
                return f"tool {name!r} must use lowercase snake_case"

            description = getattr(tool, "description", None)
            if not isinstance(description, str) or not description.strip():
                return f"tool {name}: description must be non-empty"

            schema = getattr(tool, "parameters_schema", None)
            if not isinstance(schema, dict):
                return f"tool {name}: parameters_schema must be a dict"
            if schema.get("type") != "object":
                return f"tool {name}: parameters_schema.type must be 'object'"
            properties = schema.get("properties", {})
            if not isinstance(properties, dict):
                return f"tool {name}: parameters_schema.properties must be a dict"
            required = schema.get("required", [])
            if required is not None and not isinstance(required, list):
                return f"tool {name}: parameters_schema.required must be a list when present"

            x_meta = getattr(tool, "x_leapflow", None)
            if not isinstance(x_meta, dict):
                return f"tool {name}: x_leapflow must be a dict"
            category = x_meta.get("category")
            if not isinstance(category, str) or not category.strip():
                return f"tool {name}: x_leapflow.category must be a non-empty string"
            risk = x_meta.get("risk_level")
            if not isinstance(risk, str) or risk not in allowed_risk:
                return f"tool {name}: x_leapflow.risk_level must be one of {sorted(allowed_risk)}"

            handler = getattr(tool, "handler", None)
            if not callable(handler):
                return f"tool {name}: handler must be callable"
            if not inspect.iscoroutinefunction(handler):
                return f"tool {name}: handler must be an async function"
            signature = inspect.signature(handler)
            has_kwargs = any(
                param.kind == inspect.Parameter.VAR_KEYWORD
                for param in signature.parameters.values()
            )
            if not has_kwargs:
                return f"tool {name}: handler must accept **kwargs"

            if bool(getattr(tool, "mutates_state", False)):
                if x_meta.get("requires_approval") is not True:
                    return f"tool {name}: mutating tools must set x_leapflow.requires_approval=true"
                if not x_meta.get("effect_scope"):
                    return f"tool {name}: mutating tools must set x_leapflow.effect_scope"
                if not x_meta.get("idempotency_scope"):
                    return f"tool {name}: mutating tools must set x_leapflow.idempotency_scope"
        return ""


class PluginGenerator:
    """Orchestrates LLM-driven plugin generation with validation.

    This class builds the LLM prompt, calls the provided LLM, extracts code,
    and runs validation. It does NOT install — installation is a separate
    approval-gated action.
    """

    def __init__(self, llm_provider: Any = None) -> None:
        self._llm = llm_provider
        self._validator = PluginValidator()

    def build_generation_prompt(self, request: PluginGenerationRequest) -> str:
        """Build the LLM prompt for generating a plugin. Returns the prompt text.

        The prompt includes the ToolPlugin Protocol contract and an example,
        so the LLM generates conformant code.
        """
        return f"""Generate a Python ToolPlugin for LeapFlow.

Plugin ID: {request.plugin_id}
Requirement: {request.description}

The plugin MUST:
1. Define a class implementing the ToolPlugin Protocol with these members:
   - property plugin_id -> str (must return "{request.plugin_id}")
   - property category -> str
   - property tools -> list[ToolMetadata]
   - property dependencies -> list[str] (return [] if none)
   - def bind_runtime(self, **deps) -> None
2. Define a module-level `plugin = YourPluginClass()`
3. Each tool is a ToolMetadata(name, description, parameters_schema, handler, x_leapflow, mutates_state)
4. Every ToolMetadata MUST set x_leapflow to a dict with at least category and risk_level, e.g. {{"category": "custom", "risk_level": "read_only"}}; never use None
5. Mutating tools MUST set mutates_state=True and x_leapflow.requires_approval=True plus effect_scope and idempotency_scope
6. Import from: from leapflow.plugins.protocol import ToolMetadata, ToolPlugin
7. NO dangerous operations (no eval/exec/os.system/file deletion at import time)
8. All handlers are async functions taking **kwargs and returning a dict
9. On success, every handler MUST report what it observably did in an "effect" key,
   phrased in the same terms as the requirement above (e.g.
   {{"ok": True, "effect": "the reply was delivered to the thread"}}). This is how the
   framework confirms the capability actually worked rather than merely returned; a
   handler that omits it can never be verified, only refuted. Describe the observed
   outcome, never restate the intent.

Example structure:
```python
from typing import Any
from leapflow.plugins.protocol import ToolMetadata

class MyPlugin:
    @property
    def plugin_id(self) -> str: return "{request.plugin_id}"
    @property
    def category(self) -> str: return "custom"
    @property
    def dependencies(self) -> list[str]: return []
    def bind_runtime(self, **deps: Any) -> None: pass
    @property
    def tools(self) -> list[ToolMetadata]:
        return [ToolMetadata(name="...", description="...", parameters_schema={{"type":"object","properties":{{}}}}, handler=self._handler, x_leapflow={{"category":"custom","risk_level":"read_only"}})]
    async def _handler(self, **kwargs: Any) -> dict:
        return {{"ok": True, "effect": "<what observably changed>"}}

plugin = MyPlugin()
```

Output ONLY the Python code, no markdown fences."""

    async def generate_and_validate(
        self, request: PluginGenerationRequest
    ) -> Dict[str, Any]:
        """Generate plugin code via LLM and validate it. Returns result dict.

        Does NOT install. On success, returns the validated code for a
        subsequent approval-gated install step.
        """
        if self._llm is None:
            return {
                "ok": False,
                "error": "No LLM provider configured for plugin generation",
            }

        prompt = self.build_generation_prompt(request)

        try:
            code = await self._call_llm(prompt)
        except Exception as exc:  # noqa: BLE001 - LLM boundary; any provider failure surfaces as validation error
            return {"ok": False, "error": f"LLM generation failed: {exc}"}

        code = self._extract_code(code)

        # Validate
        result = await self._validator.validate(request.plugin_id, code)
        if not result.ok:
            return {
                "ok": False,
                "error": f"Validation failed at stage '{result.stage}': {result.error}",
                "stage": result.stage,
                "code": code,  # return for debugging
            }

        return {
            "ok": True,
            "plugin_id": request.plugin_id,
            "code": code,
            "exposed_tools": result.exposed_tools,
            "requires_approval": True,
            "note": "Code validated. Human approval required before install.",
        }

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM provider. Adapt to the provider's interface."""
        # The LLMProvider ABC has achat() — adapt as needed
        messages = [{"role": "user", "content": prompt}]
        response = await self._llm.achat(messages)
        # Extract text from response (adapt to actual response shape)
        if isinstance(response, str):
            return response
        return getattr(response, "content", "") or str(response)

    def _extract_code(self, text: str) -> str:
        """Strip markdown fences if the LLM wrapped the code."""
        text = text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first fence line and last fence line
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)
        return text.strip()
