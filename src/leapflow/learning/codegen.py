"""LLM-driven skill code generation from distillation candidates.

Transforms DistillationCandidate (descriptive JSON) into executable Python async functions
that use the VSI ports (ExecutionPort, PerceptionPort) for system interaction.

Two strategies:
- LLMSkillCodeGenerator: Full LLM generation for complex/novel patterns
- TemplateSkillCodeGenerator: Template-based for known patterns (zero LLM cost)
"""

from __future__ import annotations

import ast
import json
import logging
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from ..skills.registry import SkillParameter

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════

_DEFAULT_SAFETY_CONSTRAINTS: List[str] = [
    "No os.system or subprocess calls",
    "No network requests without explicit user permission",
    "No file deletion without confirmation",
    "Only use provided ExecutionPort and PerceptionPort interfaces",
]

_FORBIDDEN_MODULES = frozenset({
    "os", "subprocess", "shutil", "sys", "ctypes", "importlib",
    "socket", "http", "urllib", "requests", "multiprocessing",
    "signal", "pty", "tempfile",
})

_FORBIDDEN_CALLS = frozenset({
    "eval", "exec", "compile", "__import__", "globals", "locals",
    "getattr", "setattr", "delattr", "open",
})

_FORBIDDEN_ATTR_CALLS = frozenset({
    "os.system", "os.popen", "os.exec", "os.spawn",
    "subprocess.run", "subprocess.call", "subprocess.Popen",
    "shutil.rmtree", "shutil.move",
})


@dataclass(frozen=True)
class CodeGenContext:
    """Context provided to the code generator for informed generation."""

    available_ports: List[str] = field(default_factory=list)
    available_methods: List[str] = field(default_factory=list)
    existing_skills: List[str] = field(default_factory=list)
    safety_constraints: List[str] = field(default_factory=lambda: list(_DEFAULT_SAFETY_CONSTRAINTS))
    episode: Any = None


@dataclass
class GeneratedSkill:
    """Output of the code generation process."""

    function_name: str
    code: str
    parameters: List[SkillParameter]
    imports: List[str]
    description: str
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    triggers: List[str] = field(default_factory=list)
    test_cases: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    generation_method: str = "llm"

    @property
    def is_valid(self) -> bool:
        """Basic validity check."""
        return bool(self.function_name and self.code and self.confidence > 0)


@dataclass
class ValidationResult:
    """Result of code validation."""

    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.valid and not self.errors


# ═══════════════════════════════════════════════════════════════════════
# Protocol
# ═══════════════════════════════════════════════════════════════════════


@runtime_checkable
class SkillCodeGenerator(Protocol):
    """Interface for skill code generation (DIP)."""

    async def generate(self, candidate: Any, context: CodeGenContext) -> Optional[GeneratedSkill]:
        ...


# ═══════════════════════════════════════════════════════════════════════
# VSI Interface Documentation (for prompt injection)
# ═══════════════════════════════════════════════════════════════════════

_VSI_INTERFACE_DOCS = textwrap.dedent("""\
    ## Available Interfaces

    ### PerceptionPort (read-only system observation)
    - `await perception.subscribe_fs(paths: List[str]) -> str`
        Subscribe to filesystem changes at given paths. Returns subscription ID.
    - `await perception.read_ui_tree(app_id: Optional[str] = None) -> UINode`
        Read the accessibility tree of the focused app (or specified app).
    - `await perception.get_clipboard() -> Dict[str, Any]`
        Read current clipboard content. Returns {"text": ..., "type": ...}.
    - `async for event in perception.stream_events() -> AsyncIterator[SystemEvent]`
        Stream real-time system events.

    ### ExecutionPort (write actions)
    - `await execution.perform_file_op(op: str, params: Dict[str, Any]) -> Dict[str, Any]`
        File operations. op: "copy"|"move"|"create"|"delete"|"rename"
        params: {"source": path, "destination": path} or {"path": path, "content": str}
    - `await execution.perform_ui_action(node_id: str, action: str, params: Optional[Dict] = None) -> Dict`
        Interact with UI elements. action: "click"|"type"|"select"|"scroll"
    - `await execution.launch_app(app_id: str) -> Dict[str, Any]`
        Launch an application by bundle ID.
    - `await execution.run_intent(intent_name: str, params: Dict[str, Any]) -> Dict[str, Any]`
        Execute a system intent (share, open-with, etc.).
    - `await execution.exec_shell(command: str) -> Dict[str, Any]`
        Execute a shell command. Returns {"stdout": ..., "stderr": ..., "exit_code": int}.
""")


# ═══════════════════════════════════════════════════════════════════════
# LLM Code Generator
# ═══════════════════════════════════════════════════════════════════════


class LLMSkillCodeGenerator:
    """LLM-driven code generation with AST validation and safety checks."""

    def __init__(self, llm: Any, *, sandbox_enabled: bool = True) -> None:
        """
        Args:
            llm: LLMProvider instance (must support achat()).
            sandbox_enabled: Whether to perform AST safety validation.
        """
        self._llm = llm
        self._sandbox_enabled = sandbox_enabled

    async def generate(self, candidate: Any, context: CodeGenContext) -> Optional[GeneratedSkill]:
        """Generate executable skill code from a DistillationCandidate.

        Pipeline:
        1. Build prompt with VSI interface docs + candidate info
        2. Call LLM to generate code
        3. Parse response (extract code block + metadata)
        4. AST validation (syntax + safety)
        5. Return GeneratedSkill or None on failure
        """
        from ..llm.message_builder import build_system_message, build_user_message_text

        prompt = self._build_prompt(candidate, context)

        try:
            resp = await self._llm.achat(
                [
                    build_system_message(
                        "You are an expert Python code generator for desktop automation skills. "
                        "Generate clean, type-annotated async functions that use the provided "
                        "ExecutionPort and PerceptionPort interfaces. Output ONLY valid Python."
                    ),
                    build_user_message_text(prompt),
                ],
                stream=True,
                enable_thinking=False,
            )
        except Exception:
            logger.warning("LLM code generation call failed", exc_info=True)
            return None

        parsed = self._parse_response(resp.content or "")
        if parsed is None:
            logger.debug("Failed to parse LLM response for codegen")
            return None

        code = parsed["code"]

        # AST validation
        if self._sandbox_enabled:
            validation = self.validate_code(code)
            if not validation.passed:
                logger.warning(
                    "Generated code failed validation: %s", validation.errors
                )
                return None

        # Build parameters from parsed metadata
        parameters = [
            SkillParameter(
                name=p.get("name", ""),
                type=p.get("type", "str"),
                required=p.get("required", False),
                default=p.get("default"),
                description=p.get("description", ""),
            )
            for p in parsed.get("parameters", [])
        ]

        return GeneratedSkill(
            function_name=parsed.get("function_name", "generated_skill"),
            code=code,
            parameters=parameters,
            imports=parsed.get("imports", []),
            description=parsed.get("description", candidate.title),
            preconditions=parsed.get("preconditions", list(candidate.pre_conditions)),
            postconditions=parsed.get("postconditions", list(candidate.post_conditions)),
            triggers=parsed.get("triggers", list(candidate.trigger_phrases)),
            test_cases=parsed.get("test_cases", []),
            confidence=parsed.get("confidence", candidate.confidence * 0.8),
            generation_method="llm",
        )

    def _build_prompt(self, candidate: Any, context: CodeGenContext) -> str:
        """Construct the code generation prompt."""
        steps_text = "\n".join(f"  {i + 1}. {s}" for i, s in enumerate(candidate.steps))
        params_text = "\n".join(
            f"  - {p.get('name', '?')}: {p.get('description', '')}"
            for p in candidate.parameters
        ) or "  (none)"

        existing_skills_text = ", ".join(context.existing_skills[:20]) or "(none)"
        constraints_text = "\n".join(f"  - {c}" for c in context.safety_constraints)

        episode_section = ""
        if context.episode and hasattr(context.episode, "semantic_actions"):
            actions = context.episode.semantic_actions or []
            actions_text = "\n".join(
                f"  {i + 1}. {a.action_name}: {a.description} "
                f"(params: {dict(list(a.parameters.items())[:5])})"
                for i, a in enumerate(actions[:10])
            )
            episode_section = f"""
            ## Demonstrated Actions (from user recording)
            {actions_text}

            CRITICAL: Generate code that implements the USER'S DEMONSTRATED WORKFLOW
            above, not a generic version of the skill name. The demonstrated actions
            show exactly what the user did — your code should replicate that pattern
            in a parameterized, reusable way.
            """

        return textwrap.dedent(f"""\
            Generate an async Python function that implements the following skill.

            ## Skill Description
            Title: {candidate.title}
            Steps:
            {steps_text}
            Parameters:
            {params_text}
            Preconditions: {candidate.pre_conditions}
            {episode_section}
            {_VSI_INTERFACE_DOCS}

            ## Existing Skills (available for composition)
            {existing_skills_text}

            ## Safety Constraints
            {constraints_text}

            ## Output Format
            Return a JSON object with:
            ```json
            {{
              "function_name": "snake_case_name",
              "code": "async def ...(execution, perception, **params):\\n    ...",
              "parameters": [{{"name": "...", "type": "str|int|float|bool|path|list|dict", "required": true/false, "default": null, "description": "..."}}],
              "imports": ["from typing import Dict, Any"],
              "description": "One-line description",
              "preconditions": ["..."],
              "postconditions": ["..."],
              "triggers": ["natural language trigger phrase"],
              "test_cases": [{{"input": {{}}, "expected_behavior": "..."}}],
              "confidence": 0.0-1.0
            }}
            ```

            IMPORTANT:
            - The function signature MUST be: `async def name(execution, perception, **params)`
            - `execution` is an ExecutionPort instance; `perception` is a PerceptionPort instance
            - Extract named parameters from `params` dict inside the function
            - Use ONLY the ExecutionPort/PerceptionPort methods listed above
            - NO os, subprocess, shutil, or direct filesystem access
            - Return a dict with {{"ok": bool, "result": ...}} or raise on failure
            - Include type hints and a docstring

            NAMING:
            - function_name MUST be derived from the skill title "{candidate.title}"
              (snake_case), NOT from internal action names like batch_rename or file.modify

            PARAMETER DESIGN:
            - Mark a parameter as "required" ONLY if the user MUST provide it for execution
              (e.g., source_dir for "organize files in X")
            - Paths or patterns with sensible defaults should be optional with defaults
            - Prefer fewer required params (0-1) for higher execution success rate
        """)

    def _parse_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Extract code block and metadata from LLM response."""
        # Try to extract JSON block
        json_match = re.search(r"```json\s*\n(.*?)```", response_text, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                if "code" in data and "function_name" in data:
                    return data
            except json.JSONDecodeError:
                pass

        # Try raw JSON (the whole response might be JSON)
        start = response_text.find("{")
        end = response_text.rfind("}")
        if start != -1 and end > start:
            try:
                data = json.loads(response_text[start:end + 1])
                if "code" in data:
                    # Infer function name from code if not provided
                    if "function_name" not in data:
                        fn_match = re.search(r"async\s+def\s+(\w+)", data["code"])
                        data["function_name"] = fn_match.group(1) if fn_match else "generated_skill"
                    return data
            except json.JSONDecodeError:
                pass

        # Fallback: extract python code block directly
        code_match = re.search(r"```python\s*\n(.*?)```", response_text, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
            fn_match = re.search(r"async\s+def\s+(\w+)", code)
            return {
                "function_name": fn_match.group(1) if fn_match else "generated_skill",
                "code": code,
                "parameters": [],
                "imports": [],
                "description": "",
                "confidence": 0.5,
            }

        return None

    def validate_code(self, code: str) -> ValidationResult:
        """Static analysis of generated code via AST.

        Checks:
        1. Valid Python syntax
        2. No forbidden imports
        3. No forbidden function calls
        4. Async function with proper signature
        5. Uses only whitelisted APIs
        """
        errors: List[str] = []
        warnings: List[str] = []

        # 1. Syntax check
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return ValidationResult(valid=False, errors=[f"SyntaxError: {e}"])

        # 2. Forbidden imports
        errors.extend(self._check_forbidden_imports(tree))

        # 3. Forbidden calls
        errors.extend(self._check_forbidden_calls(tree))

        # 4. Function signature
        sig_issues = self._check_function_signature(tree)
        # Signature issues are warnings, not errors (lenient)
        warnings.extend(sig_issues)

        return ValidationResult(valid=not errors, errors=errors, warnings=warnings)

    def _check_forbidden_imports(self, tree: ast.AST) -> List[str]:
        """Check for forbidden import statements."""
        errors: List[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_root = alias.name.split(".")[0]
                    if module_root in _FORBIDDEN_MODULES:
                        errors.append(f"Forbidden import: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_root = node.module.split(".")[0]
                    if module_root in _FORBIDDEN_MODULES:
                        errors.append(f"Forbidden import from: {node.module}")
        return errors

    def _check_forbidden_calls(self, tree: ast.AST) -> List[str]:
        """Walk AST to find dangerous function calls."""
        errors: List[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue

            # Direct calls: eval(...), exec(...)
            if isinstance(node.func, ast.Name):
                if node.func.id in _FORBIDDEN_CALLS:
                    errors.append(f"Forbidden call: {node.func.id}()")

            # Attribute calls: os.system(...), subprocess.run(...)
            elif isinstance(node.func, ast.Attribute):
                full_name = _resolve_attr_name(node.func)
                if full_name in _FORBIDDEN_ATTR_CALLS:
                    errors.append(f"Forbidden call: {full_name}()")

        return errors

    def _check_function_signature(self, tree: ast.AST) -> List[str]:
        """Verify the generated function is async and has proper signature."""
        issues: List[str] = []
        async_defs = [
            node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef)
        ]

        if not async_defs:
            issues.append("No async function definition found")
            return issues

        func = async_defs[0]
        args = func.args

        # Expect at least (execution, perception) parameters
        arg_names = [a.arg for a in args.args]
        if len(arg_names) < 2:
            issues.append(
                f"Expected at least 2 positional args (execution, perception), "
                f"got {len(arg_names)}: {arg_names}"
            )

        return issues


# ═══════════════════════════════════════════════════════════════════════
# Template Code Generator
# ═══════════════════════════════════════════════════════════════════════

# Pre-written templates for common automation patterns
_BUILTIN_TEMPLATES: Dict[str, str] = {
    "cross_app_transfer": textwrap.dedent('''\
        async def cross_app_transfer(execution, perception, **params):
            """Transfer data between two applications via clipboard."""
            source_app: str = params.get("source_app", "")
            target_app: str = params.get("target_app", "")
            element_id: str = params.get("element_id", "")

            # 1. Focus source app and copy
            await execution.launch_app(source_app)
            if element_id:
                await execution.perform_ui_action(element_id, "click")
            await execution.perform_ui_action("", "shortcut", {"keys": "cmd+c"})

            # 2. Get clipboard content
            clipboard = await perception.get_clipboard()

            # 3. Switch to target app and paste
            await execution.launch_app(target_app)
            await execution.perform_ui_action("", "shortcut", {"keys": "cmd+v"})

            return {"ok": True, "result": {"transferred": clipboard.get("text", "")}}
    '''),

    "file_organize": textwrap.dedent('''\
        async def file_organize(execution, perception, **params):
            """Organize files in a directory by extension or pattern."""
            source_dir: str = params.get("source_dir", "")
            rules: dict = params.get("rules", {})

            # 1. List files in source directory
            listing = await execution.exec_shell(f"ls -1 {source_dir}")
            files = listing.get("stdout", "").strip().split("\\n")

            moved = []
            for filename in files:
                if not filename:
                    continue
                ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
                dest_subdir = rules.get(ext)
                if dest_subdir:
                    source_path = f"{source_dir}/{filename}"
                    dest_path = f"{source_dir}/{dest_subdir}/{filename}"
                    await execution.perform_file_op("move", {
                        "source": source_path,
                        "destination": dest_path,
                    })
                    moved.append(filename)

            return {"ok": True, "result": {"moved_count": len(moved), "files": moved}}
    '''),

    "web_download": textwrap.dedent('''\
        async def web_download(execution, perception, **params):
            """Download a file via browser automation."""
            url: str = params.get("url", "")
            save_path: str = params.get("save_path", "~/Downloads")

            # 1. Open URL in default browser
            await execution.run_intent("open_url", {"url": url})

            # 2. Wait for download dialog / auto-download
            # Observe filesystem for new file
            sub_id = await perception.subscribe_fs([save_path])

            return {"ok": True, "result": {"subscription_id": sub_id, "target_path": save_path}}
    '''),

    "batch_rename": textwrap.dedent('''\
        async def batch_rename(execution, perception, **params):
            """Batch rename files using a pattern."""
            directory: str = params.get("directory", "")
            pattern: str = params.get("pattern", "")
            replacement: str = params.get("replacement", "")

            import re as _re

            listing = await execution.exec_shell(f"ls -1 {directory}")
            files = listing.get("stdout", "").strip().split("\\n")

            renamed = []
            for filename in files:
                if not filename:
                    continue
                new_name = _re.sub(pattern, replacement, filename)
                if new_name != filename:
                    await execution.perform_file_op("rename", {
                        "source": f"{directory}/{filename}",
                        "destination": f"{directory}/{new_name}",
                    })
                    renamed.append({"from": filename, "to": new_name})

            return {"ok": True, "result": {"renamed_count": len(renamed), "files": renamed}}
    '''),

    "clipboard_transform": textwrap.dedent('''\
        async def clipboard_transform(execution, perception, **params):
            """Read clipboard, apply a transformation, and write back."""
            transform: str = params.get("transform", "upper")

            # 1. Read current clipboard
            clipboard = await perception.get_clipboard()
            text = clipboard.get("text", "")

            if not text:
                return {"ok": False, "result": {"error": "Clipboard is empty"}}

            # 2. Apply transformation
            if transform == "upper":
                result = text.upper()
            elif transform == "lower":
                result = text.lower()
            elif transform == "strip":
                result = text.strip()
            elif transform == "title":
                result = text.title()
            else:
                result = text

            # 3. Write back via shell pbcopy
            await execution.exec_shell(f"echo {repr(result)} | pbcopy")

            return {"ok": True, "result": {"original": text[:100], "transformed": result[:100]}}
    '''),
}

# Mapping from pattern names to their parameter declarations
_TEMPLATE_PARAMETERS: Dict[str, List[SkillParameter]] = {
    "cross_app_transfer": [
        SkillParameter(name="source_app", type="str", required=True, description="Source app bundle ID"),
        SkillParameter(name="target_app", type="str", required=True, description="Target app bundle ID"),
        SkillParameter(name="element_id", type="str", required=False, default="", description="UI element to select before copy"),
    ],
    "file_organize": [
        SkillParameter(name="source_dir", type="path", required=True, description="Directory to organize"),
        SkillParameter(name="rules", type="dict", required=False, default={}, description="Mapping of extension to subdirectory"),
    ],
    "web_download": [
        SkillParameter(name="url", type="str", required=True, description="URL to download"),
        SkillParameter(name="save_path", type="path", required=False, default="~/Downloads", description="Download destination"),
    ],
    "batch_rename": [
        SkillParameter(name="directory", type="path", required=True, description="Directory containing files"),
        SkillParameter(name="pattern", type="str", required=False, default=".*", description="Regex pattern to match"),
        SkillParameter(name="replacement", type="str", required=False, default="", description="Replacement string"),
    ],
    "clipboard_transform": [
        SkillParameter(name="transform", type="str", required=False, default="upper", description="Transform type: upper|lower|strip|title"),
    ],
}


class TemplateSkillCodeGenerator:
    """Template-based code generation for known patterns (zero LLM cost).

    Maps well-known pattern names to pre-written code templates,
    providing instant, reliable code generation for common automation tasks.
    """

    def __init__(self) -> None:
        self._templates: Dict[str, str] = dict(_BUILTIN_TEMPLATES)

    async def generate(self, candidate: Any, context: CodeGenContext) -> Optional[GeneratedSkill]:
        """Generate from template if candidate matches a known pattern."""
        pattern_name = self._match_pattern(candidate)
        if pattern_name is None:
            return None

        template = self._templates[pattern_name]
        code = self._fill_template(template, candidate)

        # Extract function name from template
        fn_match = re.search(r"async\s+def\s+(\w+)", code)
        function_name = fn_match.group(1) if fn_match else pattern_name

        return GeneratedSkill(
            function_name=function_name,
            code=code,
            parameters=list(_TEMPLATE_PARAMETERS.get(pattern_name, [])),
            imports=["from typing import Dict, Any"],
            description=candidate.title,
            preconditions=list(candidate.pre_conditions),
            postconditions=list(candidate.post_conditions),
            triggers=list(candidate.trigger_phrases),
            confidence=0.95,
            generation_method="template",
        )

    def has_template(self, pattern_name: str) -> bool:
        """Check if a template exists for the given pattern."""
        return pattern_name in self._templates

    def register_template(self, name: str, code: str, parameters: List[SkillParameter]) -> None:
        """Register a custom template for a pattern."""
        self._templates[name] = code
        _TEMPLATE_PARAMETERS[name] = parameters

    def _match_pattern(self, candidate: Any) -> Optional[str]:
        """Match a candidate to a known template pattern via keyword heuristics."""
        title_lower = candidate.title.lower()
        steps_text = " ".join(candidate.steps).lower()
        combined = f"{title_lower} {steps_text}"

        # Score each template by keyword overlap
        pattern_keywords: Dict[str, List[str]] = {
            "cross_app_transfer": ["transfer", "copy", "paste", "clipboard", "between apps", "跨应用"],
            "file_organize": ["organize", "sort", "classify", "整理", "move files", "by extension"],
            "web_download": ["download", "url", "browser", "save", "下载"],
            "batch_rename": ["rename", "batch", "pattern", "重命名", "批量"],
            "clipboard_transform": ["clipboard", "transform", "convert", "剪贴板", "format"],
        }

        best_pattern: Optional[str] = None
        best_score = 0

        for pattern, keywords in pattern_keywords.items():
            score = sum(1 for kw in keywords if kw in combined)
            if score > best_score:
                best_score = score
                best_pattern = pattern

        # Require at least 2 keyword matches for confidence
        return best_pattern if best_score >= 2 else None

    def _fill_template(self, template: str, candidate: Any) -> str:
        """Fill template with candidate-specific values (minimal customization).

        Templates are already parameterized via **params, so the main
        customization is the docstring.
        """
        # Replace the docstring with the candidate's title
        lines = template.split("\n")
        for i, line in enumerate(lines):
            if '"""' in line and i > 0:
                lines[i] = f'    """{candidate.title}"""'
                break
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# Composite Generator (Strategy Pattern)
# ═══════════════════════════════════════════════════════════════════════


class CompositeSkillCodeGenerator:
    """LLM-first, template-fallback. Strategy pattern."""

    def __init__(
        self,
        llm_generator: LLMSkillCodeGenerator,
        template_generator: Optional[TemplateSkillCodeGenerator] = None,
    ) -> None:
        self._llm = llm_generator
        self._template = template_generator or TemplateSkillCodeGenerator()

    async def generate(self, candidate: Any, context: CodeGenContext) -> Optional[GeneratedSkill]:
        """LLM-first for semantic accuracy, template fallback for zero-LLM cost."""
        result = await self._llm.generate(candidate, context)
        if result and result.is_valid:
            logger.info(
                "codegen.llm generated skill=%s confidence=%.2f",
                result.function_name, result.confidence,
            )
            return result

        logger.warning("codegen.llm_failed, falling back to template")
        return await self._template.generate(candidate, context)


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════


def _resolve_attr_name(node: ast.Attribute) -> str:
    """Resolve a.b.c attribute chain to dotted string."""
    parts: List[str] = [node.attr]
    current = node.value
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    parts.reverse()
    return ".".join(parts)


def build_default_context(
    existing_skills: Optional[List[str]] = None, episode: Any = None,
) -> CodeGenContext:
    """Factory: build a CodeGenContext with standard VSI interface documentation."""
    return CodeGenContext(
        available_ports=[
            "PerceptionPort.subscribe_fs(paths: List[str]) -> str",
            "PerceptionPort.read_ui_tree(app_id: Optional[str]) -> UINode",
            "PerceptionPort.get_clipboard() -> Dict[str, Any]",
            "PerceptionPort.stream_events() -> AsyncIterator[SystemEvent]",
            "ExecutionPort.perform_file_op(op: str, params: Dict) -> Dict",
            "ExecutionPort.perform_ui_action(node_id: str, action: str, params: Optional[Dict]) -> Dict",
            "ExecutionPort.launch_app(app_id: str) -> Dict",
            "ExecutionPort.run_intent(intent_name: str, params: Dict) -> Dict",
            "ExecutionPort.exec_shell(command: str) -> Dict",
        ],
        available_methods=[
            "file.list", "file.read", "file.write",
            "ax.tree", "ax.focused",
            "clipboard.read", "clipboard.write",
            "app.launch", "app.list",
            "shell.exec",
        ],
        existing_skills=existing_skills or [],
        episode=episode,
    )
