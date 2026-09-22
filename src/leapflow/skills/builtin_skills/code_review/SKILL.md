---
name: code_review
description: "Structured code review for security, performance, and style"
version: 1.0.0
metadata:
  leapflow:
    category: "development"
    source: "builtin"
    confidence: 1.0
    quality_score: 1.0
  hermes:
    category: "development"
    tags: ["code", "review", "security", "performance", "quality"]
    requires_tools: ["file_read", "shell_run"]
platforms: []
triggers:
  - "review this code"
  - "code review"
  - "check this code for issues"
  - "audit this code"
  - "review my changes"
  - "find bugs in this code"
  - "security review"
  - "review pull request"
---

# Code Review

## Purpose

Perform a thorough, structured code review that evaluates changes across four
dimensions: correctness, security, performance, and maintainability.  Produce
actionable feedback organized by severity so the author can prioritize fixes.

## Guiding Principles

1. **Understand intent first** — Before critiquing implementation, understand
   what the code is trying to achieve.  Read surrounding context, commit
   messages, and related files.
2. **Severity matters** — Distinguish between blocking issues (bugs, security
   holes) and suggestions (style, naming).  Never bury a critical finding in
   a list of nitpicks.
3. **Be specific and actionable** — "This is bad" is not feedback.  Always
   explain *why* something is a problem and *how* to fix it.
4. **Assume competence** — The author made deliberate choices.  When something
   looks wrong, consider whether there is a reason before flagging it.
5. **Scope discipline** — Review the change, not the entire codebase.  Flag
   pre-existing issues only when the change makes them worse or when they
   create a direct interaction risk.

## Workflow

### Phase 1 — Scope the Review

Determine what changed and establish context:

1. Identify the **files changed** — use `file_read` to examine each file or
   use `shell_run` with `git diff` to get the change set.
2. Read **surrounding code** for each changed function/class to understand
   the integration surface.
3. Check for a **test file** corresponding to each changed source file.
4. Note the **language, framework, and project conventions** in use.

Produce a brief scope summary:
- N files changed, M lines added/removed
- Primary area: <module or feature>
- Languages: <lang list>

### Phase 2 — Correctness Analysis

Walk through the logic of each change:

- Does the code do what it claims to do?
- Are **edge cases** handled (null/empty inputs, boundary values, error paths)?
- Is **error handling** present and appropriate?  Are exceptions caught at the
  right granularity?
- Do **types and contracts** match (function signatures, return types, API
  schemas)?
- If tests exist, do they cover the new/changed paths?  Are assertions
  meaningful?

### Phase 3 — Security Analysis

Examine each change for common vulnerability patterns:

- **Injection**: SQL, command, template, XSS — is user input sanitized before
  use in queries, shell commands, or rendered output?
- **Authentication / Authorization**: Does the change bypass or weaken
  access controls?  Are secrets hardcoded?
- **Data exposure**: Could the change leak sensitive data in logs, error
  messages, or API responses?
- **Deserialization**: Is untrusted data deserialized without validation?
- **Dependencies**: Are new dependencies introduced?  Are they from trusted
  sources, actively maintained, and free of known CVEs?
- **Concurrency**: Race conditions, TOCTOU, shared mutable state without
  synchronization.

For each finding, assess exploitability (not just theoretical possibility).

### Phase 4 — Performance Analysis

Look for patterns that could degrade runtime or resource usage:

- **Algorithmic complexity**: O(n²) loops, unnecessary repeated computation,
  missing caches for expensive operations.
- **I/O patterns**: Unbounded queries, N+1 database calls, missing pagination,
  synchronous blocking in async code.
- **Memory**: Large allocations in hot paths, unbounded collections, missing
  resource cleanup.
- **Concurrency**: Lock contention, excessive context switching, thread-unsafe
  shared state.

Flag only issues that are *likely* to matter at the project's scale.

### Phase 5 — Maintainability & Style

Evaluate code quality and long-term health:

- **Naming**: Are variables, functions, and classes named clearly and
  consistently?
- **Structure**: Is the code well-factored?  Are responsibilities separated?
- **Duplication**: Is there copy-paste that should be extracted?
- **Documentation**: Are public APIs documented?  Are complex algorithms
  explained?
- **Consistency**: Does the change follow the project's existing conventions?

Style issues are lowest priority — flag them, but clearly separate from
blocking concerns.

### Phase 6 — Report

Produce a structured review:

```
## Review Summary
<1–3 sentence overall assessment: approve / request changes / needs discussion>

## Critical Issues (must fix)
1. **[SECURITY]** <file>:<line> — <description>
   **Fix**: <specific remediation>

## Warnings (should fix)
1. **[PERF]** <file>:<line> — <description>

## Suggestions (nice to have)
1. **[STYLE]** <file>:<line> — <description>

## Positive Notes
- <Things done well — always include at least one>
```

Rules:
- Categorize every finding: `[BUG]`, `[SECURITY]`, `[PERF]`, `[STYLE]`,
  `[TEST]`, `[DOC]`.
- Include file path and line number (or function name) for each finding.
- "Critical" = the change should not merge without addressing this.
- "Warning" = strongly recommended but not blocking.
- "Suggestion" = optional improvement.
- Always end with something positive — reinforce good patterns.

## Error Handling

| Situation | Action |
|---|---|
| Cannot read a changed file | Report which files were inaccessible; review what you can. |
| No test files found | Flag as a warning: "No corresponding tests found for <file>." |
| Unfamiliar language or framework | State your confidence level; focus on universal principles (logic, security, naming). |
| Change is too large (>1000 lines) | Suggest splitting the review; focus on the highest-risk files first. |
