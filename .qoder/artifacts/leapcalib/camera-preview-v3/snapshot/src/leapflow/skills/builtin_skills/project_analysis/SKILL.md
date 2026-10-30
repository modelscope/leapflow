---
name: project_analysis
description: "Project structure analysis, dependency audit, and tech stack identification"
version: 1.0.0
metadata:
  leapflow:
    category: "development"
    source: "builtin"
    confidence: 1.0
    quality_score: 1.0
  hermes:
    category: "development"
    tags: ["project", "analysis", "dependencies", "architecture", "tech-stack"]
    requires_tools: ["file_read", "file_list", "shell_run"]
platforms: []
triggers:
  - "analyze this project"
  - "what does this project do"
  - "project overview"
  - "audit dependencies"
  - "identify the tech stack"
  - "project structure"
  - "codebase analysis"
  - "explain this repository"
---

# Project Analysis

## Purpose

Produce a comprehensive overview of a software project: its structure, tech
stack, dependencies, build system, and overall health.  The output is a
structured report that helps someone unfamiliar with the project understand
its architecture and identify areas of concern.

## Guiding Principles

1. **Evidence-based** — Every claim about the project must be grounded in a
   file or command output you actually observed.  Do not infer a framework is
   used unless you see its config or import.
2. **Proportional depth** — Spend more time on large or complex areas; do not
   enumerate every file in a trivial directory.
3. **Actionable output** — The report should help the reader *do* something:
   onboard faster, fix a vulnerability, upgrade a dependency.
4. **Non-destructive** — Never modify project files.  All shell commands must
   be read-only (e.g., `ls`, `cat`, `find`, `wc`, `grep`, package manager
   list/audit commands).

## Workflow

### Phase 1 — Top-Level Scan

Get the lay of the land:

1. **List root directory** — `file_list` at the project root.  Note key
   marker files:
   - `package.json` / `yarn.lock` / `pnpm-lock.yaml` → Node.js/JS/TS
   - `pyproject.toml` / `setup.py` / `requirements.txt` / `uv.lock` → Python
   - `go.mod` → Go
   - `Cargo.toml` → Rust
   - `pom.xml` / `build.gradle` → Java/Kotlin
   - `Gemfile` → Ruby
   - `Makefile`, `CMakeLists.txt`, `Dockerfile`, `docker-compose.yml`
2. **Read config files** — `file_read` on the primary manifest
   (`package.json`, `pyproject.toml`, etc.) to extract:
   - Project name, version, description
   - Entry points / main modules
   - Scripts / build commands
3. **Identify source layout** — conventional layouts (`src/`, `lib/`, `app/`,
   `cmd/`, `internal/`, `pkg/`) vs flat structure.

### Phase 2 — Tech Stack Identification

From the evidence gathered, compile:

| Layer | Technology | Evidence |
|---|---|---|
| Language(s) | e.g., Python 3.11 | `pyproject.toml` `requires-python` |
| Framework | e.g., FastAPI | import in `app/main.py` |
| Database | e.g., PostgreSQL | `DATABASE_URL` in `.env.example` |
| Build tool | e.g., uv | `uv.lock` present |
| CI/CD | e.g., GitHub Actions | `.github/workflows/` |
| Container | e.g., Docker | `Dockerfile` |
| Testing | e.g., pytest | `[tool.pytest]` in `pyproject.toml` |

Do not guess — leave a cell blank if no evidence is found.

### Phase 3 — Directory Structure Map

Produce a tree-style outline of the major directories (depth ≤ 3), annotating
each with its purpose:

```
project-root/
├── src/app/           # Application core (FastAPI routes, services)
├── src/models/        # Database models (SQLAlchemy)
├── tests/             # Test suite (pytest)
├── migrations/        # Alembic DB migrations
├── scripts/           # Dev/ops helper scripts
├── docs/              # Documentation
└── .github/workflows/ # CI pipelines
```

For each major directory, note:
- Approximate file count and dominant file types
- Key entry points or important files
- Any unusual or non-standard organization

### Phase 4 — Dependency Audit

Examine the project's declared dependencies:

1. **Read the manifest** — extract all dependencies and their version
   constraints.
2. **Categorize** — runtime vs dev-only vs optional.
3. **Flag concerns**:
   - **Pinning**: Are versions pinned or floating?  Wide ranges in production
     dependencies are a risk.
   - **Outdated**: If possible (e.g., `npm outdated`, `pip list --outdated`),
     identify significantly outdated packages.
   - **Known vulnerabilities**: If an audit command is available
     (`npm audit`, `pip-audit`, `cargo audit`), run it.
   - **Unused**: If the project has an unused-dependency detector, note any
     findings.
   - **License**: Flag any copyleft (GPL) dependencies in an otherwise
     permissive-licensed project.
4. **Count**: Total dependencies, direct vs transitive (if lockfile present).

### Phase 5 — Build & Run Assessment

Determine how to build and run the project:

1. Look for documented commands in `Makefile`, `package.json` scripts,
   `pyproject.toml` scripts, or a `README`.
2. Check for required environment variables (`.env.example`, config files).
3. Note any non-obvious setup steps (database migration, code generation,
   native compilation).
4. Assess whether a new contributor could get running with the documented
   steps alone.

### Phase 6 — Report

Produce the final report:

```
## Project Overview
- **Name**: <name>
- **Description**: <1–2 sentences>
- **Primary Language**: <lang>
- **License**: <license>

## Tech Stack
<table from Phase 2>

## Directory Structure
<annotated tree from Phase 3>

## Dependencies
- **Total**: N direct, M transitive
- **Health**: <summary>
- **Concerns**: <bulleted list of flagged issues>

## Build & Run
- **Build command**: `<cmd>`
- **Run command**: `<cmd>`
- **Prerequisites**: <list>
- **Onboarding friction**: <low/medium/high with explanation>

## Observations & Recommendations
1. <Actionable recommendation>
2. ...
```

## Error Handling

| Situation | Action |
|---|---|
| No recognizable manifest file | State that the project type could not be identified; describe what was found at the root. |
| Cannot run audit commands (tool not installed) | Skip the vulnerability check; note it was not performed and suggest the user run it manually. |
| Very large monorepo | Focus on the top-level structure and the most active or largest sub-packages; do not attempt to map everything. |
| Binary or generated files dominate | Note this and focus analysis on the source directories. |
