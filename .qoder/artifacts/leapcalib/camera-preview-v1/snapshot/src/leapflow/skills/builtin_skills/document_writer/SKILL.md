---
name: document_writer
description: "Structured document generation: technical docs, API references, reports, and READMEs"
version: 1.0.0
metadata:
  leapflow:
    category: "productivity"
    source: "builtin"
    confidence: 1.0
    quality_score: 1.0
  hermes:
    category: "productivity"
    tags: ["documentation", "technical-writing", "README", "report", "API-docs"]
    requires_tools: ["file_read", "file_write"]
platforms: []
triggers:
  - "write document"
  - "generate docs"
  - "create README"
  - "write report"
  - "写文档"
  - "生成文档"
  - "technical documentation"
  - "API documentation"
---

# Document Writer

## Purpose

Generate well-structured, audience-appropriate documents — technical documentation,
API references, project READMEs, architecture decision records, runbooks, and
reports.  This skill treats documentation as a product: it must be correct, usable,
and maintained like code.

## Guiding Principles

1. **Audience first** — Identify who will read the document and what they need to
   accomplish.  An onboarding guide for new developers is not an API reference.
2. **Inverted pyramid** — Put the most important information first.  Readers skim;
   the answer should be findable in 30 seconds.
3. **Show, don't tell** — Concrete examples beat abstract descriptions.  Every
   non-trivial concept gets a code sample, diagram, or worked example.
4. **Single source of truth** — Documentation should live next to the code it
   describes.  Cross-references over duplication.
5. **Evergreen writing** — Avoid dates, version-specific language, and
   "currently" phrasing that rots.  Write for the next reader, not today's.

## Workflow

### Phase 1 — Analyze the Request

Before generating any content, establish:

1. **Document type**: README, API reference, architecture doc, runbook,
   changelog, tutorial, or report.
2. **Audience**: end users, developers integrating an API, internal team members,
   or stakeholders.  Determine technical depth.
3. **Scope**: what must be covered, what is explicitly out of scope.
4. **Existing documentation**: read relevant existing docs to avoid contradiction
   and find the right insertion point.
5. **Conventions**: check the project for documentation standards (file naming,
   heading style, admonition syntax, link format).

### Phase 2 — Structure the Document

Choose a template based on the document type:

**README**:
```
# Project Name
> One-line description

## Quick Start
## Features
## Installation
## Usage
## Configuration
## Contributing
## License
```

**API Reference**:
```
# API Reference — <Service>
## Authentication
## Endpoints
### <Method> <Path>
  - Description
  - Parameters (table)
  - Request example
  - Response example
  - Error codes
## Rate Limits
## Changelog
```

**Architecture Decision Record (ADR)**:
```
# ADR-NNN: <Title>
## Status: <Proposed|Accepted|Deprecated|Superseded>
## Context
## Decision
## Consequences
## Alternatives Considered
```

**Runbook**:
```
# Runbook: <Procedure>
## Prerequisites
## Steps (numbered, with verification after each)
## Rollback
## Contacts
```

Adapt the template to fit the project; never force content into a section that
adds no value.

### Phase 3 — Generate Content

For each section:

1. Read the relevant source code, config files, or data with `file_read`.
2. Extract facts: function signatures, config keys, environment variables,
   error codes, dependencies.
3. Write prose that is:
   - **Concise**: one idea per paragraph, short sentences.
   - **Precise**: use the exact names from the codebase (no paraphrasing
     class names or API paths).
   - **Active voice**: "The server starts on port 8080" not "Port 8080 is
     used by the server for starting."
4. Add **code examples** for every usage pattern.  Examples must be runnable
   and syntactically correct.
5. Use **tables** for structured data (parameters, config keys, error codes).
6. Use **admonitions** (note, warning, tip) sparingly for critical information
   that readers must not miss.

### Phase 4 — Quality Check

Before delivering:

- **Accuracy**: do code examples actually work?  Do file paths exist?
- **Completeness**: does every public API, config option, or workflow step
  appear?
- **Consistency**: are heading levels, list styles, and code fence languages
  uniform throughout?
- **Links**: are all cross-references valid?  No broken relative paths.
- **Spelling and grammar**: proofread, especially proper nouns and technical
  terms.

Write the final document to the appropriate file with `file_write`.

## Error Handling

| Situation | Action |
|---|---|
| Source code is too complex to fully document | Focus on the public API surface; note internal details as out of scope. |
| Existing docs contradict the code | Trust the code.  Update the docs and flag the discrepancy to the user. |
| No clear project conventions | Default to CommonMark, ATX headings, fenced code blocks, and sentence-case headings. |
| User requests a format you cannot render (PDF, Confluence) | Generate Markdown and advise on conversion tools (pandoc, markdown-to-confluence). |

## Limitations

- This skill generates Markdown (or plain text) documents.  Rich formats
  (PDF, DOCX, HTML) require external conversion.
- Diagram generation is descriptive (Mermaid code blocks); rendering depends
  on the viewer.
- Accuracy depends on reading the codebase; if source files are inaccessible,
  the document will have gaps.
