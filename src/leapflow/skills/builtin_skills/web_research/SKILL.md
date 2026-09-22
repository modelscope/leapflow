---
name: web_research
description: "Multi-step web search with cross-verification and structured synthesis"
version: 1.0.0
metadata:
  leapflow:
    category: "research"
    source: "builtin"
    confidence: 1.0
    quality_score: 1.0
  hermes:
    category: "research"
    tags: ["web", "search", "research", "synthesis", "fact-checking"]
    requires_tools: ["web_search", "web_fetch"]
platforms: []
triggers:
  - "research this topic"
  - "search the web for"
  - "find information about"
  - "look up"
  - "investigate online"
  - "web research"
  - "deep search"
  - "find and summarize"
---

# Web Research

## Purpose

Conduct rigorous, multi-step web research that goes beyond a single search query.
This skill orchestrates a structured research workflow: planning a search strategy,
executing multiple targeted queries, cross-verifying claims across independent
sources, and synthesizing findings into a coherent, well-sourced report.

## Guiding Principles

1. **Breadth before depth** — Cast a wide net with diverse queries before drilling
   into any single source.  Reformulate queries when initial results are thin.
2. **Source triangulation** — Never trust a single source.  Every key claim must
   appear in at least two independent sources before it is treated as established.
3. **Recency awareness** — Prefer recent sources for fast-moving topics.  Flag
   when the most recent source is older than the topic warrants.
4. **Bias detection** — Note when sources share a common owner, funding body, or
   obvious editorial slant.  Weight accordingly.
5. **Transparency** — Every factual statement in the final report must link back
   to the source URL it was derived from.

## Workflow

### Phase 1 — Understand the Question

Before any search, decompose the user's request:

- Identify the **core question** (what must be answered).
- List **sub-questions** that feed into the core answer.
- Note any **constraints**: date range, geography, domain, language.
- Decide on a **search budget**: how many queries are proportionate to the task
  (typically 3–8 queries; simple factual lookups need fewer).

Output a brief research plan (≤ 6 bullet points) and confirm with yourself
before proceeding.

### Phase 2 — Execute Searches

For each sub-question, craft a targeted search query:

1. Use **specific, keyword-rich** queries — avoid full sentences.
2. Vary phrasing across queries to surface different result sets.
3. Include **domain-specific qualifiers** when useful
   (e.g., `site:arxiv.org`, `filetype:pdf`, `"exact phrase"`).
4. After each `web_search` call, scan the snippets.  If a result looks
   authoritative or contains the needed data, fetch the full page with
   `web_fetch`.
5. Record each source: URL, title, date (if available), and a 1–2 sentence
   summary of what it contributes.

Do NOT fetch every result — only those that plausibly advance an answer.

### Phase 3 — Cross-Verify

Lay out the key claims you have gathered and check:

- Does each claim appear in **≥ 2 independent sources**?
- Are any claims **contradicted** by another source?  If so, note the
  contradiction and assess which source is more credible (recency, authority,
  methodology).
- Are there **gaps** — sub-questions that no source adequately addresses?
  If so, run 1–2 additional targeted searches.

### Phase 4 — Synthesize

Produce a structured report:

```
## Summary
<2–4 sentence executive summary>

## Key Findings
1. <Finding with inline source reference [1]>
2. ...

## Contradictions / Uncertainties
- <Any unresolved conflicts or low-confidence claims>

## Sources
[1] <title> — <url> (accessed <date>)
[2] ...
```

Rules for the report:
- Lead with the answer — do not bury it after methodology.
- Use numbered source references; every factual claim must cite at least one.
- Clearly separate established facts from speculation or single-source claims.
- If the research is inconclusive, say so explicitly rather than hedging.

### Phase 5 — Self-Check

Before delivering:

- Re-read the original question.  Does the report actually answer it?
- Are all source URLs valid (no hallucinated links)?
- Is the report length proportionate to the question's complexity?

## Error Handling

| Situation | Action |
|---|---|
| `web_search` returns no results | Reformulate the query with broader or alternative terms; try up to 2 reformulations before reporting the gap. |
| `web_fetch` fails or returns empty | Note the URL as inaccessible; do not cite it.  Rely on the search snippet if it contained the needed fact. |
| Contradictory sources with equal credibility | Present both perspectives explicitly; do not silently pick one. |
| Topic is too broad | Ask the user to narrow scope before consuming the full search budget. |

## Limitations

- This skill relies on `web_search` and `web_fetch` tools.  If they are
  unavailable, the skill cannot execute.
- Results are bounded by what the search engine indexes; paywalled or
  dynamically rendered content may be inaccessible.
- Always note when a topic requires expertise that web search alone cannot
  reliably provide (medical, legal, financial advice).
