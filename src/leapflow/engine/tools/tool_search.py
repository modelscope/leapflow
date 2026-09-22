# Copyright (c) Alibaba, Inc. and its affiliates.
"""BM25-based tool search engine and budget-driven listing renderer.

This module provides:

1. :class:`ToolSearchIndex` — thread-safe BM25 search over tool catalog entries
2. :func:`render_tool_listing` — budget-aware tool catalog renderer for system
   prompts with automatic degradation: full → names_only → grouped → none
3. :func:`entries_from_tool_definitions` — converter from OpenAI-format tool
   definitions to search-index entries

The search engine is invoked exclusively by the ``tool_search`` meta-tool
(LLM-initiated), never by :class:`DisclosurePlanner` (which operates on
structural signals only, per the design contract in ``context_disclosure.py``).
"""
from __future__ import annotations

import math
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

# ── Lightweight English stemmer ────────────────────────────────────────
# Two-pass suffix-stripping: plurals first, then verbal/nominal suffixes.
# No dictionary lookup required.  Sufficient for tool name / description
# matching where the vocabulary is domain-limited.  Both indexing and
# querying use the same stemmer, so consistency outweighs perfection.

_PASS2_RULES: tuple[tuple[str, str], ...] = (
    ("ational", "ate"),
    ("tional", "tion"),
    ("ization", "ize"),
    ("ation", "ate"),
    ("ously", "ous"),
    ("ively", "ive"),
    ("ness", ""),
    ("ment", ""),
    ("ible", ""),
    ("able", ""),
    ("ful", ""),
    ("ally", "al"),
    ("ing", ""),
    ("tion", ""),
    ("sion", ""),
    ("ed", ""),
    ("ly", ""),
    ("er", ""),
)

_MIN_STEM = 3


def _stem(word: str) -> str:
    """Simplified English suffix-stripping stemmer.

    Two-pass: normalize plurals first, then verbal/nominal suffixes.
    A minimum stem length of 3 prevents over-stripping short words.
    """
    if len(word) <= _MIN_STEM:
        return word

    # Pass 1 — plurals
    if word.endswith("ies") and len(word) > 4:
        word = word[:-3] + "i"
    elif word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("es") and len(word) > 4:
        pre = word[:-2]
        if pre[-1] in "sxz" or pre.endswith(("ch", "sh")):
            word = pre
        else:
            word = word[:-1]  # keep the 'e': "files" → "file"
    elif word.endswith("s") and not word.endswith("ss") and len(word) > 4:
        word = word[:-1]

    if len(word) <= _MIN_STEM:
        return word

    # Pass 2 — verbal / nominal suffixes (longest match first)
    for suffix, replacement in _PASS2_RULES:
        if word.endswith(suffix):
            stem = word[: -len(suffix)] + replacement
            if len(stem) >= _MIN_STEM:
                return stem

    return word


_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumeric boundaries, stem each token.

    Single-character tokens are dropped (articles, stray letters).
    """
    return [_stem(tok) for tok in _TOKEN_RE.findall(text.lower()) if len(tok) > 1]


# ── BM25 search index ─────────────────────────────────────────────────


@dataclass
class _Document:
    """Internal document representation for a single tool."""

    name: str
    category: str
    summary: str
    tokens: list[str] = field(default_factory=list)
    tf: dict[str, int] = field(default_factory=dict)
    length: int = 0


class ToolSearchIndex:
    """Thread-safe BM25 search index over tool catalog entries.

    Build the index from tool-entry dicts via :meth:`rebuild`, then search
    with :meth:`search`.  Parameters *k1* and *b* follow Robertson's
    recommendations for short documents.
    """

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._lock = threading.Lock()
        self._docs: list[_Document] = []
        self._df: dict[str, int] = {}
        self._avgdl: float = 0.0
        self._N: int = 0
        self._name_lookup: dict[str, int] = {}  # lowercase name → doc index

    def rebuild(self, entries: Sequence[Mapping[str, Any]]) -> None:
        """Rebuild the index from tool entries.

        Each entry must have at minimum ``name`` and ``summary`` keys.
        Optional: ``category``, ``parameter_names``.
        """
        docs: list[_Document] = []
        df: dict[str, int] = {}
        name_lookup: dict[str, int] = {}
        total_length = 0

        for idx, entry in enumerate(entries):
            name = str(entry.get("name", ""))
            category = str(entry.get("category", ""))
            summary = str(entry.get("summary", ""))
            param_names = entry.get("parameter_names") or ()

            text_parts = [name, category, summary]
            text_parts.extend(str(p) for p in param_names)
            text = " ".join(text_parts)

            tokens = _tokenize(text)
            tf: dict[str, int] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1

            doc = _Document(
                name=name,
                category=category,
                summary=summary,
                tokens=tokens,
                tf=tf,
                length=len(tokens),
            )
            docs.append(doc)
            if name:
                name_lookup[name.lower()] = idx
            total_length += len(tokens)

            for tok in set(tokens):
                df[tok] = df.get(tok, 0) + 1

        avgdl = total_length / len(docs) if docs else 0.0

        with self._lock:
            self._docs = docs
            self._df = df
            self._avgdl = avgdl
            self._N = len(docs)
            self._name_lookup = name_lookup

    def search(self, query: str, max_results: int = 10) -> list[dict[str, Any]]:
        """Search tools by BM25 relevance.

        Returns a list of dicts with ``name``, ``category``, ``summary``,
        ``score``.

        Filtering gates:

        - **Gate Token**: the highest-IDF query term must appear in the
          document.
        - **Term Coverage**: for queries with >= 4 unique terms, >= 50%
          of terms must match.
        - **Exact name match**: always ranks first (``score="exact_match"``).
        """
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        with self._lock:
            docs = list(self._docs)
            df = dict(self._df)
            avgdl = self._avgdl
            N = self._N

        if N == 0:
            return []

        # IDF for each unique query token
        unique_query = list(dict.fromkeys(query_tokens))
        idfs: dict[str, float] = {}
        for tok in unique_query:
            n_tok = df.get(tok, 0)
            idfs[tok] = math.log((N - n_tok + 0.5) / (n_tok + 0.5) + 1.0)

        gate_token = max(unique_query, key=lambda t: idfs[t])
        num_query_terms = len(unique_query)

        # Exact name matching: normalize query → potential tool name
        exact_name = query.strip().lower().replace("-", "_").replace(" ", "_")

        results: list[tuple[float, _Document]] = []

        for doc in docs:
            # Exact name match → infinite score
            if doc.name.lower() == exact_name:
                results.append((float("inf"), doc))
                continue

            # Gate Token: document must contain the highest-IDF query term
            if gate_token not in doc.tf:
                continue

            # Term Coverage: long queries (>= 4 terms) require >= 50% match
            if num_query_terms >= 4:
                matched = sum(1 for tok in unique_query if tok in doc.tf)
                if matched / num_query_terms < 0.5:
                    continue

            # BM25 score
            score = 0.0
            for tok in unique_query:
                tf_val = doc.tf.get(tok, 0)
                if tf_val == 0:
                    continue
                idf = idfs[tok]
                numerator = tf_val * (self._k1 + 1)
                denominator = tf_val + self._k1 * (
                    1 - self._b + self._b * doc.length / avgdl
                )
                score += idf * numerator / denominator

            if score > 0:
                results.append((score, doc))

        # Sort: descending score, then ascending name for stability
        results.sort(key=lambda x: (-x[0], x[1].name))

        return [
            {
                "name": doc.name,
                "category": doc.category,
                "summary": doc.summary,
                "score": "exact_match" if math.isinf(score) else round(score, 4),
            }
            for score, doc in results[:max_results]
        ]


# ── Budget-driven listing renderer ────────────────────────────────────


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


class ListingLevel:
    """Rendering levels for progressive listing degradation."""

    FULL = "full"
    NAMES_ONLY = "names_only"
    GROUPED = "grouped"
    NONE = "none"


def render_tool_listing(
    tool_definitions: Sequence[Mapping[str, Any]],
    token_budget: int = 2000,
) -> tuple[str, str]:
    """Render a tool catalog listing that fits within *token_budget*.

    Returns ``(rendered_text, listing_level)`` where *listing_level* is one
    of ``ListingLevel.FULL``, ``NAMES_ONLY``, ``GROUPED``, ``NONE``.

    Degradation order: full → names_only → grouped → none.  Within each
    level the largest categories are first candidates for reduction.

    The output is **byte-stable**: categories and tools within categories
    are sorted alphabetically, which is prefix-cache friendly.
    """
    # Function-local import: cross-sub-package import kept local per
    # engine module architecture rules (no top-level cross-sub-package imports).
    from leapflow.engine.context.context_disclosure import (
        CapabilityManifest,
        build_capability_manifests,
    )

    manifests = build_capability_manifests(tool_definitions)
    manifest_by_name: dict[str, CapabilityManifest] = {m.name: m for m in manifests}

    # Group by category, each list sorted by name for byte stability
    category_tools: dict[str, list[tuple[str, str, str]]] = {}
    for td in tool_definitions:
        func = td.get("function", {}) if isinstance(td, dict) else {}
        name = str(func.get("name") or td.get("name") or "")
        if not name:
            continue
        desc = str(func.get("description") or "")
        params = ", ".join(
            sorted(func.get("parameters", {}).get("properties", {}).keys())
        )
        manifest = manifest_by_name.get(name)
        cat = manifest.category if manifest else "unclassified"
        category_tools.setdefault(cat, []).append((name, params, desc))

    for tools_list in category_tools.values():
        tools_list.sort(key=lambda x: x[0])

    sorted_cats = sorted(category_tools.keys())

    # Level 1: FULL — **name**(params) [tag]: description
    full_lines: list[str] = []
    for cat in sorted_cats:
        for name, params, desc in category_tools[cat]:
            manifest = manifest_by_name.get(name)
            tag = (
                f" [capability_expand category: {manifest.category}]"
                if manifest and not manifest.is_core
                else ""
            )
            full_lines.append(f"- **{name}**({params}){tag}: {desc}")
    full_text = "\n".join(full_lines)
    if _estimate_tokens(full_text) <= token_budget:
        return full_text, ListingLevel.FULL

    # Level 2: NAMES_ONLY — name [category]: short summary
    names_lines: list[str] = []
    for cat in sorted_cats:
        for name, _, desc in category_tools[cat]:
            manifest = manifest_by_name.get(name)
            tag = f" [{manifest.category}]" if manifest else ""
            short = desc[:80].rstrip()
            if len(desc) > 80:
                short += "..."
            names_lines.append(f"- {name}{tag}: {short}")
    names_text = "\n".join(names_lines)
    if _estimate_tokens(names_text) <= token_budget:
        return names_text, ListingLevel.NAMES_ONLY

    # Level 3: GROUPED — **category**: tool1, tool2, tool3
    grouped_lines: list[str] = []
    for cat in sorted_cats:
        names = ", ".join(n for n, _, _ in category_tools[cat])
        grouped_lines.append(f"**{cat}**: {names}")
    grouped_text = "\n".join(grouped_lines)
    if _estimate_tokens(grouped_text) <= token_budget:
        return grouped_text, ListingLevel.GROUPED

    # Level 4: NONE — count only
    total = sum(len(tl) for tl in category_tools.values())
    none_text = (
        f"{total} tools available across {len(sorted_cats)} categories. "
        "Use `tool_search` to find specific tools."
    )
    return none_text, ListingLevel.NONE


# ── Helper: convert OpenAI tool definitions to search entries ──────────


def entries_from_tool_definitions(
    tool_definitions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Convert OpenAI-format tool definitions to search index entries.

    Each returned dict contains ``name``, ``category``, ``summary``, and
    ``parameter_names`` — the fields :class:`ToolSearchIndex` indexes.
    """
    from leapflow.engine.context.context_disclosure import build_capability_manifests

    manifests = build_capability_manifests(tool_definitions)
    manifest_by_name = {m.name: m for m in manifests}

    entries: list[dict[str, Any]] = []
    for td in tool_definitions:
        func = td.get("function", {}) if isinstance(td, dict) else {}
        name = str(func.get("name") or td.get("name") or "")
        if not name:
            continue
        manifest = manifest_by_name.get(name)
        param_names = list(func.get("parameters", {}).get("properties", {}).keys())
        entries.append(
            {
                "name": name,
                "category": manifest.category if manifest else "",
                "summary": (
                    manifest.summary
                    if manifest
                    else str(func.get("description", ""))
                ),
                "parameter_names": param_names,
            }
        )
    return entries
