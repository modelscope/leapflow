# Copyright (c) Alibaba, Inc. and its affiliates.
"""Content extraction for fetched web responses.

A fetch tool that hands raw markup to the model is not usable: a single page can
exceed the whole turn's context budget, and the useful text is a small fraction
of it. Extraction is therefore part of the capability, not an afterthought.

Routing is driven by the response's ``Content-Type`` header rather than by
sniffing the body, so the decision follows the protocol contract instead of
guessing from content. Two extractor implementations exist for HTML: a
dependency-free stdlib parser that always ships, and trafilatura when the
``web`` extra is installed. The stdlib one is not a placeholder — it stays the
fallback whenever trafilatura is absent *or* declines a page (it returns nothing
on layouts it cannot model), so both paths are load-bearing.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urljoin

logger = logging.getLogger(__name__)

# Kinds the tool reports back to the model. Binary never enters the transcript;
# it is written to cache and referenced by path instead.
KIND_JSON = "json"
KIND_TEXT = "text"
KIND_HTML = "html"
KIND_BINARY = "binary"

_JSON_TYPES = ("application/json", "text/json")
_HTML_TYPES = ("text/html", "application/xhtml+xml")
_TEXT_TYPES = (
    "text/plain", "text/markdown", "text/csv", "text/tab-separated-values",
    "application/xml", "text/xml", "application/javascript", "text/javascript",
    "application/x-yaml", "text/yaml",
)

# Elements whose text is chrome, navigation, or code that never belongs in an
# extracted reading of the page.
_SKIP_TAGS = frozenset({
    "script", "style", "noscript", "svg", "canvas", "iframe", "form",
    "nav", "footer", "aside", "template", "select", "button", "head",
})
_BLOCK_TAGS = frozenset({
    "p", "div", "section", "article", "main", "header", "br", "hr", "li", "tr",
    "td", "th", "pre", "blockquote", "figcaption", "dt", "dd", "table", "ul",
    "ol", "h1", "h2", "h3", "h4", "h5", "h6",
})
_HEADING_LEVELS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}
_MAX_LINKS = 30
_BLANK_RUN = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class ExtractedContent:
    """Normalized, context-ready view of a response body."""

    kind: str
    text: str = ""
    data: Any = None
    title: str = ""
    links: tuple[tuple[str, str], ...] = ()
    truncated: bool = False
    extractor: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)


@runtime_checkable
class ContentExtractor(Protocol):
    """Turns an HTML document into readable text."""

    name: str

    def available(self) -> bool:
        """Whether this extractor can run in the current environment."""
        ...

    def extract(self, html: str, *, url: str) -> ExtractedContent | None:
        """Return extracted content, or ``None`` to defer to the next extractor."""
        ...


def kind_for_content_type(content_type: str) -> str:
    """Map a Content-Type header to an extraction kind."""
    base = (content_type or "").split(";", 1)[0].strip().lower()
    if not base:
        return KIND_TEXT
    if base in _JSON_TYPES or base.endswith("+json"):
        return KIND_JSON
    if base in _HTML_TYPES:
        return KIND_HTML
    if base in _TEXT_TYPES or base.startswith("text/"):
        return KIND_TEXT
    return KIND_BINARY


class _HtmlTextParser(HTMLParser):
    """Collect readable text, a title, and links from an HTML document."""

    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self._base_url = base_url
        self._skip_depth = 0
        self._in_title = False
        self._pending_heading = 0
        self._parts: list[str] = []
        self._links: list[tuple[str, str]] = []
        self._link_text: list[str] = []
        self._link_href = ""
        self.title = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            # <title> lives inside <head>, so keep reading it while skipping the
            # rest of the head's machinery.
            self._skip_depth += 1
            return
        if self._skip_depth and tag != "title":
            return
        if tag == "title":
            self._in_title = True
            return
        if tag in _HEADING_LEVELS:
            self._newline()
            self._pending_heading = _HEADING_LEVELS[tag]
            return
        if tag == "li":
            self._newline()
            self._parts.append("- ")
            return
        if tag == "a":
            self._link_href = next((v or "" for k, v in attrs if k == "href"), "")
            self._link_text = []
            return
        if tag in _BLOCK_TAGS:
            self._newline()

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag == "title":
            self._in_title = False
            return
        if self._skip_depth:
            return
        if tag == "a" and self._link_href:
            text = " ".join("".join(self._link_text).split())
            url = urljoin(self._base_url, self._link_href)
            if text and url.startswith(("http://", "https://")):
                self._links.append((text, url))
            self._link_href = ""
            self._link_text = []
            return
        if tag in _BLOCK_TAGS or tag in _HEADING_LEVELS:
            self._newline()

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title = (self.title + data).strip()
            return
        if self._skip_depth or not data.strip():
            return
        if self._pending_heading:
            self._parts.append("#" * self._pending_heading + " ")
            self._pending_heading = 0
        if self._link_href:
            self._link_text.append(data)
        self._parts.append(data)

    def _newline(self) -> None:
        if self._parts and not self._parts[-1].endswith("\n"):
            self._parts.append("\n")

    def text(self) -> str:
        raw = "".join(self._parts)
        lines = [" ".join(line.split()) for line in raw.splitlines()]
        return _BLANK_RUN.sub("\n\n", "\n".join(lines)).strip()

    def links(self) -> tuple[tuple[str, str], ...]:
        seen: set[str] = set()
        unique: list[tuple[str, str]] = []
        for text, url in self._links:
            if url in seen:
                continue
            seen.add(url)
            unique.append((text, url))
            if len(unique) >= _MAX_LINKS:
                break
        return tuple(unique)


class StdlibHtmlExtractor:
    """Dependency-free HTML reader built on ``html.parser``.

    Always available, so a LeapFlow install with no extras can still read the
    web. It removes chrome by element type rather than scoring text density, so
    expect more boilerplate than trafilatura on article pages — the trade is zero
    dependencies and no failure mode where extraction is simply unavailable.
    """

    name = "stdlib"

    def available(self) -> bool:
        return True

    def extract(self, html: str, *, url: str) -> ExtractedContent | None:
        parser = _HtmlTextParser(url)
        try:
            parser.feed(html)
            parser.close()
        except (AssertionError, ValueError) as exc:
            # Malformed markup: report what was collected rather than failing the
            # whole fetch, since a partial reading is still useful.
            logger.debug("stdlib html extraction incomplete for %s: %s", url, exc)
        return ExtractedContent(
            kind=KIND_HTML,
            text=parser.text(),
            title=parser.title,
            links=parser.links(),
            extractor=self.name,
        )


class TrafilaturaExtractor:
    """Boilerplate-removing extractor backed by the optional ``web`` extra.

    Returns ``None`` when trafilatura is missing or produces nothing for a page,
    which hands the document to the stdlib extractor instead of reporting an
    empty body.
    """

    name = "trafilatura"

    def available(self) -> bool:
        try:
            import trafilatura  # noqa: F401
        except ImportError:
            return False
        return True

    def extract(self, html: str, *, url: str) -> ExtractedContent | None:
        try:
            import trafilatura
        except ImportError:
            return None
        try:
            text = trafilatura.extract(
                html,
                url=url,
                include_links=True,
                include_tables=True,
                output_format="markdown",
                with_metadata=False,
            )
        except Exception as exc:  # noqa: BLE001 - never fail a fetch on extraction
            logger.debug("trafilatura extraction failed for %s: %s", url, exc)
            return None
        if not text or not text.strip():
            return None
        title = ""
        links: tuple[tuple[str, str], ...] = ()
        # Title and links are not part of trafilatura's markdown output, so take
        # them from the always-present structural reader.
        structural = StdlibHtmlExtractor().extract(html, url=url)
        if structural is not None:
            title = structural.title
            links = structural.links
        return ExtractedContent(
            kind=KIND_HTML,
            text=text.strip(),
            title=title,
            links=links,
            extractor=self.name,
        )


def html_extractors(prefer: str = "auto") -> tuple[ContentExtractor, ...]:
    """Return the extractor chain, most capable first.

    ``prefer='stdlib'`` pins the dependency-free reader so behavior can be made
    reproducible regardless of which extras happen to be installed.
    """
    stdlib = StdlibHtmlExtractor()
    if prefer == "stdlib":
        return (stdlib,)
    candidates: list[ContentExtractor] = [TrafilaturaExtractor(), stdlib]
    return tuple(item for item in candidates if item.available())


def extract_html(html: str, *, url: str, prefer: str = "auto") -> ExtractedContent:
    """Extract readable content, walking the extractor chain until one answers."""
    for extractor in html_extractors(prefer):
        result = extractor.extract(html, url=url)
        if result is not None and result.text:
            return result
    return ExtractedContent(kind=KIND_HTML, text="", extractor="none")


def select_path(data: Any, path: str) -> tuple[Any, str]:
    """Return ``(value, error)`` for a dotted path into decoded JSON.

    Dotted segments with integer indices (``chart.result.0.meta``) cover the
    shape of real API payloads without pulling in a JSONPath dependency. The
    error string names the segment that failed and what was available, so the
    model can correct the path in the same turn instead of re-fetching blindly.
    """
    if not path:
        return data, ""
    current = data
    walked: list[str] = []
    for segment in [item for item in path.split(".") if item]:
        if isinstance(current, dict):
            if segment not in current:
                available = ", ".join(sorted(current)[:12]) or "no keys"
                return None, (
                    f"select path {path!r} failed at {'.'.join(walked + [segment])!r}: "
                    f"available keys: {available}"
                )
            current = current[segment]
        elif isinstance(current, list):
            try:
                index = int(segment)
            except ValueError:
                return None, (
                    f"select path {path!r} failed at {'.'.join(walked + [segment])!r}: "
                    f"expected a list index, list has {len(current)} items"
                )
            if not -len(current) <= index < len(current):
                return None, (
                    f"select path {path!r} failed at {'.'.join(walked + [segment])!r}: "
                    f"index out of range, list has {len(current)} items"
                )
            current = current[index]
        else:
            return None, (
                f"select path {path!r} failed at {'.'.join(walked + [segment])!r}: "
                f"{type(current).__name__} is not indexable"
            )
        walked.append(segment)
    return current, ""


def decode_json(body: str) -> tuple[Any, str]:
    """Decode a JSON body, returning ``(value, error)``."""
    try:
        return json.loads(body), ""
    except json.JSONDecodeError as exc:
        preview = " ".join(body[:200].split())
        return None, (
            f"Response was declared as JSON but did not parse: {exc.msg} "
            f"at line {exc.lineno} column {exc.colno}. Body starts with: {preview!r}"
        )


__all__ = [
    "KIND_BINARY",
    "KIND_HTML",
    "KIND_JSON",
    "KIND_TEXT",
    "ContentExtractor",
    "ExtractedContent",
    "StdlibHtmlExtractor",
    "TrafilaturaExtractor",
    "decode_json",
    "extract_html",
    "html_extractors",
    "kind_for_content_type",
    "select_path",
]
