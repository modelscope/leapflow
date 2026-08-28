"""Tests for web_fetch: transport contract, egress gating, and extraction.

Hermetic by construction: no test performs a real request. Transports are
replaced with a fake, and URLs use IP literals so target classification never
touches DNS.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from leapflow.security.actions import ActionDescriptor, ActionKind
from leapflow.security.network import UrlRejected, classify_url
from leapflow.security.risk import DefaultRiskClassifier, RiskLevel
from leapflow.tools import web_fetch as wf
from leapflow.tools.web_extract import (
    KIND_BINARY,
    KIND_HTML,
    KIND_JSON,
    KIND_TEXT,
    StdlibHtmlExtractor,
    decode_json,
    kind_for_content_type,
    select_path,
)

PUBLIC_URL = "https://93.184.216.34/data"
LOOPBACK_URL = "http://127.0.0.1:8765/state"
METADATA_URL = "http://169.254.169.254/latest/meta-data/iam/security-credentials/"


def _settings(**overrides):
    base = {
        "web_transport": "auto",
        "web_timeout_s": 5.0,
        "web_max_bytes": 1_000_000,
        "web_max_retries": 0,
        "web_max_redirects": 3,
        "web_user_agent": "",
        "web_extractor": "stdlib",
        "web_private_targets": "approval",
        # Off by default so transport behavior is tested without cache interference;
        # the cache tests below opt in with a real layout.
        "web_cache_ttl_s": 0.0,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _settings_with_cache(tmp_path, **overrides):
    """Settings backed by a real CacheLayout under ``tmp_path``."""
    from leapflow.layout import build_layout

    profile_layout = build_layout(tmp_path / "home").ensure(profile_id="default")
    workspace = tmp_path / "ws"
    workspace.mkdir(exist_ok=True)
    return _settings(
        web_cache_ttl_s=overrides.pop("web_cache_ttl_s", 600.0),
        profile="default",
        profile_layout=profile_layout,
        workspace_root=str(workspace),
        **overrides,
    )


def _outcome(*, status=200, content_type="application/json", body=b"{}", truncated=False, url=None):
    return wf.FetchOutcome(
        status=status,
        final_url=url or PUBLIC_URL,
        content_type=content_type,
        body=body,
        truncated=truncated,
        transport="fake",
        elapsed_ms=12,
    )


class _FakeTransport:
    """Replays a scripted sequence of outcomes or raised failures."""

    name = "fake"

    def __init__(self, *script):
        self._script = list(script)
        self.requests: list[wf.FetchRequest] = []

    def available(self) -> bool:
        return True

    async def fetch(self, request):
        self.requests.append(request)
        item = self._script.pop(0) if len(self._script) > 1 else self._script[0]
        if isinstance(item, Exception):
            raise item
        return item


def _install(monkeypatch, transport, **settings_overrides):
    monkeypatch.setattr(wf, "transports_for", lambda preference: (transport,))
    monkeypatch.setattr(wf, "_settings", lambda: _settings(**settings_overrides))


def _run(params):
    return asyncio.run(wf.web_fetch(params))


# ── target classification (SSRF surface) ─────────────────────────────

def test_classify_rejects_non_http_schemes() -> None:
    """A fetch tool that could open file:// would bypass the workspace boundary."""
    for url in ("file:///etc/passwd", "gopher://x/1", "data:text/plain,hi"):
        with pytest.raises(UrlRejected) as excinfo:
            asyncio.run(classify_url(url, resolve=False))
        assert excinfo.value.reason == "unsupported_scheme"


def test_classify_literal_addresses_by_category() -> None:
    cases = {
        "http://127.0.0.1/x": "loopback",
        "http://10.1.2.3/x": "private",
        "http://192.168.1.1/x": "private",
        "http://169.254.169.254/x": "metadata",
        "http://100.100.100.200/x": "metadata",
        "http://169.254.10.10/x": "link_local",
        "http://0.0.0.0/x": "unspecified",
        "https://93.184.216.34/x": "public",
        "http://[::1]/x": "loopback",
        "http://[fd00:ec2::254]/x": "metadata",
        "https://[2606:4700::1111]/x": "public",
        # Shared address space (CGNAT) is not RFC1918 private but is not globally
        # routable either; treating it as public would expose a whole class of
        # internal endpoints.
        "http://100.64.0.1/x": "reserved",
        "http://198.18.0.1/x": "private",
    }
    for url, expected in cases.items():
        target = asyncio.run(classify_url(url, resolve=False))
        assert target.category == expected, url
        assert target.is_internal is (expected != "public"), url


def test_classify_extracts_origin_and_credentials() -> None:
    target = asyncio.run(classify_url("https://user:pw@93.184.216.34:8443/p?q=1", resolve=False))
    assert target.origin == "https://93.184.216.34:8443"
    assert target.has_credentials is True
    # Default ports stay out of the origin so grants match the common spelling.
    plain = asyncio.run(classify_url("https://93.184.216.34/p", resolve=False))
    assert plain.origin == "https://93.184.216.34"


# ── risk classification: approval, never a silent block ──────────────

def test_public_read_is_low_enough_to_skip_prompting() -> None:
    """The common path must not prompt, or the tool is worse than shell."""
    target = asyncio.run(classify_url(PUBLIC_URL, resolve=False))
    action = ActionDescriptor.network_fetch(
        target.url, origin=target.origin, metadata=target.to_metadata()
    )
    risk = DefaultRiskClassifier().assess(action)
    assert risk.level == RiskLevel.LOW
    # policy asks whenever level is HIGH/MEDIUM or score >= 0.35.
    assert risk.score < 0.35


def test_internal_targets_ask_and_are_never_hardline() -> None:
    """Internal targets are gated by approval, not refused outright.

    CRITICAL or hardline would make policy deny the action with no prompt, which
    is not the configured behavior: the user decides.
    """
    for url in (LOOPBACK_URL, METADATA_URL, "http://10.0.0.5/admin"):
        target = asyncio.run(classify_url(url, resolve=False))
        action = ActionDescriptor.network_fetch(
            target.url, origin=target.origin, metadata=target.to_metadata()
        )
        risk = DefaultRiskClassifier().assess(action)
        assert risk.level == RiskLevel.HIGH, url
        assert risk.hardline is False, url
        assert risk.allow_permanent is False, url
        assert risk.explanation, url


def test_credentialed_url_is_high_risk() -> None:
    target = asyncio.run(classify_url("https://u:p@93.184.216.34/x", resolve=False))
    action = ActionDescriptor.network_fetch(
        target.url, origin=target.origin, metadata=target.to_metadata()
    )
    risk = DefaultRiskClassifier().assess(action)
    assert risk.level == RiskLevel.HIGH
    assert "url_embedded_credentials" in risk.reasons


def test_grant_signature_is_scoped_to_origin() -> None:
    """One approval must cover a host, not a single URL.

    Keying the grant on the full URL would re-prompt on every path and query,
    which turns progressive trust into an unusable prompt loop.
    """
    first = ActionDescriptor.network_fetch(
        "https://example.test/a?x=1", origin="https://example.test"
    )
    second = ActionDescriptor.network_fetch(
        "https://example.test/b/c?y=2", origin="https://example.test"
    )
    other = ActionDescriptor.network_fetch(
        "https://other.test/a", origin="https://other.test"
    )
    assert first.signature() == second.signature()
    assert first.signature() != other.signature()
    assert first.kind == ActionKind.NETWORK_FETCH.value


# ── extraction ───────────────────────────────────────────────────────

def test_kind_routing_follows_content_type_header() -> None:
    assert kind_for_content_type("application/json; charset=utf-8") == KIND_JSON
    assert kind_for_content_type("application/vnd.api+json") == KIND_JSON
    assert kind_for_content_type("text/html;charset=gbk") == KIND_HTML
    assert kind_for_content_type("text/markdown") == KIND_TEXT
    assert kind_for_content_type("application/pdf") == KIND_BINARY
    assert kind_for_content_type("image/png") == KIND_BINARY


def test_stdlib_extractor_drops_chrome_and_keeps_structure() -> None:
    html = """
    <html><head><title>Quarterly Report</title><style>.x{color:red}</style></head>
    <body>
      <nav><a href="/home">Home</a></nav>
      <script>var tracking = 1;</script>
      <h1>Revenue</h1>
      <p>Revenue grew 12% this quarter.</p>
      <ul><li>Cloud up</li><li>Ads flat</li></ul>
      <a href="/detail/q3">Full breakdown</a>
      <footer><a href="/legal">Legal</a></footer>
    </body></html>
    """
    result = StdlibHtmlExtractor().extract(html, url="https://site.test/report")
    assert result is not None
    assert result.title == "Quarterly Report"
    assert "Revenue grew 12% this quarter." in result.text
    assert "# Revenue" in result.text
    assert "- Cloud up" in result.text
    # Chrome and code must not leak into the reading.
    assert "tracking" not in result.text
    assert "color:red" not in result.text
    urls = [url for _, url in result.links]
    assert "https://site.test/detail/q3" in urls
    assert not any(url.endswith("/home") or url.endswith("/legal") for url in urls)


def test_select_path_walks_dicts_and_list_indices() -> None:
    payload = {"chart": {"result": [{"meta": {"price": 128.99}}]}}
    value, error = select_path(payload, "chart.result.0.meta.price")
    assert error == ""
    assert value == 128.99


def test_select_path_errors_name_the_failing_segment() -> None:
    payload = {"chart": {"result": [{"meta": {"price": 128.99}}]}}
    _, missing_key = select_path(payload, "chart.results")
    assert "chart.results" in missing_key and "available keys" in missing_key
    _, bad_index = select_path(payload, "chart.result.9")
    assert "index out of range" in bad_index
    _, not_indexable = select_path(payload, "chart.result.0.meta.price.deeper")
    assert "not indexable" in not_indexable
    _, bad_list_key = select_path(payload, "chart.result.first")
    assert "expected a list index" in bad_list_key


def test_decode_json_reports_position_and_prefix() -> None:
    data, error = decode_json("Edge: Too Many Requests")
    assert data is None
    assert "did not parse" in error
    assert "Edge: Too Many Requests" in error


# ── tool behavior ────────────────────────────────────────────────────

def test_json_fetch_returns_selected_value(monkeypatch) -> None:
    body = b'{"chart":{"result":[{"meta":{"regularMarketPrice":128.99}}]}}'
    transport = _FakeTransport(_outcome(body=body))
    _install(monkeypatch, transport)

    result = _run({"url": PUBLIC_URL, "select": "chart.result.0.meta.regularMarketPrice"})

    assert result["ok"] is True
    assert result["status"] == 200
    assert result["kind"] == KIND_JSON
    assert result["data"] == 128.99
    assert result["select"] == "chart.result.0.meta.regularMarketPrice"


def test_rate_limited_response_reports_status_not_a_traceback(monkeypatch) -> None:
    """The regression this tool exists to prevent.

    Fetching through a shell pipeline turned a 429 with a non-JSON body into a
    Python traceback from `json.load`. Here the status is the answer, the body is
    quoted as evidence, and the failure is marked retryable.
    """
    transport = _FakeTransport(
        _outcome(status=429, content_type="text/plain", body=b"Edge: Too Many Requests")
    )
    _install(monkeypatch, transport)

    result = _run({"url": PUBLIC_URL})

    assert result["ok"] is False
    assert result["status"] == 429
    assert result["error_type"] == "http_error"
    assert result["retryable"] is True
    assert "Edge: Too Many Requests" in result["body_excerpt"]
    assert "Traceback" not in result["error"]


def test_declared_json_that_does_not_parse_is_a_content_error(monkeypatch) -> None:
    transport = _FakeTransport(
        _outcome(status=200, content_type="application/json", body=b"<html>nope</html>")
    )
    _install(monkeypatch, transport)

    result = _run({"url": PUBLIC_URL})

    assert result["ok"] is False
    assert result["error_type"] == "invalid_json"
    assert result["status"] == 200
    assert "<html>nope</html>" in result["body_excerpt"]


def test_bad_selector_lists_available_keys(monkeypatch) -> None:
    transport = _FakeTransport(_outcome(body=b'{"a":1,"b":2}'))
    _install(monkeypatch, transport)

    result = _run({"url": PUBLIC_URL, "select": "missing.key"})

    assert result["ok"] is False
    assert result["error_type"] == "invalid_selector"
    assert result["retryable"] is True
    assert result["available_top_level_keys"] == ["a", "b"]


def test_html_fetch_extracts_text_and_links(monkeypatch) -> None:
    html = b"<html><head><title>T</title></head><body><p>Hello world</p>" \
           b'<a href="https://site.test/next">Next</a></body></html>'
    transport = _FakeTransport(_outcome(content_type="text/html", body=html))
    _install(monkeypatch, transport)

    result = _run({"url": PUBLIC_URL})

    assert result["ok"] is True
    assert result["kind"] == KIND_HTML
    assert result["title"] == "T"
    assert "Hello world" in result["text"]
    assert result["links"][0]["url"] == "https://site.test/next"
    assert result["extractor"] == "stdlib"


def test_raw_text_fetch_honours_the_approved_encoding_override(monkeypatch) -> None:
    body = "中文报价".encode("gb18030")
    transport = _FakeTransport(_outcome(content_type="text/plain", body=body))
    _install(monkeypatch, transport)

    result = _run(
        {"url": PUBLIC_URL, "extract": "raw_text", "encoding": "gb18030"}
    )

    assert result["ok"] is True
    assert result["text"] == "中文报价"


def test_raw_text_fetch_rejects_arbitrary_encoding(monkeypatch) -> None:
    transport = _FakeTransport(_outcome(content_type="text/plain", body=b"unused"))
    _install(monkeypatch, transport)

    result = _run({"url": PUBLIC_URL, "extract": "raw_text", "encoding": "utf-16"})

    assert result["ok"] is False
    assert result["error_type"] == "invalid_encoding"
    assert transport.requests == []


def test_binary_content_is_not_returned_inline(monkeypatch) -> None:
    transport = _FakeTransport(
        _outcome(content_type="application/pdf", body=b"%PDF-1.7 binary...")
    )
    _install(monkeypatch, transport)

    result = _run({"url": PUBLIC_URL})

    assert result["kind"] == KIND_BINARY
    assert result["text"] == ""
    assert "not returned inline" in result["note"]


def test_retryable_status_is_retried_then_succeeds(monkeypatch) -> None:
    transport = _FakeTransport(
        _outcome(status=503, content_type="text/plain", body=b"busy"),
        _outcome(status=200, content_type="text/plain", body=b"ready"),
    )
    _install(monkeypatch, transport, web_max_retries=1)
    # Collapse the backoff instead of patching asyncio.sleep: the retry loop is
    # what is under test, not the timer.
    monkeypatch.setattr(wf, "_retry_delay", lambda attempt: 0.0)

    result = _run({"url": PUBLIC_URL})

    assert result["ok"] is True
    assert result["text"] == "ready"
    assert len(transport.requests) == 2


def test_transport_failure_is_typed_and_retryable(monkeypatch) -> None:
    transport = _FakeTransport(wf.TransportFailure("timeout", "timed out", retryable=True))
    _install(monkeypatch, transport)

    result = _run({"url": PUBLIC_URL})

    assert result["ok"] is False
    assert result["error_type"] == "timeout"
    assert result["retryable"] is True


def test_client_rejection_fails_over_to_the_next_transport(monkeypatch) -> None:
    """A 429 from one stack must try the other before reporting failure.

    Observed against a real CDN edge: the same URL with the same user agent answers
    429 intermittently, and which transport gets refused changes between runs. One
    extra attempt on the other transport is therefore worth it — without it the
    model is pushed back to running curl through the shell, which is exactly what
    this tool exists to replace.
    """
    refused = _FakeTransport(_outcome(status=429, content_type="text/html", body=b"Too Many Requests"))
    accepted = _FakeTransport(_outcome(status=200, body=b'{"price": 128.99}'))
    monkeypatch.setattr(wf, "transports_for", lambda preference: (refused, accepted))
    monkeypatch.setattr(wf, "_settings", lambda: _settings())

    result = _run({"url": PUBLIC_URL, "select": "price"})

    assert result["ok"] is True
    assert result["data"] == 128.99
    assert len(refused.requests) == 1
    assert len(accepted.requests) == 1


def test_rejection_by_every_transport_reports_the_last_status(monkeypatch) -> None:
    first = _FakeTransport(_outcome(status=403, content_type="text/plain", body=b"denied"))
    second = _FakeTransport(_outcome(status=429, content_type="text/plain", body=b"slow down"))
    monkeypatch.setattr(wf, "transports_for", lambda preference: (first, second))
    monkeypatch.setattr(wf, "_settings", lambda: _settings())

    result = _run({"url": PUBLIC_URL})

    assert result["ok"] is False
    assert result["status"] == 429
    assert result["error_type"] == "http_error"
    assert "slow down" in result["body_excerpt"]


def test_ordinary_error_status_does_not_fail_over(monkeypatch) -> None:
    """A 404 is the answer; trying another stack would only waste a request."""
    first = _FakeTransport(_outcome(status=404, content_type="text/plain", body=b"nope"))
    second = _FakeTransport(_outcome(status=200, body=b"{}"))
    monkeypatch.setattr(wf, "transports_for", lambda preference: (first, second))
    monkeypatch.setattr(wf, "_settings", lambda: _settings())

    result = _run({"url": PUBLIC_URL})

    assert result["status"] == 404
    assert second.requests == []


def test_failure_in_first_transport_falls_through(monkeypatch) -> None:
    broken = _FakeTransport(wf.TransportFailure("tls_error", "bad cert", retryable=False))
    working = _FakeTransport(_outcome(status=200, content_type="text/plain", body=b"ok"))
    monkeypatch.setattr(wf, "transports_for", lambda preference: (broken, working))
    monkeypatch.setattr(wf, "_settings", lambda: _settings())

    result = _run({"url": PUBLIC_URL})

    assert result["ok"] is True
    assert result["text"] == "ok"


def test_unsupported_scheme_is_refused_before_any_request(monkeypatch) -> None:
    transport = _FakeTransport(_outcome())
    _install(monkeypatch, transport)

    result = _run({"url": "file:///etc/passwd"})

    assert result["ok"] is False
    assert result["error_type"] == "unsupported_scheme"
    assert result["retryable"] is False
    assert transport.requests == []


def test_internal_target_requires_approval_and_fails_closed(monkeypatch) -> None:
    """With no gate installed, an internal target must not be reachable."""
    transport = _FakeTransport(_outcome())
    _install(monkeypatch, transport)
    monkeypatch.setattr(wf, "_approval_gate", None)

    result = _run({"url": LOOPBACK_URL})

    assert result["ok"] is False
    assert result["error_type"] == "blocked_target"
    assert result["requires_approval"] is True
    assert result["target_category"] == "loopback"
    assert transport.requests == []


def test_internal_target_proceeds_once_approved(monkeypatch) -> None:
    transport = _FakeTransport(_outcome(content_type="text/plain", body=b"internal ok"))
    _install(monkeypatch, transport)
    seen: list[ActionDescriptor] = []

    class _Gate:
        async def evaluate(self, action):
            seen.append(action)
            return SimpleNamespace(approved=True)

    monkeypatch.setattr(wf, "_approval_gate", _Gate())

    result = _run({"url": LOOPBACK_URL})

    assert result["ok"] is True
    assert result["text"] == "internal ok"
    assert seen[0].kind == ActionKind.NETWORK_FETCH.value
    assert seen[0].resource == "http://127.0.0.1:8765"


def test_denied_approval_blocks_the_fetch(monkeypatch) -> None:
    transport = _FakeTransport(_outcome())
    _install(monkeypatch, transport)

    class _Gate:
        async def evaluate(self, action):
            return SimpleNamespace(approved=False, denial_message="user said no")

    monkeypatch.setattr(wf, "_approval_gate", _Gate())

    result = _run({"url": METADATA_URL})

    assert result["ok"] is False
    assert result["error"] == "user said no"
    assert transport.requests == []


def test_broken_gate_denies_rather_than_opening(monkeypatch) -> None:
    transport = _FakeTransport(_outcome())
    _install(monkeypatch, transport)

    class _Gate:
        async def evaluate(self, action):
            raise RuntimeError("gate exploded")

    monkeypatch.setattr(wf, "_approval_gate", _Gate())

    result = _run({"url": LOOPBACK_URL})

    assert result["ok"] is False
    assert "could not be consulted" in result["error"]
    assert transport.requests == []


def test_private_targets_deny_mode_skips_the_prompt(monkeypatch) -> None:
    """Operators running unattended need a hard refusal without a prompt."""
    transport = _FakeTransport(_outcome())
    _install(monkeypatch, transport, web_private_targets="deny")

    class _Gate:
        async def evaluate(self, action):  # pragma: no cover - must not be reached
            raise AssertionError("deny mode must not consult the gate")

    monkeypatch.setattr(wf, "_approval_gate", _Gate())

    result = _run({"url": LOOPBACK_URL})

    assert result["ok"] is False
    assert result["error_type"] == "blocked_target"
    assert result.get("requires_approval") is None
    assert transport.requests == []


def test_private_targets_allow_mode_needs_no_gate(monkeypatch) -> None:
    transport = _FakeTransport(_outcome(content_type="text/plain", body=b"dashboard"))
    _install(monkeypatch, transport, web_private_targets="allow")
    monkeypatch.setattr(wf, "_approval_gate", None)

    result = _run({"url": LOOPBACK_URL})

    assert result["ok"] is True
    assert result["text"] == "dashboard"


def test_browser_user_agent_is_sent_and_overridable(monkeypatch) -> None:
    """Default UA is browser-style because many CDNs reject library agents."""
    transport = _FakeTransport(_outcome())
    _install(monkeypatch, transport)
    _run({"url": PUBLIC_URL})
    assert transport.requests[0].user_agent == wf.DEFAULT_USER_AGENT
    assert "Mozilla/5.0" in wf.DEFAULT_USER_AGENT

    custom = _FakeTransport(_outcome())
    _install(monkeypatch, custom, web_user_agent="LeapFlow/test")
    _run({"url": PUBLIC_URL})
    assert custom.requests[0].user_agent == "LeapFlow/test"


def test_limits_come_from_settings_and_params(monkeypatch) -> None:
    transport = _FakeTransport(_outcome())
    _install(monkeypatch, transport, web_timeout_s=7.0, web_max_bytes=4096)
    _run({"url": PUBLIC_URL})
    assert transport.requests[0].timeout_s == 7.0
    assert transport.requests[0].max_bytes == 4096

    override = _FakeTransport(_outcome())
    _install(monkeypatch, override, web_timeout_s=7.0)
    _run({"url": PUBLIC_URL, "timeout": 3, "max_bytes": 128})
    assert override.requests[0].timeout_s == 3
    assert override.requests[0].max_bytes == 128


def test_missing_url_is_a_retryable_argument_error() -> None:
    result = _run({})
    assert result["ok"] is False
    assert result["retryable"] is True


def test_no_transport_available_is_reported_clearly(monkeypatch) -> None:
    monkeypatch.setattr(wf, "transports_for", lambda preference: ())
    monkeypatch.setattr(wf, "_settings", lambda: _settings())

    result = _run({"url": PUBLIC_URL})

    assert result["ok"] is False
    assert result["error_type"] == "transport_unavailable"
    assert "httpx" in result["error"]


def _redirect(location: str, *, status: int = 302):
    return wf.FetchOutcome(
        status=status,
        final_url=PUBLIC_URL,
        content_type="text/html",
        body=b"",
        truncated=False,
        transport="fake",
        elapsed_ms=3,
        location=location,
    )


# ── redirects: every hop must be re-gated ───────────────────────

def test_redirect_to_internal_target_is_gated(monkeypatch) -> None:
    """A public URL must not be able to bounce the request into the network.

    If the HTTP client follows redirects itself, the egress gate only ever sees
    the first hop, and any server able to answer 302 can read loopback services
    or cloud instance metadata on the agent's behalf.
    """
    transport = _FakeTransport(
        _redirect("http://169.254.169.254/latest/meta-data/iam/security-credentials/"),
        _outcome(content_type="text/plain", body=b"AWS_SECRET"),
    )
    _install(monkeypatch, transport)
    monkeypatch.setattr(wf, "_approval_gate", None)

    result = _run({"url": PUBLIC_URL})

    assert result["ok"] is False
    assert result["error_type"] == "blocked_target"
    assert result["target_category"] == "metadata"
    assert "Redirected to" in result["error"]
    assert result["blocked_url"].startswith("http://169.254.169.254/")
    # The second hop must never have been issued.
    assert len(transport.requests) == 1


def test_redirect_to_loopback_is_gated(monkeypatch) -> None:
    transport = _FakeTransport(
        _redirect("http://127.0.0.1:8765/admin"),
        _outcome(content_type="text/plain", body=b"internal"),
    )
    _install(monkeypatch, transport)
    monkeypatch.setattr(wf, "_approval_gate", None)

    result = _run({"url": PUBLIC_URL})

    assert result["ok"] is False
    assert result["target_category"] == "loopback"
    assert len(transport.requests) == 1


def test_public_redirect_is_followed_and_reported(monkeypatch) -> None:
    """Ordinary redirects still work, with the chain visible."""
    transport = _FakeTransport(
        _redirect("https://93.184.216.34/moved"),
        _outcome(body=b'{"price": 7}'),
    )
    _install(monkeypatch, transport)

    result = _run({"url": PUBLIC_URL, "select": "price"})

    assert result["ok"] is True
    assert result["data"] == 7
    assert result["redirect_chain"] == [PUBLIC_URL, "https://93.184.216.34/moved"]
    assert len(transport.requests) == 2


def test_relative_redirect_is_resolved(monkeypatch) -> None:
    transport = _FakeTransport(
        _redirect("/elsewhere"),
        _outcome(content_type="text/plain", body=b"landed"),
    )
    _install(monkeypatch, transport)

    result = _run({"url": PUBLIC_URL})

    assert result["ok"] is True
    assert transport.requests[1].url == "https://93.184.216.34/elsewhere"


def test_redirect_loop_is_reported(monkeypatch) -> None:
    transport = _FakeTransport(_redirect(PUBLIC_URL))
    _install(monkeypatch, transport)

    result = _run({"url": PUBLIC_URL})

    assert result["ok"] is False
    assert result["error_type"] == "redirect_loop"


def test_redirect_budget_is_bounded(monkeypatch) -> None:
    """An endless chain of distinct hops must stop at the configured budget."""
    hops = iter(range(1, 50))

    class _Chain:
        name = "chain"

        def __init__(self) -> None:
            self.requests: list[wf.FetchRequest] = []

        def available(self) -> bool:
            return True

        async def fetch(self, request):
            self.requests.append(request)
            return _redirect(f"https://93.184.216.34/hop{next(hops)}")

    transport = _Chain()
    monkeypatch.setattr(wf, "transports_for", lambda preference: (transport,))
    monkeypatch.setattr(wf, "_settings", lambda: _settings(web_max_redirects=3))

    result = _run({"url": PUBLIC_URL})

    assert result["ok"] is False
    assert result["error_type"] == "too_many_redirects"
    assert len(transport.requests) == 4  # initial request + 3 redirects


def test_transports_do_not_follow_redirects_themselves() -> None:
    """Auto-follow in the client would bypass the per-hop gate entirely."""
    import inspect

    httpx_source = inspect.getsource(wf.HttpxTransport.fetch)
    assert "follow_redirects=False" in httpx_source

    curl_source = inspect.getsource(wf.CurlTransport.fetch)
    assert '"--location"' not in curl_source
    assert "redirect_url" in curl_source


# ── caching ───────────────────────────────────────────────

def test_second_fetch_is_served_from_cache(monkeypatch, tmp_path) -> None:
    """A repeated read must not re-hit the network within its TTL."""
    transport = _FakeTransport(_outcome(body=b'{"price": 1}'))
    settings = _settings_with_cache(tmp_path)
    monkeypatch.setattr(wf, "transports_for", lambda preference: (transport,))
    monkeypatch.setattr(wf, "_settings", lambda: settings)

    first = _run({"url": PUBLIC_URL, "select": "price"})
    second = _run({"url": PUBLIC_URL, "select": "price"})

    assert first["ok"] is True and first.get("from_cache") is None
    assert second["ok"] is True and second["from_cache"] is True
    assert second["data"] == 1
    assert len(transport.requests) == 1


def test_cache_entry_is_session_scoped_and_not_syncable(monkeypatch, tmp_path) -> None:
    """Fetched content is reproducible and may be per-session: never sync it out."""
    from leapflow.cache.manager import CacheManager

    transport = _FakeTransport(_outcome(body=b'{"a": 1}'))
    settings = _settings_with_cache(tmp_path)
    monkeypatch.setattr(wf, "transports_for", lambda preference: (transport,))
    monkeypatch.setattr(wf, "_settings", lambda: settings)

    _run({"url": PUBLIC_URL})

    manager = CacheManager(settings.profile_layout.cache, profile_id="default")
    entries = [e for e in manager.list_entries() if e.owner_component == "web_fetch"]
    assert entries, "fetch body was not indexed"
    entry = entries[0]
    assert entry.scope == "session"
    assert entry.syncable is False
    assert entry.category == "web_fetch"
    assert entry.expires_at is not None


def test_zero_ttl_disables_caching(monkeypatch, tmp_path) -> None:
    transport = _FakeTransport(_outcome(body=b"{}"), _outcome(body=b"{}"))
    settings = _settings_with_cache(tmp_path, web_cache_ttl_s=0.0)
    monkeypatch.setattr(wf, "transports_for", lambda preference: (transport,))
    monkeypatch.setattr(wf, "_settings", lambda: settings)

    _run({"url": PUBLIC_URL})
    second = _run({"url": PUBLIC_URL})

    assert second.get("from_cache") is None
    assert len(transport.requests) == 2


def test_binary_content_is_saved_and_referenced_by_path(monkeypatch, tmp_path) -> None:
    """Binary stays out of the transcript but must not become a dead end."""
    transport = _FakeTransport(
        _outcome(content_type="application/pdf", body=b"%PDF-1.7 payload")
    )
    settings = _settings_with_cache(tmp_path)
    monkeypatch.setattr(wf, "transports_for", lambda preference: (transport,))
    monkeypatch.setattr(wf, "_settings", lambda: settings)

    result = _run({"url": PUBLIC_URL})

    assert result["kind"] == KIND_BINARY
    assert result["text"] == ""
    stored = Path(result["cache_path"])
    assert stored.read_bytes() == b"%PDF-1.7 payload"


def test_cache_is_consulted_only_after_the_egress_gate(monkeypatch, tmp_path) -> None:
    """A cached body must not become a bypass for an unapproved internal target."""
    transport = _FakeTransport(_outcome(content_type="text/plain", body=b"internal"))
    settings = _settings_with_cache(tmp_path)
    monkeypatch.setattr(wf, "transports_for", lambda preference: (transport,))
    monkeypatch.setattr(wf, "_settings", lambda: settings)

    class _Gate:
        def __init__(self) -> None:
            self.approve = True

        async def evaluate(self, action):
            return SimpleNamespace(approved=self.approve, denial_message="denied now")

    gate = _Gate()
    monkeypatch.setattr(wf, "_approval_gate", gate)

    first = _run({"url": LOOPBACK_URL})
    assert first["ok"] is True

    # Revoking approval must block the second read even though it is cached.
    gate.approve = False
    second = _run({"url": LOOPBACK_URL})
    assert second["ok"] is False
    assert second["error_type"] == "blocked_target"


def test_approval_detail_never_carries_query_secrets(monkeypatch) -> None:
    """The approval prompt and audit log persist the action detail.

    A query string routinely holds an API key or signed token, so origin+path is
    the most that may be written down.
    """
    transport = _FakeTransport(_outcome(content_type="text/plain", body=b"ok"))
    _install(monkeypatch, transport)
    seen: list[ActionDescriptor] = []

    class _Gate:
        async def evaluate(self, action):
            seen.append(action)
            return SimpleNamespace(approved=True)

    monkeypatch.setattr(wf, "_approval_gate", _Gate())

    _run({"url": "http://127.0.0.1:8765/admin?token=SUPERSECRET&x=1#frag"})

    action = seen[0]
    assert "SUPERSECRET" not in action.detail
    assert "SUPERSECRET" not in json.dumps(action.metadata)
    assert action.detail == "http://127.0.0.1:8765/admin"
    assert action.resource == "http://127.0.0.1:8765"


def test_credentialed_url_is_not_echoed_into_approval(monkeypatch) -> None:
    transport = _FakeTransport(_outcome(content_type="text/plain", body=b"ok"))
    _install(monkeypatch, transport)
    seen: list[ActionDescriptor] = []

    class _Gate:
        async def evaluate(self, action):
            seen.append(action)
            return SimpleNamespace(approved=True)

    monkeypatch.setattr(wf, "_approval_gate", _Gate())

    _run({"url": "https://user:hunter2@93.184.216.34/private"})

    action = seen[0]
    assert "hunter2" not in action.detail
    assert "hunter2" not in json.dumps(action.metadata)


def test_failure_evidence_keeps_status_and_body(monkeypatch) -> None:
    """A compacted HTTP failure must still explain itself to the model."""
    from leapflow.engine.context_control import ToolEvidenceBuilder

    failure = {
        "ok": False,
        "status": 429,
        "error": "HTTP 429 from https://api.test",
        "error_type": "http_error",
        "retryable": True,
        "body_excerpt": "Edge: Too Many Requests",
    }
    evidence = ToolEvidenceBuilder(max_content_chars=400).build("web_fetch", {}, failure)

    assert evidence["ok"] is False
    assert evidence["status"] == 429
    assert evidence["retryable"] is True
    assert "Edge: Too Many Requests" in evidence["body_excerpt"]


# ── loop integration contracts ───────────────────────────────────────

def test_web_fetch_is_read_only_for_the_execution_ledger() -> None:
    """read_only is the whole point: retries stay safe and batches keep running."""
    from leapflow.engine.engine import _SIDE_EFFECT_STOP_POLICIES, _default_tool_registry
    from leapflow.engine.tool_execution import (
        effect_is_uncertain_on_failure,
        execution_policy_for,
    )

    registry = _default_tool_registry()
    policy = execution_policy_for("web_fetch", registry.specs.get("web_fetch"))
    assert policy == "read_only"
    assert effect_is_uncertain_on_failure(policy) is False
    assert policy not in _SIDE_EFFECT_STOP_POLICIES


def test_web_fetch_is_disclosed_in_the_core_tier() -> None:
    """A network capability the model cannot see is why it fell back to shell."""
    from leapflow.engine.context_disclosure import DisclosurePlanner, DisclosureRuntimeState
    from leapflow.plugins import get_registry
    _tool_reg = get_registry()
    TOOL_DEFINITIONS = _tool_reg.tool_definitions
    TOOL_HANDLERS = _tool_reg.tool_handlers

    plan = DisclosurePlanner().plan(
        TOOL_DEFINITIONS, DisclosureRuntimeState(native_tools_enabled=True)
    )
    names = [item["function"]["name"] for item in plan.tool_definitions]
    assert "web_fetch" in names
    assert "web_fetch" in TOOL_HANDLERS


def test_evidence_builder_caps_fetched_bodies() -> None:
    from leapflow.engine.context_control import ToolEvidenceBuilder

    builder = ToolEvidenceBuilder(max_content_chars=400)
    result = {
        "ok": True,
        "status": 200,
        "url": PUBLIC_URL,
        "final_url": PUBLIC_URL,
        "content_type": "text/html",
        "title": "Big page",
        "text": "HEAD" + ("x" * 20_000) + "TAIL",
        "links": [{"text": f"l{i}", "url": f"https://site.test/{i}"} for i in range(50)],
        "extractor": "stdlib",
    }
    evidence = builder.build("web_fetch", {"url": PUBLIC_URL}, result)

    assert evidence["kind"] == "web_fetch_evidence"
    assert evidence["status"] == 200
    assert evidence["title"] == "Big page"
    assert len(evidence["text"]) < 1000
    assert "HEAD" in evidence["text"] and "TAIL" in evidence["text"]
    assert len(evidence["links"]) <= 50


def test_web_config_keys_are_user_visible() -> None:
    """Every durable limit must be reachable from `leap config` / `/config`."""
    from leapflow.config import get_settings
    from leapflow.config_service import ConfigService

    keys = set(ConfigService(get_settings()).writable_keys())
    for key in (
        "web.transport",
        "web.timeout_s",
        "web.max_bytes",
        "web.max_retries",
        "web.user_agent",
        "web.private_targets",
        "web.extractor",
    ):
        assert key in keys, key
