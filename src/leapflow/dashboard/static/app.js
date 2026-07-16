// LeapFlow dashboard: a minimal Server-Driven UI renderer.
// Fetches a ViewSpec from /api/view, renders the fixed component catalog into
// the DOM, connects a WebSocket for live monitor events, and posts interactive
// actions back to /api/action.
(function () {
  "use strict";

  const params = new URLSearchParams(location.search);
  const TOKEN = params.get("token") || "";
  const rootEl = document.getElementById("root");
  const statusEl = document.getElementById("status");
  const toastsEl = document.getElementById("toasts");
  let current = { action: params.get("action") || "home", target: params.get("target") || "" };

  function api(path) {
    const url = new URL(path, location.origin);
    url.searchParams.set("token", TOKEN);
    return url.toString();
  }

  async function fetchView(intent) {
    current = Object.assign({}, current, intent || {});
    const url = new URL("/api/view", location.origin);
    url.searchParams.set("token", TOKEN);
    Object.entries(current).forEach(([k, v]) => v && url.searchParams.set(k, v));
    try {
      const resp = await fetch(url.toString());
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      render(await resp.json());
    } catch (err) {
      rootEl.innerHTML = '<div class="empty">Failed to load view: ' + esc(String(err)) + "</div>";
    }
  }

  async function postAction(action) {
    if (action && action.kind === "nav") { handleNav(action); return; }
    try {
      const resp = await fetch(api("/api/action"), {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-Dashboard-Token": TOKEN },
        body: JSON.stringify(action),
      });
      const result = await resp.json();
      if (action.kind === "rpc") fetchView(); // reflect control changes
      return result;
    } catch (err) {
      toast({ title: "Action failed", summary: String(err), severity: "alert" });
    }
  }

  // nav actions are purely client-side (no server round-trip).
  function handleNav(action) {
    const p = action.params || {};
    if (action.name === "openWatch" && p.target) fetchView({ action: "open", target: p.target });
    else if (action.name === "openLink" && p.url) window.open(p.url, "_blank", "noopener");
  }

  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"]/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }

  // ── Renderers keyed by catalog type; unknown types fall back to text ──
  function el(tag, cls, html) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  }

  function renderChildren(node, parent) {
    (node.children || []).forEach((c) => parent.appendChild(renderNode(c)));
    return parent;
  }

  function bindAction(dom, node) {
    if (node.action) {
      dom.style.cursor = "pointer";
      dom.addEventListener("click", (ev) => { ev.stopPropagation(); postAction(node.action); });
    }
    return dom;
  }

  // Escape-hatch renderers for the `Custom` component, keyed by props.render.
  const CUSTOM_RENDERERS = {
    candlestick: (p) => { const data = Array.isArray(p.data) ? p.data : [];
      const d = el("div", "card"); d.appendChild(el("div", "card-title", "Candlestick"));
      d.appendChild(el("div", "md", esc(data.length + " series"))); return d; },
    gauge: (p) => { const d = el("div", "stat"); d.appendChild(el("div", "label", "Gauge"));
      d.appendChild(el("div", "value", esc(p.data != null ? p.data : "\u2014"))); return d; },
  };

  const RENDERERS = {
    Page: (n) => { const d = el("div", "page");
      const t = (n.props && n.props.title); if (t) d.appendChild(el("div", "page-title", esc(t)));
      return renderChildren(n, d); },
    Section: (n) => renderChildren(n, el("div", "section")),
    Grid: (n) => renderChildren(n, el("div", "grid")),
    Row: (n) => renderChildren(n, el("div", "row")),
    Col: (n) => renderChildren(n, el("div", "col")),
    Card: (n) => { const d = el("div", "card");
      const t = n.props && n.props.title; if (t) d.appendChild(el("div", "card-title", esc(t)));
      return renderChildren(n, d); },
    Board: (n) => { const d = el("div", "board");
      const t = n.props && n.props.title; if (t) d.appendChild(el("div", "board-title", esc(t)));
      return renderChildren(n, d); },
    Toolbar: (n) => renderChildren(n, el("div", "toolbar")),
    Stat: (n) => { const p = n.props || {}; const d = el("div", "stat");
      d.appendChild(el("div", "label", esc(p.label))); d.appendChild(el("div", "value", esc(p.value)));
      return d; },
    Markdown: (n) => el("div", "md", esc((n.props || {}).text)),
    StoryPanel: (n) => { const p = n.props || {}; const d = el("div", "card");
      d.appendChild(el("div", "card-title", esc(p.title || "Story")));
      d.appendChild(el("div", "md", esc(p.text))); return d; },
    List: (n) => { const items = ((n.props || {}).data) || []; const ul = el("ul");
      (Array.isArray(items) ? items : []).forEach((it) =>
        ul.appendChild(el("li", null, esc(typeof it === "object" ? JSON.stringify(it) : it))));
      return ul; },
    SuggestionChips: (n) => { const items = ((n.props || {}).data) || []; const d = el("div", "row");
      (Array.isArray(items) ? items : []).forEach((it) => d.appendChild(el("button", null, esc(it))));
      return d; },
    Gauge: (n) => { const p = n.props || {}; const d = el("div", "stat");
      d.appendChild(el("div", "label", esc(p.label || "Gauge")));
      d.appendChild(el("div", "value", esc(p.data != null ? p.data : (p.value != null ? p.value : "\u2014")))); return d; },
    Custom: (n) => { const p = n.props || {}; const fn = CUSTOM_RENDERERS[p.render];
      return fn ? fn(p) : el("div", "card md", esc("Custom: " + (p.render || "?"))); },
    FindingCard: renderFinding,
    InsightCard: renderFinding,
    Button: (n) => el("button", null, esc((n.props || {}).label || (n.props || {}).text || "Action")),
    FilterBar: (n) => el("div", "toolbar", ""),
  };

  function renderFinding(node) {
    const p = node.props || {};
    const sev = (p.severity || "info").toLowerCase();
    const d = el("div", "finding sev-" + sev);
    d.appendChild(el("div", "sev", esc(sev)));
    d.appendChild(el("div", "card-title", esc(p.title)));
    if (p.summary) d.appendChild(el("div", "summary", esc(p.summary)));
    return d;
  }

  function renderNode(node) {
    if (!node || typeof node !== "object") return el("div", "md", esc(node));
    const fn = RENDERERS[node.type];
    let dom;
    if (fn) {
      dom = fn(node);
    } else {
      dom = el("div", "card"); // safe fallback for unknown catalog types
      dom.appendChild(el("div", "sev", esc(node.type || "unknown")));
      dom.appendChild(el("div", "md", esc((node.props && node.props.text) || JSON.stringify(node.props || {}))));
      renderChildren(node, dom);
    }
    return bindAction(dom, node);
  }

  function render(spec) {
    rootEl.innerHTML = "";
    (spec.root || []).forEach((n) => rootEl.appendChild(renderNode(n)));
    if (!(spec.root || []).length) rootEl.appendChild(el("div", "empty", "No content yet."));
    document.title = spec.title ? spec.title + " \u00b7 LeapBoard" : "LeapBoard";
  }

  function toast(finding) {
    const sev = (finding.severity || "info").toLowerCase();
    const t = el("div", "toast sev-" + sev);
    t.appendChild(el("div", "card-title", esc(finding.title)));
    if (finding.summary) t.appendChild(el("div", "summary", esc(finding.summary)));
    toastsEl.appendChild(t);
    setTimeout(() => t.remove(), 8000);
  }

  // ── Live updates over WebSocket ──
  function connectWS() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(proto + "://" + location.host + "/ws?token=" + encodeURIComponent(TOKEN));
    ws.onopen = () => { statusEl.textContent = "live"; };
    ws.onclose = () => { statusEl.textContent = "reconnecting…"; setTimeout(connectWS, 3000); };
    ws.onmessage = (ev) => {
      let msg; try { msg = JSON.parse(ev.data); } catch (_) { return; }
      if (msg.type === "monitor.finding") { toast(msg.payload || {}); if (current.action === "home") fetchView(); }
      else if (msg.type === "watch.state") { if (current.action === "home") fetchView(); }
      else if (msg.type === "view.replace" && msg.spec) { render(msg.spec); }
    };
  }

  document.querySelectorAll("[data-action]").forEach((a) =>
    a.addEventListener("click", (ev) => { ev.preventDefault(); fetchView({ action: a.dataset.action, target: a.dataset.action === "session" ? "session" : "" }); }));

  fetchView();
  connectWS();
})();
