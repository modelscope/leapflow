// LeapBoard: a minimal Server-Driven UI renderer.
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
  const localeEl = document.getElementById("locale-switch");
  const storedLocale = localStorage.getItem("leapboard.locale") || "";
  const browserLocale = (navigator.language || "en").slice(0, 2).toLowerCase();
  let locale = storedLocale || (["en", "zh", "fr", "es", "ar", "ru"].includes(browserLocale) ? browserLocale : "en");
  let current = { template: params.get("template") || "" };
  let figSeq = 0;  // academic figure counter, reset each render()
  let tblSeq = 0;  // academic table counter, reset each render()

  const I18N = {
    zh: { "Overview": "概览", "Session": "会话", "Language": "语言", "connecting…": "连接中…", "live": "实时", "reconnecting…": "重连中…", "Loading…": "加载中…", "No content yet.": "暂无内容。", "Failed to load view": "视图加载失败", "Action failed": "操作失败", "Candlestick": "K线", "Series": "序列", "Gauge": "仪表", "Custom": "自定义", "unknown": "未知", "Watch portfolio": "观察组合", "Refresh cadence": "刷新节奏", "Active watches": "活跃观察", "Recent findings": "最新发现", "Watches": "观察任务", "Findings": "发现", "Signals": "信号", "Watch": "观察", "Price action": "价格行为", "Signal mix": "信号结构", "Market brief": "市场简报", "Latest sentiment": "最新情绪", "Mentions": "提及", "Sentiment structure": "情绪结构", "Narrative pulse": "叙事脉搏", "New papers": "新论文", "Research pipeline": "研究管线", "Evidence stream": "证据流", "Executive brief": "执行摘要", "Storyline": "叙事线", "Insights": "洞察", "Action items": "行动项", "Decisions": "决策", "Open questions": "待回答问题", "Entities": "实体", "Suggested next prompts": "建议追问", "Timeline": "时间线", "Severity mix": "严重度结构", "alert": "警报", "notable": "重要", "info": "信息", "Observation status": "观察状态", "Refresh state": "刷新状态", "Refresh reason": "刷新原因", "Coverage": "覆盖率", "Artifacts": "副产物", "Observed context": "已观察上下文", "File artifacts": "文件副产物", "File": "文件", "Status": "状态", "Note": "说明", "manual_refresh": "手动刷新", "first_observation": "首次观察", "artifact_changed": "文件副产物变化", "batch_turns": "轮次阈值", "batch_tokens": "上下文阈值", "model_salience": "模型显著性" },
    fr: { "Overview": "Vue d’ensemble", "Session": "Session", "Language": "Langue", "connecting…": "connexion…", "live": "direct", "reconnecting…": "reconnexion…", "Loading…": "chargement…", "No content yet.": "Aucun contenu.", "Failed to load view": "Échec du chargement", "Action failed": "Action échouée", "Watch": "Veille", "Watches": "Veilles", "Findings": "Constats", "Recent findings": "Constats récents", "Signals": "Signaux", "Insights": "Analyses", "Action items": "Actions", "Decisions": "Décisions", "Open questions": "Questions ouvertes", "Entities": "Entités", "Suggested next prompts": "Prochaines invites", "Executive brief": "Synthèse exécutive", "Storyline": "Narratif", "Timeline": "Chronologie", "Severity mix": "Mix de sévérité", "alert": "alerte", "notable": "notable", "info": "info", "Observation status": "Statut d’observation", "Refresh state": "État", "Refresh reason": "Raison", "Coverage": "Couverture", "Artifacts": "Artefacts", "Observed context": "Contexte observé", "File artifacts": "Fichiers", "File": "Fichier", "Status": "Statut", "Note": "Note", "manual_refresh": "actualisation manuelle", "first_observation": "première observation", "artifact_changed": "artefact modifié", "batch_turns": "seuil de tours", "batch_tokens": "seuil de jetons", "model_salience": "saillance modèle" },
    es: { "Overview": "Resumen", "Session": "Sesión", "Language": "Idioma", "connecting…": "conectando…", "live": "en vivo", "reconnecting…": "reconectando…", "Loading…": "cargando…", "No content yet.": "Sin contenido.", "Failed to load view": "Error al cargar", "Action failed": "Acción fallida", "Watch": "Vigilancia", "Watches": "Vigilancias", "Findings": "Hallazgos", "Recent findings": "Hallazgos recientes", "Signals": "Señales", "Insights": "Ideas", "Action items": "Acciones", "Decisions": "Decisiones", "Open questions": "Preguntas abiertas", "Entities": "Entidades", "Suggested next prompts": "Siguientes preguntas", "Executive brief": "Resumen ejecutivo", "Storyline": "Narrativa", "Timeline": "Cronología", "Severity mix": "Mezcla de severidad", "alert": "alerta", "notable": "relevante", "info": "info", "Observation status": "Estado de observación", "Refresh state": "Estado", "Refresh reason": "Motivo", "Coverage": "Cobertura", "Artifacts": "Artefactos", "Observed context": "Contexto observado", "File artifacts": "Archivos", "File": "Archivo", "Status": "Estado", "Note": "Nota", "manual_refresh": "actualización manual", "first_observation": "primera observación", "artifact_changed": "artefacto cambiado", "batch_turns": "umbral de turnos", "batch_tokens": "umbral de tokens", "model_salience": "relevancia del modelo" },
    ar: { "Overview": "نظرة عامة", "Session": "الجلسة", "Language": "اللغة", "connecting…": "جارٍ الاتصال…", "live": "مباشر", "reconnecting…": "إعادة الاتصال…", "Loading…": "جارٍ التحميل…", "No content yet.": "لا يوجد محتوى بعد.", "Failed to load view": "فشل تحميل العرض", "Action failed": "فشل الإجراء", "Watch": "مراقبة", "Watches": "المراقبات", "Findings": "النتائج", "Recent findings": "أحدث النتائج", "Signals": "الإشارات", "Insights": "الرؤى", "Action items": "إجراءات", "Decisions": "قرارات", "Open questions": "أسئلة مفتوحة", "Entities": "كيانات", "Suggested next prompts": "أسئلة مقترحة", "Executive brief": "ملخص تنفيذي", "Storyline": "السرد", "Timeline": "الخط الزمني", "Severity mix": "توزيع الشدة", "alert": "تنبيه", "notable": "مهم", "info": "معلومة", "Observation status": "حالة المراقبة", "Refresh state": "حالة التحديث", "Refresh reason": "سبب التحديث", "Coverage": "التغطية", "Artifacts": "المخرجات", "Observed context": "السياق المرصود", "File artifacts": "ملفات", "File": "ملف", "Status": "الحالة", "Note": "ملاحظة", "manual_refresh": "تحديث يدوي", "first_observation": "أول مراقبة", "artifact_changed": "تغير ملف", "batch_turns": "حد الجولات", "batch_tokens": "حد الرموز", "model_salience": "أهمية النموذج" },
    ru: { "Overview": "Обзор", "Session": "Сессия", "Language": "Язык", "connecting…": "подключение…", "live": "онлайн", "reconnecting…": "переподключение…", "Loading…": "загрузка…", "No content yet.": "Пока нет данных.", "Failed to load view": "Не удалось загрузить", "Action failed": "Действие не выполнено", "Watch": "Наблюдение", "Watches": "Наблюдения", "Findings": "Находки", "Recent findings": "Последние находки", "Signals": "Сигналы", "Insights": "Инсайты", "Action items": "Действия", "Decisions": "Решения", "Open questions": "Открытые вопросы", "Entities": "Сущности", "Suggested next prompts": "Следующие запросы", "Executive brief": "Краткий обзор", "Storyline": "Сюжет", "Timeline": "Хронология", "Severity mix": "Структура важности", "alert": "тревога", "notable": "важно", "info": "инфо", "Observation status": "Статус наблюдения", "Refresh state": "Состояние", "Refresh reason": "Причина", "Coverage": "Покрытие", "Artifacts": "Артефакты", "Observed context": "Наблюдаемый контекст", "File artifacts": "Файлы", "File": "Файл", "Status": "Статус", "Note": "Заметка", "manual_refresh": "ручное обновление", "first_observation": "первое наблюдение", "artifact_changed": "файл изменён", "batch_turns": "порог ходов", "batch_tokens": "порог токенов", "model_salience": "значимость модели" }
  };

  function t(key) { return (I18N[locale] && I18N[locale][key]) || key; }
  function tx(value) { return typeof value === "string" ? t(value) : value; }

  function applyLocale() {
    document.documentElement.lang = locale === "zh" ? "zh-CN" : locale;
    document.documentElement.dir = locale === "ar" ? "rtl" : "ltr";
    if (localeEl) localeEl.value = locale;
    document.querySelectorAll("[data-i18n]").forEach((node) => { node.textContent = t(node.dataset.i18n); });
  }

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
      const spec = await resp.json();
      render(spec);
      renderNav(spec.meta || {});
    } catch (err) {
      rootEl.innerHTML = '<div class="empty">' + esc(t("Failed to load view") + ": " + String(err)) + "</div>";
    }
  }

  // Template switcher: the current session, rendered through each lens.
  function renderNav(meta) {
    const nav = document.getElementById("nav");
    if (!nav) return;
    const names = Array.isArray(meta.templates) ? meta.templates : [];
    const active = meta.active_template || "";
    nav.innerHTML = "";
    names.forEach((name) => {
      const a = el("a", name === active ? "active" : "");
      a.href = "#";
      a.textContent = name;
      a.addEventListener("click", (ev) => { ev.preventDefault(); fetchView({ template: name }); });
      nav.appendChild(a);
    });
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
      toast({ title: t("Action failed"), summary: String(err), severity: "alert" });
    }
  }

  // nav actions are purely client-side (no server round-trip).
  function handleNav(action) {
    const p = action.params || {};
    if (action.name === "openLink" && p.url) window.open(p.url, "_blank", "noopener");
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
      const d = el("div", "mini-chart card"); d.appendChild(el("div", "card-title", t("Candlestick")));
      d.appendChild(el("div", "chart-placeholder", esc(data.length + " " + t("Series")))); return d; },
    gauge: (p) => renderGaugeValue(p.label || "Gauge", p.data),
  };

  function asArray(value) { return Array.isArray(value) ? value : []; }

  function severityOf(item) { return String((item && item.severity) || "info").toLowerCase(); }

  function severityCounts(items) {
    return asArray(items).reduce((acc, item) => { const sev = severityOf(item); acc[sev] = (acc[sev] || 0) + 1; return acc; }, {});
  }

  // Academic numbering: build a caption node ("Fig. N" / "Table N") + text.
  function captionInto(host, label, text) {
    const num = el("span", "fignum"); num.textContent = label; host.appendChild(num);
    if (text) host.appendChild(document.createTextNode(String(text)));
    return host;
  }
  function figcaption(text) { return captionInto(el("figcaption", "figcaption"), "Fig. " + (++figSeq), tx(text)); }
  function tableCaption(text) { return captionInto(document.createElement("caption"), "Table " + (++tblSeq), tx(text)); }
  function chartNode(dom, props) { if (props && props.caption) dom.appendChild(figcaption(props.caption)); return dom; }

  // Format the storyline like a paper abstract: bold lead-in sentence + body.
  function renderAbstract(text) {
    const s = String(text == null ? "" : text).trim();
    const box = el("div", "abstract");
    if (!s) return box;
    const idx = s.search(/[.!?\u3002\uff01\uff1f]/);
    if (idx > -1 && idx < 160) {
      const lead = el("span", "lead"); lead.textContent = s.slice(0, idx + 1); box.appendChild(lead);
      const rest = s.slice(idx + 1).trim();
      if (rest) box.appendChild(document.createTextNode(" " + rest));
    } else {
      box.textContent = s;
    }
    return box;
  }

  // List: a definition list when items carry a summary, else compact bullets.
  function renderList(node) {
    const items = asArray((node.props || {}).data);
    const structured = items.some((it) => it && typeof it === "object" && (it.summary || it.detail || it.value));
    if (structured) {
      const dl = el("dl", "dl");
      items.forEach((it) => {
        const obj = it && typeof it === "object";
        dl.appendChild(el("dt", null, esc(obj ? (it.title || it.name || it.label || "") : it)));
        dl.appendChild(el("dd", null, esc(obj ? (it.summary || it.detail || it.value || "") : "")));
      });
      return dl;
    }
    const ul = el("ul", "insight-list");
    items.forEach((it) => ul.appendChild(el("li", null, esc(typeof it === "object" ? (it.title || it.summary || JSON.stringify(it)) : it))));
    return ul;
  }

  function renderGaugeValue(label, value) {
    const d = el("div", "stat gauge-stat");
    d.appendChild(el("div", "label", esc(tx(label || "Gauge"))));
    d.appendChild(el("div", "value", esc(value != null && value !== "" ? value : "\u2014")));
    return d;
  }

  function renderChartBars(data, title) {
    const counts = severityCounts(data);
    const rows = ["alert", "notable", "info"].map((key) => ({ key, value: counts[key] || 0 }));
    const max = Math.max(1, ...rows.map((r) => r.value));
    const d = el("div", "chart card");
    if (title) d.appendChild(el("div", "card-title", esc(tx(title))));
    rows.forEach((row) => {
      const line = el("div", "bar-row");
      line.appendChild(el("span", "bar-label", esc(t(row.key))));
      const track = el("span", "bar-track");
      const fill = el("span", "bar-fill sev-" + row.key); fill.style.width = Math.round((row.value / max) * 100) + "%";
      track.appendChild(fill); line.appendChild(track); line.appendChild(el("span", "bar-value", esc(row.value))); d.appendChild(line);
    });
    return d;
  }

  function renderSparkline(data, title) {
    const values = asArray(data).slice(0, 18).map((_, i) => 20 + ((i * 17) % 53));
    const d = el("div", "chart card");
    if (title) d.appendChild(el("div", "card-title", esc(tx(title))));
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("viewBox", "0 0 180 72"); svg.setAttribute("class", "sparkline");
    const points = (values.length ? values : [38, 42, 36, 48]).map((v, i, arr) => (i * (180 / Math.max(1, arr.length - 1))) + "," + (72 - v)).join(" ");
    const poly = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    poly.setAttribute("points", points); svg.appendChild(poly); d.appendChild(svg); return d;
  }

  function renderPie(data, title) {
    const counts = severityCounts(data); const total = Math.max(1, (counts.alert || 0) + (counts.notable || 0) + (counts.info || 0));
    const d = el("div", "chart card pie-card");
    if (title) d.appendChild(el("div", "card-title", esc(tx(title))));
    const pie = el("div", "pie");
    pie.style.background = "conic-gradient(var(--alert) 0 " + ((counts.alert || 0) / total * 100) + "%, var(--notable) 0 " + (((counts.alert || 0) + (counts.notable || 0)) / total * 100) + "%, var(--info) 0 100%)";
    d.appendChild(pie); d.appendChild(renderLegend(["alert", "notable", "info"], counts)); return d;
  }

  function renderLegend(keys, counts) {
    const box = el("div", "legend");
    keys.forEach((key) => box.appendChild(el("span", "legend-item sev-" + key, esc(t(key) + " " + (counts[key] || 0)))));
    return box;
  }

  function renderTable(node) {
    const p = node.props || {}; const rows = asArray(p.data); const cols = asArray(p.columns);
    const table = el("table", "data-table");
    if (p.caption) table.appendChild(tableCaption(p.caption));
    const head = document.createElement("thead"); const headRow = document.createElement("tr");
    cols.forEach((c) => headRow.appendChild(el("th", null, esc(tx(c.label || c.key || c))))); head.appendChild(headRow); table.appendChild(head);
    const body = document.createElement("tbody"); rows.forEach((row) => { const tr = document.createElement("tr"); cols.forEach((c) => tr.appendChild(el("td", null, esc(row && row[c.key || c] != null ? row[c.key || c] : "")))); body.appendChild(tr); }); table.appendChild(body); return table;
  }

  function renderTimeline(node) {
    const items = asArray((node.props || {}).data); const d = el("div", "timeline");
    items.forEach((it) => { const row = el("div", "timeline-item sev-" + severityOf(it)); row.appendChild(el("div", "timeline-title", esc(it.title || ""))); if (it.summary) row.appendChild(el("div", "summary", esc(it.summary))); d.appendChild(row); });
    return d;
  }

  const RENDERERS = {
    Page: (n) => { const d = el("div", "page");
      const t0 = (n.props && n.props.title); if (t0) d.appendChild(el("div", "page-title", esc(tx(t0))));
      return renderChildren(n, d); },
    Section: (n) => { const p = n.props || {}; const d = el("section", "section");
      if (p.title) d.appendChild(el("div", "section-title", esc(tx(p.title))));
      if (p.subtitle) d.appendChild(el("div", "section-subtitle", esc(tx(p.subtitle))));
      return renderChildren(n, d); },
    Grid: (n) => renderChildren(n, el("div", "grid")),
    Row: (n) => renderChildren(n, el("div", "row")),
    Col: (n) => renderChildren(n, el("div", "col")),
    Card: (n) => { const d = el("div", "card");
      const title = n.props && n.props.title; if (title) d.appendChild(el("div", "card-title", esc(tx(title))));
      const kicker = n.props && n.props.kicker; if (kicker) d.appendChild(el("div", "kicker", esc(tx(kicker))));
      return renderChildren(n, d); },
    Board: (n) => { const d = el("div", "board");
      const title = n.props && n.props.title; if (title) d.appendChild(el("div", "board-title", esc(tx(title))));
      return renderChildren(n, d); },
    Toolbar: (n) => renderChildren(n, el("div", "toolbar")),
    Stat: (n) => { const p = n.props || {}; const d = el("div", "stat");
      d.appendChild(el("div", "label", esc(tx(p.label)))); d.appendChild(el("div", "value", esc(p.value != null && p.value !== "" ? p.value : "\u2014")));
      return d; },
    Markdown: (n) => el("div", "md prose", esc((n.props || {}).text)),
    StoryPanel: (n) => { const p = n.props || {}; const d = el("div", "card story-panel");
      d.appendChild(el("div", "card-title", esc(tx(p.title || "Storyline"))));
      d.appendChild(renderAbstract(p.text)); return d; },
    List: renderList,
    SuggestionChips: (n) => { const items = ((n.props || {}).data) || []; const d = el("div", "chips");
      asArray(items).forEach((it) => d.appendChild(el("button", null, esc(it)))); return d; },
    Gauge: (n) => { const p = n.props || {}; return renderGaugeValue(p.label || "Gauge", p.data != null ? p.data : p.value); },
    ProgressBar: (n) => { const p = n.props || {}; const d = el("div", "progress"); const fill = el("span", "progress-fill"); fill.style.width = Math.max(0, Math.min(100, Number(p.value || 0))) + "%"; d.appendChild(fill); return d; },
    Badge: (n) => { const p = n.props || {}; return el("span", "badge sev-" + String(p.tone || p.severity || "info").toLowerCase(), esc(tx(p.label || p.text || "info"))); },
    Table: renderTable,
    Timeline: renderTimeline,
    BarChart: (n) => chartNode(renderChartBars((n.props || {}).data, (n.props || {}).title || "Severity mix"), n.props || {}),
    AreaChart: (n) => chartNode(renderSparkline((n.props || {}).data, (n.props || {}).title), n.props || {}),
    LineChart: (n) => chartNode(renderSparkline((n.props || {}).data, (n.props || {}).title), n.props || {}),
    Sparkline: (n) => chartNode(renderSparkline((n.props || {}).data, (n.props || {}).title), n.props || {}),
    PieChart: (n) => chartNode(renderPie((n.props || {}).data, (n.props || {}).title || "Severity mix"), n.props || {}),
    Quote: (n) => { const p = n.props || {}; const q = el("blockquote", "quote", esc(p.text)); if (p.source) q.appendChild(el("cite", null, esc(p.source))); return q; },
    CitationList: (n) => { const items = asArray((n.props || {}).data); const ol = el("ol", "citations"); items.forEach((it) => ol.appendChild(el("li", null, esc(it.label || it.title || it.url || it)))); return ol; },
    EntityGraph: (n) => { const items = asArray((n.props || {}).data); const d = el("div", "entity-cloud"); items.forEach((it) => d.appendChild(el("span", "badge", esc(it.name || it.title || it)))); return d; },
    Custom: (n) => { const p = n.props || {}; const fn = CUSTOM_RENDERERS[p.render];
      return fn ? fn(p) : el("div", "card md", esc(t("Custom") + ": " + (p.render || "?"))); },
    FindingCard: renderFinding,
    InsightCard: renderFinding,
    Button: (n) => el("button", null, esc(tx((n.props || {}).label || (n.props || {}).text || "Action"))),
    FilterBar: (n) => el("div", "toolbar", ""),
  };

  function renderFinding(node) {
    const p = node.props || {};
    const sev = (p.severity || "info").toLowerCase();
    const d = el("div", "finding sev-" + sev);
    d.appendChild(el("div", "sev", esc(t(sev))));
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
    figSeq = 0; tblSeq = 0;
    (spec.root || []).forEach((n) => rootEl.appendChild(renderNode(n)));
    if (!(spec.root || []).length) rootEl.appendChild(el("div", "empty", esc(t("No content yet."))));
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
    ws.onopen = () => { statusEl.textContent = t("live"); };
    ws.onclose = () => { statusEl.textContent = t("reconnecting…"); setTimeout(connectWS, 3000); };
    ws.onmessage = (ev) => {
      let msg; try { msg = JSON.parse(ev.data); } catch (_) { return; }
      if (msg.type === "monitor.finding") { toast(msg.payload || {}); fetchView(); }
      else if (msg.type === "watch.state") { fetchView(); }
      else if (msg.type === "view.replace" && msg.spec) { render(msg.spec); }
    };
  }

  if (localeEl) {
    localeEl.addEventListener("change", () => {
      locale = localeEl.value || "en";
      localStorage.setItem("leapboard.locale", locale);
      applyLocale();
      fetchView();
    });
  }

  applyLocale();
  fetchView();
  connectWS();
})();
