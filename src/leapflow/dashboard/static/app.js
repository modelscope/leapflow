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
  const HIDDEN_NAV_TEMPLATES = new Set(["finance", "research", "sentiment"]);
  let figSeq = 0;  // academic figure counter, reset each render()
  let tblSeq = 0;  // academic table counter, reset each render()

  // ── Signal auto-refresh state ──
  let _signalRefreshTimer = null;
  let _signalEventCount = 0;

  function getCurrentTemplate() { return current.template || ""; }

  function startSignalAutoRefresh() {
    stopSignalAutoRefresh();
    _signalRefreshTimer = setInterval(function () {
      if (getCurrentTemplate() === "signals") fetchView();
    }, 5000);
  }

  function stopSignalAutoRefresh() {
    if (_signalRefreshTimer) { clearInterval(_signalRefreshTimer); _signalRefreshTimer = null; }
  }

  function incrementSignalCounter() {
    _signalEventCount++;
    var counterEl = document.getElementById("signal-event-counter");
    if (counterEl) counterEl.textContent = String(_signalEventCount);
  }

  function injectSignalRefreshBtn() {
    if (getCurrentTemplate() !== "signals") return;
    // Find the page title or first section title to attach the controls.
    var title = rootEl.querySelector(".page-title") || rootEl.querySelector(".section-title");
    if (!title) return;
    // Avoid duplicates; controls belong to the title node, never the page
    // container, so the page keeps its vertical document flow.
    var header = title;
    if (header.querySelector(".refresh-btn")) return;
    header.classList.add("with-actions");
    var btn = document.createElement("button");
    btn.className = "refresh-btn";
    btn.textContent = "\u21bb";  // ↻
    btn.title = "Refresh signal metrics";
    btn.addEventListener("click", function () {
      btn.disabled = true;
      btn.classList.add("refreshing");
      fetchView().then(function () {
        setTimeout(function () { btn.disabled = false; btn.classList.remove("refreshing"); }, 400);
      }).catch(function () {
        btn.disabled = false; btn.classList.remove("refreshing");
      });
    });
    header.appendChild(btn);
    // Also inject event counter badge next to the button
    var counter = document.createElement("span");
    counter.className = "signal-counter-badge";
    counter.id = "signal-event-counter";
    counter.textContent = String(_signalEventCount);
    counter.title = "WebSocket events received";
    header.appendChild(counter);
  }

  const I18N = {
    en: { "manual_refresh": "manual refresh", "first_observation": "first observation", "artifact_changed": "artifact changed", "batch_turns": "turn threshold", "batch_tokens": "token threshold", "model_salience": "model salience", "text_only": "conversation text", "text_and_artifacts": "conversation + files", "partial_artifacts": "partial files" },
    zh: {"Abstract": "摘要", "Action failed": "操作失败", "Action items": "行动项", "Active event-driven monitors": "活跃的事件驱动监视器", "Active triggers": "活跃触发器", "Active watches": "活跃观察", "Artifacts": "副产物", "Buffer dropped": "缓冲丢弃", "Calibrated at": "校准时间", "Calibration health": "校准健康度", "Candlestick": "K线", "Channels that have never been calibrated or whose calibration has expired are shown first.": "从未校准或校准已过期的通道排在最前。", "Context map": "上下文图谱", "Coverage": "覆盖率", "Coverage · storyline · severity": "覆盖率 · 叙事 · 严重度", "Custom": "自定义", "Days since": "距今天数", "Debounced": "去抖", "Decisions": "决策", "Decisions and actions": "决策与行动", "Domain": "领域", "Entities": "实体", "Entities and follow-ups": "实体与后续", "Evidence stream": "证据流", "Executive brief": "执行摘要", "Extracted from this session's tool/file output (not model-generated).": "数据来自本次会话的工具/文件产物（非模型生成）。", "Failed to load view": "视图加载失败", "File": "文件", "File artifacts": "文件副产物", "Findings": "发现", "Gauge": "仪表", "Insight count by severity.": "按严重度统计的洞察数。", "Insights": "洞察", "Key observations": "关键观察", "Language": "语言", "Latest observation results": "最新观测结果", "Latest sentiment": "最新情绪", "Live signal stream": "实时信号流", "Loading…": "加载中…", "Market brief": "市场简报", "Mentions": "提及", "Name": "名称", "Narrative pulse": "叙事脉搏", "New papers": "新论文", "Next prompts": "后续追问", "Next recal due": "下次校准期限", "No content yet.": "暂无内容。", "No entries.": "暂无条目。", "Note": "说明", "Observation": "观察", "Observation status": "观察状态", "Observed context": "已观察上下文", "Open questions": "待回答问题", "Operating agenda": "行动议程", "Overview": "概览", "Per-channel calibration state, freshness, and residual correction": "各通道的校准状态、时效性与残差校正", "Price action": "价格行为", "Reason": "原因", "Recent events (last 50)": "最近事件（最新50条）", "Recent findings": "最新发现", "Refresh cadence": "刷新节奏", "Refresh reason": "刷新原因", "Refresh state": "刷新状态", "Research pipeline": "研究管线", "Residual": "残差", "Sentiment structure": "情绪结构", "Series": "序列", "Session": "会话", "Session Analysis": "会话分析", "Session file artifacts.": "会话文件副产物。", "Severity mix": "严重度结构", "Signal flow": "信号流", "Signal mix": "信号结构", "Signals": "信号", "State": "状态", "Status": "状态", "Storyline": "叙事线", "Subscribers": "订阅者", "Suggested next prompts": "建议追问", "Timeline": "时间线", "Tokens": "词元", "Trigger": "触发器", "Trigger and context": "触发与上下文", "Turns": "轮次", "Watch": "观察", "Watch portfolio": "观察组合", "Watches": "观察任务", "alert": "警报", "artifact_changed": "文件副产物变化", "batch_tokens": "上下文阈值", "batch_turns": "轮次阈值", "connecting…": "连接中…", "first_observation": "首次观察", "info": "信息", "live": "实时", "manual_refresh": "手动刷新", "model_salience": "模型显著性", "notable": "重要", "reconnecting…": "重连中…", "unknown": "未知"},
    fr: {"Abstract": "Résumé", "Action failed": "Action échouée", "Action items": "Actions", "Active event-driven monitors": "Moniteurs événementiels actifs", "Active triggers": "Déclencheurs actifs", "Artifacts": "Artefacts", "Buffer dropped": "Tampon perdu", "Calibrated at": "Calibré le", "Calibration health": "État de calibration", "Channels that have never been calibrated or whose calibration has expired are shown first.": "Les canaux jamais calibrés ou dont la calibration a expiré apparaissent en premier.", "Context map": "Carte de contexte", "Coverage": "Couverture", "Coverage · storyline · severity": "Couverture · récit · sévérité", "Days since": "Jours écoulés", "Debounced": "Antirebond", "Decisions": "Décisions", "Decisions and actions": "Décisions et actions", "Domain": "Domaine", "Entities": "Entités", "Entities and follow-ups": "Entités et suivis", "Executive brief": "Synthèse exécutive", "Failed to load view": "Échec du chargement", "File": "Fichier", "File artifacts": "Fichiers", "Findings": "Constats", "Insight count by severity.": "Nombre d’analyses par sévérité.", "Insights": "Analyses", "Key observations": "Observations clés", "Language": "Langue", "Latest observation results": "Derniers résultats d'observation", "Live signal stream": "Flux de signaux en direct", "Loading…": "chargement…", "Name": "Nom", "Next prompts": "Invites suivantes", "Next recal due": "Prochaine recalibration", "No content yet.": "Aucun contenu.", "No entries.": "Aucune entrée.", "Note": "Note", "Observation": "Observation", "Observation status": "Statut d’observation", "Observed context": "Contexte observé", "Open questions": "Questions ouvertes", "Operating agenda": "Programme d’action", "Overview": "Vue d’ensemble", "Per-channel calibration state, freshness, and residual correction": "État de calibration, fraîcheur et correction résiduelle par canal", "Reason": "Raison", "Recent events (last 50)": "Événements récents (50 derniers)", "Recent findings": "Constats récents", "Refresh reason": "Raison", "Refresh state": "État", "Residual": "Résidu", "Session": "Session", "Session Analysis": "Analyse de session", "Session file artifacts.": "Artefacts de fichiers de session.", "Severity mix": "Mix de sévérité", "Signal flow": "Flux de signaux", "Signals": "Signaux", "State": "État", "Status": "Statut", "Storyline": "Narratif", "Subscribers": "Abonnés", "Suggested next prompts": "Prochaines invites", "Timeline": "Chronologie", "Tokens": "Jetons", "Trigger": "Déclencheur", "Trigger and context": "Déclencheur et contexte", "Turns": "Tours", "Watch": "Veille", "Watches": "Veilles", "alert": "alerte", "artifact_changed": "artefact modifié", "batch_tokens": "seuil de jetons", "batch_turns": "seuil de tours", "connecting…": "connexion…", "first_observation": "première observation", "info": "info", "live": "direct", "manual_refresh": "actualisation manuelle", "model_salience": "saillance modèle", "notable": "notable", "reconnecting…": "reconnexion…"},
    es: {"Abstract": "Resumen", "Action failed": "Acción fallida", "Action items": "Acciones", "Active event-driven monitors": "Monitores por eventos activos", "Active triggers": "Disparadores activos", "Artifacts": "Artefactos", "Buffer dropped": "Buffer perdido", "Calibrated at": "Calibrado el", "Calibration health": "Estado de calibración", "Channels that have never been calibrated or whose calibration has expired are shown first.": "Los canales nunca calibrados o con calibración vencida se muestran primero.", "Context map": "Mapa de contexto", "Coverage": "Cobertura", "Coverage · storyline · severity": "Cobertura · relato · severidad", "Days since": "Días desde", "Debounced": "Antirrebote", "Decisions": "Decisiones", "Decisions and actions": "Decisiones y acciones", "Domain": "Dominio", "Entities": "Entidades", "Entities and follow-ups": "Entidades y seguimientos", "Executive brief": "Resumen ejecutivo", "Failed to load view": "Error al cargar", "File": "Archivo", "File artifacts": "Archivos", "Findings": "Hallazgos", "Insight count by severity.": "Recuento de hallazgos por severidad.", "Insights": "Ideas", "Key observations": "Observaciones clave", "Language": "Idioma", "Latest observation results": "Últimos resultados de observación", "Live signal stream": "Flujo de señales en vivo", "Loading…": "cargando…", "Name": "Nombre", "Next prompts": "Siguientes prompts", "Next recal due": "Próxima recalibración", "No content yet.": "Sin contenido.", "No entries.": "Sin entradas.", "Note": "Nota", "Observation": "Observación", "Observation status": "Estado de observación", "Observed context": "Contexto observado", "Open questions": "Preguntas abiertas", "Operating agenda": "Agenda operativa", "Overview": "Resumen", "Per-channel calibration state, freshness, and residual correction": "Estado de calibración, vigencia y corrección residual por canal", "Reason": "Motivo", "Recent events (last 50)": "Eventos recientes (últimos 50)", "Recent findings": "Hallazgos recientes", "Refresh reason": "Motivo", "Refresh state": "Estado", "Residual": "Residuo", "Session": "Sesión", "Session Analysis": "Análisis de sesión", "Session file artifacts.": "Artefactos de archivos de sesión.", "Severity mix": "Mezcla de severidad", "Signal flow": "Flujo de señales", "Signals": "Señales", "State": "Estado", "Status": "Estado", "Storyline": "Narrativa", "Subscribers": "Suscriptores", "Suggested next prompts": "Siguientes preguntas", "Timeline": "Cronología", "Tokens": "Tokens", "Trigger": "Disparador", "Trigger and context": "Disparador y contexto", "Turns": "Turnos", "Watch": "Vigilancia", "Watches": "Vigilancias", "alert": "alerta", "artifact_changed": "artefacto cambiado", "batch_tokens": "umbral de tokens", "batch_turns": "umbral de turnos", "connecting…": "conectando…", "first_observation": "primera observación", "info": "info", "live": "en vivo", "manual_refresh": "actualización manual", "model_salience": "relevancia del modelo", "notable": "relevante", "reconnecting…": "reconectando…"},
    ar: {"Abstract": "ملخص", "Action failed": "فشل الإجراء", "Action items": "إجراءات", "Active event-driven monitors": "مراقبات حدثية نشطة", "Active triggers": "المُشغِّلات النشطة", "Artifacts": "المخرجات", "Buffer dropped": "ذاكرة مؤقتة مُسقَطة", "Calibrated at": "تاريخ المعايرة", "Calibration health": "سلامة المعايرة", "Channels that have never been calibrated or whose calibration has expired are shown first.": "تظهر أولاً القنوات التي لم تُعاير قط أو التي انتهت صلاحية معايرتها.", "Context map": "خريطة السياق", "Coverage": "التغطية", "Coverage · storyline · severity": "التغطية · السرد · الخطورة", "Days since": "الأيام المنقضية", "Debounced": "مُزال الارتداد", "Decisions": "قرارات", "Decisions and actions": "القرارات والإجراءات", "Domain": "المجال", "Entities": "كيانات", "Entities and follow-ups": "الكيانات والمتابعات", "Executive brief": "ملخص تنفيذي", "Failed to load view": "فشل تحميل العرض", "File": "ملف", "File artifacts": "ملفات", "Findings": "النتائج", "Insight count by severity.": "عدد الرؤى حسب الخطورة.", "Insights": "الرؤى", "Key observations": "ملاحظات رئيسية", "Language": "اللغة", "Latest observation results": "أحدث نتائج الرصد", "Live signal stream": "تدفق الإشارات المباشر", "Loading…": "جارٍ التحميل…", "Name": "الاسم", "Next prompts": "المطالبات التالية", "Next recal due": "موعد إعادة المعايرة", "No content yet.": "لا يوجد محتوى بعد.", "No entries.": "لا توجد إدخالات.", "Note": "ملاحظة", "Observation": "الرصد", "Observation status": "حالة المراقبة", "Observed context": "السياق المرصود", "Open questions": "أسئلة مفتوحة", "Operating agenda": "خطة العمل", "Overview": "نظرة عامة", "Per-channel calibration state, freshness, and residual correction": "حالة المعايرة وحداثتها وتصحيح المتبقي لكل قناة", "Reason": "السبب", "Recent events (last 50)": "الأحداث الأخيرة (آخر 50)", "Recent findings": "أحدث النتائج", "Refresh reason": "سبب التحديث", "Refresh state": "حالة التحديث", "Residual": "المتبقي", "Session": "الجلسة", "Session Analysis": "تحليل الجلسة", "Session file artifacts.": "مخرجات ملفات الجلسة.", "Severity mix": "توزيع الشدة", "Signal flow": "تدفق الإشارات", "Signals": "الإشارات", "State": "الحالة", "Status": "الحالة", "Storyline": "السرد", "Subscribers": "المشتركون", "Suggested next prompts": "أسئلة مقترحة", "Timeline": "الخط الزمني", "Tokens": "الرموز", "Trigger": "المُشغِّل", "Trigger and context": "المُشغِّل والسياق", "Turns": "الأدوار", "Watch": "مراقبة", "Watches": "المراقبات", "alert": "تنبيه", "artifact_changed": "تغير ملف", "batch_tokens": "حد الرموز", "batch_turns": "حد الجولات", "connecting…": "جارٍ الاتصال…", "first_observation": "أول مراقبة", "info": "معلومة", "live": "مباشر", "manual_refresh": "تحديث يدوي", "model_salience": "أهمية النموذج", "notable": "مهم", "reconnecting…": "إعادة الاتصال…"},
    ru: {"Abstract": "Аннотация", "Action failed": "Действие не выполнено", "Action items": "Действия", "Active event-driven monitors": "Активные событийные мониторы", "Active triggers": "Активные триггеры", "Artifacts": "Артефакты", "Buffer dropped": "Потери буфера", "Calibrated at": "Калиброван", "Calibration health": "Состояние калибровки", "Channels that have never been calibrated or whose calibration has expired are shown first.": "Каналы, которые никогда не калибровались или чья калибровка истекла, показаны первыми.", "Context map": "Карта контекста", "Coverage": "Покрытие", "Coverage · storyline · severity": "Покрытие · сюжет · важность", "Days since": "Дней с тех пор", "Debounced": "Дебаунс", "Decisions": "Решения", "Decisions and actions": "Решения и действия", "Domain": "Домен", "Entities": "Сущности", "Entities and follow-ups": "Сущности и продолжения", "Executive brief": "Краткий обзор", "Failed to load view": "Не удалось загрузить", "File": "Файл", "File artifacts": "Файлы", "Findings": "Находки", "Insight count by severity.": "Число инсайтов по важности.", "Insights": "Инсайты", "Key observations": "Ключевые наблюдения", "Language": "Язык", "Latest observation results": "Последние результаты наблюдений", "Live signal stream": "Поток сигналов (live)", "Loading…": "загрузка…", "Name": "Имя", "Next prompts": "Следующие запросы", "Next recal due": "Следующая рекалибровка", "No content yet.": "Пока нет данных.", "No entries.": "Нет записей.", "Note": "Заметка", "Observation": "Наблюдение", "Observation status": "Статус наблюдения", "Observed context": "Наблюдаемый контекст", "Open questions": "Открытые вопросы", "Operating agenda": "Рабочая повестка", "Overview": "Обзор", "Per-channel calibration state, freshness, and residual correction": "Состояние калибровки, актуальность и остаточная поправка по каналам", "Reason": "Причина", "Recent events (last 50)": "Последние события (50)", "Recent findings": "Последние находки", "Refresh reason": "Причина", "Refresh state": "Состояние", "Residual": "Остаток", "Session": "Сессия", "Session Analysis": "Анализ сессии", "Session file artifacts.": "Файловые артефакты сессии.", "Severity mix": "Структура важности", "Signal flow": "Поток сигналов", "Signals": "Сигналы", "State": "Состояние", "Status": "Статус", "Storyline": "Сюжет", "Subscribers": "Подписчики", "Suggested next prompts": "Следующие запросы", "Timeline": "Хронология", "Tokens": "Токены", "Trigger": "Триггер", "Trigger and context": "Триггер и контекст", "Turns": "Ходы", "Watch": "Наблюдение", "Watches": "Наблюдения", "alert": "тревога", "artifact_changed": "файл изменён", "batch_tokens": "порог токенов", "batch_turns": "порог ходов", "connecting…": "подключение…", "first_observation": "первое наблюдение", "info": "инфо", "live": "онлайн", "manual_refresh": "ручное обновление", "model_salience": "значимость модели", "notable": "важно", "reconnecting…": "переподключение…"}
  };

  const I18N_PATCH = {
    en: {
      "All": "All",
      "connecting…": "connecting",
      "live": "connected",
      "reconnecting…": "reconnecting",
      "seconds ago": "{count}s ago",
      "minutes ago": "{count}m ago",
      "hours ago": "{count}h ago",
      "Showing {shown} of {total} recent events.": "Showing {shown} of {total} recent events.",
      "Showing {shown} of {total} {family} events.": "Showing {shown} of {total} {family} events.",
      "stale build": "stale build",
      "stale_build_title": "This LeapBoard server (pid {pid}) predates the current source tree. Restart it to pick up recent changes.",
      "Stream events": "Stream events", "Active watches": "Active watches", "Watch portfolio": "Watch portfolio", "Noise suppressed": "Noise suppressed", "Source dropped": "Source dropped", "Reorder pending": "Reorder pending",
      "Signal health summary": "Signal health summary", "Ingress": "Ingress", "Pressure": "Pressure", "Recent event families": "Recent event families",
      "Finding severity mix": "Finding severity mix", "Watch state mix": "Watch state mix", "Watch states": "Watch states", "Trigger coverage": "Trigger coverage",
      "Latest daemon events · grouped by signal family · newest first.": "Latest daemon events · grouped by signal family · newest first.",
      "Ingress fan-out, pipeline pressure, and recent dimensional mix.": "Ingress fan-out, pipeline pressure, and recent dimensional mix.",
      "Event count by normalized family in the live ring buffer.": "Event count by normalized family in the live ring buffer.",
      "Observation count by severity across recent findings.": "Observation count by severity across recent findings.",
      "Current monitor lifecycle states.": "Current monitor lifecycle states.", "Active and completed event-driven monitors.": "Active and completed event-driven monitors.",
      "Latest observation results.": "Latest observation results.", "Event patterns registered with the monitor event bridge.": "Event patterns registered with the monitor event bridge.",
      "Triggers": "Triggers", "Watches": "Watches", "Pattern": "Pattern", "Triggered": "Triggered", "Last event": "Last event", "Value": "Value", "Dimension": "Dimension", "Signal": "Signal",
      "armed": "armed", "done": "done", "suspended": "suspended", "yes": "yes", "no": "no",
      "signal.family.fs": "fs", "signal.family.gateway": "gateway", "signal.family.ui": "ui", "signal.family.clipboard": "clipboard", "signal.family.app": "app", "signal.family.unknown": "unknown", "signal.family.hw": "hardware",
      "Physical bench": "Physical bench", "Devices": "Devices", "Charted channels": "Charted channels",
      "Recent events": "Recent events", "Unpersisted windows": "Unpersisted windows",
      "Raw samples written": "Raw samples written", "Watch state": "Watch state",
      "Channel traces": "Channel traces", "Sampled channels": "Sampled channels",
      "Envelope conformance": "Envelope conformance", "Window conformance": "Window conformance",
      "Device events": "Device events", "Sampling health": "Sampling health",
      "Learned command outcomes": "Learned command outcomes", "inside": "inside", "near": "near",
      "outside": "outside", "unknown": "unknown"
    },
    zh: {
      "All": "全部",
      "connecting…": "正在连接",
      "live": "已连接",
      "reconnecting…": "正在重连",
      "seconds ago": "{count}秒前", "minutes ago": "{count}分钟前", "hours ago": "{count}小时前",
      "Showing {shown} of {total} recent events.": "显示最近 {total} 个事件中的 {shown} 个。",
      "Showing {shown} of {total} {family} events.": "显示 {total} 个{family}事件中的 {shown} 个。",
      "stale build": "构建已过期", "stale_build_title": "LeapBoard 服务（pid {pid}）早于当前源码树启动。请重启以加载最近的更改。",
      "Stream events": "流事件", "Active watches": "活跃观察", "Watch portfolio": "观察组合", "Noise suppressed": "已压制噪声", "Source dropped": "源丢弃", "Reorder pending": "重排待处理",
      "Signal health summary": "信号健康摘要", "Ingress": "输入", "Pressure": "压力", "Recent event families": "最近事件类别",
      "Finding severity mix": "发现严重度分布", "Watch state mix": "观察状态分布", "Watch states": "观察状态", "Trigger coverage": "触发覆盖",
      "Latest daemon events · grouped by signal family · newest first.": "最新 daemon 事件 · 按信号类别分组 · 最新优先。",
      "Ingress fan-out, pipeline pressure, and recent dimensional mix.": "输入扇出、管线压力和最近维度分布。",
      "Event count by normalized family in the live ring buffer.": "实时环形缓冲区中按标准化类别统计的事件数。",
      "Observation count by severity across recent findings.": "最近发现中按严重度统计的观察数。",
      "Current monitor lifecycle states.": "当前监视器生命周期状态。", "Active and completed event-driven monitors.": "活跃和已完成的事件驱动监视器。",
      "Latest observation results.": "最新观察结果。", "Event patterns registered with the monitor event bridge.": "监视器事件桥注册的事件模式。",
      "Triggers": "触发器", "Watches": "观察任务", "Pattern": "模式", "Triggered": "已触发", "Last event": "最后事件", "Value": "值", "Dimension": "维度", "Signal": "信号",
      "armed": "已布防", "done": "完成", "suspended": "已暂停", "yes": "是", "no": "否",
      "signal.family.fs": "文件", "signal.family.gateway": "网关", "signal.family.ui": "界面", "signal.family.clipboard": "剪贴板", "signal.family.app": "应用", "signal.family.unknown": "未知", "signal.family.hw": "硬件",
      "Physical bench": "物理台面", "Devices": "设备", "Charted channels": "已绘通道",
      "Recent events": "近期事件", "Unpersisted windows": "未落盘窗口",
      "Raw samples written": "原始样本写入", "Watch state": "监视状态",
      "Channel traces": "通道轨迹", "Sampled channels": "采样通道",
      "Envelope conformance": "包络遵从性", "Window conformance": "窗口遵从性",
      "Device events": "设备事件", "Sampling health": "采样健康度",
      "Learned command outcomes": "已学习的命令结果", "inside": "范围内", "near": "接近边界",
      "outside": "越界", "unknown": "未知"
    },
    fr: {
      "All": "Tout", "connecting…": "connexion", "live": "connecté", "reconnecting…": "reconnexion", "seconds ago": "il y a {count} s", "minutes ago": "il y a {count} min", "hours ago": "il y a {count} h",
      "Showing {shown} of {total} recent events.": "Affichage de {shown} sur {total} événements récents.", "Showing {shown} of {total} {family} events.": "Affichage de {shown} sur {total} événements {family}.",
      "stale build": "build obsolète", "stale_build_title": "Ce serveur LeapBoard (pid {pid}) est antérieur à l'arbre source actuel. Redémarrez-le pour charger les changements récents.",
      "Stream events": "Événements de flux", "Active watches": "Veilles actives", "Watch portfolio": "Portefeuille de veilles", "Noise suppressed": "Bruit supprimé", "Source dropped": "Source rejetée", "Reorder pending": "Réordonnancement en attente",
      "Signal health summary": "Résumé santé des signaux", "Ingress": "Entrée", "Pressure": "Pression", "Recent event families": "Familles d'événements récentes", "Finding severity mix": "Répartition des constats", "Watch state mix": "États des veilles", "Watch states": "États des veilles", "Trigger coverage": "Couverture des déclencheurs",
      "Latest daemon events · grouped by signal family · newest first.": "Derniers événements daemon · groupés par famille · plus récents d'abord.", "Ingress fan-out, pipeline pressure, and recent dimensional mix.": "Diffusion d'entrée, pression du pipeline et dimensions récentes.", "Event count by normalized family in the live ring buffer.": "Nombre d'événements par famille normalisée dans le tampon live.", "Observation count by severity across recent findings.": "Nombre d'observations par sévérité dans les constats récents.", "Current monitor lifecycle states.": "États courants du cycle de vie des moniteurs.", "Active and completed event-driven monitors.": "Moniteurs événementiels actifs et terminés.", "Latest observation results.": "Derniers résultats d'observation.", "Event patterns registered with the monitor event bridge.": "Motifs d'événements enregistrés dans le pont des moniteurs.",
      "Triggers": "Déclencheurs", "Watches": "Veilles", "Pattern": "Motif", "Triggered": "Déclenché", "Last event": "Dernier événement", "Value": "Valeur", "Dimension": "Dimension", "Signal": "Signal", "armed": "armé", "done": "terminé", "suspended": "suspendu", "yes": "oui", "no": "non",
      "signal.family.fs": "fichiers", "signal.family.gateway": "passerelle", "signal.family.ui": "interface", "signal.family.clipboard": "presse-papiers", "signal.family.app": "application", "signal.family.unknown": "inconnu", "signal.family.hw": "matériel",
      "Physical bench": "Banc physique", "Devices": "Appareils", "Charted channels": "Voies tracées",
      "Recent events": "Événements récents", "Unpersisted windows": "Fenêtres non persistées",
      "Raw samples written": "Échantillons bruts écrits", "Watch state": "État de surveillance",
      "Channel traces": "Tracés des voies", "Sampled channels": "Voies échantillonnées",
      "Envelope conformance": "Conformité à l'enveloppe", "Window conformance": "Conformité des fenêtres",
      "Device events": "Événements matériels", "Sampling health": "Santé de l'échantillonnage",
      "Learned command outcomes": "Résultats de commandes appris", "inside": "dans les limites",
      "near": "proche de la limite", "outside": "hors limites", "unknown": "inconnu"
    },
    es: {
      "All": "Todo", "connecting…": "conectando", "live": "conectado", "reconnecting…": "reconectando", "seconds ago": "hace {count} s", "minutes ago": "hace {count} min", "hours ago": "hace {count} h",
      "Showing {shown} of {total} recent events.": "Mostrando {shown} de {total} eventos recientes.", "Showing {shown} of {total} {family} events.": "Mostrando {shown} de {total} eventos {family}.",
      "stale build": "build obsoleto", "stale_build_title": "Este servidor LeapBoard (pid {pid}) es anterior al árbol de código actual. Reinícialo para cargar los cambios recientes.",
      "Stream events": "Eventos de flujo", "Active watches": "Vigilancias activas", "Watch portfolio": "Cartera de vigilancias", "Noise suppressed": "Ruido suprimido", "Source dropped": "Fuente descartada", "Reorder pending": "Reordenación pendiente",
      "Signal health summary": "Resumen de salud de señales", "Ingress": "Entrada", "Pressure": "Presión", "Recent event families": "Familias de eventos recientes", "Finding severity mix": "Mezcla de severidad", "Watch state mix": "Estados de vigilancia", "Watch states": "Estados de vigilancia", "Trigger coverage": "Cobertura de disparadores",
      "Latest daemon events · grouped by signal family · newest first.": "Últimos eventos del daemon · agrupados por familia · recientes primero.", "Ingress fan-out, pipeline pressure, and recent dimensional mix.": "Difusión de entrada, presión del pipeline y mezcla dimensional reciente.", "Event count by normalized family in the live ring buffer.": "Conteo de eventos por familia normalizada en el búfer live.", "Observation count by severity across recent findings.": "Conteo de observaciones por severidad en hallazgos recientes.", "Current monitor lifecycle states.": "Estados actuales del ciclo de vida de monitores.", "Active and completed event-driven monitors.": "Monitores por eventos activos y completados.", "Latest observation results.": "Últimos resultados de observación.", "Event patterns registered with the monitor event bridge.": "Patrones de eventos registrados en el puente de monitores.",
      "Triggers": "Disparadores", "Watches": "Vigilancias", "Pattern": "Patrón", "Triggered": "Disparado", "Last event": "Último evento", "Value": "Valor", "Dimension": "Dimensión", "Signal": "Señal", "armed": "armado", "done": "terminado", "suspended": "suspendido", "yes": "sí", "no": "no",
      "signal.family.fs": "archivos", "signal.family.gateway": "gateway", "signal.family.ui": "interfaz", "signal.family.clipboard": "portapapeles", "signal.family.app": "aplicación", "signal.family.unknown": "desconocido", "signal.family.hw": "hardware",
      "Physical bench": "Banco físico", "Devices": "Dispositivos", "Charted channels": "Canales graficados",
      "Recent events": "Eventos recientes", "Unpersisted windows": "Ventanas no persistidas",
      "Raw samples written": "Muestras brutas escritas", "Watch state": "Estado del monitor",
      "Channel traces": "Trazas de canal", "Sampled channels": "Canales muestreados",
      "Envelope conformance": "Conformidad con la envolvente", "Window conformance": "Conformidad de ventanas",
      "Device events": "Eventos del dispositivo", "Sampling health": "Salud del muestreo",
      "Learned command outcomes": "Resultados de comandos aprendidos", "inside": "dentro",
      "near": "cerca del límite", "outside": "fuera", "unknown": "desconocido"
    },
    ar: {
      "All": "الكل", "connecting…": "جارٍ الاتصال", "live": "متصل", "reconnecting…": "جارٍ إعادة الاتصال", "seconds ago": "قبل {count} ث", "minutes ago": "قبل {count} د", "hours ago": "قبل {count} س",
      "Showing {shown} of {total} recent events.": "عرض {shown} من أصل {total} حدثاً حديثاً.", "Showing {shown} of {total} {family} events.": "عرض {shown} من أصل {total} من أحداث {family}.",
      "stale build": "بناء قديم", "stale_build_title": "خادم LeapBoard (pid {pid}) أقدم من شجرة المصدر الحالية. أعد تشغيله لتحميل التغييرات الأخيرة.",
      "Stream events": "أحداث التدفق", "Active watches": "المراقبات النشطة", "Watch portfolio": "محفظة المراقبات", "Noise suppressed": "الضجيج المحجوب", "Source dropped": "مصدر مُسقط", "Reorder pending": "إعادة الترتيب معلقة",
      "Signal health summary": "ملخص صحة الإشارات", "Ingress": "الدخول", "Pressure": "الضغط", "Recent event families": "عائلات الأحداث الأخيرة", "Finding severity mix": "توزيع شدة النتائج", "Watch state mix": "توزيع حالات المراقبة", "Watch states": "حالات المراقبة", "Trigger coverage": "تغطية المُشغّلات",
      "Latest daemon events · grouped by signal family · newest first.": "أحدث أحداث daemon · مجمعة حسب عائلة الإشارة · الأحدث أولاً.", "Ingress fan-out, pipeline pressure, and recent dimensional mix.": "تفرع الدخول وضغط الأنبوب وتوزيع الأبعاد الأخير.", "Event count by normalized family in the live ring buffer.": "عدد الأحداث حسب العائلة الموحدة في المخزن الحلقي المباشر.", "Observation count by severity across recent findings.": "عدد الملاحظات حسب الشدة في النتائج الأخيرة.", "Current monitor lifecycle states.": "حالات دورة حياة المراقبات الحالية.", "Active and completed event-driven monitors.": "المراقبات الحدثية النشطة والمكتملة.", "Latest observation results.": "أحدث نتائج الرصد.", "Event patterns registered with the monitor event bridge.": "أنماط الأحداث المسجلة في جسر أحداث المراقبة.",
      "Triggers": "المُشغّلات", "Watches": "المراقبات", "Pattern": "النمط", "Triggered": "تم التشغيل", "Last event": "آخر حدث", "Value": "القيمة", "Dimension": "البعد", "Signal": "الإشارة", "armed": "مسلح", "done": "منتهي", "suspended": "معلق", "yes": "نعم", "no": "لا",
      "signal.family.fs": "ملفات", "signal.family.gateway": "بوابة", "signal.family.ui": "واجهة", "signal.family.clipboard": "الحافظة", "signal.family.app": "تطبيق", "signal.family.unknown": "مجهول", "signal.family.hw": "عتاد",
      "Physical bench": "المنصة الفيزيائية", "Devices": "الأجهزة", "Charted channels": "القنوات المرسومة",
      "Recent events": "الأحداث الأخيرة", "Unpersisted windows": "نوافذ غير محفوظة",
      "Raw samples written": "العينات الخام المكتوبة", "Watch state": "حالة المراقبة",
      "Channel traces": "مسارات القنوات", "Sampled channels": "القنوات المُعيَّنة",
      "Envelope conformance": "مطابقة الحدود", "Window conformance": "مطابقة النوافذ",
      "Device events": "أحداث الجهاز", "Sampling health": "سلامة أخذ العينات",
      "Learned command outcomes": "نتائج الأوامر المُتعلَّمة", "inside": "داخل الحدود",
      "near": "قريب من الحد", "outside": "خارج الحدود", "unknown": "مجهول"
    },
    ru: {
      "All": "Все", "connecting…": "подключение", "live": "подключено", "reconnecting…": "переподключение", "seconds ago": "{count} с назад", "minutes ago": "{count} мин назад", "hours ago": "{count} ч назад",
      "Showing {shown} of {total} recent events.": "Показано {shown} из {total} последних событий.", "Showing {shown} of {total} {family} events.": "Показано {shown} из {total} событий {family}.",
      "stale build": "устаревшая сборка", "stale_build_title": "Сервер LeapBoard (pid {pid}) старее текущего дерева исходников. Перезапустите его, чтобы применить изменения.",
      "Stream events": "События потока", "Active watches": "Активные наблюдения", "Watch portfolio": "Портфель наблюдений", "Noise suppressed": "Шум подавлен", "Source dropped": "Источник отброшен", "Reorder pending": "Ожидает сортировки",
      "Signal health summary": "Сводка здоровья сигналов", "Ingress": "Вход", "Pressure": "Давление", "Recent event families": "Недавние семейства событий", "Finding severity mix": "Важность находок", "Watch state mix": "Состояния наблюдений", "Watch states": "Состояния наблюдений", "Trigger coverage": "Покрытие триггеров",
      "Latest daemon events · grouped by signal family · newest first.": "Последние события daemon · по семействам сигналов · новые первыми.", "Ingress fan-out, pipeline pressure, and recent dimensional mix.": "Входной fan-out, давление конвейера и недавнее распределение измерений.", "Event count by normalized family in the live ring buffer.": "Число событий по нормализованным семействам в live-буфере.", "Observation count by severity across recent findings.": "Число наблюдений по важности среди последних находок.", "Current monitor lifecycle states.": "Текущие состояния жизненного цикла мониторов.", "Active and completed event-driven monitors.": "Активные и завершённые событийные мониторы.", "Latest observation results.": "Последние результаты наблюдений.", "Event patterns registered with the monitor event bridge.": "Шаблоны событий, зарегистрированные в мосте мониторов.",
      "Triggers": "Триггеры", "Watches": "Наблюдения", "Pattern": "Шаблон", "Triggered": "Сработал", "Last event": "Последнее событие", "Value": "Значение", "Dimension": "Измерение", "Signal": "Сигнал", "armed": "взведено", "done": "готово", "suspended": "приостановлено", "yes": "да", "no": "нет",
      "signal.family.fs": "файлы", "signal.family.gateway": "шлюз", "signal.family.ui": "интерфейс", "signal.family.clipboard": "буфер", "signal.family.app": "приложение", "signal.family.unknown": "неизвестно", "signal.family.hw": "оборудование",
      "Physical bench": "Физический стенд", "Devices": "Устройства", "Charted channels": "Каналы на графике",
      "Recent events": "Недавние события", "Unpersisted windows": "Несохранённые окна",
      "Raw samples written": "Записано сырых отсчётов", "Watch state": "Состояние наблюдения",
      "Channel traces": "Трассы каналов", "Sampled channels": "Опрашиваемые каналы",
      "Envelope conformance": "Соответствие допускам", "Window conformance": "Соответствие окон",
      "Device events": "События устройства", "Sampling health": "Состояние опроса",
      "Learned command outcomes": "Изученные результаты команд", "inside": "в допуске",
      "near": "у границы", "outside": "вне допуска", "unknown": "неизвестно"
    }
  };
  const I18N_TEMPLATES = {
    // Every literal a board template renders, per locale. Held apart from I18N and
    // I18N_PATCH because those two grew with the first two lenses and were never
    // extended: five of seven templates shipped untranslated in every language, and
    // the i18n test only checked signal keys, so nothing failed. Keyed by the English
    // source string, so an untranslated key still renders readable English.
    zh: {"A ratio below 1.0 means the sampling loop is not keeping its declared cadence.": "比值低于 1.0 表示采样循环未能维持其声明的节奏。", "Action": "动作", "After": "变更后", "An unverified declaration has its writable channels demoted to read-only.": "未核验的声明，其可写通道会被降级为只读。", "Approval": "审批", "Autonomous governance": "自主治理", "Autonomy": "自主级别", "Before": "变更前", "Calibrated at": "校准时间", "Calibration health": "校准健康度", "Calls (decisions)": "观点（决策）", "Candlestick": "K 线", "Capability": "能力", "Capability adaptation": "能力适配", "Channel": "通道", "Channels": "通道数", "Channels that have never been calibrated or whose calibration has expired are shown first.": "从未校准或校准已过期的通道排在最前。", "Command": "命令", "Commanded versus observed, best tracking first": "命令值与实测值对比，跟随最好者在前", "Concerns (open questions)": "关切（待答问题）", "Counted across every charted channel. 'near' means within 5% of a declared bound.": "统计所有绘制通道。“接近”指处于声明边界的 5% 以内。", "Days since": "距今天数", "Decisions read as calls; action items as the execution checklist.": "决策即观点，行动项即执行清单。", "Declared Hz": "声明频率 (Hz)", "Desk brief": "交易台简报", "Device": "设备", "Dropped samples": "丢弃的样本", "Entities as references, and recommended next prompts to advance the work.": "实体作为参考，并给出推进工作的后续追问。", "Entities in play and the open risks still to resolve.": "涉及的实体，以及尚未解决的敞口风险。", "Envelope, rate, staleness and quality observations · newest first": "包络、速率、失联与质量观测 · 最新在前", "Environment": "环境", "Environment, selected plugin tools, and orchestration order.": "环境、已选插件工具及编排顺序。", "Events paced out": "被配速抑制的事件", "Evidence": "证据", "Executable": "可执行", "Execution checklist": "执行清单", "Extracted from this session's tool/file output (not model-generated).": "数据来自本次会话的工具/文件产物（非模型生成）。", "Failures": "失败次数", "Finance lens": "金融视图", "Follow-ups": "后续事项", "Halt": "可急停", "How often each window sat inside, near, or outside its declared limits": "各窗口处于声明限值内、接近边界或越界的频次", "Inquiry brief": "研究简报", "Insights carded as evidence, capped for fast review.": "洞察以证据卡呈现，数量受限以便快速浏览。", "Instruments & counterparties": "标的与交易对手", "Latest capability decision": "最新能力决策", "Lifecycle timeline": "生命周期时间线", "Line of inquiry": "研究主线", "Location": "位置", "Loop phase": "循环阶段", "Mean of each downsample window. Declared limits are listed per channel below.": "每个降采样窗口的均值。各通道的声明限值见下方。", "Mutation": "变更", "Narrative": "叙事", "Narrative pulse": "叙事脉搏", "Next recal due": "下次校准期限", "Normalized error": "归一化误差", "Normalized error is the residual as a share of the channel's declared span.": "归一化误差是残差占该通道声明量程的比例。", "OHLC extracted from captured session market data.": "OHLC 提取自本次会话捕获的行情数据。", "Observation backlog, proposal state, policy decisions, and lifecycle outcomes.": "观测待办、提案状态、策略决策与生命周期结果。", "Observations": "观测数", "Observed Hz": "实测频率 (Hz)", "Observed rate against declared rate": "实测速率与声明速率对比", "Open": "已连接", "Open risks": "敞口风险", "Open/high/low/close from captured tool output.": "开/高/低/收，来自捕获的工具输出。", "Origin": "来源", "Outcome": "结果", "Per-channel calibration state, freshness, and residual correction": "各通道的校准状态、时效性与残差校正", "Plan": "计划", "Plan steps": "计划步骤", "Plugin": "插件", "Policy": "策略", "Positions & actions": "持仓与操作", "Price action": "价格行为", "Proposal": "提案", "Proposal status": "提案状态", "Pulse": "脉搏", "Ratio": "比值", "References & follow-ups": "参考与后续", "References (entities)": "参考（实体）", "Registry delta": "注册表变化", "Representative observations, capped for quick scanning.": "代表性观察，数量受限以便快速浏览。", "Requirements": "能力需求", "Research lens": "研究视图", "Residual": "残差", "Sampled history per channel, newest on the right": "按通道的采样历史，最新在右侧", "Selection delta": "选择变化", "Sentiment lens": "情绪视图", "Series": "序列", "Session analysis": "会话分析", "Signal strength": "信号强度", "Skipped slots": "跳过的采样点", "State": "状态", "Storyline and signal strength before drilling into positions and actions.": "先看叙事与信号强度，再深入持仓与操作。", "Streaming": "采样中", "The line of investigation and where the open questions concentrate.": "研究主线，以及待答问题的集中之处。", "The narrative arc and how strongly themes are trending.": "叙事走向，以及主题的趋势强度。", "Theme intensity": "主题强度", "Themes": "主题", "Tool": "工具", "Transport": "传输方式", "Transport, provenance and channel counts": "传输方式、来源与通道数量", "Trust": "信任级别", "Verified": "已核验", "Voices & concerns": "声音与关切", "Watchlist": "关注列表", "Who/what is in the conversation, and the concerns still open.": "谁/什么在被讨论，以及尚未解决的关切。", "Writable": "可写"},
    fr: {"A ratio below 1.0 means the sampling loop is not keeping its declared cadence.": "Un ratio inférieur à 1,0 signifie que la boucle d’échantillonnage ne tient pas sa cadence déclarée.", "Action": "Action", "After": "Après", "An unverified declaration has its writable channels demoted to read-only.": "Une déclaration non vérifiée voit ses canaux inscriptibles rétrogradés en lecture seule.", "Approval": "Approbation", "Autonomous governance": "Gouvernance autonome", "Autonomy": "Autonomie", "Before": "Avant", "Calibrated at": "Calibré le", "Calibration health": "État de calibration", "Calls (decisions)": "Recommandations (décisions)", "Candlestick": "Chandeliers", "Capability": "Capacité", "Capability adaptation": "Adaptation des capacités", "Channel": "Canal", "Channels": "Canaux", "Channels that have never been calibrated or whose calibration has expired are shown first.": "Les canaux jamais calibrés ou dont la calibration a expiré apparaissent en premier.", "Command": "Commande", "Commanded versus observed, best tracking first": "Commandé contre observé, meilleur suivi d’abord", "Concerns (open questions)": "Préoccupations (questions ouvertes)", "Counted across every charted channel. 'near' means within 5% of a declared bound.": "Compté sur tous les canaux tracés. « près » signifie à moins de 5 % d’une borne déclarée.", "Days since": "Jours écoulés", "Decisions read as calls; action items as the execution checklist.": "Les décisions se lisent comme des recommandations ; les actions comme la liste d’exécution.", "Declared Hz": "Hz déclarés", "Desk brief": "Note de desk", "Device": "Appareil", "Dropped samples": "Échantillons perdus", "Entities as references, and recommended next prompts to advance the work.": "Entités comme références, et invites suivantes recommandées pour avancer.", "Entities in play and the open risks still to resolve.": "Entités concernées et risques ouverts à résoudre.", "Envelope, rate, staleness and quality observations · newest first": "Observations d’enveloppe, de débit, d’obsolescence et de qualité · les plus récentes d’abord", "Environment": "Environnement", "Environment, selected plugin tools, and orchestration order.": "Environnement, outils de plugin sélectionnés et ordre d’orchestration.", "Events paced out": "Événements limités", "Evidence": "Preuve", "Executable": "Exécutable", "Execution checklist": "Liste d’exécution", "Extracted from this session's tool/file output (not model-generated).": "Extrait des sorties d’outils/fichiers de cette session (non généré par le modèle).", "Failures": "Échecs", "Finance lens": "Vue finance", "Follow-ups": "Suivis", "Halt": "Arrêt", "How often each window sat inside, near, or outside its declared limits": "Fréquence à laquelle chaque fenêtre était dans, près de, ou hors de ses limites déclarées", "Inquiry brief": "Note d’enquête", "Insights carded as evidence, capped for fast review.": "Analyses présentées comme preuves, limitées pour une revue rapide.", "Instruments & counterparties": "Instruments et contreparties", "Latest capability decision": "Dernière décision de capacité", "Lifecycle timeline": "Chronologie du cycle de vie", "Line of inquiry": "Ligne d’enquête", "Location": "Emplacement", "Loop phase": "Phase de boucle", "Mean of each downsample window. Declared limits are listed per channel below.": "Moyenne de chaque fenêtre de sous-échantillonnage. Les limites déclarées figurent par canal ci-dessous.", "Mutation": "Mutation", "Narrative": "Récit", "Narrative pulse": "Pouls narratif", "Next recal due": "Prochaine recalibration", "Normalized error": "Erreur normalisée", "Normalized error is the residual as a share of the channel's declared span.": "L’erreur normalisée est le résidu en proportion de l’étendue déclarée du canal.", "OHLC extracted from captured session market data.": "OHLC extrait des données de marché capturées durant la session.", "Observation backlog, proposal state, policy decisions, and lifecycle outcomes.": "File d’observations, état des propositions, décisions de politique et résultats du cycle de vie.", "Observations": "Observations", "Observed Hz": "Hz observés", "Observed rate against declared rate": "Débit observé par rapport au débit déclaré", "Open": "Ouvert", "Open risks": "Risques ouverts", "Open/high/low/close from captured tool output.": "Ouverture/haut/bas/clôture issus des sorties d’outils capturées.", "Origin": "Origine", "Outcome": "Résultat", "Per-channel calibration state, freshness, and residual correction": "État de calibration, fraîcheur et correction résiduelle par canal", "Plan": "Plan", "Plan steps": "Étapes du plan", "Plugin": "Plugin", "Policy": "Politique", "Positions & actions": "Positions et actions", "Price action": "Action des prix", "Proposal": "Proposition", "Proposal status": "Statut de la proposition", "Pulse": "Pouls", "Ratio": "Ratio", "References & follow-ups": "Références et suivis", "References (entities)": "Références (entités)", "Registry delta": "Delta du registre", "Representative observations, capped for quick scanning.": "Observations représentatives, limitées pour une lecture rapide.", "Requirements": "Exigences", "Research lens": "Vue recherche", "Residual": "Résidu", "Sampled history per channel, newest on the right": "Historique échantillonné par canal, le plus récent à droite", "Selection delta": "Delta de sélection", "Sentiment lens": "Vue sentiment", "Series": "Série", "Session analysis": "Analyse de session", "Signal strength": "Force du signal", "Skipped slots": "Créneaux manqués", "State": "État", "Storyline and signal strength before drilling into positions and actions.": "Récit et force du signal avant d’examiner positions et actions.", "Streaming": "Diffusion", "The line of investigation and where the open questions concentrate.": "La ligne d’investigation et où se concentrent les questions ouvertes.", "The narrative arc and how strongly themes are trending.": "L’arc narratif et l’intensité des tendances thématiques.", "Theme intensity": "Intensité des thèmes", "Themes": "Thèmes", "Tool": "Outil", "Transport": "Transport", "Transport, provenance and channel counts": "Transport, provenance et nombre de canaux", "Trust": "Confiance", "Verified": "Vérifié", "Voices & concerns": "Voix et préoccupations", "Watchlist": "Liste de suivi", "Who/what is in the conversation, and the concerns still open.": "Qui/quoi est dans la conversation, et les préoccupations encore ouvertes.", "Writable": "Inscriptible"},
    es: {"A ratio below 1.0 means the sampling loop is not keeping its declared cadence.": "Una relación inferior a 1,0 significa que el bucle de muestreo no mantiene su cadencia declarada.", "Action": "Acción", "After": "Después", "An unverified declaration has its writable channels demoted to read-only.": "Una declaración no verificada degrada sus canales escribibles a solo lectura.", "Approval": "Aprobación", "Autonomous governance": "Gobernanza autónoma", "Autonomy": "Autonomía", "Before": "Antes", "Calibrated at": "Calibrado el", "Calibration health": "Estado de calibración", "Calls (decisions)": "Recomendaciones (decisiones)", "Candlestick": "Velas", "Capability": "Capacidad", "Capability adaptation": "Adaptación de capacidades", "Channel": "Canal", "Channels": "Canales", "Channels that have never been calibrated or whose calibration has expired are shown first.": "Los canales nunca calibrados o con calibración vencida se muestran primero.", "Command": "Comando", "Commanded versus observed, best tracking first": "Comandado frente a observado, mejor seguimiento primero", "Concerns (open questions)": "Inquietudes (preguntas abiertas)", "Counted across every charted channel. 'near' means within 5% of a declared bound.": "Contado en todos los canales graficados. «cerca» significa dentro del 5 % de un límite declarado.", "Days since": "Días desde", "Decisions read as calls; action items as the execution checklist.": "Las decisiones se leen como recomendaciones; las acciones como la lista de ejecución.", "Declared Hz": "Hz declarados", "Desk brief": "Informe de mesa", "Device": "Dispositivo", "Dropped samples": "Muestras descartadas", "Entities as references, and recommended next prompts to advance the work.": "Entidades como referencias y siguientes preguntas recomendadas para avanzar.", "Entities in play and the open risks still to resolve.": "Entidades implicadas y riesgos abiertos por resolver.", "Envelope, rate, staleness and quality observations · newest first": "Observaciones de envolvente, tasa, obsolescencia y calidad · las más recientes primero", "Environment": "Entorno", "Environment, selected plugin tools, and orchestration order.": "Entorno, herramientas de plugin seleccionadas y orden de orquestación.", "Events paced out": "Eventos limitados", "Evidence": "Evidencia", "Executable": "Ejecutable", "Execution checklist": "Lista de ejecución", "Extracted from this session's tool/file output (not model-generated).": "Extraído de la salida de herramientas/archivos de esta sesión (no generado por el modelo).", "Failures": "Fallos", "Finance lens": "Vista financiera", "Follow-ups": "Seguimientos", "Halt": "Parada", "How often each window sat inside, near, or outside its declared limits": "Con qué frecuencia cada ventana estuvo dentro, cerca o fuera de sus límites declarados", "Inquiry brief": "Informe de indagación", "Insights carded as evidence, capped for fast review.": "Hallazgos presentados como evidencia, limitados para revisión rápida.", "Instruments & counterparties": "Instrumentos y contrapartes", "Latest capability decision": "Última decisión de capacidad", "Lifecycle timeline": "Cronología del ciclo de vida", "Line of inquiry": "Línea de indagación", "Location": "Ubicación", "Loop phase": "Fase del bucle", "Mean of each downsample window. Declared limits are listed per channel below.": "Media de cada ventana de submuestreo. Los límites declarados se listan por canal abajo.", "Mutation": "Mutación", "Narrative": "Narrativa", "Narrative pulse": "Pulso narrativo", "Next recal due": "Próxima recalibración", "Normalized error": "Error normalizado", "Normalized error is the residual as a share of the channel's declared span.": "El error normalizado es el residuo como fracción del rango declarado del canal.", "OHLC extracted from captured session market data.": "OHLC extraído de los datos de mercado capturados en la sesión.", "Observation backlog, proposal state, policy decisions, and lifecycle outcomes.": "Cola de observaciones, estado de propuestas, decisiones de política y resultados del ciclo de vida.", "Observations": "Observaciones", "Observed Hz": "Hz observados", "Observed rate against declared rate": "Tasa observada frente a la tasa declarada", "Open": "Abierto", "Open risks": "Riesgos abiertos", "Open/high/low/close from captured tool output.": "Apertura/máximo/mínimo/cierre desde la salida de herramientas capturada.", "Origin": "Origen", "Outcome": "Resultado", "Per-channel calibration state, freshness, and residual correction": "Estado de calibración, vigencia y corrección residual por canal", "Plan": "Plan", "Plan steps": "Pasos del plan", "Plugin": "Plugin", "Policy": "Política", "Positions & actions": "Posiciones y acciones", "Price action": "Acción del precio", "Proposal": "Propuesta", "Proposal status": "Estado de la propuesta", "Pulse": "Pulso", "Ratio": "Relación", "References & follow-ups": "Referencias y seguimientos", "References (entities)": "Referencias (entidades)", "Registry delta": "Delta del registro", "Representative observations, capped for quick scanning.": "Observaciones representativas, limitadas para lectura rápida.", "Requirements": "Requisitos", "Research lens": "Vista de investigación", "Residual": "Residuo", "Sampled history per channel, newest on the right": "Historial muestreado por canal, el más reciente a la derecha", "Selection delta": "Delta de selección", "Sentiment lens": "Vista de sentimiento", "Series": "Serie", "Session analysis": "Análisis de sesión", "Signal strength": "Fuerza de la señal", "Skipped slots": "Ranuras omitidas", "State": "Estado", "Storyline and signal strength before drilling into positions and actions.": "Narrativa y fuerza de la señal antes de entrar en posiciones y acciones.", "Streaming": "Transmisión", "The line of investigation and where the open questions concentrate.": "La línea de investigación y dónde se concentran las preguntas abiertas.", "The narrative arc and how strongly themes are trending.": "El arco narrativo y con qué fuerza se mueven los temas.", "Theme intensity": "Intensidad temática", "Themes": "Temas", "Tool": "Herramienta", "Transport": "Transporte", "Transport, provenance and channel counts": "Transporte, procedencia y número de canales", "Trust": "Confianza", "Verified": "Verificado", "Voices & concerns": "Voces e inquietudes", "Watchlist": "Lista de seguimiento", "Who/what is in the conversation, and the concerns still open.": "Quién/qué está en la conversación y las inquietudes aún abiertas.", "Writable": "Escribible"},
    ar: {"A ratio below 1.0 means the sampling loop is not keeping its declared cadence.": "نسبة أقل من 1.0 تعني أن حلقة أخذ العينات لا تحافظ على وتيرتها المعلنة.", "Action": "الإجراء", "After": "بعد", "An unverified declaration has its writable channels demoted to read-only.": "الإعلان غير المُتحقَّق منه تُخفَّض قنواته القابلة للكتابة إلى القراءة فقط.", "Approval": "الموافقة", "Autonomous governance": "الحكم الذاتي", "Autonomy": "الاستقلالية", "Before": "قبل", "Calibrated at": "تاريخ المعايرة", "Calibration health": "سلامة المعايرة", "Calls (decisions)": "التوصيات (القرارات)", "Candlestick": "الشموع", "Capability": "القدرة", "Capability adaptation": "تكييف القدرات", "Channel": "القناة", "Channels": "القنوات", "Channels that have never been calibrated or whose calibration has expired are shown first.": "تظهر أولاً القنوات التي لم تُعاير قط أو التي انتهت صلاحية معايرتها.", "Command": "الأمر", "Commanded versus observed, best tracking first": "المأمور مقابل المرصود، الأفضل تتبعاً أولاً", "Concerns (open questions)": "المخاوف (أسئلة مفتوحة)", "Counted across every charted channel. 'near' means within 5% of a declared bound.": "محسوب على كل قناة مرسومة. \"قريب\" تعني داخل 5% من حد معلن.", "Days since": "الأيام المنقضية", "Decisions read as calls; action items as the execution checklist.": "القرارات تُقرأ كتوصيات؛ والإجراءات كقائمة تنفيذ.", "Declared Hz": "الهرتز المعلن", "Desk brief": "موجز المكتب", "Device": "الجهاز", "Dropped samples": "العينات المفقودة", "Entities as references, and recommended next prompts to advance the work.": "الكيانات كمراجع، والمطالبات التالية الموصى بها لدفع العمل.", "Entities in play and the open risks still to resolve.": "الكيانات المعنية والمخاطر المفتوحة التي لم تُحل.", "Envelope, rate, staleness and quality observations · newest first": "رصدات المغلف والمعدل والتقادم والجودة · الأحدث أولاً", "Environment": "البيئة", "Environment, selected plugin tools, and orchestration order.": "البيئة والأدوات المختارة وترتيب التنسيق.", "Events paced out": "الأحداث المُقيَّدة", "Evidence": "الدليل", "Executable": "قابل للتنفيذ", "Execution checklist": "قائمة التنفيذ", "Extracted from this session's tool/file output (not model-generated).": "مستخرج من مخرجات الأدوات/الملفات في هذه الجلسة (ليس من إنشاء النموذج).", "Failures": "الأعطال", "Finance lens": "منظور مالي", "Follow-ups": "المتابعات", "Halt": "إيقاف", "How often each window sat inside, near, or outside its declared limits": "عدد المرات التي كانت فيها كل نافذة داخل حدودها المعلنة أو قريبة منها أو خارجها", "Inquiry brief": "موجز الاستقصاء", "Insights carded as evidence, capped for fast review.": "الرؤى معروضة كأدلة، ومحدودة العدد للمراجعة السريعة.", "Instruments & counterparties": "الأدوات والأطراف المقابلة", "Latest capability decision": "أحدث قرار للقدرات", "Lifecycle timeline": "الخط الزمني لدورة الحياة", "Line of inquiry": "خط الاستقصاء", "Location": "الموقع", "Loop phase": "مرحلة الحلقة", "Mean of each downsample window. Declared limits are listed per channel below.": "متوسط كل نافذة تخفيض للعينات. الحدود المعلنة مدرجة لكل قناة أدناه.", "Mutation": "التغيير", "Narrative": "السرد", "Narrative pulse": "نبض السرد", "Next recal due": "موعد إعادة المعايرة", "Normalized error": "الخطأ المعياري", "Normalized error is the residual as a share of the channel's declared span.": "الخطأ المعياري هو المتبقي كنسبة من المدى المعلن للقناة.", "OHLC extracted from captured session market data.": "OHLC مستخرج من بيانات السوق المسجلة في الجلسة.", "Observation backlog, proposal state, policy decisions, and lifecycle outcomes.": "قائمة الرصد وحالة المقترحات وقرارات السياسة ونتائج دورة الحياة.", "Observations": "الرصدات", "Observed Hz": "الهرتز المرصود", "Observed rate against declared rate": "المعدل المرصود مقابل المعدل المعلن", "Open": "مفتوح", "Open risks": "المخاطر المفتوحة", "Open/high/low/close from captured tool output.": "الافتتاح/الأعلى/الأدنى/الإغلاق من مخرجات الأدوات المسجلة.", "Origin": "المصدر", "Outcome": "النتيجة", "Per-channel calibration state, freshness, and residual correction": "حالة المعايرة وحداثتها وتصحيح المتبقي لكل قناة", "Plan": "الخطة", "Plan steps": "خطوات الخطة", "Plugin": "الملحق", "Policy": "السياسة", "Positions & actions": "المراكز والإجراءات", "Price action": "حركة السعر", "Proposal": "المقترح", "Proposal status": "حالة المقترح", "Pulse": "النبض", "Ratio": "النسبة", "References & follow-ups": "المراجع والمتابعات", "References (entities)": "المراجع (الكيانات)", "Registry delta": "فرق السجل", "Representative observations, capped for quick scanning.": "رصدات تمثيلية، محدودة العدد للقراءة السريعة.", "Requirements": "المتطلبات", "Research lens": "منظور بحثي", "Residual": "المتبقي", "Sampled history per channel, newest on the right": "سجل العينات لكل قناة، الأحدث على اليمين", "Selection delta": "فرق الاختيار", "Sentiment lens": "منظور المشاعر", "Series": "السلسلة", "Session analysis": "تحليل الجلسة", "Signal strength": "قوة الإشارة", "Skipped slots": "الفتحات المتخطاة", "State": "الحالة", "Storyline and signal strength before drilling into positions and actions.": "السرد وقوة الإشارة قبل التوسع في المراكز والإجراءات.", "Streaming": "بث", "The line of investigation and where the open questions concentrate.": "خط البحث وأين تتركز الأسئلة المفتوحة.", "The narrative arc and how strongly themes are trending.": "قوس السرد ومدى قوة اتجاه الموضوعات.", "Theme intensity": "شدة الموضوعات", "Themes": "الموضوعات", "Tool": "الأداة", "Transport": "النقل", "Transport, provenance and channel counts": "النقل والمنشأ وعدد القنوات", "Trust": "الثقة", "Verified": "مُتحقَّق", "Voices & concerns": "الأصوات والمخاوف", "Watchlist": "قائمة المتابعة", "Who/what is in the conversation, and the concerns still open.": "من/ما هو في المحادثة، والمخاوف التي لا تزال مفتوحة.", "Writable": "قابل للكتابة"},
    ru: {"A ratio below 1.0 means the sampling loop is not keeping its declared cadence.": "Отношение ниже 1,0 означает, что цикл выборки не выдерживает объявленный ритм.", "Action": "Действие", "After": "После", "An unverified declaration has its writable channels demoted to read-only.": "У непроверенного объявления записываемые каналы понижаются до только чтения.", "Approval": "Согласование", "Autonomous governance": "Автономное управление", "Autonomy": "Автономность", "Before": "До", "Calibrated at": "Калиброван", "Calibration health": "Состояние калибровки", "Calls (decisions)": "Рекомендации (решения)", "Candlestick": "Свечи", "Capability": "Возможность", "Capability adaptation": "Адаптация возможностей", "Channel": "Канал", "Channels": "Каналы", "Channels that have never been calibrated or whose calibration has expired are shown first.": "Каналы, которые никогда не калибровались или чья калибровка истекла, показаны первыми.", "Command": "Команда", "Commanded versus observed, best tracking first": "Заданное против наблюдаемого, лучшее отслеживание первым", "Concerns (open questions)": "Опасения (открытые вопросы)", "Counted across every charted channel. 'near' means within 5% of a declared bound.": "Подсчитано по всем отображаемым каналам. «У границы» — в пределах 5% от объявленного предела.", "Days since": "Дней с тех пор", "Decisions read as calls; action items as the execution checklist.": "Решения читаются как рекомендации; действия — как чек-лист исполнения.", "Declared Hz": "Объявл. Гц", "Desk brief": "Сводка деска", "Device": "Устройство", "Dropped samples": "Отброшенные образцы", "Entities as references, and recommended next prompts to advance the work.": "Сущности как ссылки и рекомендуемые следующие запросы.", "Entities in play and the open risks still to resolve.": "Задействованные сущности и нерешённые риски.", "Envelope, rate, staleness and quality observations · newest first": "Наблюдения по огибающей, частоте, устареванию и качеству · сначала новые", "Environment": "Окружение", "Environment, selected plugin tools, and orchestration order.": "Окружение, выбранные инструменты плагинов и порядок оркестрации.", "Events paced out": "Событий подавлено", "Evidence": "Обоснование", "Executable": "Исполнимо", "Execution checklist": "Чек-лист исполнения", "Extracted from this session's tool/file output (not model-generated).": "Извлечено из вывода инструментов/файлов этой сессии (не сгенерировано моделью).", "Failures": "Сбои", "Finance lens": "Финансовый ракурс", "Follow-ups": "Продолжения", "Halt": "Останов", "How often each window sat inside, near, or outside its declared limits": "Как часто каждое окно было внутри, у границы или вне объявленных пределов", "Inquiry brief": "Сводка исследования", "Insights carded as evidence, capped for fast review.": "Инсайты как карточки-обоснования, ограничены для быстрого просмотра.", "Instruments & counterparties": "Инструменты и контрагенты", "Latest capability decision": "Последнее решение о возможностях", "Lifecycle timeline": "Хронология жизненного цикла", "Line of inquiry": "Линия исследования", "Location": "Расположение", "Loop phase": "Фаза цикла", "Mean of each downsample window. Declared limits are listed per channel below.": "Среднее по каждому окну прореживания. Объявленные пределы указаны по каналам ниже.", "Mutation": "Изменение", "Narrative": "Сюжет", "Narrative pulse": "Нарративный пульс", "Next recal due": "Следующая рекалибровка", "Normalized error": "Нормированная ошибка", "Normalized error is the residual as a share of the channel's declared span.": "Нормированная ошибка — остаток как доля объявленного диапазона канала.", "OHLC extracted from captured session market data.": "OHLC извлечён из рыночных данных, записанных в сессии.", "Observation backlog, proposal state, policy decisions, and lifecycle outcomes.": "Очередь наблюдений, состояние предложений, решения политики и итоги жизненного цикла.", "Observations": "Наблюдения", "Observed Hz": "Наблюд. Гц", "Observed rate against declared rate": "Наблюдаемая частота против объявленной", "Open": "Открыт", "Open risks": "Открытые риски", "Open/high/low/close from captured tool output.": "Открытие/максимум/минимум/закрытие из записанного вывода инструментов.", "Origin": "Источник", "Outcome": "Результат", "Per-channel calibration state, freshness, and residual correction": "Состояние калибровки, актуальность и остаточная поправка по каналам", "Plan": "План", "Plan steps": "Шаги плана", "Plugin": "Плагин", "Policy": "Политика", "Positions & actions": "Позиции и действия", "Price action": "Ценовое движение", "Proposal": "Предложение", "Proposal status": "Статус предложения", "Pulse": "Пульс", "Ratio": "Отношение", "References & follow-ups": "Ссылки и продолжения", "References (entities)": "Ссылки (сущности)", "Registry delta": "Изменение реестра", "Representative observations, capped for quick scanning.": "Показательные наблюдения, ограничены для быстрого просмотра.", "Requirements": "Требования", "Research lens": "Исследовательский ракурс", "Residual": "Остаток", "Sampled history per channel, newest on the right": "История выборок по каналам, самое новое справа", "Selection delta": "Изменение выбора", "Sentiment lens": "Ракурс тональности", "Series": "Серия", "Session analysis": "Анализ сессии", "Signal strength": "Сила сигнала", "Skipped slots": "Пропущенные слоты", "State": "Состояние", "Storyline and signal strength before drilling into positions and actions.": "Сюжет и сила сигнала до перехода к позициям и действиям.", "Streaming": "Потоковая передача", "The line of investigation and where the open questions concentrate.": "Линия исследования и где сосредоточены открытые вопросы.", "The narrative arc and how strongly themes are trending.": "Нарративная дуга и насколько сильно растут темы.", "Theme intensity": "Интенсивность тем", "Themes": "Темы", "Tool": "Инструмент", "Transport": "Транспорт", "Transport, provenance and channel counts": "Транспорт, происхождение и число каналов", "Trust": "Доверие", "Verified": "Проверено", "Voices & concerns": "Голоса и опасения", "Watchlist": "Список наблюдения", "Who/what is in the conversation, and the concerns still open.": "Кто/что в разговоре и какие опасения остаются.", "Writable": "Записываемый"}
  };
  Object.keys(I18N).concat(Object.keys(I18N_PATCH), Object.keys(I18N_TEMPLATES))
    .filter((lang, at, all) => all.indexOf(lang) === at)
    .forEach((lang) => {
      // Later sources win, so a locale-specific template string overrides the English
      // fallback while an absent one still resolves to readable English.
      I18N[lang] = Object.assign(
        {}, I18N.en || {}, I18N[lang] || {},
        I18N_PATCH[lang] || {}, I18N_TEMPLATES[lang] || {},
      );
    });

  function t(key) { return (I18N[locale] && I18N[locale][key]) || (I18N.en && I18N.en[key]) || key; }
  function tx(value) { return typeof value === "string" ? t(value) : value; }
  function fmt(key, vars) {
    return t(key).replace(/\{(\w+)\}/g, (_m, name) => (vars && vars[name] != null ? String(vars[name]) : ""));
  }

  function applyLocale() {
    document.documentElement.lang = locale === "zh" ? "zh-CN" : locale;
    document.documentElement.dir = locale === "ar" ? "rtl" : "ltr";
    if (localeEl) localeEl.value = locale;
    document.querySelectorAll("[data-i18n]").forEach((node) => { node.textContent = t(node.dataset.i18n); });
  }

  function setConnectionStatus(key) {
    if (!statusEl) return;
    statusEl.dataset.i18n = key;
    statusEl.textContent = t(key);
  }

  function api(path) {
    const url = new URL(path, location.origin);
    url.searchParams.set("token", TOKEN);
    return url.toString();
  }

  async function fetchView(intent) {
    var prevTemplate = current.template;
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
      renderServerHealth(spec.meta || {});
      // Manage signal auto-refresh lifecycle on template switch
      var newTemplate = (spec.meta && spec.meta.active_template) || current.template || "";
      if (newTemplate === "signals") {
        startSignalAutoRefresh();
        injectSignalRefreshBtn();
      } else if (prevTemplate === "signals" && newTemplate !== "signals") {
        stopSignalAutoRefresh();
      }
    } catch (err) {
      rootEl.innerHTML = '<div class="empty">' + esc(t("Failed to load view") + ": " + String(err)) + "</div>";
    }
  }

  // Long-lived-process staleness: warns when this dashboard server process
  // predates the current source tree (see leapflow.utils.build_info). Purely
  // informational — the page still renders whatever data the stale process
  // returns; this just tells the developer *why* it might look wrong.
  function renderServerHealth(meta) {
    if (!statusEl) return;
    var old = document.getElementById("server-stale-badge");
    if (old) old.remove();
    var server = meta.server;
    if (!server || server.stale !== true) return;
    var build = server.build || {};
    var badge = el("span", "server-stale-badge");
    badge.id = "server-stale-badge";
    badge.textContent = "\u26a0 " + t("stale build");
    badge.title = fmt("stale_build_title", { pid: build.pid || "?" });
    statusEl.insertAdjacentElement("afterend", badge);
  }

  // Template switcher: the current session, rendered through each lens.
  function renderNav(meta) {
    const nav = document.getElementById("nav");
    if (!nav) return;
    const hidden = new Set(Array.isArray(meta.hidden_templates) ? meta.hidden_templates : []);
    HIDDEN_NAV_TEMPLATES.forEach((name) => hidden.add(name));
    const seen = new Set();
    const names = (Array.isArray(meta.templates) ? meta.templates : []).filter((name) => {
      name = String(name || "");
      if (!name || hidden.has(name) || seen.has(name)) return false;
      seen.add(name);
      return true;
    });
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
    signalTimeline: renderSignalTimeline,
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

  // Layout helpers: template-driven grid column count and child spans, so a view
  // can compose dense asymmetric grids without introducing new component types.
  function _clampInt(value, lo, hi) { const n = parseInt(value, 10); return Number.isFinite(n) ? Math.max(lo, Math.min(hi, n)) : 0; }
  function gridCols(props) { const c = _clampInt(props.cols, 2, 6); return c ? " cols-" + c : ""; }
  function applySpan(dom, props) { const s = _clampInt(props.span, 2, 4); if (s && dom && dom.classList) dom.classList.add("span-" + s); }

  function signalFamily(item) {
    var raw = String((item && (item.family || item.event_type || item.title)) || "unknown").replace(":", ".");
    return raw.split(".", 1)[0] || "unknown";
  }

  function signalTimestamp(item) {
    var raw = Number(item && item.ts);
    if (!Number.isFinite(raw) || raw <= 0) return 0;
    return raw < 100000000000 ? raw * 1000 : raw;
  }

  function signalFamilyLabel(family) {
    const key = "signal.family." + String(family || "unknown");
    const label = t(key);
    return label === key ? String(family || "unknown") : label;
  }

  function signalTimeLabel(item) {
    var ms = signalTimestamp(item);
    if (!ms) return "--:--:--";
    var d = new Date(ms);
    var clock = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    var age = Math.max(0, Date.now() - ms);
    if (age < 60000) return clock + " · " + fmt("seconds ago", { count: Math.max(0, Math.round(age / 1000)) });
    if (age < 3600000) return clock + " · " + fmt("minutes ago", { count: Math.round(age / 60000) });
    return clock + " · " + fmt("hours ago", { count: Math.round(age / 3600000) });
  }

  function normalizeSignalItems(data) {
    return asArray(data).filter((it) => it && typeof it === "object")
      .map((it) => Object.assign({}, it, { family: signalFamily(it), _ts: signalTimestamp(it) }))
      .sort((a, b) => (b._ts || 0) - (a._ts || 0));
  }

  function signalCategories(items) {
    const counts = {}; items.forEach((it) => { counts[it.family] = (counts[it.family] || 0) + 1; });
    return [{ key: "all", label: t("All"), count: items.length }]
      .concat(Object.keys(counts).sort().map((key) => ({ key, label: signalFamilyLabel(key), count: counts[key] })));
  }

  function renderSignalTimeline(props) {
    const box = el("div", "signal-timeline");
    box._signalTimelineOptions = { maxItems: _clampInt(props.max_items || props.maxItems || 12, 1, 24) || 12 };
    renderSignalTimelineInto(box, props.data || []);
    window._signalStream = normalizeSignalItems(props.data || []);
    return box;
  }

  function renderSignalTimelineInto(box, data) {
    const opts = box._signalTimelineOptions || { maxItems: 12 };
    const items = normalizeSignalItems(data);
    const categories = signalCategories(items);
    var active = window._signalTimelineActiveFamily || box.dataset.activeFamily || "all";
    if (!categories.some((c) => c.key === active)) active = "all";
    window._signalTimelineActiveFamily = active;
    box.dataset.activeFamily = active;
    box.innerHTML = "";
    if (!items.length) { box.appendChild(el("div", "empty-inline", esc(t("No entries.")))); return box; }

    const tabs = el("div", "signal-tabs");
    categories.forEach((cat) => {
      const btn = el("button", "signal-tab" + (cat.key === active ? " active" : ""));
      btn.type = "button"; btn.textContent = cat.label + " " + cat.count;
      btn.addEventListener("click", () => { window._signalTimelineActiveFamily = cat.key; renderSignalTimelineInto(box, items); });
      tabs.appendChild(btn);
    });
    box.appendChild(tabs);

    const filtered = active === "all" ? items : items.filter((it) => it.family === active);
    const shown = filtered.slice(0, opts.maxItems);
    const list = el("div", "signal-stream-list timeline");
    shown.forEach((it) => {
      const row = el("div", "signal-row timeline-item sev-" + severityOf(it));
      const meta = el("div", "signal-event-meta");
      meta.appendChild(el("span", "signal-time", esc(signalTimeLabel(it))));
      meta.appendChild(el("span", "signal-family", esc(signalFamilyLabel(it.family))));
      row.appendChild(meta);
      row.appendChild(el("div", "timeline-title signal-type", esc(it.event_type || it.title || "")));
      if (it.source || it.summary) row.appendChild(el("div", "summary signal-source", esc(it.source || it.summary)));
      list.appendChild(row);
    });
    box.appendChild(list);
    const footer = el("div", "signal-stream-footer");
    footer.textContent = active === "all"
      ? fmt("Showing {shown} of {total} recent events.", { shown: shown.length, total: filtered.length })
      : fmt("Showing {shown} of {total} {family} events.", { shown: shown.length, total: filtered.length, family: signalFamilyLabel(active) });
    box.appendChild(footer);
    return box;
  }

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
    if (!items.length) return el("div", "empty-inline", esc(t("No entries.")));
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
    items.forEach((it) => ul.appendChild(el("li", null, esc(typeof it === "object" ? (it.title || it.summary || JSON.stringify(it)) : tx(it)))));
    return ul;
  }

  function renderGaugeValue(label, value) {
    const d = el("div", "stat gauge-stat");
    d.appendChild(el("div", "label", esc(tx(label || "Gauge"))));
    d.appendChild(el("div", "value", esc(value != null && value !== "" ? value : "\u2014")));
    return d;
  }

  function svgEl(tag) { return document.createElementNS("http://www.w3.org/2000/svg", tag); }

  // Distribution bars (label -> value) or, as a fallback, the severity mix of a
  // findings/insights array. Real values only — never synthetic.
  function renderChartBars(data, title) {
    const arr = asArray(data);
    let dist = null;
    if (arr.length && arr[0] && Array.isArray(arr[0].items)) dist = asArray(arr[0].items);
    else if (arr.length && arr.every((it) => it && typeof it === "object" && "value" in it && ("label" in it || "name" in it))) dist = arr;
    const d = el("div", "chart card");
    if (title) d.appendChild(el("div", "card-title", esc(tx(title))));
    let rows; let severity = false;
    if (dist) {
      rows = dist.map((it) => ({ key: "", label: String(it.label || it.name || ""), value: Number(it.value) || 0 }));
    } else {
      severity = true;
      const counts = severityCounts(data);
      rows = ["alert", "notable", "info"].map((key) => ({ key, label: t(key), value: counts[key] || 0 }));
    }
    const max = Math.max(1, ...rows.map((r) => r.value));
    rows.forEach((row) => {
      const line = el("div", "bar-row");
      line.appendChild(el("span", "bar-label", esc(tx(row.label))));
      const track = el("span", "bar-track");
      const fill = el("span", "bar-fill" + (severity ? " sev-" + row.key : "")); fill.style.width = Math.round((row.value / max) * 100) + "%";
      track.appendChild(fill); line.appendChild(track); line.appendChild(el("span", "bar-value", esc(row.value))); d.appendChild(line);
    });
    return d;
  }

  // Normalize a bound value into series groups [{label, points:[{x,y}]}].
  function seriesGroups(data) {
    const arr = asArray(data);
    if (arr.length && arr[0] && Array.isArray(arr[0].points)) return arr;
    const pts = arr.filter((p) => p && typeof p === "object" && "y" in p);
    return pts.length ? [{ label: "", points: pts }] : [];
  }

  // Real line/area chart: plots actual {x,y} points, auto-scaled. No fake data.
  function renderSparkline(data, title, opts) {
    const d = el("div", "chart card");
    if (title) d.appendChild(el("div", "card-title", esc(tx(title))));
    const groups = seriesGroups(data).slice(0, 4)
      .map((g) => ({ label: String(g.label || ""), points: asArray(g.points).map((p, i) => ({ x: p.x != null ? p.x : i, y: Number(p.y) })).filter((p) => Number.isFinite(p.y)) }))
      .filter((g) => g.points.length >= 2);
    if (!groups.length) { d.appendChild(el("div", "chart-placeholder", esc(t("No entries.")))); return d; }
    const ys = []; groups.forEach((g) => g.points.forEach((p) => ys.push(p.y)));
    const min = Math.min.apply(null, ys), max = Math.max.apply(null, ys), span = (max - min) || 1;
    const W = 320, H = 96, pad = 4;
    const svg = svgEl("svg"); svg.setAttribute("viewBox", "0 0 " + W + " " + H); svg.setAttribute("preserveAspectRatio", "none"); svg.setAttribute("class", "sparkline");
    const strokes = ["var(--accent)", "var(--info)", "var(--notable)", "var(--faint)"];
    groups.forEach((g, gi) => {
      const n = Math.max(1, g.points.length - 1);
      const coords = g.points.map((p, i) => (i * (W / n)).toFixed(1) + "," + (H - pad - ((p.y - min) / span) * (H - pad * 2)).toFixed(1)).join(" ");
      if (opts && opts.area) {
        const poly = svgEl("polygon"); poly.setAttribute("points", "0," + (H - pad) + " " + coords + " " + W + "," + (H - pad));
        poly.setAttribute("style", "fill:" + strokes[gi % strokes.length] + ";opacity:.12;stroke:none"); svg.appendChild(poly);
      }
      const line = svgEl("polyline"); line.setAttribute("points", coords);
      line.setAttribute("style", "stroke:" + strokes[gi % strokes.length]); svg.appendChild(line);
    });
    d.appendChild(svg);
    // Always name the line(s) so the chart is self-describing, even for a single
    // series; skip blank labels.
    const labeled = groups.filter((g) => g.label);
    if (labeled.length) { const lg = el("div", "legend"); labeled.forEach((g) => lg.appendChild(el("span", "legend-item", esc(g.label)))); d.appendChild(lg); }
    return d;
  }

  // Real candlestick: OHLC bars from captured market data, auto-scaled.
  function renderCandlestick(data, title) {
    const arr = asArray(data);
    let bars = (arr.length && arr[0] && Array.isArray(arr[0].bars)) ? asArray(arr[0].bars) : arr;
    bars = bars.map((b) => ({ o: Number(b && b.o), h: Number(b && b.h), l: Number(b && b.l), c: Number(b && b.c) }))
      .filter((b) => Number.isFinite(b.o) && Number.isFinite(b.h) && Number.isFinite(b.l) && Number.isFinite(b.c));
    const d = el("div", "chart card");
    if (title) d.appendChild(el("div", "card-title", esc(tx(title))));
    if (bars.length < 2) { d.appendChild(el("div", "chart-placeholder", esc(t("No entries.")))); return d; }
    const lo = Math.min.apply(null, bars.map((b) => b.l)), hi = Math.max.apply(null, bars.map((b) => b.h)), span = (hi - lo) || 1;
    const W = 320, H = 120, pad = 6, step = W / bars.length, bw = Math.max(2, step * 0.6);
    const y = (v) => H - pad - ((v - lo) / span) * (H - pad * 2);
    const svg = svgEl("svg"); svg.setAttribute("viewBox", "0 0 " + W + " " + H); svg.setAttribute("preserveAspectRatio", "none"); svg.setAttribute("class", "sparkline");
    bars.forEach((b, i) => {
      const cx = i * step + step / 2, color = b.c >= b.o ? "var(--info)" : "var(--alert)";
      const wick = svgEl("line"); wick.setAttribute("x1", cx.toFixed(1)); wick.setAttribute("x2", cx.toFixed(1));
      wick.setAttribute("y1", y(b.h).toFixed(1)); wick.setAttribute("y2", y(b.l).toFixed(1));
      wick.setAttribute("style", "stroke:" + color + ";stroke-width:1"); svg.appendChild(wick);
      const top = y(Math.max(b.o, b.c)), bot = y(Math.min(b.o, b.c));
      const rect = svgEl("rect"); rect.setAttribute("x", (cx - bw / 2).toFixed(1)); rect.setAttribute("y", top.toFixed(1));
      rect.setAttribute("width", bw.toFixed(1)); rect.setAttribute("height", Math.max(1, bot - top).toFixed(1));
      rect.setAttribute("style", "fill:" + color); svg.appendChild(rect);
    });
    d.appendChild(svg); return d;
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
    if (!rows.length) return el("div", "empty-inline", esc(t("No entries.")));
    const table = el("table", "data-table");
    if (p.caption) table.appendChild(tableCaption(p.caption));
    const head = document.createElement("thead"); const headRow = document.createElement("tr");
    cols.forEach((c) => headRow.appendChild(el("th", null, esc(tx(c.label || c.key || c))))); head.appendChild(headRow); table.appendChild(head);
    const body = document.createElement("tbody"); rows.forEach((row) => { const tr = document.createElement("tr"); cols.forEach((c) => { const v = row && row[c.key || c] != null ? row[c.key || c] : ""; tr.appendChild(el("td", null, esc(tx(v)))); }); body.appendChild(tr); }); table.appendChild(body); return table;
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
    Grid: (n) => renderChildren(n, el("div", "grid" + gridCols(n.props || {}))),
    Row: (n) => { const v = (n.props || {}).variant; const cls = v === "metrics" ? " metric-strip" : (v === "meta" ? " row-meta" : ""); return renderChildren(n, el("div", "row" + cls)); },
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
      d.appendChild(el("div", "label", esc(tx(p.label))));
      const v = (p.value != null && p.value !== "") ? (p.i18nValue ? tx(p.value) : p.value) : "\u2014";
      d.appendChild(el("div", "value", esc(v)));
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
    AreaChart: (n) => chartNode(renderSparkline((n.props || {}).data, (n.props || {}).title, { area: true }), n.props || {}),
    LineChart: (n) => chartNode(renderSparkline((n.props || {}).data, (n.props || {}).title), n.props || {}),
    Sparkline: (n) => chartNode(renderSparkline((n.props || {}).data, (n.props || {}).title), n.props || {}),
    CandlestickChart: (n) => chartNode(renderCandlestick((n.props || {}).data, (n.props || {}).title || "Candlestick"), n.props || {}),
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
    applySpan(dom, node.props || {});
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
    ws.onopen = () => { setConnectionStatus("live"); };
    ws.onclose = () => { setConnectionStatus("reconnecting…"); setTimeout(connectWS, 3000); };
    ws.onmessage = (ev) => {
      let msg; try { msg = JSON.parse(ev.data); } catch (_) { return; }
      if (msg.type === "monitor.finding") { toast(msg.payload || {}); fetchView(); }
      else if (msg.type === "watch.state") { fetchView(); }
      else if (msg.type === "signal.stream") {
        // Append to local signal stream buffer (max 50)
        if (!window._signalStream) window._signalStream = [];
        var payload = msg.payload || {};
        window._signalStream.push(payload);
        if (window._signalStream.length > 50) window._signalStream.shift();
        updateSignalTimeline(window._signalStream);
        // Increment live event counter
        incrementSignalCounter();
      }
      else if (msg.type === "view.replace" && msg.spec) { render(msg.spec); }
    };
  }

  function updateSignalTimeline(stream) {
    if (current.template !== "signals") return;
    var custom = document.querySelector(".signal-timeline");
    if (custom) { renderSignalTimelineInto(custom, stream); return; }
    var container = document.querySelector(".timeline");
    if (!container) return;
    container.innerHTML = "";
    normalizeSignalItems(stream).slice(0, 12).forEach(function (item) {
      var row = el("div", "timeline-item sev-info");
      row.appendChild(el("div", "timeline-title", esc(item.event_type || item.title || "")));
      if (item.source || item.summary) row.appendChild(el("div", "summary", esc(item.source || item.summary)));
      container.appendChild(row);
    });
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
