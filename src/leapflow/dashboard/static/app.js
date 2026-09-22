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
  // Custom live lenses are recreated from the authoritative snapshot after every
  // fetch. WebSocket increments only update the currently rendered instances.
  let _evolutionLiveControllers = [];

  // ── Signal auto-refresh state ──
  let _signalRefreshTimer = null;
  let _signalEventCount = 0;

  // ── Subagent auto-refresh state ──
  let _subagentRefreshTimer = null;

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

  function startSubagentAutoRefresh() {
    stopSubagentAutoRefresh();
    _subagentRefreshTimer = setInterval(function () {
      if (getCurrentTemplate() === "subagents") fetchView();
    }, 4000);
  }

  function stopSubagentAutoRefresh() {
    if (_subagentRefreshTimer) { clearInterval(_subagentRefreshTimer); _subagentRefreshTimer = null; }
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
    en: {"preview.in_use": "In use now", "preview.viewers": "{count} viewer(s)", "preview.profile": "Preview quality", "preview.economy": "Economy", "preview.balanced": "Balanced", "preview.detail": "Detail", "preview.one_sample": "One level sample captured. Choose session allow for a live waveform.", "preview.one_frame": "One frame captured. Choose session allow for a live preview.", "preview.level_waveform": "Live microphone level waveform",  "Every change is previewed first, then requires approval": "Every change is previewed first, then requires approval", "approval.allow_once": "Allow once", "approval.allow_session": "Allow for this session", "approval.allow_all_session": "Allow all this session", "approval.allow_always": "Always allow", "approval.allow": "Allow", "approval.deny": "Deny", "approval.deny_always": "Always deny", "approval.cancel_workflow": "Cancel the workflow", "Admission decisions": "Admission decisions", "Attached devices": "Attached devices", "Class": "Class", "Connection": "Connection", "Controls": "Controls", "Declaration": "Declaration", "Declared by": "Declared by", "Declared limits": "Declared limits", "Detail": "Detail", "Device unavailable": "Device unavailable", "Events": "Events", "Field": "Field", "History": "History", "Hz": "Hz", "Other devices": "Other devices", "Preview": "Preview", "Previewable": "Previewable", "Privacy": "Privacy", "Quality": "Quality", "Rule": "Rule", "Sampled": "Sampled", "Shape": "Shape", "Trend": "Trend", "Unit": "Unit", "Latest sampled value against the declared limits": "Latest sampled value against the declared limits", "Downsampled windows per channel, newest on the right": "Downsampled windows per channel, newest on the right", "Every change is previewed first, then requires approval where your session is authenticated": "Every change is previewed first, then requires approval where your session is authenticated", "Grouped by declared class · select a device for live channels, preview and controls": "Grouped by declared class · select a device for live channels, preview and controls", "Pulled on demand at the declared ceiling · never sampled into stored history": "Pulled on demand at the declared ceiling · never sampled into stored history", "Rejected declarations are unusable; demoted ones remain readable.": "Rejected declarations are unusable; demoted ones remain readable.", "Select a row to open the device. An unverified declaration has its writable channels demoted to read-only.": "Select a row to open the device. An unverified declaration has its writable channels demoted to read-only.", "Why a device or channel is not what its declaration asked for": "Why a device or channel is not what its declaration asked for", "Where this device's knowledge came from, and who is accountable for it": "Where this device's knowledge came from, and who is accountable for it", "An unverified declaration keeps its readable channels and loses its writable ones.": "An unverified declaration keeps its readable channels and loses its writable ones.", "'not sampled' means no loop reads this channel, so it has no current value anyone measured.": "'not sampled' means no loop reads this channel, so it has no current value anyone measured.", "ceiling": "ceiling", "consent required": "consent required", "Not streaming.": "Not streaming.", "Start preview": "Start preview", "Stop preview": "Stop preview", "Requesting access…": "Requesting access…", "Preview unavailable": "Preview unavailable", "Preview stream ended.": "Preview stream ended.", "Streaming.": "Streaming.", "Control": "Control", "limits": "limits", "reversible": "reversible", "irreversible": "irreversible", "Preview change": "Preview change", "Request approval": "Request approval", "Checking…": "Checking…", "Requesting approval…": "Requesting approval…", "No response.": "No response.", "Approval is given where your session is authenticated (TUI or leap hw).": "Approval is given where your session is authenticated (TUI or leap hw).", "Approval required": "Approval required", "Answer this where your session is authenticated (TUI or leap hw).": "Answer this where your session is authenticated (TUI or leap hw).", "Distribution": "Distribution", "manual_refresh": "manual refresh", "first_observation": "first observation", "artifact_changed": "artifact changed", "batch_turns": "turn threshold", "batch_tokens": "token threshold", "model_salience": "model salience", "text_only": "conversation text", "text_and_artifacts": "conversation + files", "partial_artifacts": "partial files" },
    zh: {"preview.in_use": "正在使用", "preview.viewers": "{count} 个观看端", "preview.profile": "预览质量", "preview.economy": "省资源", "preview.balanced": "平衡", "preview.detail": "高画质", "preview.one_sample": "已获取一次电平。选择“本会话内允许”即可显示实时波形。", "preview.one_frame": "已获取一帧。选择“本会话内允许”即可开始实时预览。", "preview.level_waveform": "实时麦克风电平波形", "Every change is previewed first, then requires approval": "每次变更先预览，然后需要审批", "approval.allow_once": "仅本次允许", "approval.allow_session": "本会话内允许", "approval.allow_all_session": "本会话内全部允许", "approval.allow_always": "始终允许", "approval.allow": "允许", "approval.deny": "拒绝", "approval.deny_always": "始终拒绝", "approval.cancel_workflow": "取消整个流程", "Admission decisions": "准入决定", "Attached devices": "已连接设备", "Class": "类别", "Connection": "连接", "Controls": "控制", "Declaration": "声明", "Declared by": "声明来源", "Declared limits": "声明限值", "Detail": "详情", "Device unavailable": "设备不可用", "Events": "事件", "Field": "字段", "History": "历史", "Hz": "赫兹", "Other devices": "其他设备", "Preview": "预览", "Previewable": "可预览", "Privacy": "隐私", "Quality": "质量", "Rule": "规则", "Sampled": "已采样", "Shape": "形态", "Trend": "趋势", "Unit": "单位", "Latest sampled value against the declared limits": "最新采样值与声明限值对比", "Downsampled windows per channel, newest on the right": "每通道的降采样窗口，最新在右", "Every change is previewed first, then requires approval where your session is authenticated": "每次变更先预览，再在已认证的会话中审批", "Grouped by declared class · select a device for live channels, preview and controls": "按声明类别分组 · 选择设备查看实时通道、预览与控制", "Pulled on demand at the declared ceiling · never sampled into stored history": "按需拉取，不超过声明上限 · 从不写入历史存储", "Rejected declarations are unusable; demoted ones remain readable.": "被拒绝的声明不可用；被降级的仍可读取。", "Select a row to open the device. An unverified declaration has its writable channels demoted to read-only.": "选择一行打开设备。未经确认的声明其可写通道被降级为只读。", "Why a device or channel is not what its declaration asked for": "设备或通道为何与其声明不一致", "Where this device's knowledge came from, and who is accountable for it": "该设备知识的来源，以及谁为其负责", "An unverified declaration keeps its readable channels and loses its writable ones.": "未经确认的声明保留可读通道，失去可写通道。", "'not sampled' means no loop reads this channel, so it has no current value anyone measured.": "“未采样”表示没有循环读取该通道，因此没有任何人测得的当前值。", "ceiling": "上限", "consent required": "需要授权", "Not streaming.": "未在传输。", "Start preview": "开始预览", "Stop preview": "停止预览", "Requesting access…": "正在请求访问…", "Preview unavailable": "预览不可用", "Preview stream ended.": "预览流已结束。", "Streaming.": "正在传输。", "Control": "控制", "limits": "限值", "reversible": "可逆", "irreversible": "不可逆", "Preview change": "预览变更", "Request approval": "申请审批", "Checking…": "正在检查…", "Requesting approval…": "正在申请审批…", "No response.": "无响应。", "Approval is given where your session is authenticated (TUI or leap hw).": "审批需在已认证的会话中完成（TUI 或 leap hw）。", "Approval required": "需要审批", "Answer this where your session is authenticated (TUI or leap hw).": "请在已认证的会话中回应（TUI 或 leap hw）。", "Distribution": "分布", "Abstract": "摘要", "Action failed": "操作失败", "Action items": "行动项", "Active event-driven monitors": "活跃的事件驱动监视器", "Active triggers": "活跃触发器", "Active watches": "活跃观察", "Artifacts": "副产物", "Buffer dropped": "缓冲丢弃", "Calibrated at": "校准时间", "Calibration health": "校准健康度", "Candlestick": "K线", "Channels that have never been calibrated or whose calibration has expired are shown first.": "从未校准或校准已过期的通道排在最前。", "Context map": "上下文图谱", "Coverage": "覆盖率", "Coverage · storyline · severity": "覆盖率 · 叙事 · 严重度", "Custom": "自定义", "Days since": "距今天数", "Debounced": "去抖", "Decisions": "决策", "Decisions and actions": "决策与行动", "Domain": "领域", "Entities": "实体", "Entities and follow-ups": "实体与后续", "Evidence stream": "证据流", "Executive brief": "执行摘要", "Extracted from this session's tool/file output (not model-generated).": "数据来自本次会话的工具/文件产物（非模型生成）。", "Failed to load view": "视图加载失败", "File": "文件", "File artifacts": "文件副产物", "Findings": "发现", "Gauge": "仪表", "Insight count by severity.": "按严重度统计的洞察数。", "Insights": "洞察", "Key observations": "关键观察", "Language": "语言", "Latest observation results": "最新观测结果", "Latest sentiment": "最新情绪", "Live signal stream": "实时信号流", "Loading…": "加载中…", "Market brief": "市场简报", "Mentions": "提及", "Name": "名称", "Narrative pulse": "叙事脉搏", "New papers": "新论文", "Next prompts": "后续追问", "Next recal due": "下次校准期限", "No content yet.": "暂无内容。", "No entries.": "暂无条目。", "Note": "说明", "Observation": "观察", "Observation status": "观察状态", "Observed context": "已观察上下文", "Open questions": "待回答问题", "Operating agenda": "行动议程", "Overview": "概览", "Per-channel calibration state, freshness, and residual correction": "各通道的校准状态、时效性与残差校正", "Price action": "价格行为", "Reason": "原因", "Recent events (last 50)": "最近事件（最新50条）", "Recent findings": "最新发现", "Refresh cadence": "刷新节奏", "Refresh reason": "刷新原因", "Refresh state": "刷新状态", "Research pipeline": "研究管线", "Residual": "残差", "Sentiment structure": "情绪结构", "Series": "序列", "Session": "会话", "Session Analysis": "会话分析", "Session file artifacts.": "会话文件副产物。", "Severity mix": "严重度结构", "Signal flow": "信号流", "Signal mix": "信号结构", "Signals": "信号", "State": "状态", "Status": "状态", "Storyline": "叙事线", "Subscribers": "订阅者", "Suggested next prompts": "建议追问", "Timeline": "时间线", "Tokens": "词元", "Trigger": "触发器", "Trigger and context": "触发与上下文", "Turns": "轮次", "Watch": "观察", "Watch portfolio": "观察组合", "Watches": "观察任务", "alert": "警报", "artifact_changed": "文件副产物变化", "batch_tokens": "上下文阈值", "batch_turns": "轮次阈值", "connecting…": "连接中…", "first_observation": "首次观察", "info": "信息", "live": "实时", "manual_refresh": "手动刷新", "model_salience": "模型显著性", "notable": "重要", "reconnecting…": "重连中…", "unknown": "未知"},
    fr: {"preview.in_use": "En cours d’utilisation", "preview.viewers": "{count} spectateur(s)", "preview.profile": "Qualité de l’aperçu", "preview.economy": "Économie", "preview.balanced": "Équilibré", "preview.detail": "Détail", "preview.one_sample": "Un niveau a été capturé. Autorisez la session pour la forme d’onde en direct.", "preview.one_frame": "Une image a été capturée. Autorisez la session pour l’aperçu en direct.", "preview.level_waveform": "Forme d’onde du niveau de microphone en direct", "Every change is previewed first, then requires approval": "Chaque changement est d'abord prévisualisé, puis approuvé", "approval.allow_once": "Autoriser une fois", "approval.allow_session": "Autoriser pour la session", "approval.allow_all_session": "Tout autoriser pour la session", "approval.allow_always": "Toujours autoriser", "approval.allow": "Autoriser", "approval.deny": "Refuser", "approval.deny_always": "Toujours refuser", "approval.cancel_workflow": "Annuler le flux", "Admission decisions": "Décisions d'admission", "Attached devices": "Appareils connectés", "Class": "Classe", "Connection": "Connexion", "Controls": "Contrôles", "Declaration": "Déclaration", "Declared by": "Déclaré par", "Declared limits": "Limites déclarées", "Detail": "Détail", "Device unavailable": "Appareil indisponible", "Events": "Événements", "Field": "Champ", "History": "Historique", "Hz": "Hz", "Other devices": "Autres appareils", "Preview": "Aperçu", "Previewable": "Prévisualisable", "Privacy": "Confidentialité", "Quality": "Qualité", "Rule": "Règle", "Sampled": "Échantillonné", "Shape": "Forme", "Trend": "Tendance", "Unit": "Unité", "Latest sampled value against the declared limits": "Dernière valeur échantillonnée face aux limites déclarées", "Downsampled windows per channel, newest on the right": "Fenêtres sous-échantillonnées par canal, la plus récente à droite", "Every change is previewed first, then requires approval where your session is authenticated": "Chaque changement est d'abord prévisualisé, puis approuvé là où votre session est authentifiée", "Grouped by declared class · select a device for live channels, preview and controls": "Groupés par classe déclarée · sélectionnez un appareil pour les canaux en direct, l'aperçu et les contrôles", "Pulled on demand at the declared ceiling · never sampled into stored history": "Récupéré à la demande dans la limite déclarée · jamais échantillonné dans l'historique", "Rejected declarations are unusable; demoted ones remain readable.": "Les déclarations rejetées sont inutilisables ; celles rétrogradées restent lisibles.", "Select a row to open the device. An unverified declaration has its writable channels demoted to read-only.": "Sélectionnez une ligne pour ouvrir l'appareil. Une déclaration non vérifiée voit ses canaux inscriptibles rétrogradés en lecture seule.", "Why a device or channel is not what its declaration asked for": "Pourquoi un appareil ou un canal n'est pas ce que sa déclaration demandait", "Where this device's knowledge came from, and who is accountable for it": "D'où vient la connaissance de cet appareil et qui en est responsable", "An unverified declaration keeps its readable channels and loses its writable ones.": "Une déclaration non vérifiée conserve ses canaux lisibles et perd ceux inscriptibles.", "'not sampled' means no loop reads this channel, so it has no current value anyone measured.": "« non échantillonné » signifie qu'aucune boucle ne lit ce canal : il n'a donc aucune valeur actuelle mesurée.", "ceiling": "plafond", "consent required": "consentement requis", "Not streaming.": "Pas de flux.", "Start preview": "Démarrer l'aperçu", "Stop preview": "Arrêter l'aperçu", "Requesting access…": "Demande d'accès…", "Preview unavailable": "Aperçu indisponible", "Preview stream ended.": "Le flux d'aperçu s'est arrêté.", "Streaming.": "Diffusion en cours.", "Control": "Contrôle", "limits": "limites", "reversible": "réversible", "irreversible": "irréversible", "Preview change": "Prévisualiser le changement", "Request approval": "Demander l'approbation", "Checking…": "Vérification…", "Requesting approval…": "Demande d'approbation…", "No response.": "Aucune réponse.", "Approval is given where your session is authenticated (TUI or leap hw).": "L'approbation se donne là où votre session est authentifiée (TUI ou leap hw).", "Approval required": "Approbation requise", "Answer this where your session is authenticated (TUI or leap hw).": "Répondez là où votre session est authentifiée (TUI ou leap hw).", "Distribution": "Distribution", "Abstract": "Résumé", "Action failed": "Action échouée", "Action items": "Actions", "Active event-driven monitors": "Moniteurs événementiels actifs", "Active triggers": "Déclencheurs actifs", "Artifacts": "Artefacts", "Buffer dropped": "Tampon perdu", "Calibrated at": "Calibré le", "Calibration health": "État de calibration", "Channels that have never been calibrated or whose calibration has expired are shown first.": "Les canaux jamais calibrés ou dont la calibration a expiré apparaissent en premier.", "Context map": "Carte de contexte", "Coverage": "Couverture", "Coverage · storyline · severity": "Couverture · récit · sévérité", "Days since": "Jours écoulés", "Debounced": "Antirebond", "Decisions": "Décisions", "Decisions and actions": "Décisions et actions", "Domain": "Domaine", "Entities": "Entités", "Entities and follow-ups": "Entités et suivis", "Executive brief": "Synthèse exécutive", "Failed to load view": "Échec du chargement", "File": "Fichier", "File artifacts": "Fichiers", "Findings": "Constats", "Insight count by severity.": "Nombre d’analyses par sévérité.", "Insights": "Analyses", "Key observations": "Observations clés", "Language": "Langue", "Latest observation results": "Derniers résultats d'observation", "Live signal stream": "Flux de signaux en direct", "Loading…": "chargement…", "Name": "Nom", "Next prompts": "Invites suivantes", "Next recal due": "Prochaine recalibration", "No content yet.": "Aucun contenu.", "No entries.": "Aucune entrée.", "Note": "Note", "Observation": "Observation", "Observation status": "Statut d’observation", "Observed context": "Contexte observé", "Open questions": "Questions ouvertes", "Operating agenda": "Programme d’action", "Overview": "Vue d’ensemble", "Per-channel calibration state, freshness, and residual correction": "État de calibration, fraîcheur et correction résiduelle par canal", "Reason": "Raison", "Recent events (last 50)": "Événements récents (50 derniers)", "Recent findings": "Constats récents", "Refresh reason": "Raison", "Refresh state": "État", "Residual": "Résidu", "Session": "Session", "Session Analysis": "Analyse de session", "Session file artifacts.": "Artefacts de fichiers de session.", "Severity mix": "Mix de sévérité", "Signal flow": "Flux de signaux", "Signals": "Signaux", "State": "État", "Status": "Statut", "Storyline": "Narratif", "Subscribers": "Abonnés", "Suggested next prompts": "Prochaines invites", "Timeline": "Chronologie", "Tokens": "Jetons", "Trigger": "Déclencheur", "Trigger and context": "Déclencheur et contexte", "Turns": "Tours", "Watch": "Veille", "Watches": "Veilles", "alert": "alerte", "artifact_changed": "artefact modifié", "batch_tokens": "seuil de jetons", "batch_turns": "seuil de tours", "connecting…": "connexion…", "first_observation": "première observation", "info": "info", "live": "direct", "manual_refresh": "actualisation manuelle", "model_salience": "saillance modèle", "notable": "notable", "reconnecting…": "reconnexion…"},
    es: {"preview.in_use": "En uso ahora", "preview.viewers": "{count} espectador(es)", "preview.profile": "Calidad de vista previa", "preview.economy": "Ahorro", "preview.balanced": "Equilibrado", "preview.detail": "Detalle", "preview.one_sample": "Se capturó una muestra de nivel. Permita la sesión para la forma de onda en vivo.", "preview.one_frame": "Se capturó un fotograma. Permita la sesión para la vista previa en vivo.", "preview.level_waveform": "Forma de onda del nivel de micrófono en vivo", "Every change is previewed first, then requires approval": "Cada cambio se previsualiza primero y luego requiere aprobación", "approval.allow_once": "Permitir una vez", "approval.allow_session": "Permitir en esta sesión", "approval.allow_all_session": "Permitir todo en la sesión", "approval.allow_always": "Permitir siempre", "approval.allow": "Permitir", "approval.deny": "Denegar", "approval.deny_always": "Denegar siempre", "approval.cancel_workflow": "Cancelar el flujo", "Admission decisions": "Decisiones de admisión", "Attached devices": "Dispositivos conectados", "Class": "Clase", "Connection": "Conexión", "Controls": "Controles", "Declaration": "Declaración", "Declared by": "Declarado por", "Declared limits": "Límites declarados", "Detail": "Detalle", "Device unavailable": "Dispositivo no disponible", "Events": "Eventos", "Field": "Campo", "History": "Historial", "Hz": "Hz", "Other devices": "Otros dispositivos", "Preview": "Vista previa", "Previewable": "Previsualizable", "Privacy": "Privacidad", "Quality": "Calidad", "Rule": "Regla", "Sampled": "Muestreado", "Shape": "Forma", "Trend": "Tendencia", "Unit": "Unidad", "Latest sampled value against the declared limits": "Último valor muestreado frente a los límites declarados", "Downsampled windows per channel, newest on the right": "Ventanas submuestreadas por canal, la más reciente a la derecha", "Every change is previewed first, then requires approval where your session is authenticated": "Cada cambio se previsualiza primero y luego requiere aprobación donde su sesión está autenticada", "Grouped by declared class · select a device for live channels, preview and controls": "Agrupados por clase declarada · seleccione un dispositivo para canales en vivo, vista previa y controles", "Pulled on demand at the declared ceiling · never sampled into stored history": "Obtenido a demanda dentro del techo declarado · nunca se muestrea en el historial", "Rejected declarations are unusable; demoted ones remain readable.": "Las declaraciones rechazadas son inutilizables; las degradadas siguen siendo legibles.", "Select a row to open the device. An unverified declaration has its writable channels demoted to read-only.": "Seleccione una fila para abrir el dispositivo. Una declaración no verificada tiene sus canales escribibles degradados a solo lectura.", "Why a device or channel is not what its declaration asked for": "Por qué un dispositivo o canal no es lo que pedía su declaración", "Where this device's knowledge came from, and who is accountable for it": "De dónde proviene el conocimiento de este dispositivo y quién responde por él", "An unverified declaration keeps its readable channels and loses its writable ones.": "Una declaración no verificada conserva sus canales legibles y pierde los escribibles.", "'not sampled' means no loop reads this channel, so it has no current value anyone measured.": "«no muestreado» significa que ningún bucle lee este canal, por lo que no tiene un valor actual medido.", "ceiling": "techo", "consent required": "se requiere consentimiento", "Not streaming.": "Sin transmisión.", "Start preview": "Iniciar vista previa", "Stop preview": "Detener vista previa", "Requesting access…": "Solicitando acceso…", "Preview unavailable": "Vista previa no disponible", "Preview stream ended.": "La transmisión de vista previa terminó.", "Streaming.": "Transmitiendo.", "Control": "Control", "limits": "límites", "reversible": "reversible", "irreversible": "irreversible", "Preview change": "Previsualizar cambio", "Request approval": "Solicitar aprobación", "Checking…": "Comprobando…", "Requesting approval…": "Solicitando aprobación…", "No response.": "Sin respuesta.", "Approval is given where your session is authenticated (TUI or leap hw).": "La aprobación se otorga donde su sesión está autenticada (TUI o leap hw).", "Approval required": "Se requiere aprobación", "Answer this where your session is authenticated (TUI or leap hw).": "Responda donde su sesión está autenticada (TUI o leap hw).", "Distribution": "Distribución", "Abstract": "Resumen", "Action failed": "Acción fallida", "Action items": "Acciones", "Active event-driven monitors": "Monitores por eventos activos", "Active triggers": "Disparadores activos", "Artifacts": "Artefactos", "Buffer dropped": "Buffer perdido", "Calibrated at": "Calibrado el", "Calibration health": "Estado de calibración", "Channels that have never been calibrated or whose calibration has expired are shown first.": "Los canales nunca calibrados o con calibración vencida se muestran primero.", "Context map": "Mapa de contexto", "Coverage": "Cobertura", "Coverage · storyline · severity": "Cobertura · relato · severidad", "Days since": "Días desde", "Debounced": "Antirrebote", "Decisions": "Decisiones", "Decisions and actions": "Decisiones y acciones", "Domain": "Dominio", "Entities": "Entidades", "Entities and follow-ups": "Entidades y seguimientos", "Executive brief": "Resumen ejecutivo", "Failed to load view": "Error al cargar", "File": "Archivo", "File artifacts": "Archivos", "Findings": "Hallazgos", "Insight count by severity.": "Recuento de hallazgos por severidad.", "Insights": "Ideas", "Key observations": "Observaciones clave", "Language": "Idioma", "Latest observation results": "Últimos resultados de observación", "Live signal stream": "Flujo de señales en vivo", "Loading…": "cargando…", "Name": "Nombre", "Next prompts": "Siguientes prompts", "Next recal due": "Próxima recalibración", "No content yet.": "Sin contenido.", "No entries.": "Sin entradas.", "Note": "Nota", "Observation": "Observación", "Observation status": "Estado de observación", "Observed context": "Contexto observado", "Open questions": "Preguntas abiertas", "Operating agenda": "Agenda operativa", "Overview": "Resumen", "Per-channel calibration state, freshness, and residual correction": "Estado de calibración, vigencia y corrección residual por canal", "Reason": "Motivo", "Recent events (last 50)": "Eventos recientes (últimos 50)", "Recent findings": "Hallazgos recientes", "Refresh reason": "Motivo", "Refresh state": "Estado", "Residual": "Residuo", "Session": "Sesión", "Session Analysis": "Análisis de sesión", "Session file artifacts.": "Artefactos de archivos de sesión.", "Severity mix": "Mezcla de severidad", "Signal flow": "Flujo de señales", "Signals": "Señales", "State": "Estado", "Status": "Estado", "Storyline": "Narrativa", "Subscribers": "Suscriptores", "Suggested next prompts": "Siguientes preguntas", "Timeline": "Cronología", "Tokens": "Tokens", "Trigger": "Disparador", "Trigger and context": "Disparador y contexto", "Turns": "Turnos", "Watch": "Vigilancia", "Watches": "Vigilancias", "alert": "alerta", "artifact_changed": "artefacto cambiado", "batch_tokens": "umbral de tokens", "batch_turns": "umbral de turnos", "connecting…": "conectando…", "first_observation": "primera observación", "info": "info", "live": "en vivo", "manual_refresh": "actualización manual", "model_salience": "relevancia del modelo", "notable": "relevante", "reconnecting…": "reconectando…"},
    ar: {"preview.in_use": "قيد الاستخدام الآن", "preview.viewers": "{count} مشاهد", "preview.profile": "جودة المعاينة", "preview.economy": "اقتصادي", "preview.balanced": "متوازن", "preview.detail": "تفاصيل", "preview.one_sample": "تم التقاط عينة مستوى واحدة. اسمح للجلسة لعرض موجة مباشرة.", "preview.one_frame": "تم التقاط إطار واحد. اسمح للجلسة لمعاينة مباشرة.", "preview.level_waveform": "موجة مستوى الميكروفون المباشرة", "Every change is previewed first, then requires approval": "تُعاين كل تغيير أولاً ثم يتطلب موافقة", "approval.allow_once": "السماح مرة واحدة", "approval.allow_session": "السماح خلال الجلسة", "approval.allow_all_session": "السماح بالكل خلال الجلسة", "approval.allow_always": "السماح دائماً", "approval.allow": "السماح", "approval.deny": "رفض", "approval.deny_always": "الرفض دائماً", "approval.cancel_workflow": "إلغاء سير العمل", "Admission decisions": "قرارات القبول", "Attached devices": "الأجهزة المتصلة", "Class": "الفئة", "Connection": "الاتصال", "Controls": "أدوات التحكم", "Declaration": "الإعلان", "Declared by": "أُعلن بواسطة", "Declared limits": "الحدود المعلنة", "Detail": "التفاصيل", "Device unavailable": "الجهاز غير متاح", "Events": "الأحداث", "Field": "الحقل", "History": "السجل", "Hz": "هرتز", "Other devices": "أجهزة أخرى", "Preview": "معاينة", "Previewable": "قابل للمعاينة", "Privacy": "الخصوصية", "Quality": "الجودة", "Rule": "القاعدة", "Sampled": "تم أخذ العينات", "Shape": "الشكل", "Trend": "الاتجاه", "Unit": "الوحدة", "Latest sampled value against the declared limits": "أحدث قيمة مُقاسة مقابل الحدود المعلنة", "Downsampled windows per channel, newest on the right": "نوافذ مُخفّضة العينات لكل قناة، الأحدث على اليمين", "Every change is previewed first, then requires approval where your session is authenticated": "تُعاين كل تغيير أولاً ثم يتطلب موافقة حيث تكون جلستك موثّقة", "Grouped by declared class · select a device for live channels, preview and controls": "مُجمَّعة حسب الفئة المعلنة · اختر جهازاً لعرض القنوات الحيّة والمعاينة والتحكم", "Pulled on demand at the declared ceiling · never sampled into stored history": "يُسحب عند الطلب وفق الحد الأعلى المعلن · لا يُخزَّن في السجل", "Rejected declarations are unusable; demoted ones remain readable.": "الإعلانات المرفوضة غير قابلة للاستخدام؛ والمُخفَّضة تبقى قابلة للقراءة.", "Select a row to open the device. An unverified declaration has its writable channels demoted to read-only.": "اختر صفاً لفتح الجهاز. الإعلان غير المُتحقَّق منه تُخفَّض قنواته القابلة للكتابة إلى القراءة فقط.", "Why a device or channel is not what its declaration asked for": "لماذا لا يطابق الجهاز أو القناة ما طلبه إعلانه", "Where this device's knowledge came from, and who is accountable for it": "من أين جاءت معرفة هذا الجهاز ومن المسؤول عنها", "An unverified declaration keeps its readable channels and loses its writable ones.": "الإعلان غير المُتحقَّق منه يحتفظ بقنواته القابلة للقراءة ويفقد القابلة للكتابة.", "'not sampled' means no loop reads this channel, so it has no current value anyone measured.": "«لم تُقس» تعني أن لا حلقة تقرأ هذه القناة، فليست لها قيمة حالية قاسها أحد.", "ceiling": "الحد الأعلى", "consent required": "يتطلب موافقة", "Not streaming.": "لا يوجد بث.", "Start preview": "بدء المعاينة", "Stop preview": "إيقاف المعاينة", "Requesting access…": "جارٍ طلب الوصول…", "Preview unavailable": "المعاينة غير متاحة", "Preview stream ended.": "انتهى بث المعاينة.", "Streaming.": "جارٍ البث.", "Control": "تحكم", "limits": "الحدود", "reversible": "قابل للعكس", "irreversible": "غير قابل للعكس", "Preview change": "معاينة التغيير", "Request approval": "طلب الموافقة", "Checking…": "جارٍ التحقق…", "Requesting approval…": "جارٍ طلب الموافقة…", "No response.": "لا استجابة.", "Approval is given where your session is authenticated (TUI or leap hw).": "تُمنح الموافقة حيث تكون جلستك موثّقة (TUI أو leap hw).", "Approval required": "مطلوب موافقة", "Answer this where your session is authenticated (TUI or leap hw).": "أجب عن هذا حيث تكون جلستك موثّقة (TUI أو leap hw).", "Distribution": "التوزيع", "Abstract": "ملخص", "Action failed": "فشل الإجراء", "Action items": "إجراءات", "Active event-driven monitors": "مراقبات حدثية نشطة", "Active triggers": "المُشغِّلات النشطة", "Artifacts": "المخرجات", "Buffer dropped": "ذاكرة مؤقتة مُسقَطة", "Calibrated at": "تاريخ المعايرة", "Calibration health": "سلامة المعايرة", "Channels that have never been calibrated or whose calibration has expired are shown first.": "تظهر أولاً القنوات التي لم تُعاير قط أو التي انتهت صلاحية معايرتها.", "Context map": "خريطة السياق", "Coverage": "التغطية", "Coverage · storyline · severity": "التغطية · السرد · الخطورة", "Days since": "الأيام المنقضية", "Debounced": "مُزال الارتداد", "Decisions": "قرارات", "Decisions and actions": "القرارات والإجراءات", "Domain": "المجال", "Entities": "كيانات", "Entities and follow-ups": "الكيانات والمتابعات", "Executive brief": "ملخص تنفيذي", "Failed to load view": "فشل تحميل العرض", "File": "ملف", "File artifacts": "ملفات", "Findings": "النتائج", "Insight count by severity.": "عدد الرؤى حسب الخطورة.", "Insights": "الرؤى", "Key observations": "ملاحظات رئيسية", "Language": "اللغة", "Latest observation results": "أحدث نتائج الرصد", "Live signal stream": "تدفق الإشارات المباشر", "Loading…": "جارٍ التحميل…", "Name": "الاسم", "Next prompts": "المطالبات التالية", "Next recal due": "موعد إعادة المعايرة", "No content yet.": "لا يوجد محتوى بعد.", "No entries.": "لا توجد إدخالات.", "Note": "ملاحظة", "Observation": "الرصد", "Observation status": "حالة المراقبة", "Observed context": "السياق المرصود", "Open questions": "أسئلة مفتوحة", "Operating agenda": "خطة العمل", "Overview": "نظرة عامة", "Per-channel calibration state, freshness, and residual correction": "حالة المعايرة وحداثتها وتصحيح المتبقي لكل قناة", "Reason": "السبب", "Recent events (last 50)": "الأحداث الأخيرة (آخر 50)", "Recent findings": "أحدث النتائج", "Refresh reason": "سبب التحديث", "Refresh state": "حالة التحديث", "Residual": "المتبقي", "Session": "الجلسة", "Session Analysis": "تحليل الجلسة", "Session file artifacts.": "مخرجات ملفات الجلسة.", "Severity mix": "توزيع الشدة", "Signal flow": "تدفق الإشارات", "Signals": "الإشارات", "State": "الحالة", "Status": "الحالة", "Storyline": "السرد", "Subscribers": "المشتركون", "Suggested next prompts": "أسئلة مقترحة", "Timeline": "الخط الزمني", "Tokens": "الرموز", "Trigger": "المُشغِّل", "Trigger and context": "المُشغِّل والسياق", "Turns": "الأدوار", "Watch": "مراقبة", "Watches": "المراقبات", "alert": "تنبيه", "artifact_changed": "تغير ملف", "batch_tokens": "حد الرموز", "batch_turns": "حد الجولات", "connecting…": "جارٍ الاتصال…", "first_observation": "أول مراقبة", "info": "معلومة", "live": "مباشر", "manual_refresh": "تحديث يدوي", "model_salience": "أهمية النموذج", "notable": "مهم", "reconnecting…": "إعادة الاتصال…"},
    ru: {"preview.in_use": "Используется сейчас", "preview.viewers": "{count} зритель(ей)", "preview.profile": "Качество предпросмотра", "preview.economy": "Экономия", "preview.balanced": "Баланс", "preview.detail": "Детально", "preview.one_sample": "Получено одно измерение уровня. Разрешите на сеанс для живой волны.", "preview.one_frame": "Получен один кадр. Разрешите на сеанс для живого предпросмотра.", "preview.level_waveform": "Живая волна уровня микрофона", "Every change is previewed first, then requires approval": "Каждое изменение сначала предпросматривается, затем требует подтверждения", "approval.allow_once": "Разрешить один раз", "approval.allow_session": "Разрешить на сеанс", "approval.allow_all_session": "Разрешить всё на сеанс", "approval.allow_always": "Разрешать всегда", "approval.allow": "Разрешить", "approval.deny": "Отклонить", "approval.deny_always": "Отклонять всегда", "approval.cancel_workflow": "Отменить весь процесс", "Admission decisions": "Решения о допуске", "Attached devices": "Подключённые устройства", "Class": "Класс", "Connection": "Соединение", "Controls": "Управление", "Declaration": "Декларация", "Declared by": "Объявлено через", "Declared limits": "Заявленные пределы", "Detail": "Подробности", "Device unavailable": "Устройство недоступно", "Events": "События", "Field": "Поле", "History": "История", "Hz": "Гц", "Other devices": "Другие устройства", "Preview": "Предпросмотр", "Previewable": "Доступно для просмотра", "Privacy": "Приватность", "Quality": "Качество", "Rule": "Правило", "Sampled": "Опрашивается", "Shape": "Форма", "Trend": "Тренд", "Unit": "Единица", "Latest sampled value against the declared limits": "Последнее измеренное значение относительно заявленных пределов", "Downsampled windows per channel, newest on the right": "Прореженные окна по каналам, новейшее справа", "Every change is previewed first, then requires approval where your session is authenticated": "Каждое изменение сначала предпросматривается, затем требует подтверждения там, где сеанс аутентифицирован", "Grouped by declared class · select a device for live channels, preview and controls": "Сгруппировано по заявленному классу · выберите устройство для живых каналов, предпросмотра и управления", "Pulled on demand at the declared ceiling · never sampled into stored history": "Запрашивается по требованию в пределах заявленного лимита · никогда не пишется в историю", "Rejected declarations are unusable; demoted ones remain readable.": "Отклонённые декларации непригодны; понижённые остаются доступными для чтения.", "Select a row to open the device. An unverified declaration has its writable channels demoted to read-only.": "Выберите строку, чтобы открыть устройство. У непроверенной декларации записываемые каналы понижаются до чтения.", "Why a device or channel is not what its declaration asked for": "Почему устройство или канал не соответствует своей декларации", "Where this device's knowledge came from, and who is accountable for it": "Откуда взяты сведения об устройстве и кто за них отвечает", "An unverified declaration keeps its readable channels and loses its writable ones.": "Непроверенная декларация сохраняет читаемые каналы и теряет записываемые.", "'not sampled' means no loop reads this channel, so it has no current value anyone measured.": "«не опрашивается» означает, что канал не читает ни один цикл, поэтому измеренного текущего значения нет.", "ceiling": "лимит", "consent required": "требуется согласие", "Not streaming.": "Поток не идёт.", "Start preview": "Начать предпросмотр", "Stop preview": "Остановить предпросмотр", "Requesting access…": "Запрос доступа…", "Preview unavailable": "Предпросмотр недоступен", "Preview stream ended.": "Поток предпросмотра завершён.", "Streaming.": "Идёт поток.", "Control": "Управление", "limits": "пределы", "reversible": "обратимо", "irreversible": "необратимо", "Preview change": "Предпросмотр изменения", "Request approval": "Запросить подтверждение", "Checking…": "Проверка…", "Requesting approval…": "Запрос подтверждения…", "No response.": "Нет ответа.", "Approval is given where your session is authenticated (TUI or leap hw).": "Подтверждение даётся там, где сеанс аутентифицирован (TUI или leap hw).", "Approval required": "Требуется подтверждение", "Answer this where your session is authenticated (TUI or leap hw).": "Ответьте там, где сеанс аутентифицирован (TUI или leap hw).", "Distribution": "Распределение", "Abstract": "Аннотация", "Action failed": "Действие не выполнено", "Action items": "Действия", "Active event-driven monitors": "Активные событийные мониторы", "Active triggers": "Активные триггеры", "Artifacts": "Артефакты", "Buffer dropped": "Потери буфера", "Calibrated at": "Калиброван", "Calibration health": "Состояние калибровки", "Channels that have never been calibrated or whose calibration has expired are shown first.": "Каналы, которые никогда не калибровались или чья калибровка истекла, показаны первыми.", "Context map": "Карта контекста", "Coverage": "Покрытие", "Coverage · storyline · severity": "Покрытие · сюжет · важность", "Days since": "Дней с тех пор", "Debounced": "Дебаунс", "Decisions": "Решения", "Decisions and actions": "Решения и действия", "Domain": "Домен", "Entities": "Сущности", "Entities and follow-ups": "Сущности и продолжения", "Executive brief": "Краткий обзор", "Failed to load view": "Не удалось загрузить", "File": "Файл", "File artifacts": "Файлы", "Findings": "Находки", "Insight count by severity.": "Число инсайтов по важности.", "Insights": "Инсайты", "Key observations": "Ключевые наблюдения", "Language": "Язык", "Latest observation results": "Последние результаты наблюдений", "Live signal stream": "Поток сигналов (live)", "Loading…": "загрузка…", "Name": "Имя", "Next prompts": "Следующие запросы", "Next recal due": "Следующая рекалибровка", "No content yet.": "Пока нет данных.", "No entries.": "Нет записей.", "Note": "Заметка", "Observation": "Наблюдение", "Observation status": "Статус наблюдения", "Observed context": "Наблюдаемый контекст", "Open questions": "Открытые вопросы", "Operating agenda": "Рабочая повестка", "Overview": "Обзор", "Per-channel calibration state, freshness, and residual correction": "Состояние калибровки, актуальность и остаточная поправка по каналам", "Reason": "Причина", "Recent events (last 50)": "Последние события (50)", "Recent findings": "Последние находки", "Refresh reason": "Причина", "Refresh state": "Состояние", "Residual": "Остаток", "Session": "Сессия", "Session Analysis": "Анализ сессии", "Session file artifacts.": "Файловые артефакты сессии.", "Severity mix": "Структура важности", "Signal flow": "Поток сигналов", "Signals": "Сигналы", "State": "Состояние", "Status": "Статус", "Storyline": "Сюжет", "Subscribers": "Подписчики", "Suggested next prompts": "Следующие запросы", "Timeline": "Хронология", "Tokens": "Токены", "Trigger": "Триггер", "Trigger and context": "Триггер и контекст", "Turns": "Ходы", "Watch": "Наблюдение", "Watches": "Наблюдения", "alert": "тревога", "artifact_changed": "файл изменён", "batch_tokens": "порог токенов", "batch_turns": "порог ходов", "connecting…": "подключение…", "first_observation": "первое наблюдение", "info": "инфо", "live": "онлайн", "manual_refresh": "ручное обновление", "model_salience": "значимость модели", "notable": "важно", "reconnecting…": "переподключение…"}
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
      "outside": "outside", "unknown": "unknown",
      "Sub-Agent Monitor": "Sub-Agent Monitor", "No subagent activity": "No subagent activity", "This board shows delegated task execution. No subagent has been dispatched yet.": "This board shows delegated task execution. No subagent has been dispatched yet.", "What are subagents?": "What are subagents?", "> Subagents are spawned when the main agent delegates a task via `delegate_task`. Each runs in an isolated context with its own tool set and message history. Only the summary flows back to the parent. Activity will appear here once a delegation occurs.": "> Subagents are spawned when the main agent delegates a task via `delegate_task`. Each runs in an isolated context with its own tool set and message history. Only the summary flows back to the parent. Activity will appear here once a delegation occurs.",
      "Max depth": "Max depth", "Max concurrent": "Max concurrent", "Summary budget": "Summary budget", "Delegation overview": "Delegation overview", "Aggregate counters for all subagent executions in this daemon lifetime.": "Aggregate counters for all subagent executions in this daemon lifetime.", "Active": "Active", "Completed": "Completed", "Failed": "Failed", "Avg duration": "Avg duration", "Success rate": "Success rate",
      "Active subagents": "Active subagents", "Currently running delegated tasks.": "Currently running delegated tasks.", "ID": "ID", "Goal": "Goal", "Depth": "Depth", "Elapsed (s)": "Elapsed (s)", "Parent": "Parent", "Recent completions": "Recent completions", "Last 50 subagent executions, newest first.": "Last 50 subagent executions, newest first.",
      "Execution detail": "Execution detail", "Tabular view of recent subagent runs with outcome and duration.": "Tabular view of recent subagent runs with outcome and duration.", "Duration (s)": "Duration (s)", "Delegation Tree": "Delegation Tree", "Parent → child relationships": "Parent → child relationships", "How tasks were delegated across depth levels.": "How tasks were delegated across depth levels.", "Child": "Child",
      "Configuration": "Configuration", "Current subagent isolation settings.": "Current subagent isolation settings.", "Statistics": "Statistics", "Delegation by depth": "Delegation by depth", "How many subagents ran at each recursion depth.": "How many subagents ran at each recursion depth.", "Executions by depth": "Executions by depth", "Outcomes": "Outcomes", "Distribution of subagent execution outcomes.": "Distribution of subagent execution outcomes.", "Executions by outcome": "Executions by outcome", "Cumulative metrics": "Cumulative metrics", "Total delegated": "Total delegated", "Total tool calls": "Total tool calls", "Total duration": "Total duration",
      "Active delegations": "Active delegations", "Aggregate counters and success metrics for all subagent executions in this daemon lifetime.": "Aggregate counters and success metrics for all subagent executions in this daemon lifetime.", "Attention required": "Attention required", "Avg delegation depth": "Avg delegation depth", "Avg depth": "Avg depth", "Avg tools/task": "Avg tools/task", "Bucketed execution times. A right-skewed distribution is normal; heavy >60s tail warrants investigation.": "Bucketed execution times. A right-skewed distribution is normal; heavy >60s tail warrants investigation.", "Current isolation settings governing all subagent delegations.": "Current isolation settings governing all subagent delegations.", "Currently running subagent tasks. Elapsed time updates on each refresh cycle.": "Currently running subagent tasks. Elapsed time updates on each refresh cycle.", "Deepest goal": "Deepest goal", "Delegation Graph": "Delegation Graph", "Delegation frequency": "Delegation frequency", "Delegation topology": "Delegation topology", "Delegations aggregated by time window. Rising trend indicates increasing task complexity or workload.": "Delegations aggregated by time window. Rising trend indicates increasing task complexity or workload.", "Delegations by hour-of-day (00:00–23:00). Peaks indicate active work sessions.": "Delegations by hour-of-day (00:00–23:00). Peaks indicate active work sessions.", "Depth analysis": "Depth analysis", "Depth saturation": "Depth saturation", "Description": "Description", "Distribution of delegation recursion depth. Most delegations should cluster at depth 0-1; deeper levels indicate multi-step decomposition.": "Distribution of delegation recursion depth. Most delegations should cluster at depth 0-1; deeper levels indicate multi-step decomposition.", "Duration analysis": "Duration analysis", "Duration distribution": "Duration distribution", "Effective subagent configuration": "Effective subagent configuration", "Efficiency metrics": "Efficiency metrics", "Error": "Error", "Execution History": "Execution History", "Executions by depth level": "Executions by depth level", "Failure rate": "Failure rate", "Health anomalies detected across recent delegations. Shown regardless of the open tab.": "Health anomalies detected across recent delegations. Shown regardless of the open tab.", "High failure rate": "High failure rate", "Hourly delegation pattern": "Hourly delegation pattern", "How deeply tasks are delegated. Depth saturation indicates potential recursion limits.": "How deeply tasks are delegated. Depth saturation indicates potential recursion limits.", "How many tool calls each delegation required. High counts may indicate task complexity or inefficient tool use.": "How many tool calls each delegation required. High counts may indicate task complexity or inefficient tool use.", "How tasks were delegated across depth levels, with execution outcome and duration.": "How tasks were delegated across depth levels, with execution outcome and duration.", "Last 50 subagent executions, newest first. Each entry shows outcome, duration, and tool usage.": "Last 50 subagent executions, newest first. Each entry shows outcome, duration, and tool usage.", "Latency percentiles computed from all recent execution durations.": "Latency percentiles computed from all recent execution durations.", "Max depth reached": "Max depth reached", "Max observed depth": "Max observed depth", "Median duration": "Median duration", "Outcome": "Outcome", "Outcome distribution": "Outcome distribution", "P50": "P50", "P75": "P75", "P90": "P90", "P95": "P95", "Percentile summary": "Percentile summary", "Performance Analytics": "Performance Analytics", "Samples": "Samples", "Setting": "Setting", "Statistical breakdown of execution times across all completed delegations.": "Statistical breakdown of execution times across all completed delegations.", "Status": "Status", "Success vs failure ratio. A healthy system shows predominantly 'completed' outcomes.": "Success vs failure ratio. A healthy system shows predominantly 'completed' outcomes.", "Tabular view with full execution metadata. Error column populated for failed delegations.": "Tabular view with full execution metadata. Error column populated for failed delegations.", "Temporal distribution": "Temporal distribution", "This board reports delegated task execution. No subagent has been dispatched yet.": "This board reports delegated task execution. No subagent has been dispatched yet.", "Timeout detected": "Timeout detected", "Tool calls per delegation": "Tool calls per delegation", "Tool usage patterns and outcome ratios across delegations.": "Tool usage patterns and outcome ratios across delegations.", "Tools": "Tools", "Unique goals": "Unique goals", "Visual map of parent→child delegation pairs. Badge color reflects execution outcome.": "Visual map of parent→child delegation pairs. Badge color reflects execution outcome.", "When delegations occur within the day. Helps identify burst patterns and quiet periods.": "When delegations occur within the day. Helps identify burst patterns and quiet periods.", "> **Configuration tuning guide:** `max_depth` controls recursion — increase for complex multi-step tasks, decrease to prevent runaway delegation chains. `max_concurrent` bounds parallelism — higher values improve throughput but increase resource contention. `summary_max_chars` limits the summary flowing back to the parent; too low truncates critical context, too high wastes the parent's context budget.": "> **Configuration tuning guide:** `max_depth` controls recursion — increase for complex multi-step tasks, decrease to prevent runaway delegation chains. `max_concurrent` bounds parallelism — higher values improve throughput but increase resource contention. `summary_max_chars` limits the summary flowing back to the parent; too low truncates critical context, too high wastes the parent's context budget.", "> **Reading guide:** P50 is the median — half of all delegations complete faster. P90 and P95 capture tail latency, the durations that affect perceived responsiveness. A large gap between P50 and P95 suggests bimodal execution: some tasks are fast lookups, others are deep multi-step operations.": "> **Reading guide:** P50 is the median — half of all delegations complete faster. P90 and P95 capture tail latency, the durations that affect perceived responsiveness. A large gap between P50 and P95 suggests bimodal execution: some tasks are fast lookups, others are deep multi-step operations.", "> **Tip:** Delegations appear automatically when the agent calls `delegate_task`. Check that the subagent executor is configured and that the task requires delegation depth > 0.": "> **Tip:** Delegations appear automatically when the agent calls `delegate_task`. Check that the subagent executor is configured and that the task requires delegation depth > 0."
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
      "outside": "越界", "unknown": "未知",
      "Sub-Agent Monitor": "子代理监控", "No subagent activity": "无子代理活动", "This board shows delegated task execution. No subagent has been dispatched yet.": "此面板显示委托任务执行情况。目前尚未派发子代理。", "What are subagents?": "什么是子代理？", "> Subagents are spawned when the main agent delegates a task via `delegate_task`. Each runs in an isolated context with its own tool set and message history. Only the summary flows back to the parent. Activity will appear here once a delegation occurs.": "> 当主代理通过 `delegate_task` 委托任务时会创建子代理。每个子代理运行在隔离的上下文中，拥有独立的工具集和消息历史。只有摘要会返回给父代理。一旦发生委托，活动将显示在此处。",
      "Max depth": "最大深度", "Max concurrent": "最大并发", "Summary budget": "摘要预算", "Delegation overview": "委托概览", "Aggregate counters for all subagent executions in this daemon lifetime.": "此 daemon 生命周期内所有子代理执行的汇总计数。", "Active": "活跃", "Completed": "已完成", "Failed": "失败", "Avg duration": "平均耗时", "Success rate": "成功率",
      "Active subagents": "活跃子代理", "Currently running delegated tasks.": "当前正在运行的委托任务。", "ID": "标识", "Goal": "目标", "Depth": "深度", "Elapsed (s)": "已用时间 (s)", "Parent": "父代理", "Recent completions": "最近完成", "Last 50 subagent executions, newest first.": "最近50次子代理执行，最新优先。",
      "Execution detail": "执行详情", "Tabular view of recent subagent runs with outcome and duration.": "最近子代理运行的表格视图，含结果和耗时。", "Duration (s)": "耗时 (s)", "Delegation Tree": "委托树", "Parent → child relationships": "父→子关系", "How tasks were delegated across depth levels.": "任务在各深度层级间的委托方式。", "Child": "子代理",
      "Configuration": "配置", "Current subagent isolation settings.": "当前子代理隔离设置。", "Statistics": "统计", "Delegation by depth": "按深度委托", "How many subagents ran at each recursion depth.": "每个递归深度运行了多少子代理。", "Executions by depth": "按深度执行次数", "Outcomes": "执行结果", "Distribution of subagent execution outcomes.": "子代理执行结果分布。", "Executions by outcome": "按结果执行次数", "Cumulative metrics": "累计指标", "Total delegated": "总委托数", "Total tool calls": "总工具调用", "Total duration": "总耗时",
      "Active delegations": "活跃委托", "Aggregate counters and success metrics for all subagent executions in this daemon lifetime.": "此 daemon 生命周期内所有子代理执行的汇总计数和成功指标。", "Attention required": "需要关注", "Avg delegation depth": "平均委托深度", "Avg depth": "平均深度", "Avg tools/task": "平均工具/任务", "Bucketed execution times. A right-skewed distribution is normal; heavy >60s tail warrants investigation.": "分桶执行时间。右偏分布正常；>60s 的长尾值得关注。", "Current isolation settings governing all subagent delegations.": "管控所有子代理委托的当前隔离设置。", "Currently running subagent tasks. Elapsed time updates on each refresh cycle.": "当前运行的子代理任务。已用时间在每次刷新时更新。", "Deepest goal": "最深目标", "Delegation Graph": "委托图", "Delegation frequency": "委托频率", "Delegation topology": "委托拓扑", "Delegations aggregated by time window. Rising trend indicates increasing task complexity or workload.": "按时间窗口聚合的委托。上升趋势表示任务复杂度或工作量增加。", "Delegations by hour-of-day (00:00–23:00). Peaks indicate active work sessions.": "按小时（00:00–23:00）的委托。峰值表示活跃的工作时段。", "Depth analysis": "深度分析", "Depth saturation": "深度饱和", "Description": "描述", "Distribution of delegation recursion depth. Most delegations should cluster at depth 0-1; deeper levels indicate multi-step decomposition.": "委托递归深度分布。大多数委托应集中在深度 0-1。", "Duration analysis": "耗时分析", "Duration distribution": "耗时分布", "Effective subagent configuration": "生效的子代理配置", "Efficiency metrics": "效率指标", "Error": "错误", "Execution History": "执行历史", "Executions by depth level": "按深度层级的执行次数", "Failure rate": "失败率", "Health anomalies detected across recent delegations. Shown regardless of the open tab.": "在最近的委托中检测到健康异常。无论打开哪个标签页都会显示。", "High failure rate": "高失败率", "Hourly delegation pattern": "每小时委托模式", "How deeply tasks are delegated. Depth saturation indicates potential recursion limits.": "任务委托的深度。深度饱和表示可能达到递归限制。", "How many tool calls each delegation required. High counts may indicate task complexity or inefficient tool use.": "每次委托所需的工具调用次数。高值可能表示任务复杂或工具使用低效。", "How tasks were delegated across depth levels, with execution outcome and duration.": "任务在各深度层级间的委托方式，含执行结果和耗时。", "Last 50 subagent executions, newest first. Each entry shows outcome, duration, and tool usage.": "最近 50 次子代理执行，最新优先。每条显示结果、耗时和工具使用。", "Latency percentiles computed from all recent execution durations.": "根据所有最近执行耗时计算的延迟百分位数。", "Max depth reached": "已达最大深度", "Max observed depth": "最大观测深度", "Median duration": "中位耗时", "Outcome": "结果", "Outcome distribution": "结果分布", "P50": "P50", "P75": "P75", "P90": "P90", "P95": "P95", "Percentile summary": "百分位摘要", "Performance Analytics": "性能分析", "Samples": "样本数", "Setting": "设置项", "Statistical breakdown of execution times across all completed delegations.": "所有已完成委托的执行时间统计分析。", "Status": "状态", "Success vs failure ratio. A healthy system shows predominantly 'completed' outcomes.": "成功与失败比率。健康系统以已完成结果为主。", "Tabular view with full execution metadata. Error column populated for failed delegations.": "带完整执行元数据的表格视图。失败的委托会填充错误列。", "Temporal distribution": "时间分布", "This board reports delegated task execution. No subagent has been dispatched yet.": "此面板报告委托任务的执行情况。目前尚未派发子代理。", "Timeout detected": "检测到超时", "Tool calls per delegation": "每次委托的工具调用", "Tool usage patterns and outcome ratios across delegations.": "委托中的工具使用模式和结果比率。", "Tools": "工具", "Unique goals": "唯一目标数", "Visual map of parent→child delegation pairs. Badge color reflects execution outcome.": "父→子委托对的可视化地图。徽章颜色反映执行结果。", "When delegations occur within the day. Helps identify burst patterns and quiet periods.": "一天中委托发生的时间。有助于识别突发模式和安静时段。", "> **Configuration tuning guide:** `max_depth` controls recursion — increase for complex multi-step tasks, decrease to prevent runaway delegation chains. `max_concurrent` bounds parallelism — higher values improve throughput but increase resource contention. `summary_max_chars` limits the summary flowing back to the parent; too low truncates critical context, too high wastes the parent's context budget.": "> **配置调优指南：** `max_depth` 控制递归。`max_concurrent` 限制并行度。`summary_max_chars` 限制返回父代理的摘要长度。", "> **Reading guide:** P50 is the median — half of all delegations complete faster. P90 and P95 capture tail latency, the durations that affect perceived responsiveness. A large gap between P50 and P95 suggests bimodal execution: some tasks are fast lookups, others are deep multi-step operations.": "> **阅读指南：** P50 是中位数。P90 和 P95 捕获尾部延迟。P50 和 P95 之间的大差距表明双峰执行。", "> **Tip:** Delegations appear automatically when the agent calls `delegate_task`. Check that the subagent executor is configured and that the task requires delegation depth > 0.": "> **提示：** 当代理调用 `delegate_task` 时委托自动出现。请检查子代理执行器是否已配置且任务深度 > 0。"
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
      "near": "proche de la limite", "outside": "hors limites", "unknown": "inconnu",
      "Sub-Agent Monitor": "Moniteur de sous-agents", "No subagent activity": "Aucune activité de sous-agent", "This board shows delegated task execution. No subagent has been dispatched yet.": "Ce tableau affiche l'exécution des tâches déléguées. Aucun sous-agent n'a encore été lancé.", "What are subagents?": "Que sont les sous-agents ?", "> Subagents are spawned when the main agent delegates a task via `delegate_task`. Each runs in an isolated context with its own tool set and message history. Only the summary flows back to the parent. Activity will appear here once a delegation occurs.": "> Les sous-agents sont créés lorsque l'agent principal délègue une tâche via `delegate_task`. Chacun s'exécute dans un contexte isolé avec ses propres outils et historique de messages. Seul le résumé remonte au parent. L'activité apparaîtra ici dès qu'une délégation aura lieu.",
      "Max depth": "Profondeur max", "Max concurrent": "Simultanéité max", "Summary budget": "Budget de résumé", "Delegation overview": "Vue d'ensemble des délégations", "Aggregate counters for all subagent executions in this daemon lifetime.": "Compteurs agrégés de toutes les exécutions de sous-agents durant la vie de ce daemon.", "Active": "Actifs", "Completed": "Terminés", "Failed": "Échoués", "Avg duration": "Durée moy.", "Success rate": "Taux de réussite",
      "Active subagents": "Sous-agents actifs", "Currently running delegated tasks.": "Tâches déléguées en cours d'exécution.", "ID": "ID", "Goal": "Objectif", "Depth": "Profondeur", "Elapsed (s)": "Écoulé (s)", "Parent": "Parent", "Recent completions": "Complétions récentes", "Last 50 subagent executions, newest first.": "50 dernières exécutions de sous-agents, plus récentes d'abord.",
      "Execution detail": "Détail d'exécution", "Tabular view of recent subagent runs with outcome and duration.": "Vue tabulaire des exécutions récentes avec résultat et durée.", "Duration (s)": "Durée (s)", "Delegation Tree": "Arbre de délégation", "Parent → child relationships": "Relations parent → enfant", "How tasks were delegated across depth levels.": "Comment les tâches ont été déléguées à travers les niveaux.", "Child": "Enfant",
      "Configuration": "Configuration", "Current subagent isolation settings.": "Paramètres d'isolation actuels des sous-agents.", "Statistics": "Statistiques", "Delegation by depth": "Délégation par profondeur", "How many subagents ran at each recursion depth.": "Nombre de sous-agents exécutés à chaque profondeur de récursion.", "Executions by depth": "Exécutions par profondeur", "Outcomes": "Résultats", "Distribution of subagent execution outcomes.": "Distribution des résultats d'exécution des sous-agents.", "Executions by outcome": "Exécutions par résultat", "Cumulative metrics": "Métriques cumulées", "Total delegated": "Total délégué", "Total tool calls": "Total d'appels d'outils", "Total duration": "Durée totale",
      "Active delegations": "Délégations actives", "Aggregate counters and success metrics for all subagent executions in this daemon lifetime.": "Compteurs agrégés et métriques de succès.", "Attention required": "Attention requise", "Avg delegation depth": "Prof. moy. de délégation", "Avg depth": "Prof. moy.", "Avg tools/task": "Outils moy./tâche", "Bucketed execution times. A right-skewed distribution is normal; heavy >60s tail warrants investigation.": "Temps d'exécution par tranches. Queue >60s à investiguer.", "Current isolation settings governing all subagent delegations.": "Paramètres d'isolation régissant toutes les délégations.", "Currently running subagent tasks. Elapsed time updates on each refresh cycle.": "Tâches de sous-agents en cours. Mise à jour à chaque cycle.", "Deepest goal": "Objectif le plus profond", "Delegation Graph": "Graphe de délégation", "Delegation frequency": "Fréquence de délégation", "Delegation topology": "Topologie de délégation", "Delegations aggregated by time window. Rising trend indicates increasing task complexity or workload.": "Délégations agrégées par fenêtre. Tendance haussière = complexité croissante.", "Delegations by hour-of-day (00:00–23:00). Peaks indicate active work sessions.": "Délégations par heure (00:00–23:00). Les pics indiquent les sessions actives.", "Depth analysis": "Analyse de profondeur", "Depth saturation": "Saturation de profondeur", "Description": "Description", "Distribution of delegation recursion depth. Most delegations should cluster at depth 0-1; deeper levels indicate multi-step decomposition.": "Distribution de la profondeur de récursion. Majorité en 0-1.", "Duration analysis": "Analyse de durée", "Duration distribution": "Distribution de durée", "Effective subagent configuration": "Configuration effective des sous-agents", "Efficiency metrics": "Métriques d'efficacité", "Error": "Erreur", "Execution History": "Historique d'exécution", "Executions by depth level": "Exécutions par niveau", "Failure rate": "Taux d'échec", "Health anomalies detected across recent delegations. Shown regardless of the open tab.": "Anomalies détectées. Affiché quel que soit l'onglet.", "High failure rate": "Taux d'échec élevé", "Hourly delegation pattern": "Modèle horaire de délégation", "How deeply tasks are delegated. Depth saturation indicates potential recursion limits.": "Profondeur de délégation. La saturation indique des limites.", "How many tool calls each delegation required. High counts may indicate task complexity or inefficient tool use.": "Appels d'outils par délégation. Nombre élevé = complexité.", "How tasks were delegated across depth levels, with execution outcome and duration.": "Comment les tâches ont été déléguées, avec résultat et durée.", "Last 50 subagent executions, newest first. Each entry shows outcome, duration, and tool usage.": "50 dernières exécutions. Résultat, durée et outils.", "Latency percentiles computed from all recent execution durations.": "Centiles de latence calculés à partir des durées récentes.", "Max depth reached": "Profondeur max atteinte", "Max observed depth": "Profondeur max observée", "Median duration": "Durée médiane", "Outcome": "Résultat", "Outcome distribution": "Distribution des résultats", "P50": "P50", "P75": "P75", "P90": "P90", "P95": "P95", "Percentile summary": "Résumé des centiles", "Performance Analytics": "Analyse de performance", "Samples": "Échantillons", "Setting": "Paramètre", "Statistical breakdown of execution times across all completed delegations.": "Analyse statistique des temps d'exécution.", "Status": "Statut", "Success vs failure ratio. A healthy system shows predominantly 'completed' outcomes.": "Ratio succès/échecs. Système sain = résultats terminés.", "Tabular view with full execution metadata. Error column populated for failed delegations.": "Vue tabulaire avec métadonnées. Colonne erreur pour les échecs.", "Temporal distribution": "Distribution temporelle", "This board reports delegated task execution. No subagent has been dispatched yet.": "Ce tableau rapporte l'exécution des tâches déléguées.", "Timeout detected": "Timeout détecté", "Tool calls per delegation": "Appels d'outils par délégation", "Tool usage patterns and outcome ratios across delegations.": "Patterns d'outils et ratios de résultats.", "Tools": "Outils", "Unique goals": "Objectifs uniques", "Visual map of parent→child delegation pairs. Badge color reflects execution outcome.": "Carte visuelle des paires parent→enfant.", "When delegations occur within the day. Helps identify burst patterns and quiet periods.": "Quand les délégations surviennent. Aide à identifier pics et accalmies.", "> **Configuration tuning guide:** `max_depth` controls recursion — increase for complex multi-step tasks, decrease to prevent runaway delegation chains. `max_concurrent` bounds parallelism — higher values improve throughput but increase resource contention. `summary_max_chars` limits the summary flowing back to the parent; too low truncates critical context, too high wastes the parent's context budget.": "> **Guide de réglage :** `max_depth` contrôle la récursion. `max_concurrent` limite le parallélisme. `summary_max_chars` limite le résumé.", "> **Reading guide:** P50 is the median — half of all delegations complete faster. P90 and P95 capture tail latency, the durations that affect perceived responsiveness. A large gap between P50 and P95 suggests bimodal execution: some tasks are fast lookups, others are deep multi-step operations.": "> **Guide de lecture :** P50 est la médiane. P90 et P95 mesurent la latence de queue. Écart P50–P95 = exécution bimodale.", "> **Tip:** Delegations appear automatically when the agent calls `delegate_task`. Check that the subagent executor is configured and that the task requires delegation depth > 0.": "> **Astuce :** Les délégations apparaissent automatiquement. Vérifiez l'exécuteur et la profondeur > 0."
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
      "near": "cerca del límite", "outside": "fuera", "unknown": "desconocido",
      "Sub-Agent Monitor": "Monitor de subagentes", "No subagent activity": "Sin actividad de subagentes", "This board shows delegated task execution. No subagent has been dispatched yet.": "Este panel muestra la ejecución de tareas delegadas. Aún no se ha lanzado ningún subagente.", "What are subagents?": "¿Qué son los subagentes?", "> Subagents are spawned when the main agent delegates a task via `delegate_task`. Each runs in an isolated context with its own tool set and message history. Only the summary flows back to the parent. Activity will appear here once a delegation occurs.": "> Los subagentes se crean cuando el agente principal delega una tarea mediante `delegate_task`. Cada uno se ejecuta en un contexto aislado con su propio conjunto de herramientas e historial de mensajes. Solo el resumen vuelve al padre. La actividad aparecerá aquí una vez que ocurra una delegación.",
      "Max depth": "Profundidad máx.", "Max concurrent": "Concurrencia máx.", "Summary budget": "Presupuesto de resumen", "Delegation overview": "Resumen de delegaciones", "Aggregate counters for all subagent executions in this daemon lifetime.": "Contadores agregados de todas las ejecuciones de subagentes en la vida de este daemon.", "Active": "Activos", "Completed": "Completados", "Failed": "Fallidos", "Avg duration": "Duración prom.", "Success rate": "Tasa de éxito",
      "Active subagents": "Subagentes activos", "Currently running delegated tasks.": "Tareas delegadas en ejecución.", "ID": "ID", "Goal": "Objetivo", "Depth": "Profundidad", "Elapsed (s)": "Transcurrido (s)", "Parent": "Padre", "Recent completions": "Completados recientes", "Last 50 subagent executions, newest first.": "Últimas 50 ejecuciones de subagentes, más recientes primero.",
      "Execution detail": "Detalle de ejecución", "Tabular view of recent subagent runs with outcome and duration.": "Vista tabular de ejecuciones recientes con resultado y duración.", "Duration (s)": "Duración (s)", "Delegation Tree": "Árbol de delegación", "Parent → child relationships": "Relaciones padre → hijo", "How tasks were delegated across depth levels.": "Cómo se delegaron las tareas entre niveles de profundidad.", "Child": "Hijo",
      "Configuration": "Configuración", "Current subagent isolation settings.": "Configuración actual de aislamiento de subagentes.", "Statistics": "Estadísticas", "Delegation by depth": "Delegación por profundidad", "How many subagents ran at each recursion depth.": "Cuántos subagentes se ejecutaron en cada profundidad de recursión.", "Executions by depth": "Ejecuciones por profundidad", "Outcomes": "Resultados", "Distribution of subagent execution outcomes.": "Distribución de resultados de ejecución de subagentes.", "Executions by outcome": "Ejecuciones por resultado", "Cumulative metrics": "Métricas acumuladas", "Total delegated": "Total delegado", "Total tool calls": "Total de llamadas", "Total duration": "Duración total",
      "Active delegations": "Delegaciones activas", "Aggregate counters and success metrics for all subagent executions in this daemon lifetime.": "Contadores agregados y métricas de éxito.", "Attention required": "Atención requerida", "Avg delegation depth": "Prof. prom. de delegación", "Avg depth": "Prof. prom.", "Avg tools/task": "Herr. prom./tarea", "Bucketed execution times. A right-skewed distribution is normal; heavy >60s tail warrants investigation.": "Tiempos agrupados. Cola >60s a investigar.", "Current isolation settings governing all subagent delegations.": "Configuración de aislamiento.", "Currently running subagent tasks. Elapsed time updates on each refresh cycle.": "Tareas en ejecución. Tiempo actualizado en cada ciclo.", "Deepest goal": "Objetivo más profundo", "Delegation Graph": "Gráfico de delegación", "Delegation frequency": "Frecuencia de delegación", "Delegation topology": "Topología de delegación", "Delegations aggregated by time window. Rising trend indicates increasing task complexity or workload.": "Delegaciones por ventana. Tendencia ascendente = mayor complejidad.", "Delegations by hour-of-day (00:00–23:00). Peaks indicate active work sessions.": "Delegaciones por hora (00:00–23:00). Picos = sesiones activas.", "Depth analysis": "Análisis de profundidad", "Depth saturation": "Saturación de profundidad", "Description": "Descripción", "Distribution of delegation recursion depth. Most delegations should cluster at depth 0-1; deeper levels indicate multi-step decomposition.": "Distribución de profundidad. Mayoría en 0-1.", "Duration analysis": "Análisis de duración", "Duration distribution": "Distribución de duración", "Effective subagent configuration": "Configuración efectiva", "Efficiency metrics": "Métricas de eficiencia", "Error": "Error", "Execution History": "Historial de ejecución", "Executions by depth level": "Ejecuciones por nivel", "Failure rate": "Tasa de fallos", "Health anomalies detected across recent delegations. Shown regardless of the open tab.": "Anomalías detectadas. Se muestra en cualquier pestaña.", "High failure rate": "Alta tasa de fallos", "Hourly delegation pattern": "Patrón horario", "How deeply tasks are delegated. Depth saturation indicates potential recursion limits.": "Profundidad de delegación. Saturación = límites.", "How many tool calls each delegation required. High counts may indicate task complexity or inefficient tool use.": "Llamadas por delegación. Conteos altos = complejidad.", "How tasks were delegated across depth levels, with execution outcome and duration.": "Cómo se delegaron, con resultado y duración.", "Last 50 subagent executions, newest first. Each entry shows outcome, duration, and tool usage.": "Últimas 50 ejecuciones. Resultado, duración y herramientas.", "Latency percentiles computed from all recent execution durations.": "Percentiles de latencia calculados.", "Max depth reached": "Prof. máx. alcanzada", "Max observed depth": "Prof. máx. observada", "Median duration": "Duración mediana", "Outcome": "Resultado", "Outcome distribution": "Distribución de resultados", "P50": "P50", "P75": "P75", "P90": "P90", "P95": "P95", "Percentile summary": "Resumen de percentiles", "Performance Analytics": "Análisis de rendimiento", "Samples": "Muestras", "Setting": "Ajuste", "Statistical breakdown of execution times across all completed delegations.": "Desglose estadístico de tiempos.", "Status": "Estado", "Success vs failure ratio. A healthy system shows predominantly 'completed' outcomes.": "Ratio éxito/fallo. Sistema sano = completados.", "Tabular view with full execution metadata. Error column populated for failed delegations.": "Vista tabular con metadatos. Columna de error para fallos.", "Temporal distribution": "Distribución temporal", "This board reports delegated task execution. No subagent has been dispatched yet.": "Este panel informa sobre ejecución de tareas delegadas.", "Timeout detected": "Timeout detectado", "Tool calls per delegation": "Llamadas por delegación", "Tool usage patterns and outcome ratios across delegations.": "Patrones y ratios de resultados.", "Tools": "Herramientas", "Unique goals": "Objetivos únicos", "Visual map of parent→child delegation pairs. Badge color reflects execution outcome.": "Mapa visual de pares padre→hijo.", "When delegations occur within the day. Helps identify burst patterns and quiet periods.": "Cuándo ocurren las delegaciones. Patrones y periodos.", "> **Configuration tuning guide:** `max_depth` controls recursion — increase for complex multi-step tasks, decrease to prevent runaway delegation chains. `max_concurrent` bounds parallelism — higher values improve throughput but increase resource contention. `summary_max_chars` limits the summary flowing back to the parent; too low truncates critical context, too high wastes the parent's context budget.": "> **Guía de ajuste:** `max_depth` controla la recursión. `max_concurrent` limita el paralelismo. `summary_max_chars` limita el resumen.", "> **Reading guide:** P50 is the median — half of all delegations complete faster. P90 and P95 capture tail latency, the durations that affect perceived responsiveness. A large gap between P50 and P95 suggests bimodal execution: some tasks are fast lookups, others are deep multi-step operations.": "> **Guía de lectura:** P50 es la mediana. P90 y P95 capturan latencia de cola. Brecha P50–P95 = bimodal.", "> **Tip:** Delegations appear automatically when the agent calls `delegate_task`. Check that the subagent executor is configured and that the task requires delegation depth > 0.": "> **Consejo:** Delegaciones automáticas al llamar `delegate_task`. Verifique ejecutor y profundidad > 0."
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
      "near": "قريب من الحد", "outside": "خارج الحدود", "unknown": "مجهول",
      "Sub-Agent Monitor": "مراقب الوكلاء الفرعيين", "No subagent activity": "لا يوجد نشاط للوكلاء الفرعيين", "This board shows delegated task execution. No subagent has been dispatched yet.": "تعرض هذه اللوحة تنفيذ المهام المفوَّضة. لم يتم إرسال أي وكيل فرعي بعد.", "What are subagents?": "ما هي الوكلاء الفرعيون؟", "> Subagents are spawned when the main agent delegates a task via `delegate_task`. Each runs in an isolated context with its own tool set and message history. Only the summary flows back to the parent. Activity will appear here once a delegation occurs.": "> يتم إنشاء الوكلاء الفرعيين عندما يفوّض الوكيل الرئيسي مهمة عبر `delegate_task`. يعمل كل منهم في سياق معزول بأدواته وسجل رسائله الخاص. يُرجَع الملخص فقط إلى الوكيل الأب. سيظهر النشاط هنا عند حدوث تفويض.",
      "Max depth": "أقصى عمق", "Max concurrent": "أقصى تزامن", "Summary budget": "ميزانية الملخص", "Delegation overview": "نظرة عامة على التفويض", "Aggregate counters for all subagent executions in this daemon lifetime.": "عدادات تراكمية لجميع عمليات تنفيذ الوكلاء الفرعيين خلال حياة هذا الـ daemon.", "Active": "نشط", "Completed": "مكتمل", "Failed": "فشل", "Avg duration": "متوسط المدة", "Success rate": "معدل النجاح",
      "Active subagents": "الوكلاء الفرعيون النشطون", "Currently running delegated tasks.": "المهام المفوّضة قيد التنفيذ حالياً.", "ID": "المعرّف", "Goal": "الهدف", "Depth": "العمق", "Elapsed (s)": "المنقضي (ث)", "Parent": "الأب", "Recent completions": "الإنجازات الأخيرة", "Last 50 subagent executions, newest first.": "آخر 50 عملية تنفيذ للوكلاء الفرعيين، الأحدث أولاً.",
      "Execution detail": "تفاصيل التنفيذ", "Tabular view of recent subagent runs with outcome and duration.": "عرض جدولي لعمليات التنفيذ الأخيرة مع النتيجة والمدة.", "Duration (s)": "المدة (ث)", "Delegation Tree": "شجرة التفويض", "Parent → child relationships": "علاقات الأب → الابن", "How tasks were delegated across depth levels.": "كيف تم تفويض المهام عبر مستويات العمق.", "Child": "الابن",
      "Configuration": "الإعدادات", "Current subagent isolation settings.": "إعدادات العزل الحالية للوكلاء الفرعيين.", "Statistics": "الإحصائيات", "Delegation by depth": "التفويض حسب العمق", "How many subagents ran at each recursion depth.": "عدد الوكلاء الفرعيين الذين عملوا في كل مستوى تكرار.", "Executions by depth": "عمليات التنفيذ حسب العمق", "Outcomes": "النتائج", "Distribution of subagent execution outcomes.": "توزيع نتائج تنفيذ الوكلاء الفرعيين.", "Executions by outcome": "عمليات التنفيذ حسب النتيجة", "Cumulative metrics": "مقاييس تراكمية", "Total delegated": "إجمالي المفوّض", "Total tool calls": "إجمالي استدعاءات الأدوات", "Total duration": "المدة الإجمالية",
      "Active delegations": "التفويضات النشطة", "Aggregate counters and success metrics for all subagent executions in this daemon lifetime.": "عدادات تراكمية ومقاييس نجاح.", "Attention required": "يتطلب الانتباه", "Avg delegation depth": "متوسط عمق التفويض", "Avg depth": "متوسط العمق", "Avg tools/task": "متوسط الأدوات/المهمة", "Bucketed execution times. A right-skewed distribution is normal; heavy >60s tail warrants investigation.": "أوقات التنفيذ المجمّعة. ذيل >60 ثانية يستحق التحقيق.", "Current isolation settings governing all subagent delegations.": "إعدادات العزل الحالية.", "Currently running subagent tasks. Elapsed time updates on each refresh cycle.": "مهام قيد التنفيذ. الوقت يُحدَّث في كل دورة.", "Deepest goal": "أعمق هدف", "Delegation Graph": "رسم بياني للتفويض", "Delegation frequency": "تكرار التفويض", "Delegation topology": "طوبولوجيا التفويض", "Delegations aggregated by time window. Rising trend indicates increasing task complexity or workload.": "التفويضات مجمّعة. الاتجاه التصاعدي = زيادة التعقيد.", "Delegations by hour-of-day (00:00–23:00). Peaks indicate active work sessions.": "التفويضات حسب الساعة (00:00–23:00). القمم = جلسات نشطة.", "Depth analysis": "تحليل العمق", "Depth saturation": "تشبع العمق", "Description": "الوصف", "Distribution of delegation recursion depth. Most delegations should cluster at depth 0-1; deeper levels indicate multi-step decomposition.": "توزيع عمق التكرار. معظمها في 0-1.", "Duration analysis": "تحليل المدة", "Duration distribution": "توزيع المدة", "Effective subagent configuration": "الإعدادات الفعّالة", "Efficiency metrics": "مقاييس الكفاءة", "Error": "خطأ", "Execution History": "سجل التنفيذ", "Executions by depth level": "عمليات حسب المستوى", "Failure rate": "معدل الفشل", "Health anomalies detected across recent delegations. Shown regardless of the open tab.": "تم اكتشاف حالات شاذة. تُعرض بغض النظر عن التبويب.", "High failure rate": "معدل فشل مرتفع", "Hourly delegation pattern": "نمط حسب الساعة", "How deeply tasks are delegated. Depth saturation indicates potential recursion limits.": "مدى عمق التفويض. التشبع = حدود.", "How many tool calls each delegation required. High counts may indicate task complexity or inefficient tool use.": "استدعاءات لكل تفويض. أرقام عالية = تعقيد.", "How tasks were delegated across depth levels, with execution outcome and duration.": "كيف تم التفويض مع النتيجة والمدة.", "Last 50 subagent executions, newest first. Each entry shows outcome, duration, and tool usage.": "آخر 50 عملية. النتيجة والمدة والأدوات.", "Latency percentiles computed from all recent execution durations.": "مئويات التأخير المحسوبة.", "Max depth reached": "أقصى عمق", "Max observed depth": "أقصى عمق مُلاحَظ", "Median duration": "المدة الوسيطة", "Outcome": "النتيجة", "Outcome distribution": "توزيع النتائج", "P50": "P50", "P75": "P75", "P90": "P90", "P95": "P95", "Percentile summary": "ملخص المئويات", "Performance Analytics": "تحليل الأداء", "Samples": "العينات", "Setting": "الإعداد", "Statistical breakdown of execution times across all completed delegations.": "تحليل إحصائي لأوقات التنفيذ.", "Status": "الحالة", "Success vs failure ratio. A healthy system shows predominantly 'completed' outcomes.": "نسبة النجاح مقابل الفشل.", "Tabular view with full execution metadata. Error column populated for failed delegations.": "عرض جدولي مع بيانات التنفيذ.", "Temporal distribution": "التوزيع الزمني", "This board reports delegated task execution. No subagent has been dispatched yet.": "تعرض هذه اللوحة تقارير التنفيذ.", "Timeout detected": "تم اكتشاف انتهاء مهلة", "Tool calls per delegation": "استدعاءات لكل تفويض", "Tool usage patterns and outcome ratios across delegations.": "أنماط الأدوات ونسب النتائج.", "Tools": "الأدوات", "Unique goals": "الأهداف الفريدة", "Visual map of parent→child delegation pairs. Badge color reflects execution outcome.": "خريطة مرئية لأزواج التفويض.", "When delegations occur within the day. Helps identify burst patterns and quiet periods.": "متى تحدث التفويضات. أنماط الذروة والهدوء.", "> **Configuration tuning guide:** `max_depth` controls recursion — increase for complex multi-step tasks, decrease to prevent runaway delegation chains. `max_concurrent` bounds parallelism — higher values improve throughput but increase resource contention. `summary_max_chars` limits the summary flowing back to the parent; too low truncates critical context, too high wastes the parent's context budget.": "> **دليل الضبط:** `max_depth` يتحكم في التكرار. `max_concurrent` يحد التوازي. `summary_max_chars` يحد الملخص.", "> **Reading guide:** P50 is the median — half of all delegations complete faster. P90 and P95 capture tail latency, the durations that affect perceived responsiveness. A large gap between P50 and P95 suggests bimodal execution: some tasks are fast lookups, others are deep multi-step operations.": "> **دليل القراءة:** P50 هو الوسيط. P90 وP95 = تأخير الذيل. فجوة = تنفيذ ثنائي.", "> **Tip:** Delegations appear automatically when the agent calls `delegate_task`. Check that the subagent executor is configured and that the task requires delegation depth > 0.": "> **تلميح:** التفويضات تظهر تلقائياً. تأكد من المنفّذ والعمق > 0."
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
      "near": "у границы", "outside": "вне допуска", "unknown": "неизвестно",
      "Sub-Agent Monitor": "Монитор субагентов", "No subagent activity": "Нет активности субагентов", "This board shows delegated task execution. No subagent has been dispatched yet.": "Эта панель показывает выполнение делегированных задач. Ни один субагент ещё не был запущен.", "What are subagents?": "Что такое субагенты?", "> Subagents are spawned when the main agent delegates a task via `delegate_task`. Each runs in an isolated context with its own tool set and message history. Only the summary flows back to the parent. Activity will appear here once a delegation occurs.": "> Субагенты создаются, когда основной агент делегирует задачу через `delegate_task`. Каждый работает в изолированном контексте со своим набором инструментов и историей сообщений. Только резюме возвращается родителю. Активность появится здесь при делегировании.",
      "Max depth": "Макс. глубина", "Max concurrent": "Макс. параллельно", "Summary budget": "Лимит резюме", "Delegation overview": "Обзор делегирования", "Aggregate counters for all subagent executions in this daemon lifetime.": "Суммарные счётчики всех выполнений субагентов за время работы daemon.", "Active": "Активные", "Completed": "Завершены", "Failed": "Ошибки", "Avg duration": "Сред. длительность", "Success rate": "Успешность",
      "Active subagents": "Активные субагенты", "Currently running delegated tasks.": "Делегированные задачи, выполняющиеся сейчас.", "ID": "ID", "Goal": "Цель", "Depth": "Глубина", "Elapsed (s)": "Прошло (с)", "Parent": "Родитель", "Recent completions": "Недавние завершения", "Last 50 subagent executions, newest first.": "Последние 50 выполнений субагентов, новейшие первыми.",
      "Execution detail": "Детали выполнения", "Tabular view of recent subagent runs with outcome and duration.": "Табличное представление недавних выполнений с результатом и длительностью.", "Duration (s)": "Длит. (с)", "Delegation Tree": "Дерево делегирования", "Parent → child relationships": "Связи родитель → потомок", "How tasks were delegated across depth levels.": "Как задачи делегировались по уровням глубины.", "Child": "Потомок",
      "Configuration": "Конфигурация", "Current subagent isolation settings.": "Текущие настройки изоляции субагентов.", "Statistics": "Статистика", "Delegation by depth": "Делегирование по глубине", "How many subagents ran at each recursion depth.": "Сколько субагентов работало на каждой глубине рекурсии.", "Executions by depth": "Выполнения по глубине", "Outcomes": "Исходы", "Distribution of subagent execution outcomes.": "Распределение результатов выполнения субагентов.", "Executions by outcome": "Выполнения по результату", "Cumulative metrics": "Накопительные метрики", "Total delegated": "Всего делегировано", "Total tool calls": "Всего вызовов", "Total duration": "Общая длительность",
      "Active delegations": "Активные делегирования", "Aggregate counters and success metrics for all subagent executions in this daemon lifetime.": "Суммарные счётчики и метрики успешности.", "Attention required": "Требуется внимание", "Avg delegation depth": "Сред. глубина делегирования", "Avg depth": "Сред. глубина", "Avg tools/task": "Сред. инструм./задачу", "Bucketed execution times. A right-skewed distribution is normal; heavy >60s tail warrants investigation.": "Гистограмма времени. Хвост >60 с требует изучения.", "Current isolation settings governing all subagent delegations.": "Настройки изоляции.", "Currently running subagent tasks. Elapsed time updates on each refresh cycle.": "Задачи субагентов сейчас. Время обновляется.", "Deepest goal": "Самая глубокая цель", "Delegation Graph": "Граф делегирования", "Delegation frequency": "Частота делегирования", "Delegation topology": "Топология делегирования", "Delegations aggregated by time window. Rising trend indicates increasing task complexity or workload.": "Делегирования по окнам. Рост = усложнение.", "Delegations by hour-of-day (00:00–23:00). Peaks indicate active work sessions.": "Делегирования по часам (00:00–23:00). Пики = активные сеансы.", "Depth analysis": "Анализ глубины", "Depth saturation": "Насыщение глубины", "Description": "Описание", "Distribution of delegation recursion depth. Most delegations should cluster at depth 0-1; deeper levels indicate multi-step decomposition.": "Распределение глубины. Большинство на 0-1.", "Duration analysis": "Анализ длительности", "Duration distribution": "Распределение длительности", "Effective subagent configuration": "Действующая конфигурация", "Efficiency metrics": "Метрики эффективности", "Error": "Ошибка", "Execution History": "История выполнений", "Executions by depth level": "Выполнения по уровню", "Failure rate": "Процент ошибок", "Health anomalies detected across recent delegations. Shown regardless of the open tab.": "Аномалии. Отображается независимо от вкладки.", "High failure rate": "Высокий процент ошибок", "Hourly delegation pattern": "Почасовой шаблон", "How deeply tasks are delegated. Depth saturation indicates potential recursion limits.": "Глубина делегирования. Насыщение = лимиты.", "How many tool calls each delegation required. High counts may indicate task complexity or inefficient tool use.": "Вызовов на делегирование. Высокие = сложность.", "How tasks were delegated across depth levels, with execution outcome and duration.": "Как задачи делегировались, с результатом и длительностью.", "Last 50 subagent executions, newest first. Each entry shows outcome, duration, and tool usage.": "Последние 50 выполнений. Результат, длительность и инструменты.", "Latency percentiles computed from all recent execution durations.": "Перцентили задержки.", "Max depth reached": "Макс. достигнутая глубина", "Max observed depth": "Макс. наблюдаемая глубина", "Median duration": "Медианная длительность", "Outcome": "Результат", "Outcome distribution": "Распределение результатов", "P50": "P50", "P75": "P75", "P90": "P90", "P95": "P95", "Percentile summary": "Сводка перцентилей", "Performance Analytics": "Аналитика производительности", "Samples": "Выборки", "Setting": "Параметр", "Statistical breakdown of execution times across all completed delegations.": "Статистический анализ времени.", "Status": "Статус", "Success vs failure ratio. A healthy system shows predominantly 'completed' outcomes.": "Соотношение успехов и ошибок.", "Tabular view with full execution metadata. Error column populated for failed delegations.": "Табличное представление с метаданными.", "Temporal distribution": "Временное распределение", "This board reports delegated task execution. No subagent has been dispatched yet.": "Панель отражает делегированные задачи. Ни один субагент не запущен.", "Timeout detected": "Обнаружен таймаут", "Tool calls per delegation": "Вызовов на делегирование", "Tool usage patterns and outcome ratios across delegations.": "Паттерны инструментов и соотношения.", "Tools": "Инструменты", "Unique goals": "Уникальные цели", "Visual map of parent→child delegation pairs. Badge color reflects execution outcome.": "Визуальная карта пар родитель→потомок.", "When delegations occur within the day. Helps identify burst patterns and quiet periods.": "Когда делегирования происходят. Пики и затишья.", "> **Configuration tuning guide:** `max_depth` controls recursion — increase for complex multi-step tasks, decrease to prevent runaway delegation chains. `max_concurrent` bounds parallelism — higher values improve throughput but increase resource contention. `summary_max_chars` limits the summary flowing back to the parent; too low truncates critical context, too high wastes the parent's context budget.": "> **Руководство:** `max_depth` управляет рекурсией. `max_concurrent` ограничивает параллелизм. `summary_max_chars` ограничивает резюме.", "> **Reading guide:** P50 is the median — half of all delegations complete faster. P90 and P95 capture tail latency, the durations that affect perceived responsiveness. A large gap between P50 and P95 suggests bimodal execution: some tasks are fast lookups, others are deep multi-step operations.": "> **Справка:** P50 — медиана. P90 и P95 = хвостовая задержка. Разрыв P50–P95 = бимодальность.", "> **Tip:** Delegations appear automatically when the agent calls `delegate_task`. Check that the subagent executor is configured and that the task requires delegation depth > 0.": "> **Совет:** Делегирования автоматически. Проверьте исполнителя и глубину > 0."
    }
  };
  const I18N_LIVE = {
    en: {
      "Evolution live": "Evolution live", "Read-only event lanes for environment, decision, and governance evidence.": "Read-only event lanes for environment, decision, and governance evidence.",
      "Run posture": "Run posture", "Current snapshot and live increments stay distinct; missing evidence remains visible.": "Current snapshot and live increments stay distinct; missing evidence remains visible.",
      "Episodes": "Episodes", "Pipeline evidence": "Pipeline evidence", "Unadmitted intents": "Unadmitted intents", "Regressions": "Regressions",
      "Live event lanes": "Live event lanes", "Presentation-only trace projection. Reconnect resynchronizes from the current evolution snapshot.": "Presentation-only trace projection. Reconnect resynchronizes from the current evolution snapshot.",
      "Counterfactual / mutation matrix": "Counterfactual / mutation matrix", "Latest durable episode projection; rows do not imply an unrecorded install or approval.": "Latest durable episode projection; rows do not imply an unrecorded install or approval.",
      "Trigger": "Trigger", "Capability": "Capability", "Decision": "Decision", "Mutation": "Mutation", "Evidence tier": "Evidence tier", "Gap closure": "Gap closure",
      "Environment lane": "Environment lane", "Decision lane": "Decision lane", "Governance lane": "Governance lane", "Live events": "Live events", "Event count": "Event count", "No live events": "No live events", "Event detail": "Event detail", "Select a live event to inspect its evidence references.": "Select a live event to inspect its evidence references.",
      "Live since snapshot": "Live since snapshot",
      "{count} event(s) arrived after this snapshot. Snapshot metrics refresh on the next monitor cycle.": "{count} event(s) arrived after this snapshot. Snapshot metrics refresh on the next monitor cycle.",
      "Environment events appear when a probe records a change in the surroundings.": "Environment events appear when a probe records a change in the surroundings.",
      "Decision events appear when a recorded observation drives a capability choice.": "Decision events appear when a recorded observation drives a capability choice.",
      "Governance events appear when trust, quarantine or reclamation moves.": "Governance events appear when trust, quarantine or reclamation moves."
    },
    zh: {
      "Evolution live": "演化实时视图", "Read-only event lanes for environment, decision, and governance evidence.": "面向环境、决策和治理证据的只读事件泳道。",
      "Run posture": "运行态势", "Current snapshot and live increments stay distinct; missing evidence remains visible.": "当前快照与实时增量保持区分；缺失证据保持可见。",
      "Episodes": "剧集", "Pipeline evidence": "管道证据", "Unadmitted intents": "未准入意图", "Regressions": "回归",
      "Live event lanes": "实时事件泳道", "Presentation-only trace projection. Reconnect resynchronizes from the current evolution snapshot.": "仅展示的追踪投影。重连时从当前演化快照重新同步。",
      "Counterfactual / mutation matrix": "反事实 / 变更矩阵", "Latest durable episode projection; rows do not imply an unrecorded install or approval.": "最新持久剧集投影；行内容不代表未记录的安装或审批。",
      "Trigger": "触发源", "Capability": "能力", "Decision": "决策", "Mutation": "变更", "Evidence tier": "证据层级", "Gap closure": "缺口闭合",
      "Environment lane": "环境泳道", "Decision lane": "决策泳道", "Governance lane": "治理泳道", "Live events": "实时事件", "Event count": "事件数", "No live events": "暂无实时事件", "Event detail": "事件详情", "Select a live event to inspect its evidence references.": "选择实时事件以查看其证据引用。",
      "Live since snapshot": "快照后新增",
      "{count} event(s) arrived after this snapshot. Snapshot metrics refresh on the next monitor cycle.": "{count} 条事件在此快照之后到达。快照类指标将在下一个监控周期刷新。",
      "Environment events appear when a probe records a change in the surroundings.": "当探针记录到周边环境发生变化时，环境事件会出现在这里。",
      "Decision events appear when a recorded observation drives a capability choice.": "当已记录的观测驱动一次能力选择时，决策事件会出现在这里。",
      "Governance events appear when trust, quarantine or reclamation moves.": "当信任、隔离或回收发生变动时，治理事件会出现在这里。"
    },
    fr: {
      "Evolution live": "Évolution en direct", "Read-only event lanes for environment, decision, and governance evidence.": "Voies d’événements en lecture seule pour les preuves d’environnement, de décision et de gouvernance.",
      "Run posture": "État d’exécution", "Current snapshot and live increments stay distinct; missing evidence remains visible.": "L’instantané et les incréments restent distincts ; les preuves manquantes restent visibles.",
      "Episodes": "Épisodes", "Pipeline evidence": "Preuves du pipeline", "Unadmitted intents": "Intentions non admises", "Regressions": "Régressions",
      "Live event lanes": "Voies d’événements live", "Presentation-only trace projection. Reconnect resynchronizes from the current evolution snapshot.": "Projection de trace réservée à l’affichage. La reconnexion se resynchronise depuis l’instantané d’évolution actuel.",
      "Counterfactual / mutation matrix": "Matrice contrefactuelle / mutations", "Latest durable episode projection; rows do not imply an unrecorded install or approval.": "Projection du dernier épisode durable ; les lignes n’impliquent ni installation ni approbation non enregistrée.",
      "Trigger": "Déclencheur", "Capability": "Capacité", "Decision": "Décision", "Mutation": "Mutation", "Evidence tier": "Niveau de preuve", "Gap closure": "Clôture de l’écart",
      "Environment lane": "Voie environnement", "Decision lane": "Voie décision", "Governance lane": "Voie gouvernance", "Live events": "Événements live", "Event count": "Nombre d’événements", "No live events": "Aucun événement live", "Event detail": "Détail de l’événement", "Select a live event to inspect its evidence references.": "Sélectionnez un événement live pour examiner ses références de preuve.",
      "Live since snapshot": "En direct depuis l’instantané",
      "{count} event(s) arrived after this snapshot. Snapshot metrics refresh on the next monitor cycle.": "{count} événement(s) sont arrivés après cet instantané. Les métriques d’instantané seront actualisées au prochain cycle de surveillance.",
      "Environment events appear when a probe records a change in the surroundings.": "Les événements d’environnement apparaissent lorsqu’une sonde enregistre un changement des alentours.",
      "Decision events appear when a recorded observation drives a capability choice.": "Les événements de décision apparaissent lorsqu’une observation enregistrée motive un choix de capacité.",
      "Governance events appear when trust, quarantine or reclamation moves.": "Les événements de gouvernance apparaissent lorsque la confiance, la quarantaine ou la récupération évolue."
    },
    es: {
      "Evolution live": "Evolución en vivo", "Read-only event lanes for environment, decision, and governance evidence.": "Carriles de eventos de solo lectura para evidencia de entorno, decisión y gobernanza.",
      "Run posture": "Estado de ejecución", "Current snapshot and live increments stay distinct; missing evidence remains visible.": "La instantánea y los incrementos en vivo siguen siendo distintos; la evidencia faltante permanece visible.",
      "Episodes": "Episodios", "Pipeline evidence": "Evidencia del pipeline", "Unadmitted intents": "Intenciones no admitidas", "Regressions": "Regresiones",
      "Live event lanes": "Carriles de eventos en vivo", "Presentation-only trace projection. Reconnect resynchronizes from the current evolution snapshot.": "Proyección de trazas solo para presentación. La reconexión resincroniza desde la instantánea de evolución actual.",
      "Counterfactual / mutation matrix": "Matriz contrafactual / de mutaciones", "Latest durable episode projection; rows do not imply an unrecorded install or approval.": "Proyección del último episodio durable; las filas no implican instalación ni aprobación no registradas.",
      "Trigger": "Desencadenante", "Capability": "Capacidad", "Decision": "Decisión", "Mutation": "Mutación", "Evidence tier": "Nivel de evidencia", "Gap closure": "Cierre de la brecha",
      "Environment lane": "Carril de entorno", "Decision lane": "Carril de decisión", "Governance lane": "Carril de gobernanza", "Live events": "Eventos en vivo", "Event count": "Conteo de eventos", "No live events": "No hay eventos en vivo", "Event detail": "Detalle del evento", "Select a live event to inspect its evidence references.": "Seleccione un evento en vivo para inspeccionar sus referencias de evidencia.",
      "Live since snapshot": "En vivo desde la instantánea",
      "{count} event(s) arrived after this snapshot. Snapshot metrics refresh on the next monitor cycle.": "{count} evento(s) llegaron después de esta instantánea. Las métricas de instantánea se actualizarán en el próximo ciclo de monitoreo.",
      "Environment events appear when a probe records a change in the surroundings.": "Los eventos de entorno aparecen cuando una sonda registra un cambio en el entorno.",
      "Decision events appear when a recorded observation drives a capability choice.": "Los eventos de decisión aparecen cuando una observación registrada impulsa una elección de capacidad.",
      "Governance events appear when trust, quarantine or reclamation moves.": "Los eventos de gobernanza aparecen cuando cambian la confianza, la cuarentena o la recuperación."
    },
    ar: {
      "Evolution live": "التطور المباشر", "Read-only event lanes for environment, decision, and governance evidence.": "مسارات أحداث للقراءة فقط لأدلة البيئة والقرار والحوكمة.",
      "Run posture": "حالة التشغيل", "Current snapshot and live increments stay distinct; missing evidence remains visible.": "تظل اللقطة والزيادات المباشرة منفصلة؛ وتبقى الأدلة المفقودة مرئية.",
      "Episodes": "الحلقات", "Pipeline evidence": "أدلة المسار", "Unadmitted intents": "نيات غير مقبولة", "Regressions": "الانحدارات",
      "Live event lanes": "مسارات الأحداث المباشرة", "Presentation-only trace projection. Reconnect resynchronizes from the current evolution snapshot.": "إسقاط أثر للعرض فقط. تعيد إعادة الاتصال المزامنة من لقطة التطور الحالية.",
      "Counterfactual / mutation matrix": "مصفوفة الافتراضات المضادة / التغييرات", "Latest durable episode projection; rows do not imply an unrecorded install or approval.": "إسقاط أحدث حلقة دائمة؛ لا تعني الصفوف تثبيتًا أو موافقة غير مسجلة.",
      "Trigger": "المحفز", "Capability": "القدرة", "Decision": "القرار", "Mutation": "التغيير", "Evidence tier": "طبقة الدليل", "Gap closure": "إغلاق الفجوة",
      "Environment lane": "مسار البيئة", "Decision lane": "مسار القرار", "Governance lane": "مسار الحوكمة", "Live events": "الأحداث المباشرة", "Event count": "عدد الأحداث", "No live events": "لا توجد أحداث مباشرة", "Event detail": "تفاصيل الحدث", "Select a live event to inspect its evidence references.": "اختر حدثًا مباشرًا لفحص مراجع أدلته.",
      "Live since snapshot": "مباشر منذ اللقطة",
      "{count} event(s) arrived after this snapshot. Snapshot metrics refresh on the next monitor cycle.": "وصل {count} حدث بعد هذه اللقطة. تُحدَّث مقاييس اللقطة في دورة المراقبة التالية.",
      "Environment events appear when a probe records a change in the surroundings.": "تظهر أحداث البيئة عندما يسجّل مِجَسّ تغيّرًا في المحيط.",
      "Decision events appear when a recorded observation drives a capability choice.": "تظهر أحداث القرار عندما تدفع ملاحظة مسجّلة إلى اختيار قدرة.",
      "Governance events appear when trust, quarantine or reclamation moves.": "تظهر أحداث الحوكمة عند تغيّر الثقة أو الحجر أو الاستعادة."
    },
    ru: {
      "Evolution live": "Эволюция в реальном времени", "Read-only event lanes for environment, decision, and governance evidence.": "Потоки событий только для чтения для доказательств окружения, решений и управления.",
      "Run posture": "Состояние запуска", "Current snapshot and live increments stay distinct; missing evidence remains visible.": "Снимок и живые приращения остаются раздельными; отсутствующие доказательства остаются видимыми.",
      "Episodes": "Эпизоды", "Pipeline evidence": "Свидетельства конвейера", "Unadmitted intents": "Непринятые намерения", "Regressions": "Регрессии",
      "Live event lanes": "Потоки живых событий", "Presentation-only trace projection. Reconnect resynchronizes from the current evolution snapshot.": "Проекция трассы только для отображения. При переподключении выполняется синхронизация из текущего снимка эволюции.",
      "Counterfactual / mutation matrix": "Контрфактическая матрица / мутации", "Latest durable episode projection; rows do not imply an unrecorded install or approval.": "Проекция последнего долговечного эпизода; строки не означают незафиксированную установку или согласование.",
      "Trigger": "Триггер", "Capability": "Возможность", "Decision": "Решение", "Mutation": "Изменение", "Evidence tier": "Уровень доказательств", "Gap closure": "Закрытие пробела",
      "Environment lane": "Поток окружения", "Decision lane": "Поток решений", "Governance lane": "Поток управления", "Live events": "Живые события", "Event count": "Число событий", "No live events": "Нет живых событий", "Event detail": "Детали события", "Select a live event to inspect its evidence references.": "Выберите живое событие, чтобы просмотреть ссылки на его доказательства.",
      "Live since snapshot": "Живых с момента снимка",
      "{count} event(s) arrived after this snapshot. Snapshot metrics refresh on the next monitor cycle.": "{count} событий поступило после этого снимка. Метрики снимка обновятся в следующем цикле мониторинга.",
      "Environment events appear when a probe records a change in the surroundings.": "События окружения появляются, когда зонд фиксирует изменение обстановки.",
      "Decision events appear when a recorded observation drives a capability choice.": "События решений появляются, когда записанное наблюдение приводит к выбору возможности.",
      "Governance events appear when trust, quarantine or reclamation moves.": "События управления появляются при изменении доверия, карантина или высвобождения."
    }
  };
  const I18N_TEMPLATES = {
    // Every literal a board template renders, per locale. Held apart from I18N and
    // I18N_PATCH because those two grew with the first two lenses and were never
    // extended: five of seven templates shipped untranslated in every language, and
    // the i18n test only checked signal keys, so nothing failed. Keyed by the English
    // source string, so an untranslated key still renders readable English.
    zh: {"> **No effect verdict has been recorded yet**, so there is no reward signal to measure. The rates above are blank rather than zero on purpose. An effect can only be verified when the requirement declared one, and only world-model-authored requirements carry an expected effect today.": "> **尚未记录任何效果判定**，因此没有可度量的奖励信号。上面的比率有意留空而非显示 0%。只有当需求声明了预期效果才可能验证，而目前只有世界模型撰写的需求带有预期效果。", "> **Regression: a closed gap has recurred.** An evolution that looked successful did not hold. This is the one finding on this board that warrants immediate attention.": "> **回归：已闭合的缺口再次复发。** 一次看起来成功的演进并未站住。这是本看板上唯一需要立即处理的发现。", "> **Snapshot only.** There is no causal history to rebuild yet, so the timeline is absent rather than empty. Why is stated by the `Policy decisions` row under pipeline reachability; the live snapshot and the reachability table itself are unaffected.": "> **仅快照。** 目前尚无可重建的因果历史，因此时间线是「缺席」而非「空白」。原因由管道贯通度中的 `策略决策` 一行说明；实时快照与贯通度表本身不受影响。", "> **Some plugins are frozen by an internal defect.** A frozen plugin still reports `DRAFT`, and the trust dimension only *scores*, so it stays selectable unless it is also unregistered — check the `Selectable` column.": "> **部分插件因内部缺陷被冻结。** 冻结的插件仍报告 `DRAFT`，而信任维度只做「打分」，因此若未同时注销，它仍可被选中——请查看 `可被选中` 列。", "> **Verification tier: L2 (declared fitness).** A retired observation means a candidate *declared* it provides the capability, not that the capability was observed to work. Effect verification (L3) is not wired yet, so no closure on this board should be read as proven.": "> **验证层级：L2（声明式适配）。** 观测被退役，只意味着某个候选**声明**自己提供该能力，并不意味着该能力被观测到确实生效。效果验证（L3）尚未接线，因此本看板上的任何闭合都不应被读作「已证实」。", "> A watch has to complete one cycle before there is anything to show. If this persists, check that the scheduler is enabled and that the `framework-evolution` watch is armed and not muted.": "> 需要至少完成一个观测周期才会有内容。若持续为空，请检查调度器是否启用、`framework-evolution` watch 是否已 armed 且未静音。", "> An episode is written when an environment observation leads to a capability decision. None has been recorded, which is either a quiet system or a pipeline that stops earlier — the **Pipeline** tab names the segment where it stops, and what would unblock it.": "> 当一次环境观测导向一次能力决策时，才会写下一条剧集。目前尚无记录——这既可能是系统本就安静，也可能是管道更早就断了：**管道**页签会指出它断在哪一段，以及什么能解除阻塞。", "> Nothing reclaims these automatically. Each holds a tool name and appears in the capability list without being selectable, so the registry grows in a direction no requirement can use.": "> 目前没有任何机制自动回收它们。每一个都占着一个工具名、出现在能力列表里，却不可被选中——注册表朝着没有任何需求能用的方向增长。", "> These proposals entered no pipeline, so they appear in no decision record and no observation. Admitting them is a configuration choice.": "> 这些提议未进入任何管道，因此不会出现在任何决策记录或观测中。是否准入是一项配置选择。", "A ratio below 1.0 means the sampling loop is not keeping its declared cadence.": "比值低于 1.0 表示采样循环未能维持其声明的节奏。", "Abstained": "弃权", "Acquisition authority": "获取授权", "Acquisition lifecycle": "获取生命周期", "Action": "动作", "After": "变更后", "An unverified declaration has its writable channels demoted to read-only.": "未核验的声明，其可写通道会被降级为只读。", "Approval": "审批", "Autonomous governance": "自主治理", "Autonomy": "自主级别", "Before": "变更前", "CANDIDATE": "候选级", "Calibrated at": "校准时间", "Calibration health": "校准健康度", "Calls": "调用次数", "Calls (decisions)": "观点（决策）", "Candlestick": "K 线", "Capability": "能力", "Capability adaptation": "能力适配", "Capability observations": "能力观测", "Capability ownership": "能力归属", "Capability topology": "能力拓扑", "Change": "变化", "Channel": "通道", "Channels": "通道数", "Channels that have never been calibrated or whose calibration has expired are shown first.": "从未校准或校准已过期的通道排在最前。", "Command": "命令", "Commanded versus observed, best tracking first": "命令值与实测值对比，跟随最好者在前", "Composition": "组成", "Concerns (open questions)": "关切（待答问题）", "Confidence": "置信度", "Counted across every charted channel. 'near' means within 5% of a declared bound.": "统计所有绘制通道。“接近”指处于声明边界的 5% 以内。", "Cycles run": "已运行周期", "DRAFT": "草稿级", "Days since": "距今天数", "Decision": "决策", "Decisions read as calls; action items as the execution checklist.": "决策即观点，行动项即执行清单。", "Declared Hz": "声明频率 (Hz)", "Desk brief": "交易台简报", "Device": "设备", "Dropped samples": "丢弃的样本", "Each row names one blocked segment and the change that would unblock it.": "每一行指出一个受阻环节，以及能解除阻塞的那项变更。", "Effect verification (L3)": "效果验证（L3）", "Effects declared": "已声明效果", "Entities as references, and recommended next prompts to advance the work.": "实体作为参考，并给出推进工作的后续追问。", "Entities in play and the open risks still to resolve.": "涉及的实体，以及尚未解决的敞口风险。", "Envelope, rate, staleness and quality observations · newest first": "包络、速率、失联与质量观测 · 最新在前", "Environment": "环境", "Environment to framework": "环境 → 框架", "Environment, selected plugin tools, and orchestration order.": "环境、已选插件工具及编排顺序。", "Error rate": "错误率", "Events paced out": "被配速抑制的事件", "Ever used": "是否用过", "Evidence": "证据", "Evidence admission": "证据准入", "Evolution": "演进", "Evolution timeline": "演进时间线", "Executable": "可执行", "Execution checklist": "执行清单", "Extracted from this session's tool/file output (not model-generated).": "数据来自本次会话的工具/文件产物（非模型生成）。", "Failures": "失败次数", "Fiber": "Fiber 状态", "Fiber state changes since the previous cycle, including load retries.": "自上一周期以来的 Fiber 状态变化，含加载重试。", "Finance lens": "金融视图", "Follow-ups": "后续事项", "Framework change": "框架变更", "Framework changes as they happened, from runtime probes.": "来自运行时探针的框架变更实况。", "Framework evolution": "框架演进", "Framework size and how much of the evolution pipeline shows runtime evidence.": "框架规模，以及演进管道中有多少环节呈现运行时证据。", "From": "从", "Frozen plugins": "已冻结插件", "Gap closure": "缺口闭合", "Halt": "可急停", "How closures are verified": "闭合是如何验证的", "How much of the effect signal is usable as feedback. This decides whether a learning policy is worth building.": "效果信号中有多少可真正用作反馈。这决定了是否值得构建学习策略。", "How much of the framework it grew itself, and how much of the pipeline shows runtime evidence.": "框架中有多少是它自己长出来的，以及演进管道中有多少环节呈现运行时证据。", "How often each window sat inside, near, or outside its declared limits": "各窗口处于声明限值内、接近边界或越界的频次", "Inquiry brief": "研究简报", "Insights carded as evidence, capped for fast review.": "洞察以证据卡呈现，数量受限以便快速浏览。", "Instruments & counterparties": "标的与交易对手", "Kept": "保留", "Latest capability decision": "最新能力决策", "Lifecycle records": "生命周期记录", "Lifecycle timeline": "生命周期时间线", "Lifecycle transitions": "生命周期迁移", "Line of inquiry": "研究主线", "Live activity": "实时动态", "Location": "位置", "Loop phase": "循环阶段", "Mean of each downsample window. Declared limits are listed per channel below.": "每个降采样窗口的均值。各通道的声明限值见下方。", "Model's reasoning": "模型的推理", "Mutation": "变更", "Narrative": "叙事", "Narrative pulse": "叙事脉搏", "Needs attention": "需要关注", "Next recal due": "下次校准期限", "Next step": "下一步", "No causal history yet": "尚无因果历史", "Normalized error": "归一化误差", "Normalized error is the residual as a share of the channel's declared span.": "归一化误差是残差占该通道声明量程的比例。", "Not yet observed": "尚未观测", "Nothing has driven a framework change, so there is no episode to narrate.": "尚无任何事驱动过框架变更，因此没有可讲述的剧集。", "OHLC extracted from captured session market data.": "OHLC 提取自本次会话捕获的行情数据。", "Observation backlog, proposal state, policy decisions, and lifecycle outcomes.": "观测待办、提案状态、策略决策与生命周期结果。", "Observations": "观测数", "Observed Hz": "实测频率 (Hz)", "Observed rate against declared rate": "实测速率与声明速率对比", "One global namespace, arbitrated first-wins. The challenger is recorded, never silently dropped.": "单一全局命名空间，先注册者胜。挑战者会被记录，绝不静默丢弃。", "Open": "已连接", "Open risks": "敞口风险", "Open/high/low/close from captured tool output.": "开/高/低/收，来自捕获的工具输出。", "Origin": "来源", "Outcome": "结果", "PRODUCTION": "生产级", "Per episode: the trigger, the decision, the change, and whether the gap closed.": "逐条剧集：触发源、决策、变更，以及缺口是否闭合。", "Per-channel calibration state, freshness, and residual correction": "各通道的校准状态、时效性与残差校正", "Per-segment runtime evidence. A module existing is not evidence that anything calls it.": "逐段运行时证据。模块存在并不等于有任何代码调用它。", "Pipeline": "管道", "Pipeline evidence": "管道证据", "Pipeline reachability": "管道贯通度", "Plan": "计划", "Plan steps": "计划步骤", "Plugin": "插件", "Plugin roster and maturity": "插件名册与成熟度", "Plugins": "插件数", "Plugins by origin": "按来源分布的插件", "Plugins by maturity": "按成熟度分布的插件", "Policy": "策略", "Policy decisions": "策略决策", "Positions & actions": "持仓与操作", "Posture": "态势", "Price action": "价格行为", "Proposal": "提案", "Proposal status": "提案状态", "Proposed, not admitted": "已提议，未准入", "Pulse": "脉搏", "Quarantine feed": "隔离进料", "Ratio": "比值", "Read live from the registry and maturity ledger every cycle.": "每个周期从注册表与成熟度账本实时读取。", "Recent episodes": "近期剧集", "Reclaim candidates": "可回收候选", "Reclaimable": "可回收", "References & follow-ups": "参考与后续", "References (entities)": "参考（实体）", "Registry": "注册表", "Registry delta": "注册表变化", "Registry version": "注册表版本", "Regressions": "回归", "Rejected": "被拒", "Representative observations, capped for quick scanning.": "代表性观察，数量受限以便快速浏览。", "Requirements": "能力需求", "Research lens": "研究视图", "Residual": "残差", "Reward signal bandwidth": "奖励信号带宽", "Runtime evidence": "运行时证据", "Sampled history per channel, newest on the right": "按通道的采样历史，最新在右侧", "Segment": "管道段", "Segments by status": "按状态分布的管道段", "Selectable": "可被选中", "Selection delta": "选择变化", "Self-acquired": "自获取", "Self-acquired plugins that are registered but unselectable or never once used.": "已注册但不可被选中、或从未被使用过的自获取插件。", "Sentiment lens": "情绪视图", "Series": "序列", "Session analysis": "会话分析", "Signal strength": "信号强度", "Signals that something grew wrong, or was withheld. Shown regardless of the open tab.": "表明某处长错了、或被扣下未放行的信号。无论打开哪个页签都会显示。", "Skipped slots": "跳过的采样点", "State": "状态", "Storyline and signal strength before drilling into positions and actions.": "先看叙事与信号强度，再深入持仓与操作。", "Streaming": "采样中", "Suggested next steps": "建议的下一步", "The line of investigation and where the open questions concentrate.": "研究主线，以及待答问题的集中之处。", "The narrative arc and how strongly themes are trending.": "叙事走向，以及主题的趋势强度。", "The world model asked for these capabilities and nothing took them up.": "世界模型请求了这些能力，但无人受理。", "Theme intensity": "主题强度", "Themes": "主题", "This board reports how the framework changes itself. Nothing has been recorded yet.": "本看板报告框架如何改变自身。目前尚无任何记录。", "To": "到", "Tool": "工具", "Tool-name conflicts": "工具名冲突", "Tools": "工具数", "Transport": "传输方式", "Transport, provenance and channel counts": "传输方式、来源与通道数量", "Maturity": "成熟度", "Maturity accrual": "成熟度累积", "Maturity class": "成熟度", "Unselectable reclamation": "不可选回收", "Usable": "可用", "VERIFIED": "已验证级", "Verdicts": "判定数", "Verdicts by reason": "按原因分布的判定", "Verified": "已核验", "Verified by": "验证依据", "Voices & concerns": "声音与关切", "Watchlist": "关注列表", "What changed in the environment, and what the framework did about it.": "环境发生了什么变化，框架又为此做了什么。", "Which plugin owns which tool, and which capability that tool provides.": "哪个插件拥有哪个工具，以及该工具提供什么能力。", "Who/what is in the conversation, and the concerns still open.": "谁/什么在被讨论，以及尚未解决的关切。", "Why": "原因", "Why not admitted": "未准入原因", "Why this page is empty": "这个页面为何是空的", "World-model driver": "世界模型驱动器", "Writable": "可写", "aborted": "已中断", "accruing": "正在累积", "active": "运行中", "appeared": "新出现", "armed": "已就绪", "assess_compatibility": "评估兼容性", "built_in": "内置", "capability_expand": "扩展能力", "committed": "已定论", "conformance": "合规", "declared_fitness": "声明式适配", "disable": "停用", "disposed": "已释放", "effect_observed": "效果已观测", "environment_probe": "环境探测", "execution_failed": "执行失败", "expected_effect_absent": "预期效果未出现", "failed": "已失败", "frozen": "已冻结", "gone": "已消失", "idle": "空闲无变化", "install": "安装", "loading": "加载中", "manual": "人工", "moved": "已迁移", "new_unproven": "新，未验证", "no": "否", "no_evidence": "无证据", "no_expected_effect_declared": "未声明预期效果", "no_outcome_observed": "未观测到结果", "none": "无", "not_admitted": "未准入", "not_applicable": "不适用", "observe_only": "仅观察", "observed_effect": "观测效果", "open": "进行中", "pending": "待启", "reload": "重载", "remove": "移除", "reopened": "已复发", "resolved": "已闭合", "rollback": "回滚", "runtime": "运行时", "self_acquired": "自获取", "still_open": "仍未闭合", "tool_reported_no_effect": "工具未报告效果", "trusted": "已信任", "unknown": "未知", "unknown_tool": "未知工具", "unloading": "卸载中", "unscheduled": "未调度", "unverifiable": "无法核实", "unverified": "未验证", "waiting": "等待首个周期", "watching": "监视中", "wired": "已贯通", "world_model": "世界模型", "yes": "是", "Causal trace": "因果追踪", "Read-only environment-to-governance evidence for one framework-evolution snapshot.": "单个框架演进快照的只读环境到治理证据。", "Evidence boundary": "证据边界", "A causal trace shows recorded facts; it does not infer missing approval or observed effect.": "因果追踪展示已记录的事实；不推断缺失的审批或已观测效果。", "Episodes": "剧集", "Declared-fitness closures": "声明式适配闭合", "Counterfactual / mutation matrix": "反事实 / 变更矩阵", "Each row preserves the driver, decision, registry delta, and verification tier.": "每一行保留驱动因素、决策、注册表变化和验证层级。", "Trigger": "触发源", "Evidence tier": "证据层级", "Episode timeline": "剧集时间线", "Rebuilt from existing decision and observation records.": "从现有决策和观测记录重建。", "Durable trace feed": "持久追踪流", "Rejected and no-op decisions remain visible when their trace sink is installed.": "安装追踪接收器后，被拒绝和无操作决策仍保持可见。", "Lifecycle": "生命周期", "Pipeline evidence over time": "管道证据随时间变化", "One point per recorded change in framework state, oldest first.": "每一个点对应一次已记录的框架状态变化，最旧在左。", "Samples": "样本数", "Net change": "净变化"},
    fr: {"> **No effect verdict has been recorded yet**, so there is no reward signal to measure. The rates above are blank rather than zero on purpose. An effect can only be verified when the requirement declared one, and only world-model-authored requirements carry an expected effect today.": "> **Aucun verdict d'effet n'a encore été enregistré**, il n'y a donc aucun signal de récompense à mesurer. Les taux ci-dessus sont volontairement vides plutôt que nuls. Un effet ne peut être vérifié que si l'exigence en a déclaré un, et seules les exigences rédigées par le modèle du monde en portent aujourd'hui.", "> **Regression: a closed gap has recurred.** An evolution that looked successful did not hold. This is the one finding on this board that warrants immediate attention.": "> **Régression : un écart comblé s'est reproduit.** Une évolution qui semblait réussie n'a pas tenu. C'est le seul constat de ce tableau qui exige une attention immédiate.", "> **Snapshot only.** There is no causal history to rebuild yet, so the timeline is absent rather than empty. Why is stated by the `Policy decisions` row under pipeline reachability; the live snapshot and the reachability table itself are unaffected.": "> **Instantané seulement.** Aucun historique causal à reconstruire pour l'instant : la chronologie est absente, non vide. La raison est indiquée par la ligne `Décisions de politique` sous la couverture du pipeline ; l'instantané et le tableau de couverture ne sont pas affectés.", "> **Some plugins are frozen by an internal defect.** A frozen plugin still reports `DRAFT`, and the trust dimension only *scores*, so it stays selectable unless it is also unregistered — check the `Selectable` column.": "> **Certains plugins sont gelés par un défaut interne.** Un plugin gelé signale toujours `DRAFT`, et la dimension de confiance ne fait que *noter*, donc il reste sélectionnable tant qu'il n'est pas également désenregistré — voir la colonne `Sélectionnable`.", "> **Verification tier: L2 (declared fitness).** A retired observation means a candidate *declared* it provides the capability, not that the capability was observed to work. Effect verification (L3) is not wired yet, so no closure on this board should be read as proven.": "> **Niveau de vérification : L2 (aptitude déclarée).** Une observation retirée signifie qu'un candidat a *déclaré* fournir la capacité, non que la capacité a été observée en fonctionnement. La vérification d'effet (L3) n'est pas câblée, donc aucune clôture de ce tableau ne doit être lue comme prouvée.", "> A watch has to complete one cycle before there is anything to show. If this persists, check that the scheduler is enabled and that the `framework-evolution` watch is armed and not muted.": "> Un cycle d'observation doit s'achever avant qu'il y ait quoi que ce soit à montrer. Si cela persiste, vérifiez que le planificateur est actif et que la surveillance `framework-evolution` est armée et non silencée.", "> An episode is written when an environment observation leads to a capability decision. None has been recorded, which is either a quiet system or a pipeline that stops earlier — the **Pipeline** tab names the segment where it stops, and what would unblock it.": "> Un épisode est écrit lorsqu'une observation de l'environnement conduit à une décision de capacité. Aucun n'a été enregistré : soit le système est calme, soit le pipeline s'arrête plus tôt — l'onglet **Pipeline** nomme le segment où il s'arrête et ce qui le débloquerait.", "> Nothing reclaims these automatically. Each holds a tool name and appears in the capability list without being selectable, so the registry grows in a direction no requirement can use.": "> Rien ne les récupère automatiquement. Chacun occupe un nom d'outil et figure dans la liste des capacités sans être sélectionnable : le registre grandit dans une direction qu'aucune exigence ne peut utiliser.", "> These proposals entered no pipeline, so they appear in no decision record and no observation. Admitting them is a configuration choice.": "> Ces propositions n'ont intégré aucun pipeline : elles n'apparaissent donc dans aucun enregistrement de décision ni observation. Les admettre est un choix de configuration.", "A ratio below 1.0 means the sampling loop is not keeping its declared cadence.": "Un ratio inférieur à 1,0 signifie que la boucle d’échantillonnage ne tient pas sa cadence déclarée.", "Abstained": "Abstention", "Acquisition authority": "Autorité d'acquisition", "Acquisition lifecycle": "Cycle de vie d'acquisition", "Action": "Action", "After": "Après", "An unverified declaration has its writable channels demoted to read-only.": "Une déclaration non vérifiée voit ses canaux inscriptibles rétrogradés en lecture seule.", "Approval": "Approbation", "Autonomous governance": "Gouvernance autonome", "Autonomy": "Autonomie", "Before": "Avant", "CANDIDATE": "Candidat", "Calibrated at": "Calibré le", "Calibration health": "État de calibration", "Calls": "Appels", "Calls (decisions)": "Recommandations (décisions)", "Candlestick": "Chandeliers", "Capability": "Capacité", "Capability adaptation": "Adaptation des capacités", "Capability observations": "Observations de capacités", "Capability ownership": "Propriété des capacités", "Capability topology": "Topologie des capacités", "Change": "Changement", "Channel": "Canal", "Channels": "Canaux", "Channels that have never been calibrated or whose calibration has expired are shown first.": "Les canaux jamais calibrés ou dont la calibration a expiré apparaissent en premier.", "Command": "Commande", "Commanded versus observed, best tracking first": "Commandé contre observé, meilleur suivi d’abord", "Composition": "Composition", "Concerns (open questions)": "Préoccupations (questions ouvertes)", "Confidence": "Confiance", "Counted across every charted channel. 'near' means within 5% of a declared bound.": "Compté sur tous les canaux tracés. « près » signifie à moins de 5 % d’une borne déclarée.", "Cycles run": "Cycles exécutés", "DRAFT": "Brouillon", "Days since": "Jours écoulés", "Decision": "Décision", "Decisions read as calls; action items as the execution checklist.": "Les décisions se lisent comme des recommandations ; les actions comme la liste d’exécution.", "Declared Hz": "Hz déclarés", "Desk brief": "Note de desk", "Device": "Appareil", "Dropped samples": "Échantillons perdus", "Each row names one blocked segment and the change that would unblock it.": "Chaque ligne nomme un segment bloqué et le changement qui le débloquerait.", "Effect verification (L3)": "Vérification d'effet (L3)", "Effects declared": "Effets déclarés", "Entities as references, and recommended next prompts to advance the work.": "Entités comme références, et invites suivantes recommandées pour avancer.", "Entities in play and the open risks still to resolve.": "Entités concernées et risques ouverts à résoudre.", "Envelope, rate, staleness and quality observations · newest first": "Observations d’enveloppe, de débit, d’obsolescence et de qualité · les plus récentes d’abord", "Environment": "Environnement", "Environment to framework": "De l'environnement au framework", "Environment, selected plugin tools, and orchestration order.": "Environnement, outils de plugin sélectionnés et ordre d’orchestration.", "Error rate": "Taux d'erreur", "Events paced out": "Événements limités", "Ever used": "Déjà utilisé", "Evidence": "Preuve", "Evidence admission": "Admission des preuves", "Evolution": "Évolution", "Evolution timeline": "Chronologie de l'évolution", "Executable": "Exécutable", "Execution checklist": "Liste d’exécution", "Extracted from this session's tool/file output (not model-generated).": "Extrait des sorties d’outils/fichiers de cette session (non généré par le modèle).", "Failures": "Échecs", "Fiber": "Fibre", "Fiber state changes since the previous cycle, including load retries.": "Changements d'état de fiber depuis le cycle précédent, y compris les tentatives de chargement.", "Finance lens": "Vue finance", "Follow-ups": "Suivis", "Framework change": "Changement du framework", "Framework changes as they happened, from runtime probes.": "Changements du framework en temps réel, via les sondes d'exécution.", "Framework evolution": "Évolution du framework", "Framework size and how much of the evolution pipeline shows runtime evidence.": "Taille du framework et part du pipeline d'évolution qui présente des preuves d'exécution.", "From": "De", "Frozen plugins": "Plugins gelés", "Gap closure": "Clôture de l'écart", "Halt": "Arrêt", "How closures are verified": "Comment les clôtures sont vérifiées", "How much of the effect signal is usable as feedback. This decides whether a learning policy is worth building.": "Quelle part du signal d'effet est exploitable comme rétroaction. C'est ce qui détermine s'il vaut la peine de construire une politique d'apprentissage.", "How much of the framework it grew itself, and how much of the pipeline shows runtime evidence.": "Quelle part du framework il a fait croître lui-même, et quelle part du pipeline présente des preuves d'exécution.", "How often each window sat inside, near, or outside its declared limits": "Fréquence à laquelle chaque fenêtre était dans, près de, ou hors de ses limites déclarées", "Inquiry brief": "Note d’enquête", "Insights carded as evidence, capped for fast review.": "Analyses présentées comme preuves, limitées pour une revue rapide.", "Instruments & counterparties": "Instruments et contreparties", "Kept": "Conservé", "Latest capability decision": "Dernière décision de capacité", "Lifecycle records": "Enregistrements de cycle de vie", "Lifecycle timeline": "Chronologie du cycle de vie", "Lifecycle transitions": "Transitions de cycle de vie", "Line of inquiry": "Ligne d’enquête", "Live activity": "Activité en direct", "Location": "Emplacement", "Loop phase": "Phase de boucle", "Mean of each downsample window. Declared limits are listed per channel below.": "Moyenne de chaque fenêtre de sous-échantillonnage. Les limites déclarées figurent par canal ci-dessous.", "Model's reasoning": "Raisonnement du modèle", "Mutation": "Mutation", "Narrative": "Récit", "Narrative pulse": "Pouls narratif", "Needs attention": "Requiert attention", "Next recal due": "Prochaine recalibration", "Next step": "Étape suivante", "No causal history yet": "Pas encore d'historique causal", "Normalized error": "Erreur normalisée", "Normalized error is the residual as a share of the channel's declared span.": "L’erreur normalisée est le résidu en proportion de l’étendue déclarée du canal.", "Not yet observed": "Pas encore observé", "Nothing has driven a framework change, so there is no episode to narrate.": "Rien n'a encore déclenché de changement du framework : il n'y a donc aucun épisode à raconter.", "OHLC extracted from captured session market data.": "OHLC extrait des données de marché capturées durant la session.", "Observation backlog, proposal state, policy decisions, and lifecycle outcomes.": "File d’observations, état des propositions, décisions de politique et résultats du cycle de vie.", "Observations": "Observations", "Observed Hz": "Hz observés", "Observed rate against declared rate": "Débit observé par rapport au débit déclaré", "One global namespace, arbitrated first-wins. The challenger is recorded, never silently dropped.": "Un espace de noms global unique, arbitré au premier arrivé. Le concurrent est enregistré, jamais supprimé en silence.", "Open": "Ouvert", "Open risks": "Risques ouverts", "Open/high/low/close from captured tool output.": "Ouverture/haut/bas/clôture issus des sorties d’outils capturées.", "Origin": "Origine", "Outcome": "Résultat", "PRODUCTION": "Production", "Per episode: the trigger, the decision, the change, and whether the gap closed.": "Par épisode : le déclencheur, la décision, le changement, et si l'écart a été comblé.", "Per-channel calibration state, freshness, and residual correction": "État de calibration, fraîcheur et correction résiduelle par canal", "Per-segment runtime evidence. A module existing is not evidence that anything calls it.": "Preuves d'exécution par segment. L'existence d'un module ne prouve pas qu'il soit appelé.", "Pipeline": "Pipeline", "Pipeline evidence": "Preuves du pipeline", "Pipeline reachability": "Accessibilité du pipeline", "Plan": "Plan", "Plan steps": "Étapes du plan", "Plugin": "Plugin", "Plugin roster and maturity": "Registre des plugins et maturité", "Plugins": "Plugins", "Plugins by origin": "Plugins par origine", "Plugins by maturity": "Plugins par maturité", "Policy": "Politique", "Policy decisions": "Décisions de politique", "Positions & actions": "Positions et actions", "Posture": "Posture", "Price action": "Action des prix", "Proposal": "Proposition", "Proposal status": "Statut de la proposition", "Proposed, not admitted": "Proposé, non admis", "Pulse": "Pouls", "Quarantine feed": "Flux de quarantaine", "Ratio": "Ratio", "Read live from the registry and maturity ledger every cycle.": "Lu en direct depuis le registre et le registre de maturité à chaque cycle.", "Recent episodes": "Épisodes récents", "Reclaim candidates": "Candidats à la récupération", "Reclaimable": "Récupérable", "References & follow-ups": "Références et suivis", "References (entities)": "Références (entités)", "Registry": "Registre", "Registry delta": "Delta du registre", "Registry version": "Version du registre", "Regressions": "Régressions", "Rejected": "Rejeté", "Representative observations, capped for quick scanning.": "Observations représentatives, limitées pour une lecture rapide.", "Requirements": "Exigences", "Research lens": "Vue recherche", "Residual": "Résidu", "Reward signal bandwidth": "Bande passante du signal de récompense", "Runtime evidence": "Preuve d'exécution", "Sampled history per channel, newest on the right": "Historique échantillonné par canal, le plus récent à droite", "Segment": "Segment", "Segments by status": "Segments par statut", "Selectable": "Sélectionnable", "Selection delta": "Delta de sélection", "Self-acquired": "Auto-acquis", "Self-acquired plugins that are registered but unselectable or never once used.": "Plugins auto-acquis qui sont enregistrés mais non sélectionnables, ou jamais utilisés une seule fois.", "Sentiment lens": "Vue sentiment", "Series": "Série", "Session analysis": "Analyse de session", "Signal strength": "Force du signal", "Signals that something grew wrong, or was withheld. Shown regardless of the open tab.": "Signaux indiquant qu'une évolution a mal tourné ou a été retenue. Affichés quel que soit l'onglet ouvert.", "Skipped slots": "Créneaux manqués", "State": "État", "Storyline and signal strength before drilling into positions and actions.": "Récit et force du signal avant d’examiner positions et actions.", "Streaming": "Diffusion", "Suggested next steps": "Prochaines étapes suggérées", "The line of investigation and where the open questions concentrate.": "La ligne d’investigation et où se concentrent les questions ouvertes.", "The narrative arc and how strongly themes are trending.": "L’arc narratif et l’intensité des tendances thématiques.", "The world model asked for these capabilities and nothing took them up.": "Le modèle du monde a demandé ces capacités et personne ne les a prises en charge.", "Theme intensity": "Intensité des thèmes", "Themes": "Thèmes", "This board reports how the framework changes itself. Nothing has been recorded yet.": "Ce tableau rend compte de la façon dont le framework se modifie lui-même. Rien n'a encore été enregistré.", "To": "Vers", "Tool": "Outil", "Tool-name conflicts": "Conflits de noms d'outils", "Tools": "Outils", "Transport": "Transport", "Transport, provenance and channel counts": "Transport, provenance et nombre de canaux", "Maturity": "Maturité", "Maturity accrual": "Accumulation de maturité", "Maturity class": "Classe de maturité", "Unselectable reclamation": "Récupération non sélectionnable", "Usable": "Exploitable", "VERIFIED": "Vérifié", "Verdicts": "Verdicts", "Verdicts by reason": "Verdicts par motif", "Verified": "Vérifié", "Verified by": "Vérifié par", "Voices & concerns": "Voix et préoccupations", "Watchlist": "Liste de suivi", "What changed in the environment, and what the framework did about it.": "Ce qui a changé dans l'environnement, et ce que le framework a fait en réponse.", "Which plugin owns which tool, and which capability that tool provides.": "Quel plugin possède quel outil, et quelle capacité cet outil fournit.", "Who/what is in the conversation, and the concerns still open.": "Qui/quoi est dans la conversation, et les préoccupations encore ouvertes.", "Why": "Pourquoi", "Why not admitted": "Motif de non-admission", "Why this page is empty": "Pourquoi cette page est vide", "World-model driver": "Pilote du modèle du monde", "Writable": "Inscriptible", "aborted": "Abandonné", "accruing": "En accumulation", "active": "Actif", "appeared": "Apparu", "armed": "Armé", "assess_compatibility": "Évaluer la compatibilité", "built_in": "Intégré", "capability_expand": "Étendre les capacités", "committed": "Conclu", "conformance": "Conformité", "declared_fitness": "Aptitude déclarée", "disable": "Désactiver", "disposed": "Libéré", "effect_observed": "Effet observé", "environment_probe": "Sonde d'environnement", "execution_failed": "Échec d'exécution", "expected_effect_absent": "Effet attendu absent", "failed": "Échoué", "frozen": "Gelé", "gone": "Disparu", "idle": "Au repos", "install": "Installer", "loading": "Chargement", "manual": "Manuel", "moved": "Déplacé", "new_unproven": "Nouveau, non éprouvé", "no": "Non", "no_evidence": "Aucune preuve", "no_expected_effect_declared": "Aucun effet attendu déclaré", "no_outcome_observed": "Aucun résultat observé", "none": "Aucun", "not_admitted": "Non admis", "not_applicable": "Sans objet", "observe_only": "Observer seulement", "observed_effect": "Effet observé", "open": "Ouvert", "pending": "En attente", "reload": "Recharger", "remove": "Supprimer", "reopened": "Réouvert", "resolved": "Résolu", "rollback": "Annuler", "runtime": "Exécution", "self_acquired": "Auto-acquis", "still_open": "Toujours ouvert", "tool_reported_no_effect": "L'outil n'a signalé aucun effet", "trusted": "De confiance", "unknown": "Inconnu", "unknown_tool": "Outil inconnu", "unloading": "Déchargement", "unscheduled": "Non planifié", "unverifiable": "Invérifiable", "unverified": "Non vérifié", "waiting": "En attente", "watching": "En surveillance", "wired": "Câblé", "world_model": "Modèle du monde", "yes": "Oui", "Causal trace": "Trace causale", "Read-only environment-to-governance evidence for one framework-evolution snapshot.": "Preuves en lecture seule reliant l’environnement à la gouvernance pour un instantané d’évolution du framework.", "Evidence boundary": "Limite des preuves", "A causal trace shows recorded facts; it does not infer missing approval or observed effect.": "Une trace causale montre les faits enregistrés ; elle n’infère ni approbation manquante ni effet observé.", "Episodes": "Épisodes", "Declared-fitness closures": "Clôtures par aptitude déclarée", "Counterfactual / mutation matrix": "Matrice contrefactuelle / mutations", "Each row preserves the driver, decision, registry delta, and verification tier.": "Chaque ligne conserve le déclencheur, la décision, le delta du registre et le niveau de vérification.", "Trigger": "Déclencheur", "Evidence tier": "Niveau de preuve", "Episode timeline": "Chronologie des épisodes", "Rebuilt from existing decision and observation records.": "Reconstruite à partir des enregistrements existants de décision et d’observation.", "Durable trace feed": "Flux de traces durables", "Rejected and no-op decisions remain visible when their trace sink is installed.": "Les décisions rejetées et sans action restent visibles lorsque leur récepteur de traces est installé.", "Lifecycle": "Cycle de vie", "Pipeline evidence over time": "Évolution des preuves du convéoyeur", "One point per recorded change in framework state, oldest first.": "Un point par changement enregistré de l’état du framework, du plus ancien au plus récent.", "Samples": "Échantillons", "Net change": "Variation nette"},
    es: {"> **No effect verdict has been recorded yet**, so there is no reward signal to measure. The rates above are blank rather than zero on purpose. An effect can only be verified when the requirement declared one, and only world-model-authored requirements carry an expected effect today.": "> **Aún no se ha registrado ningún veredicto de efecto**, por lo que no hay señal de recompensa que medir. Las tasas anteriores están en blanco a propósito, no en cero. Un efecto solo puede verificarse si el requisito declaró uno, y hoy solo los requisitos redactados por el modelo del mundo lo llevan.", "> **Regression: a closed gap has recurred.** An evolution that looked successful did not hold. This is the one finding on this board that warrants immediate attention.": "> **Regresión: una brecha cerrada ha vuelto a aparecer.** Una evolución que parecía exitosa no se sostuvo. Es el único hallazgo de este panel que exige atención inmediata.", "> **Snapshot only.** There is no causal history to rebuild yet, so the timeline is absent rather than empty. Why is stated by the `Policy decisions` row under pipeline reachability; the live snapshot and the reachability table itself are unaffected.": "> **Solo instantánea.** Todavía no hay historia causal que reconstruir, por lo que la cronología está ausente, no vacía. El motivo lo indica la fila `Decisiones de política` bajo la cobertura del pipeline; la instantánea y la tabla de cobertura no se ven afectadas.", "> **Some plugins are frozen by an internal defect.** A frozen plugin still reports `DRAFT`, and the trust dimension only *scores*, so it stays selectable unless it is also unregistered — check the `Selectable` column.": "> **Algunos plugins están congelados por un defecto interno.** Un plugin congelado sigue informando `DRAFT`, y la dimensión de confianza solo *puntúa*, por lo que permanece seleccionable a menos que también se desregistre — consulte la columna `Seleccionable`.", "> **Verification tier: L2 (declared fitness).** A retired observation means a candidate *declared* it provides the capability, not that the capability was observed to work. Effect verification (L3) is not wired yet, so no closure on this board should be read as proven.": "> **Nivel de verificación: L2 (aptitud declarada).** Una observación retirada significa que un candidato *declaró* que proporciona la capacidad, no que se observara funcionando. La verificación de efecto (L3) no está conectada, así que ningún cierre de este panel debe leerse como probado.", "> A watch has to complete one cycle before there is anything to show. If this persists, check that the scheduler is enabled and that the `framework-evolution` watch is armed and not muted.": "> Debe completarse un ciclo de observación antes de que haya algo que mostrar. Si persiste, compruebe que el planificador está activo y que la vigilancia `framework-evolution` está armada y no silenciada.", "> An episode is written when an environment observation leads to a capability decision. None has been recorded, which is either a quiet system or a pipeline that stops earlier — the **Pipeline** tab names the segment where it stops, and what would unblock it.": "> Un episodio se escribe cuando una observación del entorno conduce a una decisión de capacidad. No se ha registrado ninguno: o el sistema está tranquilo o el pipeline se detiene antes — la pestaña **Pipeline** nombra el segmento donde se detiene y qué lo desbloquearía.", "> Nothing reclaims these automatically. Each holds a tool name and appears in the capability list without being selectable, so the registry grows in a direction no requirement can use.": "> Nada los recupera automáticamente. Cada uno ocupa un nombre de herramienta y aparece en la lista de capacidades sin ser seleccionable: el registro crece en una dirección que ningún requisito puede usar.", "> These proposals entered no pipeline, so they appear in no decision record and no observation. Admitting them is a configuration choice.": "> Estas propuestas no entraron en ningún pipeline, por lo que no aparecen en ningún registro de decisión ni observación. Admitirlas es una elección de configuración.", "A ratio below 1.0 means the sampling loop is not keeping its declared cadence.": "Una relación inferior a 1,0 significa que el bucle de muestreo no mantiene su cadencia declarada.", "Abstained": "Abstenido", "Acquisition authority": "Autoridad de adquisición", "Acquisition lifecycle": "Ciclo de vida de adquisición", "Action": "Acción", "After": "Después", "An unverified declaration has its writable channels demoted to read-only.": "Una declaración no verificada degrada sus canales escribibles a solo lectura.", "Approval": "Aprobación", "Autonomous governance": "Gobernanza autónoma", "Autonomy": "Autonomía", "Before": "Antes", "CANDIDATE": "Candidato", "Calibrated at": "Calibrado el", "Calibration health": "Estado de calibración", "Calls": "Llamadas", "Calls (decisions)": "Recomendaciones (decisiones)", "Candlestick": "Velas", "Capability": "Capacidad", "Capability adaptation": "Adaptación de capacidades", "Capability observations": "Observaciones de capacidad", "Capability ownership": "Propiedad de capacidades", "Capability topology": "Topología de capacidades", "Change": "Cambio", "Channel": "Canal", "Channels": "Canales", "Channels that have never been calibrated or whose calibration has expired are shown first.": "Los canales nunca calibrados o con calibración vencida se muestran primero.", "Command": "Comando", "Commanded versus observed, best tracking first": "Comandado frente a observado, mejor seguimiento primero", "Composition": "Composición", "Concerns (open questions)": "Inquietudes (preguntas abiertas)", "Confidence": "Confianza", "Counted across every charted channel. 'near' means within 5% of a declared bound.": "Contado en todos los canales graficados. «cerca» significa dentro del 5 % de un límite declarado.", "Cycles run": "Ciclos ejecutados", "DRAFT": "Borrador", "Days since": "Días desde", "Decision": "Decisión", "Decisions read as calls; action items as the execution checklist.": "Las decisiones se leen como recomendaciones; las acciones como la lista de ejecución.", "Declared Hz": "Hz declarados", "Desk brief": "Informe de mesa", "Device": "Dispositivo", "Dropped samples": "Muestras descartadas", "Each row names one blocked segment and the change that would unblock it.": "Cada fila nombra un segmento bloqueado y el cambio que lo desbloquearía.", "Effect verification (L3)": "Verificación de efecto (L3)", "Effects declared": "Efectos declarados", "Entities as references, and recommended next prompts to advance the work.": "Entidades como referencias y siguientes preguntas recomendadas para avanzar.", "Entities in play and the open risks still to resolve.": "Entidades implicadas y riesgos abiertos por resolver.", "Envelope, rate, staleness and quality observations · newest first": "Observaciones de envolvente, tasa, obsolescencia y calidad · las más recientes primero", "Environment": "Entorno", "Environment to framework": "Del entorno al framework", "Environment, selected plugin tools, and orchestration order.": "Entorno, herramientas de plugin seleccionadas y orden de orquestación.", "Error rate": "Tasa de error", "Events paced out": "Eventos limitados", "Ever used": "Alguna vez usado", "Evidence": "Evidencia", "Evidence admission": "Admisión de evidencia", "Evolution": "Evolución", "Evolution timeline": "Cronología de la evolución", "Executable": "Ejecutable", "Execution checklist": "Lista de ejecución", "Extracted from this session's tool/file output (not model-generated).": "Extraído de la salida de herramientas/archivos de esta sesión (no generado por el modelo).", "Failures": "Fallos", "Fiber": "Fibra", "Fiber state changes since the previous cycle, including load retries.": "Cambios de estado de fiber desde el ciclo anterior, incluidos los reintentos de carga.", "Finance lens": "Vista financiera", "Follow-ups": "Seguimientos", "Framework change": "Cambio del framework", "Framework changes as they happened, from runtime probes.": "Cambios del framework en tiempo real, desde sondas de ejecución.", "Framework evolution": "Evolución del framework", "Framework size and how much of the evolution pipeline shows runtime evidence.": "Tamaño del framework y qué parte del pipeline de evolución muestra evidencia en ejecución.", "From": "Desde", "Frozen plugins": "Plugins congelados", "Gap closure": "Cierre de la brecha", "Halt": "Parada", "How closures are verified": "Cómo se verifican los cierres", "How much of the effect signal is usable as feedback. This decides whether a learning policy is worth building.": "Cuánto de la señal de efecto es utilizable como retroalimentación. Esto decide si vale la pena construir una política de aprendizaje.", "How much of the framework it grew itself, and how much of the pipeline shows runtime evidence.": "Cuánto del framework hizo crecer por sí mismo y cuánto del pipeline muestra evidencia de ejecución.", "How often each window sat inside, near, or outside its declared limits": "Con qué frecuencia cada ventana estuvo dentro, cerca o fuera de sus límites declarados", "Inquiry brief": "Informe de indagación", "Insights carded as evidence, capped for fast review.": "Hallazgos presentados como evidencia, limitados para revisión rápida.", "Instruments & counterparties": "Instrumentos y contrapartes", "Kept": "Conservado", "Latest capability decision": "Última decisión de capacidad", "Lifecycle records": "Registros de ciclo de vida", "Lifecycle timeline": "Cronología del ciclo de vida", "Lifecycle transitions": "Transiciones de ciclo de vida", "Line of inquiry": "Línea de indagación", "Live activity": "Actividad en vivo", "Location": "Ubicación", "Loop phase": "Fase del bucle", "Mean of each downsample window. Declared limits are listed per channel below.": "Media de cada ventana de submuestreo. Los límites declarados se listan por canal abajo.", "Model's reasoning": "Razonamiento del modelo", "Mutation": "Mutación", "Narrative": "Narrativa", "Narrative pulse": "Pulso narrativo", "Needs attention": "Requiere atención", "Next recal due": "Próxima recalibración", "Next step": "Siguiente paso", "No causal history yet": "Aún no hay historia causal", "Normalized error": "Error normalizado", "Normalized error is the residual as a share of the channel's declared span.": "El error normalizado es el residuo como fracción del rango declarado del canal.", "Not yet observed": "Aún no observado", "Nothing has driven a framework change, so there is no episode to narrate.": "Nada ha impulsado todavía un cambio del framework, por lo que no hay ningún episodio que narrar.", "OHLC extracted from captured session market data.": "OHLC extraído de los datos de mercado capturados en la sesión.", "Observation backlog, proposal state, policy decisions, and lifecycle outcomes.": "Cola de observaciones, estado de propuestas, decisiones de política y resultados del ciclo de vida.", "Observations": "Observaciones", "Observed Hz": "Hz observados", "Observed rate against declared rate": "Tasa observada frente a la tasa declarada", "One global namespace, arbitrated first-wins. The challenger is recorded, never silently dropped.": "Un único espacio de nombres global, arbitrado por orden de llegada. El aspirante queda registrado, nunca se descarta en silencio.", "Open": "Abierto", "Open risks": "Riesgos abiertos", "Open/high/low/close from captured tool output.": "Apertura/máximo/mínimo/cierre desde la salida de herramientas capturada.", "Origin": "Origen", "Outcome": "Resultado", "PRODUCTION": "Producción", "Per episode: the trigger, the decision, the change, and whether the gap closed.": "Por episodio: el desencadenante, la decisión, el cambio y si la brecha se cerró.", "Per-channel calibration state, freshness, and residual correction": "Estado de calibración, vigencia y corrección residual por canal", "Per-segment runtime evidence. A module existing is not evidence that anything calls it.": "Evidencia en ejecución por segmento. Que un módulo exista no prueba que algo lo invoque.", "Pipeline": "Pipeline", "Pipeline evidence": "Evidencia del pipeline", "Pipeline reachability": "Alcanzabilidad del pipeline", "Plan": "Plan", "Plan steps": "Pasos del plan", "Plugin": "Plugin", "Plugin roster and maturity": "Registro de plugins y madurez", "Plugins": "Plugins", "Plugins by origin": "Plugins por origen", "Plugins by maturity": "Plugins por madurez", "Policy": "Política", "Policy decisions": "Decisiones de política", "Positions & actions": "Posiciones y acciones", "Posture": "Postura", "Price action": "Acción del precio", "Proposal": "Propuesta", "Proposal status": "Estado de la propuesta", "Proposed, not admitted": "Propuesto, no admitido", "Pulse": "Pulso", "Quarantine feed": "Entrada de cuarentena", "Ratio": "Relación", "Read live from the registry and maturity ledger every cycle.": "Leído en vivo del registro y del libro de madurez en cada ciclo.", "Recent episodes": "Episodios recientes", "Reclaim candidates": "Candidatos a recuperación", "Reclaimable": "Recuperable", "References & follow-ups": "Referencias y seguimientos", "References (entities)": "Referencias (entidades)", "Registry": "Registro", "Registry delta": "Delta del registro", "Registry version": "Versión del registro", "Regressions": "Regresiones", "Rejected": "Rechazado", "Representative observations, capped for quick scanning.": "Observaciones representativas, limitadas para lectura rápida.", "Requirements": "Requisitos", "Research lens": "Vista de investigación", "Residual": "Residuo", "Reward signal bandwidth": "Ancho de banda de la señal de recompensa", "Runtime evidence": "Evidencia en ejecución", "Sampled history per channel, newest on the right": "Historial muestreado por canal, el más reciente a la derecha", "Segment": "Segmento", "Segments by status": "Segmentos por estado", "Selectable": "Seleccionable", "Selection delta": "Delta de selección", "Self-acquired": "Autoadquirido", "Self-acquired plugins that are registered but unselectable or never once used.": "Plugins autoadquiridos que están registrados pero no son seleccionables, o nunca se han usado.", "Sentiment lens": "Vista de sentimiento", "Series": "Serie", "Session analysis": "Análisis de sesión", "Signal strength": "Fuerza de la señal", "Signals that something grew wrong, or was withheld. Shown regardless of the open tab.": "Señales de que algo creció mal o fue retenido. Se muestran independientemente de la pestaña abierta.", "Skipped slots": "Ranuras omitidas", "State": "Estado", "Storyline and signal strength before drilling into positions and actions.": "Narrativa y fuerza de la señal antes de entrar en posiciones y acciones.", "Streaming": "Transmisión", "Suggested next steps": "Próximos pasos sugeridos", "The line of investigation and where the open questions concentrate.": "La línea de investigación y dónde se concentran las preguntas abiertas.", "The narrative arc and how strongly themes are trending.": "El arco narrativo y con qué fuerza se mueven los temas.", "The world model asked for these capabilities and nothing took them up.": "El modelo del mundo pidió estas capacidades y nada las asumió.", "Theme intensity": "Intensidad temática", "Themes": "Temas", "This board reports how the framework changes itself. Nothing has been recorded yet.": "Este panel informa de cómo el framework se modifica a sí mismo. Todavía no se ha registrado nada.", "To": "Hasta", "Tool": "Herramienta", "Tool-name conflicts": "Conflictos de nombres de herramientas", "Tools": "Herramientas", "Transport": "Transporte", "Transport, provenance and channel counts": "Transporte, procedencia y número de canales", "Maturity": "Madurez", "Maturity accrual": "Acumulación de madurez", "Maturity class": "Clase de madurez", "Unselectable reclamation": "Recuperación no seleccionable", "Usable": "Utilizable", "VERIFIED": "Verificado", "Verdicts": "Veredictos", "Verdicts by reason": "Veredictos por motivo", "Verified": "Verificado", "Verified by": "Verificado por", "Voices & concerns": "Voces e inquietudes", "Watchlist": "Lista de seguimiento", "What changed in the environment, and what the framework did about it.": "Qué cambió en el entorno y qué hizo el framework al respecto.", "Which plugin owns which tool, and which capability that tool provides.": "Qué plugin posee qué herramienta y qué capacidad proporciona esa herramienta.", "Who/what is in the conversation, and the concerns still open.": "Quién/qué está en la conversación y las inquietudes aún abiertas.", "Why": "Por qué", "Why not admitted": "Motivo de no admisión", "Why this page is empty": "Por qué esta página está vacía", "World-model driver": "Controlador del modelo del mundo", "Writable": "Escribible", "aborted": "Abortado", "accruing": "Acumulando", "active": "Activo", "appeared": "Apareció", "armed": "Armado", "assess_compatibility": "Evaluar compatibilidad", "built_in": "Integrado", "capability_expand": "Ampliar capacidad", "committed": "Concluido", "conformance": "Conformidad", "declared_fitness": "Aptitud declarada", "disable": "Desactivar", "disposed": "Liberado", "effect_observed": "Efecto observado", "environment_probe": "Sonda de entorno", "execution_failed": "Ejecución fallida", "expected_effect_absent": "Efecto esperado ausente", "failed": "Fallido", "frozen": "Congelado", "gone": "Desapareció", "idle": "Inactivo", "install": "Instalar", "loading": "Cargando", "manual": "Manual", "moved": "Se movió", "new_unproven": "Nuevo, no probado", "no": "No", "no_evidence": "Sin evidencia", "no_expected_effect_declared": "Sin efecto esperado declarado", "no_outcome_observed": "Sin resultado observado", "none": "Ninguno", "not_admitted": "No admitido", "not_applicable": "No aplicable", "observe_only": "Solo observar", "observed_effect": "Efecto observado", "open": "Abierto", "pending": "Pendiente", "reload": "Recargar", "remove": "Eliminar", "reopened": "Reabierto", "resolved": "Resuelto", "rollback": "Revertir", "runtime": "Tiempo de ejecución", "self_acquired": "Autoadquirido", "still_open": "Aún abierto", "tool_reported_no_effect": "La herramienta no informó efecto", "trusted": "De confianza", "unknown": "Desconocido", "unknown_tool": "Herramienta desconocida", "unloading": "Descargando", "unscheduled": "No planificado", "unverifiable": "No verificable", "unverified": "No verificado", "waiting": "En espera", "watching": "Vigilando", "wired": "Conectado", "world_model": "Modelo del mundo", "yes": "Sí", "Causal trace": "Traza causal", "Read-only environment-to-governance evidence for one framework-evolution snapshot.": "Evidencia de solo lectura del entorno a la gobernanza para una instantánea de evolución del framework.", "Evidence boundary": "Límite de evidencia", "A causal trace shows recorded facts; it does not infer missing approval or observed effect.": "Una traza causal muestra hechos registrados; no infiere aprobación ausente ni efecto observado.", "Episodes": "Episodios", "Declared-fitness closures": "Cierres por aptitud declarada", "Counterfactual / mutation matrix": "Matriz contrafactual / de mutaciones", "Each row preserves the driver, decision, registry delta, and verification tier.": "Cada fila conserva el desencadenante, la decisión, el delta del registro y el nivel de verificación.", "Trigger": "Desencadenante", "Evidence tier": "Nivel de evidencia", "Episode timeline": "Cronología de episodios", "Rebuilt from existing decision and observation records.": "Reconstruida a partir de registros existentes de decisión y observación.", "Durable trace feed": "Flujo de trazas durables", "Rejected and no-op decisions remain visible when their trace sink is installed.": "Las decisiones rechazadas y sin acción permanecen visibles cuando se instala su receptor de trazas.", "Lifecycle": "Ciclo de vida", "Pipeline evidence over time": "Evidencia del canal a lo largo del tiempo", "One point per recorded change in framework state, oldest first.": "Un punto por cada cambio registrado del estado del framework, del más antiguo al más reciente.", "Samples": "Muestras", "Net change": "Cambio neto"},
    ar: {"> **No effect verdict has been recorded yet**, so there is no reward signal to measure. The rates above are blank rather than zero on purpose. An effect can only be verified when the requirement declared one, and only world-model-authored requirements carry an expected effect today.": "> **لم يُسجَّل أي حكم على الأثر بعد**، لذا لا توجد إشارة مكافأة لقياسها. النسب أعلاه فارغة عن قصد وليست صفرًا. لا يمكن التحقق من الأثر إلا إذا أعلنه المطلب، واليوم لا تحمل الأثر المتوقع سوى المطالب التي كتبها نموذج العالم.", "> **Regression: a closed gap has recurred.** An evolution that looked successful did not hold. This is the one finding on this board that warrants immediate attention.": "> **انحدار: فجوة أُغلقت عادت للظهور.** تطوّر بدا ناجحًا لم يصمد. هذا هو الاكتشاف الوحيد في هذه اللوحة الذي يستدعي انتباهًا فوريًا.", "> **Snapshot only.** There is no causal history to rebuild yet, so the timeline is absent rather than empty. Why is stated by the `Policy decisions` row under pipeline reachability; the live snapshot and the reachability table itself are unaffected.": "> **لقطة فقط.** لا يوجد بعد تاريخ سببي لإعادة بنائه، لذا فالخط الزمني غائب وليس فارغًا. السبب مبيَّن في صف `قرارات السياسة` تحت تغطية المسار؛ اللقطة الحيّة وجدول التغطية غير متأثرين.", "> **Some plugins are frozen by an internal defect.** A frozen plugin still reports `DRAFT`, and the trust dimension only *scores*, so it stays selectable unless it is also unregistered — check the `Selectable` column.": "> **بعض الإضافات مُجمَّدة بسبب خلل داخلي.** الإضافة المُجمَّدة لا تزال تُبلِّغ `DRAFT`، وبُعد الثقة يقوم بالتقييم فقط، لذا تبقى قابلة للاختيار إلا إذا أُلغي تسجيلها أيضًا — راجع عمود `قابل للاختيار`.", "> **Verification tier: L2 (declared fitness).** A retired observation means a candidate *declared* it provides the capability, not that the capability was observed to work. Effect verification (L3) is not wired yet, so no closure on this board should be read as proven.": "> **مستوى التحقق: L2 (الملاءمة المُعلنة).** سحب الرصد يعني أن مرشّحًا *أعلن* أنه يوفّر القدرة، لا أن القدرة رُصدت وهي تعمل. التحقق من الأثر (L3) غير موصول، لذا لا ينبغي قراءة أي إغلاق في هذه اللوحة كأمر مُثبَت.", "> A watch has to complete one cycle before there is anything to show. If this persists, check that the scheduler is enabled and that the `framework-evolution` watch is armed and not muted.": "> يجب أن تكتمل دورة مراقبة واحدة قبل ظهور أي محتوى. إذا استمر ذلك، تحقّق من تمكين المُجدول وأن مراقبة `framework-evolution` مُسلّحة وغير مكتومة.", "> An episode is written when an environment observation leads to a capability decision. None has been recorded, which is either a quiet system or a pipeline that stops earlier — the **Pipeline** tab names the segment where it stops, and what would unblock it.": "> تُكتب الحلقة عندما يؤدي رصد للبيئة إلى قرار بشأن قدرة. لم يُسجَّل أي منها، وهذا يعني إمّا نظامًا هادئًا أو مسارًا يتوقف قبل ذلك — تبويب **المسار** يحدّد الجزء الذي يتوقف عنده وما الذي يزيل التعطيل.", "> Nothing reclaims these automatically. Each holds a tool name and appears in the capability list without being selectable, so the registry grows in a direction no requirement can use.": "> لا شيء يستعيدها تلقائيًا. كل واحدة تحتجز اسم أداة وتظهر في قائمة القدرات دون أن تكون قابلة للاختيار، فينمو السجل في اتجاه لا يمكن لأي مطلب استخدامه.", "> These proposals entered no pipeline, so they appear in no decision record and no observation. Admitting them is a configuration choice.": "> لم تدخل هذه المقترحات أي مسار، لذا لا تظهر في أي سجل قرار أو رصد. قبولها خيار في الإعدادات.", "A ratio below 1.0 means the sampling loop is not keeping its declared cadence.": "نسبة أقل من 1.0 تعني أن حلقة أخذ العينات لا تحافظ على وتيرتها المعلنة.", "Abstained": "امتناع", "Acquisition authority": "سلطة الاكتساب", "Acquisition lifecycle": "دورة حياة الاكتساب", "Action": "الإجراء", "After": "بعد", "An unverified declaration has its writable channels demoted to read-only.": "الإعلان غير المُتحقَّق منه تُخفَّض قنواته القابلة للكتابة إلى القراءة فقط.", "Approval": "الموافقة", "Autonomous governance": "الحكم الذاتي", "Autonomy": "الاستقلالية", "Before": "قبل", "CANDIDATE": "مرشّح", "Calibrated at": "تاريخ المعايرة", "Calibration health": "سلامة المعايرة", "Calls": "الاستدعاءات", "Calls (decisions)": "التوصيات (القرارات)", "Candlestick": "الشموع", "Capability": "القدرة", "Capability adaptation": "تكييف القدرات", "Capability observations": "رصد القدرات", "Capability ownership": "ملكية القدرات", "Capability topology": "طوبولوجيا القدرات", "Change": "التغيير", "Channel": "القناة", "Channels": "القنوات", "Channels that have never been calibrated or whose calibration has expired are shown first.": "تظهر أولاً القنوات التي لم تُعاير قط أو التي انتهت صلاحية معايرتها.", "Command": "الأمر", "Commanded versus observed, best tracking first": "المأمور مقابل المرصود، الأفضل تتبعاً أولاً", "Composition": "التركيب", "Concerns (open questions)": "المخاوف (أسئلة مفتوحة)", "Confidence": "الثقة", "Counted across every charted channel. 'near' means within 5% of a declared bound.": "محسوب على كل قناة مرسومة. \"قريب\" تعني داخل 5% من حد معلن.", "Cycles run": "الدورات المنفَّذة", "DRAFT": "مسوّدة", "Days since": "الأيام المنقضية", "Decision": "القرار", "Decisions read as calls; action items as the execution checklist.": "القرارات تُقرأ كتوصيات؛ والإجراءات كقائمة تنفيذ.", "Declared Hz": "الهرتز المعلن", "Desk brief": "موجز المكتب", "Device": "الجهاز", "Dropped samples": "العينات المفقودة", "Each row names one blocked segment and the change that would unblock it.": "كل صف يحدّد جزءًا معطَّلًا والتغيير الذي يزيل التعطيل.", "Effect verification (L3)": "التحقق من الأثر (L3)", "Effects declared": "الآثار المُعلنة", "Entities as references, and recommended next prompts to advance the work.": "الكيانات كمراجع، والمطالبات التالية الموصى بها لدفع العمل.", "Entities in play and the open risks still to resolve.": "الكيانات المعنية والمخاطر المفتوحة التي لم تُحل.", "Envelope, rate, staleness and quality observations · newest first": "رصدات المغلف والمعدل والتقادم والجودة · الأحدث أولاً", "Environment": "البيئة", "Environment to framework": "من البيئة إلى الإطار", "Environment, selected plugin tools, and orchestration order.": "البيئة والأدوات المختارة وترتيب التنسيق.", "Error rate": "معدل الأخطاء", "Events paced out": "الأحداث المُقيَّدة", "Ever used": "استُخدم سابقًا", "Evidence": "الدليل", "Evidence admission": "قبول الأدلة", "Evolution": "التطور", "Evolution timeline": "الخط الزمني للتطور", "Executable": "قابل للتنفيذ", "Execution checklist": "قائمة التنفيذ", "Extracted from this session's tool/file output (not model-generated).": "مستخرج من مخرجات الأدوات/الملفات في هذه الجلسة (ليس من إنشاء النموذج).", "Failures": "الأعطال", "Fiber": "الخيط", "Fiber state changes since the previous cycle, including load retries.": "تغييرات حالة الـ fiber منذ الدورة السابقة، بما في ذلك محاولات التحميل.", "Finance lens": "منظور مالي", "Follow-ups": "المتابعات", "Framework change": "تغيير الإطار", "Framework changes as they happened, from runtime probes.": "تغييرات الإطار لحظة حدوثها، من مجسّات وقت التشغيل.", "Framework evolution": "تطور الإطار", "Framework size and how much of the evolution pipeline shows runtime evidence.": "حجم الإطار ومقدار ما يُظهره مسار التطور من أدلة وقت التشغيل.", "From": "من", "Frozen plugins": "الإضافات المُجمَّدة", "Gap closure": "إغلاق الفجوة", "Halt": "إيقاف", "How closures are verified": "كيف يُتحقَّق من الإغلاقات", "How much of the effect signal is usable as feedback. This decides whether a learning policy is worth building.": "ما مقدار إشارة الأثر القابل للاستخدام كتغذية راجعة. هذا يحدّد ما إذا كان بناء سياسة تعلّم يستحق العناء.", "How much of the framework it grew itself, and how much of the pipeline shows runtime evidence.": "ما مقدار ما نمّاه الإطار بنفسه، وما مقدار المسار الذي يُظهر أدلة وقت التشغيل.", "How often each window sat inside, near, or outside its declared limits": "عدد المرات التي كانت فيها كل نافذة داخل حدودها المعلنة أو قريبة منها أو خارجها", "Inquiry brief": "موجز الاستقصاء", "Insights carded as evidence, capped for fast review.": "الرؤى معروضة كأدلة، ومحدودة العدد للمراجعة السريعة.", "Instruments & counterparties": "الأدوات والأطراف المقابلة", "Kept": "المحتفظ به", "Latest capability decision": "أحدث قرار للقدرات", "Lifecycle records": "سجلات دورة الحياة", "Lifecycle timeline": "الخط الزمني لدورة الحياة", "Lifecycle transitions": "انتقالات دورة الحياة", "Line of inquiry": "خط الاستقصاء", "Live activity": "النشاط المباشر", "Location": "الموقع", "Loop phase": "مرحلة الحلقة", "Mean of each downsample window. Declared limits are listed per channel below.": "متوسط كل نافذة تخفيض للعينات. الحدود المعلنة مدرجة لكل قناة أدناه.", "Model's reasoning": "استدلال النموذج", "Mutation": "التغيير", "Narrative": "السرد", "Narrative pulse": "نبض السرد", "Needs attention": "يستدعي الانتباه", "Next recal due": "موعد إعادة المعايرة", "Next step": "الخطوة التالية", "No causal history yet": "لا يوجد تاريخ سببي بعد", "Normalized error": "الخطأ المعياري", "Normalized error is the residual as a share of the channel's declared span.": "الخطأ المعياري هو المتبقي كنسبة من المدى المعلن للقناة.", "Not yet observed": "لم يُرصد بعد", "Nothing has driven a framework change, so there is no episode to narrate.": "لم يدفع أي شيء بعد إلى تغيير في الإطار، لذا لا توجد حلقة لسردها.", "OHLC extracted from captured session market data.": "OHLC مستخرج من بيانات السوق المسجلة في الجلسة.", "Observation backlog, proposal state, policy decisions, and lifecycle outcomes.": "قائمة الرصد وحالة المقترحات وقرارات السياسة ونتائج دورة الحياة.", "Observations": "الرصدات", "Observed Hz": "الهرتز المرصود", "Observed rate against declared rate": "المعدل المرصود مقابل المعدل المعلن", "One global namespace, arbitrated first-wins. The challenger is recorded, never silently dropped.": "مساحة أسماء عالمية واحدة، تُحكَّم بأسبقية التسجيل. يُسجَّل المتنافس ولا يُهمَل بصمت.", "Open": "مفتوح", "Open risks": "المخاطر المفتوحة", "Open/high/low/close from captured tool output.": "الافتتاح/الأعلى/الأدنى/الإغلاق من مخرجات الأدوات المسجلة.", "Origin": "المصدر", "Outcome": "النتيجة", "PRODUCTION": "إنتاج", "Per episode: the trigger, the decision, the change, and whether the gap closed.": "لكل حلقة: المُحفِّز والقرار والتغيير وما إذا أُغلقت الفجوة.", "Per-channel calibration state, freshness, and residual correction": "حالة المعايرة وحداثتها وتصحيح المتبقي لكل قناة", "Per-segment runtime evidence. A module existing is not evidence that anything calls it.": "أدلة وقت التشغيل لكل مقطع. وجود وحدة لا يعني أن شيئًا يستدعيها.", "Pipeline": "المسار", "Pipeline evidence": "أدلة المسار", "Pipeline reachability": "إمكانية الوصول إلى المسار", "Plan": "الخطة", "Plan steps": "خطوات الخطة", "Plugin": "الملحق", "Plugin roster and maturity": "قائمة الملحقات والنضج", "Plugins": "الملحقات", "Plugins by origin": "الإضافات حسب المصدر", "Plugins by maturity": "الإضافات حسب النضج", "Policy": "السياسة", "Policy decisions": "قرارات السياسة", "Positions & actions": "المراكز والإجراءات", "Posture": "الوضع", "Price action": "حركة السعر", "Proposal": "المقترح", "Proposal status": "حالة المقترح", "Proposed, not admitted": "مُقترح وغير مقبول", "Pulse": "النبض", "Quarantine feed": "تغذية الحجر", "Ratio": "النسبة", "Read live from the registry and maturity ledger every cycle.": "يُقرأ مباشرة من السجل ودفتر النضج في كل دورة.", "Recent episodes": "الحلقات الأخيرة", "Reclaim candidates": "مرشّحو الاسترجاع", "Reclaimable": "قابل للاسترجاع", "References & follow-ups": "المراجع والمتابعات", "References (entities)": "المراجع (الكيانات)", "Registry": "السجل", "Registry delta": "فرق السجل", "Registry version": "إصدار السجل", "Regressions": "الانحدارات", "Rejected": "المرفوض", "Representative observations, capped for quick scanning.": "رصدات تمثيلية، محدودة العدد للقراءة السريعة.", "Requirements": "المتطلبات", "Research lens": "منظور بحثي", "Residual": "المتبقي", "Reward signal bandwidth": "نطاق إشارة المكافأة", "Runtime evidence": "دليل وقت التشغيل", "Sampled history per channel, newest on the right": "سجل العينات لكل قناة، الأحدث على اليمين", "Segment": "المقطع", "Segments by status": "الأجزاء حسب الحالة", "Selectable": "قابل للاختيار", "Selection delta": "فرق الاختيار", "Self-acquired": "مُكتسَب ذاتيًا", "Self-acquired plugins that are registered but unselectable or never once used.": "إضافات مُكتسَبة ذاتيًا مُسجَّلة لكنها غير قابلة للاختيار أو لم تُستخدم قطّ.", "Sentiment lens": "منظور المشاعر", "Series": "السلسلة", "Session analysis": "تحليل الجلسة", "Signal strength": "قوة الإشارة", "Signals that something grew wrong, or was withheld. Shown regardless of the open tab.": "إشارات على أن شيئًا نما بشكل خاطئ أو تم حجبه. تظهر أيًا كان التبويب المفتوح.", "Skipped slots": "الفتحات المتخطاة", "State": "الحالة", "Storyline and signal strength before drilling into positions and actions.": "السرد وقوة الإشارة قبل التوسع في المراكز والإجراءات.", "Streaming": "بث", "Suggested next steps": "الخطوات التالية المقترحة", "The line of investigation and where the open questions concentrate.": "خط البحث وأين تتركز الأسئلة المفتوحة.", "The narrative arc and how strongly themes are trending.": "قوس السرد ومدى قوة اتجاه الموضوعات.", "The world model asked for these capabilities and nothing took them up.": "طلب نموذج العالم هذه القدرات ولم يتبنّها شيء.", "Theme intensity": "شدة الموضوعات", "Themes": "الموضوعات", "This board reports how the framework changes itself. Nothing has been recorded yet.": "تُبلِّغ هذه اللوحة عن كيفية تغيير الإطار لنفسه. لم يُسجَّل أي شيء بعد.", "To": "إلى", "Tool": "الأداة", "Tool-name conflicts": "تعارضات أسماء الأدوات", "Tools": "الأدوات", "Transport": "النقل", "Transport, provenance and channel counts": "النقل والمنشأ وعدد القنوات", "Maturity": "النضج", "Maturity accrual": "تراكم النضج", "Maturity class": "فئة النضج", "Unselectable reclamation": "استرجاع غير القابل للاختيار", "Usable": "قابل للاستخدام", "VERIFIED": "مُتحقَّق", "Verdicts": "الأحكام", "Verdicts by reason": "الأحكام حسب السبب", "Verified": "مُتحقَّق", "Verified by": "تم التحقق بواسطة", "Voices & concerns": "الأصوات والمخاوف", "Watchlist": "قائمة المتابعة", "What changed in the environment, and what the framework did about it.": "ما تغيّر في البيئة، وما فعله الإطار حيال ذلك.", "Which plugin owns which tool, and which capability that tool provides.": "أي ملحق يملك أي أداة، وأي قدرة توفرها تلك الأداة.", "Who/what is in the conversation, and the concerns still open.": "من/ما هو في المحادثة، والمخاوف التي لا تزال مفتوحة.", "Why": "السبب", "Why not admitted": "سبب عدم القبول", "Why this page is empty": "لماذا هذه الصفحة فارغة", "World-model driver": "مُشغِّل نموذج العالم", "Writable": "قابل للكتابة", "aborted": "مُلغى", "accruing": "قيد التراكم", "active": "نشط", "appeared": "ظهر", "armed": "مُسلّح", "assess_compatibility": "تقييم التوافق", "built_in": "مدمج", "capability_expand": "توسيع القدرة", "committed": "مُنجَز", "conformance": "المطابقة", "declared_fitness": "الملاءمة المُعلنة", "disable": "تعطيل", "disposed": "تم التخلص منه", "effect_observed": "تم رصد الأثر", "environment_probe": "مِجَس البيئة", "execution_failed": "فشل التنفيذ", "expected_effect_absent": "الأثر المتوقع غائب", "failed": "فشل", "frozen": "مُجمَّد", "gone": "اختفى", "idle": "خامل", "install": "تثبيت", "loading": "قيد التحميل", "manual": "يدوي", "moved": "انتقل", "new_unproven": "جديد وغير مُثبَت", "no": "لا", "no_evidence": "لا يوجد دليل", "no_expected_effect_declared": "لم يُعلَن أثر متوقع", "no_outcome_observed": "لم يُرصد أي ناتج", "none": "لا شيء", "not_admitted": "غير مقبول", "not_applicable": "غير منطبق", "observe_only": "المراقبة فقط", "observed_effect": "الأثر المرصود", "open": "مفتوح", "pending": "معلّق", "reload": "إعادة تحميل", "remove": "إزالة", "reopened": "أُعيد فتحه", "resolved": "تم الحل", "rollback": "تراجع", "runtime": "وقت التشغيل", "self_acquired": "مُكتسَب ذاتيًا", "still_open": "لا يزال مفتوحًا", "tool_reported_no_effect": "الأداة لم تُبلِّغ عن أثر", "trusted": "موثوق", "unknown": "غير معروف", "unknown_tool": "أداة غير معروفة", "unloading": "قيد الإلغاء", "unscheduled": "غير مُجدول", "unverifiable": "غير قابل للتحقق", "unverified": "غير مُتحقَّق", "waiting": "في الانتظار", "watching": "يراقب", "wired": "موصول", "world_model": "نموذج العالم", "yes": "نعم", "Causal trace": "الأثر السببي", "Read-only environment-to-governance evidence for one framework-evolution snapshot.": "دليل للقراءة فقط يربط البيئة بالحوكمة للّقطة واحدة من تطور الإطار.", "Evidence boundary": "حدود الدليل", "A causal trace shows recorded facts; it does not infer missing approval or observed effect.": "يعرض الأثر السببي الحقائق المسجلة ولا يستنتج موافقة مفقودة أو أثرًا مرصودًا.", "Episodes": "الحلقات", "Declared-fitness closures": "إغلاقات الملاءمة المعلنة", "Counterfactual / mutation matrix": "مصفوفة الافتراضات المضادة / التغييرات", "Each row preserves the driver, decision, registry delta, and verification tier.": "يحفظ كل صف المحفز والقرار وفرق السجل ومستوى التحقق.", "Trigger": "المحفز", "Evidence tier": "طبقة الدليل", "Episode timeline": "الخط الزمني للحلقات", "Rebuilt from existing decision and observation records.": "أُعيد بناؤه من سجلات القرار والرصد الموجودة.", "Durable trace feed": "تدفق آثار دائم", "Rejected and no-op decisions remain visible when their trace sink is installed.": "تبقى القرارات المرفوضة والتي بلا إجراء مرئية عند تثبيت مستقبل آثارها.", "Lifecycle": "دورة الحياة", "Pipeline evidence over time": "أدلة المسار عبر الزمن", "One point per recorded change in framework state, oldest first.": "نقطة واحدة لكل تغيير مسجّل في حالة الإطار، الأقدم أولًا.", "Samples": "العينات", "Net change": "التغير الصافي"},
    ru: {"> **No effect verdict has been recorded yet**, so there is no reward signal to measure. The rates above are blank rather than zero on purpose. An effect can only be verified when the requirement declared one, and only world-model-authored requirements carry an expected effect today.": "> **Ни одного заключения об эффекте пока не записано**, поэтому измерять нечего. Показатели выше намеренно пусты, а не равны нулю. Эффект можно проверить только если требование его заявило, а сегодня заявленный эффект несут лишь требования, составленные моделью мира.", "> **Regression: a closed gap has recurred.** An evolution that looked successful did not hold. This is the one finding on this board that warrants immediate attention.": "> **Регрессия: закрытый пробел возобновился.** Эволюция, казавшаяся успешной, не удержалась. Это единственный вывод на этой панели, требующий немедленного внимания.", "> **Snapshot only.** There is no causal history to rebuild yet, so the timeline is absent rather than empty. Why is stated by the `Policy decisions` row under pipeline reachability; the live snapshot and the reachability table itself are unaffected.": "> **Только снимок.** Причинной истории для восстановления пока нет, поэтому хронология отсутствует, а не пуста. Причина указана в строке `Решения политики` под покрытием конвейера; снимок и таблица покрытия не затронуты.", "> **Some plugins are frozen by an internal defect.** A frozen plugin still reports `DRAFT`, and the trust dimension only *scores*, so it stays selectable unless it is also unregistered — check the `Selectable` column.": "> **Некоторые плагины заморожены из-за внутреннего дефекта.** Замороженный плагин по-прежнему сообщает `DRAFT`, а измерение доверия только *оценивает*, поэтому он остаётся выбираемым, пока не будет также снят с регистрации — см. столбец `Выбираемо`.", "> **Verification tier: L2 (declared fitness).** A retired observation means a candidate *declared* it provides the capability, not that the capability was observed to work. Effect verification (L3) is not wired yet, so no closure on this board should be read as proven.": "> **Уровень проверки: L2 (заявленная пригодность).** Снятое наблюдение означает, что кандидат *заявил* о предоставлении возможности, а не что возможность наблюдалась в работе. Проверка эффекта (L3) не подключена, поэтому ни одно закрытие на этой панели не следует считать доказанным.", "> A watch has to complete one cycle before there is anything to show. If this persists, check that the scheduler is enabled and that the `framework-evolution` watch is armed and not muted.": "> Прежде чем появятся данные, должен завершиться хотя бы один цикл наблюдения. Если это сохраняется, проверьте, включён ли планировщик и что наблюдение `framework-evolution` активно и не отключено.", "> An episode is written when an environment observation leads to a capability decision. None has been recorded, which is either a quiet system or a pipeline that stops earlier — the **Pipeline** tab names the segment where it stops, and what would unblock it.": "> Эпизод записывается, когда наблюдение окружения приводит к решению о возможности. Ни одного не зафиксировано: либо система спокойна, либо конвейер останавливается раньше — вкладка **Конвейер** называет сегмент остановки и то, что его разблокирует.", "> Nothing reclaims these automatically. Each holds a tool name and appears in the capability list without being selectable, so the registry grows in a direction no requirement can use.": "> Ничто не утилизирует их автоматически. Каждый занимает имя инструмента и присутствует в списке возможностей, не будучи выбираемым: реестр растёт в направлении, непригодном ни для одного требования.", "> These proposals entered no pipeline, so they appear in no decision record and no observation. Admitting them is a configuration choice.": "> Эти предложения не вошли ни в один конвейер, поэтому не отражены ни в одной записи решения или наблюдения. Их приём — вопрос конфигурации.", "A ratio below 1.0 means the sampling loop is not keeping its declared cadence.": "Отношение ниже 1,0 означает, что цикл выборки не выдерживает объявленный ритм.", "Abstained": "Воздержалось", "Acquisition authority": "Право на получение", "Acquisition lifecycle": "Жизненный цикл получения", "Action": "Действие", "After": "После", "An unverified declaration has its writable channels demoted to read-only.": "У непроверенного объявления записываемые каналы понижаются до только чтения.", "Approval": "Согласование", "Autonomous governance": "Автономное управление", "Autonomy": "Автономность", "Before": "До", "CANDIDATE": "Кандидат", "Calibrated at": "Калиброван", "Calibration health": "Состояние калибровки", "Calls": "Вызовы", "Calls (decisions)": "Рекомендации (решения)", "Candlestick": "Свечи", "Capability": "Возможность", "Capability adaptation": "Адаптация возможностей", "Capability observations": "Наблюдения возможностей", "Capability ownership": "Владение возможностями", "Capability topology": "Топология возможностей", "Change": "Изменение", "Channel": "Канал", "Channels": "Каналы", "Channels that have never been calibrated or whose calibration has expired are shown first.": "Каналы, которые никогда не калибровались или чья калибровка истекла, показаны первыми.", "Command": "Команда", "Commanded versus observed, best tracking first": "Заданное против наблюдаемого, лучшее отслеживание первым", "Composition": "Состав", "Concerns (open questions)": "Опасения (открытые вопросы)", "Confidence": "Уверенность", "Counted across every charted channel. 'near' means within 5% of a declared bound.": "Подсчитано по всем отображаемым каналам. «У границы» — в пределах 5% от объявленного предела.", "Cycles run": "Выполнено циклов", "DRAFT": "Черновик", "Days since": "Дней с тех пор", "Decision": "Решение", "Decisions read as calls; action items as the execution checklist.": "Решения читаются как рекомендации; действия — как чек-лист исполнения.", "Declared Hz": "Объявл. Гц", "Desk brief": "Сводка деска", "Device": "Устройство", "Dropped samples": "Отброшенные образцы", "Each row names one blocked segment and the change that would unblock it.": "Каждая строка называет заблокированный сегмент и изменение, которое его разблокирует.", "Effect verification (L3)": "Проверка эффекта (L3)", "Effects declared": "Заявлено эффектов", "Entities as references, and recommended next prompts to advance the work.": "Сущности как ссылки и рекомендуемые следующие запросы.", "Entities in play and the open risks still to resolve.": "Задействованные сущности и нерешённые риски.", "Envelope, rate, staleness and quality observations · newest first": "Наблюдения по огибающей, частоте, устареванию и качеству · сначала новые", "Environment": "Окружение", "Environment to framework": "От окружения к фреймворку", "Environment, selected plugin tools, and orchestration order.": "Окружение, выбранные инструменты плагинов и порядок оркестрации.", "Error rate": "Частота ошибок", "Events paced out": "Событий подавлено", "Ever used": "Использовался", "Evidence": "Обоснование", "Evidence admission": "Приём данных", "Evolution": "Эволюция", "Evolution timeline": "Хронология эволюции", "Executable": "Исполнимо", "Execution checklist": "Чек-лист исполнения", "Extracted from this session's tool/file output (not model-generated).": "Извлечено из вывода инструментов/файлов этой сессии (не сгенерировано моделью).", "Failures": "Сбои", "Fiber": "Файбер", "Fiber state changes since the previous cycle, including load retries.": "Изменения состояния fiber с предыдущего цикла, включая повторные загрузки.", "Finance lens": "Финансовый ракурс", "Follow-ups": "Продолжения", "Framework change": "Изменение фреймворка", "Framework changes as they happened, from runtime probes.": "Изменения фреймворка в момент их появления, от рантайм-зондов.", "Framework evolution": "Эволюция фреймворка", "Framework size and how much of the evolution pipeline shows runtime evidence.": "Размер фреймворка и какая часть конвейера эволюции показывает свидетельства времени выполнения.", "From": "Из", "Frozen plugins": "Замороженные плагины", "Gap closure": "Закрытие пробела", "Halt": "Останов", "How closures are verified": "Как проверяются закрытия", "How much of the effect signal is usable as feedback. This decides whether a learning policy is worth building.": "Какая часть сигнала об эффекте пригодна как обратная связь. Это определяет, стоит ли строить обучающую политику.", "How much of the framework it grew itself, and how much of the pipeline shows runtime evidence.": "Какую часть фреймворка он вырастил сам и какая часть конвейера показывает данные времени выполнения.", "How often each window sat inside, near, or outside its declared limits": "Как часто каждое окно было внутри, у границы или вне объявленных пределов", "Inquiry brief": "Сводка исследования", "Insights carded as evidence, capped for fast review.": "Инсайты как карточки-обоснования, ограничены для быстрого просмотра.", "Instruments & counterparties": "Инструменты и контрагенты", "Kept": "Оставлен", "Latest capability decision": "Последнее решение о возможностях", "Lifecycle records": "Записи жизненного цикла", "Lifecycle timeline": "Хронология жизненного цикла", "Lifecycle transitions": "Переходы жизненного цикла", "Line of inquiry": "Линия исследования", "Live activity": "Текущая активность", "Location": "Расположение", "Loop phase": "Фаза цикла", "Mean of each downsample window. Declared limits are listed per channel below.": "Среднее по каждому окну прореживания. Объявленные пределы указаны по каналам ниже.", "Model's reasoning": "Обоснование модели", "Mutation": "Изменение", "Narrative": "Сюжет", "Narrative pulse": "Нарративный пульс", "Needs attention": "Требует внимания", "Next recal due": "Следующая рекалибровка", "Next step": "Следующий шаг", "No causal history yet": "Причинной истории пока нет", "Normalized error": "Нормированная ошибка", "Normalized error is the residual as a share of the channel's declared span.": "Нормированная ошибка — остаток как доля объявленного диапазона канала.", "Not yet observed": "Ещё не наблюдалось", "Nothing has driven a framework change, so there is no episode to narrate.": "Ничто пока не вызвало изменения фреймворка, поэтому рассказывать не о чем.", "OHLC extracted from captured session market data.": "OHLC извлечён из рыночных данных, записанных в сессии.", "Observation backlog, proposal state, policy decisions, and lifecycle outcomes.": "Очередь наблюдений, состояние предложений, решения политики и итоги жизненного цикла.", "Observations": "Наблюдения", "Observed Hz": "Наблюд. Гц", "Observed rate against declared rate": "Наблюдаемая частота против объявленной", "One global namespace, arbitrated first-wins. The challenger is recorded, never silently dropped.": "Единое глобальное пространство имён, арбитраж по первому пришедшему. Претендент записывается, а не отбрасывается молча.", "Open": "Открыт", "Open risks": "Открытые риски", "Open/high/low/close from captured tool output.": "Открытие/максимум/минимум/закрытие из записанного вывода инструментов.", "Origin": "Источник", "Outcome": "Результат", "PRODUCTION": "Продакшн", "Per episode: the trigger, the decision, the change, and whether the gap closed.": "По эпизодам: триггер, решение, изменение и закрылся ли пробел.", "Per-channel calibration state, freshness, and residual correction": "Состояние калибровки, актуальность и остаточная поправка по каналам", "Per-segment runtime evidence. A module existing is not evidence that anything calls it.": "Свидетельства времени выполнения по сегментам. Наличие модуля не доказывает, что его кто-то вызывает.", "Pipeline": "Конвейер", "Pipeline evidence": "Свидетельства конвейера", "Pipeline reachability": "Достижимость конвейера", "Plan": "План", "Plan steps": "Шаги плана", "Plugin": "Плагин", "Plugin roster and maturity": "Реестр плагинов и зрелость", "Plugins": "Плагины", "Plugins by origin": "Плагины по происхождению", "Plugins by maturity": "Плагины по зрелости", "Policy": "Политика", "Policy decisions": "Решения политики", "Positions & actions": "Позиции и действия", "Posture": "Состояние", "Price action": "Ценовое движение", "Proposal": "Предложение", "Proposal status": "Статус предложения", "Proposed, not admitted": "Предложено, не принято", "Pulse": "Пульс", "Quarantine feed": "Поток карантина", "Ratio": "Отношение", "Read live from the registry and maturity ledger every cycle.": "Читается напрямую из реестра и журнала зрелости каждый цикл.", "Recent episodes": "Недавние эпизоды", "Reclaim candidates": "Кандидаты на утилизацию", "Reclaimable": "Утилизируемо", "References & follow-ups": "Ссылки и продолжения", "References (entities)": "Ссылки (сущности)", "Registry": "Реестр", "Registry delta": "Изменение реестра", "Registry version": "Версия реестра", "Regressions": "Регрессии", "Rejected": "Отклонён", "Representative observations, capped for quick scanning.": "Показательные наблюдения, ограничены для быстрого просмотра.", "Requirements": "Требования", "Research lens": "Исследовательский ракурс", "Residual": "Остаток", "Reward signal bandwidth": "Пропускная способность сигнала вознаграждения", "Runtime evidence": "Свидетельство времени выполнения", "Sampled history per channel, newest on the right": "История выборок по каналам, самое новое справа", "Segment": "Сегмент", "Segments by status": "Сегменты по статусу", "Selectable": "Выбираемый", "Selection delta": "Изменение выбора", "Self-acquired": "Самостоятельно получено", "Self-acquired plugins that are registered but unselectable or never once used.": "Самостоятельно полученные плагины, которые зарегистрированы, но невыбираемы или ни разу не использовались.", "Sentiment lens": "Ракурс тональности", "Series": "Серия", "Session analysis": "Анализ сессии", "Signal strength": "Сила сигнала", "Signals that something grew wrong, or was withheld. Shown regardless of the open tab.": "Признаки того, что что-то выросло неверно или было задержано. Показываются независимо от открытой вкладки.", "Skipped slots": "Пропущенные слоты", "State": "Состояние", "Storyline and signal strength before drilling into positions and actions.": "Сюжет и сила сигнала до перехода к позициям и действиям.", "Streaming": "Потоковая передача", "Suggested next steps": "Рекомендуемые следующие шаги", "The line of investigation and where the open questions concentrate.": "Линия исследования и где сосредоточены открытые вопросы.", "The narrative arc and how strongly themes are trending.": "Нарративная дуга и насколько сильно растут темы.", "The world model asked for these capabilities and nothing took them up.": "Модель мира запросила эти возможности, и никто их не принял.", "Theme intensity": "Интенсивность тем", "Themes": "Темы", "This board reports how the framework changes itself. Nothing has been recorded yet.": "Эта панель сообщает, как фреймворк изменяет сам себя. Пока ничего не записано.", "To": "В", "Tool": "Инструмент", "Tool-name conflicts": "Конфликты имён инструментов", "Tools": "Инструменты", "Transport": "Транспорт", "Transport, provenance and channel counts": "Транспорт, происхождение и число каналов", "Maturity": "Зрелость", "Maturity accrual": "Накопление зрелости", "Maturity class": "Класс зрелости", "Unselectable reclamation": "Утилизация невыбираемого", "Usable": "Пригодно", "VERIFIED": "Проверено", "Verdicts": "Заключений", "Verdicts by reason": "Заключения по причине", "Verified": "Проверено", "Verified by": "Подтверждено", "Voices & concerns": "Голоса и опасения", "Watchlist": "Список наблюдения", "What changed in the environment, and what the framework did about it.": "Что изменилось в окружении и что фреймворк с этим сделал.", "Which plugin owns which tool, and which capability that tool provides.": "Какой плагин владеет каким инструментом и какую возможность этот инструмент предоставляет.", "Who/what is in the conversation, and the concerns still open.": "Кто/что в разговоре и какие опасения остаются.", "Why": "Почему", "Why not admitted": "Причина отклонения", "Why this page is empty": "Почему эта страница пуста", "World-model driver": "Драйвер модели мира", "Writable": "Записываемый", "aborted": "Прервано", "accruing": "Накапливается", "active": "Активно", "appeared": "Появился", "armed": "Активно", "assess_compatibility": "Оценка совместимости", "built_in": "Встроенный", "capability_expand": "Расширение возможностей", "committed": "Завершено", "conformance": "Соответствие", "declared_fitness": "Заявленная пригодность", "disable": "Отключение", "disposed": "Освобождено", "effect_observed": "Эффект наблюдался", "environment_probe": "Зонд окружения", "execution_failed": "Сбой выполнения", "expected_effect_absent": "Ожидаемый эффект отсутствует", "failed": "Сбой", "frozen": "Заморожено", "gone": "Исчез", "idle": "Простой", "install": "Установка", "loading": "Загрузка", "manual": "Вручную", "moved": "Перешёл", "new_unproven": "Новое, непроверенное", "no": "Нет", "no_evidence": "Нет данных", "no_expected_effect_declared": "Ожидаемый эффект не заявлен", "no_outcome_observed": "Результат не наблюдался", "none": "Нет", "not_admitted": "Не принято", "not_applicable": "Неприменимо", "observe_only": "Только наблюдение", "observed_effect": "Наблюдаемый эффект", "open": "Открыто", "pending": "Ожидает", "reload": "Перезагрузка", "remove": "Удаление", "reopened": "Возобновлено", "resolved": "Закрыто", "rollback": "Откат", "runtime": "Среда выполнения", "self_acquired": "Самостоятельно получено", "still_open": "Всё ещё открыто", "tool_reported_no_effect": "Инструмент не сообщил об эффекте", "trusted": "Доверенное", "unknown": "Неизвестно", "unknown_tool": "Неизвестный инструмент", "unloading": "Выгрузка", "unscheduled": "Не запланировано", "unverifiable": "Не проверяемо", "unverified": "Непроверенное", "waiting": "Ожидание", "watching": "Наблюдает", "wired": "Подключено", "world_model": "Модель мира", "yes": "Да", "Causal trace": "Причинная трасса", "Read-only environment-to-governance evidence for one framework-evolution snapshot.": "Доказательства только для чтения от окружения к управлению для одного снимка эволюции фреймворка.", "Evidence boundary": "Граница доказательств", "A causal trace shows recorded facts; it does not infer missing approval or observed effect.": "Причинная трасса показывает записанные факты и не выводит отсутствующее согласование или наблюдаемый эффект.", "Episodes": "Эпизоды", "Declared-fitness closures": "Закрытия по заявленной пригодности", "Counterfactual / mutation matrix": "Контрфактическая матрица / мутации", "Each row preserves the driver, decision, registry delta, and verification tier.": "Каждая строка сохраняет триггер, решение, изменение реестра и уровень проверки.", "Trigger": "Триггер", "Evidence tier": "Уровень доказательств", "Episode timeline": "Хронология эпизодов", "Rebuilt from existing decision and observation records.": "Восстановлена из существующих записей решений и наблюдений.", "Durable trace feed": "Поток долговечных трасс", "Rejected and no-op decisions remain visible when their trace sink is installed.": "Отклонённые решения и решения без действия остаются видимыми, когда установлен их приёмник трасс.", "Lifecycle": "Жизненный цикл", "Pipeline evidence over time": "Свидетельства конвейера во времени", "One point per recorded change in framework state, oldest first.": "Одна точка на каждое зафиксированное изменение состояния фреймворка, старшие слева.", "Samples": "Выборки", "Net change": "Итоговое изменение"}
  };
  // Page-chrome freshness strings. A table of their own because the provenance bar
  // is not a lens: it renders above every board, so keying it off any one lens's
  // table would tie a page-wide fact to an unrelated view.
  const I18N_PROVENANCE = {
    en: {
      "Observed": "Observed", "Not yet observed": "Not yet observed",
      "every {interval}": "every {interval}", "next in {interval}": "next in {interval}",
      "event-driven": "event-driven", "muted": "muted",
      "Stale: no cycle completed in over {interval}. Nothing below can be trusted as current.": "Stale: no cycle completed in over {interval}. Nothing below can be trusted as current.",
      "confirmed unchanged {interval} later": "confirmed unchanged {interval} later",
      "{count} sec": "{count}s", "{count} min": "{count}m", "{count} hr": "{count}h", "{count} days": "{count}d",
      "Refresh now": "Refresh now", "Refreshing\u2026": "Refreshing\u2026"
    },
    zh: {
      "Observed": "观测于", "Not yet observed": "尚未观测",
      "every {interval}": "每 {interval}", "next in {interval}": "{interval} 后下一次",
      "event-driven": "事件驱动", "muted": "已静音",
      "Stale: no cycle completed in over {interval}. Nothing below can be trusted as current.": "已过期：超过 {interval} 无任何周期完成。下方内容不可被当作当前状态。",
      "confirmed unchanged {interval} later": "{interval} 后确认未变",
      "{count} sec": "{count} 秒", "{count} min": "{count} 分钟", "{count} hr": "{count} 小时", "{count} days": "{count} 天",
      "Refresh now": "立即刷新", "Refreshing\u2026": "刷新中…"
    },
    fr: {
      "Observed": "Observé à", "Not yet observed": "Pas encore observé",
      "every {interval}": "toutes les {interval}", "next in {interval}": "prochain dans {interval}",
      "event-driven": "piloté par événements", "muted": "en sourdine",
      "Stale: no cycle completed in over {interval}. Nothing below can be trusted as current.": "Périmé : aucun cycle achevé depuis plus de {interval}. Rien ci-dessous ne peut être considéré comme à jour.",
      "confirmed unchanged {interval} later": "confirmé inchangé {interval} plus tard",
      "{count} sec": "{count} s", "{count} min": "{count} min", "{count} hr": "{count} h", "{count} days": "{count} j",
      "Refresh now": "Actualiser", "Refreshing\u2026": "Actualisation…"
    },
    es: {
      "Observed": "Observado a las", "Not yet observed": "Aún no observado",
      "every {interval}": "cada {interval}", "next in {interval}": "siguiente en {interval}",
      "event-driven": "impulsado por eventos", "muted": "silenciado",
      "Stale: no cycle completed in over {interval}. Nothing below can be trusted as current.": "Obsoleto: ningún ciclo completado en más de {interval}. Nada de lo siguiente puede considerarse actual.",
      "confirmed unchanged {interval} later": "confirmado sin cambios {interval} después",
      "{count} sec": "{count} s", "{count} min": "{count} min", "{count} hr": "{count} h", "{count} days": "{count} d",
      "Refresh now": "Actualizar ahora", "Refreshing\u2026": "Actualizando…"
    },
    ar: {
      "Observed": "رُصِد عند", "Not yet observed": "لم يُرصَد بعد",
      "every {interval}": "كل {interval}", "next in {interval}": "التالي بعد {interval}",
      "event-driven": "مدفوع بالأحداث", "muted": "مكتوم",
      "Stale: no cycle completed in over {interval}. Nothing below can be trusted as current.": "قديم: لم تكتمل أي دورة لأكثر من {interval}. لا يمكن اعتبار أي مما أدناه حاليًا.",
      "confirmed unchanged {interval} later": "تأكّد عدم تغيَّره بعد {interval}",
      "{count} sec": "{count} ث", "{count} min": "{count} د", "{count} hr": "{count} س", "{count} days": "{count} ي",
      "Refresh now": "تحديث الآن", "Refreshing\u2026": "جارٍ التحديث…"
    },
    ru: {
      "Observed": "Наблюдено в", "Not yet observed": "Ещё не наблюдалось",
      "every {interval}": "каждые {interval}", "next in {interval}": "следующее через {interval}",
      "event-driven": "по событиям", "muted": "без уведомлений",
      "Stale: no cycle completed in over {interval}. Nothing below can be trusted as current.": "Устарело: ни один цикл не завершён более {interval}. Ничто ниже нельзя считать актуальным.",
      "confirmed unchanged {interval} later": "подтверждено без изменений через {interval}",
      "{count} sec": "{count} с", "{count} min": "{count} мин", "{count} hr": "{count} ч", "{count} days": "{count} д",
      "Refresh now": "Обновить", "Refreshing\u2026": "Обновление…"
    }
  };
  Object.keys(I18N).concat(Object.keys(I18N_PATCH), Object.keys(I18N_LIVE), Object.keys(I18N_TEMPLATES), Object.keys(I18N_PROVENANCE))
    .filter((lang, at, all) => all.indexOf(lang) === at)
    .forEach((lang) => {
      // Later sources win, so a locale-specific template string overrides the English
      // fallback while an absent one still resolves to readable English.
      I18N[lang] = Object.assign(
        {}, I18N.en || {}, I18N[lang] || {},
        I18N_PATCH[lang] || {}, I18N_LIVE[lang] || {}, I18N_TEMPLATES[lang] || {},
        I18N_PROVENANCE[lang] || {},
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

  const _previewCleanup = new Set();

  function disposePreviews() {
    _previewCleanup.forEach((stop) => { try { stop(); } catch (_) {} });
    _previewCleanup.clear();
  }

  function renderViewError(error) {
    rootEl.innerHTML = "";
    const card = el("div", "card view-error");
    card.appendChild(el("div", "card-title", esc(t("Failed to load view"))));
    card.appendChild(el("div", "summary", esc(String(error.message || error.code || t("Preview unavailable")))));
    if (error.request_id) card.appendChild(el("div", "kicker", esc("Request ID: " + error.request_id)));
    const retry = el("button", "primary", esc(t("Retry")));
    retry.addEventListener("click", () => fetchView());
    card.appendChild(retry);
    rootEl.appendChild(card);
  }

  async function fetchView(intent) {
    var prevTemplate = current.template;
    current = Object.assign({}, current, intent || {});
    const url = new URL("/api/view", location.origin);
    url.searchParams.set("token", TOKEN);
    Object.entries(current).forEach(([k, v]) => v && url.searchParams.set(k, v));
    try {
      const resp = await fetch(url.toString());
      let payload = {};
      try { payload = await resp.json(); } catch (_) {}
      if (!resp.ok) {
        const remote = payload && payload.error;
        const detail = remote && typeof remote === "object"
          ? remote
          : { code: "http_" + resp.status, message: String(remote || "HTTP " + resp.status) };
        throw detail;
      }
      render(payload);
      renderNav(payload.meta || {});
      renderServerHealth(payload.meta || {});
      renderProvenance(payload.meta || {});
      // Manage signal auto-refresh lifecycle on template switch
      var newTemplate = (payload.meta && payload.meta.active_template) || current.template || "";
      if (newTemplate === "signals") {
        startSignalAutoRefresh();
        injectSignalRefreshBtn();
      } else if (prevTemplate === "signals" && newTemplate !== "signals") {
        stopSignalAutoRefresh();
      }
      // Manage subagent auto-refresh lifecycle on template switch
      if (newTemplate === "subagents") {
        startSubagentAutoRefresh();
      } else if (prevTemplate === "subagents" && newTemplate !== "subagents") {
        stopSubagentAutoRefresh();
      }
    } catch (err) {
      const detail = err && typeof err === "object"
        ? err
        : { code: "view_request_failed", message: String(err) };
      renderViewError(detail);
    }
  }

  // Freshness, stated on every lens. A finding-backed board renders the newest
  // *persisted* finding, and findings dedup on a content fingerprint, so an
  // unchanged subject keeps its previous finding and the page can be minutes old
  // while looking current. ``observed_at`` was already in the payload and rendered
  // nowhere, so a reader had no way to tell -- and a board that prescribes a fix it
  // cannot confirm landed is a broken loop, not a slow one.
  //
  // Placed above the rendered tree rather than inside it: it describes the whole
  // page, and putting it in the component tree would make it a template author's
  // job to remember on each of ten lenses.
  function renderProvenance(meta) {
    const root = document.getElementById("root");
    const old = document.getElementById("provenance-bar");
    if (old) old.remove();
    const p = meta.provenance;
    if (!root || !p || typeof p !== "object") return;
    const bar = el("div", "provenance" + (p.stale ? " is-stale" : ""));
    bar.id = "provenance-bar";

    const facts = [];
    if (p.observed) {
      facts.push(t("Observed") + " " + clockLabel(p.observed_at)
        + " \u00b7 " + relativeLabel(p.age_seconds));
    } else {
      // No instant to age. Saying "0s ago" here would invent one, which is the
      // failure this bar exists to prevent rather than commit.
      facts.push(t("Not yet observed"));
    }
    // The second instant, and the reason the bar is trustworthy. A cycle that found
    // nothing changed writes no finding, so without this the page looked frozen
    // whenever it was in fact being confirmed correct on every tick.
    if (p.checked && p.unchanged_for_seconds > 1) {
      facts.push(fmt("confirmed unchanged {interval} later",
        { interval: durationLabel(p.unchanged_for_seconds) }));
    }
    if (p.cadence_seconds > 0) {
      facts.push(fmt("every {interval}", { interval: durationLabel(p.cadence_seconds) }));
      if (p.next_due_in_seconds > 0) {
        facts.push(fmt("next in {interval}", { interval: durationLabel(p.next_due_in_seconds) }));
      }
    } else {
      facts.push(t("event-driven"));
    }
    if (p.muted) facts.push(t("muted"));
    bar.appendChild(el("span", "provenance-facts", esc(facts.join("  \u00b7  "))));

    if (p.stale) {
      // The verdict is about the watch, not the content: no cycle has completed in
      // two cadences, so nothing on the page can be trusted to reflect the present.
      // Old-but-confirmed content is stable, not stale, and is not flagged here.
      bar.appendChild(el("span", "provenance-stale", esc("\u26a0 " + fmt(
        "Stale: no cycle completed in over {interval}. Nothing below can be trusted as current.",
        { interval: durationLabel(p.cadence_seconds * 2) },
      ))));
    }
    // The affordance that closes the loop. Several boards prescribe a fix ("Next
    // step: run this command"); without a way to re-run the cycle the reader had to
    // wait out a whole cadence to learn whether it landed, and had no way to tell a
    // failed change from a page that had not caught up. ``watch.refresh`` is already
    // on the server's allow-list and re-runs one producer -- it decides nothing.
    if (p.watch_id) {
      const button = el("button", "provenance-refresh", esc(t("Refresh now")));
      button.type = "button";
      button.addEventListener("click", async () => {
        button.disabled = true;
        button.textContent = t("Refreshing\u2026");
        // postAction already refetches the view for an rpc, so the bar it rebuilds
        // carries the new observed_at without a second request.
        await postAction({ kind: "rpc", name: "watch.refresh", params: { watch_id: p.watch_id } });
      });
      bar.appendChild(button);
    }
    root.insertAdjacentElement("beforebegin", bar);
  }

  // Wall-clock, for "which cycle am I looking at". Locale-independent digits on
  // purpose: this is matched against log lines, not read as prose.
  function clockLabel(epochSeconds) {
    const at = new Date(Number(epochSeconds || 0) * 1000);
    if (!isFinite(at.getTime())) return "";
    const pad = (n) => String(n).padStart(2, "0");
    return pad(at.getHours()) + ":" + pad(at.getMinutes()) + ":" + pad(at.getSeconds());
  }

  function durationLabel(seconds) {
    const s = Math.max(0, Math.round(Number(seconds || 0)));
    if (s < 60) return fmt("{count} sec", { count: s });
    if (s < 3600) return fmt("{count} min", { count: Math.round(s / 60) });
    if (s < 86400) return fmt("{count} hr", { count: Math.round(s / 3600) });
    return fmt("{count} days", { count: Math.round(s / 86400) });
  }

  // Reuses the signal timeline's relative keys, which are interpolated on {count}.
  function relativeLabel(seconds) {
    const s = Math.max(0, Math.round(Number(seconds || 0)));
    if (s < 60) return fmt("seconds ago", { count: s });
    if (s < 3600) return fmt("minutes ago", { count: Math.round(s / 60) });
    return fmt("hours ago", { count: Math.round(s / 3600) });
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
      a.addEventListener("click", (ev) => {
        ev.preventDefault();
        // Choosing a lens clears any drilled-in target. ``current`` is merged rather
        // than replaced by fetchView, so without this a device selected earlier would
        // stay in the query string for every later view.
        fetchView({ template: name, device: "", channel: "" });
      });
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
    if (action.name === "openLink" && p.url) { window.open(p.url, "_blank", "noopener"); return; }
    // Drill-down into a target view. Routed through fetchView rather than a location
    // change so the token stays out of history and the WebSocket is not torn down.
    if (p.template) { fetchView({ template: p.template, device: p.device || "", channel: p.channel || "" }); }
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

  // ── Markdown: the three constructs the templates actually author ─────────
  //
  // The ``Markdown`` node used to escape its text and print it verbatim, so every
  // aside on the evolution board read as raw markup -- a leading ``>`` and literal
  // ``**`` in front of the very sentence meant to carry the most weight.
  //
  // Escaping runs first and every transform below operates on the *escaped* text,
  // so prose can never introduce an element: the only tags in the result are the
  // ones added here. A general Markdown library is deliberately not pulled in --
  // the SDUI contract is a closed component catalog whose whole security argument
  // is that no untrusted string reaches innerHTML unescaped, and widening that
  // surface to gain three constructs is a bad trade.
  //
  // Recognised: ``> `` aside (consecutive lines fold into one blockquote),
  // ``**bold**``, and ``` `code` ```. Anything else stays literal, which is the
  // honest outcome for a construct the renderer does not claim to support.
  function mdInline(escaped) {
    const parts = String(escaped).split("`");
    // An unmatched trailing backtick is literal text, not an unterminated code
    // span: an odd count re-joins the tail so a stray backtick cannot swallow the
    // rest of the sentence into <code>.
    if (parts.length % 2 === 0) {
      const tail = parts.pop();
      parts[parts.length - 1] += "`" + tail;
    }
    return parts
      .map((part, at) => (at % 2
        // Odd segments sit between a backtick pair. Bold is not applied inside
        // them, because a config key containing ``**`` is a key, not emphasis.
        ? "<code>" + part + "</code>"
        : part.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")))
      .join("");
  }

  function renderMarkdown(raw) {
    const host = el("div", "md prose");
    let quote = [];
    let para = [];
    // Both flushes join with a space rather than preserving the newline: the
    // templates author these as YAML folded scalars, so a line break in the source
    // is a wrapping artifact of the file and never an intended hard break.
    const flushQuote = () => {
      if (!quote.length) return;
      host.appendChild(el("blockquote", "quote", mdInline(esc(quote.join(" ")))));
      quote = [];
    };
    const flushPara = () => {
      if (!para.length) return;
      host.appendChild(el("p", null, mdInline(esc(para.join(" ")))));
      para = [];
    };
    String(raw == null ? "" : raw).split(/\r?\n/).forEach((line) => {
      const text = line.trim();
      if (!text) { flushQuote(); flushPara(); return; }
      // Tested on the raw line, before escaping turns ``>`` into ``&gt;``.
      if (text.charAt(0) === ">") {
        flushPara();
        quote.push(text.slice(1).trim());
      } else if (quote.length) {
        // Lazy continuation: an unprefixed line directly under an aside belongs to
        // it. The templates author these as YAML folded scalars, which arrive
        // pre-joined, so this path exists for text that reaches the renderer any
        // other way -- a literal scalar, or a bound payload string. Without it such
        // a notice splits into a one-line quote plus an orphan paragraph.
        quote.push(text);
      } else {
        para.push(text);
      }
    });
    flushQuote();
    flushPara();
    return host;
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
    evolutionLive: renderEvolutionLive,
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

  function normalizeEvolutionEvent(value) {
    if (!value || typeof value !== "object") return null;
    const stage = String(value.stage || "");
    const kind = String(value.kind || "");
    if (!stage || !kind) return null;
    const correlation = value.correlation && typeof value.correlation === "object" ? value.correlation : {};
    return {
      event_id: String(value.event_id || value.trace_id || (stage + ":" + kind + ":" + String(value.ts || ""))),
      episode_id: String(value.episode_id || correlation.record_id || correlation.intent_id || correlation.requirement_id || value.trace_id || ""),
      stage: stage,
      kind: kind,
      ts: Number(value.ts || 0),
      summary: String(value.summary || ""),
      evidence_level: String(value.evidence_level || "runtime_trace"),
      verification_tier: String(value.verification_tier || "not_recorded"),
      side_effect_state: String(value.side_effect_state || "not_recorded"),
      payload_ref: value.payload_ref && typeof value.payload_ref === "object" ? value.payload_ref : correlation,
    };
  }

  function evolutionLaneFor(stage) {
    if (stage === "observe") return "environment";
    if (stage === "orient" || stage === "decide") return "decision";
    return "governance";
  }

  function evolutionSeverity(event) {
    if (event.kind === "trust_frozen" || event.side_effect_state === "unknown") return "alert";
    if (event.kind.indexOf("unregistered") >= 0 || event.verification_tier === "not_recorded") return "notable";
    return "info";
  }

  function evolutionTimeLabel(event) {
    const ms = Number(event.ts || 0) * 1000;
    if (!Number.isFinite(ms) || ms <= 0) return "--:--:--";
    return new Date(ms).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  }

  function showEvolutionDrawer(drawer, event) {
    drawer.innerHTML = "";
    if (!event) {
      drawer.appendChild(el("div", "card-title", esc(t("Event detail"))));
      drawer.appendChild(el("div", "summary", esc(t("Select a live event to inspect its evidence references."))));
      return;
    }
    drawer.appendChild(el("div", "card-title", esc(event.stage + " · " + event.kind)));
    drawer.appendChild(el("div", "summary", esc(event.summary || t("No entries."))));
    drawer.appendChild(el("pre", "evolution-live-json", esc(JSON.stringify({
      episode_id: event.episode_id,
      evidence_level: event.evidence_level,
      verification_tier: event.verification_tier,
      side_effect_state: event.side_effect_state,
      payload_ref: event.payload_ref,
    }, null, 2))));
  }

  // Two provenances meet in this lens and must not be blended. ``Event count`` and
  // ``Live since snapshot`` are recomputed from the event list on every append, so
  // they are genuinely live. Everything else needs a full producer cycle -- a
  // presentation event carries one fact, never a recount -- so those are labelled
  // as of the snapshot instead of being redrawn as though the increment had moved
  // them. The producer's own fingerprint docstring names this failure: a rendered
  // value that cannot be refreshed "freezes on the page while still looking
  // current, which is worse than not showing it". It held on the server and was
  // then committed here, where five of six metrics sat on load-time values while
  // the lanes advanced beside them.
  function evolutionLiveStat(label, value, provenance) {
    const stat = el("div", "stat prov-" + provenance);
    stat.appendChild(el("div", "label", esc(t(label))));
    stat.appendChild(el("div", "value", esc(value)));
    return stat;
  }

  function renderEvolutionLive(props) {
    const snapshot = props.data && typeof props.data === "object" ? props.data : {};
    const maxEvents = _clampInt(props.max_events || 60, 8, 120) || 60;
    // The instant the snapshot metrics describe. Events at or before it are already
    // counted in them; only strictly newer ones are uncounted increments.
    const baselineAt = Number(snapshot.observed_at || 0);
    const host = el("div", "evolution-live");
    const metrics = el("div", "evolution-live-metrics metric-strip");
    const drift = el("div", "evolution-live-drift");
    const lanes = el("div", "evolution-live-lanes");
    const drawer = el("div", "evolution-live-drawer card");
    const controller = {
      events: asArray(snapshot.traces).map(normalizeEvolutionEvent).filter(Boolean),
      maxEvents: maxEvents,
      render: function () {
        this.events.sort((left, right) => Number(right.ts || 0) - Number(left.ts || 0));
        this.events = this.events.slice(0, this.maxEvents);
        const since = baselineAt
          ? this.events.filter((event) => Number(event.ts || 0) > baselineAt).length
          : 0;
        metrics.innerHTML = "";
        const summary = snapshot.summary && typeof snapshot.summary === "object" ? snapshot.summary : {};
        [
          ["Event count", this.events.length, "live"],
          ["Live since snapshot", since, "live"],
          ["Episodes", summary.episode_count || 0, "snapshot"],
          ["Mutation", summary.mutation_count || 0, "snapshot"],
          ["Regressions", summary.regression_count || 0, "snapshot"],
          ["Unadmitted intents", summary.unadmitted_intent_count || 0, "snapshot"],
          ["Pipeline evidence", String(summary.segments_with_evidence || 0) + "/" + String(summary.segments_total || 0), "snapshot"],
        ].forEach((triple) => {
          metrics.appendChild(evolutionLiveStat(triple[0], triple[1], triple[2]));
        });
        // Stated only when it is true. A permanent "these may be stale" caption is
        // noise a reader learns to skip; a count that appears exactly when the two
        // provenances have diverged is a fact, and it names what closes the gap.
        drift.innerHTML = "";
        if (since > 0) {
          drift.appendChild(el("div", "drift-note", esc(fmt(
            "{count} event(s) arrived after this snapshot. Snapshot metrics refresh on the next monitor cycle.",
            { count: since },
          ))));
        }
        lanes.innerHTML = "";
        const definitions = [
          ["environment", "Environment lane", "Environment events appear when a probe records a change in the surroundings."],
          ["decision", "Decision lane", "Decision events appear when a recorded observation drives a capability choice."],
          ["governance", "Governance lane", "Governance events appear when trust, quarantine or reclamation moves."],
        ];
        definitions.forEach((definition) => {
          const lane = el("section", "evolution-live-lane " + definition[0]);
          lane.appendChild(el("div", "card-title", esc(t(definition[1]))));
          const items = this.events.filter((event) => evolutionLaneFor(event.stage) === definition[0]);
          if (!items.length) {
            // An empty lane answers what would appear here and why nothing has, so
            // "quiet" is distinguishable from "broken" without leaving the page.
            // Reporting only "No live events" left the reader unable to tell which.
            lane.appendChild(el("div", "empty-inline", esc(t("No live events"))));
            lane.appendChild(el("div", "lane-hint", esc(t(definition[2]))));
          } else {
            items.forEach((event) => {
              const row = el("button", "evolution-live-event sev-" + evolutionSeverity(event));
              row.type = "button";
              row.appendChild(el("span", "evolution-live-time", esc(evolutionTimeLabel(event))));
              row.appendChild(el("span", "evolution-live-kind", esc(event.stage + " \u00b7 " + event.kind)));
              if (event.summary) row.appendChild(el("span", "evolution-live-summary", esc(event.summary)));
              row.addEventListener("click", () => { showEvolutionDrawer(drawer, event); });
              lane.appendChild(row);
            });
          }
          lanes.appendChild(lane);
        });
      },
      append: function (event) {
        if (this.events.some((item) => item.event_id === event.event_id)) return;
        this.events.unshift(event);
        this.render();
      },
    };
    showEvolutionDrawer(drawer, null);
    controller.render();
    _evolutionLiveControllers.push(controller);
    host.appendChild(metrics);
    host.appendChild(drift);
    host.appendChild(lanes);
    host.appendChild(drawer);
    return host;
  }

  function updateEvolutionLive(payload) {
    if (getCurrentTemplate() !== "evolution_live") return;
    const event = normalizeEvolutionEvent(payload);
    if (!event) return;
    _evolutionLiveControllers.forEach((controller) => controller.append(event));
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
      const label = el("span", "bar-label", esc(tx(row.label)));
      label.title = tx(row.label);
      line.appendChild(label);
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
    // One point is kept, not discarded. A series with a single sample used to be
    // filtered out and the chart reported "No entries." -- the reader was told there
    // was no data when in fact there was exactly one observation, which for a board
    // whose subject has just started evolving is the most important reading it has.
    // A lone sample is drawn as a marker rather than an invented line.
    const groups = seriesGroups(data).slice(0, 4)
      .map((g) => ({ label: String(g.label || ""), points: asArray(g.points).map((p, i) => ({ x: p.x != null ? p.x : i, y: Number(p.y) })).filter((p) => Number.isFinite(p.y)) }))
      .filter((g) => g.points.length >= 1);
    if (!groups.length) { d.appendChild(el("div", "chart-placeholder", esc(t("No entries.")))); return d; }
    const ys = []; groups.forEach((g) => g.points.forEach((p) => ys.push(p.y)));
    const min = Math.min.apply(null, ys), max = Math.max.apply(null, ys), span = (max - min) || 1;
    const W = 320, H = 96, pad = 4;
    const svg = svgEl("svg"); svg.setAttribute("viewBox", "0 0 " + W + " " + H); svg.setAttribute("preserveAspectRatio", "none"); svg.setAttribute("class", "sparkline");
    const strokes = ["var(--accent)", "var(--info)", "var(--notable)", "var(--faint)"];
    groups.forEach((g, gi) => {
      const stroke = strokes[gi % strokes.length];
      const yAt = (p) => H - pad - ((p.y - min) / span) * (H - pad * 2);
      if (g.points.length === 1) {
        // Centred, because a single sample has no span to lay out along and pinning
        // it to the left edge would read as the start of a trend that does not exist.
        const dot = svgEl("circle");
        dot.setAttribute("cx", String(W / 2));
        dot.setAttribute("cy", yAt(g.points[0]).toFixed(1));
        dot.setAttribute("r", "3.5");
        dot.setAttribute("style", "fill:" + stroke + ";stroke:none");
        svg.appendChild(dot);
        return;
      }
      const n = Math.max(1, g.points.length - 1);
      const coords = g.points.map((p, i) => (i * (W / n)).toFixed(1) + "," + yAt(p).toFixed(1)).join(" ");
      if (opts && opts.area) {
        const poly = svgEl("polygon"); poly.setAttribute("points", "0," + (H - pad) + " " + coords + " " + W + "," + (H - pad));
        poly.setAttribute("style", "fill:" + stroke + ";opacity:.12;stroke:none"); svg.appendChild(poly);
      }
      const line = svgEl("polyline"); line.setAttribute("points", coords);
      line.setAttribute("style", "stroke:" + stroke); svg.appendChild(line);
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
    const buttons = asArray(p.row_buttons);
    const table = el("table", "data-table");
    if (p.caption) table.appendChild(tableCaption(p.caption));
    const head = document.createElement("thead"); const headRow = document.createElement("tr");
    cols.forEach((c) => headRow.appendChild(el("th", null, esc(tx(c.label || c.key || c)))));
    if (buttons.length) headRow.appendChild(el("th", "row-actions-head", ""));
    head.appendChild(headRow); table.appendChild(head);
    const body = document.createElement("tbody");
    rows.forEach((row) => {
      const tr = document.createElement("tr");
      cols.forEach((c) => { const v = row && row[c.key || c] != null ? row[c.key || c] : ""; tr.appendChild(el("td", null, esc(tx(v)))); });
      if (buttons.length) tr.appendChild(rowActionCell(buttons, row));
      bindRowAction(tr, p.row_action, row);
      body.appendChild(tr);
    });
    table.appendChild(body); return table;
  }

  // Explicit per-row buttons, because "select a row to open the device" is an instruction
  // rather than an affordance: nothing about the row says it is clickable, and a row that
  // could do two things (open it, preview it) cannot say which one a click means. The
  // whole-row action stays as a shortcut for the primary one.
  function rowActionCell(buttons, row) {
    const cell = el("td", "row-actions");
    buttons.forEach((spec) => {
      // ``require`` names a column that must be truthy for this button to apply, so a
      // Preview button appears only on devices that have something to preview. Deciding
      // per row from the row's own data keeps the template free of device knowledge.
      if (spec.require && !truthy(row[spec.require])) return;
      const params = resolveRowParams(spec, row);
      if (params === null) return;
      const button = el("button", spec.variant === "primary" ? "row-action primary" : "row-action",
                        esc(tx(spec.label || "")));
      button.addEventListener("click", (ev) => {
        ev.stopPropagation();  // a row click means "open"; this button means something else
        postAction({ kind: spec.kind, name: spec.name || "", params: params });
      });
      cell.appendChild(button);
    });
    return cell;
  }

  // Merge a spec's static params with the columns it reads from this row. Returns null
  // when a required column is missing, so the caller drops the action instead of firing
  // it with an empty target -- which would navigate to a device page for no device.
  function resolveRowParams(spec, row) {
    const params = Object.assign({}, spec.params || {});
    const fromRow = spec.param_from_row || {};
    let complete = true;
    Object.keys(fromRow).forEach((key) => {
      const value = row[fromRow[key]];
      if (value == null || value === "") { complete = false; return; }
      params[key] = value;
    });
    return complete ? params : null;
  }

  // A per-row action, resolved against the row rather than the template. Rows are
  // expanded here on the client, so a template cannot interpolate a row value into the
  // action -- it names the column to read instead (`param_from_row`) and this fills it
  // in. Without the mapping the action is dropped rather than fired with a missing
  // target, which would navigate to an empty device page.
  function bindRowAction(tr, spec, row) {
    if (!spec || !spec.kind || !row) return;
    const params = resolveRowParams(spec, row);
    if (params === null) return;
    tr.classList.add("row-actionable");
    tr.addEventListener("click", (ev) => {
      ev.stopPropagation();
      postAction({ kind: spec.kind, name: spec.name || "", params: params });
    });
  }

  function renderTimeline(node) {
    const items = asArray((node.props || {}).data); const d = el("div", "timeline");
    items.forEach((it) => { const row = el("div", "timeline-item sev-" + severityOf(it)); row.appendChild(el("div", "timeline-title", esc(it.title || ""))); if (it.summary) row.appendChild(el("div", "summary", esc(it.summary))); d.appendChild(row); });
    return d;
  }

  // ── Device panels: preview, level meter, controls ──
  //
  // The client holds no policy. Whether a preview may run is decided by the daemon's
  // approval chain, so these panels ask and report the answer: a refusal arrives as a
  // JSON body with a message that already names the next step, and is shown verbatim.
  // Duplicating the rule here would create a second gate that could disagree with the
  // one that actually enforces it.

  function mediaUrl(path, props, options) {
    const url = new URL(path, location.origin);
    url.searchParams.set("token", TOKEN);
    url.searchParams.set("device", props.device || "");
    url.searchParams.set("channel", props.channel || "");
    const request = options || {};
    ["fps", "max_width", "quality", "viewer_id"].forEach((key) => {
      if (request[key]) url.searchParams.set(key, String(request[key]));
    });
    return url.toString();
  }

  const _CAMERA_PROFILES = [
    { id: "economy", fps: 4, max_width: 640, quality: 60, label: "preview.economy" },
    { id: "balanced", fps: 8, max_width: 960, quality: 75, label: "preview.balanced" },
    { id: "detail", fps: 30, max_width: 1280, quality: 85, label: "preview.detail" },
  ];

  // The browser picks a profile, never an unbounded capture request. The values are
  // clipped again by the daemon's PreviewBroker against both these published runtime caps
  // and the channel's declaration, so editing a URL cannot turn the camera into a compute
  // or bandwidth sink. A saved choice is per physical channel; selecting Detail for the
  // desk camera must not silently make the built-in camera expensive too.
  function cameraProfiles(props) {
    const fpsCap = Math.max(0, Number(props.max_fps || 0));
    const widthCap = Math.max(0, Number(props.max_width || 0));
    const qualityCap = Math.max(0, Number(props.max_quality || 0));
    return _CAMERA_PROFILES.map((profile) => ({
      id: profile.id,
      label: profile.label,
      fps: fpsCap ? Math.min(profile.fps, fpsCap) : profile.fps,
      max_width: widthCap ? Math.min(profile.max_width, widthCap) : profile.max_width,
      quality: qualityCap ? Math.min(profile.quality, qualityCap) : profile.quality,
    }));
  }

  function previewPreferenceKey(props) {
    return "leapboard.preview." + String(props.device || "") + "." + String(props.channel || "");
  }

  function buildCameraSettings(props) {
    const profiles = cameraProfiles(props);
    const key = previewPreferenceKey(props);
    let selected = "balanced";
    try { selected = localStorage.getItem(key) || selected; } catch (_) {}
    if (!profiles.some((profile) => profile.id === selected)) selected = "balanced";
    const wrap = el("div", "preview-settings");
    const label = el("label", "preview-profile-label", esc(t("preview.profile")));
    const select = document.createElement("select");
    select.className = "preview-profile";
    profiles.forEach((profile) => {
      const option = document.createElement("option");
      option.value = profile.id;
      option.textContent = tx(t(profile.label));
      select.appendChild(option);
    });
    select.value = selected;
    const summary = el("span", "preview-profile-summary");
    function current() {
      return profiles.find((profile) => profile.id === select.value) || profiles[0];
    }
    function refreshSummary() {
      const profile = current();
      summary.innerHTML = esc(profile.max_width + " px · " + profile.fps + " fps · JPEG " + profile.quality);
    }
    select.addEventListener("change", () => {
      try { localStorage.setItem(key, select.value); } catch (_) {}
      refreshSummary();
      wrap.dispatchEvent(new CustomEvent("leap:preview-profile", { detail: current() }));
    });
    refreshSummary();
    wrap.appendChild(label);
    wrap.appendChild(select);
    wrap.appendChild(summary);
    return { dom: wrap, current: current };
  }

  function previewWsUrl(props, profile) {
    const url = new URL(mediaUrl("/api/media/ws", props, profile));
    url.protocol = url.protocol === "https:" ? "wss:" : "ws:";
    return url.toString();
  }

  function renderMediaPreview(node) {
    const p = node.props || {};
    // A picture only for a declared frame; every other previewable shape is a live scalar
    // meter. Keyed on the declared representation, never on a device class: the props used
    // to carry "camera"/"microphone", so a level source that was not a microphone rendered
    // as one. Defaulting the unknown case to the meter is the cheaper mistake -- it polls a
    // value instead of demanding a stream the daemon would refuse.
    const isLevel = String(p.representation || "") !== "frame";
    const card = el("div", "card media-preview");
    card.appendChild(el("div", "card-title", esc(tx(p.title || t("Preview")))));

    const meta = [
      isLevel ? p.unit || "dBFS" : p.media_type,
      !isLevel && p.max_fps ? p.max_fps + " fps " + t("ceiling") : "",
    ].filter(Boolean);
    if (meta.length) card.appendChild(el("div", "kicker", esc(meta.join(" \u00b7 "))));
    // The daemon's own lease, not this tab's state: it reports a device already being
    // watched -- by another browser, or by a session this tab cannot see. Without it the
    // page said "Not streaming" beside a camera whose light was on, which is the reading
    // that makes somebody distrust the indicator rather than the page.
    if (truthy(p.active)) {
      const viewers = Math.max(0, Number(p.viewers) || 0);
      const age = Math.max(0, Math.round(Number(p.frame_age_ms) || 0));
      const parts = [t("preview.in_use"), fmt("preview.viewers", { count: viewers })];
      if (age > 0) parts.push(age + " ms");
      const live = el("div", "preview-live");
      live.appendChild(el("span", "badge sev-notable", esc(parts[0])));
      live.appendChild(el("span", null, esc(parts.slice(1).join(" \u00b7 "))));
      card.appendChild(live);
    }
    const settings = isLevel ? null : buildCameraSettings(p);
    if (settings) card.appendChild(settings.dom);
    if (truthy(p.consent_required)) {
      const notice = el("div", "consent-notice");
      notice.appendChild(el("span", "badge sev-notable", esc(t("consent required"))));
      notice.appendChild(el("span", null, esc(tx(p.consent_reason || ""))));
      card.appendChild(notice);
    }

    const stage = el("div", "media-stage");
    const consentSlot = el("div", "media-consent");
    const status = el("div", "media-status", esc(t("Not streaming.")));
    const button = el("button", "media-toggle", esc(t("Start preview")));
    let running = false;
    let starting = false;
    let timer = null;
    let socket = null;
    let viewerId = "";

    function releaseViewer() {
      const owner = viewerId;
      viewerId = "";
      if (!owner) return;
      const payload = JSON.stringify({ viewer_id: owner });
      const url = api("/api/media/release");
      try {
        fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: payload,
          keepalive: true,
        }).catch(() => {});
      } catch (_) {}
    }

    function stop(preserveStage) {
      running = false;
      starting = false;
      if (timer) { clearInterval(timer); timer = null; }
      if (socket) {
        socket.onopen = null; socket.onmessage = null; socket.onerror = null; socket.onclose = null;
        try { socket.close(); } catch (_) {}
        socket = null;
      }
      releaseViewer();
      if (!preserveStage) {
        stage.innerHTML = "";
        status.innerHTML = esc(t("Not streaming."));
      }
      button.disabled = false;
      button.innerHTML = esc(t("Start preview"));
    }

    // A prompt belongs next to the panel whose request raised it. The promise does not
    // settle until the first media sample, so approval stays in this slot while the daemon
    // is waiting rather than falling through to a detached page-level notification.
    async function withConsentSlot(work) {
      let decision = "";
      _consentSink = (approval) => {
        consentSlot.innerHTML = "";
        approval.addEventListener("leap:resolved", (event) => { decision = String(event.detail || ""); });
        consentSlot.appendChild(approval);
      };
      try {
        return { value: await work(), decision: decision };
      } finally {
        _consentSink = null;
        consentSlot.innerHTML = "";
      }
    }

    async function pollLevel(meter) {
      let resp;
      try {
        resp = await fetch(mediaUrl("/api/media/level", p, { viewer_id: viewerId }));
      } catch (err) {
        status.innerHTML = esc(t("Preview unavailable") + ": " + String(err));
        return "stop";
      }
      let body = {};
      try { body = await resp.json(); } catch (_) {}
      if (body.viewer_id) viewerId = String(body.viewer_id);
      if (!resp.ok) {
        status.innerHTML = esc(body.error || body.code || "HTTP " + resp.status);
        return body.code === "level_not_ready" ? "retry" : "stop";
      }
      meter.update(body.value, body.unit || p.unit);
      status.innerHTML = esc(t("Streaming."));
      return "ok";
    }

    function openCameraStream(profile) {
      stage.innerHTML = "";
      const canvas = document.createElement("canvas");
      canvas.className = "media-frame";
      canvas.setAttribute("aria-label", String(p.title || "preview"));
      stage.appendChild(canvas);
      const context = canvas.getContext("2d");
      if (!context || !window.createImageBitmap) {
        status.innerHTML = esc(t("Preview unavailable") + ": browser image decoding is unavailable.");
        return Promise.resolve(false);
      }
      return new Promise((resolve) => {
        let settled = false;
        let decoded = false;
        let pendingBlob = null;
        let firstFrame = false;
        const finish = (value) => { if (!settled) { settled = true; resolve(value); } };
        const drawLatest = async () => {
          if (decoded || !pendingBlob) return;
          decoded = true;
          const blob = pendingBlob;
          pendingBlob = null;
          try {
            const bitmap = await createImageBitmap(blob);
            const dpr = Math.max(1, window.devicePixelRatio || 1);
            const width = bitmap.width;
            const height = bitmap.height;
            canvas.width = width * dpr;
            canvas.height = height * dpr;
            canvas.style.width = "min(100%, " + width + "px)";
            canvas.style.height = "auto";
            context.setTransform(dpr, 0, 0, dpr, 0, 0);
            context.clearRect(0, 0, width, height);
            context.drawImage(bitmap, 0, 0, width, height);
            bitmap.close();
            if (!firstFrame) {
              firstFrame = true;
              running = true;
              status.innerHTML = esc(t("Streaming."));
              button.innerHTML = esc(t("Stop preview"));
              finish(true);
            }
          } catch (err) {
            status.innerHTML = esc(t("Preview unavailable") + ": " + String(err));
            finish(false);
          } finally {
            decoded = false;
            if (pendingBlob) void drawLatest();
          }
        };
        socket = new WebSocket(previewWsUrl(p, profile));
        socket.onmessage = (event) => {
          if (typeof event.data === "string") {
            let message = {}; try { message = JSON.parse(event.data); } catch (_) {}
            if (message.type === "opened") viewerId = String(message.viewer_id || "");
            if (message.type === "error") {
              status.innerHTML = esc(message.error || message.code || t("Preview unavailable"));
              finish(false);
            }
            return;
          }
          pendingBlob = event.data instanceof Blob ? event.data : new Blob([event.data], { type: "image/jpeg" });
          void drawLatest();
        };
        socket.onerror = () => {
          status.innerHTML = esc(t("Preview unavailable"));
          finish(false);
        };
        socket.onclose = () => {
          socket = null;
          if (!firstFrame && !settled) finish(false);
          if (running) {
            running = false;
            status.innerHTML = esc(t("Preview stream ended."));
            button.innerHTML = esc(t("Start preview"));
          }
          releaseViewer();
        };
      });
    }

    async function start() {
      if (running || starting) return;
      starting = true;
      button.disabled = true;
      status.innerHTML = esc(t("Requesting access…"));
      try {
        if (isLevel) {
          const meter = buildLevelMeter(p);
          stage.innerHTML = "";
          stage.appendChild(meter.dom);
          const first = await withConsentSlot(() => pollLevel(meter));
          if (first.value === "stop") { stop(); return; }
          if (first.decision === "allow_once") {
            releaseViewer();
            status.innerHTML = esc(t("preview.one_sample"));
            return;
          }
          running = true;
          button.innerHTML = esc(t("Stop preview"));
          timer = setInterval(async () => {
            if (!running) return;
            if ((await pollLevel(meter)) === "stop") stop();
          }, 250);
          return;
        }
        const ready = await withConsentSlot(() => openCameraStream(settings.current()));
        if (!ready.value) { stop(); return; }
        if (ready.decision === "allow_once") {
          stop(true);
          status.innerHTML = esc(t("preview.one_frame"));
        }
      } finally {
        starting = false;
        button.disabled = false;
        if (!running) button.innerHTML = esc(t("Start preview"));
      }
    }

    button.addEventListener("click", (ev) => { ev.stopPropagation(); running ? stop() : void start(); });
    if (settings) {
      settings.dom.addEventListener("leap:preview-profile", () => {
        if (running) { stop(); void start(); }
      });
    }
    _previewCleanup.add(stop);
    card.appendChild(stage);
    card.appendChild(consentSlot);
    card.appendChild(status);
    card.appendChild(button);
    return card;
  }

  // A live meter with an update handle, shared by the polled microphone preview and the
  // declarative LevelMeter component. A narrow browser-local waveform is intentionally
  // not persisted: it answers "what is the input doing now" without turning room audio
  // into stored history.
  function buildLevelMeter(props) {
    const dom = el("div", "level-meter");
    const track = el("div", "level-track");
    const fill = el("span", "level-fill");
    track.appendChild(fill);
    const readout = el("div", "value", "\u2014");
    const canvas = document.createElement("canvas");
    canvas.className = "level-waveform";
    canvas.setAttribute("aria-label", String(t("preview.level_waveform")));
    dom.appendChild(track);
    dom.appendChild(readout);
    dom.appendChild(canvas);
    const floor = Number(props && props.floor != null && props.floor !== "" ? props.floor : -60);
    const samples = [];

    function draw() {
      const dpr = Math.max(1, window.devicePixelRatio || 1);
      const width = Math.max(240, Math.floor(canvas.clientWidth || dom.clientWidth || 520));
      const height = 72;
      if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
        canvas.width = width * dpr; canvas.height = height * dpr;
        canvas.style.height = height + "px";
      }
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);
      ctx.strokeStyle = "rgba(118, 124, 140, .25)";
      ctx.lineWidth = 1;
      [0.25, 0.5, 0.75].forEach((ratio) => { ctx.beginPath(); ctx.moveTo(0, height * ratio); ctx.lineTo(width, height * ratio); ctx.stroke(); });
      if (!samples.length) return;
      ctx.strokeStyle = "#b7332c";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      samples.forEach((level, index) => {
        const ratio = Math.max(0, Math.min(1, (level - floor) / (0 - floor)));
        const x = samples.length === 1 ? width : (index / (samples.length - 1)) * width;
        const y = height - ratio * height;
        if (index === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }

    return {
      dom: dom,
      update: function (value, unit) {
        const level = Number(value);
        if (!Number.isFinite(level)) {
          fill.style.width = "0%";
          readout.innerHTML = esc("\u2014");
          return;
        }
        const ratio = (level - floor) / (0 - floor);
        fill.style.width = Math.max(0, Math.min(100, ratio * 100)) + "%";
        readout.innerHTML = esc(level.toFixed(1) + " " + (unit || ""));
        samples.push(level);
        if (samples.length > 96) samples.splice(0, samples.length - 96);
        requestAnimationFrame(draw);
      },
    };
  }

  function renderLevelMeter(node) {
    const p = node.props || {};
    const wrap = el("div", "level-meter-block");
    if (p.label) wrap.appendChild(el("div", "label", esc(tx(p.label))));
    const meter = buildLevelMeter(p);
    meter.update(p.data != null ? p.data : p.value, p.unit);
    wrap.appendChild(meter.dom);
    return wrap;
  }

  // A control derived entirely from the channel's declared envelope: an enumerated
  // domain is a select, a bounded numeric is a slider, anything else is a field. The
  // submit path is deliberately two-step -- preview what would be sent, then request
  // approval -- because the dry-run runs every feasibility check without touching the
  // device, and confirming intent before an irreversible effect is cheap.
  function renderControlForm(node) {
    const p = node.props || {};
    const form = el("div", "card control-form");
    form.appendChild(el("div", "card-title", esc(tx(p.title || p.channel || t("Control")))));
    const kicker = [p.effect, p.limits ? t("limits") + ": " + p.limits : "", truthy(p.reversible) ? t("reversible") : t("irreversible")].filter(Boolean);
    form.appendChild(el("div", "kicker", esc(kicker.join(" \u00b7 "))));

    const input = buildControlInput(p);
    const row = el("div", "control-row");
    row.appendChild(input);
    if (p.unit) row.appendChild(el("span", "unit", esc(String(p.unit))));
    form.appendChild(row);

    const status = el("div", "control-status");
    const preview = el("button", null, esc(t("Preview change")));
    const submit = el("button", "primary", esc(t("Request approval")));

    async function send(name, label) {
      status.innerHTML = esc(label);
      const result = await postAction({
        kind: "rpc",
        name: name,
        params: { device: p.device || "", channel: p.channel || "", value: input.value },
      });
      if (!result) { status.innerHTML = esc(t("No response.")); return; }
      const inner = result.result || result;
      status.innerHTML = esc(inner.error || inner.detail || inner.message || JSON.stringify(inner));
    }

    preview.addEventListener("click", (ev) => { ev.stopPropagation(); send("hardware.configure_preview", t("Checking…")); });
    submit.addEventListener("click", (ev) => { ev.stopPropagation(); send("hardware.request_write", t("Requesting approval…")); });

    const actions = el("div", "control-actions");
    actions.appendChild(preview);
    actions.appendChild(submit);
    form.appendChild(actions);
    form.appendChild(status);
    form.appendChild(el("div", "kicker", esc(t("Approval is given where your session is authenticated (TUI or leap hw)."))));
    return form;
  }

  function buildControlInput(p) {
    const options = asArray(p.options);
    if (String(p.control) === "select" && options.length) {
      const select = document.createElement("select");
      options.forEach((option) => {
        const item = document.createElement("option");
        item.value = String(option); item.textContent = String(option);
        select.appendChild(item);
      });
      return select;
    }
    const input = document.createElement("input");
    if (String(p.control) === "slider") {
      input.type = "range";
      input.min = String(p.min_value); input.max = String(p.max_value);
      if (p.step) input.step = String(p.step);
      input.value = String(p.min_value);
    } else {
      input.type = "text";
    }
    return input;
  }

  function renderTabs(node) {
    const p = node.props || {};
    const wrap = el("div", "tabs");
    const bar = el("div", "tab-bar");
    const body = el("div", "tab-body");
    const panes = (node.children || []).map((child) => renderNode(child));
    panes.forEach((pane, index) => {
      const label = ((node.children[index] || {}).props || {}).title || "Tab " + (index + 1);
      const button = el("button", index === 0 ? "active" : "", esc(tx(label)));
      button.addEventListener("click", (ev) => {
        ev.stopPropagation();
        Array.from(bar.children).forEach((b) => b.classList.remove("active"));
        button.classList.add("active");
        panes.forEach((other, i) => { other.style.display = i === index ? "" : "none"; });
      });
      bar.appendChild(button);
      pane.style.display = index === 0 ? "" : "none";
      body.appendChild(pane);
    });
    if (p.title) wrap.appendChild(el("div", "section-title", esc(tx(p.title))));
    wrap.appendChild(bar);
    wrap.appendChild(body);
    return wrap;
  }

  function truthy(value) {
    // Template interpolation renders every prop through a string, so a bound false
    // arrives as "false" -- which is truthy in JS and would have shown a consent notice
    // on every channel, including the ones that need none.
    return value === true || value === "true" || value === 1 || value === "1";
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
    // Translated like every other text prop. It was not, so a Markdown notice
    // stayed English in all six locales; interpolated text still cannot match a
    // dictionary key, which is why templates keep counts in Stat and the prose
    // here literal.
    Markdown: (n) => renderMarkdown(tx((n.props || {}).text)),
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
    // Device panels. First-class rather than Custom renderers: previewing and setting a
    // peripheral is a core capability of the board, and every catalog type a shipped
    // template references must have a renderer here or it degrades to a card printing
    // its own type name (tests/test_dashboard_sdui.py enforces this).
    MediaPreview: renderMediaPreview,
    LevelMeter: renderLevelMeter,
    Form: renderControlForm,
    Tabs: renderTabs,
    Tab: (n) => renderChildren(n, el("div", "tab-pane")),
    Select: (n) => { const p = n.props || {}; const d = el("div", "control-row"); d.appendChild(buildControlInput({ control: "select", options: p.options || p.data })); return d; },
    Slider: (n) => { const p = n.props || {}; const d = el("div", "control-row"); d.appendChild(buildControlInput({ control: "slider", min_value: p.min_value, max_value: p.max_value, step: p.step })); return d; },
    LinkCard: (n) => { const p = n.props || {}; const d = el("div", "card link-card");
      const a = el("a", null, esc(tx(p.title || p.url || "link"))); a.href = String(p.url || "#");
      a.target = "_blank"; a.rel = "noopener noreferrer"; d.appendChild(a);
      if (p.summary) d.appendChild(el("div", "summary", esc(tx(p.summary)))); return d; },
    ApprovalPrompt: (n) => { const p = n.props || {}; const d = el("div", "card approval-prompt");
      d.appendChild(el("div", "card-title", esc(tx(p.title || t("Approval required")))));
      if (p.summary) d.appendChild(el("div", "summary", esc(tx(p.summary))));
      // Deliberately not a decision button. A browser session is a weaker identity than
      // the TUI process that holds the approval route, so this reports the pending
      // request and says where to answer it.
      d.appendChild(el("div", "kicker", esc(t("Answer this where your session is authenticated (TUI or leap hw).")))); return d; },
    Heatmap: (n) => { const p = n.props || {}; return chartNode(renderChartBars(p.data, p.title || t("Distribution")), p); },
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
    disposePreviews();
    _evolutionLiveControllers = [];
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
    // A reconnect may have missed bounded live events; the authoritative snapshot
    // restores the live lens without asking the daemon to replay a mutation.
    ws.addEventListener("open", () => { fetchView(); });
    ws.onclose = () => { setConnectionStatus("reconnecting…"); setTimeout(connectWS, 3000); };
    ws.onmessage = (ev) => {
      let msg; try { msg = JSON.parse(ev.data); } catch (_) { return; }
      if (msg.type === "monitor.finding") { toast(msg.payload || {}); fetchView(); }
      else if (msg.type === "approval_request") { showApproval(msg.payload || {}); }
      else if (msg.type === "watch.state") { fetchView(); }
      else if (msg.type === "evolution.presentation") { updateEvolutionLive(msg.payload || {}); }
      else if (msg.type === "view.resync") { fetchView(); }
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
      else if (msg.type === "subagent.started" || msg.type === "subagent.completed" || msg.type === "subagent.failed") {
        // Subagent lifecycle event: refresh the view if on subagents template
        if (getCurrentTemplate() === "subagents") fetchView();
      }
      else if (msg.type === "view.replace" && msg.spec) { render(msg.spec); }
    };
  }

  // ── Approval prompts raised by this page's own requests ──
  //
  // The board answers only what it asked for: the prompt arrives because a Start preview
  // click is still waiting on the daemon, and answering it completes that request. Every
  // decision the card offers is one the *policy* allowed -- the choices come from the
  // approval request, never from a list here -- and the answer goes back through
  // ``approval.resolve``, so the grant and the audit record are the orchestrator's.
  //
  // Delivered **in place**: into the panel that is waiting, right under its button, so the
  // question sits next to the thing it is about. A page-level modal was worse in both
  // directions -- it covered the panel whose consent was being asked for, and when its
  // stylesheet was stale it degraded into an unstyled block at the foot of the document,
  // which is where the user found it.
  let _consentSink = null;
  const _approvalSeen = new Set();

  function showApproval(approval) {
    const pendingId = String(approval.pending_id || "");
    if (!pendingId || _approvalSeen.has(pendingId)) return;
    _approvalSeen.add(pendingId);
    const card = buildApprovalCard(approval, pendingId);
    if (_consentSink) { _consentSink(card); return; }
    // Nothing claimed it -- an approval raised by something other than a panel on this
    // page. Float it, because an unanchored prompt still has to be answerable.
    const overlay = el("div", "approval-overlay");
    card.classList.add("approval-floating");
    overlay.appendChild(card);
    card.addEventListener("leap:resolved", () => overlay.remove());
    document.body.appendChild(overlay);
  }

  function buildApprovalCard(approval, pendingId) {
    const display = approval.display || {};
    const card = el("div", "approval-inline");
    card.appendChild(el("div", "approval-title", esc(tx(display.title || t("Approval required")))));
    if (display.summary) card.appendChild(el("div", "approval-summary", esc(String(display.summary))));
    if (display.reason) card.appendChild(el("div", "approval-reason", esc(String(display.reason))));

    const actions = el("div", "approval-actions");
    // Offered verbatim, in the order the policy gave them, minus ``show_details``: this
    // card already shows the title, the summary and the risk explanation, and
    // ``_normalize_decision`` maps that choice to *deny* -- so rendering it would be a
    // button labelled "show details" that silently refuses.
    const policyOrder = ["allow_session", "allow_once", "deny", "deny_always"];
    const choices = asArray(approval.choices)
      .filter((choice) => String(choice) !== "show_details")
      .sort((left, right) => policyOrder.indexOf(String(left)) - policyOrder.indexOf(String(right)));
    (choices.length ? choices : ["allow_once", "deny"]).forEach((choice) => {
      const label = t("approval." + choice);
      const button = el(
        "button",
        String(choice) === "allow_session" ? "approval-choice primary" : "approval-choice",
        esc(label === "approval." + choice ? String(choice) : label),
      );
      button.addEventListener("click", async (ev) => {
        ev.stopPropagation();
        Array.from(actions.children).forEach((b) => { b.disabled = true; });
        await postAction({
          kind: "approval",
          params: { pending_id: pendingId, decision: choice },
        });
        card.dispatchEvent(new CustomEvent("leap:resolved", { detail: String(choice) }));
        card.remove();
      });
      actions.appendChild(button);
    });
    card.appendChild(actions);
    return card;
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

  window.addEventListener("pagehide", disposePreviews);
  applyLocale();
  fetchView();
  connectWS();
})();
