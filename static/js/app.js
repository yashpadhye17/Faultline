/* ==========================================================================
   Text helpers and markdown
   ========================================================================== */
function escapeHTML(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function inlineMarkdown(text) {
  return escapeHTML(text)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/__(.+?)__/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*(?!\*)([^*]+?)\*(?!\*)/g, "$1<em>$2</em>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
}

function renderMarkdown(src) {
  if (!src) return "";
  const text = String(src).replace(/\r\n/g, "\n").trim();
  if (!text) return "";
  try {
    if (window.marked) {
      const parse = typeof window.marked.parse === "function" ? window.marked.parse.bind(window.marked) : window.marked;
      if (typeof window.marked.setOptions === "function") {
        window.marked.setOptions({ breaks: true, gfm: true });
      }
      return parse(text);
    }
  } catch (err) {
    console.warn("marked parse failed, using fallback", err);
  }
  return renderMarkdownFallback(text);
}

function renderMarkdownFallback(src) {
  const lines = String(src).split("\n");
  const html = [];
  let inUl = false;
  let inOl = false;
  let para = [];

  const closeLists = () => {
    if (inUl) { html.push("</ul>"); inUl = false; }
    if (inOl) { html.push("</ol>"); inOl = false; }
  };
  const flushPara = () => {
    if (!para.length) return;
    html.push(`<p>${inlineMarkdown(para.join(" "))}</p>`);
    para = [];
  };

  for (const raw of lines) {
    const trimmed = raw.trim();
    if (!trimmed) { flushPara(); closeLists(); continue; }
    if (/^(-{3,}|\*{3,}|_{3,})$/.test(trimmed)) { flushPara(); closeLists(); html.push("<hr />"); continue; }
    const heading = trimmed.match(/^(#{1,6})\s+(.+)$/);
    if (heading) {
      flushPara(); closeLists();
      const level = Math.min(heading[1].length + 1, 5);
      html.push(`<h${level}>${inlineMarkdown(heading[2])}</h${level}>`);
      continue;
    }
    const ol = trimmed.match(/^(\d+)[\.\)]\s+(.+)$/);
    if (ol) {
      flushPara();
      if (inUl) { html.push("</ul>"); inUl = false; }
      if (!inOl) { html.push("<ol>"); inOl = true; }
      html.push(`<li>${inlineMarkdown(ol[2])}</li>`);
      continue;
    }
    const ul = trimmed.match(/^[-*•]\s+(.+)$/);
    if (ul) {
      flushPara();
      if (inOl) { html.push("</ol>"); inOl = false; }
      if (!inUl) { html.push("<ul>"); inUl = true; }
      html.push(`<li>${inlineMarkdown(ul[1])}</li>`);
      continue;
    }
    closeLists();
    para.push(trimmed);
  }
  flushPara();
  closeLists();
  return html.join("\n");
}

function formatScore(value) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(3) : "—";
}

function timeAgo(ts) {
  const sec = Math.max(1, Math.round((Date.now() - ts) / 1000));
  if (sec < 60) return "just now";
  const min = Math.round(sec / 60);
  if (min < 60) return `${min}m ago`;
  const hr = Math.round(min / 60);
  if (hr < 24) return `${hr}h ago`;
  return new Date(ts).toLocaleDateString();
}

/* ==========================================================================
   Feedback: status line, toasts
   ========================================================================== */
function setStatus(text) {
  const el = document.getElementById("status");
  if (el) el.textContent = text;
}

function showToast(message, tone = "default", durationMs = 3200) {
  const stack = document.getElementById("toast-stack");
  if (!stack) return;
  const node = document.createElement("div");
  node.className = tone === "error" ? "toast error" : "toast";
  node.textContent = message;
  stack.appendChild(node);
  setTimeout(() => node.remove(), durationMs);
}

async function postJSON(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || data);
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return data;
}

/* ==========================================================================
   Theme, navigation rail, drawers, tabs
   ========================================================================== */
function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem("eirc-theme", theme); } catch {}
}

function initTheme() {
  document.querySelectorAll("[data-theme-toggle]").forEach((btn) => {
    btn.addEventListener("click", () => {
      setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
    });
  });
}

function closeRail() {
  document.body.classList.remove("rail-open");
  const toggle = document.getElementById("rail-toggle");
  const backdrop = document.getElementById("rail-backdrop");
  if (toggle) toggle.setAttribute("aria-expanded", "false");
  if (backdrop) backdrop.hidden = true;
}

function initRail() {
  const toggle = document.getElementById("rail-toggle");
  const backdrop = document.getElementById("rail-backdrop");
  if (!toggle) return;
  toggle.addEventListener("click", () => {
    const open = !document.body.classList.contains("rail-open");
    document.body.classList.toggle("rail-open", open);
    toggle.setAttribute("aria-expanded", String(open));
    if (backdrop) backdrop.hidden = !open;
  });
  if (backdrop) backdrop.addEventListener("click", closeRail);
}

function openPanel(panel) {
  if (!panel) return;
  closePanels();
  panel.classList.add("open");
  panel.setAttribute("aria-hidden", "false");
  const backdrop = document.getElementById("overlay-backdrop");
  if (backdrop) backdrop.hidden = false;
  const focusable = panel.querySelector("button, input, select, a");
  if (focusable) focusable.focus({ preventScroll: true });
}

function closePanels() {
  document.querySelectorAll(".panel-drawer.open").forEach((p) => {
    p.classList.remove("open");
    p.setAttribute("aria-hidden", "true");
  });
  const backdrop = document.getElementById("overlay-backdrop");
  if (backdrop) backdrop.hidden = true;
}

function openDrawer(title, bodyHtml) {
  const drawer = document.getElementById("source-drawer");
  if (!drawer) return;
  document.getElementById("drawer-title").textContent = title;
  document.getElementById("drawer-body").innerHTML = bodyHtml;
  openPanel(drawer);
}

function initPanels() {
  const backdrop = document.getElementById("overlay-backdrop");
  if (backdrop) backdrop.addEventListener("click", closePanels);
  document.addEventListener("click", (e) => {
    if (e.target.closest("[data-panel-close]")) closePanels();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      closePanels();
      closeRail();
    }
  });
}

function activateTab(root, name) {
  root.querySelectorAll("[data-tab]").forEach((btn) => {
    const on = btn.dataset.tab === name;
    btn.classList.toggle("active", on);
    btn.setAttribute("aria-selected", String(on));
  });
  root.querySelectorAll("[data-panel]").forEach((panel) => {
    panel.hidden = panel.dataset.panel !== name;
  });
}

function initTabs() {
  document.querySelectorAll("[data-tabs]").forEach((root) => {
    root.querySelectorAll("[data-tab]").forEach((btn) => {
      btn.addEventListener("click", () => activateTab(root, btn.dataset.tab));
    });
  });
}

/* ==========================================================================
   System status (top bar) and landing page data
   ========================================================================== */
async function fetchSystemMetrics() {
  const res = await fetch("/metrics");
  return res.json();
}

async function initSystemPills() {
  const el = document.getElementById("system-pills");
  if (!el) return;
  try {
    const [ready, metrics] = await Promise.all([fetch("/readyz").then((r) => r.json()), fetchSystemMetrics()]);
    const vs = metrics.vectorstore || {};
    const pills = [];
    if (ready.vectorstore_loaded) {
      pills.push(`<span class="badge ok">${escapeHTML(vs.doc_count ?? ready.doc_count ?? 0)} reports indexed</span>`);
    } else {
      pills.push('<span class="badge warn">Index not ready</span>');
    }
    if (metrics.hybrid_search && metrics.bm25_enabled) pills.push('<span class="badge brand">Hybrid search</span>');
    el.innerHTML = pills.join("");
  } catch {
    el.innerHTML = '<span class="badge warn">Status unavailable</span>';
  }
}

function aggregateFromRun(data) {
  const metrics = (data && data.metrics) || {};
  const cells = metrics.cells || [];
  if (cells.length) return { agg: cells[0].aggregate || {}, label: cells[0].llm_model };
  return { agg: metrics.aggregate || (data.manifest || {}).aggregate_metrics || {}, label: (data.manifest || {}).model_name };
}

/** Metrics that meet their configured threshold (score >= threshold). */
function passingMetricKeys(agg, thresholds) {
  return Object.keys(agg || {}).filter((k) => {
    const thr = thresholds[k];
    return thr != null && Number.isFinite(Number(thr)) && Number(agg[k]) >= Number(thr);
  });
}

async function initLanding() {
  const proof = document.getElementById("proof-bar");
  if (!proof) return;
  try {
    const m = await fetchSystemMetrics();
    const vs = m.vectorstore || {};
    const values = {
      docs: vs.doc_count != null ? Number(vs.doc_count).toLocaleString() : "—",
      chunks: vs.chunk_count != null ? Number(vs.chunk_count).toLocaleString() : "—",
      retrieval: m.hybrid_search && m.bm25_enabled ? "Hybrid" : "Semantic",
      rerank: m.reranker_enabled ? "On" : "Off",
    };
    proof.querySelectorAll("[data-proof]").forEach((el) => {
      el.textContent = values[el.dataset.proof];
      el.classList.remove("skeleton");
    });
  } catch {
    proof.querySelectorAll("[data-proof]").forEach((el) => el.classList.remove("skeleton"));
  }
}

function scoreCardHTML(name, score, threshold, cls = "score-card") {
  const n = Number(score);
  const pct = Number.isFinite(n) ? Math.max(0, Math.min(100, n * 100)) : 0;
  const hasThr = threshold != null && Number.isFinite(Number(threshold));
  const pass = hasThr ? n >= Number(threshold) : null;
  const badge = pass == null ? "" : pass ? '<span class="badge ok">Pass</span>' : '<span class="badge danger">Below target</span>';
  const mark = hasThr ? `<span class="meter-mark" style="left:${Math.min(100, Number(threshold) * 100)}%"></span>` : "";
  return `<div class="${cls} score-card">
    <div class="score-card-top"><span>${escapeHTML(String(name).replaceAll("_", " "))}</span>${badge}</div>
    <strong>${formatScore(score)}</strong>
    <div class="meter"><div class="meter-fill ${pass == null ? "" : pass ? "pass" : "fail"}" style="width:${pct}%"></div>${mark}</div>
    <small>${hasThr ? `Target ${formatScore(threshold)}` : "No target set"}</small>
  </div>`;
}

/* ==========================================================================
   Local history
   ========================================================================== */
const historyStore = {
  limit: 20,
  key(feature) { return `eirc-history-${feature}`; },
  list(feature) {
    try { return JSON.parse(localStorage.getItem(this.key(feature)) || "[]"); } catch { return []; }
  },
  save(feature, entry) {
    const items = this.list(feature).filter((e) => e.id !== entry.id);
    items.unshift(entry);
    try { localStorage.setItem(this.key(feature), JSON.stringify(items.slice(0, this.limit))); } catch {}
  },
  get(feature, id) { return this.list(feature).find((e) => e.id === id); },
  clear(feature) { try { localStorage.removeItem(this.key(feature)); } catch {} },
};

/* ==========================================================================
   Response rendering
   ========================================================================== */
const responseStore = new Map();

function responseRecord(rid) {
  if (!responseStore.has(rid)) responseStore.set(rid, { markdown: "", citations: [] });
  return responseStore.get(rid);
}

function listHTML(items) {
  if (!items || !items.length) return "<p class='muted small'>None extracted.</p>";
  return `<ul class="md-body">${items.map((i) => `<li>${inlineMarkdown(String(i))}</li>`).join("")}</ul>`;
}

function accordion(title, bodyHtml, { open = false, count = null } = {}) {
  const countHtml = count != null ? `<span class="acc-count">${count}</span>` : "";
  return `<details class="acc"${open ? " open" : ""}>
    <summary><span>${escapeHTML(title)}${countHtml}</span></summary>
    <div class="acc-body">${bodyHtml}</div>
  </details>`;
}

function answerCard(title, markdownSrc, rid) {
  responseRecord(rid).markdown = markdownSrc || "";
  const md = renderMarkdown(markdownSrc || "");
  return `<div class="answer-card">
    <div class="answer-head">
      <h3>${escapeHTML(title)}</h3>
      <div class="answer-actions">
        <button type="button" class="btn btn-ghost btn-sm" data-copy="markdown" data-rid="${rid}">Copy markdown</button>
        <button type="button" class="btn btn-ghost btn-sm" data-copy="plain" data-rid="${rid}">Copy text</button>
      </div>
    </div>
    <div class="md-body" data-answer-body="${rid}">${md || "<p class='muted'>No answer text returned.</p>"}</div>
  </div>`;
}

function sourcesHTML(citations, rid) {
  const list = [...(citations || [])].sort((a, b) => Number(b.score || 0) - Number(a.score || 0));
  responseRecord(rid).citations = list;
  if (!list.length) return '<p class="muted small">No citations returned for this answer.</p>';
  const pills = list
    .map((c, i) => `<button type="button" class="source-pill" data-source="${i}" data-rid="${rid}" title="${escapeHTML(c.title || c.doc_id)}">
        <span class="source-num">${i + 1}</span><span class="source-title">${escapeHTML(c.title || c.doc_id)}</span>
      </button>`)
    .join("");
  return `<div class="sources">
    <div class="sources-head"><span>${list.length} source${list.length === 1 ? "" : "s"}</span>
      <button type="button" class="link-btn small" data-sources-all data-rid="${rid}">View all</button></div>
    <div class="source-pills">${pills}</div>
  </div>`;
}

function runMetaHTML(meta) {
  if (!meta) return "";
  const parts = [
    meta.latency_ms != null ? `${(Number(meta.latency_ms) / 1000).toFixed(1)}s` : null,
    meta.model_name || null,
    meta.num_contexts != null ? `${meta.num_contexts} passages` : null,
    meta.prompt_id ? `${meta.prompt_id}@${meta.prompt_version}` : null,
    meta.request_id ? `id ${String(meta.request_id).slice(0, 8)}` : null,
  ].filter(Boolean);
  return `<div class="run-meta">${parts.map((p) => `<span>${escapeHTML(p)}</span>`).join("")}</div>`;
}

function sourceDetailHTML(c, i) {
  return `<article class="source-detail">
    <div class="source-meta">
      <span class="badge plain">Source ${i + 1}</span>
      <span class="badge brand">Relevance ${formatScore(c.score)}</span>
    </div>
    <h3>${escapeHTML(c.title || c.doc_id)}</h3>
    <p class="mono subtle">${escapeHTML(c.doc_id || "")}</p>
    ${c.snippet ? `<div class="source-snippet">${escapeHTML(c.snippet)}</div>` : ""}
    ${c.citation_url ? `<a class="btn btn-secondary btn-sm" href="${escapeHTML(c.citation_url)}" target="_blank" rel="noopener">Open in NTRS</a>` : ""}
  </article>`;
}

function allSourcesHTML(list, rid) {
  return `<div class="source-list">${list
    .map((c, i) => `<button type="button" class="source-row" data-source="${i}" data-rid="${rid}">
        <span class="source-num">${i + 1}</span>
        <span><strong>${escapeHTML(c.title || c.doc_id)}</strong><small>${escapeHTML(c.doc_id || "")}</small></span>
        <span class="mono subtle">${formatScore(c.score)}</span>
      </button>`)
    .join("")}</div>`;
}

function initResponseActions() {
  document.addEventListener("click", async (e) => {
    const copyBtn = e.target.closest("[data-copy]");
    if (copyBtn) {
      const rid = copyBtn.dataset.rid;
      let text = responseRecord(rid).markdown;
      if (copyBtn.dataset.copy === "plain") {
        const body = document.querySelector(`[data-answer-body="${rid}"]`);
        text = body ? body.innerText : text;
      }
      try {
        await navigator.clipboard.writeText(text);
        showToast("Copied to clipboard");
      } catch {
        showToast("Could not copy", "error");
      }
      return;
    }
    const sourceBtn = e.target.closest("[data-source]");
    if (sourceBtn) {
      const list = responseRecord(sourceBtn.dataset.rid).citations;
      const idx = Number(sourceBtn.dataset.source);
      if (list[idx]) openDrawer(`Source ${idx + 1} of ${list.length}`, sourceDetailHTML(list[idx], idx));
      return;
    }
    const allBtn = e.target.closest("[data-sources-all]");
    if (allBtn) {
      const rid = allBtn.dataset.rid;
      const list = responseRecord(rid).citations;
      openDrawer(`${list.length} sources`, allSourcesHTML(list, rid));
    }
  });
}

function renderRCA(data, rid) {
  const causes = data.likely_causes || [];
  const causesHtml = causes.length
    ? `<ol class="md-body">${causes
        .map((c) => `<li><strong>${escapeHTML(c.cause)}</strong>${c.confidence ? ` <span class="badge plain">${escapeHTML(c.confidence)}</span>` : ""}<br><span class="muted">${inlineMarkdown(String(c.evidence || ""))}</span></li>`)
        .join("")}</ol>`
    : "<p class='muted small'>None parsed.</p>";
  return (
    answerCard("Analysis", data.raw_answer || data.summary || "", rid) +
    sourcesHTML(data.citations, rid) +
    accordion("Likely causes", causesHtml, { open: true, count: causes.length }) +
    accordion("Recommended checks", listHTML(data.recommended_checks), { count: (data.recommended_checks || []).length }) +
    accordion("Corrective actions", listHTML(data.corrective_actions), { count: (data.corrective_actions || []).length }) +
    accordion("Problem statement", `<div class="md-body">${renderMarkdown(data.problem_statement || "") || "<p class='muted small'>Not provided.</p>"}</div>`) +
    runMetaHTML(data.run_meta)
  );
}

function renderChat(data, rid) {
  return (
    answerCard("Answer", data.answer || "", rid) +
    sourcesHTML(data.citations, rid) +
    accordion("Troubleshooting steps", listHTML(data.steps), { open: true, count: (data.steps || []).length }) +
    runMetaHTML(data.run_meta)
  );
}

function renderComparison(data, rid) {
  const title = `${data.topic_a || "Topic A"} vs ${data.topic_b || "Topic B"}`;
  return (
    answerCard(title, data.raw_answer || data.summary || "", rid) +
    sourcesHTML(data.citations, rid) +
    accordion("Similarities", listHTML(data.similarities), { count: (data.similarities || []).length }) +
    accordion("Differences", listHTML(data.differences), { open: true, count: (data.differences || []).length }) +
    accordion("Risks", listHTML(data.risks), { count: (data.risks || []).length }) +
    accordion("Recommendations", listHTML(data.recommendations), { count: (data.recommendations || []).length }) +
    runMetaHTML(data.run_meta)
  );
}

function renderResearch(data, rid) {
  return answerCard("Answer", data.answer || "", rid) + sourcesHTML(data.citations, rid) + runMetaHTML(data.run_meta);
}

/* ==========================================================================
   Boot
   ========================================================================== */
initTheme();
initRail();
initPanels();
initTabs();
initResponseActions();
initSystemPills();
initLanding();
