/**
 * Conversation-style copilot workspace.
 *
 * cfg.feature       history namespace, e.g. "rca"
 * cfg.primaryField  field focused/prefilled from examples and #q= links
 * cfg.buildPayload  (form, sessionId) => request body
 * cfg.userText      (payload) => text shown in the user bubble and history
 * cfg.render        (data, rid) => response HTML
 * cfg.clearFields   field names emptied after sending
 */
function initCopilot(cfg) {
  const form = document.getElementById("copilot-form");
  if (!form) return;
  const thread = document.getElementById("thread");
  const empty = document.getElementById("empty-state");
  const scroller = document.getElementById("ws-scroll");
  const sendBtn = document.getElementById("send-btn");
  const feature = cfg.feature;
  let sessionId = null;
  let activeId = null;
  let busy = false;

  const field = (name) => form.querySelector(`[name="${name}"]`);

  const autogrow = (ta) => {
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 240)}px`;
  };
  form.querySelectorAll("textarea[data-autogrow]").forEach((ta) => {
    ta.addEventListener("input", () => autogrow(ta));
    autogrow(ta);
  });

  const fillFields = (values) => {
    Object.entries(values).forEach(([name, value]) => {
      const f = field(name);
      if (!f) return;
      f.value = value;
      f.dispatchEvent(new Event("input", { bubbles: true }));
    });
  };

  document.querySelectorAll("[data-fill]").forEach((card) => {
    card.addEventListener("click", () => {
      let values = {};
      try { values = JSON.parse(card.dataset.fill); } catch {}
      fillFields(values);
      const target = field(cfg.primaryField) || field(Object.keys(values)[0]);
      if (target) target.focus();
    });
  });

  const hash = window.location.hash.slice(1);
  if (hash.startsWith("q=") && cfg.primaryField) {
    fillFields({ [cfg.primaryField]: decodeURIComponent(hash.slice(2)) });
    const f = field(cfg.primaryField);
    if (f) f.focus();
    history.replaceState(null, "", window.location.pathname);
  }

  /* Attachments */
  const attachBtn = document.getElementById("attach-btn");
  const uploadInput = document.getElementById("upload-input");
  const chip = document.getElementById("attach-chip");

  const clearAttachment = () => {
    sessionId = null;
    chip.hidden = true;
    chip.innerHTML = "";
  };

  const upload = async (file) => {
    if (!file) return;
    if (!/\.pdf$/i.test(file.name)) {
      showToast("Only PDF files can be attached", "error");
      return;
    }
    chip.hidden = false;
    chip.innerHTML = `<span>Uploading ${escapeHTML(file.name)}…</span>`;
    const body = new FormData();
    body.append("file", file);
    if (sessionId) body.append("session_id", sessionId);
    try {
      const res = await fetch("/api/v1/uploads", { method: "POST", body });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || "Upload failed");
      sessionId = data.session_id;
      chip.innerHTML = `<span>${escapeHTML(data.filename || file.name)} · ${data.chunks} passages</span>
        <button type="button" class="chip-x" aria-label="Remove attachment">×</button>`;
      chip.querySelector(".chip-x").addEventListener("click", clearAttachment);
      showToast("PDF attached as extra context");
    } catch (err) {
      clearAttachment();
      showToast(err.message, "error");
    }
  };

  attachBtn.addEventListener("click", () => uploadInput.click());
  uploadInput.addEventListener("change", async () => {
    await upload(uploadInput.files && uploadInput.files[0]);
    uploadInput.value = "";
  });
  form.addEventListener("dragover", (e) => { e.preventDefault(); form.classList.add("dragover"); });
  form.addEventListener("dragleave", () => form.classList.remove("dragover"));
  form.addEventListener("drop", (e) => {
    e.preventDefault();
    form.classList.remove("dragover");
    upload(e.dataTransfer.files && e.dataTransfer.files[0]);
  });

  form.addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
      e.preventDefault();
      form.requestSubmit();
    }
  });

  /* Thread */
  const assistantHead = () => '<div class="msg-head"><span class="avatar">E</span><span>Reliability Copilot</span></div>';
  const userBubble = (text) => `<div class="msg msg-user"><div class="bubble">${escapeHTML(text)}</div></div>`;
  const loadingHTML = () => `${assistantHead()}
    <div class="skeleton-card">
      <ol class="steps-inline"><li class="active">Retrieving reports</li><li>Reranking evidence</li><li>Writing answer</li></ol>
      <div class="skeleton-line w-90"></div><div class="skeleton-line w-75"></div><div class="skeleton-line w-60"></div>
    </div>`;
  const newId = () => `r${Date.now().toString(36)}${Math.random().toString(36).slice(2, 6)}`;

  const setBusy = (on) => {
    busy = on;
    sendBtn.disabled = on;
    sendBtn.classList.toggle("is-loading", on);
  };

  const startSteps = (item) => {
    const steps = item.querySelectorAll(".steps-inline li");
    let i = 0;
    return setInterval(() => {
      if (i >= steps.length - 1) return;
      steps[i].classList.replace("active", "done");
      i += 1;
      steps[i].classList.add("active");
    }, 2400);
  };

  const showEmpty = (on) => { empty.hidden = !on; };

  const run = async (payload, text, item) => {
    item.innerHTML = loadingHTML();
    const timer = startSteps(item);
    setBusy(true);
    scroller.scrollTop = scroller.scrollHeight;
    try {
      const data = await postJSON(form.dataset.endpoint, payload);
      const rid = newId();
      item.dataset.rid = rid;
      item.innerHTML = `${assistantHead()}<div class="msg-body">${cfg.render(data, rid)}</div>`;
      historyStore.save(feature, { id: rid, ts: Date.now(), text, data });
      activeId = rid;
      renderRecent();
    } catch (err) {
      item.innerHTML = `${assistantHead()}<div class="error-card">
        <strong>Something went wrong</strong>
        <p>${escapeHTML(err.message)}</p>
        <button type="button" class="btn btn-secondary btn-sm" data-retry>Try again</button>
      </div>`;
      item.querySelector("[data-retry]").addEventListener("click", () => run(payload, text, item));
    } finally {
      clearInterval(timer);
      setBusy(false);
      item.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    if (busy) return;
    if (!form.checkValidity()) {
      form.reportValidity();
      return;
    }
    const payload = cfg.buildPayload(form, sessionId);
    const text = cfg.userText(payload).trim();
    if (!text) {
      showToast("Type a question first", "error");
      return;
    }
    (cfg.clearFields || []).forEach((name) => {
      const f = field(name);
      if (!f) return;
      f.value = "";
      if (f.matches("textarea[data-autogrow]")) autogrow(f);
    });
    showEmpty(false);
    thread.insertAdjacentHTML("beforeend", userBubble(text));
    const item = document.createElement("div");
    item.className = "msg msg-assistant";
    thread.appendChild(item);
    run(payload, text, item);
  });

  /* History */
  const recentSection = document.getElementById("recent-section");
  const recentList = document.getElementById("recent-list");

  const renderRecent = () => {
    if (!recentSection) return;
    recentSection.hidden = false;
    const items = historyStore.list(feature);
    if (!items.length) {
      recentList.innerHTML = '<li class="recent-empty">Your questions will appear here</li>';
      return;
    }
    recentList.innerHTML = items
      .map((entry) => `<li><button type="button" class="recent-item${entry.id === activeId ? " active" : ""}" data-history="${entry.id}">
          <span class="recent-title">${escapeHTML(entry.text)}</span>
          <span class="recent-time">${timeAgo(entry.ts)}</span>
        </button></li>`)
      .join("");
  };

  const showEntry = (entry) => {
    activeId = entry.id;
    showEmpty(false);
    thread.innerHTML = `${userBubble(entry.text)}
      <div class="msg msg-assistant" data-rid="${entry.id}">${assistantHead()}<div class="msg-body">${cfg.render(entry.data, entry.id)}</div></div>`;
    scroller.scrollTop = 0;
    renderRecent();
    closeRail();
  };

  if (recentList) {
    recentList.addEventListener("click", (e) => {
      const btn = e.target.closest("[data-history]");
      if (!btn) return;
      const entry = historyStore.get(feature, btn.dataset.history);
      if (entry) showEntry(entry);
    });
  }

  const clearBtn = document.getElementById("clear-history");
  if (clearBtn) {
    clearBtn.addEventListener("click", () => {
      historyStore.clear(feature);
      activeId = null;
      renderRecent();
      showToast("History cleared");
    });
  }

  const newChat = document.getElementById("new-chat");
  if (newChat) {
    newChat.addEventListener("click", () => {
      thread.innerHTML = "";
      activeId = null;
      clearAttachment();
      showEmpty(true);
      renderRecent();
      closeRail();
      const f = field(cfg.primaryField);
      if (f) f.focus();
    });
  }

  renderRecent();
}
