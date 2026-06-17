"use strict";

const state = {
  panel: [],
  judge: "",
  presets: {},
  allowOverrides: false,
  fusionModel: "openfusion",
  preset: "quality",
  busy: false,
};

const $ = (id) => document.getElementById(id);

async function loadConfig() {
  try {
    const res = await fetch("/v1/config");
    if (!res.ok) throw new Error("config " + res.status);
    const cfg = await res.json();
    state.presets = cfg.presets || {};
    state.allowOverrides = !!cfg.allow_request_overrides;
    state.fusionModel = cfg.fusion_model || "openfusion";
    state.panel = (cfg.panel || []).slice();
    state.judge = cfg.judge || "";
    $("webSearch").checked = !!(cfg.tools && cfg.tools.web_search);
    state.preset = cfg.preset || "custom";
    syncPresetTabs();
    if (!state.allowOverrides) {
      $("configNote").textContent =
        "This server uses a fixed config (set allow_request_overrides: true to edit the panel from here).";
    }
  } catch (err) {
    $("configNote").textContent = "Could not load server config: " + err.message;
  }
  renderPanel();
  renderJudge();
}

function chip(value, onRemove, onEdit) {
  const el = document.createElement("span");
  el.className = "chip";
  const input = document.createElement("input");
  input.value = value;
  input.disabled = !state.allowOverrides;
  input.addEventListener("change", () => onEdit(input.value.trim()));
  el.appendChild(input);
  if (state.allowOverrides && onRemove) {
    const btn = document.createElement("button");
    btn.textContent = "✕";
    btn.title = "Remove";
    btn.addEventListener("click", onRemove);
    el.appendChild(btn);
  }
  return el;
}

function renderPanel() {
  const box = $("panelChips");
  box.innerHTML = "";
  state.panel.forEach((model, i) => {
    box.appendChild(
      chip(
        model,
        () => {
          state.panel.splice(i, 1);
          renderPanel();
        },
        (v) => {
          state.panel[i] = v;
        }
      )
    );
  });
  $("addModel").style.display = state.allowOverrides ? "" : "none";
}

function renderJudge() {
  const box = $("judgeChip");
  box.innerHTML = "";
  box.appendChild(chip(state.judge, null, (v) => (state.judge = v)));
}

function syncPresetTabs() {
  document.querySelectorAll(".preset-tab").forEach((tab) => {
    tab.classList.toggle("is-active", tab.dataset.preset === state.preset);
  });
}

function selectPreset(name) {
  state.preset = name;
  syncPresetTabs();
  if (name !== "custom" && state.presets[name]) {
    state.panel = state.presets[name].panel.slice();
    state.judge = state.presets[name].judge;
    $("webSearch").checked = true;
    renderPanel();
    renderJudge();
  }
}

function setStatus(text) {
  $("status").textContent = text;
}

function renderAnalysis(analysis) {
  const box = $("analysisBody");
  box.innerHTML = "";
  if (analysis.raw) {
    const pre = document.createElement("div");
    pre.style.whiteSpace = "pre-wrap";
    pre.textContent = analysis.raw;
    box.appendChild(pre);
  } else {
    for (const [key, value] of Object.entries(analysis)) {
      const h = document.createElement("div");
      h.className = "analysis-key";
      h.textContent = key.replace(/_/g, " ");
      box.appendChild(h);
      const items = Array.isArray(value) ? value : [value];
      const ul = document.createElement("ul");
      items.forEach((item) => {
        const li = document.createElement("li");
        li.textContent = typeof item === "string" ? item : JSON.stringify(item);
        ul.appendChild(li);
      });
      box.appendChild(ul);
    }
  }
  $("analysisBox").hidden = false;
}

function renderUsage(usage) {
  const total = usage.total || usage.panel_total || usage;
  if (!total) return;
  const parts = [];
  if (total.total_tokens != null) parts.push(`<b>${total.total_tokens}</b> tokens`);
  if (total.cost != null) parts.push(`<b>$${Number(total.cost).toFixed(4)}</b>`);
  if (Array.isArray(usage.panel)) parts.push(`${usage.panel.length} panel members + judge`);
  const box = $("usage");
  box.innerHTML = parts.join("");
  box.hidden = parts.length === 0;
}

function handleEvent(eventName, data) {
  if (eventName === "progress") {
    setStatus(data.message + (data.panel_count != null ? ` (${data.panel_count} answers)` : ""));
    return;
  }
  if (eventName === "analysis") {
    renderAnalysis(data);
    return;
  }
  if (eventName === "usage") {
    renderUsage(data);
    return;
  }
  // default: a chat.completion.chunk
  if (data.error) {
    setStatus("Error: " + (data.error.message || "upstream error"));
    return;
  }
  const delta = data.choices && data.choices[0] && data.choices[0].delta;
  if (delta && delta.content) {
    $("answer").textContent += delta.content;
  }
}

function parseBlock(block) {
  let eventName = null;
  const dataLines = [];
  block.split("\n").forEach((line) => {
    if (line.startsWith("event:")) eventName = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  });
  if (dataLines.length === 0) return;
  const raw = dataLines.join("\n");
  if (raw === "[DONE]") return;
  let data;
  try {
    data = JSON.parse(raw);
  } catch (_) {
    return;
  }
  handleEvent(eventName, data);
}

function buildOverride() {
  if (!state.allowOverrides) return null;
  return {
    panel: state.panel.filter(Boolean),
    judge: state.judge,
    tools: { web_search: $("webSearch").checked },
  };
}

async function send() {
  if (state.busy) return;
  const prompt = $("prompt").value.trim();
  if (!prompt) return;
  state.busy = true;
  $("send").disabled = true;
  $("output").hidden = false;
  $("answer").textContent = "";
  $("analysisBox").hidden = true;
  $("usage").hidden = true;
  setStatus("Sending…");

  const payload = {
    model: state.fusionModel,
    messages: [{ role: "user", content: prompt }],
    stream: true,
  };
  const override = buildOverride();
  if (override) payload.openfusion = override;

  const headers = { "Content-Type": "application/json" };
  const token = $("gatewayToken").value.trim();
  if (token) headers.Authorization = "Bearer " + token;

  try {
    const res = await fetch("/v1/chat/completions", {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      setStatus("Error " + res.status + ": " + ((body.error && body.error.message) || res.statusText));
      return;
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf("\n\n")) !== -1) {
        const block = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        parseBlock(block);
      }
    }
    if (buffer.trim()) parseBlock(buffer);
    if (!$("status").textContent.startsWith("Error")) setStatus("Done.");
  } catch (err) {
    setStatus("Request failed: " + err.message);
  } finally {
    state.busy = false;
    $("send").disabled = false;
  }
}

function init() {
  document.querySelectorAll(".preset-tab").forEach((tab) => {
    tab.addEventListener("click", () => selectPreset(tab.dataset.preset));
  });
  $("addModel").addEventListener("click", () => {
    if (!state.allowOverrides) return;
    state.panel.push("");
    state.preset = "custom";
    syncPresetTabs();
    renderPanel();
  });
  $("send").addEventListener("click", send);
  $("prompt").addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") send();
  });
  loadConfig();
}

init();
