// Annotator UI for the Egocentric Implicit Ownership benchmark.
// Pure vanilla JS to keep the deploy story simple — no build step.

const LABELS = ["MINE", "PERSON_k", "SHARED", "AMBIGUOUS"];
const TAXONOMIES = ["A", "B", "C", "D"];
const STATUSES = ["draft", "in_review", "verified", "rejected"];
const COLORS = {
  MINE: "#f59e0b",
  PERSON_k: "#a855f7",
  SHARED: "#22c55e",
  AMBIGUOUS: "#94a3b8",
};

const state = {
  scenes: [],
  active: null,        // current SceneRecord
  edits: { scene_label: null, scene_taxonomy: null, review_status: null, notes: null, object_overrides: {} },
};

const els = {
  list: document.getElementById("scene-list"),
  empty: document.getElementById("empty-state"),
  detail: document.getElementById("scene-detail"),
  clipId: document.getElementById("clip-id"),
  datasetBadge: document.getElementById("dataset-badge"),
  taxonomyBadge: document.getElementById("taxonomy-badge"),
  statusBadge: document.getElementById("status-badge"),
  confBadge: document.getElementById("conf-badge"),
  narration: document.getElementById("narration"),
  framesGrid: document.getElementById("frames-grid"),
  sceneLabelButtons: document.getElementById("scene-label-buttons"),
  taxonomyButtons: document.getElementById("taxonomy-buttons"),
  statusButtons: document.getElementById("status-buttons"),
  notes: document.getElementById("notes"),
  save: document.getElementById("save"),
  saveFeedback: document.getElementById("save-feedback"),
  objectsTbody: document.querySelector("#objects-table tbody"),
  nObjects: document.getElementById("n-objects"),
  relationsList: document.getElementById("relations-list"),
  nRelations: document.getElementById("n-relations"),
  editsList: document.getElementById("edits-list"),
  nEdits: document.getElementById("n-edits"),
  activityList: document.getElementById("activity-list"),
  stats: document.getElementById("stats"),
  refresh: document.getElementById("refresh"),
  filterStatus: document.getElementById("filter-status"),
  filterTaxonomy: document.getElementById("filter-taxonomy"),
  filterLabel: document.getElementById("filter-label"),
  annotator: document.getElementById("annotator"),
};

// Restore annotator name from localStorage.
els.annotator.value = localStorage.getItem("egoown.annotator") || "";
els.annotator.addEventListener("change", () => {
  localStorage.setItem("egoown.annotator", els.annotator.value || "");
});

async function fetchJSON(url, opts = {}) {
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error(`${url} → ${res.status}`);
  return res.json();
}

async function loadScenes() {
  const params = new URLSearchParams();
  if (els.filterStatus.value) params.set("status", els.filterStatus.value);
  if (els.filterTaxonomy.value) params.set("taxonomy", els.filterTaxonomy.value);
  if (els.filterLabel.value) params.set("label", els.filterLabel.value);
  state.scenes = await fetchJSON(`/api/scenes?${params}`);
  renderScenes();
}

function renderScenes() {
  els.list.innerHTML = "";
  state.scenes.forEach((s) => {
    const li = document.createElement("li");
    if (state.active && state.active.clip.clip_id === s.clip_id) li.classList.add("active");
    li.innerHTML = `
      <div class="row">
        <span class="clip-id">${escapeHtml(s.clip_id)}</span>
        <span class="status-pill ${s.review_status}">${s.review_status}</span>
      </div>
      <div class="row">
        <small style="color: var(--muted)">${escapeHtml(s.dataset)} · ${escapeHtml(s.verb || '-')}</small>
        <span class="label-pill ${s.scene_label || 'UNLABELED'}">${s.scene_label || '—'}</span>
      </div>
      <div class="row">
        <small style="color: var(--muted)">tax ${s.taxonomy} · ${s.n_objects} objs</small>
        <small style="color: var(--muted)">${s.auto_label_confidence != null ? "conf " + s.auto_label_confidence.toFixed(2) : ''}</small>
      </div>`;
    li.addEventListener("click", () => loadScene(s.clip_id));
    els.list.appendChild(li);
  });
}

async function loadScene(clipId) {
  state.active = await fetchJSON(`/api/scenes/${encodeURIComponent(clipId)}`);
  state.edits = {
    scene_label: state.active.scene_label,
    scene_taxonomy: state.active.scene_taxonomy || state.active.clip.taxonomy,
    review_status: state.active.review_status,
    notes: state.active.notes || "",
    object_overrides: {},
  };
  renderActive();
  renderScenes();    // re-highlight in sidebar
}

function renderActive() {
  if (!state.active) {
    els.empty.hidden = false;
    els.detail.hidden = true;
    return;
  }
  els.empty.hidden = true;
  els.detail.hidden = false;

  const r = state.active;
  els.clipId.textContent = r.clip.clip_id;
  els.datasetBadge.textContent = r.clip.dataset;
  els.taxonomyBadge.textContent = `Tax ${state.edits.scene_taxonomy}`;
  els.statusBadge.textContent = state.edits.review_status;
  els.confBadge.textContent = r.auto_label_confidence != null ? `conf ${r.auto_label_confidence.toFixed(2)}` : '';
  els.narration.textContent = r.clip.narration || "";

  renderSegButtons(els.sceneLabelButtons, LABELS, state.edits.scene_label, (v) => { state.edits.scene_label = v; renderActive(); }, (v) => `label-pill ${v}`);
  renderSegButtons(els.taxonomyButtons, TAXONOMIES, state.edits.scene_taxonomy, (v) => { state.edits.scene_taxonomy = v; renderActive(); });
  renderSegButtons(els.statusButtons, STATUSES, state.edits.review_status, (v) => { state.edits.review_status = v; renderActive(); });
  els.notes.value = state.edits.notes || "";
  els.notes.oninput = () => { state.edits.notes = els.notes.value; };

  renderFrames(r);
  renderObjectTable(r);
  renderRelations(r);
  renderEdits(r);
}

function renderSegButtons(container, options, current, onSelect, classFn) {
  container.innerHTML = "";
  options.forEach((opt) => {
    const btn = document.createElement("button");
    btn.className = "seg-btn" + (opt === current ? " active" : "");
    btn.textContent = opt;
    if (classFn) btn.classList.add(classFn(opt));
    btn.onclick = (e) => { e.preventDefault(); onSelect(opt); };
    container.appendChild(btn);
  });
}

function renderFrames(r) {
  els.framesGrid.innerHTML = "";
  r.frames.forEach((f) => {
    const card = document.createElement("div");
    card.className = "frame-card";
    card.innerHTML = `
      <span class="tag">${f.tag}</span>
      <span class="ts">${f.timestamp_sec.toFixed(2)}s</span>
      <div class="canvas-wrap"><canvas></canvas></div>
      <div class="legend"></div>`;
    const canvas = card.querySelector("canvas");
    const legend = card.querySelector(".legend");
    drawFrame(canvas, legend, f);
    els.framesGrid.appendChild(card);
  });
}

function drawFrame(canvas, legend, frame) {
  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onload = () => {
    const W = img.naturalWidth;
    const H = img.naturalHeight;
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0);

    // Persons (dashed white).
    (frame.persons || []).forEach((p) => {
      drawBox(ctx, p.bbox, W, H, "#ffffff", 2, true, p.person_id || "person");
    });
    // Objects (color-coded by ownership).
    (frame.objects || []).forEach((o) => {
      const color = COLORS[o.ownership] || "#3b82f6";
      drawBox(ctx, o.bbox, W, H, color, 2, false, `${o.label}${o.instance_id ? "·" + o.instance_id.split("_").slice(-1)[0] : ""}`);
    });
  };
  img.onerror = () => {
    canvas.width = 320; canvas.height = 240;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#222"; ctx.fillRect(0, 0, 320, 240);
    ctx.fillStyle = "#888"; ctx.font = "12px monospace";
    ctx.fillText("frame missing", 8, 20);
    ctx.fillText(frame.frame_path || "(no path)", 8, 40);
  };
  if (frame.frame_path) {
    img.src = "/frames/" + encodeURI(frame.frame_path.replace(/^\/+/, ""));
  } else {
    img.onerror();
  }

  // Build legend chips for both persons and objects.
  legend.innerHTML = "";
  (frame.persons || []).forEach((p) => {
    legend.appendChild(chip("#ffffff", `${p.person_id || "person"}`, true));
  });
  (frame.objects || []).forEach((o) => {
    const c = COLORS[o.ownership] || "#3b82f6";
    legend.appendChild(chip(c, `${o.label}${o.instance_id ? '·' + o.instance_id : ''}${o.ownership ? ' (' + o.ownership + ')' : ''}`, false));
  });
}

function chip(color, text, dashed) {
  const span = document.createElement("span");
  span.style.cssText = `display:inline-flex;align-items:center;gap:4px;padding:1px 6px;border-radius:8px;background:rgba(255,255,255,0.05);border:1px ${dashed ? "dashed" : "solid"} ${color};`;
  span.innerHTML = `<i style="display:inline-block;width:8px;height:8px;background:${color};border-radius:2px"></i>${escapeHtml(text)}`;
  return span;
}

function drawBox(ctx, bbox, W, H, color, lineWidth, dashed, label) {
  const x = bbox.x_min * W;
  const y = bbox.y_min * H;
  const w = (bbox.x_max - bbox.x_min) * W;
  const h = (bbox.y_max - bbox.y_min) * H;
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  if (dashed) ctx.setLineDash([6, 4]);
  ctx.strokeRect(x, y, w, h);
  ctx.setLineDash([]);
  // label
  ctx.font = "12px monospace";
  const padding = 3;
  const tw = ctx.measureText(label).width + padding * 2;
  ctx.fillStyle = color;
  ctx.fillRect(x, Math.max(0, y - 16), tw, 16);
  ctx.fillStyle = "black";
  ctx.fillText(label, x + padding, Math.max(12, y - 4));
  ctx.restore();
}

function renderObjectTable(r) {
  els.objectsTbody.innerHTML = "";
  let count = 0;
  r.frames.forEach((f) => {
    f.objects.forEach((o) => {
      count++;
      const tr = document.createElement("tr");
      const overrideSel = document.createElement("select");
      ["", ...LABELS].forEach((lbl) => {
        const opt = document.createElement("option");
        opt.value = lbl;
        opt.textContent = lbl || "(no override)";
        const current = (o.instance_id && state.edits.object_overrides[o.instance_id]) || "";
        if (current === lbl) opt.selected = true;
        overrideSel.appendChild(opt);
      });
      overrideSel.onchange = () => {
        if (!o.instance_id) return;
        if (overrideSel.value) {
          state.edits.object_overrides[o.instance_id] = overrideSel.value;
        } else {
          delete state.edits.object_overrides[o.instance_id];
        }
      };
      tr.innerHTML = `
        <td>${f.tag}</td>
        <td><code>${escapeHtml(o.instance_id || '-')}</code></td>
        <td>${escapeHtml(o.label)}</td>
        <td>${o.score != null ? o.score.toFixed(2) : '-'}</td>
        <td><span class="label-pill ${o.ownership || 'UNLABELED'}">${o.ownership || '—'}</span></td>
        <td></td>
        <td><small style="color: var(--muted)">${escapeHtml((o.ownership_evidence || []).join(', '))}</small></td>`;
      tr.children[5].appendChild(overrideSel);
      els.objectsTbody.appendChild(tr);
    });
  });
  els.nObjects.textContent = count;
}

function renderRelations(r) {
  els.relationsList.innerHTML = "";
  let count = 0;
  r.frames.forEach((f) => {
    (f.relations || []).forEach((rel) => {
      count++;
      const li = document.createElement("li");
      li.innerHTML = `<small style="color: var(--muted)">${f.tag}</small> <code>${escapeHtml(rel.subject_id)}</code> — <b>${escapeHtml(rel.predicate)}</b> → <code>${escapeHtml(rel.object_id)}</code> ${rel.score != null ? '<small>(' + rel.score.toFixed(2) + ')</small>' : ''} ${rel.note ? '<small style="color: var(--muted)">— ' + escapeHtml(rel.note) + '</small>' : ''}`;
      els.relationsList.appendChild(li);
    });
  });
  els.nRelations.textContent = count;
}

function renderEdits(r) {
  els.editsList.innerHTML = "";
  (r.edits || []).slice(-30).reverse().forEach((e) => {
    const li = document.createElement("li");
    li.innerHTML = `<small style="color: var(--muted)">${escapeHtml(e.when || '')}</small> <b>${escapeHtml(e.annotator)}</b> ${escapeHtml(e.field)}: <code>${escapeHtml(e.old_value || '∅')}</code> → <code>${escapeHtml(e.new_value || '∅')}</code>`;
    els.editsList.appendChild(li);
  });
  els.nEdits.textContent = (r.edits || []).length;
}

async function saveActive() {
  if (!state.active) return;
  const annotator = els.annotator.value || "anonymous";
  const body = {
    annotator,
    scene_label: state.edits.scene_label,
    scene_taxonomy: state.edits.scene_taxonomy,
    review_status: state.edits.review_status,
    notes: state.edits.notes,
    object_overrides: state.edits.object_overrides,
  };
  els.saveFeedback.textContent = "saving…";
  els.saveFeedback.style.color = "var(--muted)";
  try {
    const updated = await fetchJSON(`/api/scenes/${encodeURIComponent(state.active.clip.clip_id)}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    state.active = updated;
    state.edits.object_overrides = {};
    els.saveFeedback.textContent = "saved ✓";
    els.saveFeedback.style.color = "var(--shared)";
    renderActive();
    refreshSidebar();
  } catch (e) {
    els.saveFeedback.textContent = "error: " + e.message;
    els.saveFeedback.style.color = "var(--danger)";
  }
}

async function refreshSidebar() {
  await loadScenes();
  await loadActivity();
  await loadStats();
}

async function loadActivity() {
  const items = await fetchJSON("/api/activity?limit=40");
  els.activityList.innerHTML = "";
  items.forEach((a) => {
    const li = document.createElement("li");
    const t = (a.ts || "").replace("T", " ").slice(0, 19);
    li.innerHTML = `<div class="who">${escapeHtml(a.annotator)}</div>
      <div><span class="clip">${escapeHtml(a.clip_id)}</span></div>
      <div><small>${escapeHtml(a.field)}: ${escapeHtml(a.old || '∅')} → ${escapeHtml(a.new || '∅')}</small></div>
      <div><small style="color: var(--muted)">${escapeHtml(t)}</small></div>`;
    els.activityList.appendChild(li);
  });
}

async function loadStats() {
  const s = await fetchJSON("/api/stats");
  const parts = [`total ${s.total}`];
  for (const [k, v] of Object.entries(s.by_status || {})) parts.push(`${k}:${v}`);
  els.stats.textContent = parts.join("  ·  ");
}

function escapeHtml(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

els.refresh.onclick = refreshSidebar;
[els.filterStatus, els.filterTaxonomy, els.filterLabel].forEach((sel) =>
  sel.addEventListener("change", () => loadScenes())
);
els.save.onclick = saveActive;

// Bootstrap.
refreshSidebar();
// Poll activity every 5s so collaborators see each other's edits.
setInterval(() => { loadActivity(); loadStats(); }, 5000);
