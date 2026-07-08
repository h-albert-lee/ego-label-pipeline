// Annotator UI for the Egocentric Implicit Ownership benchmark.
// Vanilla JS. Designed around a "label → save → next" loop with keyboard
// shortcuts so an annotator can work for an hour without leaving the home row.

const LABELS = ["MINE", "PERSON_k", "SHARED", "AMBIGUOUS"];
const TAXONOMIES = ["A", "B", "C", "D"];
const STATUSES = ["draft", "in_review", "verified", "rejected"];
const COLORS = {
  MINE: "#f59e0b",
  PERSON_k: "#a855f7",
  SHARED: "#22c55e",
  AMBIGUOUS: "#94a3b8",
};
const INSTANCE_PALETTE = ["#60a5fa", "#f472b6", "#34d399", "#fbbf24", "#a78bfa", "#fb923c", "#22d3ee", "#f87171"];

const state = {
  config: { videos_available: false },
  scenes: [],
  active: null,
  edits: { scene_label: null, scene_taxonomy: null, review_status: null, notes: null, object_overrides: {} },
  show: { zones: false, relations: false, persons: true },
  videoPresent: false,
  lastSaved: null,
  dirty: false,
};

// ---- DOM cache ----
const $ = (id) => document.getElementById(id);
const els = {
  list: $("scene-list"),
  empty: $("empty-state"),
  detail: $("scene-detail"),
  clipId: $("clip-id"),
  datasetBadge: $("dataset-badge"),
  taxonomyBadge: $("taxonomy-badge"),
  statusBadge: $("status-badge"),
  confBadge: $("conf-badge"),
  dirtyBadge: $("dirty-badge"),
  lastSaved: $("last-saved"),
  verbNoun: $("verb-noun"),
  autoLabelDisplay: $("auto-label-display"),
  narration: $("narration"),
  framesGrid: $("frames-grid"),
  sceneLabelButtons: $("scene-label-buttons"),
  taxonomyButtons: $("taxonomy-buttons"),
  statusButtons: $("status-buttons"),
  evidencePanel: $("evidence-panel"),
  notes: $("notes"),
  save: $("save"),
  prev: $("prev"),
  next: $("next"),
  saveFeedback: $("save-feedback"),
  quickApprove: $("quick-approve"),
  quickReject: $("quick-reject"),
  instancesGrid: $("instances-grid"),
  nInstances: $("n-instances"),
  relationsList: $("relations-list"),
  nRelations: $("n-relations"),
  editsList: $("edits-list"),
  nEdits: $("n-edits"),
  activityList: $("activity-list"),
  refresh: null,
  filterDataset: $("filter-dataset"),
  filterStatus: $("filter-status"),
  filterTaxonomy: $("filter-taxonomy"),
  filterLabel: $("filter-label"),
  filterVlmAgreement: $("filter-vlm-agreement"),
  filterSort: $("filter-sort"),
  search: $("search"),
  annotator: $("annotator"),
  autoAdvance: $("auto-advance"),
  showZones: $("show-zones"),
  showRelations: $("show-relations"),
  showPersons: $("show-persons"),
  videoBlock: $("video-block"),
  clipVideo: $("clip-video"),
  videoTime: $("video-time"),
  progressText: $("progress-text"),
  progressFill: $("progress-fill"),
  kbdHint: $("kbd-hint"),
  zoomModal: $("zoom-modal"),
  zoomCanvas: $("zoom-canvas"),
  zoomTitle: $("zoom-title"),
  zoomClose: $("zoom-close"),
  helpModal: $("help-modal"),
  helpClose: $("help-close"),
  statsStatus: $("stats-status"),
  statsLabel: $("stats-label"),
  statsTaxonomy: $("stats-taxonomy"),
  statsVlm: $("stats-vlm"),
  vlmBadge: $("vlm-badge"),
  vlmPanel: $("vlm-panel"),
};

// ---- preference persistence ----
els.annotator.value = localStorage.getItem("egoown.annotator") || "";
els.annotator.addEventListener("change", () => localStorage.setItem("egoown.annotator", els.annotator.value || ""));
els.autoAdvance.checked = localStorage.getItem("egoown.autoAdvance") !== "0";
els.autoAdvance.addEventListener("change", () => localStorage.setItem("egoown.autoAdvance", els.autoAdvance.checked ? "1" : "0"));

// ---- networking ----
async function fetchJSON(url, opts = {}) {
  const res = await fetch(url, opts);
  if (!res.ok) throw new Error(`${url} → ${res.status}`);
  return res.json();
}

async function bootstrap() {
  try {
    state.config = await fetchJSON("/api/config");
  } catch (e) {
    console.warn("config failed", e);
  }
  await loadDatasets();
  await refreshSidebar();
}

// ---- scenes list ----
async function loadDatasets() {
  try {
    const datasets = await fetchJSON("/api/datasets");
    if (datasets.length <= 1) return; // nothing to filter — single dataset (or none) being served
    const current = els.filterDataset.value;
    els.filterDataset.innerHTML = `<option value="">all</option>` +
      datasets.map((d) => `<option value="${escapeHtml(d)}">${escapeHtml(d)}</option>`).join("");
    els.filterDataset.value = current;
  } catch (e) {
    console.warn("datasets failed", e);
  }
}

async function loadScenes() {
  const params = new URLSearchParams();
  if (els.filterDataset.value) params.set("dataset", els.filterDataset.value);
  if (els.filterStatus.value) params.set("status", els.filterStatus.value);
  if (els.filterTaxonomy.value) params.set("taxonomy", els.filterTaxonomy.value);
  if (els.filterLabel.value) params.set("label", els.filterLabel.value);
  if (els.filterVlmAgreement.value) params.set("vlm_agreement", els.filterVlmAgreement.value);
  if (els.filterSort.value) params.set("sort", els.filterSort.value);
  state.scenes = await fetchJSON(`/api/scenes?${params}`);
  renderScenes();
}

function renderScenes() {
  const q = (els.search.value || "").trim().toLowerCase();
  els.list.innerHTML = "";
  state.scenes
    .filter((s) => {
      if (!q) return true;
      const blob = `${s.clip_id} ${s.verb || ""} ${(s.nouns || []).join(" ")}`.toLowerCase();
      return blob.includes(q);
    })
    .forEach((s) => {
      const li = document.createElement("li");
      if (state.active && state.active.clip.clip_id === s.clip_id) li.classList.add("active");
      const conf = s.auto_label_confidence;
      const confClass = conf == null ? "" : conf < 0.4 ? "low" : conf < 0.75 ? "med" : "high";
      const vlmPill = s.has_vlm_judgement
        ? `<span class="vlm-pill ${s.vlm_agrees ? 'agree' : 'disagree'}" title="VLM cross-check ${s.vlm_agrees ? 'agrees' : 'disagrees'}">${s.vlm_agrees ? '✓' : '✗'} vlm</span>`
        : '';
      li.innerHTML = `
        <div class="row">
          <span class="clip-id" title="${escapeHtml(s.clip_id)}">${escapeHtml(s.clip_id)}</span>
          <span class="status-pill ${s.review_status}">${s.review_status}</span>
        </div>
        <div class="row">
          <small>${escapeHtml(s.dataset)} · ${escapeHtml(s.verb || '—')}</small>
          <span class="label-pill ${s.scene_label || 'UNLABELED'}">${s.scene_label || '—'}</span>
        </div>
        <div class="row">
          <small>tax ${s.taxonomy} · ${s.n_objects} objs${s.n_edits ? ' · ' + s.n_edits + ' edits' : ''}</small>
          ${vlmPill}
          <span class="conf-mini ${confClass}">${conf != null ? conf.toFixed(2) : '—'}</span>
        </div>`;
      li.onclick = () => loadScene(s.clip_id);
      els.list.appendChild(li);
    });
}

// ---- active scene ----
async function loadScene(clipId, { focus = true } = {}) {
  state.active = await fetchJSON(`/api/scenes/${encodeURIComponent(clipId)}`);
  state.edits = {
    scene_label: state.active.scene_label,
    scene_taxonomy: state.active.scene_taxonomy || state.active.clip.taxonomy,
    review_status: state.active.review_status,
    notes: state.active.notes || "",
    object_overrides: {},
  };
  state.dirty = false;
  state.lastSaved = null;
  await checkVideoAvailability();
  renderActive();
  renderScenes();
  if (focus) window.scrollTo({ top: 0, behavior: "smooth" });
}

async function checkVideoAvailability() {
  state.videoPresent = false;
  if (!state.config.videos_available || !state.active?.clip?.video_id) return;
  try {
    const head = await fetch(`/video/${state.active.clip.video_id}`, { method: "HEAD" });
    state.videoPresent = head.ok;
  } catch {
    state.videoPresent = false;
  }
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
  els.statusBadge.className = `badge status-badge ${state.edits.review_status}`;
  els.confBadge.hidden = true;
  els.vlmBadge.hidden = !r.vlm_majority_label;
  if (r.vlm_majority_label) {
    const vlmAgrees = r.vlm_majority_label === r.scene_label;
    els.vlmBadge.textContent = vlmAgrees ? "VLM: agrees" : "VLM: disagrees";
    els.vlmBadge.className = `badge vlm-badge ${vlmAgrees ? "agree" : "disagree"}`;
  }
  els.dirtyBadge.hidden = !state.dirty;
  els.lastSaved.textContent = state.lastSaved ? `last save · ${state.lastSaved}` : "";

  const verbNoun = `${r.clip.verb || '—'} · ${(r.clip.nouns || []).join(', ') || '—'}`;
  els.verbNoun.textContent = verbNoun;
  els.autoLabelDisplay.innerHTML = `auto: <span class="label-pill ${r.scene_label || 'UNLABELED'}">${r.scene_label || '—'}</span> ${escapeHtml(r.notes || '')}`;
  els.narration.textContent = r.clip.narration || "";

  renderSegButtons(els.sceneLabelButtons, LABELS, state.edits.scene_label, (v) => setSceneLabel(v));
  renderSegButtons(els.taxonomyButtons, TAXONOMIES, state.edits.scene_taxonomy, (v) => { state.edits.scene_taxonomy = v; markDirty(); renderActive(); });
  renderSegButtons(els.statusButtons, STATUSES, state.edits.review_status, (v) => { state.edits.review_status = v; markDirty(); renderActive(); });
  renderEvidencePanel(r);
  renderVlmPanel(r);
  els.notes.value = state.edits.notes || "";
  els.notes.oninput = () => { state.edits.notes = els.notes.value; markDirty(); };

  els.showZones.checked = state.show.zones;
  els.showRelations.checked = state.show.relations;
  els.showPersons.checked = state.show.persons;

  renderVideoBlock(r);
  renderFrames(r);
  renderInstances(r);
  renderRelations(r);
  renderEdits(r);
}

function setSceneLabel(v) {
  state.edits.scene_label = v;
  markDirty();
  renderActive();
}
function markDirty() { state.dirty = true; els.dirtyBadge.hidden = false; }

function renderSegButtons(container, options, current, onSelect) {
  container.innerHTML = "";
  options.forEach((opt) => {
    const btn = document.createElement("button");
    btn.className = "seg-btn" + (opt === current ? ` active ${opt}` : "");
    btn.textContent = opt;
    btn.onclick = (e) => { e.preventDefault(); onSelect(opt); };
    container.appendChild(btn);
  });
}

// ---- video ----
function renderVideoBlock(r) {
  if (!state.videoPresent || !r.clip.video_id) {
    els.videoBlock.hidden = true;
    return;
  }
  els.videoBlock.hidden = false;
  const url = `/video/${r.clip.video_id}`;
  if (els.clipVideo.getAttribute("data-video-id") !== r.clip.video_id) {
    els.clipVideo.setAttribute("data-video-id", r.clip.video_id);
    els.clipVideo.src = url;
  }
  // Pre-seek to t-2 so the annotator sees the start of the clip.
  els.clipVideo.onloadedmetadata = () => {
    try { els.clipVideo.currentTime = r.clip.t_minus_2_sec; } catch {}
  };
  els.clipVideo.ontimeupdate = () => {
    els.videoTime.textContent = els.clipVideo.currentTime.toFixed(2) + "s";
  };
  els.videoBlock.querySelectorAll("[data-jump]").forEach((btn) => {
    btn.onclick = () => {
      const tag = btn.dataset.jump;
      const t = tag === "t-2" ? r.clip.t_minus_2_sec : tag === "t-1" ? r.clip.t_minus_1_sec : r.clip.t_sec;
      els.clipVideo.currentTime = t;
      els.clipVideo.play().catch(() => {});
    };
  });
}

// ---- frames + canvas drawing ----
function instanceColor(instanceId) {
  if (!instanceId) return "#3b82f6";
  let h = 0;
  for (let i = 0; i < instanceId.length; i++) h = (h * 31 + instanceId.charCodeAt(i)) >>> 0;
  return INSTANCE_PALETTE[h % INSTANCE_PALETTE.length];
}

function renderFrames(r) {
  els.framesGrid.innerHTML = "";
  r.frames.forEach((f, idx) => {
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
    card.onclick = () => openZoom(f);
    els.framesGrid.appendChild(card);
  });
}

function drawFrame(canvas, legend, frame) {
  const FALLBACK_W = 640;
  const FALLBACK_H = 480;

  const draw = (W, H, hasImage, ctx) => {
    if (state.show.zones && frame.zones) {
      drawZones(ctx, frame.zones, W, H);
    }
    if (state.show.persons) {
      (frame.persons || []).forEach((p) => {
        drawBox(ctx, p.bbox, W, H, "#ffffff", 2, true, p.person_id || "person");
      });
    }
    (frame.objects || []).forEach((o) => {
      const ownColor = COLORS[o.ownership] || "#6b7280";
      const insColor = instanceColor(o.instance_id);
      // Outer thin box = instance color, inner thick box = ownership color.
      drawBox(ctx, o.bbox, W, H, insColor, 1, false, "");
      drawBox(ctx, o.bbox, W, H, ownColor, 2, false, `${o.label}${o.instance_id ? "·" + (o.instance_id.split("_").pop() || "") : ""}`);
    });
    if (state.show.relations) {
      drawRelations(ctx, frame, W, H);
    }
  };

  const renderLegend = () => {
    legend.innerHTML = "";
    (frame.persons || []).forEach((p) => legend.appendChild(chip("#ffffff", `${p.person_id || "person"}`, true)));
    (frame.objects || []).forEach((o) => {
      const c = COLORS[o.ownership] || "#6b7280";
      const labelStr = `${o.label}${o.instance_id ? '·' + o.instance_id : ''}${o.ownership ? ' ' + o.ownership : ''}`;
      legend.appendChild(chip(c, labelStr, false));
    });
  };

  const drawPlaceholder = () => {
    canvas.width = FALLBACK_W; canvas.height = FALLBACK_H;
    const ctx = canvas.getContext("2d");
    ctx.fillStyle = "#101418"; ctx.fillRect(0, 0, FALLBACK_W, FALLBACK_H);
    // grid for context
    ctx.strokeStyle = "#1f262f"; ctx.lineWidth = 1;
    for (let i = 1; i < 10; i++) {
      ctx.beginPath(); ctx.moveTo((FALLBACK_W * i) / 10, 0); ctx.lineTo((FALLBACK_W * i) / 10, FALLBACK_H); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(0, (FALLBACK_H * i) / 10); ctx.lineTo(FALLBACK_W, (FALLBACK_H * i) / 10); ctx.stroke();
    }
    ctx.fillStyle = "#3a4452"; ctx.font = "16px ui-monospace, monospace";
    ctx.fillText("frame image not available · bboxes shown to scale", 12, 22);
    if (frame.frame_path) {
      ctx.fillStyle = "#576475"; ctx.font = "11px ui-monospace, monospace";
      ctx.fillText(frame.frame_path, 12, 40);
    }
    draw(FALLBACK_W, FALLBACK_H, false, ctx);
    renderLegend();
  };

  if (!frame.frame_path) {
    drawPlaceholder();
    return;
  }

  const img = new Image();
  img.crossOrigin = "anonymous";
  img.onload = () => {
    const W = img.naturalWidth;
    const H = img.naturalHeight;
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0);
    draw(W, H, true, ctx);
    renderLegend();
  };
  img.onerror = drawPlaceholder;
  img.src = "/frames/" + encodeURI(frame.frame_path.replace(/^\/+/, ""));
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
  if (label) {
    ctx.font = "12px ui-monospace, monospace";
    const padding = 3;
    const tw = ctx.measureText(label).width + padding * 2;
    ctx.fillStyle = color;
    ctx.fillRect(x, Math.max(0, y - 16), tw, 16);
    ctx.fillStyle = "#0e1116";
    ctx.fillText(label, x + padding, Math.max(12, y - 4));
  }
  ctx.restore();
}

function drawZones(ctx, zones, W, H) {
  ctx.save();
  // wearer near zone (bottom)
  ctx.fillStyle = "rgba(245,158,11,0.10)";
  ctx.fillRect(0, zones.mine_y_min * H, W, H - zones.mine_y_min * H);
  ctx.strokeStyle = "rgba(245,158,11,0.6)";
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(0, zones.mine_y_min * H);
  ctx.lineTo(W, zones.mine_y_min * H);
  ctx.stroke();
  ctx.fillStyle = "rgba(245,158,11,0.7)";
  ctx.font = "11px ui-monospace, monospace";
  ctx.fillText("MINE near zone", 6, zones.mine_y_min * H + 14);

  // shared band (vertical strip)
  ctx.setLineDash([]);
  ctx.fillStyle = "rgba(34,197,94,0.10)";
  ctx.fillRect(zones.shared_x_min * W, 0, (zones.shared_x_max - zones.shared_x_min) * W, H);
  ctx.strokeStyle = "rgba(34,197,94,0.6)";
  ctx.setLineDash([4, 4]);
  [zones.shared_x_min, zones.shared_x_max].forEach((x) => {
    ctx.beginPath(); ctx.moveTo(x * W, 0); ctx.lineTo(x * W, H); ctx.stroke();
  });
  ctx.fillStyle = "rgba(34,197,94,0.8)";
  ctx.fillText("SHARED band", zones.shared_x_min * W + 4, 14);

  // person influence zones (top)
  Object.entries(zones.person_zones || {}).forEach(([pid, b]) => {
    ctx.fillStyle = "rgba(168,85,247,0.10)";
    ctx.fillRect(b.x_min * W, b.y_min * H, (b.x_max - b.x_min) * W, (b.y_max - b.y_min) * H);
    ctx.strokeStyle = "rgba(168,85,247,0.7)";
    ctx.setLineDash([4, 4]);
    ctx.strokeRect(b.x_min * W, b.y_min * H, (b.x_max - b.x_min) * W, (b.y_max - b.y_min) * H);
    ctx.fillStyle = "rgba(168,85,247,0.9)";
    ctx.fillText(`zone:${pid}`, b.x_min * W + 4, b.y_min * H + 14);
  });

  ctx.setLineDash([]);
  ctx.restore();
}

function drawRelations(ctx, frame, W, H) {
  const idToBbox = {};
  (frame.objects || []).forEach((o) => { if (o.instance_id) idToBbox[o.instance_id] = o.bbox; });
  (frame.persons || []).forEach((p) => { if (p.person_id) idToBbox[p.person_id] = p.bbox; });
  ctx.save();
  ctx.lineWidth = 1.5;
  (frame.relations || []).forEach((rel) => {
    const a = idToBbox[rel.subject_id];
    const b = idToBbox[rel.object_id];
    if (!a || !b) return;
    const ax = (a.x_min + a.x_max) / 2 * W;
    const ay = (a.y_min + a.y_max) / 2 * H;
    const bx = (b.x_min + b.x_max) / 2 * W;
    const by = (b.y_min + b.y_max) / 2 * H;
    ctx.strokeStyle = rel.predicate === "held_by" ? "rgba(168,85,247,0.85)"
      : rel.predicate === "next_to" ? "rgba(96,165,250,0.85)"
      : "rgba(255,255,255,0.5)";
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
    ctx.fillStyle = ctx.strokeStyle;
    ctx.font = "10px ui-monospace, monospace";
    ctx.fillText(rel.predicate, (ax + bx) / 2, (ay + by) / 2 - 4);
  });
  ctx.restore();
}

function chip(color, text, dashed) {
  const span = document.createElement("span");
  span.className = "chip";
  span.style.cssText = `border:1px ${dashed ? "dashed" : "solid"} ${color};`;
  span.innerHTML = `<i style="background:${color}"></i>${escapeHtml(text)}`;
  return span;
}

const OBJECT_TYPES = ["personal", "shared", "generic_object"];
const TARGET_ZONES = ["ego_zone", "other_person_zone", "shared_zone", "center_table", "background_or_ambiguous_zone"];

// The 4 aspects the VLM judge shares with the auto evidence panel (it has no
// equivalent of "1. caption"), keyed by the field name both sides use.
// Ordered to match the auto evidence panel's numbering (2-5) so the two
// panels line up row-for-row when eyeballed side-by-side.
const VLM_SHARED_ASPECTS = [
  ["relation_graph", "relation_graph_evidence"],
  ["object_type", "object_type_evidence"],
  ["zone", "zone_evidence"],
  ["context_change", "context_change_evidence"],
];

function vlmCompositeKey(modelId, shortKey) { return `vlm::${modelId}::${shortKey}`; }

function vlmEvidenceTextForKey(r, compositeKey) {
  const [, modelId, shortKey] = compositeKey.split("::");
  const judge = (r.vlm_judgements || {})[modelId];
  const field = (VLM_SHARED_ASPECTS.find(([k]) => k === shortKey) || [])[1];
  return (judge && field && judge[field]) || "";
}

// Default rationale suggestion: for each aspect where a VLM judge's overall
// label *agrees* with the auto label and both sides actually wrote something
// for that aspect, treat it as corroborated by two independent modes and
// pre-select it (both the auto checkbox and the matching VLM checkbox).
// Only applies when nobody has already curated a selection for this row —
// an existing kev.selected_evidence means a human already made this call.
function computeDualSupportedDefault(r) {
  const kev = r.auto_key_evidence || {};
  if ((kev.selected_evidence || []).length) return null;
  const autoKeys = [];
  const vlmKeys = [];
  const sentences = [];
  Object.entries(r.vlm_judgements || {}).forEach(([modelId, judge]) => {
    if (judge.agrees !== true) return;
    VLM_SHARED_ASPECTS.forEach(([shortKey, field]) => {
      const autoText = kev[field];
      const vlmText = judge[field];
      if (autoText && vlmText) {
        autoKeys.push(shortKey);
        vlmKeys.push(vlmCompositeKey(modelId, shortKey));
        sentences.push(autoText, vlmText);
      }
    });
  });
  if (!sentences.length) return null;
  return { autoKeys, vlmKeys, text: Array.from(new Set(sentences)).join(" ") };
}

function renderEvidencePanel(r) {
  const kev = r.auto_key_evidence || {};
  if (!els.evidencePanel) return;

  const evidenceRows = [
    ["caption", "1. caption", kev.caption_evidence || "—", false],
    ["relation_graph", "2. relation graph", relationGraphControl(kev), true],
    ["object_type", "3. object type", objectTypeControl(kev), true],
    ["zone", "4. zone information", zoneControl(kev), true],
    ["context_change", "5. context change", kev.context_change_evidence || "—", false],
  ];
  const suggestion = computeDualSupportedDefault(r);
  const selected = new Set(kev.selected_evidence && kev.selected_evidence.length ? kev.selected_evidence : (suggestion ? suggestion.autoKeys : []));
  const initialRationale = (kev.selected_evidence && kev.selected_evidence.length) || !suggestion ? (kev.rationale || "") : suggestion.text;

  els.evidencePanel.innerHTML = `
    <div class="evidence-rationale">
      <div class="evidence-title">Auto rationale for GT choice${suggestion ? ' <small class="hint">(defaulted to evidence both auto + VLM support)</small>' : ""}</div>
      <textarea class="evidence-rationale-text" rows="4">${escapeHtml(initialRationale)}</textarea>
      <div class="evidence-actions">
        <button type="button" class="add-selected-evidence">add selected evidence</button>
        <button type="button" class="save-rationale">save rationale</button>
        <span class="evidence-save-feedback"></span>
      </div>
    </div>
    <div class="evidence-list">
      ${evidenceRows.map(([key, title, body, isHtml]) => `
        <div class="evidence-item">
          <label class="evidence-check">
            <input type="checkbox" data-evidence-key="${escapeHtml(key)}"${selected.has(key) ? " checked" : ""} />
            <span>${escapeHtml(title)}</span>
          </label>
          <div class="evidence-body">${isHtml ? body : escapeHtml(body)}</div>
        </div>
      `).join("")}
    </div>
  `;

  const rationaleText = els.evidencePanel.querySelector(".evidence-rationale-text");
  const feedback = els.evidencePanel.querySelector(".evidence-save-feedback");
  const selectedKeys = () => [
    ...Array.from(els.evidencePanel.querySelectorAll("[data-evidence-key]:checked")).map((x) => x.dataset.evidenceKey),
    ...Array.from(els.vlmPanel ? els.vlmPanel.querySelectorAll("[data-vlm-evidence-key]:checked") : []).map((x) => x.dataset.vlmEvidenceKey),
  ];

  els.evidencePanel.querySelector(".add-selected-evidence").onclick = () => {
    const additions = selectedKeys()
      .map((key) => (key.startsWith("vlm::") ? vlmEvidenceTextForKey(r, key) : evidenceTextForKey(kev, key)))
      .filter(Boolean);
    if (!additions.length) return;
    const current = (rationaleText.value || "").trim();
    const addition = additions.join(" ");
    rationaleText.value = current ? `${current} ${addition}` : addition;
  };

  els.evidencePanel.querySelector(".save-rationale").onclick = async () => {
    const payload = {
      annotator: els.annotator.value || "anonymous",
      rationale: rationaleText.value || "",
      selected_evidence: selectedKeys(),
    };
    feedback.textContent = "saving…";
    try {
      const updated = await fetchJSON(
        `/api/scenes/${encodeURIComponent(r.clip.clip_id)}/evidence`,
        { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }
      );
      state.active = updated;
      state.lastSaved = new Date().toLocaleTimeString();
      feedback.textContent = "saved";
      renderActive();
      refreshSidebar();
    } catch (e) {
      feedback.textContent = `error: ${e.message || e}`;
      console.error("rationale update failed", e);
    }
  };

  els.evidencePanel.querySelectorAll(".ev-select").forEach((sel) => {
    sel.onchange = async () => {
      const evKey = sel.dataset.evKey;
      const payload = { annotator: els.annotator.value || "anonymous", [evKey]: sel.value };
      try {
        const updated = await fetchJSON(
          `/api/scenes/${encodeURIComponent(r.clip.clip_id)}/evidence`,
          { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }
        );
        state.active = updated;
        state.lastSaved = new Date().toLocaleTimeString();
        renderActive();
        refreshSidebar();
      } catch (e) {
        console.error("evidence update failed", e);
      }
    };
  });

  const postRelations = async (newRelations) => {
    const payload = { annotator: els.annotator.value || "anonymous", relations: newRelations };
    try {
      const updated = await fetchJSON(
        `/api/scenes/${encodeURIComponent(r.clip.clip_id)}/evidence`,
        { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }
      );
      state.active = updated;
      state.lastSaved = new Date().toLocaleTimeString();
      renderActive();
      refreshSidebar();
    } catch (e) {
      console.error("relation update failed", e);
    }
  };

  els.evidencePanel.querySelectorAll(".relation-remove").forEach((btn) => {
    btn.onclick = () => postRelations((kev.relations || []).filter((_, i) => i !== Number(btn.dataset.relIdx)));
  });

  els.evidencePanel.querySelectorAll(".relation-reassign").forEach((sel) => {
    sel.onchange = () => {
      const idx = Number(sel.dataset.relIdx);
      if (sel.value === RELATION_REMOVE_SENTINEL) {
        postRelations((kev.relations || []).filter((_, i) => i !== idx));
        return;
      }
      const newRelations = (kev.relations || []).map((rel, i) =>
        i === idx ? { ...rel, object_id: sel.value, note: "manually corrected", score: null } : rel
      );
      postRelations(newRelations);
    };
  });
}

// Side-by-side with the auto evidence panel, not merged into it — the whole
// point of vlm-crosscheck is an *independent* second opinion, so reconciling
// disagreeing rationales is left to the human reviewer, not blended here.
// Row labels intentionally match the auto evidence panel's numbering (2-5;
// the VLM judge has no equivalent of "1. caption") so the two are easy to
// eyeball side-by-side.
const VLM_ASPECT_TITLES = {
  relation_graph: "2. relation graph",
  object_type: "3. object type",
  zone: "4. zone information",
  context_change: "5. context change",
};

function renderVlmPanel(r) {
  if (!els.vlmPanel) return;
  const judgements = r.vlm_judgements || {};
  const modelIds = Object.keys(judgements);
  if (!modelIds.length) {
    els.vlmPanel.innerHTML = `<div class="vlm-empty">No VLM cross-check judgement available for this clip yet.</div>`;
    return;
  }
  const kev = r.auto_key_evidence || {};
  const suggestion = computeDualSupportedDefault(r);
  const priorSelection = kev.selected_evidence || [];
  const selected = new Set(priorSelection.length ? priorSelection : (suggestion ? suggestion.vlmKeys : []));

  els.vlmPanel.innerHTML = modelIds.map((modelId) => {
    const j = judgements[modelId];
    const agreeClass = j.agrees == null ? "" : j.agrees ? "agree" : "disagree";
    const agreeText = j.agrees == null ? "no auto label to compare" : j.agrees ? "✓ agrees" : "✗ disagrees";
    const rows = VLM_SHARED_ASPECTS.map(([shortKey, field]) => {
      const compositeKey = vlmCompositeKey(modelId, shortKey);
      return `
        <div class="evidence-item">
          <label class="evidence-check">
            <input type="checkbox" data-vlm-evidence-key="${escapeHtml(compositeKey)}"${selected.has(compositeKey) ? " checked" : ""} />
            <span class="vlm-evidence-title">${escapeHtml(VLM_ASPECT_TITLES[shortKey])}</span>
          </label>
          <div class="evidence-body">${escapeHtml(j[field] || "—")}</div>
        </div>
      `;
    }).join("");
    const fallback = !j.object_type_evidence && !j.zone_evidence && !j.relation_graph_evidence && !j.context_change_evidence && j.rationale
      ? `<div class="evidence-item"><span class="vlm-evidence-title">raw response</span><div class="evidence-body">${escapeHtml(j.rationale)}</div></div>`
      : "";
    return `
      <div class="vlm-judge-card">
        <div class="vlm-judge-header">
          <span class="vlm-model-id">${escapeHtml(modelId)}</span>
          <span class="label-pill ${j.label || 'UNLABELED'}">${escapeHtml(j.label || '—')}</span>
          <span class="vlm-agree-pill ${agreeClass}">${agreeText}</span>
        </div>
        <div class="evidence-list">${rows}${fallback}</div>
      </div>
    `;
  }).join("");
}

function evidenceTextForKey(kev, key) {
  if (key === "caption") return kev.caption_evidence || "";
  if (key === "relation_graph") return kev.relation_graph_evidence || "";
  if (key === "object_type") return kev.object_type_evidence || (kev.object_type ? `The object type is ${kev.object_type}.` : "");
  if (key === "zone") return kev.zone_evidence || (kev.target_zone ? `The object is in ${kev.target_zone}.` : "");
  if (key === "context_change") return kev.context_change_evidence || "";
  return "";
}

function objectTypeControl(kev) {
  return `
    <select class="ev-select" data-ev-key="object_type">
      ${OBJECT_TYPES.map((t) => `<option value="${escapeHtml(t)}"${kev.object_type === t ? " selected" : ""}>${escapeHtml(t)}</option>`).join("")}
    </select>
    <span class="ev-inline-text">${escapeHtml(kev.object_type_evidence || "")}</span>
  `;
}

function zoneControl(kev) {
  return `
    <select class="ev-select" data-ev-key="target_zone">
      ${TARGET_ZONES.map((z) => `<option value="${escapeHtml(z)}"${kev.target_zone === z ? " selected" : ""}>${escapeHtml(z)}</option>`).join("")}
    </select>
    <span class="ev-inline-text">${escapeHtml(kev.zone_evidence || "")}</span>
  `;
}

function relationText(rel) {
  const parts = [`${rel.predicate || "related_to"} → ${rel.object_id || "unknown"}`];
  if (rel.note) parts.push(`(${rel.note})`);
  if (rel.score != null) parts.push(`score=${Number(rel.score).toFixed(2)}`);
  return parts.join(" ");
}

const RELATION_REMOVE_SENTINEL = "__remove__";

// Every holder id ever seen for this target across t-2/t-1/t, so a reviewer
// can reassign a held_by relation (e.g. person_5 -> wearer) instead of only
// being able to delete it and hope some other tier lands on the right answer.
function holderOptionsFor(kev) {
  const ids = new Set(["wearer"]);
  const snapshots = (kev.temporal && kev.temporal.frame_snapshots) || {};
  Object.values(snapshots).forEach((snap) => {
    (snap.persons || []).forEach((p) => { if (p.person_id) ids.add(p.person_id); });
  });
  return Array.from(ids);
}

function relationGraphControl(kev) {
  const relations = kev.relations || [];
  const summary = `<div class="ev-inline-text">${escapeHtml(kev.relation_graph_evidence || "No relations recorded.")}</div>`;
  if (!relations.length) return summary;
  const holderOptions = holderOptionsFor(kev);
  const rows = relations.map((rel, idx) => {
    const reassign = rel.predicate === "held_by" ? `
      <select class="relation-reassign" data-rel-idx="${idx}" title="Reassign this relation's holder, or remove it">
        ${holderOptions.map((id) => `<option value="${escapeHtml(id)}"${rel.object_id === id ? " selected" : ""}>${escapeHtml(id)}</option>`).join("")}
        <option value="${RELATION_REMOVE_SENTINEL}">— remove —</option>
      </select>
    ` : `
      <button type="button" class="relation-remove" data-rel-idx="${idx}" title="Remove this relation">remove</button>
    `;
    return `
      <div class="relation-row">
        <span class="relation-text">${escapeHtml(relationText(rel))}</span>
        ${reassign}
      </div>
    `;
  }).join("");
  return `${summary}<div class="relation-list">${rows}</div>`;
}

// ---- per-instance review panel ----
function renderInstances(r) {
  const groups = {};
  r.frames.forEach((f) => {
    f.objects.forEach((o) => {
      if (!o.instance_id) return;
      if (!groups[o.instance_id]) groups[o.instance_id] = { id: o.instance_id, label: o.label, byTag: {} };
      groups[o.instance_id].byTag[f.tag] = o;
    });
  });
  const list = Object.values(groups);
  els.instancesGrid.innerHTML = "";
  list.forEach((g) => {
    const card = document.createElement("div");
    card.className = "instance-card";
    const overrideLabel = state.edits.object_overrides[g.id];
    if (overrideLabel) card.classList.add("dirty");
    const finalDet = g.byTag["t"] || g.byTag["t-1"] || g.byTag["t-2"];
    const finalOwnership = overrideLabel || finalDet?.ownership || "—";

    const dotsHtml = ["t-2", "t-1", "t"].map((tag) => {
      const d = g.byTag[tag];
      const cls = d ? (d.ownership || "AMBIGUOUS") : "absent";
      return `<span class="frame-dot ${cls}" title="${tag} · ${d ? d.ownership || '—' : 'not detected'}">${tag.replace("-", "−")}</span>`;
    }).join("");

    card.innerHTML = `
      <div class="instance-head">
        <span class="instance-color" style="background:${instanceColor(g.id)}"></span>
        <span class="instance-id">${escapeHtml(g.id)}</span>
        <span class="instance-meta">${escapeHtml(g.label)}</span>
        <span class="label-pill ${finalOwnership in COLORS ? finalOwnership : 'UNLABELED'}">${escapeHtml(finalOwnership)}</span>
      </div>
      <div class="instance-frames">${dotsHtml}</div>
      <div class="instance-override"></div>
    `;

    const overrideSel = document.createElement("select");
    ["", ...LABELS].forEach((lbl) => {
      const opt = document.createElement("option");
      opt.value = lbl;
      opt.textContent = lbl || "(keep auto)";
      if ((overrideLabel || "") === lbl) opt.selected = true;
      overrideSel.appendChild(opt);
    });
    overrideSel.onchange = () => {
      if (overrideSel.value) state.edits.object_overrides[g.id] = overrideSel.value;
      else delete state.edits.object_overrides[g.id];
      markDirty();
      renderInstances(r);
    };
    card.querySelector(".instance-override").appendChild(overrideSel);

    els.instancesGrid.appendChild(card);
  });
  els.nInstances.textContent = list.length;
}

// ---- relations / edits ----
function renderRelations(r) {
  els.relationsList.innerHTML = "";
  let count = 0;
  r.frames.forEach((f) => {
    (f.relations || []).forEach((rel) => {
      count++;
      const li = document.createElement("li");
      li.innerHTML = `<small>${f.tag}</small> <code>${escapeHtml(rel.subject_id)}</code> — <b>${escapeHtml(rel.predicate)}</b> → <code>${escapeHtml(rel.object_id)}</code> ${rel.score != null ? '<small>(' + rel.score.toFixed(2) + ')</small>' : ''}${rel.note ? ' <small>· ' + escapeHtml(rel.note) + '</small>' : ''}`;
      els.relationsList.appendChild(li);
    });
  });
  els.nRelations.textContent = count;
}

function renderEdits(r) {
  els.editsList.innerHTML = "";
  (r.edits || []).slice(-30).reverse().forEach((e) => {
    const li = document.createElement("li");
    li.innerHTML = `<small>${escapeHtml((e.when || '').slice(0,19).replace('T',' '))}</small> <b>${escapeHtml(e.annotator)}</b> ${escapeHtml(e.field)}: <code>${escapeHtml(e.old_value || '∅')}</code> → <code>${escapeHtml(e.new_value || '∅')}</code>`;
    els.editsList.appendChild(li);
  });
  els.nEdits.textContent = (r.edits || []).length;
}

// ---- save / advance ----
async function saveActive({ advance = false } = {}) {
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
    state.dirty = false;
    state.lastSaved = new Date().toLocaleTimeString();
    els.saveFeedback.textContent = "saved ✓";
    els.saveFeedback.style.color = "var(--shared)";
    renderActive();
    refreshSidebar();
    if (advance && els.autoAdvance.checked) {
      await advanceToNext();
    }
  } catch (e) {
    els.saveFeedback.textContent = "error: " + e.message;
    els.saveFeedback.style.color = "var(--danger)";
  }
}

async function advanceToNext() {
  const after = state.active?.clip?.clip_id;
  try {
    const r = await fetchJSON(`/api/next-draft${after ? `?after=${encodeURIComponent(after)}` : ""}`);
    if (r.clip_id) await loadScene(r.clip_id);
    else {
      els.saveFeedback.textContent = "no more drafts — all reviewed 🎉";
      els.saveFeedback.style.color = "var(--shared)";
    }
  } catch (e) { console.warn(e); }
}

async function advanceToPrev() {
  const ids = state.scenes.map((s) => s.clip_id);
  const cur = state.active?.clip?.clip_id;
  if (!ids.length) return;
  const idx = cur ? ids.indexOf(cur) : 0;
  const target = ids[Math.max(0, idx - 1)];
  if (target) await loadScene(target);
}

async function quickApprove() {
  if (!state.active) return;
  // Approve = confirm auto-label + verified
  state.edits.scene_label = state.active.scene_label;
  state.edits.review_status = "verified";
  if (!state.edits.notes) state.edits.notes = "auto-label confirmed";
  markDirty();
  await saveActive({ advance: true });
}

async function quickReject() {
  if (!state.active) return;
  if (!confirm(`Reject ${state.active.clip.clip_id}? It will be excluded from the benchmark.`)) return;
  state.edits.review_status = "rejected";
  markDirty();
  await saveActive({ advance: true });
}

// ---- zoom modal ----
function openZoom(frame) {
  els.zoomTitle.textContent = `${frame.tag} · ${frame.timestamp_sec.toFixed(2)}s`;
  drawFrame(els.zoomCanvas, document.createElement("div"), frame);
  els.zoomModal.showModal();
}
els.zoomClose.onclick = () => els.zoomModal.close();
els.zoomModal.addEventListener("click", (e) => { if (e.target === els.zoomModal) els.zoomModal.close(); });

// ---- help modal ----
els.kbdHint.onclick = () => els.helpModal.showModal();
els.helpClose.onclick = () => els.helpModal.close();
els.helpModal.addEventListener("click", (e) => { if (e.target === els.helpModal) els.helpModal.close(); });

// ---- activity / stats ----
async function loadActivity() {
  const items = await fetchJSON("/api/activity?limit=40");
  els.activityList.innerHTML = "";
  items.forEach((a) => {
    const li = document.createElement("li");
    const t = (a.ts || "").replace("T", " ").slice(0, 19);
    li.innerHTML = `<div><span class="who">${escapeHtml(a.annotator)}</span> · <span class="clip">${escapeHtml(a.clip_id)}</span></div>
      <div><small>${escapeHtml(a.field)}: ${escapeHtml(a.old || '∅')} → ${escapeHtml(a.new || '∅')}</small></div>
      <div><small style="color: var(--muted)">${escapeHtml(t)}</small></div>`;
    els.activityList.appendChild(li);
  });
}

async function loadStats() {
  const s = await fetchJSON("/api/stats");
  // progress bar
  const verified = s.by_status?.verified || 0;
  const total = s.total || 0;
  const pct = total > 0 ? (verified / total) * 100 : 0;
  els.progressFill.style.width = `${pct}%`;
  const drafts = s.by_status?.draft || 0;
  els.progressText.innerHTML = `<strong>${verified}</strong> / ${total} verified · <strong>${drafts}</strong> drafts left`;

  // detailed stats pane
  const renderBucket = (host, bucket, classFn) => {
    host.innerHTML = "";
    Object.entries(bucket || {}).forEach(([k, v]) => {
      const row = document.createElement("div");
      row.className = "stat-row";
      row.innerHTML = `<span class="${classFn ? classFn(k) : ''}">${escapeHtml(k)}</span><span>${v}</span>`;
      host.appendChild(row);
    });
  };
  renderBucket(els.statsStatus, s.by_status, (k) => `status-pill ${k}`);
  renderBucket(els.statsLabel, s.by_label, (k) => `label-pill ${k}`);
  renderBucket(els.statsTaxonomy, s.by_taxonomy);
  renderBucket(els.statsVlm, s.by_vlm_agreement, (k) => `vlm-pill ${k === 'agree' ? 'agree' : k === 'disagree' ? 'disagree' : ''}`);
}

// tabs
document.querySelectorAll(".tab-btn").forEach((b) => {
  b.onclick = () => {
    document.querySelectorAll(".tab-btn").forEach((x) => x.classList.toggle("active", x === b));
    document.querySelectorAll(".tab-pane").forEach((p) => p.classList.toggle("active", p.id === b.dataset.tab + "-pane"));
  };
});

async function refreshSidebar() {
  await loadScenes();
  await loadActivity();
  await loadStats();
}

// ---- show toggles ----
els.showZones.onchange = () => { state.show.zones = els.showZones.checked; renderActive(); };
els.showRelations.onchange = () => { state.show.relations = els.showRelations.checked; renderActive(); };
els.showPersons.onchange = () => { state.show.persons = els.showPersons.checked; renderActive(); };

// ---- listeners ----
[els.filterDataset, els.filterStatus, els.filterTaxonomy, els.filterLabel, els.filterVlmAgreement, els.filterSort].forEach((sel) =>
  sel.addEventListener("change", () => loadScenes())
);
els.search.addEventListener("input", () => renderScenes());
els.save.onclick = () => saveActive({ advance: true });
els.next.onclick = () => advanceToNext();
els.prev.onclick = () => advanceToPrev();
els.quickApprove.onclick = quickApprove;
els.quickReject.onclick = quickReject;

// ---- keyboard shortcuts ----
function isTypingTarget(e) {
  const t = e.target;
  if (!t) return false;
  const tag = (t.tagName || "").toLowerCase();
  return tag === "input" || tag === "textarea" || tag === "select" || t.isContentEditable;
}
window.addEventListener("keydown", (e) => {
  // Cmd/Ctrl+S always saves.
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "s") {
    e.preventDefault();
    saveActive({ advance: true });
    return;
  }
  if (e.key === "Escape") {
    if (els.zoomModal.open) els.zoomModal.close();
    else if (els.helpModal.open) els.helpModal.close();
    else if (document.activeElement && document.activeElement.blur) document.activeElement.blur();
    return;
  }
  if (e.key === "?") { els.helpModal.showModal(); return; }
  if (isTypingTarget(e)) return;
  const k = e.key.toLowerCase();
  switch (k) {
    case "1": setSceneLabel("MINE"); break;
    case "2": setSceneLabel("PERSON_k"); break;
    case "3": setSceneLabel("SHARED"); break;
    case "4": setSceneLabel("AMBIGUOUS"); break;
    case "a": case "b": case "c": case "d":
      state.edits.scene_taxonomy = k.toUpperCase(); markDirty(); renderActive(); break;
    case "v": quickApprove(); break;
    case "x": quickReject(); break;
    case "j": case "n": advanceToNext(); break;
    case "k": advanceToPrev(); break;
    case "z": els.showZones.checked = state.show.zones = !state.show.zones; renderActive(); break;
    case "r": els.showRelations.checked = state.show.relations = !state.show.relations; renderActive(); break;
    case "f": e.preventDefault(); els.search.focus(); break;
  }
});

// ---- bootstrap ----
bootstrap();
setInterval(() => { loadActivity(); loadStats(); }, 5000);

// ---- helpers ----
function escapeHtml(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}
