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
  filterStatus: $("filter-status"),
  filterTaxonomy: $("filter-taxonomy"),
  filterLabel: $("filter-label"),
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
  await refreshSidebar();
}

// ---- scenes list ----
async function loadScenes() {
  const params = new URLSearchParams();
  if (els.filterStatus.value) params.set("status", els.filterStatus.value);
  if (els.filterTaxonomy.value) params.set("taxonomy", els.filterTaxonomy.value);
  if (els.filterLabel.value) params.set("label", els.filterLabel.value);
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
    const head = await fetch(`/video/${encodeURIComponent(state.active.clip.video_id)}`, { method: "HEAD" });
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
  const conf = r.auto_label_confidence;
  els.confBadge.textContent = conf != null ? `auto ${conf.toFixed(2)}` : "";
  els.confBadge.className = `badge conf-badge ${conf != null && conf < 0.5 ? "low" : ""}`;
  els.dirtyBadge.hidden = !state.dirty;
  els.lastSaved.textContent = state.lastSaved ? `last save · ${state.lastSaved}` : "";

  const verbNoun = `${r.clip.verb || '—'} · ${(r.clip.nouns || []).join(', ') || '—'}`;
  els.verbNoun.textContent = verbNoun;
  els.autoLabelDisplay.innerHTML = `auto: <span class="label-pill ${r.scene_label || 'UNLABELED'}">${r.scene_label || '—'}</span> ${escapeHtml(r.notes || '')}`;
  els.narration.textContent = r.clip.narration || "";

  renderSegButtons(els.sceneLabelButtons, LABELS, state.edits.scene_label, (v) => setSceneLabel(v));
  renderSegButtons(els.taxonomyButtons, TAXONOMIES, state.edits.scene_taxonomy, (v) => { state.edits.scene_taxonomy = v; markDirty(); renderActive(); });
  renderSegButtons(els.statusButtons, STATUSES, state.edits.review_status, (v) => { state.edits.review_status = v; markDirty(); renderActive(); });
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
  const url = `/video/${encodeURIComponent(r.clip.video_id)}`;
  if (els.clipVideo.src.split("/").pop() !== encodeURIComponent(r.clip.video_id)) {
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
      const ownColor = COLORS[o.ownership] || "#3b82f6";
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
      const c = COLORS[o.ownership] || "#3b82f6";
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
    const evidenceList = ["t-2", "t-1", "t"]
      .map((tag) => g.byTag[tag])
      .filter(Boolean)
      .flatMap((d) => (d.ownership_evidence || []).map((e) => `${e}`))
      .filter((v, i, a) => a.indexOf(v) === i)
      .join(" · ");

    const dotsHtml = ["t-2", "t-1", "t"].map((tag) => {
      const d = g.byTag[tag];
      const cls = d ? (d.ownership || "AMBIGUOUS") : "absent";
      return `<span class="frame-dot ${cls}" title="${tag} · ${d ? d.ownership || '—' : 'not detected'}">${tag.replace("t", "t").replace("-", "−")}</span>`;
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
      ${evidenceList ? `<div class="instance-evidence">${escapeHtml(evidenceList)}</div>` : ""}
    `;
    const sel = document.createElement("select");
    ["", ...LABELS].forEach((lbl) => {
      const opt = document.createElement("option");
      opt.value = lbl;
      opt.textContent = lbl || "(keep auto)";
      if ((overrideLabel || "") === lbl) opt.selected = true;
      sel.appendChild(opt);
    });
    sel.onchange = () => {
      if (sel.value) state.edits.object_overrides[g.id] = sel.value;
      else delete state.edits.object_overrides[g.id];
      markDirty();
      renderInstances(r);
    };
    card.querySelector(".instance-override").appendChild(sel);
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
[els.filterStatus, els.filterTaxonomy, els.filterLabel, els.filterSort].forEach((sel) =>
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
