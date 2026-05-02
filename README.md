# Egocentric Implicit Ownership — Benchmark Pipeline & Annotator

End-to-end scaffolding for inferring **implicit object ownership** from
first-person video using only visual + contextual cues. The pipeline surfaces
candidate scenes from **Ego4D FHO**, **EPIC-KITCHENS**, **HD-EPIC**, and
**EgoLife**; extracts sparse `(t-2, t-1, t)` frames; runs a multi-stage
detection + scene-graph stack; produces draft ownership labels; and serves a
**collaborative web annotator** for human verification.

## Taxonomy and label space

| Taxonomy | Name        | Description                                                                 |
|----------|-------------|-----------------------------------------------------------------------------|
| A        | Baseline    | Static shared-table scenes; MINE / SHARED / PERSON_k clear from single frame |
| B        | Conflict    | Visual cues disagree with context                                           |
| C        | Contextual  | Requires past frames (give / pass / put down / hand over)                   |
| D        | Ambiguous   | Symmetric / occluded / insufficient cues                                    |

Label space: `MINE`, `PERSON_k`, `SHARED`, `AMBIGUOUS`.

## Pipeline (top-down view)

```
filter → extract-frames → detect ─┬─ A. NATIVE: dataset-supplied bboxes (no GPU)
                                  ├─ B. LOCAL MODELS: DINO + RAM + SAM + Person
                                  │                  + Depth + BLIP-2 + scene graph
                                  └─ C. REMOTE VLM: Claude Opus 4.7 / GPT-4o
                                       (replaces RAM/BLIP-2; or scene judge)
        → label  ─ rule-based ownership over instance trajectories
                  ─ optional remote-VLM second opinion → vlm_judgement
                  ─ Taxonomy D auto-flag for symmetric duplicates
        → serve  ─ FastAPI + vanilla-JS UI for collaborative review
```

Three detect modes mix freely. Common path: A for fast smoke-test → B for the
full visual evidence stack → add C for attribute extraction or scene-level
second opinions.

Stages communicate via JSONL; every step is resumable.

## Install

```bash
pip install -e ".[frames,detect,serve,dev]"
```

Optional extras:
- `frames`: `imageio`, `opencv-python` for `ffmpeg`-based frame extraction
- `detect`: `torch`, `transformers` (Grounding DINO, SAM2, Depth Anything,
  BLIP-2, RAM)
- `serve`: `fastapi`, `uvicorn` for the annotator server
- `dev`: `pytest`, `ruff`, `httpx` for tests

## CLI

Installs as `egoown`:

```bash
egoown --help
egoown download {ego4d|epic|hd-epic} --out data/...

egoown filter ego4d-fho \
    --annotations data/ego4d/v2/annotations/fho_main.json \
    --taxonomy C \
    --out outputs/candidates_fho_C.jsonl

egoown extract-frames \
    --candidates outputs/candidates_fho_C.jsonl \
    --videos-root data/ego4d/videos \
    --out frames/

egoown detect \
    --candidates outputs/candidates_fho_C.jsonl \
    --frames frames/ \
    --out outputs/detections.jsonl \
    --use-ram --extract-attrs --estimate-depth

egoown label \
    --detections outputs/detections.jsonl \
    --out outputs/scene_records.jsonl

egoown serve \
    --scenes outputs/scene_records.jsonl \
    --frames-root frames/ \
    --host 0.0.0.0 --port 8000
```

## Collaborative annotator UI

`egoown serve` exposes:

- `GET /` — full HTML/JS annotator
- `GET /api/scenes` — list (filter by `status` / `taxonomy` / `label`)
- `GET /api/scenes/{clip_id}` — full SceneRecord
- `POST /api/scenes/{clip_id}` — partial update (label / status / notes /
  per-object override)
- `GET /api/activity` — recent edits across all annotators
- `GET /api/stats` — counts by status / label / taxonomy
- `GET /frames/{path}` — serves frames relative to `--frames-root`

The UI shows the three sparse frames side-by-side with bbox overlays
color-coded by ownership. Each detection can be overridden inline. Every edit
is appended to `scene_records.activity.jsonl` and surfaced live in the right
panel so multiple annotators see each other's work.

Identity is just a free-form `annotator` name (saved to `localStorage`); add
real auth in front of the server if you ship it.

## Project layout

```
src/egoownership/
    schema.py                 # SceneRecord, BBox, Person, Relation, etc.
    config.py                 # YAML verb/noun whitelists + zone thresholds
    filters.py                # Taxonomy-aware verb+noun filtering
    frames.py                 # ffmpeg / imageio frame extraction
    pipeline.py               # stage orchestration
    cli.py                    # typer entry point + serve command
    datasets/                 # Ego4D FHO, EPIC-KITCHENS, HD-EPIC, EgoLife
    download/                 # dataset-specific download helpers
    detection/
        grounding_dino.py     # text-prompted bbox proposals
        sam.py                # SAM2 mask refinement (bbox tightening)
        ram.py                # RAM (recognize-anything) bottom-up tags
        persons.py            # person detection + identity propagation
        tracking.py           # IoU + singleton-fallback instance tracking
        zones.py              # static / person-relative / depth-aware zones
        depth.py              # Depth Anything v2 wrapper
        attributes.py         # VLM-based per-object attribute extraction
        relations.py          # next_to / held_by / moved_to scene graph
        ownership.py          # rule cascade + scene-level labeling
    server/
        app.py                # FastAPI app
        store.py              # JSONL-backed store with file-locking
        entry.py              # uvicorn-reload entry point
        static/               # index.html + app.js + app.css
configs/
    taxonomy.yaml             # verb/noun whitelists, zone thresholds
tests/
    test_*.py                 # 37 unit + integration tests
```

## Data access notes

- **Ego4D** requires the license at <https://ego4d-data.org/>. The `download`
  command emits the official `ego4d` CLI invocations.
- **EPIC-KITCHENS** annotations: GitHub. Videos: University of Bristol.
- **HD-EPIC** annotations: project-site GitHub release.
- **EgoLife** annotations: project site; the adapter accepts a single JSON or
  a directory of per-clip JSONs.

## Testing

```bash
.venv/bin/pytest                # 37 tests (filters, parsers, tracking,
                                #  zones, ownership, relations, server, …)
```

The server tests use `fastapi.testclient` so no socket is required.

## Status

Heuristic ownership labeling is a starting point, not ground truth. Run the
pipeline → open the annotator → distribute the URL across the team.
