"""FastAPI app for collaborative SceneRecord review.

Endpoints:

* ``GET  /``                       — annotator UI (HTML)
* ``GET  /api/scenes``             — list of scene summaries
* ``GET  /api/scenes/{clip_id}``   — full SceneRecord
* ``POST /api/scenes/{clip_id}``   — partial update (label / status / notes / object overrides)
* ``GET  /api/activity``           — recent activity log
* ``GET  /api/stats``              — counts by status / label / taxonomy
* ``GET  /frames/{path}``          — serves frame images from FRAMES_ROOT

Multiple annotators are differentiated only by the ``annotator`` query/header
field; cookie-based auth is intentionally out of scope for the pilot.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from egoownership.schema import OwnershipLabel, Taxonomy
from egoownership.server.store import SceneStore

_STATIC_DIR = Path(__file__).parent / "static"


class SceneUpdateBody(BaseModel):
    annotator: str = "anonymous"
    scene_label: OwnershipLabel | None = None
    scene_taxonomy: Taxonomy | None = None
    review_status: str | None = None  # "draft" | "in_review" | "verified" | "rejected"
    notes: str | None = None
    object_overrides: dict[str, OwnershipLabel] | None = None


def create_app(scenes_path: Path, frames_root: Path) -> FastAPI:
    """Build a FastAPI app pointing at the given scenes JSONL and frame dir."""

    store = SceneStore(scenes_path)
    app = FastAPI(title="Egocentric Implicit Ownership annotator")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    if _STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        index_html = _STATIC_DIR / "index.html"
        if index_html.exists():
            return index_html.read_text(encoding="utf-8")
        return "<h1>Annotator UI not bundled</h1>"

    @app.get("/api/scenes")
    def list_scenes(
        status: str | None = None,
        taxonomy: str | None = None,
        label: str | None = None,
    ) -> list[dict[str, Any]]:
        rows = store.list_summaries()
        if status:
            rows = [r for r in rows if r["review_status"] == status]
        if taxonomy:
            rows = [r for r in rows if r["taxonomy"] == taxonomy]
        if label:
            rows = [r for r in rows if r["scene_label"] == label]
        return rows

    @app.get("/api/scenes/{clip_id:path}")
    def get_scene(clip_id: str) -> dict[str, Any]:
        rec = store.get(clip_id)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"clip {clip_id!r} not found")
        return rec.model_dump(mode="json")

    @app.post("/api/scenes/{clip_id:path}")
    def update_scene(clip_id: str, body: SceneUpdateBody) -> dict[str, Any]:
        updated = store.update(
            clip_id,
            annotator=body.annotator,
            scene_label=body.scene_label,
            scene_taxonomy=body.scene_taxonomy,
            review_status=body.review_status,
            notes=body.notes,
            object_overrides=body.object_overrides,
        )
        if updated is None:
            raise HTTPException(status_code=404, detail=f"clip {clip_id!r} not found")
        return updated.model_dump(mode="json")

    @app.get("/api/activity")
    def activity(limit: int = Query(default=50, ge=1, le=500)) -> list[dict[str, Any]]:
        return store.recent_activity(limit)

    @app.get("/api/stats")
    def stats() -> dict[str, Any]:
        return store.stats()

    @app.get("/frames/{path:path}")
    def frame(path: str):
        full = (frames_root / path).resolve()
        try:
            full.relative_to(frames_root.resolve())
        except ValueError:
            raise HTTPException(status_code=400, detail="path escape")
        if not full.exists():
            raise HTTPException(status_code=404, detail="frame not found")
        return FileResponse(full)

    return app
