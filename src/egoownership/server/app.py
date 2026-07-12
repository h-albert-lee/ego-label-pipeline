"""FastAPI app for collaborative SceneRecord review.

Endpoints:

* ``GET  /``                       — annotator UI (HTML)
* ``GET  /api/scenes``             — list of scene summaries
* ``GET  /api/scenes/{clip_id}``   — full SceneRecord
* ``POST /api/scenes/{clip_id}``   — partial update (label / status / notes / object overrides)
* ``GET  /api/next-draft``         — next clip needing review (after a given clip_id)
* ``GET  /api/activity``           — recent activity log
* ``GET  /api/stats``              — counts by status / label / taxonomy
* ``GET  /frames/{path}``          — serves frame images from FRAMES_ROOT
* ``GET  /video/{video_id}``       — streams local video if present in VIDEOS_ROOT
* ``HEAD /video/{video_id}``       — quick existence check (UI uses this)

Multiple annotators are differentiated only by the ``annotator`` query/header
field; cookie-based auth is intentionally out of scope for the pilot.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from egoownership.schema import OwnershipLabel, Taxonomy
from egoownership.server.store import SceneStore

_STATIC_DIR = Path(__file__).parent / "static"

_VIDEO_EXTS = (".mp4", ".MP4", ".mkv", ".webm", ".mov")


class SceneUpdateBody(BaseModel):
    annotator: str = "anonymous"
    scene_label: OwnershipLabel | None = None
    scene_taxonomy: Taxonomy | None = None
    review_status: str | None = None
    notes: str | None = None
    object_overrides: dict[str, OwnershipLabel] | None = None
    # Optional: lets the review UI's single save action persist rationale/
    # selected_evidence together with the scene fields in one pass, instead
    # of a separate call to /evidence (see SceneStore.update).
    rationale: str | None = None
    selected_evidence: list[str] | None = None


class EvidenceUpdateBody(BaseModel):
    annotator: str = "anonymous"
    object_type: str | None = None
    target_zone: str | None = None
    relations: list[dict[str, Any]] | None = None
    rationale: str | None = None
    selected_evidence: list[str] | None = None


def _resolve_video(videos_root: Path | None, video_id: str) -> Path | None:
    # video_id may be a leading-slash-stripped absolute path (from auto-label pipeline).
    # Reconstruct as absolute by prepending "/" and check if it exists.
    abs_candidate = Path("/" + video_id)
    if abs_candidate.exists():
        return abs_candidate
    if videos_root is None:
        return None
    for ext in _VIDEO_EXTS:
        candidate = videos_root / f"{video_id}{ext}"
        if candidate.exists():
            return candidate
    return None


def create_app(
    scenes_path: Path,
    frames_root: Path,
    videos_root: Path | None = None,
) -> FastAPI:
    """Build a FastAPI app pointing at the given scenes JSONL, frame dir, and
    optional videos directory.
    """

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

    @app.get("/api/config")
    def runtime_config() -> dict[str, Any]:
        """Tell the frontend what the server has, so it can show/hide features."""
        return {
            "videos_available": videos_root is not None,
            "frames_root": str(frames_root),
        }

    @app.get("/api/scenes")
    def list_scenes(
        status: str | None = None,
        taxonomy: str | None = None,
        label: str | None = None,
        dataset: str | None = None,
        vlm_agreement: str | None = None,  # agree | disagree | no_data
        sort: str = "default",  # default | confidence-asc | confidence-desc
    ) -> list[dict[str, Any]]:
        rows = store.list_summaries()
        if status:
            rows = [r for r in rows if r["review_status"] == status]
        if taxonomy:
            rows = [r for r in rows if r["taxonomy"] == taxonomy]
        if label:
            rows = [r for r in rows if r["scene_label"] == label]
        if dataset:
            rows = [r for r in rows if r["dataset"] == dataset]
        if vlm_agreement == "agree":
            rows = [r for r in rows if r["vlm_agrees"] is True]
        elif vlm_agreement == "disagree":
            rows = [r for r in rows if r["vlm_agrees"] is False]
        elif vlm_agreement == "no_data":
            rows = [r for r in rows if not r["has_vlm_judgement"]]
        if sort == "confidence-asc":
            rows.sort(key=lambda r: (r["auto_label_confidence"] is None, r["auto_label_confidence"] or 0))
        elif sort == "confidence-desc":
            rows.sort(key=lambda r: -(r["auto_label_confidence"] or 0))
        return rows

    @app.get("/api/datasets")
    def list_datasets() -> list[str]:
        """Distinct dataset values currently in the store, for the UI's filter dropdown."""
        return sorted({r["dataset"] for r in store.list_summaries() if r.get("dataset")})

    @app.get("/api/scenes/{clip_id:path}")
    def get_scene(clip_id: str) -> dict[str, Any]:
        rec = store.get(clip_id)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"clip {clip_id!r} not found")
        return rec.model_dump(mode="json")

    @app.post("/api/scenes/{clip_id}/evidence")
    @app.post("/api/scenes/{clip_id:path}/evidence")
    def update_evidence(clip_id: str, body: EvidenceUpdateBody) -> dict[str, Any]:
        updated = store.update_evidence(
            clip_id,
            annotator=body.annotator,
            object_type=body.object_type,
            target_zone=body.target_zone,
            relations=body.relations,
            rationale=body.rationale,
            selected_evidence=body.selected_evidence,
        )
        if updated is None:
            raise HTTPException(status_code=404, detail=f"clip {clip_id!r} not found")
        return updated.model_dump(mode="json")

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
            rationale=body.rationale,
            selected_evidence=body.selected_evidence,
        )
        if updated is None:
            raise HTTPException(status_code=404, detail=f"clip {clip_id!r} not found")
        return updated.model_dump(mode="json")

    @app.get("/api/next-draft")
    def next_draft(after: str | None = None) -> dict[str, Any]:
        """Return the next clip that still needs review.

        Order: clips with status in {"draft", "in_review"} sorted by ascending
        ``auto_label_confidence`` (low confidence first — those need humans
        most). When ``after`` is provided, returns the first such clip *after*
        the given clip_id in that order.
        """
        rows = store.list_summaries()
        pending = [r for r in rows if r["review_status"] in {"draft", "in_review"}]
        pending.sort(key=lambda r: (r["auto_label_confidence"] is None, r["auto_label_confidence"] or 0))
        if not pending:
            return {"clip_id": None, "remaining": 0}
        if after is None:
            return {"clip_id": pending[0]["clip_id"], "remaining": len(pending)}
        ids = [r["clip_id"] for r in pending]
        try:
            idx = ids.index(after)
        except ValueError:
            return {"clip_id": pending[0]["clip_id"], "remaining": len(pending)}
        nxt = pending[(idx + 1) % len(pending)] if len(pending) > 1 else None
        return {
            "clip_id": nxt["clip_id"] if nxt else None,
            "remaining": len(pending),
        }

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

    @app.head("/video/{video_id:path}")
    def video_head(video_id: str) -> Response:
        path = _resolve_video(videos_root, video_id)
        if path is None:
            return Response(status_code=404)
        return Response(status_code=200, headers={"Content-Length": str(path.stat().st_size)})

    @app.get("/video/{video_id:path}")
    def video_get(video_id: str, request: Request):
        """Serve a local video, supporting HTTP Range so the <video> element
        can seek to t-2 / t-1 / t timestamps without downloading the whole file.
        """
        path = _resolve_video(videos_root, video_id)
        if path is None:
            raise HTTPException(status_code=404, detail="video not found")
        size = path.stat().st_size
        range_header = request.headers.get("range")
        if range_header is None:
            return FileResponse(path, media_type="video/mp4")

        # Parse "bytes=START-END"
        try:
            units, rng = range_header.split("=", 1)
            start_s, end_s = rng.split("-", 1)
            start = int(start_s)
            end = int(end_s) if end_s else size - 1
        except ValueError:
            raise HTTPException(status_code=416, detail="invalid range")
        end = min(end, size - 1)
        chunk = end - start + 1

        def iter_chunk():
            with path.open("rb") as f:
                f.seek(start)
                remaining = chunk
                while remaining > 0:
                    buf = f.read(min(64 * 1024, remaining))
                    if not buf:
                        break
                    remaining -= len(buf)
                    yield buf

        from fastapi.responses import StreamingResponse

        return StreamingResponse(
            iter_chunk(),
            status_code=206,
            media_type="video/mp4",
            headers={
                "Accept-Ranges": "bytes",
                "Content-Range": f"bytes {start}-{end}/{size}",
                "Content-Length": str(chunk),
            },
        )

    return app
