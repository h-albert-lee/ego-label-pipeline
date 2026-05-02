"""Dataset-native bbox path — convert annotation-supplied bboxes directly to
``FrameDetections`` without running any vision model.

Useful when GPU isn't available, or when you just want to spin up the human
review UI on real data quickly. Trade-off: only objects/hands the dataset
already labeled show up — no person detector, RAM, attributes, depth, or
scene-graph features (those still need the model path).

Supported sources today: Ego4D FHO (`pre_frame.boxes`, `pnr_frame.boxes`,
`post_frame.boxes`) and HD-EPIC (`tracks[*].bboxes` keyed by frame index).
EPIC-KITCHENS-100 doesn't ship per-frame object boxes in the action CSVs, so
its EPIC clips fall through here and get empty frames — use the model path
for EPIC.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from egoownership.config import normalize_token
from egoownership.schema import (
    BBox,
    ClipCandidate,
    FrameDetections,
    ObjectDetection,
)


# ---------- Ego4D FHO ----------


def _box_to_bbox(box: dict, default_w: int = 1, default_h: int = 1) -> BBox | None:
    """FHO boxes carry one of: (x, y, w, h) absolute, or normalized.

    The format varies across FHO releases. We accept any of:
      - ``{"x": ..., "y": ..., "width": ..., "height": ...}`` absolute pixels
      - ``{"x_min": ..., "y_min": ..., "x_max": ..., "y_max": ...}`` already 0..1
      - ``{"bbox": [x1, y1, x2, y2]}`` absolute pixels
    """
    if "x_min" in box and "x_max" in box:
        try:
            return BBox(
                x_min=float(box["x_min"]),
                y_min=float(box["y_min"]),
                x_max=float(box["x_max"]),
                y_max=float(box["y_max"]),
            )
        except Exception:  # noqa: BLE001
            return None
    if "bbox" in box and isinstance(box["bbox"], (list, tuple)) and len(box["bbox"]) == 4:
        x1, y1, x2, y2 = box["bbox"]
        w = box.get("image_width") or default_w
        h = box.get("image_height") or default_h
        return BBox.from_xyxy_abs(float(x1), float(y1), float(x2), float(y2), int(w), int(h))
    if {"x", "y", "width", "height"} <= set(box):
        x = float(box["x"])
        y = float(box["y"])
        w_box = float(box["width"])
        h_box = float(box["height"])
        w = box.get("image_width") or default_w
        h = box.get("image_height") or default_h
        return BBox.from_xyxy_abs(x, y, x + w_box, y + h_box, int(w), int(h))
    return None


def _detections_from_fho_frame(frame: dict) -> tuple[list[ObjectDetection], int | None, int | None]:
    """Pull (objects + image dims) out of one FHO pre/PNR/post frame dict."""
    boxes = frame.get("boxes") or frame.get("bbox_data") or []
    width = frame.get("image_width") or frame.get("width")
    height = frame.get("image_height") or frame.get("height")
    out: list[ObjectDetection] = []
    for raw in boxes:
        if not isinstance(raw, dict):
            continue
        bbox = _box_to_bbox(raw, default_w=width or 1, default_h=height or 1)
        if bbox is None:
            continue
        obj_label = (
            raw.get("object_type")
            or raw.get("label")
            or raw.get("name")
            or "object"
        )
        is_hand = bool(raw.get("is_hand")) or "hand" in str(obj_label).lower()
        out.append(
            ObjectDetection(
                label=normalize_token(str(obj_label)) if not is_hand else "hand",
                bbox=bbox,
                score=raw.get("score") or 1.0,
            )
        )
    return out, width, height


def _fho_frames_from_clip_annotation(ann: dict) -> dict[str, dict] | None:
    """Pull the (pre, PNR, post) frame triples out of one FHO annotation."""
    pre = ann.get("pre_frame") or ann.get("pre_45_frame") or ann.get("pre")
    pnr = ann.get("pnr_frame") or ann.get("pnr") or ann.get("contact_frame")
    post = ann.get("post_frame") or ann.get("post")
    if not (pre and pnr and post):
        return None
    return {"t-2": pre, "t-1": pnr, "t": post}


def detections_for_fho(
    fho_annotations_path: Path,
) -> dict[str, list[FrameDetections]]:
    """Build a clip_id → list[FrameDetections] map from an ``fho_main.json``.

    Streams via ``ijson`` if available; falls back to ``json.load`` for fixtures.
    """
    try:
        import ijson  # type: ignore

        return _detections_for_fho_streaming(fho_annotations_path, ijson)
    except ImportError:
        with Path(fho_annotations_path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        return _detections_for_fho_dict(data)


def _detections_for_fho_dict(data: dict) -> dict[str, list[FrameDetections]]:
    out: dict[str, list[FrameDetections]] = {}
    for clip in data.get("clips", []) or []:
        clip_uid = clip.get("clip_uid") or clip.get("clip_id") or ""
        for ann in clip.get("annotations", []) or []:
            triple = _fho_frames_from_clip_annotation(ann)
            if triple is None:
                continue
            ann_key = ann.get("unique_id") or ann.get("annotation_uid")
            if not ann_key:
                # Fall back to a deterministic key (matches parser).
                pre_t = triple["t-2"].get("clip_time") or triple["t-2"].get("time") or 0.0
                ann_key = f"{float(pre_t):.3f}"
            clip_id = f"{clip_uid}:{ann_key}"
            out[clip_id] = _frames_from_triple(triple)
    return out


def _detections_for_fho_streaming(path: Path, ijson) -> dict[str, list[FrameDetections]]:
    out: dict[str, list[FrameDetections]] = {}
    with Path(path).open("rb") as f:
        for clip in ijson.items(f, "clips.item"):
            clip_uid = clip.get("clip_uid") or clip.get("clip_id") or ""
            for ann in clip.get("annotations", []) or []:
                triple = _fho_frames_from_clip_annotation(ann)
                if triple is None:
                    continue
                ann_key = ann.get("unique_id") or ann.get("annotation_uid")
                if not ann_key:
                    pre_t = triple["t-2"].get("clip_time") or triple["t-2"].get("time") or 0.0
                    ann_key = f"{float(pre_t):.3f}"
                clip_id = f"{clip_uid}:{ann_key}"
                out[clip_id] = _frames_from_triple(triple)
    return out


def _frames_from_triple(triple: dict[str, dict]) -> list[FrameDetections]:
    frames: list[FrameDetections] = []
    for tag in ("t-2", "t-1", "t"):
        raw = triple[tag]
        objs, w, h = _detections_from_fho_frame(raw)
        t = (
            raw.get("clip_time")
            or raw.get("video_time")
            or raw.get("time")
            or raw.get("pts_time")
            or 0.0
        )
        frames.append(
            FrameDetections(
                tag=tag,  # type: ignore[arg-type]
                timestamp_sec=float(t),
                width=int(w) if w else None,
                height=int(h) if h else None,
                objects=objs,
            )
        )
    return frames


# ---------- HD-EPIC ----------


def detections_for_hd_epic(
    hd_epic_annotations_path: Path,
) -> dict[str, list[FrameDetections]]:
    """Build a clip_id → frames map from HD-EPIC movement-track JSONs.

    Each track in the source carries ``bboxes`` keyed (typically) by frame
    index. We sample three frames per track at start / midpoint / end.
    """
    p = Path(hd_epic_annotations_path)
    files = [p] if p.is_file() else sorted(p.rglob("*.json"))

    out: dict[str, list[FrameDetections]] = {}
    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            data = json.load(f)
        video_id = data.get("video_id") or fp.stem
        fps = float(data.get("fps") or 0) or 60.0
        for track in data.get("tracks", []) or []:
            start = track.get("start_frame", track.get("start"))
            end = track.get("end_frame", track.get("end"))
            if start is None or end is None or end <= start:
                continue
            mid = int((start + end) / 2)
            obj_label = normalize_token(
                str(track.get("object") or track.get("noun") or track.get("label") or "object")
            )
            bboxes = track.get("bboxes") or {}
            # `bboxes` is usually a {frame_index: [x1,y1,x2,y2]} or list-of-dicts.
            def _lookup(frame_idx: int) -> Any | None:
                if isinstance(bboxes, dict):
                    return bboxes.get(str(frame_idx)) or bboxes.get(frame_idx)
                if isinstance(bboxes, list):
                    for entry in bboxes:
                        if isinstance(entry, dict) and entry.get("frame") == frame_idx:
                            return entry
                return None

            frames: list[FrameDetections] = []
            for tag, fidx in (("t-2", start), ("t-1", mid), ("t", end)):
                raw = _lookup(fidx)
                objs: list[ObjectDetection] = []
                w = data.get("image_width") or data.get("width")
                h = data.get("image_height") or data.get("height")
                if isinstance(raw, (list, tuple)) and len(raw) == 4 and w and h:
                    x1, y1, x2, y2 = raw
                    objs.append(
                        ObjectDetection(
                            label=obj_label,
                            bbox=BBox.from_xyxy_abs(
                                float(x1), float(y1), float(x2), float(y2), int(w), int(h)
                            ),
                            score=1.0,
                        )
                    )
                elif isinstance(raw, dict):
                    bb = _box_to_bbox(raw, default_w=w or 1, default_h=h or 1)
                    if bb is not None:
                        objs.append(ObjectDetection(label=obj_label, bbox=bb, score=1.0))
                frames.append(
                    FrameDetections(
                        tag=tag,  # type: ignore[arg-type]
                        timestamp_sec=fidx / fps,
                        width=int(w) if w else None,
                        height=int(h) if h else None,
                        objects=objs,
                    )
                )
            clip_id = f"{video_id}:track_{track.get('track_id', start)}"
            out[clip_id] = frames
    return out


# ---------- pipeline entry point ----------


def stage_native_detect(
    candidates: Iterable[ClipCandidate],
    annotations_path: Path,
    dataset: str,
) -> Iterator[dict]:
    """Yield (clip + frames + empty depth_bands) records mimicking model detect.

    Use this in place of :func:`pipeline.stage_detect` when no vision models
    are available. Frames carry only objects (no persons/relations/attributes).
    """
    if dataset in {"ego4d-fho", "ego4d_fho"}:
        cache = detections_for_fho(Path(annotations_path))
    elif dataset in {"hd-epic", "hd_epic"}:
        cache = detections_for_hd_epic(Path(annotations_path))
    else:
        raise ValueError(
            f"Native bbox path not implemented for dataset={dataset!r}. "
            "Use the model path (egoown detect) for this dataset."
        )
    for cand in candidates:
        frames = cache.get(cand.clip_id)
        if frames is None:
            continue
        yield {
            "clip": cand.model_dump(mode="json"),
            "frames": [fd.model_dump(mode="json") for fd in frames],
            "depth_bands": [None] * len(frames),
        }
