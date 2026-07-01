"""Ego4D FHO (Forecasting Hands and Objects) parser.

The canonical annotation file is ``fho_main.json``. Its top-level schema is::

    {
      "clips": [
        {
          "clip_uid": "...",
          "video_uid": "...",
          "parent_video_metadata": { ... },
          "annotations": [
            {
              "pre_frame":  {"frame": int, "frame_number": int, "time": float,
                             "boxes": [...]},
              "pnr_frame":  {"frame": int, "time": float, "boxes": [...]},
              "post_frame": {"frame": int, "time": float, "boxes": [...]},
              "verb":   "put_down" | "give" | ...,
              "objects": [{"object_type": "noun_label", ...}],
              "narration_text": "...",
              ...
            },
            ...
          ]
        },
        ...
      ]
    }

We stream with ``ijson`` if available (the file is multi-GB), falling back to
``json`` for small fixtures. Each annotation becomes one ``ClipCandidate`` with
``t_minus_2 = pre``, ``t_minus_1 = pnr``, ``t = post``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from egoownership.config import normalize_token
from egoownership.schema import ClipCandidate, Taxonomy


def _safe_get_time(frame_dict: Any, default: float) -> float:
    if not isinstance(frame_dict, dict):
        return default
    for key in ("clip_time", "video_time", "time", "pts_time"):
        v = frame_dict.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    # Fallback: derive from frame number if fps is there.
    frame_no = frame_dict.get("frame_number") or frame_dict.get("frame")
    fps = frame_dict.get("fps") or frame_dict.get("frame_rate")
    if isinstance(frame_no, (int, float)) and isinstance(fps, (int, float)) and fps > 0:
        return float(frame_no) / float(fps)
    return default


def _extract_nouns(ann: dict) -> list[str]:
    nouns: list[str] = []
    for obj in ann.get("objects", []) or []:
        if isinstance(obj, dict):
            for key in ("object_type", "noun", "noun_label", "label"):
                if key in obj and obj[key]:
                    nouns.append(normalize_token(str(obj[key])))
                    break
    # Also look at optional top-level noun_label (varies by FHO version).
    for key in ("noun_label", "noun", "nouns"):
        v = ann.get(key)
        if isinstance(v, str):
            nouns.append(normalize_token(v))
        elif isinstance(v, list):
            nouns.extend(normalize_token(str(n)) for n in v if n)
    return [n for n in nouns if n]


def _ann_to_candidate(clip_uid: str, video_uid: str | None, ann: dict) -> ClipCandidate | None:
    pre = ann.get("pre_frame") or ann.get("pre_45_frame") or ann.get("pre")
    pnr = ann.get("pnr_frame") or ann.get("pnr") or ann.get("contact_frame")
    post = ann.get("post_frame") or ann.get("post")
    if not (pre and pnr and post):
        return None

    # If we only have frame numbers we still want monotone timestamps; seed with
    # indices so the record is preserved even when times are missing.
    t_pre = _safe_get_time(pre, default=0.0)
    t_pnr = _safe_get_time(pnr, default=t_pre + 0.5)
    t_post = _safe_get_time(post, default=t_pnr + 0.5)

    verb = ann.get("verb") or ann.get("verb_label")
    verb_norm = normalize_token(verb) if isinstance(verb, str) else None
    narration = ann.get("narration_text") or ann.get("narration")

    ann_key = ann.get("unique_id") or ann.get("annotation_uid") or f"{t_pre:.3f}"
    return ClipCandidate(
        dataset="ego4d_fho",
        clip_id=f"{clip_uid}:{ann_key}",
        video_id=video_uid,
        taxonomy=Taxonomy.CONTEXTUAL,  # FHO pre/PNR/post is inherently contextual
        t_minus_2_sec=t_pre,
        t_minus_1_sec=t_pnr,
        t_sec=t_post,
        verb=verb_norm,
        nouns=_extract_nouns(ann),
        narration=narration,
        source={"clip_uid": clip_uid, "video_uid": video_uid},
    )


def iter_fho_candidates(annotations_path: Path) -> Iterator[ClipCandidate]:
    """Yield ``ClipCandidate`` objects from an ``fho_main.json`` file.

    Uses ``ijson`` when available to avoid loading the whole file into memory.
    Falls back to ``json.load`` for small fixtures.
    """
    
    annotations_path = Path(annotations_path)
    try:
        import ijson  # type: ignore

        with annotations_path.open("rb") as f:
            for clip in ijson.items(f, "clips.item"):
                clip_uid = clip.get("clip_uid") or clip.get("clip_id") or ""
                video_uid = clip.get("video_uid") or clip.get("video_id")
                for ann in clip.get("annotations", []) or []:
                    cand = _ann_to_candidate(clip_uid, video_uid, ann)
                    if cand is not None:
                        yield cand
    except ImportError:
        with annotations_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for clip in (data.get("clips") or data.get("videos") or []):
            clip_uid = clip.get("clip_uid") or clip.get("clip_id") or ""
            video_uid = clip.get("video_uid") or clip.get("video_id")
            for ann in clip.get("annotations", []) or []:
                cand = _ann_to_candidate(clip_uid, video_uid, ann)
                if cand is not None:
                    yield cand
