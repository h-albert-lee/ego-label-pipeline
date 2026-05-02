"""HD-EPIC object-movement-track parser.

HD-EPIC ships per-video JSON files like::

    {
      "video_id": "P01_01",
      "fps": 60.0,
      "tracks": [
        {
          "track_id": 15,
          "object": "notebook",
          "movement_type": "transfer" | "pickup" | "place",
          "start_frame": 12340,
          "end_frame": 13200,
          "bboxes": [...]   # optional, one per frame
        },
        ...
      ]
    }

Field names vary slightly between releases — the parser is defensive.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from egoownership.config import normalize_token
from egoownership.schema import ClipCandidate, Taxonomy


def _to_sec(frame: Any, fps: float) -> float:
    if fps and isinstance(frame, (int, float)):
        return float(frame) / float(fps)
    return 0.0


def _iter_one_file(path: Path) -> Iterator[ClipCandidate]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    video_id = data.get("video_id") or path.stem
    fps = float(data.get("fps") or 0) or 60.0
    for track in data.get("tracks", []) or []:
        start = track.get("start_frame", track.get("start"))
        end = track.get("end_frame", track.get("end"))
        if start is None or end is None or end <= start:
            continue
        mid = (start + end) / 2.0
        obj = track.get("object") or track.get("noun") or track.get("label") or ""
        mv = track.get("movement_type") or track.get("verb") or ""
        yield ClipCandidate(
            dataset="hd_epic",
            clip_id=f"{video_id}:track_{track.get('track_id', start)}",
            video_id=video_id,
            taxonomy=Taxonomy.CONTEXTUAL,
            t_minus_2_sec=_to_sec(start, fps),
            t_minus_1_sec=_to_sec(mid, fps),
            t_sec=_to_sec(end, fps),
            verb=normalize_token(mv) if isinstance(mv, str) else None,
            nouns=[normalize_token(obj)] if obj else [],
            narration=None,
            source={
                "track_id": track.get("track_id"),
                "start_frame": start,
                "end_frame": end,
                "fps": fps,
            },
        )


def iter_hd_epic_candidates(annotations_path: Path) -> Iterator[ClipCandidate]:
    """Accept either a single JSON file or a directory of per-video JSONs."""

    p = Path(annotations_path)
    if p.is_file():
        yield from _iter_one_file(p)
        return
    for child in sorted(p.rglob("*.json")):
        yield from _iter_one_file(child)
