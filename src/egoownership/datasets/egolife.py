"""EgoLife adapter.

EgoLife (CVPR 2025) records 6 participants over a week of cohabitation —
cooking, dining, meetings, etc. Annotations vary by release; the canonical
schema we target is a per-clip JSON like::

    {
      "clip_id": "day3_dinner_p1",
      "video_id": "day3_dinner",
      "participant": "p1",
      "scenario": "dining",
      "fps": 30,
      "events": [
        {
          "start_sec": 412.0,
          "end_sec": 414.5,
          "verb": "pass",
          "nouns": ["bowl"],
          "narration": "passes the rice bowl to p3",
          "transcript": "could you pass me the rice?",
          "av_caption": "p1 reaches across the table"
        },
        ...
      ]
    }

The adapter is defensive: missing fields are tolerated and the clip simply
gets fewer attributes. Each event becomes one ``ClipCandidate`` with
``t-2 = start``, ``t-1 = midpoint``, ``t = end``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from egoownership.config import normalize_token
from egoownership.schema import ClipCandidate, Taxonomy

_DINING_OR_MEETING = {"dining", "meeting", "dinner", "lunch", "breakfast", "discussion"}


def _scenario_taxonomy(scenario: str | None, verb: str | None) -> Taxonomy:
    if scenario and scenario.lower() in _DINING_OR_MEETING:
        # If there's an action verb, lean Contextual; otherwise Baseline.
        return Taxonomy.CONTEXTUAL if verb else Taxonomy.BASELINE
    return Taxonomy.CONTEXTUAL if verb else Taxonomy.BASELINE


def _iter_one_file(path: Path) -> Iterator[ClipCandidate]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "events" in data:
        clips = [data]
    elif isinstance(data, list):
        clips = data
    else:
        clips = [data]

    for clip in clips:
        clip_id_root = clip.get("clip_id") or path.stem
        video_id = clip.get("video_id") or clip_id_root
        scenario = clip.get("scenario")
        participant = clip.get("participant")
        for i, ev in enumerate(clip.get("events", []) or []):
            t0 = float(ev.get("start_sec", 0.0))
            t2 = float(ev.get("end_sec", t0 + 1.0))
            if t2 <= t0:
                continue
            t1 = (t0 + t2) / 2.0
            verb = ev.get("verb")
            nouns_raw = ev.get("nouns") or []
            if isinstance(nouns_raw, str):
                nouns_raw = [nouns_raw]
            nouns = [normalize_token(str(n)) for n in nouns_raw if n]
            narration_parts = [
                ev.get("narration"),
                ev.get("transcript"),
                ev.get("av_caption"),
            ]
            narration = " | ".join(p for p in narration_parts if p) or None
            yield ClipCandidate(
                dataset="egolife",
                clip_id=f"{clip_id_root}:event_{i}",
                video_id=video_id,
                taxonomy=_scenario_taxonomy(scenario, verb),
                t_minus_2_sec=t0,
                t_minus_1_sec=t1,
                t_sec=t2,
                verb=normalize_token(verb) if isinstance(verb, str) else None,
                nouns=nouns,
                narration=narration,
                source={
                    "scenario": scenario,
                    "participant": participant,
                    "transcript": ev.get("transcript"),
                    "av_caption": ev.get("av_caption"),
                },
            )


def iter_egolife_candidates(annotations_path: Path) -> Iterator[ClipCandidate]:
    """Accept either one EgoLife clip JSON or a directory of them."""
    p = Path(annotations_path)
    if p.is_file():
        yield from _iter_one_file(p)
    else:
        for child in sorted(p.rglob("*.json")):
            yield from _iter_one_file(child)
