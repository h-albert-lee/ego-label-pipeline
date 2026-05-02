"""EPIC-KITCHENS-100 parser.

EPIC ships action segments as CSVs with columns::

    participant_id, video_id, start_timestamp, stop_timestamp,
    start_frame, stop_frame, narration, verb, verb_class, noun,
    noun_class, all_nouns, all_noun_classes

``start_timestamp`` / ``stop_timestamp`` are ``HH:MM:SS.xx`` strings.

We map ``t-2 = start``, ``t-1 = midpoint``, ``t = stop``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from egoownership.config import normalize_token
from egoownership.schema import ClipCandidate, Taxonomy


def _parse_ts(ts: str) -> float:
    """Parse HH:MM:SS.xx -> seconds. Tolerant of NaNs / malformed rows."""
    if not isinstance(ts, str) or not ts:
        return 0.0
    parts = ts.split(":")
    try:
        parts_f = [float(p) for p in parts]
    except ValueError:
        return 0.0
    if len(parts_f) == 3:
        h, m, s = parts_f
    elif len(parts_f) == 2:
        h, m, s = 0.0, parts_f[0], parts_f[1]
    else:
        return 0.0
    return h * 3600.0 + m * 60.0 + s


def _parse_all_nouns(raw) -> list[str]:
    if not isinstance(raw, str) or not raw:
        return []
    # EPIC stores list-like strings: "['cup', 'tea']".
    s = raw.strip().strip("[]")
    out: list[str] = []
    for piece in s.split(","):
        piece = piece.strip().strip("'\"")
        if piece:
            out.append(normalize_token(piece))
    return out


def iter_epic_candidates(
    annotations_path: Path, default_taxonomy: Taxonomy = Taxonomy.CONTEXTUAL
) -> Iterator[ClipCandidate]:
    df = pd.read_csv(Path(annotations_path))
    for row in df.itertuples(index=False):
        rd = row._asdict() if hasattr(row, "_asdict") else dict(row._mapping)
        t0 = _parse_ts(rd.get("start_timestamp", ""))
        t2 = _parse_ts(rd.get("stop_timestamp", ""))
        if t2 <= t0:
            # Skip rows with degenerate windows.
            continue
        t1 = (t0 + t2) / 2.0
        verb = rd.get("verb")
        noun = rd.get("noun")
        nouns = _parse_all_nouns(rd.get("all_nouns"))
        if isinstance(noun, str) and noun:
            nouns = [normalize_token(noun), *nouns]
        yield ClipCandidate(
            dataset="epic_kitchens_100",
            clip_id=str(rd.get("narration_id") or f"{rd.get('video_id')}_{t0:.2f}"),
            video_id=rd.get("video_id"),
            taxonomy=default_taxonomy,
            t_minus_2_sec=t0,
            t_minus_1_sec=t1,
            t_sec=t2,
            verb=normalize_token(verb) if isinstance(verb, str) else None,
            nouns=nouns,
            narration=rd.get("narration") if isinstance(rd.get("narration"), str) else None,
            source={
                "participant_id": rd.get("participant_id"),
                "start_frame": rd.get("start_frame"),
                "stop_frame": rd.get("stop_frame"),
            },
        )
