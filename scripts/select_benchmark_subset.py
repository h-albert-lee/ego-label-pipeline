#!/usr/bin/env python3
"""Information-value-based subset selection for the EgoOwn benchmark.

Reduces a large labels JSONL (e.g. Ego4D ~10k) to a curated eval subset +
an extended pool, WITHOUT random sampling:

Priority tiers (kept in this order until --target-size is reached):
  P0  rare-label items (PERSON_k / SHARED / AMBIGUOUS)          -> keep all
  P1  narration actor is a third person (multi-person signal)    -> keep all
  P2  MINE decided only by proximity rules (depth/cy evidence)    -> T2/boundary
      candidates, kept before easy MINE
  P3  remaining MINE (strong held_by:wearer evidence)             -> quota fill

Constraints applied within every tier:
  --max-per-video      cap clips per source video (near-duplicate control)
  (verb, noun) dedup   at most --max-per-combo per video+verb+noun combo

Non-selected rows are written to --pool-out with split="extended_pool"
(frozen candidate train split — do NOT eval on these).

Usage:
    python scripts/select_benchmark_subset.py \\
        --input outputs/ego4d/labels_v3.jsonl \\
        --out outputs/ego4d/labels_v3_eval.jsonl \\
        --pool-out outputs/ego4d/labels_v3_pool.jsonl \\
        --target-size 2500 --max-per-video 4
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

_RARE_LABELS = {"PERSON_k", "SHARED", "AMBIGUOUS"}
_PROXIMITY_EVIDENCE = re.compile(r"depth-near|>=mine_y_min")
_OTHER_SUBJECT_RE = re.compile(
    r"^\s*(?:#O\b|the\s+)?(?:man|woman|person|lady|guy|boy|girl)\s+[A-Z]\b",
    re.IGNORECASE,
)


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass


def _narration_other(rec: dict) -> bool:
    # Raw pipeline rows carry `narration`; HF-summary rows (labels_v2) carry the
    # caption under `dense_caption_en`. Read whichever is present.
    narr = (
        rec.get("narration")
        or (rec.get("clip") or {}).get("narration")
        or rec.get("dense_caption_en")
        or ""
    )
    if not isinstance(narr, str):
        return False
    return narr.lstrip().startswith("#O") or bool(_OTHER_SUBJECT_RE.search(narr))


def _evidence_blob(rec: dict) -> str:
    ev = rec.get("ownership_evidence") or rec.get("auto_key_evidence") or rec.get("evidence") or []
    if isinstance(ev, dict):
        return " ".join(str(v) for v in ev.values())
    if isinstance(ev, list):
        return " ".join(str(v) for v in ev)
    return str(ev)


def _held_by_t(rec: dict):
    """Frame-t held_by from the inline scene-record summary (labels_v2), if any."""
    snaps = ((rec.get("evidence") or {}).get("temporal") or {}).get("frame_snapshots") or {}
    return (snaps.get("t") or {}).get("held_by")


def _is_proximity_mine(rec: dict) -> bool:
    """MINE decided by a proximity/zone rule rather than a held_by:wearer grab.

    Prefer the raw cascade evidence strings when present; otherwise fall back
    to the HF summary, where a MINE with no frame-t held_by attribution could
    only have come from the depth/cy/zone rules (held_by:wearer would have
    stamped 'wearer').
    """
    blob = _evidence_blob(rec)
    if _PROXIMITY_EVIDENCE.search(blob):
        return True
    if "held_by:wearer" in blob or "held_by:person" in blob:
        return False
    return _held_by_t(rec) is None


def _tier(rec: dict) -> int:
    label = rec.get("auto_ground_truth") or rec.get("auto_label") or ""
    if label in _RARE_LABELS:
        return 0
    # P2 (proximity/boundary MINE) is the scarce, information-rich mining
    # target — check it before the generic P1 multi-person bucket, which is
    # non-discriminative on all-#O slices (every caption is third-person).
    if _is_proximity_mine(rec):
        return 2
    if _narration_other(rec):
        return 1
    return 3


# gt_confidence per tier: proximity/zone MINE (P2) is the low-confidence
# mining pool whose GT the judges should re-propose; explicit rare-label
# decisions (P0) and held_by:wearer MINE (P3) are firmer.
_TIER_CONFIDENCE = {0: "high", 1: "medium", 2: "low", 3: "medium"}


def _combo_key(rec: dict) -> tuple:
    clip = rec.get("clip") or {}
    verb = rec.get("verb") or clip.get("verb") or ""
    noun = (rec.get("object") or {}).get("label") or ""
    return (rec.get("video_id") or clip.get("video_id") or "", verb, noun)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="curated eval subset JSONL")
    ap.add_argument("--pool-out", type=Path, default=None,
                    help="extended pool JSONL (default: <out stem>_pool.jsonl)")
    ap.add_argument("--target-size", type=int, default=2500)
    ap.add_argument("--max-per-video", type=int, default=4)
    ap.add_argument("--max-per-combo", type=int, default=1,
                    help="max clips per (video, verb, noun) combo")
    args = ap.parse_args()

    records = list(_iter_jsonl(args.input))
    print(f"loaded {len(records)} records from {args.input}")

    # Tier everything, then fill greedily under per-video / per-combo caps.
    tiered: dict[int, list[dict]] = defaultdict(list)
    for rec in records:
        t = _tier(rec)
        rec["subset_tier"] = f"P{t}"
        rec["gt_confidence"] = _TIER_CONFIDENCE[t]
        tiered[t].append(rec)
    for t in sorted(tiered):
        labels = Counter((r.get("auto_ground_truth") or r.get("auto_label") or "?") for r in tiered[t])
        print(f"  tier P{t}: {len(tiered[t])} rows  {dict(labels)}")

    selected, pool = [], []
    per_video: Counter = Counter()
    per_combo: Counter = Counter()

    def try_take(rec: dict, *, exempt_caps: bool = False) -> bool:
        vid = rec.get("video_id") or (rec.get("clip") or {}).get("video_id") or ""
        combo = _combo_key(rec)
        if not exempt_caps:
            if per_video[vid] >= args.max_per_video:
                return False
            if per_combo[combo] >= args.max_per_combo:
                return False
        per_video[vid] += 1
        per_combo[combo] += 1
        selected.append(rec)
        return True

    # P0/P1: keep all — rare labels & multi-person items are the scarce
    # resource; caps don't apply (they're already scarce).
    for t in (0, 1):
        for rec in tiered[t]:
            try_take(rec, exempt_caps=True)

    # P2 then P3: fill up to target under caps.
    for t in (2, 3):
        for rec in tiered[t]:
            if len(selected) >= args.target_size:
                pool.append(rec)
            elif not try_take(rec):
                pool.append(rec)

    # Anything past target in earlier loop iterations was appended to pool.
    sel_ids = {id(r) for r in selected}
    pool = [r for r in records if id(r) not in sel_ids]

    for rec in selected:
        rec["split"] = "eval"
    for rec in pool:
        rec["split"] = "extended_pool"

    pool_out = args.pool_out or args.out.with_name(args.out.stem + "_pool.jsonl")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for rec in selected:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    with pool_out.open("w", encoding="utf-8") as fh:
        for rec in pool:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    sel_labels = Counter((r.get("auto_ground_truth") or r.get("auto_label") or "?") for r in selected)
    n_videos = len({r.get("video_id") or (r.get("clip") or {}).get("video_id") for r in selected})
    print(f"\nselected {len(selected)} rows ({n_videos} videos) -> {args.out}")
    print(f"  label dist: {dict(sel_labels)}")
    print(f"extended pool {len(pool)} rows -> {pool_out}")
    print("NOTE: extended_pool is the frozen train-split candidate — never eval on it.")


if __name__ == "__main__":
    main()
