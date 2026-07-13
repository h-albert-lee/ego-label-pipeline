"""labels_v2 → labels_v3 post-hoc relabel (cascade guideline-v2-2026-07-09).

Path B (no local ego4d detections JSONL): reproduce the patched cascade's
decision *change* on the inline structured evidence, using the real module
helpers (narration_actor, _is_communal) — NOT re-implementing them — so this
stays in sync with src/egoownership/detection/ownership.py.

The patch (commit 087708d) changes exactly two things vs the old cascade:
  1. held_by hand was unconditionally the wearer's → MINE. Now a bare hand
     with a third-person narration actor uses the 2x2 rule
     (communal→SHARED, personal→PERSON_k).
  2. communal-function nouns held/zoned/depth-near stay SHARED (persistence).

Only rows the patch would flip are changed; every changed row is stamped
relabel="guideline-v2-2026-07-09-posthoc" with the fired rule. The original
label is preserved in auto_ground_truth_v2.

Reconstruction note: the inline evidence is a *summary*; the raw detector's
bare "hand" was already collapsed to "wearer" by the old cascade, so we treat
frame-t held_by=="wearer" as the hand-attribution proxy. Depth/cy-derived MINE
(held_by None) is left unchanged unless the noun is communal — faithful to the
patch, which does not add a narration gate to those rules.
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")
from egoownership.detection.ownership import (  # noqa: E402
    CASCADE_VERSION,
    _is_communal,
    narration_actor,
)

IN = Path("outputs/ego4d/labels_v2.jsonl")
OUT = Path("outputs/ego4d/labels_v3.jsonl")
TAG = f"{CASCADE_VERSION}-posthoc"


def held_at_t(rec: dict) -> str | None:
    snaps = ((rec.get("evidence") or {}).get("temporal") or {}).get("frame_snapshots") or {}
    return (snaps.get("t") or {}).get("held_by")


def relabel(rec: dict) -> tuple[str, str | None]:
    """Return (new_label, fired_rule|None)."""
    old = rec.get("auto_ground_truth") or ""
    obj_label = (rec.get("object") or {}).get("label") or ""
    communal = _is_communal(obj_label)
    actor = narration_actor(rec.get("dense_caption_en"))
    held = held_at_t(rec)
    tz = (rec.get("evidence") or {}).get("target_zone")
    held_is_hand = held in ("wearer", "hand") or (isinstance(held, str) and held.startswith("hand"))

    if old == "MINE":
        if held_is_hand:
            if actor == "other":
                if communal:
                    return "SHARED", "held_by:hand+third-party-actor+communal-persistence"
                return "PERSON_k", "held_by:hand+third-party-actor(2x2)"
            if communal:
                return "SHARED", "held_by:wearer+communal-persistence"
        elif communal:
            return "SHARED", "mine-rule+communal-persistence"
    elif old == "PERSON_k":
        if communal and (tz == "other_person_zone" or (isinstance(held, str) and held.startswith("person_"))):
            return "SHARED", "person-rule+communal-persistence"
    return old, None


def main() -> None:
    n = 0
    changed = 0
    move = Counter()      # (old -> new)
    by_rule = Counter()
    dist_v2 = Counter()
    dist_v3 = Counter()
    with IN.open(encoding="utf-8") as fin, OUT.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            n += 1
            old = rec.get("auto_ground_truth") or ""
            dist_v2[old] += 1
            new, rule = relabel(rec)
            if rule is not None and new != old:
                rec["auto_ground_truth_v2"] = old
                rec["auto_ground_truth"] = new
                rec["relabel"] = TAG
                rec["relabel_rule"] = rule
                changed += 1
                move[(old, new)] += 1
                by_rule[rule] += 1
            dist_v3[rec.get("auto_ground_truth")] += 1
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    labels = ["MINE", "PERSON_k", "SHARED", "AMBIGUOUS"]
    print(f"총 {n}개, 변경 {changed}개 ({changed/n:.1%})\n")
    print("## v2 → v3 라벨 분포")
    print(f"{'label':<12}{'v2':>7}{'v3':>7}{'delta':>8}")
    for L in labels:
        print(f"{L:<12}{dist_v2.get(L,0):>7}{dist_v3.get(L,0):>7}{dist_v3.get(L,0)-dist_v2.get(L,0):>+8}")
    print("\n## 변경 방향 행렬 (old → new)")
    for (o, nw), c in move.most_common():
        print(f"  {o:<10} → {nw:<10} : {c}")
    print("\n## 규칙별 변경 건수")
    for r, c in by_rule.most_common():
        print(f"  {c:>5}  {r}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
