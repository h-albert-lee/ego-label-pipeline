"""overlap(IAA + human baseline) 셋 — 검수 대상에서 라벨×taxonomy 층화 샘플.

제안 라벨·judge 정보를 숨긴 사본(raw 프레임+bbox만): scene_label=None,
notes="", review_status="draft". 검수자가 아무 힌트 없이 처음부터 라벨링 →
Cohen's kappa(주 검수자와의 일치) + human baseline 겸용.

결정론적 샘플(Math.random 없이 id 해시 정렬 상위 N)이라 재실행해도 동일.
"""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def stratum(rec):
    return (rec.get("scene_label"), rec["clip"].get("taxonomy"))


def hkey(rec):
    return hashlib.sha256(rec["clip"]["clip_id"].encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=Path("outputs/ego4d/review_scenerecords.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("outputs/ego4d/overlap_scenerecords.jsonl"))
    ap.add_argument("--frac", type=float, default=0.17)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input) if l.strip()]
    strata = defaultdict(list)
    for r in rows:
        strata[stratum(r)].append(r)

    picked = []
    for s, items in strata.items():
        items = sorted(items, key=hkey)          # 결정론적
        n = max(1, round(len(items) * args.frac))  # 층마다 최소 1개
        picked.extend(items[:n])

    # 숨김 사본
    blinded = []
    for r in picked:
        b = json.loads(json.dumps(r))            # deep copy
        b["scene_label"] = None                   # 제안 숨김
        b["notes"] = ""                            # judge 정보 숨김
        b["review_status"] = "draft"
        b["split"] = "overlap_blind"
        blinded.append(b)

    with args.out.open("w", encoding="utf-8") as f:
        for r in blinded:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    print(f"overlap 셋: {len(blinded)}/{len(rows)} ({len(blinded)/len(rows):.1%})")
    print("층(라벨,taxonomy)별:", dict(Counter(stratum(r) for r in picked)))
    print(f"→ {args.out} (scene_label/notes 비움 = 숨김)")


if __name__ == "__main__":
    main()
