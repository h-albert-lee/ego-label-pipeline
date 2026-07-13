"""통합 검수 대시보드 셋 — Ego4D + HD-EPIC 병합, 층화 샘플, blind, annotator 배정.

- review 대상(split=review) 병합: Ego4D review_scenerecords + EPIC review_scenerecords_epic.
  frame_path에 소스 접두("ego4d/"|"epic/") → 통합 frames_root(심볼릭)에서 resolve.
- 감사 샘플: label×taxonomy×source_dataset 층화 15% (결정론적) + AMBIGUOUS rescue 전량.
- 샘플=검수 셋, 나머지=review_status "auto_accepted"(judge 합의 GT 확정).
- 검수 셋 blind: scene_label=None, notes="" (제안/judge 숨김). 원본 제안은 side 파일 보존.
- annotator 배정: 5명 결정론적 라운드로빈 + 30% 이중배정(쌍 고르게). assigned_to 리스트.
"""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ANNOTATORS = ["jangha", "hanwool", "seungyeop", "guijin", "chaeyun"]
FRAC = 0.15
DOUBLE_FRAC = 0.30
D = Path("outputs/ego4d")
EPIC = Path("outputs/epic")


def h(s, salt=""):
    return int.from_bytes(hashlib.sha256((salt + s).encode()).digest()[:8], "big")


def label_of(rec, q):
    """확정 예정 라벨: 제안(proposed) 우선, 없으면 auto_gt(rescue 등)."""
    qr = q.get(rec["clip"]["clip_id"], {})
    return qr.get("proposed") or qr.get("gt") or rec.get("scene_label")


def main():
    q = {json.loads(l)["id"]: json.loads(l) for l in open(D / "review_queue.jsonl") if l.strip()}

    # --- 병합 (split=review만), frame_path 소스 접두 ---
    merged = []
    for path, prefix, src in [
        (D / "review_scenerecords.jsonl", "ego4d/", "ego4d"),
        (EPIC / "review_scenerecords_epic.jsonl", "epic/", "hd_epic"),
    ]:
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("split") != "review":
                continue
            for f in r["frames"]:
                if f.get("frame_path") and not f["frame_path"].startswith(prefix):
                    f["frame_path"] = prefix + f["frame_path"]
            r["clip"]["source_dataset"] = src
            merged.append(r)
    print(f"병합 review 대상: {len(merged)} ({dict(Counter(r['clip']['source_dataset'] for r in merged))})")

    # --- 층화 샘플 + AMBIGUOUS rescue 전량 ---
    is_rescue = lambda r: q.get(r["clip"]["clip_id"], {}).get("rescue") is not None
    rescue = [r for r in merged if is_rescue(r)]
    pool = [r for r in merged if not is_rescue(r)]
    strata = defaultdict(list)
    for r in pool:
        key = (label_of(r, q), r["clip"].get("taxonomy"), r["clip"]["source_dataset"])
        strata[key].append(r)
    sampled = set()
    for key, items in strata.items():
        items = sorted(items, key=lambda r: h(r["clip"]["clip_id"]))
        n = max(1, round(len(items) * FRAC))
        for r in items[:n]:
            sampled.add(r["clip"]["clip_id"])
    audit_ids = sampled | {r["clip"]["clip_id"] for r in rescue}
    print(f"감사 셋: 층화 {len(sampled)} + rescue {len(rescue)} = {len(audit_ids)}")

    # --- 배정: 라운드로빈 + 30% 이중 ---
    audit_recs = [r for r in merged if r["clip"]["clip_id"] in audit_ids]
    audit_recs.sort(key=lambda r: h(r["clip"]["clip_id"], "assign"))
    npair = len(ANNOTATORS)
    pairs = [(ANNOTATORS[i], ANNOTATORS[j])
             for i in range(npair) for j in range(i + 1, npair)]  # 10 쌍
    for idx, r in enumerate(audit_recs):
        primary = ANNOTATORS[idx % npair]
        assigned = [primary]
        if h(r["clip"]["clip_id"], "double") % 100 < DOUBLE_FRAC * 100:
            pair = pairs[h(r["clip"]["clip_id"], "pair") % len(pairs)]
            second = pair[1] if pair[0] == primary else pair[0]
            if second != primary:
                assigned.append(second)
        r["assigned_to"] = assigned

    # --- blind + 원본 제안 side 보존 ---
    side = []
    audit_set = set(audit_ids)
    out = []
    for r in merged:
        cid = r["clip"]["clip_id"]
        if cid in audit_set:
            side.append({"clip_id": cid, "proposed": label_of(r, q),
                         "auto_gt": q.get(cid, {}).get("gt"), "notes": r.get("notes"),
                         "source_dataset": r["clip"]["source_dataset"],
                         "taxonomy": r["clip"].get("taxonomy"),
                         "assigned_to": r.get("assigned_to")})
            r["scene_label"] = None       # blind
            r["notes"] = ""
            r["review_status"] = "in_review"
            r["split"] = "audit"
        else:
            r["review_status"] = "auto_accepted"   # judge 합의 GT 확정
            r["scene_label"] = label_of(r, q)
            r["split"] = "auto_accepted"
        out.append(r)

    with (D / "audit_scenerecords.jsonl").open("w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (D / "audit_proposals_side.jsonl").open("w", encoding="utf-8") as f:
        for r in side:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    dbl = sum(1 for r in audit_recs if len(r["assigned_to"]) > 1)
    print(f"\n감사 셋 {len(side)}: 단독 {len(side)-dbl} / 이중 {dbl} ({dbl/len(side):.0%})")
    print("annotator 부하:", dict(Counter(a for r in audit_recs for a in r["assigned_to"])))
    print("이중배정 쌍:", dict(Counter(tuple(sorted(r["assigned_to"])) for r in audit_recs if len(r["assigned_to"])>1)))
    print(f"auto_accepted: {len(out)-len(side)}")
    print(f"→ {D/'audit_scenerecords.jsonl'} (blind), {D/'audit_proposals_side.jsonl'} (제안 보존)")


if __name__ == "__main__":
    main()
