"""검수 완료 후 집계 — egoown_report가 읽는 JSON 3종.

입력:
  audit_scenerecords.jsonl   검수 결과(사람 라벨 = scene_label, edits[] 포함)
  audit_proposals_side.jsonl 원본 judge 제안/auto_gt/assigned_to (blind 전 보존)

산출(a/b/c):
  (a) per-label 오류율   : blind 사람판정 vs judge 제안 (제안이 틀린 비율)
  (b) Cohen's kappa      : 이중배정 쌍의 두 사람 판정 일치 (edits에서 annotator별 최종 라벨)
  (c) per-taxonomy accuracy: blind 사람판정 vs 최종 GT(=사람판정 자체가 GT이므로
      여기선 judge 제안을 예측으로 둔 human-reference accuracy로 산출)

출력: outputs/ego4d/audit_report.json (egoown_report 호환 스키마)
"""
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

D = Path("outputs/ego4d")
LABELS = ["MINE", "PERSON_k", "SHARED", "AMBIGUOUS"]


def final_label_per_annotator(rec):
    """edits[]에서 annotator별 마지막 scene_label 판정을 뽑는다."""
    out = {}
    for e in rec.get("edits", []):
        if e.get("field") == "scene_label" and e.get("new_value"):
            out[e["annotator"]] = e["new_value"]
    return out


def cohens_kappa(pairs):
    """pairs: [(labelA, labelB), ...] → Cohen's kappa."""
    n = len(pairs)
    if n == 0:
        return None
    po = sum(1 for a, b in pairs if a == b) / n
    ca = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in set(ca) | set(cb))
    return None if pe == 1 else round((po - pe) / (1 - pe), 4)


def main():
    side = {r["clip_id"]: r for r in (json.loads(l) for l in open(D / "audit_proposals_side.jsonl") if l.strip())}
    recs = {json.loads(l)["clip"]["clip_id"]: json.loads(l)
            for l in open(D / "audit_scenerecords.jsonl") if l.strip()
            if json.loads(l).get("split") == "audit"}

    # 사람 최종 판정 (verified/rejected 또는 scene_label 세팅된 것)
    human = {}
    per_annot = {}
    for cid, rec in recs.items():
        fa = final_label_per_annotator(rec)
        per_annot[cid] = fa
        # 대표 사람 라벨: scene_label(최종) 우선, 없으면 첫 annotator
        hl = rec.get("scene_label") or (next(iter(fa.values())) if fa else None)
        if hl:
            human[cid] = hl

    reviewed = len(human)
    total = len(recs)

    # (a) per-label 오류율: judge 제안 vs 사람판정 (제안 기준)
    a = {}
    for lab in LABELS:
        prop_items = [cid for cid in human if side[cid].get("proposed") == lab]
        wrong = sum(1 for cid in prop_items if human[cid] != lab)
        a[lab] = {"n_proposed": len(prop_items), "human_disagreed": wrong,
                  "error_rate": round(wrong / len(prop_items), 4) if prop_items else None}

    # (b) Cohen's kappa: 이중배정에서 두 annotator 최종 라벨 쌍
    kappa_pairs = []
    per_pair = defaultdict(list)
    for cid, fa in per_annot.items():
        who = [a for a in fa if a in (side[cid].get("assigned_to") or [])]
        if len(who) >= 2:
            for x, y in combinations(sorted(who), 2):
                kappa_pairs.append((fa[x], fa[y]))
                per_pair[(x, y)].append((fa[x], fa[y]))
    b = {"overall_kappa": cohens_kappa(kappa_pairs),
         "n_double_annotated_pairs": len(kappa_pairs),
         "per_pair_kappa": {f"{x}|{y}": cohens_kappa(v) for (x, y), v in per_pair.items()}}

    # (c) per-taxonomy: 사람판정을 reference로, judge 제안 accuracy
    c = {}
    tax = defaultdict(lambda: [0, 0])
    for cid in human:
        t = side[cid].get("taxonomy")
        tax[t][1] += 1
        if side[cid].get("proposed") == human[cid]:
            tax[t][0] += 1
    for t, (ok, n) in tax.items():
        c[str(t)] = {"n": n, "judge_matches_human": ok, "accuracy": round(ok / n, 4) if n else None}

    report = {
        "audit_summary": {"total_audit": total, "human_reviewed": reviewed,
                          "completion": round(reviewed / total, 4) if total else 0},
        "a_per_label_proposal_error": a,
        "b_inter_annotator_kappa": b,
        "c_per_taxonomy_judge_accuracy": c,
    }
    (D / "audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n→ {D/'audit_report.json'}")


if __name__ == "__main__":
    main()
