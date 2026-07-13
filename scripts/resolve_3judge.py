"""3-judge 재adjudication — 미해결 2137개를 승인만/사람판정으로 재분류.

judge 3표:
  J1 claude-jupiter (frames-only, 1차)
  J2 gpt-5.4-mini   (frames-only, 2차)
  J3 claude-jupiter (with-narration, 3차)   <- evidence_mode 다름, tie-break

3표 중 2표 이상이 한 라벨에 모이면 majority 성립 → '승인만'(proposed=majority).
3표가 전부 갈리면 majority 없음 → '사람판정' 유지.

주의: J1과 J3은 같은 모델(jupiter)이라 완전 독립이 아님. 그래서 majority를
'2/3 동의'로 보되, J3(narration)이 J2(frames gpt)와 일치해 J1을 뒤집는 경우를
별도로 집계 — 이게 narration이 실제로 기여한 케이스.
"""
import json
from collections import Counter
from pathlib import Path

D = Path("outputs/ego4d")
ev = {json.loads(l)["id"]: json.loads(l) for l in open(D / "labels_v2_eval.jsonl") if l.strip()}
cc1 = {json.loads(l)["id"]: json.loads(l) for l in open(D / "crosscheck_eval.jsonl") if l.strip()}
cc2 = {json.loads(l)["id"]: json.loads(l) for l in open(D / "crosscheck_eval_gpt.jsonl") if l.strip()}
cc3 = {json.loads(l)["id"]: json.loads(l) for l in open(D / "crosscheck_eval_narr.jsonl") if l.strip()}
J1, J2, J3 = "claude-jupiter-v1-p", "gpt-5.4-mini", "claude-jupiter-v1-p"


def lab(cc, i, jn):
    r = cc.get(i)
    return (r["judges"].get(jn) or {}).get("label") if r else None


resolved, still_open = [], []
narr_broke_tie = 0
move = Counter()
for i in cc3:                      # only the 2137 unresolved were sent to J3
    gt = cc1[i]["auto_ground_truth"]
    l1, l2, l3 = lab(cc1, i, J1), lab(cc2, i, J2), lab(cc3, i, J3)
    votes = [v for v in (l1, l2, l3) if v in ("MINE", "PERSON_k", "SHARED", "AMBIGUOUS")]
    top, cnt = (Counter(votes).most_common(1)[0] if votes else (None, 0))
    if cnt >= 2:                   # majority
        resolved.append((i, gt, top, l1, l2, l3))
        move[(gt, top)] += 1
        if l3 == l2 and l3 != l1:  # narration(J3) sided with gpt to break J1
            narr_broke_tie += 1
    else:
        still_open.append((i, gt, l1, l2, l3))

n = len(cc3)
print(f"3차 대상(미해결): {n}")
print(f"  majority 성립 → 승인만 이동: {len(resolved)} ({len(resolved)/n:.1%})")
print(f"  여전히 3표 갈림 → 사람판정 유지: {len(still_open)} ({len(still_open)/n:.1%})")
print(f"  그중 narration(J3)이 gpt와 합의해 J1 뒤집은 케이스: {narr_broke_tie}")

print("\n## 새로 해소된 것의 라벨 이동 (원 GT → 3judge majority)")
for (g, t), c in move.most_common():
    tag = "  (재확인)" if g == t else ""
    print(f"  {g:<10} → {t:<10}: {c}{tag}")

# 재분류 후 최종 큐 크기 (전체 review_queue 기준)
q = [json.loads(l) for l in open(D / "review_queue.jsonl") if l.strip()]
prev_approve = sum(1 for r in q if not r["needs_human"])
prev_human = sum(1 for r in q if r["needs_human"])
new_approve = prev_approve + len(resolved)
new_human = prev_human - len(resolved)
print("\n## 검수 큐 갱신")
print(f"  {'':16}{'이전':>8}{'이후':>8}")
print(f"  {'승인만':<14}{prev_approve:>8}{new_approve:>8}")
print(f"  {'사람판정':<12}{prev_human:>8}{new_human:>8}")
print(f"  {'큐 합계':<14}{len(q):>8}{len(q):>8}")

# 재adjudication 결과를 갱신용으로 저장
out = {i: dict(proposed=top, votes=[l1, l2, l3], resolved_by="3judge-majority")
       for (i, gt, top, l1, l2, l3) in resolved}
Path("/tmp/claude-1000/-home-gpuadmin-albert/2d6f65bb-9ea0-4f24-bd19-b48dca9a8ac3/scratchpad/resolved3.json").write_text(json.dumps(out))
print(f"\n해소 {len(out)}개 저장 → scratchpad/resolved3.json")
