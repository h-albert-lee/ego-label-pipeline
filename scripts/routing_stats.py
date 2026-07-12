"""C' 라우팅 통계 — gt_confidence × judge 동의, 검수 큐 크기, judge majority 채굴.

입력:
  labels_v2_eval.jsonl        (gt_confidence/subset_tier 태깅)
  crosscheck_eval.jsonl       (1차: claude judge, 전체 5158)
  crosscheck_eval_gpt.jsonl   (2차: gpt judge, 1차 disagree 3129)
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

D = Path("outputs/ego4d")
ev = {json.loads(l)["id"]: json.loads(l) for l in open(D / "labels_v2_eval.jsonl") if l.strip()}
cc1 = {json.loads(l)["id"]: json.loads(l) for l in open(D / "crosscheck_eval.jsonl") if l.strip()}
cc2 = {json.loads(l)["id"]: json.loads(l) for l in open(D / "crosscheck_eval_gpt.jsonl") if l.strip()}
JC = "claude-jupiter-v1-p"
JG = "gpt-5.4-mini"


def jlabel(row, jn):
    return (row["judges"].get(jn) or {}).get("label")


rows = []
for i, e in ev.items():
    c1 = cc1[i]
    la = jlabel(c1, JC)
    lg = jlabel(cc2[i], JG) if i in cc2 else None  # only disagree rows went to pass2
    rows.append({
        "id": i,
        "conf": e["gt_confidence"],
        "tier": e["subset_tier"],
        "gt": c1["auto_ground_truth"],
        "claude": la,
        "gpt": lg,                       # None if claude agreed with GT (not sent to pass2)
        "claude_agree": c1["majority_agrees"],
    })

n = len(rows)
LABELS = ["MINE", "PERSON_k", "SHARED", "AMBIGUOUS"]

# ---- a. gt_confidence × judge 동의 크로스탭 -------------------------------
print("=" * 68)
print("a. gt_confidence × claude judge 동의")
print(f"{'conf':<8}{'tier':<6}{'n':>6}{'agree':>7}{'rate':>8}{'disagree':>10}")
for conf, tier in [("high", "P0"), ("medium", "P1"), ("low", "P2")]:
    g = [r for r in rows if r["conf"] == conf]
    ag = sum(r["claude_agree"] for r in g)
    print(f"{conf:<8}{tier:<6}{len(g):>6}{ag:>7}{ag/len(g):>8.1%}{len(g)-ag:>10}")
print(f"{'전체':<14}{n:>6}{sum(r['claude_agree'] for r in rows):>7}"
      f"{sum(r['claude_agree'] for r in rows)/n:>8.1%}")

# ---- b. disagree 행에서 judge↔judge 수렴 (GT 문제 vs judge 불신) ----------
print("\n" + "=" * 68)
print("b. claude가 GT와 disagree한 행에서 claude↔gpt 수렴")
dis = [r for r in rows if not r["claude_agree"] and r["gpt"]]
jj = sum(1 for r in dis if r["claude"] == r["gpt"])
jj_bin = sum(1 for r in dis if (r["claude"] == "MINE") == (r["gpt"] == "MINE"))
gpt_back = sum(1 for r in dis if r["gpt"] == r["gt"])
print(f"  disagree 행: {len(dis)}")
print(f"  claude=gpt (4분류 완전 일치): {jj}/{len(dis)} = {jj/len(dis):.1%}")
print(f"  MINE/not 이진축 일치:         {jj_bin}/{len(dis)} = {jj_bin/len(dis):.1%}")
print(f"  gpt는 오히려 GT와 일치(claude만 튐): {gpt_back}/{len(dis)} = {gpt_back/len(dis):.1%}")
print("  해석: 이진축 높음 + gpt-GT낮음 → 두 judge가 GT에 맞서 수렴 = GT(cascade) 의심")

# ---- c. P2(low) judge majority 라벨 분포 (희소클래스 채굴) ----------------
print("\n" + "=" * 68)
print("c. P2(low=proximity MINE) 채굴 — judge majority 제안 라벨 분포")


def majority(r):
    """두 judge(가능하면)의 다수결. gpt 없으면 claude가 GT와 일치했다는 뜻."""
    votes = [v for v in (r["claude"], r["gpt"]) if v]
    if not r["gpt"]:                       # claude가 GT와 일치 → 제안 = GT
        return r["gt"]
    if r["claude"] == r["gpt"]:
        return r["claude"]
    return None                            # 두 judge 불일치 → 미해결


low = [r for r in rows if r["conf"] == "low"]
prop = Counter(majority(r) for r in low)
print(f"  P2 총 {len(low)}개 (원래 전부 GT=MINE)")
print(f"  judge 제안 라벨: {dict(prop)}")
resolved = {k: v for k, v in prop.items() if k}
mined = sum(v for k, v in resolved.items() if k != "MINE")
print(f"  → MINE 아닌 라벨로 채굴(두 judge 합의 or claude=GT아님): "
      f"{mined}/{len(low)} = {mined/len(low):.1%}")
print(f"  → 미해결(두 judge 불일치, None): {prop.get(None,0)}")

# ---- d. human 검수 큐 --------------------------------------------------
print("\n" + "=" * 68)
print("d. human 검수 큐")
q_low = [r for r in rows if r["conf"] == "low"]
q_hm_dis = [r for r in rows if r["conf"] in ("high", "medium") and not r["claude_agree"]]
# 자동 확정 가능(두 judge가 GT와 다른 라벨로 합의) vs 사람 필수(judge 불일치)
q_unresolved = [r for r in (q_low + q_hm_dis) if r["gpt"] and r["claude"] != r["gpt"]]
print(f"  (i) low-conf 전체:                 {len(q_low)}")
print(f"  (ii) high/medium인데 disagree:     {len(q_hm_dis)}")
print(f"  검수 큐 합집합:                     {len(q_low) + len(q_hm_dis)}")
print(f"  그중 두 judge 불일치(사람 필수):    {len(q_unresolved)}")
print(f"  그중 두 judge 합의(제안 라벨 승인검토): "
      f"{len(q_low)+len(q_hm_dis)-len(q_unresolved)}")

# 검수 시트용 id 목록 저장 (제안 라벨 포함)
sheet = []
for r in q_low + q_hm_dis:
    sheet.append({**r, "proposed": majority(r)})
Path("/tmp/claude-1000/-home-gpuadmin-albert/2d6f65bb-9ea0-4f24-bd19-b48dca9a8ac3/scratchpad/review_queue.json").write_text(
    json.dumps(sheet, ensure_ascii=False))
print(f"\n  검수 큐 저장: {len(sheet)}개 → scratchpad/review_queue.json")
