"""32% 불일치 진단 리포트 (a,b,c) — crosscheck.jsonl + labels_v2.jsonl 조인."""
import json
from collections import Counter, defaultdict
from pathlib import Path

CC = Path("outputs/ego4d/crosscheck.jsonl")
LAB = Path("outputs/ego4d/labels_v2.jsonl")

lab = {json.loads(l)["id"]: json.loads(l) for l in open(LAB) if l.strip()}
cc = [json.loads(l) for l in open(CC) if l.strip()]
jname = list(cc[0]["judges"].keys())[0]

rows = []
for r in cc:
    j = r["judges"].get(jname, {})
    src = lab.get(r["id"], {})
    ake = src.get("auto_key_evidence") or {}
    zev = ake.get("zone_evidence", "") or ""
    dist = None
    if "distance" in zev:
        try:
            dist = float(zev.split("distance")[1].split(".")[0] + "." + zev.split("distance")[1].split(".")[1].split()[0])
        except Exception:
            dist = None
    rows.append({
        "id": r["id"],
        "gt": r["auto_ground_truth"],
        "judge": j.get("label"),
        "agrees": r["majority_agrees"],
        "object_type": ake.get("object_type"),
        "target_zone": ake.get("target_zone"),
        "taxonomy": src.get("auto_taxonomy"),
        "needs_review": src.get("needs_review"),
        "distance": dist,
    })

n = len(rows)
overall = sum(1 for r in rows if r["agrees"]) / n
print(f"# 진단 리포트 (표본 {n}개, judge={jname})")
print(f"전체 judge↔GT 일치율: {overall:.1%}\n")

# a. GT 라벨별 일치율
print("## a. GT 라벨별 judge 일치율")
print(f"{'GT label':<12}{'n':>4}{'agree':>7}{'rate':>8}")
by_gt = defaultdict(list)
for r in rows:
    by_gt[r["gt"]].append(r["agrees"])
for gt in ["MINE", "PERSON_k", "SHARED", "AMBIGUOUS"]:
    v = by_gt.get(gt, [])
    if v:
        print(f"{gt:<12}{len(v):>4}{sum(v):>7}{sum(v)/len(v):>8.1%}")
print()

# b. cascade 규칙별 일치율 크로스탭 (object_type × target_zone)
print("## b. cascade 규칙별 일치율 (object_type × target_zone)")
print(f"{'object_type':<12}{'target_zone':<28}{'n':>4}{'agree':>7}{'rate':>8}  (대표 GT)")
by_rule = defaultdict(list)
rule_gt = defaultdict(Counter)
for r in rows:
    k = (r["object_type"], r["target_zone"])
    by_rule[k].append(r["agrees"])
    rule_gt[k][r["gt"]] += 1
for k in sorted(by_rule, key=lambda x: -len(by_rule[x])):
    v = by_rule[k]
    ot, tz = k
    top = rule_gt[k].most_common(1)[0][0]
    print(f"{str(ot):<12}{str(tz):<28}{len(v):>4}{sum(v):>7}{sum(v)/len(v):>8.1%}  {top}")
print()

# b2. depth/proximity 밴드별 (MINE 규칙 혐의: other_person_zone인데 MINE)
print("## b2. depth-band/proximity — MINE GT 중 target_zone별")
mine = [r for r in rows if r["gt"] == "MINE"]
zc = defaultdict(list)
for r in mine:
    zc[r["target_zone"]].append(r["agrees"])
for tz, v in sorted(zc.items(), key=lambda x: -len(x[1])):
    print(f"  MINE & zone={str(tz):<28} n={len(v):>3} 일치율={sum(v)/len(v):.1%}")
dists = [r["distance"] for r in mine if r["distance"] is not None]
if dists:
    print(f"  MINE proximity distance: min={min(dists):.2f} mean={sum(dists)/len(dists):.2f} max={max(dists):.2f}")
print()

# c. judge 분포 vs GT 분포
print("## c. 라벨 분포: judge vs GT")
labels = ["MINE", "PERSON_k", "SHARED", "AMBIGUOUS", "UNKNOWN", "ERROR"]
gtc = Counter(r["gt"] for r in rows)
jc = Counter(r["judge"] for r in rows)
print(f"{'label':<12}{'GT':>6}{'judge':>7}{'delta':>8}")
for L in labels:
    g, j = gtc.get(L, 0), jc.get(L, 0)
    if g or j:
        print(f"{L:<12}{g:>6}{j:>7}{j-g:>+8}")

# 혼동행렬
print("\n## c2. 혼동행렬 (행=GT, 열=judge)")
cols = [L for L in labels if any(r["judge"] == L for r in rows)]
print(f"{'GT\\judge':<12}" + "".join(f"{c:>10}" for c in cols))
for gt in ["MINE", "PERSON_k", "SHARED", "AMBIGUOUS"]:
    line = f"{gt:<12}"
    for c in cols:
        cnt = sum(1 for r in rows if r["gt"] == gt and r["judge"] == c)
        line += f"{cnt:>10}"
    print(line)
