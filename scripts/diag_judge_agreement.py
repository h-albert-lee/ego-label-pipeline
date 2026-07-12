"""judge↔judge (Claude jupiter vs GPT-5.4-mini) 상호 일치율 + 3자 비교."""
import json
from collections import Counter
from pathlib import Path

A = Path("outputs/ego4d/crosscheck.jsonl")       # claude-jupiter
B = Path("outputs/ego4d/crosscheck_gpt.jsonl")    # gpt-5.4-mini


def load(p):
    rows = [json.loads(l) for l in open(p) if l.strip()]
    jn = list(rows[0]["judges"].keys())[0]
    return {r["id"]: (r["auto_ground_truth"], r["judges"][jn].get("label")) for r in rows}, jn


a, na = load(A)
b, nb = load(B)
common = sorted(set(a) & set(b))
print(f"judge A = {na}, judge B = {nb}")
print(f"공통 레코드: {len(common)}개\n")

aa = ab = bb_gt = both_gt = neither = 0
threeway = 0
confusion = Counter()
for rid in common:
    gt, la = a[rid]
    _, lb = b[rid]
    if la == lb:
        aa += 1
        if la == gt:
            threeway += 1
    if la == gt:
        ab += 1
    if lb == gt:
        bb_gt += 1
    if la == gt and lb == gt:
        both_gt += 1
    if la != gt and lb != gt:
        neither += 1
    confusion[(la, lb)] += 1

n = len(common)
print("## (ii) judge↔judge 일치율")
print(f"  Claude ↔ GPT 라벨 일치: {aa}/{n} = {aa/n:.1%}")
print()
print("## judge↔GT 일치율 (공통 표본 기준)")
print(f"  Claude ↔ GT: {ab}/{n} = {ab/n:.1%}")
print(f"  GPT    ↔ GT: {bb_gt}/{n} = {bb_gt/n:.1%}")
print()
print("## 3자 관계")
print(f"  세 라벨 모두 일치 (Claude=GPT=GT):          {threeway}/{n} = {threeway/n:.1%}")
print(f"  두 judge 일치하나 GT와 다름 (GT 의심):      {aa-threeway}/{n} = {(aa-threeway)/n:.1%}")
print(f"  두 judge 모두 GT와 불일치 (일치 무관):      {neither}/{n} = {neither/n:.1%}")
print()

# judge끼리 일치했는데 GT와 다른 케이스의 라벨 이동 방향
print("## 두 judge가 합의했는데 GT가 다른 케이스 — GT→judge합의 라벨")
mv = Counter()
for rid in common:
    gt, la = a[rid]
    _, lb = b[rid]
    if la == lb and la != gt:
        mv[(gt, la)] += 1
for (gt, jl), c in mv.most_common():
    print(f"  GT={gt:<10} → 두 judge 모두={jl:<10} : {c}건")

print("\n## 라벨 분포 비교 (공통 50)")
gtc = Counter(a[r][0] for r in common)
ac = Counter(a[r][1] for r in common)
bc = Counter(b[r][1] for r in common)
print(f"{'label':<12}{'GT':>5}{'Claude':>8}{'GPT':>6}")
for L in ["MINE", "PERSON_k", "SHARED", "AMBIGUOUS"]:
    print(f"{L:<12}{gtc.get(L,0):>5}{ac.get(L,0):>8}{bc.get(L,0):>6}")
